#!/usr/bin/env python3
"""Prepare isolated RAG-only development panels for residual Planner DPO."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import normalize_question, write_json, write_jsonl  # noqa: E402
from finetune.tool_trace_common import sha256_file, sha256_json, stable_seed  # noqa: E402


TOOL_COUNT_WEIGHTS = {1: 0.10, 2: 0.15, 3: 0.20, 4: 0.20, 5: 0.20, 6: 0.15}
STYLES = ("legacy_train", "assignment3_api_native_train")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def source_id(row: dict[str, Any]) -> str:
    construction = row.get("construction")
    if isinstance(construction, dict) and construction.get("source_trace_id"):
        return str(construction["source_trace_id"])
    meta = row.get("meta")
    if isinstance(meta, dict) and meta.get("source_trace_id"):
        return str(meta["source_trace_id"])
    return str(row.get("id", row.get("trace_id", "")))


def quotas(total: int) -> dict[int, int]:
    raw = {key: total * value for key, value in TOOL_COUNT_WEIGHTS.items()}
    result = {key: int(value) for key, value in raw.items()}
    remainder = total - sum(result.values())
    for key in sorted(raw, key=lambda item: (-(raw[item] - result[item]), item))[
        :remainder
    ]:
        result[key] += 1
    return result


def compact_gold(row: dict[str, Any], role: str) -> dict[str, Any]:
    style = (
        "legacy_compact"
        if row["source_split"] == "legacy_train"
        else "api_native"
    )
    return {
        "id": str(row["trace_id"]),
        "问题": str(row["query"]),
        "工具": ",".join(row["gold_tools"]),
        "思考": "GT 由隔离训练侧标注确定；只用于 RAG-only DPO 选参与确认。",
        "source": "assignment3_rag_residual_dpo_dev",
        "domain": row.get("domain", ""),
        "tool_count": int(row["tool_count"]),
        "generation_split": "assignment3_rag_residual_dpo_dev",
        "style": style,
        "source_split": str(row["source_split"]),
        "combo_seen_bucket": row.get("combo_seen_bucket", ""),
        "construction": {
            **dict(row.get("construction") or {}),
            "dataset_role": role,
            "type": "rag_residual_dpo_isolated_dev",
        },
    }


def label_free(row: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "id": str(row["trace_id"]),
        "query": str(row["query"]),
        "candidate_tools": list(row["candidate_tools"]),
        "candidate_k": 10,
        "ranking_hash": sha256_json(row["candidate_tools"]),
    }
    payload["input_hash"] = sha256_json(payload)
    return payload


def choose_panel(
    pools: dict[tuple[str, int], list[dict[str, Any]]],
    per_style: int,
    used_sources: set[str],
    seed: int,
    role: str,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    exposure: Counter[str] = Counter()
    for style in STYLES:
        before = len(selected)
        for tool_count, target in quotas(per_style).items():
            pool = [
                row
                for row in pools[(style, tool_count)]
                if source_id(row) not in used_sources
            ]
            pool.sort(
                key=lambda row: stable_seed(
                    seed, role, style, tool_count, row["trace_id"]
                )
            )
            for index in range(min(target, len(pool))):
                window_end = min(len(pool), index + 256)
                row = min(
                    pool[index:window_end],
                    key=lambda item: (
                        max(exposure[tool] + 1 for tool in item["gold_tools"]),
                        sum((exposure[tool] + 1) ** 2 for tool in item["gold_tools"]),
                        stable_seed(seed, role, index, item["trace_id"]),
                    ),
                )
                selected_index = pool.index(row, index, window_end)
                pool[index], pool[selected_index] = pool[selected_index], pool[index]
                selected.append(row)
                used_sources.add(source_id(row))
                exposure.update(row["gold_tools"])
        current = len(selected) - before
        if current < per_style:
            residual = sorted(
                (
                    row
                    for (row_style, _), rows in pools.items()
                    if row_style == style
                    for row in rows
                    if source_id(row) not in used_sources
                ),
                key=lambda row: (
                    max(exposure[tool] + 1 for tool in row["gold_tools"]),
                    sum((exposure[tool] + 1) ** 2 for tool in row["gold_tools"]),
                    stable_seed(seed, role, style, "residual", row["trace_id"]),
                ),
            )
            for row in residual[: per_style - current]:
                selected.append(row)
                used_sources.add(source_id(row))
                exposure.update(row["gold_tools"])
        if len(selected) - before != per_style:
            raise ValueError(f"incorrect {role}/{style} panel size")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument("--api-train", type=Path, required=True)
    parser.add_argument("--sft-runtime", type=Path, required=True)
    parser.add_argument("--v1-preferences", type=Path, required=True)
    parser.add_argument("--v1-dev", type=Path, required=True)
    parser.add_argument("--frozen", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-per-style", type=int, default=600)
    parser.add_argument("--confirmation-per-style", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()

    compact = read_jsonl(args.compact)
    api_rows = read_jsonl(args.api_train)
    api_sources = {
        str(row["id"]): str(row["construction"]["source_trace_id"])
        for row in api_rows
    }
    sft_rows = read_jsonl(args.sft_runtime)
    sft_sources = {
        api_sources[str(row["meta"]["trace_id"])] for row in sft_rows
    }
    v1_preferences = read_jsonl(args.v1_preferences)
    v1_preference_sources = {source_id(row) for row in v1_preferences}
    v1_dev = read_jsonl(args.v1_dev)
    v1_dev_sources = {source_id(row) for row in v1_dev}
    frozen_rows = [row for path in args.frozen for row in read_jsonl(path)]
    frozen_ids = {str(row["id"]) for row in frozen_rows}
    frozen_queries = {
        normalize_question(str(row.get("问题", row.get("query", ""))))
        for row in frozen_rows
    }

    blocked_sources = sft_sources | v1_preference_sources | v1_dev_sources
    by_source: dict[tuple[str, str], dict[str, Any]] = {}
    for row in compact:
        if len(row["candidate_tools"]) != 10 or not set(row["gold_tools"]) <= set(
            row["candidate_tools"]
        ):
            raise ValueError(f"invalid compact RAG row: {row['trace_id']}")
        source = source_id(row)
        if source in blocked_sources or str(row["trace_id"]) in frozen_ids:
            continue
        if normalize_question(str(row["query"])) in frozen_queries:
            continue
        key = (str(row["source_split"]), source)
        previous = by_source.get(key)
        if previous is None or stable_seed(
            args.seed, "source-row", row["trace_id"]
        ) < stable_seed(args.seed, "source-row", previous["trace_id"]):
            by_source[key] = row

    pools: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for (style, _), row in by_source.items():
        pools[(style, int(row["tool_count"]))].append(row)

    used_sources = set(blocked_sources)
    selection = choose_panel(
        pools,
        args.selection_per_style,
        used_sources,
        args.seed,
        "selection",
    )
    confirmation = choose_panel(
        pools,
        args.confirmation_per_style,
        used_sources,
        args.seed,
        "confirmation",
    )

    selection_sources = {source_id(row) for row in selection}
    confirmation_sources = {source_id(row) for row in confirmation}
    selection_queries = {normalize_question(str(row["query"])) for row in selection}
    confirmation_queries = {
        normalize_question(str(row["query"])) for row in confirmation
    }
    selection_multi_combos = {
        tuple(sorted(row["gold_tools"])) for row in selection if row["tool_count"] > 1
    }
    confirmation_multi_combos = {
        tuple(sorted(row["gold_tools"]))
        for row in confirmation
        if row["tool_count"] > 1
    }
    if selection_sources & confirmation_sources:
        raise ValueError("selection/confirmation source overlap")
    if selection_queries & confirmation_queries:
        raise ValueError("selection/confirmation query overlap")

    output = args.output_dir
    selection_gold = [compact_gold(row, "selection") for row in selection]
    confirmation_gold = [compact_gold(row, "confirmation") for row in confirmation]
    write_jsonl(output / "plan/selection_gold.jsonl", selection_gold)
    write_jsonl(output / "plan/confirmation_gold.jsonl", confirmation_gold)
    write_jsonl(output / "data/selection_rag_label_free.jsonl", [label_free(row) for row in selection])
    write_jsonl(
        output / "data/confirmation_rag_label_free.jsonl",
        [label_free(row) for row in confirmation],
    )

    # Previous preference sources are eligible again only when the current parent
    # still makes a fresh semantic error on them. They remain excluded from both
    # development panels, so this does not weaken selection isolation.
    blocked_for_preferences = v1_dev_sources | selection_sources | confirmation_sources
    blocked_rows = [
        {
            "id": f"blocked_{index:05d}",
            "construction": {"source_trace_id": source},
        }
        for index, source in enumerate(sorted(blocked_for_preferences))
    ]
    write_jsonl(output / "plan/preference_blocked_sources.jsonl", blocked_rows)

    represented_selection = {tool for row in selection for tool in row["gold_tools"]}
    represented_confirmation = {tool for row in confirmation for tool in row["gold_tools"]}
    audit = {
        "schema_version": 1,
        "status": "complete",
        "strict_ok": (
            len(selection) == 2 * args.selection_per_style
            and len(confirmation) == 2 * args.confirmation_per_style
            and not (selection_sources & confirmation_sources)
            and not (selection_queries & confirmation_queries)
            and not (selection_sources & blocked_sources)
            and not (confirmation_sources & blocked_sources)
            and len(represented_selection) == 100
            and len(represented_confirmation) == 100
        ),
        "protocol": {
            "view": "rag_only",
            "candidate_k": 10,
            "parent_model_fixed": "Qwen3-8B current DPO checkpoint-100",
            "wo_rag_used": False,
            "selection_rows": len(selection),
            "confirmation_rows": len(confirmation),
            "selection_confirmation_multi_combo_overlap_is_report_only": True,
            "previous_preference_sources_may_be_remined": True,
        },
        "counts": {
            "compact_rows": len(compact),
            "eligible_unique_sources": len(by_source),
            "selection": len(selection),
            "confirmation": len(confirmation),
            "selection_by_style": dict(Counter(row["source_split"] for row in selection)),
            "confirmation_by_style": dict(Counter(row["source_split"] for row in confirmation)),
            "selection_by_tool_count": dict(Counter(str(row["tool_count"]) for row in selection)),
            "confirmation_by_tool_count": dict(Counter(str(row["tool_count"]) for row in confirmation)),
            "selection_tools": len(represented_selection),
            "confirmation_tools": len(represented_confirmation),
        },
        "isolation": {
            "sft_sources": len(sft_sources),
            "v1_preference_sources": len(v1_preference_sources),
            "v1_dev_sources": len(v1_dev_sources),
            "selection_blocked_source_overlap": len(selection_sources & blocked_sources),
            "confirmation_blocked_source_overlap": len(confirmation_sources & blocked_sources),
            "selection_confirmation_source_overlap": len(selection_sources & confirmation_sources),
            "selection_confirmation_query_overlap": len(selection_queries & confirmation_queries),
            "selection_confirmation_multi_combo_overlap": len(selection_multi_combos & confirmation_multi_combos),
            "selection_frozen_query_overlap": len(selection_queries & frozen_queries),
            "confirmation_frozen_query_overlap": len(confirmation_queries & frozen_queries),
        },
        "inputs": {
            str(path): sha256_file(path)
            for path in (
                args.compact,
                args.api_train,
                args.sft_runtime,
                args.v1_preferences,
                args.v1_dev,
                *args.frozen,
            )
        },
        "outputs": {
            str(path): sha256_file(path)
            for path in (
                output / "plan/selection_gold.jsonl",
                output / "plan/confirmation_gold.jsonl",
                output / "plan/preference_blocked_sources.jsonl",
                output / "data/selection_rag_label_free.jsonl",
                output / "data/confirmation_rag_label_free.jsonl",
            )
        },
    }
    write_json(output / "plan/data_plan_audit.json", audit)
    if not audit["strict_ok"]:
        raise ValueError("residual DPO data plan audit failed")
    print(
        f"residual DPO plan selection={len(selection)} "
        f"confirmation={len(confirmation)} strict_ok=True"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
