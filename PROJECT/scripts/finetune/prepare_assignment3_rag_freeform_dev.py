#!/usr/bin/env python3
"""Filter the audited free-form DPO dev panel against current training sources."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import normalize_question, write_json, write_jsonl  # noqa: E402
from finetune.tool_trace_common import sha256_file, stable_seed  # noqa: E402


STYLES = ("legacy_compact", "api_native")


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
    return str(row.get("id", ""))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-dev", type=Path, required=True)
    parser.add_argument("--api-train", type=Path, required=True)
    parser.add_argument("--sft-runtime", type=Path, required=True)
    parser.add_argument("--preference-data", action="append", type=Path, required=True)
    parser.add_argument("--blocked-dataset", action="append", type=Path, default=[])
    parser.add_argument("--frozen", action="append", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit-out", type=Path, required=True)
    parser.add_argument("--per-style", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()

    generated = read_jsonl(args.generated_dev)
    api_rows = read_jsonl(args.api_train)
    api_sources = {
        str(row["id"]): str(row["construction"]["source_trace_id"])
        for row in api_rows
    }
    sft_sources = {
        api_sources[str(row["meta"]["trace_id"])]
        for row in read_jsonl(args.sft_runtime)
    }
    preference_sources = {
        source_id(row)
        for path in args.preference_data
        for row in read_jsonl(path)
    }
    blocked_rows = [
        row for path in args.blocked_dataset for row in read_jsonl(path)
    ]
    blocked_dataset_sources = {source_id(row) for row in blocked_rows}
    blocked_dataset_queries = {
        normalize_question(str(row.get("问题", row.get("query", ""))))
        for row in blocked_rows
    }
    blocked_sources = sft_sources | preference_sources | blocked_dataset_sources
    frozen_rows = [row for path in args.frozen for row in read_jsonl(path)]
    frozen_queries = {
        normalize_question(str(row.get("问题", row.get("query", ""))))
        for row in frozen_rows
    }

    eligible = [
        row
        for row in generated
        if source_id(row) not in blocked_sources
        and normalize_question(str(row["问题"])) not in frozen_queries
        and normalize_question(str(row["问题"])) not in blocked_dataset_queries
    ]
    selected: list[dict[str, Any]] = []
    exposure: Counter[str] = Counter()
    for style in STYLES:
        style_rows = [row for row in eligible if row.get("style") == style]
        chosen_ids: set[str] = set()
        for tool_count in range(1, 7):
            bucket = sorted(
                (row for row in style_rows if int(row["tool_count"]) == tool_count),
                key=lambda row: stable_seed(
                    args.seed, style, tool_count, row["id"]
                ),
            )
            for row in bucket[: min(args.per_style // 6, len(bucket))]:
                selected.append(row)
                chosen_ids.add(str(row["id"]))
                exposure.update(str(row["工具"]).split(","))
        shortfall = args.per_style - len(chosen_ids)
        residual = sorted(
            (row for row in style_rows if str(row["id"]) not in chosen_ids),
            key=lambda row: (
                max(exposure[tool] + 1 for tool in str(row["工具"]).split(",")),
                sum((exposure[tool] + 1) ** 2 for tool in str(row["工具"]).split(",")),
                stable_seed(args.seed, style, "residual", row["id"]),
            ),
        )
        if len(residual) < shortfall:
            raise ValueError(f"insufficient isolated {style} rows: {len(style_rows)}")
        for row in residual[:shortfall]:
            selected.append(row)
            exposure.update(str(row["工具"]).split(","))

    selected.sort(key=lambda row: stable_seed(args.seed, "final", row["id"]))
    output_rows = []
    for row in selected:
        copied = dict(row)
        copied["source"] = "assignment3_current_rag_freeform_dev"
        copied["generation_split"] = "dev"
        copied["construction"] = {
            **dict(row.get("construction") or {}),
            "dataset_role": "current_rag_residual_dpo_selection",
            "current_source_isolation_filter": True,
        }
        output_rows.append(copied)
    write_jsonl(args.out, output_rows)

    selected_sources = {source_id(row) for row in output_rows}
    selected_queries = {
        normalize_question(str(row["问题"])) for row in output_rows
    }
    represented_tools = {
        tool for row in output_rows for tool in str(row["工具"]).split(",")
    }
    errors = []
    if len(output_rows) != 2 * args.per_style:
        errors.append("row_count")
    if len(selected_sources) != len(output_rows):
        errors.append("duplicate_source")
    if len(selected_queries) != len(output_rows):
        errors.append("duplicate_query")
    if selected_sources & blocked_sources:
        errors.append("blocked_source_overlap")
    if selected_queries & blocked_dataset_queries:
        errors.append("blocked_dataset_query_overlap")
    if selected_queries & frozen_queries:
        errors.append("frozen_query_overlap")
    if len(represented_tools) != 100:
        errors.append("tool_coverage")
    audit = {
        "schema_version": 1,
        "status": "complete" if not errors else "failed",
        "strict_ok": not errors,
        "role": "checkpoint selection only; never preference or SFT training data",
        "counts": {
            "generated": len(generated),
            "eligible_after_current_source_filter": len(eligible),
            "selected": len(output_rows),
            "by_style": dict(Counter(str(row["style"]) for row in output_rows)),
            "by_style_tool_count": {
                f"{style}:{tool_count}": count
                for (style, tool_count), count in sorted(
                    Counter(
                        (str(row["style"]), int(row["tool_count"]))
                        for row in output_rows
                    ).items()
                )
            },
            "represented_tools": len(represented_tools),
        },
        "isolation": {
            "sft_sources": len(sft_sources),
            "preference_sources": len(preference_sources),
            "blocked_dataset_sources": len(blocked_dataset_sources),
            "selected_blocked_source_overlap": len(selected_sources & blocked_sources),
            "selected_blocked_dataset_query_overlap": len(
                selected_queries & blocked_dataset_queries
            ),
            "selected_frozen_query_overlap": len(selected_queries & frozen_queries),
        },
        "errors": errors,
        "inputs": {
            str(path): sha256_file(path)
            for path in (
                args.generated_dev,
                args.api_train,
                args.sft_runtime,
                *args.preference_data,
                *args.blocked_dataset,
                *args.frozen,
            )
        },
        "output": {str(args.out): sha256_file(args.out)},
    }
    write_json(args.audit_out, audit)
    print(
        f"free-form current dev rows={len(output_rows)} tools={len(represented_tools)} "
        f"strict_ok={audit['strict_ok']}"
    )
    return 0 if audit["strict_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
