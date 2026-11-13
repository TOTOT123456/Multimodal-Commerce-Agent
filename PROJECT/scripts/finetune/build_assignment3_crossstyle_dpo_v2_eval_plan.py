#!/usr/bin/env python3
"""Build query-free, source-disjoint dev/final plans for cross-style DPO v2."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import (  # noqa: E402
    CATALOG_PATH,
    TOOL_TRACE_DIR,
    catalog_by_id,
    load_catalog,
    load_jsonl,
    split_tools,
    write_json,
    write_jsonl,
)
from finetune.tool_trace_common import sha256_file, stable_seed  # noqa: E402


PROTOCOLS = (
    ("dev", "api_native"),
    ("dev", "legacy_compact"),
    ("final", "api_native"),
    ("final", "legacy_compact"),
)


def randomized_topological_order(
    tools: list[str],
    catalog: dict[str, dict[str, Any]],
    *seed_parts: object,
) -> list[str]:
    remaining = set(tools)
    result: list[str] = []
    step = 0
    while remaining:
        ready = [
            tool_id
            for tool_id in remaining
            if not (
                set(str(value) for value in catalog[tool_id].get("depends_on", []))
                & remaining
            )
        ]
        if not ready:
            raise ValueError(f"dependency cycle in {sorted(remaining)}")
        ready.sort(key=lambda tool_id: stable_seed(*seed_parts, step, tool_id))
        result.append(ready[0])
        remaining.remove(ready[0])
        step += 1
    return result


def dependency_violations(
    tools: list[str],
    catalog: dict[str, dict[str, Any]],
) -> list[str]:
    positions = {tool_id: index for index, tool_id in enumerate(tools)}
    return [
        f"{dependency}->{tool_id}"
        for tool_id in tools
        for dependency in catalog[tool_id].get("depends_on", [])
        if dependency in positions and positions[dependency] > positions[tool_id]
    ]


def blocked_source_ids(paths: list[Path]) -> set[str]:
    blocked: set[str] = set()
    for path in paths:
        for row in load_jsonl(path):
            construction = row.get("construction")
            if isinstance(construction, dict) and construction.get("source_trace_id"):
                blocked.add(str(construction["source_trace_id"]))
    return blocked


def select_rows(
    pool: list[dict[str, Any]],
    count: int,
    exposure: Counter[str],
    selected_ids: set[str],
    *seed_parts: object,
) -> list[dict[str, Any]]:
    remaining = [row for row in pool if str(row["id"]) not in selected_ids]
    chosen: list[dict[str, Any]] = []
    for local_index in range(count):
        if not remaining:
            raise ValueError(f"insufficient source rows for {seed_parts}")
        row = min(
            remaining,
            key=lambda item: (
                max(exposure[tool_id] + 1 for tool_id in split_tools(item["工具"])),
                sum(
                    (exposure[tool_id] + 1) ** 2
                    for tool_id in split_tools(item["工具"])
                ),
                stable_seed(*seed_parts, local_index, item["id"]),
            ),
        )
        remaining.remove(row)
        chosen.append(row)
        selected_ids.add(str(row["id"]))
        exposure.update(split_tools(row["工具"]))
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-train", type=Path, default=TOOL_TRACE_DIR / "train.jsonl")
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    parser.add_argument(
        "--blocked-source-plan",
        action="append",
        type=Path,
        default=[],
        help="Plan whose construction.source_trace_id values cannot be reused.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-style", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20_260_729)
    args = parser.parse_args()

    if args.per_style <= 0 or args.per_style % 6:
        raise SystemExit("--per-style must be positive and divisible by six")
    catalog_rows = load_catalog(args.catalog)
    catalog = catalog_by_id(catalog_rows)
    formal_rows = load_jsonl(args.formal_train)
    if len(catalog) != 100 or len(formal_rows) != 24_000:
        raise SystemExit("expected established 100-tool catalog and 24k formal train")

    blocked = blocked_source_ids(args.blocked_source_plan)
    by_count: dict[int, list[dict[str, Any]]] = {value: [] for value in range(1, 7)}
    for row in formal_rows:
        tools = split_tools(row["工具"])
        if len(tools) in by_count and str(row["id"]) not in blocked:
            by_count[len(tools)].append(row)

    per_count = args.per_style // 6
    selected_ids: set[str] = set()
    output_paths: dict[str, Path] = {}
    plans: dict[str, list[dict[str, Any]]] = {}
    for role, style in PROTOCOLS:
        exposure: Counter[str] = Counter()
        plan_rows: list[dict[str, Any]] = []
        for tool_count in range(1, 7):
            sources = select_rows(
                by_count[tool_count],
                per_count,
                exposure,
                selected_ids,
                args.seed,
                role,
                style,
                tool_count,
            )
            for local_index, source in enumerate(sources):
                source_tools = split_tools(source["工具"])
                source_order_invalid = bool(
                    dependency_violations(source_tools, catalog)
                )
                tools = (
                    randomized_topological_order(
                        source_tools,
                        catalog,
                        "a3-dpo-v2-eval-order",
                        args.seed,
                        role,
                        style,
                        source["id"],
                    )
                    if len(source_tools) > 1
                    and (style == "api_native" or source_order_invalid)
                    else list(source_tools)
                )
                trace_id = (
                    f"a3_dpo2_{role}_{style}_{tool_count}_{local_index:03d}"
                )
                domains = sorted({str(catalog[tool_id]["domain"]) for tool_id in tools})
                plan_rows.append(
                    {
                        "id": trace_id,
                        "tools": tools,
                        "tool_count": len(tools),
                        "domain": ",".join(domains),
                        "combo_seen_bucket": style,
                        "style": style,
                        "construction": {
                            "type": "dpo_v2_sealed_eval",
                            "dataset_role": role,
                            "style_profile": style,
                            "source_trace_id": str(source["id"]),
                            "source_tools": source_tools,
                            "source_combo_seen_bucket": source.get(
                                "combo_seen_bucket", ""
                            ),
                            "source_order_dependency_invalid": source_order_invalid,
                            "target_order_randomized": tools != source_tools,
                            "seed": args.seed,
                        },
                    }
                )
        key = f"{role}_{style}"
        path = args.output_dir / f"{key}_plan.jsonl"
        write_jsonl(path, plan_rows)
        output_paths[key] = path
        plans[key] = plan_rows

    all_rows = [row for rows in plans.values() for row in rows]
    all_source_ids = [
        str(row["construction"]["source_trace_id"]) for row in all_rows
    ]
    query_fields = {"query", "question", "问题"} & set().union(
        *(set(row) for row in all_rows)
    )
    dependency_errors = {
        str(row["id"]): dependency_violations(row["tools"], catalog)
        for row in all_rows
        if dependency_violations(row["tools"], catalog)
    }
    plan_counts = {
        key: {
            "rows": len(rows),
            "tool_count": dict(
                sorted(Counter(int(row["tool_count"]) for row in rows).items())
            ),
            "tool_coverage": len(
                {tool_id for row in rows for tool_id in row["tools"]}
            ),
            "source_trace_ids": len(
                {str(row["construction"]["source_trace_id"]) for row in rows}
            ),
        }
        for key, rows in plans.items()
    }
    strict_ok = (
        len(all_rows) == len(PROTOCOLS) * args.per_style
        and len({str(row["id"]) for row in all_rows}) == len(all_rows)
        and len(set(all_source_ids)) == len(all_source_ids)
        and not (set(all_source_ids) & blocked)
        and not query_fields
        and not dependency_errors
        and all(
            row["rows"] == args.per_style
            and row["tool_count"] == {value: per_count for value in range(1, 7)}
            and row["tool_coverage"] == 100
            for row in plan_counts.values()
        )
    )
    audit = {
        "audit_version": "assignment3-crossstyle-dpo-v2-sealed-eval-plan-v1",
        "protocol_frozen_at": "2026-07-29",
        "strict_ok": strict_ok,
        "training_use_allowed": False,
        "purpose": {
            "dev": "DPO-v2 checkpoint selection and guardrails only",
            "final": "one-shot DPO-v2 final evaluation only",
        },
        "protocol": {
            "styles": ["api_native", "legacy_compact"],
            "roles": ["dev", "final"],
            "per_style_per_role": args.per_style,
            "per_tool_count_per_style": per_count,
            "tool_counts": [1, 2, 3, 4, 5, 6],
            "source_trace_disjoint_across_all_four_cells": True,
            "api_native_target_order": "seeded dependency-valid topological order",
            "legacy_compact_target_order": "formal-train business order",
            "query_generation": "independent GPT generator plus independent judge; no query template",
        },
        "counts": plan_counts,
        "isolation": {
            "all_rows": len(all_rows),
            "unique_ids": len({str(row["id"]) for row in all_rows}),
            "unique_source_trace_ids": len(set(all_source_ids)),
            "blocked_source_trace_ids": len(blocked),
            "blocked_source_overlap": len(set(all_source_ids) & blocked),
            "query_fields_present": sorted(query_fields),
            "dependency_error_count": len(dependency_errors),
            "dependency_errors": dependency_errors,
        },
        "inputs": {
            str(args.formal_train): sha256_file(args.formal_train),
            str(args.catalog): sha256_file(args.catalog),
            **{
                str(path): sha256_file(path)
                for path in args.blocked_source_plan
            },
        },
        "outputs": {
            key: {"path": str(path), "sha256": sha256_file(path)}
            for key, path in output_paths.items()
        },
        "final_seal": {
            "sealed_before_preference_mining": True,
            "selection_or_training_use_forbidden": True,
            "post_final_tuning_forbidden": True,
        },
    }
    write_json(args.output_dir / "plan_audit.json", audit)
    print(
        f"sealed eval plans rows={len(all_rows)} "
        f"unique_sources={len(set(all_source_ids))} strict_ok={strict_ok}"
    )
    return 0 if strict_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
