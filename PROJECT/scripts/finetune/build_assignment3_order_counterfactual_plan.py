#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build query-free, reorderable target pairs for static trace evaluation."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import CATALOG_PATH, load_catalog, load_jsonl, write_json, write_jsonl  # noqa: E402
from finetune.tool_trace_common import ASSIGNMENT3_DIR, sha256_file, sha256_json  # noqa: E402


SAFE_OPERATION_TYPES = {"read", "compute", "decision"}


def eligible(tools: list[str], catalog: dict[str, dict[str, object]]) -> bool:
    tool_set = set(tools)
    if len(tool_set) != len(tools) or len(tools) < 2:
        return False
    if any(str(catalog[tool_id]["operation_type"]) not in SAFE_OPERATION_TYPES for tool_id in tools):
        return False
    return not any(
        tool_set.intersection(str(value) for value in catalog[tool_id].get("depends_on", []))
        for tool_id in tools
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-plan",
        type=Path,
        default=ASSIGNMENT3_DIR / "api_native_dev_2400" / "target_plan.jsonl",
    )
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-out", type=Path, required=True)
    args = parser.parse_args()

    catalog_rows = load_catalog(args.catalog)
    catalog = {str(row["tool_id"]): row for row in catalog_rows}
    source_rows = load_jsonl(args.source_plan)
    selected: dict[tuple[str, ...], dict[str, object]] = {}
    for row in source_rows:
        tools = [str(value) for value in row.get("tools", [])]
        if not all(tool_id in catalog for tool_id in tools) or not eligible(tools, catalog):
            continue
        key = tuple(sorted(tools))
        current = selected.get(key)
        if current is None or str(row["id"]) < str(current["id"]):
            selected[key] = row

    rows = []
    for index, source in enumerate(sorted(selected.values(), key=lambda row: str(row["id"]))):
        tools_a = [str(value) for value in source["tools"]]
        tools_b = list(reversed(tools_a))
        pair_id = f"assignment3_order_pair_{index:04d}"
        rows.append(
            {
                "pair_id": pair_id,
                "id_a": f"{pair_id}_a",
                "id_b": f"{pair_id}_b",
                "tools_a": tools_a,
                "tools_b": tools_b,
                "tool_count": len(tools_a),
                "domain": str(source.get("domain", "")),
                "combo_seen_bucket": "order_counterfactual",
            }
        )

    write_jsonl(args.output, rows)
    tool_counts = Counter(int(row["tool_count"]) for row in rows)
    errors = [
        row["pair_id"]
        for row in rows
        if set(row["tools_a"]) != set(row["tools_b"])
        or row["tools_a"] == row["tools_b"]
        or not eligible(list(row["tools_a"]), catalog)
    ]
    audit = {
        "audit_version": "assignment3-order-counterfactual-plan-v1",
        "strict_ok": bool(rows) and not errors,
        "pairs": len(rows),
        "queries_to_generate": 2 * len(rows),
        "tool_count": dict(sorted(tool_counts.items())),
        "selection_rule": {
            "source": "query-free API-native target plans only",
            "one_pair_per_unordered_tool_set": True,
            "allowed_operation_types": sorted(SAFE_OPERATION_TYPES),
            "in_set_dependencies_excluded": True,
            "counterfactual_order": "exact_reverse",
        },
        "errors": errors,
        "training_use_allowed": False,
        "hashes": {
            "source_plan": sha256_file(args.source_plan),
            "catalog": sha256_file(args.catalog),
            "output": sha256_file(args.output),
            "output_content": sha256_json(rows),
        },
    }
    write_json(args.audit_out, audit)
    print(
        f"order-counterfactual plan pairs={len(rows)} queries={2 * len(rows)} "
        f"strict_ok={audit['strict_ok']}"
    )
    return 0 if audit["strict_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
