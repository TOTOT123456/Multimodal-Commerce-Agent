#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit API-generated paired order counterfactuals before inference."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import CATALOG_PATH, load_catalog, load_jsonl, normalize_question, split_tools, write_json  # noqa: E402
from finetune.tool_trace_common import sha256_file  # noqa: E402


INTERNAL_RE = re.compile(
    r"tool[_ -]?id|工具\s*id|catalog|rag|api|mcp|候选工具|调用工具|内部字段|实现逻辑",
    re.I,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-plan", type=Path, required=True)
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--primary-dev", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plan_rows = load_jsonl(args.pair_plan)
    accepted_rows = load_jsonl(args.accepted)
    rows = load_jsonl(args.input)
    catalog = load_catalog(args.catalog)
    catalog_ids = {str(row["tool_id"]) for row in catalog}
    plan_by_pair = {str(row["pair_id"]): row for row in plan_rows}
    accepted_by_pair = {str(row["pair_id"]): row for row in accepted_rows}
    row_by_id = {str(row["id"]): row for row in rows}

    alignment_errors = []
    for pair_id, plan in plan_by_pair.items():
        accepted = accepted_by_pair.get(pair_id)
        left = row_by_id.get(str(plan["id_a"]))
        right = row_by_id.get(str(plan["id_b"]))
        reasons = []
        if accepted is None:
            reasons.append("missing_accepted_pair")
        elif (accepted.get("judgment") or {}).get("verdict") != "pass":
            reasons.append("accepted_judgment_not_pass")
        if left is None or right is None:
            reasons.append("missing_pair_query")
        else:
            if split_tools(left.get("工具", "")) != [str(value) for value in plan["tools_a"]]:
                reasons.append("tools_a_mismatch")
            if split_tools(right.get("工具", "")) != [str(value) for value in plan["tools_b"]]:
                reasons.append("tools_b_mismatch")
            if left.get("pair_id") != pair_id or right.get("pair_id") != pair_id:
                reasons.append("pair_id_mismatch")
            if left.get("pair_variant") != "a" or right.get("pair_variant") != "b":
                reasons.append("pair_variant_mismatch")
        if set(plan["tools_a"]) != set(plan["tools_b"]) or plan["tools_a"] == plan["tools_b"]:
            reasons.append("invalid_counterfactual_target")
        if reasons:
            alignment_errors.append({"pair_id": pair_id, "reasons": reasons})

    normalized = [normalize_question(str(row["问题"])) for row in rows]
    query_sets = {
        "primary_api_dev": {
            normalize_question(str(row["问题"])) for row in load_jsonl(args.primary_dev)
        },
        "train": {normalize_question(str(row["问题"])) for row in load_jsonl(args.train)},
        "frozen_test": {normalize_question(str(row["问题"])) for row in load_jsonl(args.test)},
    }
    overlap = {name: len(set(normalized) & values) for name, values in query_sets.items()}
    internal_rows = [str(row["id"]) for row in rows if INTERNAL_RE.search(str(row["问题"]))]
    tool_id_rows = [
        str(row["id"])
        for row in rows
        if any(tool_id.casefold() in str(row["问题"]).casefold() for tool_id in catalog_ids)
    ]
    forbidden_provenance = [str(row["id"]) for row in rows if "parent_id" in row]
    payload = {
        "audit_version": "assignment3-order-counterfactual-isolation-audit-v1",
        "strict_ok": False,
        "counts": {
            "pair_plan": len(plan_rows),
            "accepted_pairs": len(accepted_rows),
            "rows": len(rows),
            "unique_ids": len(row_by_id),
            "unique_normalized_queries": len(set(normalized)),
            "tool_count": dict(sorted(Counter(int(row["tool_count"]) for row in rows).items())),
        },
        "alignment_errors": alignment_errors,
        "query_overlap": overlap,
        "internal_leakage": {
            "internal_term_rows": internal_rows,
            "literal_tool_id_rows": tool_id_rows,
            "forbidden_provenance_rows": forbidden_provenance,
        },
        "training_use_allowed": False,
        "hashes": {
            str(path): sha256_file(path)
            for path in (
                args.pair_plan,
                args.accepted,
                args.input,
                args.primary_dev,
                args.train,
                args.test,
                args.catalog,
            )
        },
    }
    payload["strict_ok"] = (
        len(plan_rows) > 0
        and len(accepted_rows) == len(plan_rows)
        and len(rows) == 2 * len(plan_rows)
        and len(row_by_id) == len(rows)
        and len(set(normalized)) == len(rows)
        and not alignment_errors
        and not any(overlap.values())
        and not internal_rows
        and not tool_id_rows
        and not forbidden_provenance
    )
    write_json(args.output, payload)
    print(
        f"order-counterfactual audit pairs={len(plan_rows)} rows={len(rows)} "
        f"strict_ok={payload['strict_ok']} query_overlap={sum(overlap.values())}"
    )
    return 0 if payload["strict_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
