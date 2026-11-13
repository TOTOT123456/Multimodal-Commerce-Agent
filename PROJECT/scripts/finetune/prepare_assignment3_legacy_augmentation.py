#!/usr/bin/env python3
"""Namespace and audit a fresh Legacy train-side pool for later DPO mining."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import normalize_question  # noqa: E402
from finetune.tool_trace_common import sha256_file  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def query(row: dict[str, Any]) -> str:
    return normalize_question(str(row.get("问题") or row.get("query") or ""))


def split_tools(row: dict[str, Any]) -> list[str]:
    return [tool.strip() for tool in str(row["工具"]).split(",") if tool.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-train", type=Path, required=True)
    parser.add_argument("--raw-generation-meta", type=Path, required=True)
    parser.add_argument("--canonical-train", type=Path, required=True)
    parser.add_argument("--frozen-test", type=Path, required=True)
    parser.add_argument("--blocked-dataset", action="append", type=Path, default=[])
    parser.add_argument("--prefix", default="residual_aug1_")
    parser.add_argument("--expected-rows", type=int, default=24000)
    parser.add_argument("--generator-seed", type=int, default=20260803)
    parser.add_argument("--context-salt", type=int, default=300007)
    parser.add_argument("--expression-profile", default="train")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--meta-out", type=Path, required=True)
    parser.add_argument("--audit-out", type=Path, required=True)
    args = parser.parse_args()

    raw = read_jsonl(args.raw_train)
    raw_meta = read_jsonl(args.raw_generation_meta)
    canonical = read_jsonl(args.canonical_train)
    frozen = read_jsonl(args.frozen_test)
    blocked = [row for path in args.blocked_dataset for row in read_jsonl(path)]
    if args.expected_rows <= 0:
        raise ValueError("--expected-rows must be positive")
    if len(raw) != args.expected_rows:
        raise ValueError(
            f"expected {args.expected_rows} fresh train rows, got {len(raw)}"
        )
    raw_ids = {str(row["id"]) for row in raw}
    meta_by_id = {
        str(row["trace_id"]): row
        for row in raw_meta
        if row.get("split") == "train" and str(row["trace_id"]) in raw_ids
    }
    if len(meta_by_id) != len(raw):
        raise ValueError("generation metadata does not cover fresh train rows")

    output: list[dict[str, Any]] = []
    output_meta: list[dict[str, Any]] = []
    for row in raw:
        old_id = str(row["id"])
        new_id = args.prefix + old_id
        copied = dict(row)
        copied["id"] = new_id
        copied["source"] = "assignment3_context_augmented_legacy_dpo_mining"
        copied["style"] = "legacy_compact"
        copied["source_split"] = "legacy_augmented_train"
        copied["generation_split"] = "train"
        copied["construction"] = {
            "source_trace_id": new_id,
            "type": "fresh_context_legacy_dpo_mining",
            "raw_trace_id": old_id,
        }
        output.append(copied)
        meta = dict(meta_by_id[old_id])
        meta["trace_id"] = new_id
        output_meta.append(meta)

    selected_ids = {str(row["id"]) for row in output}
    selected_queries = {query(row) for row in output}
    canonical_queries = {query(row) for row in canonical}
    frozen_queries = {query(row) for row in frozen}
    blocked_queries = {query(row) for row in blocked}
    represented_tools = {tool for row in output for tool in split_tools(row)}
    checks = {
        "row_count_expected": len(output) == args.expected_rows,
        "unique_id": len(selected_ids) == len(output),
        "unique_query": len(selected_queries) == len(output),
        "query_disjoint_from_canonical_train": not (selected_queries & canonical_queries),
        "query_disjoint_from_frozen_test": not (selected_queries & frozen_queries),
        "query_disjoint_from_all_development_sets": not (selected_queries & blocked_queries),
        "all_100_tools_represented": len(represented_tools) == 100,
    }
    write_jsonl(args.out, output)
    write_jsonl(args.meta_out, output_meta)
    audit = {
        "schema_version": 1,
        "status": "complete" if all(checks.values()) else "failed",
        "strict_ok": all(checks.values()),
        "role": "DPO mining pool only; not a development or frozen set",
        "frozen_labels_used": False,
        "construction": {
            "style": "legacy_compact",
            "generator_seed": args.generator_seed,
            "context_salt": args.context_salt,
            "expression_profile": args.expression_profile,
            "expression_cluster_policy": "training-only clusters; frozen test clusters are excluded",
        },
        "counts": {
            "rows": len(output),
            "by_tool_count": dict(
                sorted(Counter(str(row["tool_count"]) for row in output).items())
            ),
            "represented_tools": len(represented_tools),
        },
        "overlaps": {
            "canonical_train_query": len(selected_queries & canonical_queries),
            "frozen_test_query": len(selected_queries & frozen_queries),
            "development_query": len(selected_queries & blocked_queries),
        },
        "checks": checks,
        "inputs": {
            str(path): sha256_file(path)
            for path in (
                args.raw_train,
                args.raw_generation_meta,
                args.canonical_train,
                args.frozen_test,
                *args.blocked_dataset,
            )
        },
        "outputs": {
            str(args.out): sha256_file(args.out),
            str(args.meta_out): sha256_file(args.meta_out),
        },
    }
    args.audit_out.parent.mkdir(parents=True, exist_ok=True)
    args.audit_out.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"legacy DPO augmentation rows={len(output)} tools={len(represented_tools)} "
        f"strict_ok={audit['strict_ok']}"
    )
    return 0 if audit["strict_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
