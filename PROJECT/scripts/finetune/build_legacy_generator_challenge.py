#!/usr/bin/env python3
"""Build an audited Legacy-generator challenge split without touching frozen data."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import load_jsonl, split_tools, write_json, write_jsonl  # noqa: E402
from finetune.build_planner_candidate_depth_isolated_dev import (  # noqa: E402
    dataset_combo,
    dataset_query,
    normalize_query,
    runtime_query,
    runtime_target,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_id(row: dict[str, Any]) -> str | None:
    construction = row.get("construction")
    if isinstance(construction, dict) and construction.get("source_trace_id"):
        return str(construction["source_trace_id"])
    if row.get("id"):
        return str(row["id"])
    meta = row.get("meta")
    if isinstance(meta, dict) and meta.get("trace_id"):
        return str(meta["trace_id"])
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-test", type=Path, required=True)
    parser.add_argument("--raw-stats", type=Path, required=True)
    parser.add_argument("--sft-runtime", type=Path, required=True)
    parser.add_argument("--blocked-dataset", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-out", type=Path, required=True)
    parser.add_argument("--id-prefix", default="legacy_generator_challenge")
    args = parser.parse_args()

    raw = load_jsonl(args.raw_test)
    sft = load_jsonl(args.sft_runtime)
    sft_queries = {normalize_query(runtime_query(row)) for row in sft}
    sft_combos = {frozenset(runtime_target(row)) for row in sft}
    sft_source_ids = {
        str(row["meta"]["trace_id"])
        for row in sft
        if isinstance(row.get("meta"), dict) and row["meta"].get("trace_id")
    }
    blocked_queries = set(sft_queries)
    blocked_source_ids = set(sft_source_ids)
    blocked_hashes = {}
    for path in args.blocked_dataset:
        blocked_hashes[str(path)] = sha256_file(path)
        for row in load_jsonl(path):
            query = dataset_query(row)
            if query:
                blocked_queries.add(normalize_query(query))
            identifier = source_id(row)
            if identifier:
                blocked_source_ids.add(identifier)

    rows = []
    for index, row in enumerate(raw):
        rewritten = dict(row)
        original_id = str(row["id"])
        trace_id = f"{args.id_prefix}_{index:04d}"
        rewritten["id"] = trace_id
        rewritten["source"] = "legacy_generator_context_isolated_challenge"
        rewritten["generation_split"] = "challenge"
        rewritten["style"] = "legacy_generator"
        rewritten["construction"] = {
            "type": "legacy_generator_context_isolated_challenge",
            "dataset_role": "challenge_only_not_for_training_or_tuning",
            "panel": "legacy_like",
            "source_trace_id": trace_id,
            "raw_generator_trace_id": original_id,
            "test_context_salt": 200011,
            "multi_seen_combo_ratio": 0.0,
        }
        rows.append(rewritten)

    queries = [normalize_query(str(row["问题"])) for row in rows]
    ids = [str(row["id"]) for row in rows]
    multi = [row for row in rows if int(row["tool_count"]) > 1]
    multi_combos = {frozenset(split_tools(row["工具"])) for row in multi}
    single = [row for row in rows if int(row["tool_count"]) == 1]
    expression_clusters = [
        cluster
        for row in rows
        for cluster in row.get("expression_cluster_ids", {}).values()
    ]
    raw_stats = json.loads(args.raw_stats.read_text(encoding="utf-8"))
    errors = []
    if len(rows) != 1200:
        errors.append(f"row_count:{len(rows)}")
    if len(single) != 600 or len(multi) != 600:
        errors.append(f"single_multi:{len(single)}/{len(multi)}")
    if len(queries) != len(set(queries)):
        errors.append("duplicate_normalized_queries")
    if set(queries) & blocked_queries:
        errors.append("blocked_query_overlap")
    if set(ids) & blocked_source_ids:
        errors.append("blocked_source_id_overlap")
    if multi_combos & sft_combos:
        errors.append("multi_combo_overlap_with_sft")
    if any("_test_" not in cluster for cluster in expression_clusters):
        errors.append("non_test_expression_cluster")
    if float(
        raw_stats.get("split", {}).get(
            "test_multi_rows_with_combo_seen_in_train_rate", 1.0
        )
    ) != 0.0:
        errors.append("raw_generator_seen_combo_rate_not_zero")

    write_jsonl(args.output, rows)
    audit = {
        "schema_version": 1,
        "status": "complete" if not errors else "failed",
        "strict_ok": not errors,
        "role": (
            "Legacy-generator distribution and hard-interference challenge; "
            "never training data and never the 11-variant selection set"
        ),
        "training_use_allowed": False,
        "selection_use_allowed": False,
        "counts": {
            "rows": len(rows),
            "single": len(single),
            "multi": len(multi),
            "tool_count": dict(
                sorted(Counter(int(row["tool_count"]) for row in rows).items())
            ),
            "combo_bucket": dict(
                sorted(Counter(str(row["combo_seen_bucket"]) for row in rows).items())
            ),
            "unique_multi_combos": len(multi_combos),
            "unique_expression_clusters": len(set(expression_clusters)),
        },
        "isolation": {
            "duplicate_normalized_queries": len(queries) - len(set(queries)),
            "query_overlap_with_sft_or_blocked": len(set(queries) & blocked_queries),
            "source_id_overlap_with_sft_or_blocked": len(set(ids) & blocked_source_ids),
            "multi_combo_overlap_with_sft": len(multi_combos & sft_combos),
            "singletons_are_structurally_shared_labels": True,
            "expression_clusters_are_test_only": all(
                "_test_" in cluster for cluster in expression_clusters
            ),
            "context_template_family_reuse": (
                "intentional: this diagnostic targets the Legacy generator distribution; "
                "exact query, source id, non-singleton combination, expression split, and "
                "context values are independently controlled"
            ),
        },
        "fixed_design": {
            "rows": 1200,
            "single_per_tool": 6,
            "multi_rows": 600,
            "multi_sft_seen_ratio": 0.0,
            "test_context_salt": 200011,
            "generator_seed": 20260819,
        },
        "errors": errors,
        "inputs": {
            str(args.raw_test): sha256_file(args.raw_test),
            str(args.raw_stats): sha256_file(args.raw_stats),
            str(args.sft_runtime): sha256_file(args.sft_runtime),
            **blocked_hashes,
        },
        "output": {str(args.output): sha256_file(args.output)},
    }
    write_json(args.audit_out, audit)
    print(
        f"legacy challenge rows={len(rows)} single={len(single)} multi={len(multi)} "
        f"strict_ok={audit['strict_ok']}"
    )
    return 0 if audit["strict_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
