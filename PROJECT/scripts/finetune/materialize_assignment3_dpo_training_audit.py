#!/usr/bin/env python3
"""Materialize the exact DPO source rows and audit preference provenance."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import (  # noqa: E402
    CATALOG_PATH,
    catalog_by_id,
    load_catalog,
    normalize_question,
    split_tools,
)
from finetune.tool_trace_common import (  # noqa: E402
    canonical_target,
    render_messages,
    sha256_file,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def source_id(row: dict[str, Any]) -> str:
    construction = row.get("construction") or {}
    meta = row.get("meta") or {}
    return str(
        construction.get("source_trace_id")
        or meta.get("source_trace_id")
        or row.get("id")
        or ""
    )


def preference_query(row: dict[str, Any]) -> str:
    for message in reversed(row.get("messages") or []):
        if message.get("role") != "user":
            continue
        content = str(message.get("content", ""))
        marker = "\n用户问题：\n"
        return content.rsplit(marker, 1)[-1] if marker in content else content
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preferences", type=Path, required=True)
    parser.add_argument("--mining-gold", type=Path, required=True)
    parser.add_argument("--mining-gold-extra", action="append", type=Path, default=[])
    parser.add_argument("--mining-meta", type=Path, required=True)
    parser.add_argument("--mining-meta-extra", action="append", type=Path, default=[])
    parser.add_argument("--reference-meta", action="append", type=Path, default=[])
    parser.add_argument(
        "--protected-dataset", action="append", type=Path, default=[]
    )
    parser.add_argument(
        "--previous-preferences", action="append", type=Path, default=[]
    )
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    parser.add_argument("--expected-pairs", type=int, required=True)
    parser.add_argument("--expected-parent", type=Path, required=True)
    parser.add_argument("--selected-gold-out", type=Path, required=True)
    parser.add_argument("--combined-meta-out", type=Path, required=True)
    parser.add_argument("--audit-out", type=Path, required=True)
    args = parser.parse_args()

    preferences = read_jsonl(args.preferences)
    mining_gold_paths = [args.mining_gold, *args.mining_gold_extra]
    mining_meta_paths = [args.mining_meta, *args.mining_meta_extra]
    mining_gold = [row for path in mining_gold_paths for row in read_jsonl(path)]
    mining_meta = [row for path in mining_meta_paths for row in read_jsonl(path)]
    gold_by_id = {str(row["id"]): row for row in mining_gold}
    meta_by_id = {str(row["trace_id"]): row for row in mining_meta}
    if len(gold_by_id) != len(mining_gold) or len(meta_by_id) != len(mining_meta):
        raise ValueError("duplicate mining gold or metadata IDs")

    catalog = catalog_by_id(load_catalog(args.catalog))
    expected_parent = str(args.expected_parent.resolve())
    selected_gold: list[dict[str, Any]] = []
    selected_meta: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    trace_ids: list[str] = []
    source_ids: list[str] = []
    queries: list[str] = []
    for preference in preferences:
        meta = preference.get("meta") or {}
        trace_id = str(meta.get("trace_id", ""))
        source = gold_by_id.get(trace_id)
        generation = meta_by_id.get(trace_id)
        reasons: list[str] = []
        if source is None:
            reasons.append("missing_mining_gold")
        if generation is None:
            reasons.append("missing_mining_meta")
        if source is not None:
            gold_tools = split_tools(source["工具"])
            candidates = [str(value) for value in meta.get("candidate_tools") or []]
            expected_messages = render_messages(
                str(source["问题"]), candidates, catalog, "rag"
            )
            if preference.get("messages", [])[:-1] != expected_messages:
                reasons.append("prompt_mismatch")
            if not preference.get("messages") or preference["messages"][-1].get(
                "content"
            ) != canonical_target(gold_tools):
                reasons.append("chosen_mismatch")
            if not set(gold_tools) <= set(candidates) or len(candidates) != 10:
                reasons.append("candidate_protocol")
            if int(meta.get("tool_count", -1)) != len(gold_tools):
                reasons.append("tool_count_mismatch")
            if preference_query(preference) != source["问题"]:
                reasons.append("query_mismatch")
        if str(Path(str(meta.get("parent_adapter", ""))).resolve()) != expected_parent:
            reasons.append("parent_adapter_mismatch")
        if meta.get("candidate_order_variant") != "original":
            reasons.append("non_deployed_candidate_order")
        if meta.get("semantic_pair") is not True or meta.get("synthetic") is not False:
            reasons.append("non_semantic_or_synthetic")
        if reasons:
            failures.append({"trace_id": trace_id, "reasons": ",".join(reasons)})
            continue
        selected_gold.append(source)
        selected_meta.append(generation)
        trace_ids.append(trace_id)
        source_ids.append(str(meta.get("source_trace_id", "")))
        queries.append(normalize_question(str(source["问题"])))

    protected_ids: set[str] = set()
    protected_sources: set[str] = set()
    protected_queries: set[str] = set()
    protected_rows = 0
    for path in args.protected_dataset:
        rows = read_jsonl(path)
        protected_rows += len(rows)
        protected_ids.update(str(row.get("id", "")) for row in rows)
        protected_sources.update(source_id(row) for row in rows)
        protected_queries.update(
            normalize_question(str(row.get("问题") or row.get("query") or ""))
            for row in rows
        )

    previous_ids: set[str] = set()
    previous_sources: set[str] = set()
    previous_queries: set[str] = set()
    for path in args.previous_preferences:
        for row in read_jsonl(path):
            meta = row.get("meta") or {}
            previous_ids.add(str(meta.get("trace_id", "")))
            previous_sources.add(str(meta.get("source_trace_id", "")))
            previous_queries.add(normalize_question(preference_query(row)))

    combined_meta_by_id: dict[str, dict[str, Any]] = {
        str(row["trace_id"]): row for row in selected_meta
    }
    metadata_conflicts = 0
    for path in args.reference_meta:
        for row in read_jsonl(path):
            trace_id = str(row["trace_id"])
            existing = combined_meta_by_id.get(trace_id)
            if existing is not None and existing != row:
                metadata_conflicts += 1
            else:
                combined_meta_by_id[trace_id] = row

    write_jsonl(args.selected_gold_out, selected_gold)
    write_jsonl(
        args.combined_meta_out,
        [combined_meta_by_id[key] for key in sorted(combined_meta_by_id)],
    )

    selected_id_set = set(trace_ids)
    selected_source_set = set(source_ids)
    selected_query_set = set(queries)
    overlap = {
        "protected_trace_id": len(selected_id_set & protected_ids),
        "protected_source_trace_id": len(selected_source_set & protected_sources),
        "protected_normalized_query": len(selected_query_set & protected_queries),
        "previous_preference_trace_id": len(selected_id_set & previous_ids),
        "previous_preference_source_trace_id": len(
            selected_source_set & previous_sources
        ),
        "previous_preference_normalized_query": len(
            selected_query_set & previous_queries
        ),
    }
    checks = {
        "expected_pair_count": len(preferences) == args.expected_pairs,
        "all_preferences_valid": not failures,
        "unique_trace_id": len(selected_id_set) == len(preferences),
        "unique_source_trace_id": len(selected_source_set) == len(preferences),
        "unique_normalized_query": len(selected_query_set) == len(preferences),
        "no_protected_or_previous_overlap": not any(overlap.values()),
        "metadata_no_conflicts": metadata_conflicts == 0,
        "all_100_tools_represented": len(
            {tool for row in selected_gold for tool in split_tools(row["工具"])}
        )
        == 100,
    }
    audit = {
        "schema_version": 1,
        "status": "complete" if all(checks.values()) else "failed",
        "strict_ok": all(checks.values()),
        "role": "post-construction DPO provenance and isolation audit",
        "frozen_data_used_for_training_or_selection": False,
        "counts": {
            "preferences": len(preferences),
            "selected_gold": len(selected_gold),
            "protected_rows": protected_rows,
            "by_tool_count": dict(
                sorted(Counter(str(row["tool_count"]) for row in selected_gold).items())
            ),
            "by_error_type": dict(
                sorted(
                    Counter(
                        error
                        for row in preferences
                        for error in (row.get("meta") or {}).get("error_types", [])
                    ).items()
                )
            ),
            "by_domain": dict(
                sorted(
                    Counter(
                        domain
                        for row in selected_gold
                        for domain in str(row.get("domain", "")).split(",")
                        if domain
                    ).items()
                )
            ),
            "represented_tools": len(
                {tool for row in selected_gold for tool in split_tools(row["工具"])}
            ),
        },
        "overlap": overlap,
        "metadata_conflicts": metadata_conflicts,
        "checks": checks,
        "failures": failures[:100],
        "inputs": {
            str(path): sha256_file(path)
            for path in (
                args.preferences,
                *mining_gold_paths,
                *mining_meta_paths,
                args.catalog,
                *args.reference_meta,
                *args.protected_dataset,
                *args.previous_preferences,
            )
        },
        "outputs": {
            str(args.selected_gold_out): sha256_file(args.selected_gold_out),
            str(args.combined_meta_out): sha256_file(args.combined_meta_out),
        },
    }
    write_json(args.audit_out, audit)
    print(
        f"DPO provenance pairs={len(preferences)} overlap={overlap} "
        f"strict_ok={audit['strict_ok']}"
    )
    return 0 if audit["strict_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
