#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit API-native dev isolation, train GT evidence, and SFT compact records."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import (  # noqa: E402
    CATALOG_PATH,
    TOOL_TRACE_DIR,
    catalog_by_id,
    load_catalog,
    load_jsonl,
    normalize_question,
    split_tools,
    write_json,
    write_jsonl,
)
from finetune.tool_trace_common import ASSIGNMENT3_DIR, sha256_file, sha256_json, stable_seed  # noqa: E402


def context_signature(meta: dict[str, Any]) -> str:
    query = str(meta.get("query", ""))
    phrases = sorted((meta.get("expression_phrases") or {}).values(), key=lambda value: len(str(value)), reverse=True)
    for phrase in phrases:
        if phrase:
            query = query.replace(str(phrase), "{PHRASE}")
    return normalize_question(query)


def split_sets(
    rows: list[dict[str, Any]],
    meta_by_id: dict[str, dict[str, Any]],
) -> dict[str, set[Any]]:
    def meta_for(row: dict[str, Any]) -> dict[str, Any]:
        return meta_by_id.get(row["id"], {"query": row["问题"]})

    return {
        "queries": {normalize_question(row["问题"]) for row in rows},
        "ids": {row["id"] for row in rows},
        "clusters": {
            (tool_id, cluster_id)
            for row in rows
            for tool_id, cluster_id in (meta_for(row).get("expression_cluster_ids") or {}).items()
        },
        "phrases": {
            (tool_id, normalize_question(str(phrase)))
            for row in rows
            for tool_id, phrase in (meta_for(row).get("expression_phrases") or {}).items()
        },
        "contexts": {context_signature(meta_for(row)) for row in rows},
    }


def audit_evidence(
    rows: list[dict[str, Any]],
    meta_by_id: dict[str, dict[str, Any]],
    catalog_ids: set[str],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    failures = []
    counts: Counter[str] = Counter()
    for row in rows:
        meta = meta_by_id.get(row["id"])
        reasons = []
        gold = split_tools(row["工具"])
        if not meta:
            reasons.append("missing_meta")
        else:
            meta_tools = meta.get("tools") or []
            evidence = meta.get("tool_evidence") or {}
            phrases = meta.get("expression_phrases") or {}
            if meta.get("query") != row["问题"]:
                reasons.append("query_mismatch")
            if meta_tools != gold:
                reasons.append("tools_mismatch")
            if int(meta.get("tool_count", -1)) != len(gold) or int(row.get("tool_count", -1)) != len(gold):
                reasons.append("tool_count_mismatch")
            if set(evidence) != set(gold):
                reasons.append("evidence_tool_set_mismatch")
            if set(phrases) != set(gold):
                reasons.append("phrase_tool_set_mismatch")
            for tool_id in gold:
                phrase = str(evidence.get(tool_id, ""))
                if not phrase or phrase not in row["问题"]:
                    reasons.append(f"evidence_not_in_query:{tool_id}")
                if phrase != str(phrases.get(tool_id, "")):
                    reasons.append(f"evidence_phrase_mismatch:{tool_id}")
        if len(gold) != len(set(gold)):
            reasons.append("duplicate_gold")
        if not set(gold) <= catalog_ids:
            reasons.append("unknown_gold")
        if reasons:
            failures.append({"id": row["id"], "reasons": reasons})
            counts.update(reasons)
    return failures, counts


def audit_api_rows(
    rows: list[dict[str, Any]],
    catalog_ids: set[str],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    failures = []
    counts: Counter[str] = Counter()
    for row in rows:
        reasons = []
        gold = split_tools(row["工具"])
        if len(gold) != len(set(gold)):
            reasons.append("duplicate_gold")
        if not set(gold) <= catalog_ids:
            reasons.append("unknown_gold")
        if int(row.get("tool_count", -1)) != len(gold):
            reasons.append("tool_count_mismatch")
        if not str(row.get("问题", "")).strip():
            reasons.append("empty_query")
        if reasons:
            failures.append({"id": row.get("id"), "reasons": reasons})
            counts.update(reasons)
    return failures, counts


def coverage_labels(row: dict[str, Any], by_id: dict[str, dict[str, Any]]) -> set[str]:
    tools = split_tools(row["工具"])
    labels = {
        f"tool_count:{len(tools)}",
        f"kind:{'single' if len(tools) == 1 else 'multi'}",
        f"combo:{row.get('combo_seen_bucket', '')}",
    }
    labels.update(f"tool:{tool_id}" for tool_id in tools)
    labels.update(f"domain:{by_id[tool_id]['domain']}" for tool_id in tools)
    return labels


def stratified_sample(
    rows: list[dict[str, Any]],
    count: int,
    by_id: dict[str, dict[str, Any]],
    split_name: str,
) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: stable_seed(split_name, row["id"], "gt-review"))
    required = set().union(*(coverage_labels(row, by_id) for row in rows))
    selected = []
    selected_ids = set()
    covered = set()
    while required - covered:
        missing = required - covered
        candidate = max(
            (row for row in ordered if row["id"] not in selected_ids),
            key=lambda row: (
                len(coverage_labels(row, by_id) & missing),
                -stable_seed(row["id"], "gt-review-greedy"),
            ),
        )
        selected.append(candidate)
        selected_ids.add(candidate["id"])
        covered.update(coverage_labels(candidate, by_id))
    for row in ordered:
        if len(selected) >= count:
            break
        if row["id"] not in selected_ids:
            selected.append(row)
            selected_ids.add(row["id"])
    if len(selected) != count:
        raise SystemExit(f"{split_name}: could only select {len(selected)}/{count} review rows")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    parser.add_argument("--train", type=Path, default=TOOL_TRACE_DIR / "train.jsonl")
    parser.add_argument(
        "--dev",
        type=Path,
        default=ASSIGNMENT3_DIR / "api_native_dev_2400" / "dev.jsonl",
    )
    parser.add_argument("--frozen", type=Path, default=TOOL_TRACE_DIR / "test.jsonl")
    parser.add_argument("--formal-meta", type=Path, default=TOOL_TRACE_DIR / "_aux" / "generation_meta.jsonl")
    parser.add_argument(
        "--dev-isolation-audit",
        type=Path,
        default=ASSIGNMENT3_DIR / "api_native_dev_2400" / "isolation_audit.json",
    )
    parser.add_argument("--compact-sft", type=Path, default=ASSIGNMENT3_DIR / "sft_train_compact.jsonl")
    parser.add_argument("--global-ids", type=Path, default=ASSIGNMENT3_DIR / "global_view_ids.json")
    parser.add_argument("--prompt-length-stats", type=Path, default=ASSIGNMENT3_DIR / "prompt_length_stats.json")
    parser.add_argument(
        "--generation-leakage-audit",
        type=Path,
        default=TOOL_TRACE_DIR / "_aux" / "generation_leakage_audit.json",
    )
    parser.add_argument("--out", type=Path, default=ASSIGNMENT3_DIR / "gt_quality_audit.json")
    parser.add_argument("--review-out", type=Path, default=ASSIGNMENT3_DIR / "gt_review_sample.jsonl")
    parser.add_argument("--revisions-out", type=Path, default=ASSIGNMENT3_DIR / "gt_revisions.jsonl")
    parser.add_argument("--train-review-count", type=int, default=400)
    parser.add_argument("--dev-review-count", type=int, default=0)
    args = parser.parse_args()

    catalog = load_catalog(args.catalog)
    by_id = catalog_by_id(catalog)
    catalog_ids = set(by_id)
    train = load_jsonl(args.train)
    dev = load_jsonl(args.dev)
    frozen = load_jsonl(args.frozen)
    formal_meta = load_jsonl(args.formal_meta)
    meta_by_id = {row["trace_id"]: row for row in formal_meta}
    if len(meta_by_id) != len(formal_meta):
        raise SystemExit("duplicate trace id in formal generation meta")
    for row in train + frozen:
        if row["id"] not in meta_by_id:
            raise SystemExit(f"missing generation meta: {row['id']}")
    dev_isolation = json.loads(args.dev_isolation_audit.read_text(encoding="utf-8"))

    split_rows = {"train": train, "dev": dev, "frozen": frozen}
    sets = {name: split_sets(rows, meta_by_id) for name, rows in split_rows.items()}
    pairwise = {}
    for left, right in [("train", "dev"), ("train", "frozen"), ("dev", "frozen")]:
        pairwise[f"{left}_{right}"] = {
            key: len(sets[left][key] & sets[right][key])
            for key in ["queries", "ids", "clusters", "phrases", "contexts"]
        }

    train_failures, train_failure_types = audit_evidence(train, meta_by_id, catalog_ids)
    dev_failures, dev_failure_types = audit_api_rows(dev, catalog_ids)
    compact = load_jsonl(args.compact_sft)
    compact_trace_ids = {row["trace_id"] for row in compact}
    dev_ids = sets["dev"]["ids"]
    frozen_ids = sets["frozen"]["ids"]
    dev_queries = sets["dev"]["queries"]
    frozen_queries = sets["frozen"]["queries"]
    compact_query_norms = {normalize_question(row["query"]) for row in compact}
    rag_counts = Counter(row["trace_id"] for row in compact if row["view"] == "rag")
    global_ids_payload = json.loads(args.global_ids.read_text(encoding="utf-8"))
    generation_leakage = json.loads(args.generation_leakage_audit.read_text(encoding="utf-8"))
    global_ids = set(global_ids_payload["trace_ids"])
    compact_global_ids = {row["trace_id"] for row in compact if row["view"] == "global"}
    runtime_hashes = Counter(row["prompt_hash"] for row in compact)

    review_rows = []
    for split_name, rows, count in [
        ("train", train, args.train_review_count),
        ("dev", dev, args.dev_review_count),
    ]:
        if count == 0:
            continue
        for row in stratified_sample(rows, count, by_id, split_name):
            meta = meta_by_id.get(row["id"], {})
            review_rows.append(
                {
                    "id": row["id"],
                    "split": split_name,
                    "query": row["问题"],
                    "gold_tools": split_tools(row["工具"]),
                    "tool_evidence": meta.get("tool_evidence", {}),
                    "tool_count": row["tool_count"],
                    "domains": row["domain"].split(","),
                    "combo_seen_bucket": row["combo_seen_bucket"],
                    "automatic_checks": {
                        "all_gold_have_evidence": (
                            split_name != "train"
                            or set(meta.get("tool_evidence", {}))
                            == set(split_tools(row["工具"]))
                        ),
                        "all_evidence_in_query": (
                            split_name != "train"
                            or all(
                                phrase in row["问题"]
                                for phrase in meta.get("tool_evidence", {}).values()
                            )
                        ),
                        "tool_count_matches": len(split_tools(row["工具"])) == int(row["tool_count"]),
                    },
                    "semantic_review": {
                        "status": "pending_agent_review",
                        "reviewer_type": "",
                        "notes": "",
                    },
                }
            )
    write_jsonl(args.review_out, review_rows)
    if not args.revisions_out.exists():
        write_jsonl(args.revisions_out, [])

    review_coverage = {
        "total": len(review_rows),
        "by_split": dict(Counter(row["split"] for row in review_rows)),
        "tools": len({tool_id for row in review_rows for tool_id in row["gold_tools"]}),
        "domains": len({domain for row in review_rows for domain in row["domains"]}),
        "tool_counts": sorted({int(row["tool_count"]) for row in review_rows}),
        "combo_buckets": sorted({row["combo_seen_bucket"] for row in review_rows}),
        "automatic_check_failures": sum(
            not all(row["automatic_checks"].values()) for row in review_rows
        ),
        "semantic_review_status": "pending_agent_review",
    }
    audit = {
        "status": "automatic_audit_complete_semantic_review_pending",
        "strict_automatic_ok": True,
        "split_counts": {name: len(rows) for name, rows in split_rows.items()},
        "pairwise_isolation": pairwise,
        "legacy_train_frozen_context_risk": {
            "context_signature_overlap": pairwise["train_frozen"]["contexts"],
            "context_skeleton_matcher_all_hit_at_10": generation_leakage[
                "rule_retrievers"
            ]["context_skeleton_matcher"]["@10"]["AllHit@K"],
            "existing_generation_leakage_strict_ok": generation_leakage["strict_ok"],
            "interpretation": (
                "The immutable pre-assignment train/frozen split shares non-intent "
                "context scaffolds. Assignment-3 dev is isolated from both splits; "
                "the existing leakage audit gates the legacy risk by predictive power."
            ),
        },
        "evidence_alignment": {
            "train": {
                "rows": len(train),
                "failures": len(train_failures),
                "failure_types": dict(train_failure_types),
                "examples": train_failures[:20],
            },
            "dev": {
                "rows": len(dev),
                "failures": len(dev_failures),
                "failure_types": dict(dev_failure_types),
                "examples": dev_failures[:20],
            },
        },
        "sft_leakage": {
            "compact_dev_trace_overlap": len(compact_trace_ids & dev_ids),
            "compact_frozen_trace_overlap": len(compact_trace_ids & frozen_ids),
            "compact_dev_query_overlap": len(compact_query_norms & dev_queries),
            "compact_frozen_query_overlap": len(compact_query_norms & frozen_queries),
        },
        "sft_structure": {
            "compact_rows": len(compact),
            "rag_rows": sum(row["view"] == "rag" for row in compact),
            "global_rows": sum(row["view"] == "global" for row in compact),
            "rag_trace_count_min": min(rag_counts.values()) if rag_counts else 0,
            "rag_trace_count_max": max(rag_counts.values()) if rag_counts else 0,
            "missing_rag_trace_ids": sorted(sets["train"]["ids"] - set(rag_counts))[:20],
            "global_ids_exact_match": global_ids == compact_global_ids,
            "unique_prompt_hashes": len(runtime_hashes),
        },
        "review_sample": review_coverage,
        "revisions": {
            "path": str(args.revisions_out),
            "count": len(load_jsonl(args.revisions_out)),
            "policy": "No source split is silently overwritten; every correction must be explicit.",
        },
        "hashes": {
            str(args.catalog): sha256_file(args.catalog),
            str(args.train): sha256_file(args.train),
            str(args.dev): sha256_file(args.dev),
            str(args.frozen): sha256_file(args.frozen),
            str(args.formal_meta): sha256_file(args.formal_meta),
            str(args.dev_isolation_audit): sha256_file(args.dev_isolation_audit),
            str(args.compact_sft): sha256_file(args.compact_sft),
            str(args.global_ids): sha256_file(args.global_ids),
            str(args.prompt_length_stats): sha256_file(args.prompt_length_stats),
            str(args.generation_leakage_audit): sha256_file(args.generation_leakage_audit),
            str(args.review_out): sha256_file(args.review_out),
            str(args.revisions_out): sha256_file(args.revisions_out),
        },
    }
    assignment3_dev_isolated = all(
        value == 0
        for pair_name in ("train_dev", "dev_frozen")
        for value in pairwise[pair_name].values()
    )
    legacy_identity_and_expression_isolated = all(
        pairwise["train_frozen"][key] == 0
        for key in ("queries", "ids", "clusters", "phrases")
    )
    audit["strict_automatic_ok"] = (
        assignment3_dev_isolated
        and legacy_identity_and_expression_isolated
        and generation_leakage["strict_ok"]
        and dev_isolation.get("strict_ok") is True
        and not train_failures
        and not dev_failures
        and all(value == 0 for value in audit["sft_leakage"].values())
        and audit["sft_structure"]["rag_rows"] == len(train)
        and audit["sft_structure"]["rag_trace_count_min"] == 1
        and audit["sft_structure"]["rag_trace_count_max"] == 1
        and not audit["sft_structure"]["missing_rag_trace_ids"]
        and audit["sft_structure"]["global_ids_exact_match"]
        and review_coverage["automatic_check_failures"] == 0
        and review_coverage["by_split"].get("train") == args.train_review_count
        and review_coverage["by_split"].get("dev", 0) == args.dev_review_count
        and review_coverage["tools"] == len(catalog)
        and review_coverage["domains"] == 8
        and review_coverage["tool_counts"] == [1, 2, 3, 4, 5, 6]
    )
    if not audit["strict_automatic_ok"]:
        audit["status"] = "failed"
    write_json(args.out, audit)
    print(
        f"training audit automatic_ok={audit['strict_automatic_ok']} "
        f"train_evidence_fail={len(train_failures)} dev_evidence_fail={len(dev_failures)} "
        f"review={len(review_rows)}"
    )
    return 0 if audit["strict_automatic_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
