#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create audited, label-free generator inputs for dev or frozen evaluation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import CATALOG_PATH, TOOL_TRACE_DIR, load_catalog, load_jsonl, write_json, write_jsonl  # noqa: E402
from finetune.tool_trace_common import ASSIGNMENT3_DIR, sha256_file, sha256_json  # noqa: E402


QUERY_FIELDS = {"id", "query", "input_hash"}
RAG_FIELDS = {"id", "query", "candidate_tools", "candidate_k", "ranking_hash", "input_hash"}
FORBIDDEN_FIELD_PARTS = {"gold", "工具", "label", "target", "answer"}


def authoritative_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {row["id"]: row for row in rows}
    if len(result) != len(rows):
        raise SystemExit("duplicate ids in authoritative split")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "frozen", "test"], required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--rankings", type=Path)
    parser.add_argument("--candidate-k", type=int, required=True)
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    parser.add_argument("--query-out", type=Path)
    parser.add_argument("--rag-out", type=Path)
    parser.add_argument("--audit-out", type=Path)
    args = parser.parse_args()

    if args.candidate_k not in {5, 10, 20, 30, 50}:
        raise SystemExit("--candidate-k must be one of 5, 10, 20, 30, 50")
    if args.split == "dev":
        api_dir = ASSIGNMENT3_DIR / "api_native_dev_2400"
        args.input = args.input or (api_dir / "dev.jsonl")
        args.rankings = args.rankings or (api_dir / "rag_rankings.jsonl")
        args.query_out = args.query_out or (
            api_dir / f"queries_label_free_k{args.candidate_k}.jsonl"
        )
        args.rag_out = args.rag_out or (
            api_dir / f"rag_k{args.candidate_k}_label_free.jsonl"
        )
        args.audit_out = args.audit_out or (
            api_dir / f"label_free_k{args.candidate_k}_audit.json"
        )
    elif args.split == "frozen":
        args.input = args.input or (TOOL_TRACE_DIR / "test.jsonl")
        default_rankings = (
            Path(__file__).resolve().parents[2]
            / "data"
            / "rag"
            / "_aux"
            / "reranker_finetune"
            / "bge-reranker-base-e1-g8-bs16_rank_fusion_tuned_hybrid"
            / "preds_rank_fusion_w0p6_k10.jsonl"
        )
        args.rankings = args.rankings or default_rankings
        args.query_out = args.query_out or (ASSIGNMENT3_DIR / "frozen_queries_label_free.jsonl")
        args.rag_out = args.rag_out or (
            ASSIGNMENT3_DIR / f"frozen_rag_candidates_k{args.candidate_k}_label_free.jsonl"
        )
        args.audit_out = args.audit_out or (
            ASSIGNMENT3_DIR / f"frozen_label_free_k{args.candidate_k}_audit.json"
        )
    else:
        missing = [
            name
            for name, value in (
                ("--input", args.input),
                ("--rankings", args.rankings),
                ("--query-out", args.query_out),
                ("--rag-out", args.rag_out),
                ("--audit-out", args.audit_out),
            )
            if value is None
        ]
        if missing:
            raise SystemExit(
                "custom test split requires explicit paths: " + ", ".join(missing)
            )

    catalog_ids = {row["tool_id"] for row in load_catalog(args.catalog)}
    authority = load_jsonl(args.input)
    authority_by_id = authoritative_index(authority)
    ranking_rows = load_jsonl(args.rankings)
    ranking_by_id = {row.get("id"): row for row in ranking_rows}
    if len(ranking_by_id) != len(ranking_rows):
        raise SystemExit("duplicate ids in ranking file")
    if set(ranking_by_id) != set(authority_by_id):
        raise SystemExit("ranking ids do not exactly match authoritative split")

    query_rows = []
    rag_rows = []
    candidate_errors = []
    alignment_errors = []
    ranking_file_sha256 = sha256_file(args.rankings)
    for row in authority:
        trace_id = row["id"]
        ranking = ranking_by_id[trace_id]
        if ranking.get("query") != row["问题"]:
            alignment_errors.append(trace_id)
            continue
        full_ranking = ranking.get("pred_tools")
        if not isinstance(full_ranking, list):
            candidate_errors.append({"id": trace_id, "reason": "ranking_not_list"})
            continue
        candidates = full_ranking[: args.candidate_k]
        if (
            len(candidates) != args.candidate_k
            or len(candidates) != len(set(candidates))
            or not set(candidates) <= catalog_ids
        ):
            candidate_errors.append({"id": trace_id, "reason": "invalid_candidates", "values": candidates})
            continue
        query_payload = {"id": trace_id, "query": row["问题"]}
        query_payload["input_hash"] = sha256_json(query_payload)
        ranking_hash = sha256_json(
            {
                "id": trace_id,
                "query": row["问题"],
                "full_ranking": full_ranking,
                "ranking_file_sha256": ranking_file_sha256,
            }
        )
        rag_payload = {
            "id": trace_id,
            "query": row["问题"],
            "candidate_tools": candidates,
            "candidate_k": args.candidate_k,
            "ranking_hash": ranking_hash,
        }
        rag_payload["input_hash"] = sha256_json(rag_payload)
        query_rows.append(query_payload)
        rag_rows.append(rag_payload)

    write_jsonl(args.query_out, query_rows)
    write_jsonl(args.rag_out, rag_rows)
    query_field_errors = [
        {"id": row.get("id"), "fields": sorted(set(row) - QUERY_FIELDS)}
        for row in query_rows
        if set(row) != QUERY_FIELDS
    ]
    rag_field_errors = [
        {"id": row.get("id"), "fields": sorted(set(row) - RAG_FIELDS)}
        for row in rag_rows
        if set(row) != RAG_FIELDS
    ]
    forbidden_hits = [
        {"id": row.get("id"), "field": field}
        for row in query_rows + rag_rows
        for field in row
        if any(part.lower() in field.lower() for part in FORBIDDEN_FIELD_PARTS)
    ]
    ranking_prefix_mismatches = sum(
        rag_row["candidate_tools"] != ranking_by_id[rag_row["id"]]["pred_tools"][: args.candidate_k]
        for rag_row in rag_rows
    )
    audit = {
        "strict_ok": (
            not alignment_errors
            and not candidate_errors
            and not query_field_errors
            and not rag_field_errors
            and not forbidden_hits
            and ranking_prefix_mismatches == 0
            and len(query_rows) == len(authority)
            and len(rag_rows) == len(authority)
        ),
        "split": args.split,
        "candidate_k": args.candidate_k,
        "authority_rows": len(authority),
        "query_rows": len(query_rows),
        "rag_rows": len(rag_rows),
        "id_set_match": {row["id"] for row in query_rows} == set(authority_by_id)
        and {row["id"] for row in rag_rows} == set(authority_by_id),
        "alignment_errors": alignment_errors,
        "candidate_errors": candidate_errors,
        "query_field_errors": query_field_errors,
        "rag_field_errors": rag_field_errors,
        "forbidden_field_hits": forbidden_hits,
        "ranking_prefix_mismatches": ranking_prefix_mismatches,
        "candidate_gold_injection": False,
        "hashes": {
            "catalog_snapshot": sha256_file(args.catalog),
            "authoritative_split": sha256_file(args.input),
            "gold_snapshot": sha256_file(args.input),
            "source_ranking": ranking_file_sha256,
            "query_input": sha256_file(args.query_out),
            "rag_input": sha256_file(args.rag_out),
        },
        "generator_allowed_inputs": [
            str(args.query_out),
            str(args.rag_out),
            str(args.catalog),
            "model checkpoint",
            "prompt config",
        ],
        "generator_forbidden_inputs": [str(args.input), str(args.rankings)],
    }
    write_json(args.audit_out, audit)
    print(
        f"label-free {args.split} rows={len(query_rows)} K={args.candidate_k} "
        f"strict_ok={audit['strict_ok']} -> {args.rag_out}"
    )
    return 0 if audit["strict_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
