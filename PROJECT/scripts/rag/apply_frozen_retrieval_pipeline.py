#!/usr/bin/env python3
"""Apply one already-selected retrieval route without performing any search."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from search_retrieval_pipeline import (  # noqa: E402
    assert_aligned,
    load_json,
    load_jsonl,
    metrics,
    second_rrf,
    weighted_rrf,
    write_json,
    write_jsonl,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_route(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if {
        "first_stage",
        "reranker",
        "second_stage",
    } <= set(payload):
        first_stage = payload["first_stage"]
        reranker = payload["reranker"]
        second_stage = payload["second_stage"]
        return {
            "first_candidate_n": first_stage["candidate_n_per_source"],
            "first_vector_weight": first_stage["vector_weight"],
            "first_rrf_k": first_stage["rrf_k"],
            "candidate_k": reranker["candidate_k"],
            "second_source_weight": second_stage["source_weight"],
            "second_rrf_k": second_stage["rrf_k"],
        }
    route = payload.get("selected", {}).get("route", payload.get("route", payload))
    required = {
        "first_candidate_n",
        "first_vector_weight",
        "first_rrf_k",
        "candidate_k",
        "second_source_weight",
        "second_rrf_k",
    }
    if not isinstance(route, dict) or not required <= set(route):
        raise ValueError(f"route JSON lacks required fields: {sorted(required)}")
    return {key: route[key] for key in required}


def subgroup_metrics(
    prediction_rows: list[dict],
    rankings: list[list[str]],
    metadata_rows: list[dict],
    field: str,
) -> dict[str, Any]:
    metadata = {str(row["id"]): row for row in metadata_rows}
    indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(prediction_rows):
        value = metadata[str(row["id"])].get(field, "unknown")
        indices[str(value)].append(index)
    result = {}
    for value, selected in sorted(indices.items()):
        subset_rows = [prediction_rows[index] for index in selected]
        subset_rankings = [rankings[index] for index in selected]
        result[value] = {
            "count": len(selected),
            **metrics(subset_rows, subset_rankings, [6, 10, 20, 30, 40, 50]),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vector-preds", type=Path, required=True)
    parser.add_argument("--bm25-preds", type=Path, required=True)
    parser.add_argument("--reranker-preds", type=Path, required=True)
    parser.add_argument("--route-json", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--complete-ranking-to",
        type=int,
        default=0,
        help=(
            "Preserve the exact frozen-route ranking as a prefix and, only when "
            "it is shorter than this depth, append tools from a full-source RRF."
        ),
    )
    args = parser.parse_args()
    if args.complete_ranking_to < 0:
        raise ValueError("--complete-ranking-to must be non-negative")

    vector_rows = load_jsonl(args.vector_preds)
    bm25_rows = load_jsonl(args.bm25_preds)
    reranker_rows = load_jsonl(args.reranker_preds)
    metadata_rows = load_jsonl(args.dataset)
    assert_aligned(vector_rows, bm25_rows, "vector", "bm25")
    assert_aligned(vector_rows, reranker_rows, "vector", "reranker")
    if len(metadata_rows) != len(vector_rows):
        raise ValueError(f"metadata rows mismatch: {len(metadata_rows)} != {len(vector_rows)}")
    metadata_by_id = {str(row["id"]): row for row in metadata_rows}
    if set(metadata_by_id) != {str(row.get("id", "")) for row in vector_rows}:
        raise ValueError("dataset ids do not match prediction ids")

    route = resolve_route(args.route_json)
    candidate_k = int(route["candidate_k"])
    score_maps = []
    for row in reranker_rows:
        tools = row.get("candidate_tools", [])
        scores = row.get("scores", [])
        if len(tools) != len(scores):
            raise ValueError("reranker candidate_tools and scores are not aligned")
        score_maps.append({str(tool): float(score) for tool, score in zip(tools, scores)})
    source_rankings = [
        weighted_rrf(
            vector,
            bm25,
            int(route["first_candidate_n"]),
            float(route["first_vector_weight"]),
            int(route["first_rrf_k"]),
        )
        for vector, bm25 in zip(vector_rows, bm25_rows)
    ]
    reranked = []
    for row_index, (source, score_map) in enumerate(zip(source_rankings, score_maps)):
        source_position = {tool: index for index, tool in enumerate(source)}
        candidates = source[:candidate_k]
        missing = [tool for tool in candidates if tool not in score_map]
        if missing:
            raise ValueError(
                f"reranker scores do not cover frozen Top-{candidate_k} at row "
                f"{row_index}: missing={missing}"
            )
        reranked.append(
            sorted(
                candidates,
                key=lambda tool: (-score_map[tool], source_position[tool], tool),
            )
        )
    canonical_rankings = [
        second_rrf(
            source,
            rerank,
            float(route["second_source_weight"]),
            int(route["second_rrf_k"]),
        )
        for source, rerank in zip(source_rankings, reranked)
    ]
    final_rankings: list[list[str]] = []
    completion_counts: list[int] = []
    for vector, bm25, canonical in zip(vector_rows, bm25_rows, canonical_rankings):
        completed = list(canonical)
        if len(completed) < args.complete_ranking_to:
            full_candidate_n = max(
                len(vector.get("pred_tools", [])),
                len(bm25.get("pred_tools", [])),
            )
            full_source = weighted_rrf(
                vector,
                bm25,
                full_candidate_n,
                float(route["first_vector_weight"]),
                int(route["first_rrf_k"]),
            )
            seen = set(completed)
            completed.extend(tool for tool in full_source if tool not in seen)
        if len(completed) < args.complete_ranking_to:
            raise ValueError(
                f"cannot complete frozen ranking to {args.complete_ranking_to}; "
                f"only {len(completed)} unique tools are available"
            )
        completed = completed[: max(len(canonical), args.complete_ranking_to)]
        final_rankings.append(completed)
        completion_counts.append(len(completed) - len(canonical))
    pred_rows = [
        {
            "id": row.get("id", ""),
            "query": row["query"],
            "gold_tools": row.get("gold_tools", []),
            "pred_tools": ranking,
            "source_tools": source,
            "reranked_tools": rerank,
            "tail_completion_count": completion_count,
        }
        for row, ranking, source, rerank, completion_count in zip(
            vector_rows,
            final_rankings,
            source_rankings,
            reranked,
            completion_counts,
        )
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "preds.jsonl", pred_rows)
    payload = {
        "route": route,
        "inputs": {
            str(path): sha256_file(path)
            for path in (
                args.vector_preds,
                args.bm25_preds,
                args.reranker_preds,
                args.route_json,
                args.dataset,
            )
        },
        "count": len(pred_rows),
        "ranking_completion": {
            "target_depth": args.complete_ranking_to,
            "policy": (
                "preserve_exact_frozen_route_prefix_then_append_full_source_rrf"
                if args.complete_ranking_to
                else "disabled"
            ),
            "rows_extended": sum(value > 0 for value in completion_counts),
            "tools_appended": sum(completion_counts),
            "max_tools_appended_per_row": max(completion_counts, default=0),
            "prefix_preserved": True,
        },
        "overall": {
            **metrics(pred_rows, final_rankings, [6, 10, 20, 30, 40, 50]),
            "min_output_size": min(map(len, final_rankings)),
            "max_output_size": max(map(len, final_rankings)),
            "mean_output_size": sum(map(len, final_rankings)) / len(final_rankings),
        },
        "canonical_uncompleted": metrics(
            pred_rows, canonical_rankings, [6, 10, 20, 30, 40, 50]
        ),
        "source": metrics(pred_rows, source_rankings, [6, 10, 20, 30, 40, 50]),
        "reranker": metrics(pred_rows, reranked, [6, 10, 20, 30, 40, 50]),
        "by_style": subgroup_metrics(pred_rows, final_rankings, metadata_rows, "style"),
        "by_tool_count": subgroup_metrics(pred_rows, final_rankings, metadata_rows, "tool_count"),
    }
    write_json(args.out_dir / "metrics.json", payload)
    print(f"frozen route AllHit@10={payload['overall']['allhit_at_10']:.6f} -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
