#!/usr/bin/env python3
"""Apply a frozen BM25/vector weighted-RRF route without selecting parameters."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from search_retrieval_pipeline import (  # noqa: E402
    assert_aligned,
    load_json,
    load_jsonl,
    metrics,
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


def resolve_route(path: Path, field: str) -> dict:
    payload = load_json(path)
    route = payload
    for key in field.split("."):
        if not isinstance(route, dict) or key not in route:
            raise ValueError(f"route field {field!r} missing from {path}")
        route = route[key]
    if not isinstance(route, dict):
        raise ValueError(f"route field {field!r} is not an object")
    compact = {"candidate_n", "vector_weight", "rrf_k"}
    canonical = {"candidate_n_per_source", "vector_weight", "rrf_k"}
    frozen = {"first_candidate_n", "first_vector_weight", "first_rrf_k"}
    if compact <= set(route):
        return {key: route[key] for key in compact}
    if canonical <= set(route):
        return {
            "candidate_n": route["candidate_n_per_source"],
            "vector_weight": route["vector_weight"],
            "rrf_k": route["rrf_k"],
        }
    if frozen <= set(route):
        return {
            "candidate_n": route["first_candidate_n"],
            "vector_weight": route["first_vector_weight"],
            "rrf_k": route["first_rrf_k"],
        }
    raise ValueError(
        f"route lacks compact fields {sorted(compact)} or frozen fields {sorted(frozen)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vector-preds", type=Path, required=True)
    parser.add_argument("--bm25-preds", type=Path, required=True)
    parser.add_argument("--route-json", type=Path, required=True)
    parser.add_argument("--route-field", default="mining_source_selected")
    parser.add_argument(
        "--complete-to",
        type=int,
        default=0,
        help="Preserve the frozen source prefix and append full-source RRF tools to this depth.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.complete_to < 0:
        raise ValueError("--complete-to must be non-negative")

    vector_rows = load_jsonl(args.vector_preds)
    bm25_rows = load_jsonl(args.bm25_preds)
    assert_aligned(vector_rows, bm25_rows, "vector", "bm25")
    route = resolve_route(args.route_json, args.route_field)
    candidate_n = int(route["candidate_n"])
    if any(
        len(row.get("pred_tools", [])) < candidate_n
        for row in vector_rows + bm25_rows
    ):
        raise ValueError(f"rankings shorter than frozen candidate_n={candidate_n}")

    rankings = [
        weighted_rrf(
            vector,
            bm25,
            candidate_n,
            float(route["vector_weight"]),
            int(route["rrf_k"]),
        )
        for vector, bm25 in zip(vector_rows, bm25_rows)
    ]
    completion_counts = [0] * len(rankings)
    if args.complete_to:
        completed_rankings = []
        for index, (ranking, vector, bm25) in enumerate(
            zip(rankings, vector_rows, bm25_rows)
        ):
            completed = list(ranking)
            if len(completed) < args.complete_to:
                full_n = min(len(vector["pred_tools"]), len(bm25["pred_tools"]))
                full_source = weighted_rrf(
                    vector,
                    bm25,
                    full_n,
                    float(route["vector_weight"]),
                    int(route["rrf_k"]),
                )
                seen = set(completed)
                completed.extend(tool for tool in full_source if tool not in seen)
            if len(completed) < args.complete_to:
                raise ValueError(
                    f"cannot complete first-stage row {index} to {args.complete_to}"
                )
            completed = completed[: max(len(ranking), args.complete_to)]
            completion_counts[index] = len(completed) - len(ranking)
            completed_rankings.append(completed)
        rankings = completed_rankings
    rows = [
        {
            "id": row.get("id", ""),
            "query": row["query"],
            "gold_tools": row.get("gold_tools", []),
            "pred_tools": ranking,
        }
        for row, ranking in zip(vector_rows, rankings)
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "preds.jsonl", rows)
    payload = {
        "route": route,
        "route_field": args.route_field,
        "count": len(rows),
        "mean_output_size": sum(map(len, rankings)) / max(1, len(rankings)),
        "completion": {
            "target_depth": args.complete_to,
            "policy": (
                "preserve_frozen_source_prefix_then_append_full_source_rrf"
                if args.complete_to
                else "disabled"
            ),
            "rows_extended": sum(value > 0 for value in completion_counts),
            "tools_appended": sum(completion_counts),
        },
        "metrics": metrics(rows, rankings, [6, 10, 20, 30, 40, 50]),
        "inputs": {
            str(path): sha256_file(path)
            for path in (args.vector_preds, args.bm25_preds, args.route_json)
        },
    }
    write_json(args.out_dir / "metrics.json", payload)
    print(f"applied first-stage route rows={len(rows)} -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
