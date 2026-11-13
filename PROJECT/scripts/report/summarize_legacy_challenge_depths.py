#!/usr/bin/env python3
"""Summarize the fixed-route joint-depth grid on the isolated Legacy challenge."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


FIXED_RERANKER_P50_MS = {10: 23.9481, 20: 47.0914, 30: 70.9828, 50: 116.9624}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_mcnemar_pvalue(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = min(left_only, right_only)
    one_sided = sum(
        math.comb(discordant, index) for index in range(tail + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * one_sided)


def paired(left_path: Path, right_path: Path) -> dict[str, Any]:
    left = load_jsonl(left_path)
    right = load_jsonl(right_path)
    if len(left) != len(right):
        raise ValueError("paired rows differ")
    left_only = 0
    right_only = 0
    for left_row, right_row in zip(left, right):
        if left_row["id"] != right_row["id"]:
            raise ValueError("paired ids differ")
        left_correct = bool(left_row["trace_em"])
        right_correct = bool(right_row["trace_em"])
        left_only += left_correct and not right_correct
        right_only += right_correct and not left_correct
    return {
        "rows": len(left),
        "left_correct_right_wrong": left_only,
        "right_correct_left_wrong": right_only,
        "net_rows": left_only - right_only,
        "exact_mcnemar_pvalue": exact_mcnemar_pvalue(left_only, right_only),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-audit", type=Path, required=True)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--selected-route-sidecar", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    audit = load_json(args.dataset_audit)
    if not audit.get("strict_ok") or audit.get("selection_use_allowed") is not False:
        raise ValueError("challenge audit is not strict or improperly allows selection")
    routes = load_json(args.routes)
    cells = {}
    scored_paths = {}
    route_rows = list(routes["routes"])
    if args.selected_route_sidecar is not None:
        selected_sidecar = load_json(args.selected_route_sidecar)
        route_rows.append(
            {
                "id": "selected_r20p10",
                "reranker_depth": 20,
                "planner_depth": 10,
                "second_stage": selected_sidecar["second_stage"],
                "overall": selected_sidecar["retrieval"]["overall"],
            }
        )
    for route in route_rows:
        identifier = str(route["id"])
        metric_path = args.metrics_dir / f"{identifier}.metrics.json"
        scored_path = args.metrics_dir / f"{identifier}.scored.jsonl"
        metric = load_json(metric_path)
        if metric.get("status") != "complete" or int(metric.get("count", 0)) != 1200:
            raise ValueError(f"incomplete metric for {identifier}")
        reranker_depth = int(route["reranker_depth"])
        planner_depth = int(route["planner_depth"])
        cells[identifier] = {
            "reranker_depth": reranker_depth,
            "planner_depth": planner_depth,
            "second_stage": route["second_stage"],
            "retriever_allhit_at_planner_depth": float(
                route["overall"][f"allhit_at_{planner_depth}"]
            ),
            "set_em": float(metric["overall"]["set_em"]),
            "trace_em": float(metric["overall"]["trace_em"]),
            "strict": float(metric["overall"]["strict_output_valid_rate"]),
            "mean_input_tokens": float(metric["efficiency"]["input_tokens"]["mean"]),
            "p50_generator_latency_ms": float(
                metric["efficiency"]["latency_ms_generator_only"]["p50"]
            ),
            "fixed_selection_benchmark_reranker_p50_ms": FIXED_RERANKER_P50_MS[
                reranker_depth
            ],
            "metrics": str(metric_path),
            "metrics_sha256": sha256_file(metric_path),
            "scored": str(scored_path),
            "scored_sha256": sha256_file(scored_path),
        }
        scored_paths[identifier] = scored_path

    ranked = sorted(
        cells,
        key=lambda identifier: (
            -cells[identifier]["trace_em"],
            cells[identifier]["planner_depth"],
            cells[identifier]["reranker_depth"],
            cells[identifier]["mean_input_tokens"],
            identifier,
        ),
    )
    selected = "selected_r20p10" if "selected_r20p10" in cells else "rerank20_planner10"
    raw_best = ranked[0]
    comparisons = {}
    for other in (
        "rerank10_planner10",
        "rerank20_planner10",
        "rerank20_planner20",
        "rerank30_planner10",
        "rerank30_planner20",
        "rerank50_planner10",
        "rerank50_planner20",
        "rerank50_planner30",
        "rerank50_planner50",
    ):
        comparisons[f"{selected}_vs_{other}"] = paired(
            scored_paths[selected], scored_paths[other]
        )
    p10 = cells[selected]
    best = cells[raw_best]
    report = {
        "schema_version": 1,
        "status": "complete_challenge_only_no_tuning",
        "dataset_role": (
            "Legacy-generator distribution challenge; not a training, selection, "
            "promotion, Legacy frozen, or API frozen split"
        ),
        "dataset": str(args.dataset),
        "dataset_sha256": sha256_file(args.dataset),
        "dataset_audit": str(args.dataset_audit),
        "dataset_audit_sha256": sha256_file(args.dataset_audit),
        "routes": str(args.routes),
        "routes_sha256": sha256_file(args.routes),
        "selected_route_sidecar": (
            str(args.selected_route_sidecar)
            if args.selected_route_sidecar is not None
            else None
        ),
        "selected_route_sidecar_sha256": (
            sha256_file(args.selected_route_sidecar)
            if args.selected_route_sidecar is not None
            else None
        ),
        "fixed_route_policy": (
            "all 14 depth cells reuse second-stage parameters selected before this "
            "challenge was generated; no challenge label tunes RRF or depth"
        ),
        "cells": cells,
        "ranking_by_trace_then_cost": ranked,
        "raw_best_route_id": raw_best,
        "r20p10_route_id": selected,
        "r20p10_is_raw_best": raw_best == selected,
        "r20p10_gap_from_raw_best": best["trace_em"] - p10["trace_em"],
        "r20p10_within_reasonable_0p5pp_band": (
            p10["trace_em"] >= best["trace_em"] - 0.005
        ),
        "comparisons": comparisons,
        "interpretation_constraints": [
            "The challenge diagnoses external validity and interference; it cannot select or promote a route.",
            "Raw differences must be read together with paired discordances and the fixed token/latency cost.",
            "No protected Legacy/API result may be used to retune this grid.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"legacy challenge raw_best={raw_best} trace={best['trace_em']:.9f} "
        f"r20p10={p10['trace_em']:.9f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
