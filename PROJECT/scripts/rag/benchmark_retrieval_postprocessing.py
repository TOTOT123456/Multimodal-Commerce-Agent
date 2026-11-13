#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark CPU-only retrieval post-processing on the fixed 6K split."""
from __future__ import annotations

import argparse
import json
import pickle
import platform
import statistics
import time
from pathlib import Path

from train_reranker_residual import (
    assert_aligned,
    featurize,
    load_jsonl,
    predict_scores,
    rank_predictions,
)
from tune_reranker_rank_fusion import fuse_rows
from tune_weighted_fusion import fuse_weighted_rrf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = (
    PROJECT_ROOT
    / "data/rag/_aux/embedding_finetune/bge-large-mnrl-e1-lr2e-5-bs64"
)


def cpu_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "not_reported"


def timing_summary(seconds: list[float], rows: int) -> dict:
    median_seconds = statistics.median(seconds)
    return {
        "repetitions": len(seconds),
        "seconds": {
            "min": min(seconds),
            "median": median_seconds,
            "max": max(seconds),
        },
        "mean_ms_per_query_from_median_run": median_seconds * 1000.0 / rows,
        "throughput_queries_per_second_from_median_run": rows / median_seconds,
    }


def topk_equal_rate(left: list[list[str]], right_rows: list[dict], k: int) -> float:
    equal = sum(
        int(ranking[:k] == row["pred_tools"][:k])
        for ranking, row in zip(left, right_rows)
    )
    return equal / max(1, len(left))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--vector-preds",
        type=Path,
        default=RUN_ROOT / "retrieval_fallback/preds/vector_bge-large-mnrl-e1-lr2e-5-bs64.jsonl",
    )
    parser.add_argument(
        "--bm25-preds",
        type=Path,
        default=RUN_ROOT / "retrieval_fallback/preds/bm25.jsonl",
    )
    parser.add_argument(
        "--source-preds",
        type=Path,
        default=RUN_ROOT / "fusion_tuning_rrf_fallback/preds_rrf_n50_w0p95_k5.jsonl",
    )
    parser.add_argument(
        "--reranker-preds",
        type=Path,
        default=RUN_ROOT / "reranker_tuned_hybrid_fallback/preds.jsonl",
    )
    parser.add_argument(
        "--final-reranker-preds",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data/rag/_aux/reranker_finetune/"
            "bge-reranker-base-e1-g8-bs16_eval_tuned_hybrid/preds.jsonl"
        ),
    )
    parser.add_argument(
        "--final-preds",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data/rag/_aux/reranker_finetune/"
            "bge-reranker-base-e1-g8-bs16_rank_fusion_tuned_hybrid/"
            "preds_rank_fusion_w0p6_k10.jsonl"
        ),
    )
    parser.add_argument(
        "--residual-model",
        type=Path,
        default=RUN_ROOT / "reranker_residual_logreg_balanced_tuned_hybrid/model.pkl",
    )
    parser.add_argument(
        "--residual-preds",
        type=Path,
        default=RUN_ROOT / "reranker_residual_logreg_balanced_tuned_hybrid/preds.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "data/rag/_aux/retrieval_postprocessing_efficiency.json",
    )
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")

    vector_rows = load_jsonl(args.vector_preds)
    bm25_rows = load_jsonl(args.bm25_preds)
    source_rows = load_jsonl(args.source_preds)
    reranker_rows = load_jsonl(args.reranker_preds)
    final_reranker_rows = load_jsonl(args.final_reranker_preds)
    final_rows = load_jsonl(args.final_preds)
    residual_rows = load_jsonl(args.residual_preds)
    assert_aligned(vector_rows, bm25_rows)
    assert_aligned(source_rows, reranker_rows)
    assert_aligned(source_rows, final_reranker_rows)
    assert_aligned(source_rows, final_rows)
    assert_aligned(source_rows, residual_rows)
    rows = len(source_rows)
    if not rows or any(
        len(group) != rows
        for group in [vector_rows, bm25_rows, final_reranker_rows, final_rows, residual_rows]
    ):
        raise SystemExit("all benchmark inputs must contain the same non-zero row count")

    first_stage_runs: list[float] = []
    first_stage_rankings: list[list[str]] = []
    final_fusion_runs: list[float] = []
    final_rankings: list[list[str]] = []
    for repetition in range(args.repetitions + 1):
        started = time.perf_counter()
        first_stage_rankings = [
            fuse_weighted_rrf(vector, bm25, 50, 0.95, 5)
            for vector, bm25 in zip(vector_rows, bm25_rows)
        ]
        elapsed = time.perf_counter() - started
        if repetition > 0:
            first_stage_runs.append(elapsed)

        started = time.perf_counter()
        final_rankings = fuse_rows(source_rows, final_reranker_rows, 0.6, 10)
        elapsed = time.perf_counter() - started
        if repetition > 0:
            final_fusion_runs.append(elapsed)

    with args.residual_model.open("rb") as f:
        residual_bundle = pickle.load(f)
    model = residual_bundle["model"]
    candidate_k = int(residual_bundle["candidate_k"])

    residual_phase_runs = {"feature_build": [], "model_score": [], "ranking": [], "total": []}
    residual_rankings: list[list[str]] = []
    for repetition in range(args.repetitions + 1):
        total_started = time.perf_counter()
        started = time.perf_counter()
        features, _, tools, spans = featurize(source_rows, reranker_rows, candidate_k)
        feature_seconds = time.perf_counter() - started

        started = time.perf_counter()
        scores = predict_scores(model, features)
        score_seconds = time.perf_counter() - started

        started = time.perf_counter()
        residual_rankings = rank_predictions(source_rows, tools, spans, scores)
        ranking_seconds = time.perf_counter() - started
        total_seconds = time.perf_counter() - total_started

        if repetition > 0:
            residual_phase_runs["feature_build"].append(feature_seconds)
            residual_phase_runs["model_score"].append(score_seconds)
            residual_phase_runs["ranking"].append(ranking_seconds)
            residual_phase_runs["total"].append(total_seconds)

    payload = {
        "status": "complete",
        "protocol": "single-process CPU, one warmup run, full fixed 6K batch",
        "rows": rows,
        "candidate_k": candidate_k,
        "cpu": cpu_name(),
        "first_stage_rrf": {
            **timing_summary(first_stage_runs, rows),
            "parameters": {"vector_weight": 0.95, "rrf_k": 5, "candidate_n": 50},
            "top10_equal_rate": topk_equal_rate(first_stage_rankings, source_rows, 10),
        },
        "final_rrf": {
            **timing_summary(final_fusion_runs, rows),
            "parameters": {"source_weight": 0.6, "rrf_k": 10, "candidate_n": 50},
            "top10_equal_rate": topk_equal_rate(final_rankings, final_rows, 10),
        },
        "residual_logreg": {
            "model": str(args.residual_model),
            "feature_count": int(features.shape[1]),
            "candidate_rows": int(features.shape[0]),
            "feature_build": timing_summary(residual_phase_runs["feature_build"], rows),
            "model_score": timing_summary(residual_phase_runs["model_score"], rows),
            "ranking": timing_summary(residual_phase_runs["ranking"], rows),
            "total": timing_summary(residual_phase_runs["total"], rows),
            "top10_equal_rate": topk_equal_rate(residual_rankings, residual_rows, 10),
        },
        "inputs": {
            "vector_preds": str(args.vector_preds),
            "bm25_preds": str(args.bm25_preds),
            "source_preds": str(args.source_preds),
            "reranker_preds": str(args.reranker_preds),
            "final_reranker_preds": str(args.final_reranker_preds),
            "final_preds": str(args.final_preds),
            "residual_preds": str(args.residual_preds),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"retrieval post-processing benchmark -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
