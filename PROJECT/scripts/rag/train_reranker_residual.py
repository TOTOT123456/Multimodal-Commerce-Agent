#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train a lightweight residual reranker over source/reranker features."""
from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


FEATURE_NAMES = [
    "source_rr",
    "reranker_rr",
    "log_source_rank",
    "log_reranker_rank",
    "score_z",
    "score_minmax",
    "score_gap_to_best",
    "source_top1",
    "source_top3",
    "source_top5",
    "source_top10",
    "reranker_top3",
    "reranker_top10",
    "rank_delta_scaled",
]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def assert_aligned(source_rows: list[dict], reranker_rows: list[dict]) -> None:
    if len(source_rows) != len(reranker_rows):
        raise SystemExit(f"row count mismatch: source={len(source_rows)} reranker={len(reranker_rows)}")
    for idx, (source, reranker) in enumerate(zip(source_rows, reranker_rows)):
        if source.get("id") and reranker.get("id") and source.get("id") != reranker.get("id"):
            raise SystemExit(f"id mismatch at row {idx}: {source.get('id')} vs {reranker.get('id')}")
        if source.get("query") != reranker.get("query"):
            raise SystemExit(f"query mismatch at row {idx}")


def allhit_at(rows: list[dict], rankings: list[list[str]], k: int) -> float:
    hits = 0
    for row, ranking in zip(rows, rankings):
        gold = set(row.get("gold_tools", []))
        hits += int(bool(gold) and gold <= set(ranking[:k]))
    return hits / max(1, len(rows))


def mrr(rows: list[dict], rankings: list[list[str]]) -> float:
    total = 0.0
    for row, ranking in zip(rows, rankings):
        gold = set(row.get("gold_tools", []))
        rr = 0.0
        for rank, tool_id in enumerate(ranking, start=1):
            if tool_id in gold:
                rr = 1.0 / rank
                break
        total += rr
    return total / max(1, len(rows))


def evaluate(rows: list[dict], rankings: list[list[str]], ks: list[int]) -> dict[str, float]:
    metrics = {f"allhit_at_{k}": allhit_at(rows, rankings, k) for k in ks}
    metrics["mrr"] = mrr(rows, rankings)
    return metrics


def candidate_scores(reranker_row: dict, candidate_k: int) -> dict[str, float]:
    candidates = reranker_row.get("candidate_tools", [])[:candidate_k]
    scores = reranker_row.get("scores", [])[:candidate_k]
    return {tool_id: float(score) for tool_id, score in zip(candidates, scores)}


def feature_vector(source_rank: int, reranker_rank: int, score: float, scores: list[float]) -> list[float]:
    mean = float(np.mean(scores)) if scores else 0.0
    std = float(np.std(scores)) if scores else 1.0
    if std < 1e-6:
        std = 1.0
    lo = min(scores) if scores else 0.0
    hi = max(scores) if scores else 1.0
    span = hi - lo if hi > lo else 1.0
    return [
        1.0 / source_rank,
        1.0 / reranker_rank,
        math.log1p(source_rank),
        math.log1p(reranker_rank),
        (score - mean) / std,
        (score - lo) / span,
        hi - score,
        1.0 if source_rank <= 1 else 0.0,
        1.0 if source_rank <= 3 else 0.0,
        1.0 if source_rank <= 5 else 0.0,
        1.0 if source_rank <= 10 else 0.0,
        1.0 if reranker_rank <= 3 else 0.0,
        1.0 if reranker_rank <= 10 else 0.0,
        (reranker_rank - source_rank) / 50.0,
    ]


def featurize(source_rows: list[dict], reranker_rows: list[dict], candidate_k: int) -> tuple[np.ndarray, np.ndarray, list[str], list[tuple[int, int]]]:
    features: list[list[float]] = []
    labels: list[int] = []
    tools: list[str] = []
    spans: list[tuple[int, int]] = []
    for source_row, reranker_row in zip(source_rows, reranker_rows):
        gold = set(source_row.get("gold_tools", []))
        source_pos = {tool_id: idx for idx, tool_id in enumerate(source_row["pred_tools"], start=1)}
        reranker_pos = {tool_id: idx for idx, tool_id in enumerate(reranker_row["pred_tools"], start=1)}
        score_map = candidate_scores(reranker_row, candidate_k)
        candidates = list(dict.fromkeys(reranker_row.get("candidate_tools", source_row["pred_tools"])[:candidate_k]))
        scores = [score_map.get(tool_id, min(score_map.values(), default=0.0)) for tool_id in candidates]
        start = len(features)
        for tool_id in candidates:
            source_rank = source_pos.get(tool_id, 9999)
            reranker_rank = reranker_pos.get(tool_id, 9999)
            score = score_map.get(tool_id, min(scores, default=0.0))
            features.append(feature_vector(source_rank, reranker_rank, score, scores))
            labels.append(1 if tool_id in gold else 0)
            tools.append(tool_id)
        spans.append((start, len(features)))
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int8), tools, spans


def build_model(model_type: str, seed: int) -> Any:
    if model_type == "logreg":
        return LogisticRegression(max_iter=300, n_jobs=4, C=1.0)
    if model_type == "logreg_balanced":
        return LogisticRegression(max_iter=300, n_jobs=4, C=1.0, class_weight="balanced")
    if model_type == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.05,
            l2_regularization=0.1,
            max_leaf_nodes=15,
            random_state=seed,
        )
    raise ValueError(f"unknown model type: {model_type}")


def predict_scores(model: Any, features: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    return model.decision_function(features)


def rank_predictions(source_rows: list[dict], tools: list[str], spans: list[tuple[int, int]], scores: np.ndarray) -> list[list[str]]:
    rankings = []
    for source_row, (lo, hi) in zip(source_rows, spans):
        source_pos = {tool_id: idx for idx, tool_id in enumerate(source_row["pred_tools"], start=1)}
        scored = [
            (tools[idx], float(scores[idx]), source_pos.get(tools[idx], 9999), idx - lo)
            for idx in range(lo, hi)
        ]
        scored.sort(key=lambda item: (-item[1], item[2], item[3], item[0]))
        ranking = [tool_id for tool_id, _, _, _ in scored]
        seen = set(ranking)
        ranking.extend(tool_id for tool_id in source_row["pred_tools"] if tool_id not in seen)
        rankings.append(ranking)
    return rankings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-source-preds", type=Path, required=True)
    parser.add_argument("--train-reranker-preds", type=Path, required=True)
    parser.add_argument("--eval-source-preds", type=Path, required=True)
    parser.add_argument("--eval-reranker-preds", type=Path, required=True)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--model-type", choices=["logreg", "logreg_balanced", "hgb"], default="hgb")
    parser.add_argument("--ks", nargs="+", type=int, default=[6, 10, 30, 50])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--write-eval-preds", action="store_true")
    args = parser.parse_args()

    train_source = load_jsonl(args.train_source_preds)
    train_reranker = load_jsonl(args.train_reranker_preds)
    eval_source = load_jsonl(args.eval_source_preds)
    eval_reranker = load_jsonl(args.eval_reranker_preds)
    assert_aligned(train_source, train_reranker)
    assert_aligned(eval_source, eval_reranker)

    train_x, train_y, train_tools, train_spans = featurize(train_source, train_reranker, args.candidate_k)
    eval_x, eval_y, eval_tools, eval_spans = featurize(eval_source, eval_reranker, args.candidate_k)

    model = build_model(args.model_type, args.seed)
    model.fit(train_x, train_y)
    train_scores = predict_scores(model, train_x)
    eval_scores = predict_scores(model, eval_x)

    train_rankings = rank_predictions(train_source, train_tools, train_spans, train_scores)
    eval_rankings = rank_predictions(eval_source, eval_tools, eval_spans, eval_scores)
    payload = {
        "model_type": args.model_type,
        "candidate_k": args.candidate_k,
        "feature_names": FEATURE_NAMES,
        "train_source_preds": str(args.train_source_preds),
        "train_reranker_preds": str(args.train_reranker_preds),
        "eval_source_preds": str(args.eval_source_preds),
        "eval_reranker_preds": str(args.eval_reranker_preds),
        "train_examples": int(train_x.shape[0]),
        "train_positive_rate": float(train_y.mean()),
        "eval_examples": int(eval_x.shape[0]),
        "eval_positive_rate": float(eval_y.mean()),
        "train_source_metrics": evaluate(train_source, [row["pred_tools"] for row in train_source], args.ks),
        "train_reranker_metrics": evaluate(train_source, [row["pred_tools"] for row in train_reranker], args.ks),
        "train_residual_metrics": evaluate(train_source, train_rankings, args.ks),
        "eval_source_metrics": evaluate(eval_source, [row["pred_tools"] for row in eval_source], args.ks),
        "eval_reranker_metrics": evaluate(eval_source, [row["pred_tools"] for row in eval_reranker], args.ks),
        "eval_residual_metrics": evaluate(eval_source, eval_rankings, args.ks),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "metrics.json", payload)
    with (args.out_dir / "model.pkl").open("wb") as f:
        pickle.dump({"model": model, "feature_names": FEATURE_NAMES, "candidate_k": args.candidate_k}, f)

    if args.write_eval_preds:
        pred_rows = []
        for row, ranking in zip(eval_source, eval_rankings):
            pred_rows.append(
                {
                    "id": row.get("id", ""),
                    "query": row["query"],
                    "gold_tools": row.get("gold_tools", []),
                    "pred_tools": ranking,
                }
            )
        write_jsonl(args.out_dir / "preds.jsonl", pred_rows)

    eval_a10 = payload["eval_residual_metrics"].get("allhit_at_10", 0.0)
    source_a10 = payload["eval_source_metrics"].get("allhit_at_10", 0.0)
    print(f"model={args.model_type} eval AllHit@10={eval_a10:.6f} delta_vs_source={eval_a10 - source_a10:+.6f}")
    print(f"wrote {args.out_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
