#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run a reranker over an existing retrieval candidate file."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import RAG_DIR, TOOL_CORPUS_PATH, load_jsonl, write_json
from rag.run_retrieval_baselines import evaluate_rankings


DEFAULT_PREDS = RAG_DIR / "_aux" / "rag_baseline" / "preds" / "hybrid_qwen3-embedding-0.6b_rrf.jsonl"
DEFAULT_OUTPUT_DIR = RAG_DIR / "_aux" / "reranker_baseline"


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[int(pct) - 1]


def load_docs(path: Path) -> dict[str, str]:
    docs = {}
    for row in load_jsonl(path):
        docs[row["tool_id"]] = row.get("document") or row.get("text") or row.get("enhanced_desc") or ""
    if not docs:
        raise ValueError(f"empty tool docs: {path}")
    return docs


def load_reranker(args: argparse.Namespace):
    if args.backend == "sentence-transformers":
        from sentence_transformers import CrossEncoder
        import torch

        model_kwargs: dict[str, Any] = (
            {"torch_dtype": torch.float16} if args.use_fp16 else {}
        )
        return CrossEncoder(
            args.model,
            device=args.device,
            trust_remote_code=args.trust_remote_code,
            max_length=args.max_length,
            model_kwargs=model_kwargs,
        )

    from FlagEmbedding import FlagAutoReranker

    kwargs: dict[str, Any] = {
        "use_fp16": args.use_fp16,
        "trust_remote_code": args.trust_remote_code,
        "devices": [args.device] if args.device else None,
        "batch_size": args.pair_batch_size,
    }
    if args.model_class:
        kwargs["model_class"] = args.model_class
    return FlagAutoReranker.from_finetuned(args.model, **kwargs)


def flatten_scores(scores: Any) -> list[float]:
    if hasattr(scores, "tolist"):
        scores = scores.tolist()
    if isinstance(scores, (float, int)):
        return [float(scores)]
    return [float(x) for x in scores]


def compute_scores(reranker: Any, pairs: list[tuple[str, str]], args: argparse.Namespace) -> list[float]:
    if not pairs:
        return []
    if args.backend == "sentence-transformers":
        return flatten_scores(
            reranker.predict(
                pairs,
                batch_size=args.pair_batch_size,
                show_progress_bar=False,
            )
        )
    return flatten_scores(
        reranker.compute_score(pairs, batch_size=args.pair_batch_size, max_length=args.max_length)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", type=Path, default=DEFAULT_PREDS)
    parser.add_argument("--docs", type=Path, default=TOOL_CORPUS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default="Qwen/Qwen3-Reranker-0.6B")
    parser.add_argument("--backend", choices=["flag", "sentence-transformers"], default="flag")
    parser.add_argument("--model-class", default="decoder-only-base")
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--ks", nargs="+", type=int, default=[6, 8, 10])
    parser.add_argument("--query-batch-size", type=int, default=24)
    parser.add_argument("--pair-batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--use-fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", dest="use_fp16", action="store_false")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()

    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("invalid shard count/index")

    docs = load_docs(args.docs)
    rows = load_jsonl(args.preds)
    if args.limit:
        rows = rows[: args.limit]
    rows = [
        row
        for index, row in enumerate(rows)
        if index % args.shard_count == args.shard_index
    ]
    if not rows:
        raise SystemExit(f"empty predictions: {args.preds}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred_out = args.output_dir / "preds.jsonl"
    metrics_out = args.output_dir / "metrics.json"

    load_t0 = time.perf_counter()
    reranker = load_reranker(args)
    load_seconds = time.perf_counter() - load_t0

    out_rows = []
    latencies = []
    t0 = time.perf_counter()
    for start in range(0, len(rows), args.query_batch_size):
        batch = rows[start : start + args.query_batch_size]
        pairs: list[tuple[str, str]] = []
        spans: list[tuple[int, int, list[str]]] = []
        for row in batch:
            candidates = [tool for tool in row.get("pred_tools", [])[: args.candidate_k] if tool in docs]
            spans.append((len(pairs), len(pairs) + len(candidates), candidates))
            pairs.extend((row["query"], docs[tool]) for tool in candidates)
        chunk_t0 = time.perf_counter()
        scores = compute_scores(reranker, pairs, args)
        chunk_ms_per_query = (time.perf_counter() - chunk_t0) * 1000.0 / max(1, len(batch))
        latencies.extend([chunk_ms_per_query] * len(batch))
        for row, (lo, hi, candidates) in zip(batch, spans):
            candidate_scores = scores[lo:hi]
            ranked = [
                tool
                for tool, _, _ in sorted(
                    zip(candidates, candidate_scores, range(len(candidates))),
                    key=lambda item: (-item[1], item[2]),
                )
            ]
            out_rows.append(
                {
                    "id": row.get("id", ""),
                    "query": row["query"],
                    "gold_tools": row.get("gold_tools", []),
                    "pred_tools": ranked,
                    "candidate_tools": candidates,
                    "candidate_count": len(candidates),
                    "scores": candidate_scores,
                }
            )

    examples = [{"工具": ",".join(row.get("gold_tools", []))} for row in out_rows]
    reranked = [row["pred_tools"] for row in out_rows]
    candidates = [row["candidate_tools"] for row in out_rows]
    metrics = {
        "model": args.model,
        "backend": args.backend,
        "model_class": args.model_class,
        "source_predictions": str(args.preds),
        "tool_docs": str(args.docs),
        "predictions": str(pred_out),
        "num_queries": len(out_rows),
        "candidate_k": args.candidate_k,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "k_values": args.ks,
        "load_seconds": round(load_seconds, 4),
        "elapsed_seconds": round(time.perf_counter() - t0, 4),
        "timing": {
            "rerank_p50_ms_per_query": round(statistics.median(latencies), 4) if latencies else 0.0,
            "rerank_p95_ms_per_query": round(percentile(latencies, 95), 4),
        },
        "candidate_metrics": evaluate_rankings(examples, candidates, [args.candidate_k]),
        "rerank_metrics": evaluate_rankings(examples, reranked, args.ks),
    }

    with pred_out.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(metrics_out, metrics)
    print(f"reranker queries={len(out_rows)} candidate_k={args.candidate_k} -> {metrics_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
