#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build fixed train/dev RAG rankings for assignment-3 generation.

Train mode fuses already-produced, aligned source/reranker rankings. Dev mode
runs the frozen embedding, BM25, task reranker, and both fixed RRF stages.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import (  # noqa: E402
    RAG_DIR,
    TOOL_CORPUS_PATH,
    TOOL_TRACE_DIR,
    load_jsonl,
    split_tools,
    write_json,
    write_jsonl,
)
from finetune.tool_trace_common import ASSIGNMENT3_DIR, percentile, sha256_file  # noqa: E402
from rag.run_retrieval_baselines import (  # noqa: E402
    DEFAULT_QUERY_INSTRUCTION,
    run_bm25,
    run_vector_db,
)


DEFAULT_TRAIN_SOURCE = (
    RAG_DIR
    / "_aux"
    / "embedding_finetune"
    / "bge-large-mnrl-e1-lr2e-5-bs64"
    / "fusion_tuning_rrf_train_fallback"
    / "preds_rrf_n50_w0p95_k5.jsonl"
)
DEFAULT_TRAIN_RERANKER = (
    RAG_DIR
    / "_aux"
    / "reranker_finetune"
    / "bge-reranker-base-e1-g8-bs16_eval_tuned_hybrid_train"
    / "preds.jsonl"
)
DEFAULT_TRAIN_UNTUNED = (
    RAG_DIR
    / "_aux"
    / "embedding_finetune"
    / "bge-large-mnrl-e1-lr2e-5-bs64"
    / "retrieval_train_fallback"
    / "preds"
    / "hybrid_bge-large-mnrl-e1-lr2e-5-bs64_rrf.jsonl"
)
DEFAULT_FINE_DB = RAG_DIR / "vector_dbs_finetuned" / "bge-large-mnrl-e1-lr2e-5-bs64"
DEFAULT_BASE_DB = RAG_DIR / "vector_dbs" / "bge-large"
DEFAULT_RERANKER = RAG_DIR / "_aux" / "reranker_finetune" / "bge-reranker-base-e1-g8-bs16"


def weighted_rrf(
    left: list[str],
    right: list[str],
    left_weight: float,
    rrf_k: int,
    left_n: int | None = None,
    right_n: int | None = None,
) -> list[str]:
    left_values = left[:left_n] if left_n else left
    right_values = right[:right_n] if right_n else right
    left_pos = {tool_id: rank for rank, tool_id in enumerate(left_values, start=1)}
    right_pos = {tool_id: rank for rank, tool_id in enumerate(right_values, start=1)}
    missing = 9999
    scored = []
    for tool_id in set(left_pos) | set(right_pos):
        score = (
            left_weight / (rrf_k + left_pos.get(tool_id, missing))
            + (1.0 - left_weight) / (rrf_k + right_pos.get(tool_id, missing))
        )
        scored.append((tool_id, score, left_pos.get(tool_id, missing), right_pos.get(tool_id, missing)))
    scored.sort(key=lambda item: (-item[1], item[2], item[3], item[0]))
    return [tool_id for tool_id, _, _, _ in scored]


def authority_rows(path: Path) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    if not rows:
        raise SystemExit(f"empty authoritative split: {path}")
    return rows


def validate_ranking(
    authority: list[dict[str, Any]],
    rankings: list[dict[str, Any]],
    name: str,
) -> None:
    if len(authority) != len(rankings):
        raise SystemExit(f"{name}: row count {len(rankings)} != authority {len(authority)}")
    authority_by_id = {row["id"]: row for row in authority}
    if len(authority_by_id) != len(authority):
        raise SystemExit("duplicate ids in authoritative split")
    ranking_by_id = {row.get("id"): row for row in rankings}
    if set(ranking_by_id) != set(authority_by_id):
        raise SystemExit(f"{name}: id set mismatch")
    for trace_id, row in authority_by_id.items():
        ranking = ranking_by_id[trace_id]
        if ranking.get("query") != row["问题"]:
            raise SystemExit(f"{name}: query mismatch for {trace_id}")
        if "gold_tools" in ranking and ranking["gold_tools"] != split_tools(row["工具"]):
            raise SystemExit(f"{name}: gold mismatch for {trace_id}")
        tools = ranking.get("pred_tools")
        if not isinstance(tools, list) or len(tools) != len(set(tools)):
            raise SystemExit(f"{name}: invalid or duplicate ranking for {trace_id}")


def retrieval_metrics(rows: list[dict[str, Any]], ks: list[int]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k in ks:
        hits = sum(set(row["gold_tools"]) <= set(row["pred_tools"][:k]) for row in rows)
        metrics[f"allhit_at_{k}"] = hits / max(1, len(rows))
    reciprocal = []
    for row in rows:
        gold = set(row["gold_tools"])
        reciprocal.append(
            next((1.0 / rank for rank, tool_id in enumerate(row["pred_tools"], 1) if tool_id in gold), 0.0)
        )
    metrics["mrr"] = sum(reciprocal) / max(1, len(reciprocal))
    return metrics


def build_train(args: argparse.Namespace) -> dict[str, Any]:
    authority = authority_rows(args.input)
    source = load_jsonl(args.train_source)
    reranker = load_jsonl(args.train_reranker)
    untuned = load_jsonl(args.train_untuned)
    validate_ranking(authority, source, "train source")
    validate_ranking(authority, reranker, "train reranker")
    validate_ranking(authority, untuned, "train untuned")
    source_by_id = {row["id"]: row for row in source}
    reranker_by_id = {row["id"]: row for row in reranker}
    untuned_by_id = {row["id"]: row for row in untuned}
    final_rows = []
    untuned_rows = []
    for auth in authority:
        trace_id = auth["id"]
        gold = split_tools(auth["工具"])
        final_rank = weighted_rrf(
            source_by_id[trace_id]["pred_tools"],
            reranker_by_id[trace_id]["pred_tools"],
            args.final_source_weight,
            args.final_rrf_k,
        )
        final_rows.append(
            {
                "id": trace_id,
                "query": auth["问题"],
                "gold_tools": gold,
                "pred_tools": final_rank,
            }
        )
        untuned_rows.append(
            {
                "id": trace_id,
                "query": auth["问题"],
                "gold_tools": gold,
                "pred_tools": untuned_by_id[trace_id]["pred_tools"],
            }
        )
    write_jsonl(args.output, final_rows)
    write_jsonl(args.untuned_output, untuned_rows)
    return {
        "mode": "train_existing_fusion",
        "count": len(final_rows),
        "metrics": retrieval_metrics(final_rows, args.ks),
        "untuned_metrics": retrieval_metrics(untuned_rows, args.ks),
        "timing": {},
        "input_hashes": {
            str(args.input): sha256_file(args.input),
            str(args.train_source): sha256_file(args.train_source),
            str(args.train_reranker): sha256_file(args.train_reranker),
            str(args.train_untuned): sha256_file(args.train_untuned),
        },
    }


def load_docs(path: Path) -> list[dict[str, Any]]:
    docs = load_jsonl(path)
    if not docs:
        raise SystemExit(f"empty tool corpus: {path}")
    return docs


def rerank(
    rows: list[dict[str, Any]],
    source_rankings: list[list[str]],
    docs: dict[str, str],
    model_path: Path,
    device: str,
    candidate_k: int,
    pair_batch_size: int,
    query_batch_size: int,
) -> tuple[list[list[str]], dict[str, float]]:
    from sentence_transformers import CrossEncoder

    load_start = time.perf_counter()
    model = CrossEncoder(
        str(model_path),
        device=device,
        trust_remote_code=True,
        max_length=512,
        model_kwargs={"torch_dtype": "auto"},
    )
    load_seconds = time.perf_counter() - load_start
    rankings: list[list[str]] = []
    latencies: list[float] = []
    for start in range(0, len(rows), query_batch_size):
        chunk_rows = rows[start : start + query_batch_size]
        chunk_ranks = source_rankings[start : start + query_batch_size]
        pairs: list[tuple[str, str]] = []
        spans: list[tuple[int, int, list[str]]] = []
        for row, ranking in zip(chunk_rows, chunk_ranks):
            candidates = [tool_id for tool_id in ranking[:candidate_k] if tool_id in docs]
            spans.append((len(pairs), len(pairs) + len(candidates), candidates))
            pairs.extend((row["问题"], docs[tool_id]) for tool_id in candidates)
        start_time = time.perf_counter()
        scores = model.predict(pairs, batch_size=pair_batch_size, show_progress_bar=False)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0 / max(1, len(chunk_rows))
        latencies.extend([elapsed_ms] * len(chunk_rows))
        score_values = scores.tolist() if hasattr(scores, "tolist") else list(scores)
        for lo, hi, candidates in spans:
            ranked = [
                tool_id
                for tool_id, _, position in sorted(
                    zip(candidates, score_values[lo:hi], range(len(candidates))),
                    key=lambda item: (-float(item[1]), item[2]),
                )
            ]
            rankings.append(ranked)
    return rankings, {
        "model_load_seconds": load_seconds,
        "p50_ms_per_query": statistics.median(latencies) if latencies else 0.0,
        "p95_ms_per_query": percentile(latencies, 95),
    }


def build_live(args: argparse.Namespace) -> dict[str, Any]:
    authority = authority_rows(args.input)
    docs = load_docs(args.docs)
    doc_text = {
        row["tool_id"]: row.get("document") or row.get("text") or row.get("enhanced_desc") or ""
        for row in docs
    }
    max_k = len(docs)
    production_started = time.perf_counter()
    bm25_rank, _, bm25_timing = run_bm25(docs, authority, max_k, args.tokenizer, args.text_field)
    fine_rank, _, fine_timing, fine_config = run_vector_db(
        args.fine_db,
        authority,
        max_k,
        args.vector_batch_size,
        args.device,
        args.query_instruction,
    )
    if args.skip_untuned:
        base_rank, base_timing, base_config = [], {}, {"status": "not_run"}
    else:
        base_rank, _, base_timing, base_config = run_vector_db(
            args.base_db,
            authority,
            max_k,
            args.vector_batch_size,
            args.device,
            args.query_instruction,
        )
    source_fusion_started = time.perf_counter()
    source = [
        weighted_rrf(
            vector,
            lexical,
            args.source_vector_weight,
            args.source_rrf_k,
            args.source_candidate_n,
            args.source_candidate_n,
        )
        for vector, lexical in zip(fine_rank, bm25_rank)
    ]
    source_fusion_ms_per_query = (
        (time.perf_counter() - source_fusion_started) * 1000.0 / len(authority)
    )
    untuned = (
        [
            weighted_rrf(
                vector,
                lexical,
                0.5,
                args.untuned_rrf_k,
                args.untuned_candidate_n,
                args.untuned_candidate_n,
            )
            for vector, lexical in zip(base_rank, bm25_rank)
        ]
        if not args.skip_untuned
        else [list(ranking) for ranking in source]
    )
    reranked, rerank_timing = rerank(
        authority,
        source,
        doc_text,
        args.reranker,
        args.device,
        args.reranker_candidate_k,
        args.pair_batch_size,
        args.query_batch_size,
    )
    final_fusion_started = time.perf_counter()
    final = [
        weighted_rrf(source_rank, reranker_rank, args.final_source_weight, args.final_rrf_k)
        for source_rank, reranker_rank in zip(source, reranked)
    ]
    final_fusion_ms_per_query = (
        (time.perf_counter() - final_fusion_started) * 1000.0 / len(authority)
    )
    production_elapsed_seconds = time.perf_counter() - production_started
    final_rows = []
    untuned_rows = []
    for auth, final_rank, untuned_rank in zip(authority, final, untuned):
        gold = split_tools(auth["工具"])
        final_rows.append(
            {"id": auth["id"], "query": auth["问题"], "gold_tools": gold, "pred_tools": final_rank}
        )
        untuned_rows.append(
            {"id": auth["id"], "query": auth["问题"], "gold_tools": gold, "pred_tools": untuned_rank}
        )
    write_jsonl(args.output, final_rows)
    write_jsonl(args.untuned_output, untuned_rows)
    return {
        "mode": "live_fixed_retrieval",
        "count": len(final_rows),
        "metrics": retrieval_metrics(final_rows, args.ks),
        "untuned_metrics": retrieval_metrics(untuned_rows, args.ks),
        "timing": {
            "bm25": bm25_timing,
            "fine_embedding": fine_timing,
            "base_embedding": base_timing,
            "task_reranker": rerank_timing,
            "source_fusion_ms_per_query": source_fusion_ms_per_query,
            "final_fusion_ms_per_query": final_fusion_ms_per_query,
            "production_elapsed_seconds_including_model_load": production_elapsed_seconds,
            "production_steady_state_avg_ms_per_query": (
                max(
                    0.0,
                    production_elapsed_seconds
                    - rerank_timing["model_load_seconds"]
                    - fine_timing.get("model_and_index_load_seconds", 0.0)
                    - base_timing.get("model_and_index_load_seconds", 0.0),
                )
                * 1000.0
                / len(authority)
            ),
        },
        "models": {
            "fine_embedding": fine_config,
            "base_embedding": base_config,
            "task_reranker": str(args.reranker),
        },
        "input_hashes": {
            str(args.input): sha256_file(args.input),
            str(args.docs): sha256_file(args.docs),
            str(args.fine_db / "config.json"): sha256_file(args.fine_db / "config.json"),
            str(args.base_db / "config.json"): sha256_file(args.base_db / "config.json"),
            str(args.reranker / "model.safetensors"): sha256_file(args.reranker / "model.safetensors"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "live"], required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--untuned-output", type=Path)
    parser.add_argument("--manifest-out", type=Path)
    parser.add_argument("--docs", type=Path, default=TOOL_CORPUS_PATH)
    parser.add_argument("--train-source", type=Path, default=DEFAULT_TRAIN_SOURCE)
    parser.add_argument("--train-reranker", type=Path, default=DEFAULT_TRAIN_RERANKER)
    parser.add_argument("--train-untuned", type=Path, default=DEFAULT_TRAIN_UNTUNED)
    parser.add_argument("--fine-db", type=Path, default=DEFAULT_FINE_DB)
    parser.add_argument("--base-db", type=Path, default=DEFAULT_BASE_DB)
    parser.add_argument("--reranker", type=Path, default=DEFAULT_RERANKER)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--vector-batch-size", type=int, default=128)
    parser.add_argument("--pair-batch-size", type=int, default=256)
    parser.add_argument("--query-batch-size", type=int, default=64)
    parser.add_argument("--tokenizer", choices=["auto", "jieba", "fallback"], default="auto")
    parser.add_argument("--text-field", default="document")
    parser.add_argument("--query-instruction", default=DEFAULT_QUERY_INSTRUCTION)
    parser.add_argument(
        "--skip-untuned",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip the untuned base-embedding diagnostic for production latency runs.",
    )
    parser.add_argument("--source-vector-weight", type=float, default=0.95)
    parser.add_argument("--source-rrf-k", type=int, default=5)
    parser.add_argument("--source-candidate-n", type=int, default=50)
    parser.add_argument("--untuned-rrf-k", type=int, default=60)
    parser.add_argument("--untuned-candidate-n", type=int, default=30)
    parser.add_argument("--reranker-candidate-k", type=int, default=50)
    parser.add_argument("--final-source-weight", type=float, default=0.6)
    parser.add_argument("--final-rrf-k", type=int, default=10)
    parser.add_argument("--ks", nargs="+", type=int, default=[10, 20, 30])
    args = parser.parse_args()

    if args.mode == "train":
        args.input = args.input or (TOOL_TRACE_DIR / "train.jsonl")
        args.output = args.output or (ASSIGNMENT3_DIR / "rag_train_rankings.jsonl")
        args.untuned_output = args.untuned_output or (ASSIGNMENT3_DIR / "rag_train_untuned_rankings.jsonl")
        args.manifest_out = args.manifest_out or (ASSIGNMENT3_DIR / "rag_train_rankings_manifest.json")
    else:
        api_dir = ASSIGNMENT3_DIR / "api_native_dev_2400"
        args.input = args.input or (api_dir / "dev.jsonl")
        args.output = args.output or (api_dir / "rag_rankings.jsonl")
        args.untuned_output = args.untuned_output or (api_dir / "rag_untuned_rankings.jsonl")
        args.manifest_out = args.manifest_out or (api_dir / "retrieval_manifest.json")

    started = time.perf_counter()
    payload = build_train(args) if args.mode == "train" else build_live(args)
    payload.update(
        {
            "status": "complete",
            "parameters": {
                "source_vector_weight": args.source_vector_weight,
                "source_rrf_k": args.source_rrf_k,
                "source_candidate_n": args.source_candidate_n,
                "untuned_rrf_k": args.untuned_rrf_k,
                "untuned_candidate_n": args.untuned_candidate_n,
                "reranker_candidate_k": args.reranker_candidate_k,
                "final_source_weight": args.final_source_weight,
                "final_rrf_k": args.final_rrf_k,
                "skip_untuned": args.skip_untuned,
            },
            "output": str(args.output),
            "output_sha256": sha256_file(args.output),
            "untuned_output": str(args.untuned_output),
            "untuned_output_sha256": sha256_file(args.untuned_output),
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    write_json(args.manifest_out, payload)
    print(
        f"RAG {args.mode} rows={payload['count']} "
        f"AllHit@10={payload['metrics'].get('allhit_at_10', 0.0):.6f} -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
