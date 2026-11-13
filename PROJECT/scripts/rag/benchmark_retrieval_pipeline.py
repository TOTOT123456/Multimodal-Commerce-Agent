#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark the complete fixed retrieval path with warmed, batched execution."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import RAG_DIR, TOOL_CORPUS_PATH, load_jsonl  # noqa: E402
from rag.search_retrieval_pipeline import second_rrf, weighted_rrf  # noqa: E402
from rag.run_retrieval_baselines import (  # noqa: E402
    BM25,
    DEFAULT_QUERY_INSTRUCTION,
    build_tokenizer,
    doc_text,
    evaluate_rankings,
    load_sentence_transformer,
    rank_from_scores,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data/tool_call_trace/_aux/assignment3/assignment3_crossstyle_rag10/"
    "dpo_v2/eval_protocol/final.jsonl"
)
DEFAULT_REFERENCE = DEFAULT_INPUT.parent / "final_rag_rankings.jsonl"
DEFAULT_FINE_DB = RAG_DIR / "vector_dbs_finetuned/bge-large-mnrl-e1-lr2e-5-bs64"
DEFAULT_RERANKER = (
    RAG_DIR / "_aux/reranker_finetune/bge-reranker-base-e1-g8-bs16"
)
DEFAULT_ROUTE = (
    RAG_DIR / "_aux/retrieval_pipeline_reselection/bs64_control_route.snapshot.json"
)
DEFAULT_OUT = RAG_DIR / "_aux/retrieval_pipeline_efficiency_20260731.json"


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def latency_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values) if values else 0.0,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values, default=0.0),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cpu_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "not_reported"


def load_route(path: Path) -> dict[str, int | float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
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
        raise SystemExit(f"route JSON lacks required fields: {sorted(required)}")
    return {key: route[key] for key in required}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--docs", type=Path, default=TOOL_CORPUS_PATH)
    parser.add_argument("--fine-db", type=Path, default=DEFAULT_FINE_DB)
    parser.add_argument("--reranker", type=Path, default=DEFAULT_RERANKER)
    parser.add_argument("--route-json", type=Path, default=DEFAULT_ROUTE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--query-batch-size", type=int, default=64)
    parser.add_argument("--pair-batch-size", type=int, default=256)
    parser.add_argument("--tokenizer", choices=["auto", "jieba", "fallback"], default="jieba")
    parser.add_argument("--text-field", default="document")
    parser.add_argument("--query-instruction", default=DEFAULT_QUERY_INSTRUCTION)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.query_batch_size < 1 or args.pair_batch_size < 1:
        parser.error("batch sizes must be positive")

    try:
        import faiss
        import torch
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise SystemExit("run this benchmark in the llm_bge environment") from exc

    authority = load_jsonl(args.input)
    reference = load_jsonl(args.reference)
    docs = load_jsonl(args.docs)
    route = load_route(args.route_json)
    if not authority or len(authority) != len(reference):
        raise SystemExit("input and reference must be aligned and non-empty")
    for index, (row, expected) in enumerate(zip(authority, reference)):
        if row.get("id") != expected.get("id") or row.get("问题") != expected.get("query"):
            raise SystemExit(f"reference mismatch at row {index}")

    tokenize = build_tokenizer(args.tokenizer)
    bm25 = BM25([tokenize(doc_text(doc, args.text_field)) for doc in docs])
    doc_by_id = {
        row["tool_id"]: row.get("document") or row.get("text") or row.get("enhanced_desc") or ""
        for row in docs
    }

    vector_config = json.loads((args.fine_db / "config.json").read_text(encoding="utf-8"))
    vector_model_dir = Path(str(vector_config["model_name_or_path"]))
    vector_model_weights = vector_model_dir / "model.safetensors"
    load_started = time.perf_counter()
    vector_model = load_sentence_transformer(vector_config["model_name_or_path"], args.device)
    vector_index = faiss.read_index(str(args.fine_db / "index.faiss"))
    vector_load_seconds = time.perf_counter() - load_started

    load_started = time.perf_counter()
    reranker = CrossEncoder(
        str(args.reranker),
        device=args.device,
        trust_remote_code=True,
        max_length=512,
        model_kwargs={"torch_dtype": torch.float16},
    )
    reranker_load_seconds = time.perf_counter() - load_started

    stage_values: dict[str, list[float]] = {
        "bm25": [],
        "vector_and_faiss": [],
        "source_rrf": [],
        "task_reranker": [],
        "final_rrf": [],
        "complete_path": [],
    }

    def synchronize() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def run_batch(batch: list[dict], record: bool) -> list[list[str]]:
        batch_started = time.perf_counter()

        started = time.perf_counter()
        lexical_rankings = []
        for row in batch:
            scores = bm25.scores(tokenize(row["问题"]))
            ranking, _ = rank_from_scores(scores, docs, len(docs))
            lexical_rankings.append(ranking)
        bm25_seconds = time.perf_counter() - started

        queries = [
            f"{args.query_instruction}{row['问题']}" if args.query_instruction else row["问题"]
            for row in batch
        ]
        synchronize()
        started = time.perf_counter()
        embeddings = vector_model.encode(
            queries,
            batch_size=args.query_batch_size,
            normalize_embeddings=bool(vector_config.get("normalize_embeddings", True)),
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vector_rankings = []
        for embedding in embeddings:
            _, indices = vector_index.search(embedding.reshape(1, -1), len(docs))
            vector_rankings.append(
                [docs[int(index)]["tool_id"] for index in indices[0] if int(index) >= 0]
            )
        synchronize()
        vector_seconds = time.perf_counter() - started

        started = time.perf_counter()
        source_rankings = [
            weighted_rrf(
                {"pred_tools": vector},
                {"pred_tools": lexical},
                int(route["first_candidate_n"]),
                float(route["first_vector_weight"]),
                int(route["first_rrf_k"]),
            )
            for vector, lexical in zip(vector_rankings, lexical_rankings)
        ]
        source_seconds = time.perf_counter() - started

        pairs: list[tuple[str, str]] = []
        spans: list[tuple[int, int, list[str]]] = []
        for row, source in zip(batch, source_rankings):
            candidates = [
                tool_id
                for tool_id in source[: int(route["candidate_k"])]
                if tool_id in doc_by_id
            ]
            spans.append((len(pairs), len(pairs) + len(candidates), candidates))
            pairs.extend((row["问题"], doc_by_id[tool_id]) for tool_id in candidates)
        synchronize()
        started = time.perf_counter()
        scores = reranker.predict(
            pairs,
            batch_size=args.pair_batch_size,
            show_progress_bar=False,
        )
        synchronize()
        reranker_seconds = time.perf_counter() - started
        score_values = scores.tolist() if hasattr(scores, "tolist") else list(scores)
        reranked = []
        for lo, hi, candidates in spans:
            reranked.append(
                [
                    tool_id
                    for tool_id, _, position in sorted(
                        zip(candidates, score_values[lo:hi], range(len(candidates))),
                        key=lambda item: (-float(item[1]), item[2]),
                    )
                ]
            )

        started = time.perf_counter()
        final_rankings = [
            second_rrf(
                source,
                reranker_rank,
                float(route["second_source_weight"]),
                int(route["second_rrf_k"]),
            )
            for source, reranker_rank in zip(source_rankings, reranked)
        ]
        final_seconds = time.perf_counter() - started
        batch_seconds = time.perf_counter() - batch_started

        if record:
            divisor = len(batch)
            measured = {
                "bm25": bm25_seconds,
                "vector_and_faiss": vector_seconds,
                "source_rrf": source_seconds,
                "task_reranker": reranker_seconds,
                "final_rrf": final_seconds,
                "complete_path": batch_seconds,
            }
            for name, seconds in measured.items():
                stage_values[name].extend([seconds * 1000.0 / divisor] * divisor)
        return final_rankings

    warmup = authority[: args.query_batch_size]
    run_batch(warmup, record=False)
    synchronize()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    predicted: list[list[str]] = []
    measured_started = time.perf_counter()
    for start in range(0, len(authority), args.query_batch_size):
        predicted.extend(run_batch(authority[start : start + args.query_batch_size], record=True))
    synchronize()
    measured_seconds = time.perf_counter() - measured_started

    top10_equal = sum(
        ranking[:10] == expected["pred_tools"][:10]
        for ranking, expected in zip(predicted, reference)
    )
    full_equal = sum(
        ranking == expected["pred_tools"]
        for ranking, expected in zip(predicted, reference)
    )
    changed_top10_ids = [
        str(row["id"])
        for row, ranking, expected in zip(authority, predicted, reference)
        if ranking[:10] != expected["pred_tools"][:10]
    ]
    quality_ks = [6, 10, 20, 30, 40, 50]
    deployment_quality = evaluate_rankings(authority, predicted, quality_ks)
    reference_quality = evaluate_rankings(
        authority, [row["pred_tools"] for row in reference], quality_ks
    )
    quality_delta = {
        f"AllHit@{k}": (
            deployment_quality[f"@{k}"]["AllHit@K"]
            - reference_quality[f"@{k}"]["AllHit@K"]
        )
        for k in quality_ks
    }
    quality_delta["MRR"] = (
        deployment_quality["overall"]["MRR"]
        - reference_quality["overall"]["MRR"]
    )
    gpu = {}
    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        gpu = {
            "name": torch.cuda.get_device_name(device_index),
            "logical_device_index": device_index,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device_index) / (1024**3),
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device_index) / (1024**3),
        }
    payload = {
        "status": "complete",
        "protocol": "warmed batched complete retrieval path",
        "rows": len(authority),
        "query_batch_size": args.query_batch_size,
        "pair_batch_size": args.pair_batch_size,
        "candidate_k": int(route["candidate_k"]),
        "warmup_batches": 1,
        "parameters": {
            "tokenizer": args.tokenizer,
            "query_instruction": args.query_instruction,
            **route,
        },
        "measured_seconds": measured_seconds,
        "throughput_queries_per_second": len(authority) / measured_seconds,
        "latency_ms_per_query": {
            name: latency_summary(values) for name, values in stage_values.items()
        },
        "model_load_seconds": {
            "vector_and_faiss": vector_load_seconds,
            "task_reranker": reranker_load_seconds,
        },
        "ranking_equivalence": {
            "top10_equal": top10_equal,
            "top10_equal_rate": top10_equal / len(authority),
            "full_equal": full_equal,
            "full_equal_rate": full_equal / len(authority),
            "changed_top10_ids": changed_top10_ids,
        },
        "quality_metrics": {
            "deployment_candidate_only_scoring": deployment_quality,
            "provided_reference_rankings": reference_quality,
            "deployment_minus_reference": quality_delta,
        },
        "hardware": {"cpu": cpu_name(), "gpu": gpu},
        "inputs": {
            "authority": str(args.input),
            "authority_sha256": sha256_file(args.input),
            "reference": str(args.reference),
            "reference_sha256": sha256_file(args.reference),
            "docs": str(args.docs),
            "docs_sha256": sha256_file(args.docs),
            "vector_db_config": str(args.fine_db / "config.json"),
            "vector_db_config_sha256": sha256_file(args.fine_db / "config.json"),
            "vector_index": str(args.fine_db / "index.faiss"),
            "vector_index_sha256": sha256_file(args.fine_db / "index.faiss"),
            "vector_model_weights": str(vector_model_weights),
            "vector_model_weights_sha256": sha256_file(vector_model_weights),
            "reranker": str(args.reranker),
            "reranker_weights_sha256": sha256_file(args.reranker / "model.safetensors"),
            "route_json": str(args.route_json),
            "route_json_sha256": sha256_file(args.route_json),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"retrieval pipeline benchmark -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
