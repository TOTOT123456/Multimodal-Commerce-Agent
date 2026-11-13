#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run BM25, vector, and hybrid retrieval baselines over the 100-tool corpus."""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import RAG_DIR, TOOL_CORPUS_PATH, TOOL_TRACE_DIR, load_jsonl, split_tools, write_json


DEFAULT_VECTOR_DBS = RAG_DIR / "vector_dbs"
DEFAULT_OUTPUT = RAG_DIR / "_aux" / "rag_baseline"
DEFAULT_QUERY_INSTRUCTION = "为电商导购助手检索最相关的工具描述："


def example_query(row: dict) -> str:
    for key in ("问题", "query"):
        if key in row:
            return str(row[key])
    raise KeyError("example has neither '问题' nor 'query'")


def example_gold_tools(row: dict) -> list[str]:
    if "工具" in row:
        return split_tools(row["工具"])
    for key in ("gold_tools", "pos_tool_ids"):
        if key in row:
            value = row[key]
            if isinstance(value, list):
                return [str(tool_id) for tool_id in value]
            return split_tools(str(value))
    raise KeyError("example has none of '工具', 'gold_tools', or 'pos_tool_ids'")


def doc_text(doc: dict, field: str) -> str:
    if field in doc:
        return str(doc[field])
    for fallback in ("text", "document", "enhanced_desc", "description"):
        if fallback in doc:
            return str(doc[fallback])
    return json.dumps(doc, ensure_ascii=False)


def fallback_tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens: list[str] = []
    for block in re.findall(r"[\u4e00-\u9fff]+|[a-z0-9_]+", text):
        if re.fullmatch(r"[\u4e00-\u9fff]+", block):
            tokens.extend(block)
            tokens.extend(block[i : i + 2] for i in range(max(0, len(block) - 1)))
            tokens.extend(block[i : i + 3] for i in range(max(0, len(block) - 2)))
        else:
            tokens.append(block)
            tokens.extend(part for part in block.split("_") if part)
    return tokens


def build_tokenizer(mode: str):
    if mode in {"auto", "jieba"}:
        try:
            import jieba

            return lambda text: [token.strip().lower() for token in jieba.lcut(text) if token.strip()]
        except ImportError:
            if mode == "jieba":
                raise
    return fallback_tokenize


class BM25:
    def __init__(self, tokenized_docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs = tokenized_docs
        self.k1 = k1
        self.b = b
        self.n = len(tokenized_docs)
        self.doc_lens = [len(doc) for doc in tokenized_docs]
        self.avgdl = sum(self.doc_lens) / max(1, self.n)
        self.term_freqs = [Counter(doc) for doc in tokenized_docs]
        df = Counter()
        for doc in tokenized_docs:
            df.update(set(doc))
        self.idf = {term: math.log(1.0 + (self.n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

    def scores(self, query_tokens: list[str]) -> list[float]:
        scores = [0.0] * self.n
        for idx, freqs in enumerate(self.term_freqs):
            dl = self.doc_lens[idx]
            score = 0.0
            for term in query_tokens:
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl)
                score += self.idf.get(term, 0.0) * tf * (self.k1 + 1.0) / denom
            scores[idx] = score
        return scores


def rank_from_scores(scores: list[float], docs: list[dict], top_k: int) -> tuple[list[str], list[float]]:
    order = sorted(range(len(scores)), key=lambda idx: (-scores[idx], docs[idx]["tool_id"]))
    order = order[:top_k]
    return [docs[idx]["tool_id"] for idx in order], [float(scores[idx]) for idx in order]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[int(pct) - 1]


def evaluate_rankings(examples: list[dict], rankings: list[list[str]], ks: list[int]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    gold_sets = [set(example_gold_tools(row)) for row in examples]
    for k in ks:
        hit_any = 0
        all_hit = 0
        recall_sum = 0.0
        precision_sum = 0.0
        for gold, ranking in zip(gold_sets, rankings):
            pred = set(ranking[:k])
            matched = gold & pred
            hit_any += int(bool(matched))
            all_hit += int(bool(gold) and gold <= pred)
            recall_sum += len(matched) / max(1, len(gold))
            precision_sum += len(matched) / k
        n = max(1, len(examples))
        out[f"@{k}"] = {
            "Hit@K": round(hit_any / n, 6),
            "AllHit@K": round(all_hit / n, 6),
            "Recall@K": round(recall_sum / n, 6),
            "Precision@K": round(precision_sum / n, 6),
        }

    mrr_sum = 0.0
    for gold, ranking in zip(gold_sets, rankings):
        rr = 0.0
        for rank, tool_id in enumerate(ranking, start=1):
            if tool_id in gold:
                rr = 1.0 / rank
                break
        mrr_sum += rr
    out["overall"] = {"MRR": round(mrr_sum / max(1, len(examples)), 6), "count": len(examples)}
    return out


def run_bm25(
    docs: list[dict],
    examples: list[dict],
    max_k: int,
    tokenizer_mode: str,
    text_field: str,
) -> tuple[list[list[str]], list[list[float]], dict[str, float]]:
    tokenize = build_tokenizer(tokenizer_mode)
    bm25 = BM25([tokenize(doc_text(doc, text_field)) for doc in docs])

    rankings: list[list[str]] = []
    score_rows: list[list[float]] = []
    latencies = []
    for row in examples:
        t0 = time.perf_counter()
        scores = bm25.scores(tokenize(example_query(row)))
        tools, top_scores = rank_from_scores(scores, docs, max_k)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        rankings.append(tools)
        score_rows.append(top_scores)

    timing = {
        "p50_ms": round(statistics.median(latencies), 4),
        "p95_ms": round(percentile(latencies, 95), 4),
    }
    return rankings, score_rows, timing


def load_sentence_transformer(model_name_or_path: str, device: str):
    from sentence_transformers import SentenceTransformer

    kwargs = {"trust_remote_code": True}
    if device != "auto":
        kwargs["device"] = device
    try:
        return SentenceTransformer(model_name_or_path, **kwargs)
    except TypeError:
        kwargs.pop("trust_remote_code", None)
        return SentenceTransformer(model_name_or_path, **kwargs)


def run_vector_db(
    db_dir: Path,
    examples: list[dict],
    max_k: int,
    batch_size: int,
    device: str,
    query_instruction: str,
) -> tuple[list[list[str]], list[list[float]], dict[str, float], dict]:
    try:
        import faiss
    except ImportError as exc:
        raise SystemExit(
            "Missing vector dependency. Use the llm_bge conda env or install the RAG requirements.\n"
            f"Original error: {exc}"
        )

    docs = load_jsonl(db_dir / "docs.jsonl")
    config = json.loads((db_dir / "config.json").read_text(encoding="utf-8"))
    load_started = time.perf_counter()
    model = load_sentence_transformer(config["model_name_or_path"], device)
    index = faiss.read_index(str(db_dir / "index.faiss"))
    load_seconds = time.perf_counter() - load_started
    queries = [
        f"{query_instruction}{example_query(row)}" if query_instruction else example_query(row)
        for row in examples
    ]

    t0 = time.perf_counter()
    query_embeddings = model.encode(
        queries,
        batch_size=batch_size,
        normalize_embeddings=bool(config.get("normalize_embeddings", True)),
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    encode_ms_per_query = (time.perf_counter() - t0) * 1000.0 / max(1, len(queries))

    search_latencies = []
    rankings: list[list[str]] = []
    score_rows: list[list[float]] = []
    for emb in query_embeddings:
        t1 = time.perf_counter()
        scores, idxs = index.search(emb.reshape(1, -1), max_k)
        search_latencies.append((time.perf_counter() - t1) * 1000.0)
        rankings.append([docs[int(idx)]["tool_id"] for idx in idxs[0] if int(idx) >= 0])
        score_rows.append([float(score) for score in scores[0]])

    timing = {
        "model_and_index_load_seconds": round(load_seconds, 6),
        "encode_avg_ms": round(encode_ms_per_query, 4),
        "search_p50_ms": round(statistics.median(search_latencies), 4),
        "search_p95_ms": round(percentile(search_latencies, 95), 4),
    }
    return rankings, score_rows, timing, config


def normalize_scores(score_map: dict[str, float]) -> dict[str, float]:
    if not score_map:
        return {}
    values = list(score_map.values())
    lo, hi = min(values), max(values)
    if math.isclose(lo, hi):
        return {key: 1.0 for key in score_map}
    return {key: (value - lo) / (hi - lo) for key, value in score_map.items()}


def fuse_union(vec_rank: list[str], bm25_rank: list[str], candidate_k: int) -> list[str]:
    sentinel = candidate_k + 1
    vec_pos = {tool_id: rank for rank, tool_id in enumerate(vec_rank[:candidate_k], start=1)}
    bm25_pos = {tool_id: rank for rank, tool_id in enumerate(bm25_rank[:candidate_k], start=1)}
    candidates = set(vec_pos) | set(bm25_pos)
    return sorted(
        candidates,
        key=lambda tool_id: (
            min(vec_pos.get(tool_id, sentinel), bm25_pos.get(tool_id, sentinel)),
            vec_pos.get(tool_id, sentinel) + bm25_pos.get(tool_id, sentinel),
            tool_id,
        ),
    )


def fuse_weighted(
    vec_rank: list[str],
    vec_scores: list[float],
    bm25_rank: list[str],
    bm25_scores: list[float],
    candidate_k: int,
    alpha: float,
) -> list[str]:
    vec = dict(zip(vec_rank[:candidate_k], vec_scores[:candidate_k]))
    bm25 = dict(zip(bm25_rank[:candidate_k], bm25_scores[:candidate_k]))
    vec_norm = normalize_scores(vec)
    bm25_norm = normalize_scores(bm25)
    candidates = set(vec_norm) | set(bm25_norm)
    scored = [
        (tool_id, alpha * vec_norm.get(tool_id, 0.0) + (1.0 - alpha) * bm25_norm.get(tool_id, 0.0))
        for tool_id in candidates
    ]
    scored.sort(key=lambda item: (-item[1], item[0]))
    return [tool_id for tool_id, _ in scored]


def fuse_rrf(vec_rank: list[str], bm25_rank: list[str], candidate_k: int, rrf_k: int) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in (vec_rank[:candidate_k], bm25_rank[:candidate_k]):
        for rank, tool_id in enumerate(ranking, start=1):
            scores[tool_id] = scores.get(tool_id, 0.0) + 1.0 / (rrf_k + rank)
    return [tool_id for tool_id, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]


def safe_name(name: str) -> str:
    return name.replace(":", "_").replace("/", "_")


def write_predictions(
    path: Path,
    examples: list[dict],
    rankings: list[list[str]],
    score_rows: list[list[float]] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for idx, (row, ranking) in enumerate(zip(examples, rankings)):
            item = {
                "id": row.get("id", row.get("trace_id", idx)),
                "query": example_query(row),
                "gold_tools": example_gold_tools(row),
                "pred_tools": ranking,
            }
            if score_rows:
                item["scores"] = score_rows[idx]
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def add_result(
    results: dict,
    name: str,
    examples: list[dict],
    rankings: list[list[str]],
    ks: list[int],
    timing: dict,
    extra: dict | None = None,
) -> None:
    results[name] = {"metrics": evaluate_rankings(examples, rankings, ks), "timing": timing}
    if extra:
        results[name]["extra"] = extra


def write_markdown(path: Path, results: dict, ks: list[int]) -> None:
    def unique_max(values: list[float | None]) -> float | None:
        vals = [value for value in values if value is not None]
        if not vals:
            return None
        best = max(vals)
        if sum(math.isclose(value, best, rel_tol=0.0, abs_tol=1e-12) for value in vals) == 1:
            return best
        return None

    def metric_cell(value: float, best: float | None) -> str:
        text = f"{value:.4f}"
        if best is not None and math.isclose(value, best, rel_tol=0.0, abs_tol=1e-12):
            return f"**{text}**"
        return text

    def allhit_uplift(name: str, k: int) -> tuple[float, float] | None:
        if not name.startswith("hybrid:"):
            return None
        parts = name.split(":")
        if len(parts) < 3:
            return None
        base_name = f"vector:{parts[1]}"
        if base_name not in results:
            return None
        current = results[name]["metrics"][f"@{k}"]["AllHit@K"]
        base_value = results[base_name]["metrics"][f"@{k}"]["AllHit@K"]
        delta = current - base_value
        pct = 0.0 if math.isclose(base_value, 0.0) else delta / base_value * 100.0
        return delta, pct

    best_by_k = {}
    for k in ks:
        names = list(results.keys())
        best_by_k[k] = {
            "Hit@K": unique_max([results[name]["metrics"][f"@{k}"]["Hit@K"] for name in names]),
            "AllHit@K": unique_max([results[name]["metrics"][f"@{k}"]["AllHit@K"] for name in names]),
            "Recall@K": unique_max([results[name]["metrics"][f"@{k}"]["Recall@K"] for name in names]),
            "Precision@K": unique_max([results[name]["metrics"][f"@{k}"]["Precision@K"] for name in names]),
            "MRR": unique_max([results[name]["metrics"]["overall"]["MRR"] for name in names]),
            "uplift": unique_max([allhit_uplift(name, k)[0] for name in names if allhit_uplift(name, k)]),
        }

    lines = [
        "# RAG Baseline Metrics",
        "",
        "| Retriever | K | Hit@K | AllHit@K | Recall@K | Precision@K | MRR | AllHit@K提升 | Timing |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for name, payload in results.items():
        for k in ks:
            mrr = payload["metrics"]["overall"]["MRR"]
            timing = ", ".join(f"{key}={value}" for key, value in payload.get("timing", {}).items())
            row = payload["metrics"][f"@{k}"]
            best = best_by_k[k]
            uplift = allhit_uplift(name, k)
            if uplift is None:
                uplift_cell = "-"
            else:
                delta, pct = uplift
                uplift_cell = f"{delta:+.4f} ({pct:.2f}%)"
                if best["uplift"] is not None and math.isclose(delta, best["uplift"], rel_tol=0.0, abs_tol=1e-12):
                    uplift_cell = f"**{uplift_cell}**"
            lines.append(
                "| {name} | {k} | {hit} | {allhit} | {recall} | {precision} | {mrr} | {uplift} | {timing} |".format(
                    name=name,
                    k=k,
                    hit=metric_cell(row["Hit@K"], best["Hit@K"]),
                    allhit=metric_cell(row["AllHit@K"], best["AllHit@K"]),
                    recall=metric_cell(row["Recall@K"], best["Recall@K"]),
                    precision=metric_cell(row["Precision@K"], best["Precision@K"]),
                    mrr=metric_cell(mrr, best["MRR"]),
                    uplift=uplift_cell,
                    timing=timing,
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def existing_vector_dbs(vector_db_dir: Path, selected: Iterable[str]) -> list[Path]:
    selected_set = set(selected)
    out = []
    for config_path in sorted(vector_db_dir.glob("*/config.json")):
        db_dir = config_path.parent
        if selected_set and db_dir.name not in selected_set:
            continue
        if (db_dir / "index.faiss").exists() and (db_dir / "docs.jsonl").exists():
            out.append(db_dir)
    return out


def write_compat_summary(path: Path, results: dict, args: argparse.Namespace, elapsed_sec: float, num_tools: int, num_queries: int) -> None:
    methods = {}
    for name, payload in results.items():
        flat = {"MRR": payload["metrics"]["overall"]["MRR"]}
        for k in args.ks:
            row = payload["metrics"][f"@{k}"]
            flat[f"Hit@{k}"] = row["Hit@K"]
            flat[f"AllHit@{k}"] = row["AllHit@K"]
            flat[f"Recall@{k}"] = row["Recall@K"]
            flat[f"Precision@{k}"] = row["Precision@K"]
        methods[name] = flat
    write_json(
        path,
        {
            "test": str(args.test_file),
            "corpus": str(args.docs),
            "predictions_dir": str(args.output_dir / "preds"),
            "metrics": str(args.output_dir / "metrics.json"),
            "num_tools": num_tools,
            "num_queries": num_queries,
            "k_values": args.ks,
            "candidate_k": args.candidate_k,
            "elapsed_sec": round(elapsed_sec, 4),
            "methods": methods,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=Path, default=TOOL_CORPUS_PATH)
    parser.add_argument("--test-file", type=Path, default=TOOL_TRACE_DIR / "test.jsonl")
    parser.add_argument("--vector-db-dir", type=Path, default=DEFAULT_VECTOR_DBS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", choices=["bm25", "vector", "hybrid", "all"], default="all")
    parser.add_argument("--models", nargs="*", default=[], help="Vector DB names to evaluate. Empty means all existing DBs.")
    parser.add_argument("--ks", nargs="+", type=int, default=[6, 7, 8, 9, 10, 15, 20, 30, 50])
    parser.add_argument("--limit", type=int, default=0, help="Debug with the first N examples.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tokenizer", choices=["auto", "jieba", "fallback"], default="auto")
    parser.add_argument("--text-field", default="document")
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--weighted-alpha", type=float, default=0.7)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--query-instruction", default=DEFAULT_QUERY_INSTRUCTION)
    parser.add_argument("--compat-metrics-out", type=Path, default=RAG_DIR / "_aux" / "retrieval_baseline_metrics.json")
    args = parser.parse_args()

    t0 = time.perf_counter()
    docs = load_jsonl(args.docs)
    examples = load_jsonl(args.test_file)
    if not docs:
        raise SystemExit(f"empty docs: {args.docs}")
    if not examples:
        raise SystemExit(f"empty test file: {args.test_file}")
    if args.limit:
        examples = examples[: args.limit]

    max_k = min(max(args.ks + [args.candidate_k]), len(docs))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    preds_dir = args.output_dir / "preds"

    results: dict = {}
    bm25_rankings: list[list[str]] | None = None
    bm25_scores: list[list[float]] | None = None

    if args.mode in {"bm25", "hybrid", "all"}:
        bm25_rankings, bm25_scores, timing = run_bm25(docs, examples, max_k, args.tokenizer, args.text_field)
        add_result(results, "bm25", examples, bm25_rankings, args.ks, timing, {"tokenizer": args.tokenizer, "text_field": args.text_field})
        write_predictions(preds_dir / "bm25.jsonl", examples, bm25_rankings, bm25_scores)
        print("evaluated bm25")

    vector_payloads: dict[str, tuple[list[list[str]], list[list[float]]]] = {}
    if args.mode in {"vector", "hybrid", "all"}:
        dbs = existing_vector_dbs(args.vector_db_dir, args.models)
        if not dbs and args.mode == "vector":
            raise SystemExit(f"no vector dbs found in {args.vector_db_dir}; run build_vector_dbs.py first")
        if not dbs and args.mode in {"hybrid", "all"}:
            print(f"no vector dbs found in {args.vector_db_dir}; skipped vector/hybrid")
        for db_dir in dbs:
            rankings, scores, timing, config = run_vector_db(
                db_dir=db_dir,
                examples=examples,
                max_k=max_k,
                batch_size=args.batch_size,
                device=args.device,
                query_instruction=args.query_instruction,
            )
            name = f"vector:{db_dir.name}"
            vector_payloads[db_dir.name] = (rankings, scores)
            add_result(results, name, examples, rankings, args.ks, timing, config)
            write_predictions(preds_dir / f"{safe_name(name)}.jsonl", examples, rankings, scores)
            print(f"evaluated {name}")

    if args.mode in {"hybrid", "all"} and bm25_rankings and bm25_scores:
        for model_key, (vec_rankings, vec_scores) in vector_payloads.items():
            strategies = (f"union_top{args.candidate_k}", "weighted", "rrf")
            for strategy in strategies:
                fused_rankings = []
                for vr, vs, br, bs in zip(vec_rankings, vec_scores, bm25_rankings, bm25_scores):
                    if strategy.startswith("union_top"):
                        fused_rankings.append(fuse_union(vr, br, args.candidate_k))
                    elif strategy == "weighted":
                        fused_rankings.append(fuse_weighted(vr, vs, br, bs, args.candidate_k, args.weighted_alpha))
                    else:
                        fused_rankings.append(fuse_rrf(vr, br, args.candidate_k, args.rrf_k))
                name = f"hybrid:{model_key}:{strategy}"
                extra = {
                    "candidate_k": args.candidate_k,
                    "weighted_alpha": args.weighted_alpha if strategy == "weighted" else None,
                    "rrf_k": args.rrf_k if strategy == "rrf" else None,
                }
                if strategy.startswith("union_top"):
                    extra["ordering"] = "union sorted by best rank, then rank sum, then tool_id"
                add_result(results, name, examples, fused_rankings, args.ks, {}, extra)
                write_predictions(preds_dir / f"{safe_name(name)}.jsonl", examples, fused_rankings)
                print(f"evaluated {name}")

    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.output_dir / "metrics.md", results, args.ks)
    write_compat_summary(args.compat_metrics_out, results, args, time.perf_counter() - t0, len(docs), len(examples))
    print(f"wrote metrics -> {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
