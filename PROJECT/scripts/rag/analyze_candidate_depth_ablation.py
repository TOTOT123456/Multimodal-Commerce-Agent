#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze how first-stage candidate depth changes hybrid retrieval quality."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE_DIR = PROJECT_ROOT / "data" / "rag" / "_aux" / "rag_baseline"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_scores(score_map: dict[str, float]) -> dict[str, float]:
    if not score_map:
        return {}
    values = list(score_map.values())
    lo, hi = min(values), max(values)
    if math.isclose(lo, hi):
        return {key: 1.0 for key in score_map}
    return {key: (value - lo) / (hi - lo) for key, value in score_map.items()}


def fuse_union(vec_rank: list[str], bm25_rank: list[str], candidate_n: int) -> list[str]:
    sentinel = candidate_n + 1
    vec_pos = {tool_id: rank for rank, tool_id in enumerate(vec_rank[:candidate_n], start=1)}
    bm25_pos = {tool_id: rank for rank, tool_id in enumerate(bm25_rank[:candidate_n], start=1)}
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
    candidate_n: int,
    alpha: float,
) -> list[str]:
    vec = dict(zip(vec_rank[:candidate_n], vec_scores[:candidate_n]))
    bm25 = dict(zip(bm25_rank[:candidate_n], bm25_scores[:candidate_n]))
    vec_norm = normalize_scores(vec)
    bm25_norm = normalize_scores(bm25)
    candidates = set(vec_norm) | set(bm25_norm)
    scored = [
        (tool_id, alpha * vec_norm.get(tool_id, 0.0) + (1.0 - alpha) * bm25_norm.get(tool_id, 0.0))
        for tool_id in candidates
    ]
    scored.sort(key=lambda item: (-item[1], item[0]))
    return [tool_id for tool_id, _ in scored]


def fuse_rrf(vec_rank: list[str], bm25_rank: list[str], candidate_n: int, rrf_k: int) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in (vec_rank[:candidate_n], bm25_rank[:candidate_n]):
        for rank, tool_id in enumerate(ranking, start=1):
            scores[tool_id] = scores.get(tool_id, 0.0) + 1.0 / (rrf_k + rank)
    return [tool_id for tool_id, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]


def allhit_at(rows: list[dict], rankings: list[list[str]], k: int) -> float:
    total = 0
    for row, ranking in zip(rows, rankings):
        gold = set(row["gold_tools"])
        total += int(bool(gold) and gold <= set(ranking[:k]))
    return total / max(1, len(rows))


def unique_max_positions(rows: list[dict], keys: list[str]) -> dict[str, int | None]:
    out: dict[str, int | None] = {}
    for key in keys:
        values = [row[key] for row in rows]
        best = max(values)
        if values.count(best) == 1:
            out[key] = values.index(best)
        else:
            out[key] = None
    return out


def fmt_metric(value: float, bold: bool) -> str:
    text = f"{value:.4f}"
    return f"**{text}**" if bold else text


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def write_markdown(path: Path, rows: list[dict], ks: list[int]) -> None:
    metric_keys = (
        ["union_pool_allhit"]
        + [f"union_allhit_at_{k}" for k in ks]
        + [f"weighted_allhit_at_{k}" for k in ks]
        + [f"rrf_allhit_at_{k}" for k in ks]
    )
    best_by_model: dict[str, dict[str, int | None]] = {}
    for model in dict.fromkeys(row["model"] for row in rows):
        model_rows = [row for row in rows if row["model"] == model]
        model_best = unique_max_positions(model_rows, metric_keys)
        offset_rows = [idx for idx, row in enumerate(rows) if row["model"] == model]
        best_by_model[model] = {
            key: (offset_rows[pos] if pos is not None else None)
            for key, pos in model_best.items()
        }
    metric_headers = (
        [f"Union A@{k}" for k in ks]
        + [f"Weighted A@{k}" for k in ks]
        + [f"RRF A@{k}" for k in ks]
    )
    table_header = "| 向量模型 | 候选N | 并集候选AllHit@pool | 平均候选数 | " + " | ".join(metric_headers) + " |"
    table_separator = "|" + "---|" + "---:|" * (3 + len(metric_headers))
    header = [
        "# 候选深度 N 消融",
        "",
        "基于 `preds/bm25.jsonl` 与各 `preds/vector_*.jsonl` 重算；候选深度 N 表示向量 TopN 与 BM25 TopN 进入融合。",
        "",
        table_header,
        table_separator,
    ]
    lines = header[:]
    previous_model = ""
    for idx, row in enumerate(rows):
        if previous_model and row["model"] != previous_model:
            lines.append("| " + " | ".join([""] * (4 + len(metric_headers))) + " |")
        previous_model = row["model"]
        best_pos = best_by_model[row["model"]]
        values = [
            row["model"],
            str(row["candidate_n"]),
            fmt_metric(row["union_pool_allhit"], best_pos["union_pool_allhit"] == idx),
            f"{row['avg_pool_size']:.2f}",
        ]
        for k in ks:
            key = f"union_allhit_at_{k}"
            values.append(fmt_metric(row[key], best_pos[key] == idx))
        for k in ks:
            key = f"weighted_allhit_at_{k}"
            values.append(fmt_metric(row[key], best_pos[key] == idx))
        for k in ks:
            key = f"rrf_allhit_at_{k}"
            values.append(fmt_metric(row[key], best_pos[key] == idx))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_predictions(path: Path, source_rows: list[dict], rankings: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row, ranking in zip(source_rows, rankings):
            item = {
                "id": row.get("id", ""),
                "query": row["query"],
                "gold_tools": row.get("gold_tools", []),
                "pred_tools": ranking,
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--candidate-ns", nargs="+", type=int, default=[10, 20, 30, 50])
    parser.add_argument("--ks", nargs="+", type=int, default=[6, 10, 30])
    parser.add_argument("--weighted-alpha", type=float, default=0.7)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--write-preds", action="store_true", help="Write selected fused rankings as reranker candidates.")
    parser.add_argument("--pred-models", nargs="*", default=[], help="Model names to write when --write-preds is set. Empty means all.")
    parser.add_argument("--pred-strategies", nargs="+", choices=["union", "weighted", "rrf"], default=["rrf"])
    parser.add_argument("--preds-out-dir", type=Path, default=None)
    args = parser.parse_args()

    preds_dir = args.baseline_dir / "preds"
    pred_model_filter = set(args.pred_models)
    preds_out_dir = args.preds_out_dir or (args.baseline_dir / "preds_candidate_depth")
    bm25_rows = load_jsonl(preds_dir / "bm25.jsonl")
    vector_paths = sorted(preds_dir.glob("vector_*.jsonl"))
    if not bm25_rows or not vector_paths:
        raise SystemExit(f"missing prediction files under {preds_dir}")

    rows = []
    for vector_path in vector_paths:
        model = vector_path.stem.removeprefix("vector_")
        vector_rows = load_jsonl(vector_path)
        if len(vector_rows) != len(bm25_rows):
            raise SystemExit(f"row count mismatch: {vector_path}")

        for candidate_n in args.candidate_ns:
            union_rankings = []
            weighted_rankings = []
            rrf_rankings = []
            pool_sizes = []
            for vec, bm25 in zip(vector_rows, bm25_rows):
                vec_rank = vec["pred_tools"]
                bm25_rank = bm25["pred_tools"]
                union_ranking = fuse_union(vec_rank, bm25_rank, candidate_n)
                union_rankings.append(union_ranking)
                pool_sizes.append(len(union_ranking))
                weighted_rankings.append(
                    fuse_weighted(
                        vec_rank,
                        vec["scores"],
                        bm25_rank,
                        bm25["scores"],
                        candidate_n,
                        args.weighted_alpha,
                    )
                )
                rrf_rankings.append(fuse_rrf(vec_rank, bm25_rank, candidate_n, args.rrf_k))

            row = {
                "model": model,
                "candidate_n": candidate_n,
                "union_pool_allhit": allhit_at(vector_rows, union_rankings, max(pool_sizes)),
                "avg_pool_size": statistics.mean(pool_sizes),
                "weighted_alpha": args.weighted_alpha,
                "rrf_k": args.rrf_k,
            }
            for k in args.ks:
                row[f"union_allhit_at_{k}"] = allhit_at(vector_rows, union_rankings, k)
                row[f"weighted_allhit_at_{k}"] = allhit_at(vector_rows, weighted_rankings, k)
                row[f"rrf_allhit_at_{k}"] = allhit_at(vector_rows, rrf_rankings, k)
            rows.append(row)

            if args.write_preds and (not pred_model_filter or model in pred_model_filter):
                strategy_rankings = {
                    "union": union_rankings,
                    "weighted": weighted_rankings,
                    "rrf": rrf_rankings,
                }
                for strategy in args.pred_strategies:
                    out_path = preds_out_dir / f"hybrid_{model}_{strategy}_n{candidate_n}.jsonl"
                    write_predictions(out_path, vector_rows, strategy_rankings[strategy])

    out_json = args.baseline_dir / "candidate_depth_ablation.json"
    out_md = args.baseline_dir / "candidate_depth_ablation.md"
    payload = {
        "source_predictions": display_path(preds_dir),
        "candidate_ns": args.candidate_ns,
        "ks": args.ks,
        "weighted_alpha": args.weighted_alpha,
        "rrf_k": args.rrf_k,
        "rows": rows,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(out_md, rows, args.ks)
    print(f"wrote {display_path(out_json)}")
    print(f"wrote {display_path(out_md)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
