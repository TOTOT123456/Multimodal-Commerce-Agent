#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize reranker results over candidate-depth ablation runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAG_BASELINE_DIR = PROJECT_ROOT / "data" / "rag" / "_aux" / "rag_baseline"
DEFAULT_RERANKER_DIR = PROJECT_ROOT / "data" / "rag" / "_aux" / "reranker_candidate_depth"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def allhit_at_k_from_predictions(path: Path, k: int) -> float:
    total = 0
    hit = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            total += 1
            if set(row["gold_tools"]).issubset(set(row["pred_tools"][:k])):
                hit += 1
    return hit / total if total else 0.0


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def fmt(value: float) -> str:
    return f"{value:.4f}"


def fmt_delta(value: float, base: float) -> str:
    pct = value / base - 1.0 if base else 0.0
    return f"{value - base:+.4f} ({pct:+.2%})"


def unique_max_positions(rows: list[dict], keys: list[str]) -> dict[str, int | None]:
    out: dict[str, int | None] = {}
    for key in keys:
        values = [row[key] for row in rows]
        best = max(values)
        out[key] = values.index(best) if values.count(best) == 1 else None
    return out


def maybe_bold(text: str, row_idx: int, best_idx: int | None) -> str:
    return f"**{text}**" if best_idx == row_idx else text


def build_rows(candidate_ablation: dict, reranker_dir: Path, model: str, strategy: str) -> list[dict]:
    candidate_by_n = {
        int(row["candidate_n"]): row
        for row in candidate_ablation["rows"]
        if row["model"] == model
    }
    rows = []
    for candidate_n in sorted(candidate_by_n):
        metrics_path = reranker_dir / f"{model}_{strategy}_n{candidate_n}" / "metrics.json"
        if not metrics_path.exists():
            continue
        candidate_row = candidate_by_n[candidate_n]
        metrics = load_json(metrics_path)
        candidate_metric_key = next(key for key in metrics["candidate_metrics"] if key.startswith("@"))
        rerank_metrics = metrics["rerank_metrics"]
        first_stage_a6 = candidate_row[f"{strategy}_allhit_at_6"]
        first_stage_a10 = candidate_row[f"{strategy}_allhit_at_10"]
        first_stage_a30 = candidate_row[f"{strategy}_allhit_at_30"]
        rerank_a6 = rerank_metrics.get("@6", {}).get("AllHit@K")
        if rerank_a6 is None:
            preds_path = metrics_path.parent / "preds.jsonl"
            rerank_a6 = allhit_at_k_from_predictions(preds_path, 6) if preds_path.exists() else None
        rerank_a10 = rerank_metrics["@10"]["AllHit@K"]
        rerank_a30 = rerank_metrics["@30"]["AllHit@K"]
        rows.append(
            {
                "model": model,
                "strategy": strategy,
                "candidate_n": candidate_n,
                "avg_pool_size": candidate_row["avg_pool_size"],
                "candidate_allhit_pool": metrics["candidate_metrics"][candidate_metric_key]["AllHit@K"],
                "first_stage_allhit_at_6": first_stage_a6,
                "first_stage_allhit_at_10": first_stage_a10,
                "first_stage_allhit_at_30": first_stage_a30,
                "reranker_allhit_at_6": rerank_a6,
                "reranker_allhit_at_10": rerank_a10,
                "reranker_allhit_at_30": rerank_a30,
                "reranker_allhit_at_50": rerank_metrics["@50"]["AllHit@K"],
                "reranker_mrr": rerank_metrics["overall"]["MRR"],
                "reranker_p50_ms": metrics["timing"]["rerank_p50_ms_per_query"],
                "reranker_p95_ms": metrics["timing"]["rerank_p95_ms_per_query"],
                "reranker_delta_at_10": rerank_a10 - first_stage_a10,
                "reranker_delta_at_30": rerank_a30 - first_stage_a30,
                "reranker_delta_at_10_text": fmt_delta(rerank_a10, first_stage_a10),
                "reranker_delta_at_30_text": fmt_delta(rerank_a30, first_stage_a30),
                "source_metrics": display_path(metrics_path),
            }
        )
    return rows


def write_markdown(path: Path, rows: list[dict]) -> None:
    include_a6 = all(row["reranker_allhit_at_6"] is not None for row in rows)
    metric_keys = [
        "candidate_allhit_pool",
        "first_stage_allhit_at_10",
        "reranker_allhit_at_10",
        "reranker_allhit_at_30",
        "reranker_allhit_at_50",
    ]
    if include_a6:
        metric_keys[1:1] = ["first_stage_allhit_at_6", "reranker_allhit_at_6"]
    best = unique_max_positions(rows, metric_keys)
    lines = [
        "# Reranker 候选深度 N 消融",
        "",
        "基于 `bge-large + BM25 + RRF` 的候选池复跑 `BAAI/bge-reranker-base`；候选 N 表示向量 TopN 与 BM25 TopN 先进入 RRF，再把完整并集候选交给 reranker。",
        "",
        (
            "| 候选N | 候选AllHit@pool | 平均候选数 | 一阶段RRF A@6 | Reranker A@6 | 一阶段RRF A@10 | Reranker A@10 | Reranker A@30 | Reranker A@50 | A@10提升 | p50(ms) |"
            if include_a6
            else "| 候选N | 候选AllHit@pool | 平均候选数 | 一阶段RRF A@10 | Reranker A@10 | Reranker A@30 | Reranker A@50 | A@10提升 | p50(ms) |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|" if include_a6 else "|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for idx, row in enumerate(rows):
        common = {
            "n": row["candidate_n"],
            "pool": maybe_bold(fmt(row["candidate_allhit_pool"]), idx, best["candidate_allhit_pool"]),
            "avg": row["avg_pool_size"],
            "first10": maybe_bold(fmt(row["first_stage_allhit_at_10"]), idx, best["first_stage_allhit_at_10"]),
            "rerank10": maybe_bold(fmt(row["reranker_allhit_at_10"]), idx, best["reranker_allhit_at_10"]),
            "rerank30": maybe_bold(fmt(row["reranker_allhit_at_30"]), idx, best["reranker_allhit_at_30"]),
            "rerank50": maybe_bold(fmt(row["reranker_allhit_at_50"]), idx, best["reranker_allhit_at_50"]),
            "delta10": row["reranker_delta_at_10_text"],
            "p50": row["reranker_p50_ms"],
        }
        if include_a6:
            common["first6"] = maybe_bold(fmt(row["first_stage_allhit_at_6"]), idx, best["first_stage_allhit_at_6"])
            common["rerank6"] = maybe_bold(fmt(row["reranker_allhit_at_6"]), idx, best["reranker_allhit_at_6"])
            lines.append("| {n} | {pool} | {avg:.2f} | {first6} | {rerank6} | {first10} | {rerank10} | {rerank30} | {rerank50} | {delta10} | {p50:.4f} |".format(**common))
        else:
            lines.append("| {n} | {pool} | {avg:.2f} | {first10} | {rerank10} | {rerank30} | {rerank50} | {delta10} | {p50:.4f} |".format(**common))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-ablation", type=Path, default=RAG_BASELINE_DIR / "candidate_depth_ablation.json")
    parser.add_argument("--reranker-dir", type=Path, default=DEFAULT_RERANKER_DIR)
    parser.add_argument("--model", default="bge-large")
    parser.add_argument("--strategy", default="rrf")
    parser.add_argument("--output-json", type=Path, default=RAG_BASELINE_DIR / "reranker_candidate_depth_ablation.json")
    parser.add_argument("--output-md", type=Path, default=RAG_BASELINE_DIR / "reranker_candidate_depth_ablation.md")
    args = parser.parse_args()

    candidate_ablation = load_json(args.candidate_ablation)
    rows = build_rows(candidate_ablation, args.reranker_dir, args.model, args.strategy)
    if not rows:
        raise SystemExit(f"no reranker metrics found under {args.reranker_dir}")

    payload = {
        "candidate_ablation": display_path(args.candidate_ablation),
        "reranker_metrics_root": display_path(args.reranker_dir),
        "model": args.model,
        "strategy": args.strategy,
        "rows": rows,
    }
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.output_md, rows)
    print(f"wrote {display_path(args.output_json)}")
    print(f"wrote {display_path(args.output_md)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
