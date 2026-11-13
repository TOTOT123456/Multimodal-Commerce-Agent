#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tune weighted BM25/vector fusion from existing prediction files."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_DIR = PROJECT_ROOT / "data" / "rag" / "_aux" / "rag_baseline"


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


def normalize_scores(score_map: dict[str, float]) -> dict[str, float]:
    if not score_map:
        return {}
    values = list(score_map.values())
    lo, hi = min(values), max(values)
    if math.isclose(lo, hi):
        return {key: 1.0 for key in score_map}
    return {key: (value - lo) / (hi - lo) for key, value in score_map.items()}


def fuse_weighted(vec: dict, bm25: dict, candidate_n: int, alpha: float) -> list[str]:
    vec_scores = dict(zip(vec["pred_tools"][:candidate_n], vec["scores"][:candidate_n]))
    bm25_scores = dict(zip(bm25["pred_tools"][:candidate_n], bm25["scores"][:candidate_n]))
    vec_norm = normalize_scores(vec_scores)
    bm25_norm = normalize_scores(bm25_scores)
    candidates = set(vec_norm) | set(bm25_norm)
    scored = [
        (tool_id, alpha * vec_norm.get(tool_id, 0.0) + (1.0 - alpha) * bm25_norm.get(tool_id, 0.0))
        for tool_id in candidates
    ]
    scored.sort(key=lambda item: (-item[1], item[0]))
    return [tool_id for tool_id, _ in scored]


def fuse_weighted_rrf(vec: dict, bm25: dict, candidate_n: int, vector_weight: float, rrf_k: int) -> list[str]:
    vec_rank = {tool_id: rank for rank, tool_id in enumerate(vec["pred_tools"][:candidate_n], start=1)}
    bm25_rank = {tool_id: rank for rank, tool_id in enumerate(bm25["pred_tools"][:candidate_n], start=1)}
    candidates = set(vec_rank) | set(bm25_rank)
    missing_rank = 9999
    scored = [
        (
            tool_id,
            vector_weight / (rrf_k + vec_rank.get(tool_id, missing_rank))
            + (1.0 - vector_weight) / (rrf_k + bm25_rank.get(tool_id, missing_rank)),
        )
        for tool_id in candidates
    ]
    scored.sort(key=lambda item: (-item[1], item[0]))
    return [tool_id for tool_id, _ in scored]


def allhit_at(rows: list[dict], rankings: list[list[str]], k: int) -> float:
    total = 0
    for row, ranking in zip(rows, rankings):
        gold = set(row["gold_tools"])
        total += int(bool(gold) and gold <= set(ranking[:k]))
    return total / max(1, len(rows))


def mrr(rows: list[dict], rankings: list[list[str]]) -> float:
    total = 0.0
    for row, ranking in zip(rows, rankings):
        gold = set(row["gold_tools"])
        rr = 0.0
        for rank, tool_id in enumerate(ranking, start=1):
            if tool_id in gold:
                rr = 1.0 / rank
                break
        total += rr
    return total / max(1, len(rows))


def load_predictions(base_dir: Path, model: str) -> tuple[list[dict], list[dict]]:
    preds_dir = base_dir / "preds"
    bm25_rows = load_jsonl(preds_dir / "bm25.jsonl")
    vector_path = preds_dir / f"vector_{model}.jsonl"
    vector_rows = load_jsonl(vector_path)
    if len(bm25_rows) != len(vector_rows):
        raise SystemExit(f"row count mismatch: {preds_dir / 'bm25.jsonl'} vs {vector_path}")
    return bm25_rows, vector_rows


def selected_rankings(
    bm25_rows: list[dict],
    vector_rows: list[dict],
    selected: dict,
) -> list[list[str]]:
    if selected["strategy"] == "score":
        return [
            fuse_weighted(vec, bm25, selected["candidate_n"], selected["alpha"])
            for vec, bm25 in zip(vector_rows, bm25_rows)
        ]
    return [
        fuse_weighted_rrf(vec, bm25, selected["candidate_n"], selected["vector_weight"], selected["rrf_k"])
        for vec, bm25 in zip(vector_rows, bm25_rows)
    ]


def prediction_filename(selected: dict, prefix: str = "preds") -> str:
    if selected["strategy"] == "score":
        param_name = "alpha" + str(selected["alpha"]).replace(".", "p")
    else:
        param_name = "w" + str(selected["vector_weight"]).replace(".", "p") + f"_k{selected['rrf_k']}"
    return f"{prefix}_{selected['strategy']}_n{selected['candidate_n']}_{param_name}.jsonl"


def prediction_rows(vector_rows: list[dict], rankings: list[list[str]]) -> list[dict]:
    return [
        {
            "id": row.get("id", ""),
            "query": row["query"],
            "gold_tools": row.get("gold_tools", []),
            "pred_tools": ranking,
        }
        for row, ranking in zip(vector_rows, rankings)
    ]


def evaluate_grid(
    bm25_rows: list[dict],
    vector_rows: list[dict],
    strategy: str,
    candidate_ns: list[int],
    alphas: list[float],
    rrf_vector_weights: list[float],
    rrf_ks: list[int],
    ks: list[int],
) -> tuple[list[dict], dict]:
    vector_rankings = [row["pred_tools"] for row in vector_rows]
    vector_metrics = {f"vector_allhit_at_{k}": allhit_at(vector_rows, vector_rankings, k) for k in ks}
    vector_metrics["vector_mrr"] = mrr(vector_rows, vector_rankings)

    rows = []
    for candidate_n in candidate_ns:
        pool_sizes = [
            len(set(vec["pred_tools"][:candidate_n]) | set(bm25["pred_tools"][:candidate_n]))
            for vec, bm25 in zip(vector_rows, bm25_rows)
        ]
        pool_rankings = [
            list(dict.fromkeys(vec["pred_tools"][:candidate_n] + bm25["pred_tools"][:candidate_n]))
            for vec, bm25 in zip(vector_rows, bm25_rows)
        ]
        param_grid: list[dict]
        if strategy == "score":
            param_grid = [{"alpha": alpha} for alpha in alphas]
        else:
            param_grid = [
                {"vector_weight": vector_weight, "rrf_k": rrf_k}
                for vector_weight in rrf_vector_weights
                for rrf_k in rrf_ks
            ]
        for params in param_grid:
            if strategy == "score":
                rankings = [
                    fuse_weighted(vec, bm25, candidate_n, params["alpha"])
                    for vec, bm25 in zip(vector_rows, bm25_rows)
                ]
            else:
                rankings = [
                    fuse_weighted_rrf(vec, bm25, candidate_n, params["vector_weight"], params["rrf_k"])
                    for vec, bm25 in zip(vector_rows, bm25_rows)
                ]
            row = {
                "strategy": strategy,
                "candidate_n": candidate_n,
                "avg_pool_size": statistics.mean(pool_sizes),
                "pool_allhit": allhit_at(vector_rows, pool_rankings, max(pool_sizes)),
                "mrr": mrr(vector_rows, rankings),
                **params,
            }
            for k in ks:
                row[f"allhit_at_{k}"] = allhit_at(vector_rows, rankings, k)
                row[f"delta_vs_vector_at_{k}"] = row[f"allhit_at_{k}"] - vector_metrics[f"vector_allhit_at_{k}"]
            rows.append(row)
    return rows, vector_metrics


def select_best(rows: list[dict], primary_k: int, tie_k: int | None) -> dict:
    def key(row: dict) -> tuple:
        alpha_or_weight = row.get("alpha", row.get("vector_weight", 0.0))
        rrf_k = row.get("rrf_k", 0)
        return (
            row[f"allhit_at_{primary_k}"],
            row[f"allhit_at_{tie_k}"] if tie_k is not None else 0.0,
            row["mrr"],
            row["candidate_n"],
            alpha_or_weight,
            -rrf_k,
        )

    return max(rows, key=key)


def write_markdown(path: Path, payload: dict, ks: list[int], primary_k: int) -> None:
    selected = payload["selected"]
    eval_selected = payload.get("eval_selected")

    def param_cell(row: dict) -> str:
        if row.get("strategy") == "rrf":
            return f"w={row['vector_weight']:.4g}, rrf_k={row['rrf_k']}"
        return f"alpha={row['alpha']:.4g}"

    lines = [
        "# Weighted Fusion Sweep",
        "",
        f"Model: `{payload['model']}`",
        f"Strategy: `{payload['strategy']}`",
        f"Tune predictions: `{payload['tune_dir']}`",
        f"Eval predictions: `{payload['eval_dir']}`",
        f"Selection metric: `AllHit@{primary_k}`",
        "",
        "## Selected",
        "",
        f"| Split | N | Params | {' | '.join(f'AllHit@{k}' for k in ks)} | MRR | Delta@{primary_k} vs vector |",
        "|---|---:|---:|" + "---:|" * (len(ks) + 2),
    ]
    for label, row, vector in [
        ("tune", selected, payload["tune_vector_metrics"]),
        ("eval", eval_selected, payload["eval_vector_metrics"]),
    ]:
        if not row:
            continue
        values = [
            label,
            str(row["candidate_n"]),
            param_cell(row),
            *[f"{row.get(f'allhit_at_{k}', 0.0):.4f}" for k in ks],
            f"{row['mrr']:.4f}",
            f"{row[f'allhit_at_{primary_k}'] - vector[f'vector_allhit_at_{primary_k}']:+.4f}",
        ]
        lines.append("| " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## Top Tune Rows",
            "",
            f"| N | Params | {' | '.join(f'AllHit@{k}' for k in ks)} | MRR | Delta@{primary_k} vs vector |",
            "|---:|---:|" + "---:|" * (len(ks) + 2),
        ]
    )
    top_rows = sorted(
        payload["tune_rows"],
        key=lambda row: (row[f"allhit_at_{primary_k}"], row.get("allhit_at_30", 0.0), row["mrr"]),
        reverse=True,
    )[:20]
    vector = payload["tune_vector_metrics"]
    for row in top_rows:
        values = [
            str(row["candidate_n"]),
            param_cell(row),
            *[f"{row.get(f'allhit_at_{k}', 0.0):.4f}" for k in ks],
            f"{row['mrr']:.4f}",
            f"{row[f'allhit_at_{primary_k}'] - vector[f'vector_allhit_at_{primary_k}']:+.4f}",
        ]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune-dir", type=Path, default=None, help="Prediction dir used to choose N/alpha. Defaults to eval dir.")
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR, help="Prediction dir used for final evaluation.")
    parser.add_argument("--model", required=True, help="Model key used in preds/vector_<model>.jsonl.")
    parser.add_argument("--strategy", choices=["score", "rrf"], default="score")
    parser.add_argument("--candidate-ns", nargs="+", type=int, default=[10, 20, 30, 50])
    parser.add_argument(
        "--weighted-alphas",
        nargs="+",
        type=float,
        default=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 1.0],
    )
    parser.add_argument("--rrf-vector-weights", nargs="+", type=float, default=[0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.99])
    parser.add_argument("--rrf-ks", nargs="+", type=int, default=[5, 10, 20, 30, 60, 100])
    parser.add_argument("--ks", nargs="+", type=int, default=[6, 10, 30])
    parser.add_argument("--primary-k", type=int, default=10)
    parser.add_argument("--tie-k", type=int, default=30)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--write-eval-preds", action="store_true")
    parser.add_argument(
        "--write-tune-preds",
        action="store_true",
        help="Write the selected first-stage rankings for the tuning split without repeating the grid search.",
    )
    args = parser.parse_args()
    if any(k < 6 for k in args.ks):
        parser.error("--ks values must be at least 6 for this project")
    if args.primary_k not in args.ks:
        parser.error("--primary-k must be included in --ks")
    if args.tie_k is not None and args.tie_k not in args.ks:
        parser.error("--tie-k must be included in --ks")

    tune_dir = args.tune_dir or args.eval_dir
    out_dir = args.out_dir or (args.eval_dir / "fusion_tuning")
    out_dir.mkdir(parents=True, exist_ok=True)

    tune_bm25, tune_vector = load_predictions(tune_dir, args.model)
    tune_rows, tune_vector_metrics = evaluate_grid(
        tune_bm25,
        tune_vector,
        args.strategy,
        args.candidate_ns,
        args.weighted_alphas,
        args.rrf_vector_weights,
        args.rrf_ks,
        args.ks,
    )
    selected = select_best(tune_rows, args.primary_k, args.tie_k)

    eval_bm25, eval_vector = load_predictions(args.eval_dir, args.model)
    eval_rows, eval_vector_metrics = evaluate_grid(
        eval_bm25,
        eval_vector,
        args.strategy,
        [selected["candidate_n"]],
        [selected.get("alpha", 0.0)],
        [selected.get("vector_weight", 0.0)],
        [selected.get("rrf_k", 0)],
        args.ks,
    )
    eval_selected = eval_rows[0]

    payload = {
        "model": args.model,
        "strategy": args.strategy,
        "tune_dir": str(tune_dir),
        "eval_dir": str(args.eval_dir),
        "candidate_ns": args.candidate_ns,
        "weighted_alphas": args.weighted_alphas,
        "rrf_vector_weights": args.rrf_vector_weights,
        "rrf_ks": args.rrf_ks,
        "ks": args.ks,
        "primary_k": args.primary_k,
        "tie_k": args.tie_k,
        "selected": selected,
        "eval_selected": eval_selected,
        "tune_vector_metrics": tune_vector_metrics,
        "eval_vector_metrics": eval_vector_metrics,
        "tune_rows": tune_rows,
    }
    (out_dir / "weighted_fusion_sweep.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(out_dir / "weighted_fusion_sweep.md", payload, args.ks, args.primary_k)

    if args.write_tune_preds:
        rankings = selected_rankings(tune_bm25, tune_vector, selected)
        write_jsonl(
            out_dir / prediction_filename(selected, prefix="preds_tune"),
            prediction_rows(tune_vector, rankings),
        )

    if args.write_eval_preds:
        rankings = selected_rankings(eval_bm25, eval_vector, selected)
        write_jsonl(
            out_dir / prediction_filename(selected),
            prediction_rows(eval_vector, rankings),
        )

    if args.strategy == "score":
        print(f"selected N={selected['candidate_n']} alpha={selected['alpha']}")
    else:
        print(f"selected N={selected['candidate_n']} vector_weight={selected['vector_weight']} rrf_k={selected['rrf_k']}")
    print(f"tune AllHit@{args.primary_k}={selected[f'allhit_at_{args.primary_k}']:.6f}")
    print(f"eval AllHit@{args.primary_k}={eval_selected[f'allhit_at_{args.primary_k}']:.6f}")
    print(f"wrote {out_dir / 'weighted_fusion_sweep.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
