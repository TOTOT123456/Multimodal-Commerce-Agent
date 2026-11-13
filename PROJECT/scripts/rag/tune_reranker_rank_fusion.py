#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tune rank fusion between a first-stage source and reranker output."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


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


def assert_aligned(left: list[dict], right: list[dict], left_name: str, right_name: str) -> None:
    if len(left) != len(right):
        raise SystemExit(f"row count mismatch: {left_name}={len(left)} vs {right_name}={len(right)}")
    for idx, (a, b) in enumerate(zip(left, right)):
        if a.get("id") and b.get("id") and a.get("id") != b.get("id"):
            raise SystemExit(f"id mismatch at row {idx}: {a.get('id')} vs {b.get('id')}")
        if a.get("query") != b.get("query"):
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


def evaluate(rows: list[dict], rankings: list[list[str]], ks: list[int]) -> dict:
    metrics = {f"allhit_at_{k}": allhit_at(rows, rankings, k) for k in ks}
    metrics["mrr"] = mrr(rows, rankings)
    return metrics


def rrf_fuse(source_rank: list[str], reranker_rank: list[str], source_weight: float, rrf_k: int) -> list[str]:
    source_pos = {tool_id: rank for rank, tool_id in enumerate(source_rank, start=1)}
    reranker_pos = {tool_id: rank for rank, tool_id in enumerate(reranker_rank, start=1)}
    missing_rank = 9999
    candidates = set(source_pos) | set(reranker_pos)
    scored = []
    for tool_id in candidates:
        score = (
            source_weight / (rrf_k + source_pos.get(tool_id, missing_rank))
            + (1.0 - source_weight) / (rrf_k + reranker_pos.get(tool_id, missing_rank))
        )
        scored.append((tool_id, score, source_pos.get(tool_id, missing_rank), reranker_pos.get(tool_id, missing_rank)))
    scored.sort(key=lambda item: (-item[1], item[2], item[3], item[0]))
    return [tool_id for tool_id, _, _, _ in scored]


def fuse_rows(source_rows: list[dict], reranker_rows: list[dict], source_weight: float, rrf_k: int) -> list[list[str]]:
    return [
        rrf_fuse(source["pred_tools"], reranker["pred_tools"], source_weight, rrf_k)
        for source, reranker in zip(source_rows, reranker_rows)
    ]


def sweep(
    source_rows: list[dict],
    reranker_rows: list[dict],
    source_weights: list[float],
    rrf_ks: list[int],
    ks: list[int],
) -> list[dict]:
    rows = []
    source_metrics = evaluate(source_rows, [row["pred_tools"] for row in source_rows], ks)
    for source_weight in source_weights:
        for rrf_k in rrf_ks:
            rankings = fuse_rows(source_rows, reranker_rows, source_weight, rrf_k)
            row = {
                "source_weight": source_weight,
                "rrf_k": rrf_k,
                **evaluate(source_rows, rankings, ks),
            }
            for k in ks:
                row[f"delta_vs_source_at_{k}"] = row[f"allhit_at_{k}"] - source_metrics[f"allhit_at_{k}"]
            rows.append(row)
    return rows


def select_best(rows: list[dict], primary_k: int, tie_k: int | None) -> dict:
    def key(row: dict) -> tuple:
        return (
            row[f"allhit_at_{primary_k}"],
            row[f"allhit_at_{tie_k}"] if tie_k is not None else 0.0,
            row["mrr"],
            row.get("allhit_at_6", 0.0),
            row["source_weight"],
            -row["rrf_k"],
        )

    return max(rows, key=key)


def write_markdown(path: Path, payload: dict, primary_k: int) -> None:
    selected = payload["selected"]
    eval_selected = payload["eval_selected"]
    ks = payload["ks"]

    def row_values(label: str, row: dict, source_metrics: dict) -> list[str]:
        return [
            label,
            f"w={row['source_weight']:.4g}, rrf_k={row['rrf_k']}",
            *[f"{row.get(f'allhit_at_{k}', 0.0):.4f}" for k in ks],
            f"{row['mrr']:.4f}",
            f"{row[f'allhit_at_{primary_k}'] - source_metrics[f'allhit_at_{primary_k}']:+.4f}",
        ]

    header_metrics = " | ".join([f"AllHit@{k}" for k in ks])
    lines = [
        "# Reranker Rank Fusion Sweep",
        "",
        f"Tune source: `{payload['tune_source_preds']}`",
        f"Tune reranker: `{payload['tune_reranker_preds']}`",
        f"Eval source: `{payload['eval_source_preds']}`",
        f"Eval reranker: `{payload['eval_reranker_preds']}`",
        f"Selection metric: `AllHit@{primary_k}`",
        "",
        "## Selected",
        "",
        f"| Split | Params | {header_metrics} | MRR | Delta@{primary_k} vs source |",
        "|---|---:|" + "---:|" * (len(ks) + 2),
    ]
    for values in [
        row_values("tune", selected, payload["tune_source_metrics"]),
        row_values("eval", eval_selected, payload["eval_source_metrics"]),
    ]:
        lines.append("| " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## Baselines",
            "",
            f"| Split | Method | {header_metrics} | MRR |",
            "|---|---|" + "---:|" * (len(ks) + 1),
        ]
    )
    for split in ["tune", "eval"]:
        for method in ["source", "reranker"]:
            metrics = payload[f"{split}_{method}_metrics"]
            values = [split, method, *[f"{metrics.get(f'allhit_at_{k}', 0.0):.4f}" for k in ks], f"{metrics['mrr']:.4f}"]
            lines.append("| " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## Top Tune Rows",
            "",
            f"| Params | {header_metrics} | MRR | Delta@{primary_k} vs source |",
            "|---:|" + "---:|" * (len(ks) + 2),
        ]
    )
    top_rows = sorted(
        payload["tune_rows"],
        key=lambda row: (row[f"allhit_at_{primary_k}"], row.get("allhit_at_30", 0.0), row["mrr"], row["source_weight"], -row["rrf_k"]),
        reverse=True,
    )[:20]
    source_metrics = payload["tune_source_metrics"]
    for row in top_rows:
        values = [
            f"w={row['source_weight']:.4g}, rrf_k={row['rrf_k']}",
            *[f"{row.get(f'allhit_at_{k}', 0.0):.4f}" for k in ks],
            f"{row['mrr']:.4f}",
            f"{row[f'allhit_at_{primary_k}'] - source_metrics[f'allhit_at_{primary_k}']:+.4f}",
        ]
        lines.append("| " + " | ".join(values) + " |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune-source-preds", type=Path, required=True)
    parser.add_argument("--tune-reranker-preds", type=Path, required=True)
    parser.add_argument("--eval-source-preds", type=Path, required=True)
    parser.add_argument("--eval-reranker-preds", type=Path, required=True)
    parser.add_argument("--source-weights", nargs="+", type=float, default=[0.7, 0.75, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.99])
    parser.add_argument("--rrf-ks", nargs="+", type=int, default=[1, 2, 3, 5, 8, 10, 20, 30, 60])
    parser.add_argument("--ks", nargs="+", type=int, default=[6, 10, 30, 50])
    parser.add_argument("--primary-k", type=int, default=10)
    parser.add_argument("--tie-k", type=int, default=30)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--write-eval-preds", action="store_true")
    args = parser.parse_args()
    if any(k < 6 for k in args.ks):
        parser.error("--ks values must be at least 6 for this project")
    if args.primary_k not in args.ks:
        parser.error("--primary-k must be included in --ks")
    if args.tie_k is not None and args.tie_k not in args.ks:
        parser.error("--tie-k must be included in --ks")

    tune_source = load_jsonl(args.tune_source_preds)
    tune_reranker = load_jsonl(args.tune_reranker_preds)
    eval_source = load_jsonl(args.eval_source_preds)
    eval_reranker = load_jsonl(args.eval_reranker_preds)
    assert_aligned(tune_source, tune_reranker, "tune_source", "tune_reranker")
    assert_aligned(eval_source, eval_reranker, "eval_source", "eval_reranker")

    tune_rows = sweep(tune_source, tune_reranker, args.source_weights, args.rrf_ks, args.ks)
    selected = select_best(tune_rows, args.primary_k, args.tie_k)
    eval_rankings = fuse_rows(eval_source, eval_reranker, selected["source_weight"], selected["rrf_k"])
    eval_selected = {
        "source_weight": selected["source_weight"],
        "rrf_k": selected["rrf_k"],
        **evaluate(eval_source, eval_rankings, args.ks),
    }

    tune_source_metrics = evaluate(tune_source, [row["pred_tools"] for row in tune_source], args.ks)
    eval_source_metrics = evaluate(eval_source, [row["pred_tools"] for row in eval_source], args.ks)
    for k in args.ks:
        eval_selected[f"delta_vs_source_at_{k}"] = eval_selected[f"allhit_at_{k}"] - eval_source_metrics[f"allhit_at_{k}"]

    payload = {
        "tune_source_preds": str(args.tune_source_preds),
        "tune_reranker_preds": str(args.tune_reranker_preds),
        "eval_source_preds": str(args.eval_source_preds),
        "eval_reranker_preds": str(args.eval_reranker_preds),
        "source_weights": args.source_weights,
        "rrf_ks": args.rrf_ks,
        "ks": args.ks,
        "primary_k": args.primary_k,
        "tie_k": args.tie_k,
        "selected": selected,
        "eval_selected": eval_selected,
        "tune_source_metrics": tune_source_metrics,
        "tune_reranker_metrics": evaluate(tune_source, [row["pred_tools"] for row in tune_reranker], args.ks),
        "eval_source_metrics": eval_source_metrics,
        "eval_reranker_metrics": evaluate(eval_source, [row["pred_tools"] for row in eval_reranker], args.ks),
        "tune_rows": tune_rows,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "reranker_rank_fusion_sweep.json", payload)
    write_markdown(args.out_dir / "reranker_rank_fusion_sweep.md", payload, args.primary_k)

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
        param = f"w{selected['source_weight']}".replace(".", "p") + f"_k{selected['rrf_k']}"
        write_jsonl(args.out_dir / f"preds_rank_fusion_{param}.jsonl", pred_rows)

    print(f"selected source_weight={selected['source_weight']} rrf_k={selected['rrf_k']}")
    print(f"tune AllHit@{args.primary_k}={selected[f'allhit_at_{args.primary_k}']:.6f}")
    print(f"eval AllHit@{args.primary_k}={eval_selected[f'allhit_at_{args.primary_k}']:.6f}")
    print(f"wrote {args.out_dir / 'reranker_rank_fusion_sweep.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
