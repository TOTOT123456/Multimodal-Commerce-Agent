#!/usr/bin/env python3
"""Paired bootstrap and exact McNemar comparison for two frozen retrieval routes."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def success(row: dict, k: int) -> float:
    gold = set(row.get("gold_tools", []))
    return float(bool(gold) and gold <= set(row.get("pred_tools", [])[:k]))


def reciprocal_rank(row: dict) -> float:
    gold = set(row.get("gold_tools", []))
    return next(
        (1.0 / rank for rank, tool in enumerate(row.get("pred_tools", []), 1) if tool in gold),
        0.0,
    )


def exact_mcnemar(baseline: np.ndarray, challenger: np.ndarray) -> dict[str, Any]:
    baseline_only = int(np.sum((baseline == 1) & (challenger == 0)))
    challenger_only = int(np.sum((baseline == 0) & (challenger == 1)))
    discordant = baseline_only + challenger_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, value) for value in range(min(baseline_only, challenger_only) + 1))
        p_value = min(1.0, 2.0 * tail / (2**discordant))
    return {
        "baseline_only": baseline_only,
        "challenger_only": challenger_only,
        "discordant": discordant,
        "exact_two_sided_p": p_value,
    }


def bootstrap_difference(
    baseline: np.ndarray,
    challenger: np.ndarray,
    samples: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(baseline)
    differences = np.empty(samples, dtype=np.float64)
    chunk = 500
    for start in range(0, samples, chunk):
        count = min(chunk, samples - start)
        indices = rng.integers(0, n, size=(count, n), endpoint=False)
        differences[start : start + count] = (challenger[indices] - baseline[indices]).mean(axis=1)
    return {
        "point_difference": float((challenger - baseline).mean()),
        "ci95_low": float(np.quantile(differences, 0.025)),
        "ci95_high": float(np.quantile(differences, 0.975)),
        "bootstrap_samples": samples,
        "seed": seed,
    }


def compare_subset(
    baseline_rows: list[dict],
    challenger_rows: list[dict],
    indices: list[int],
    samples: int,
    seed: int,
) -> dict[str, Any]:
    result = {"count": len(indices), "metrics": {}}
    for offset, k in enumerate((6, 10, 20, 30, 40, 50)):
        baseline = np.asarray([success(baseline_rows[index], k) for index in indices])
        challenger = np.asarray([success(challenger_rows[index], k) for index in indices])
        item = {
            "baseline": float(baseline.mean()),
            "challenger": float(challenger.mean()),
            **bootstrap_difference(baseline, challenger, samples, seed + offset),
        }
        if k == 10:
            item["mcnemar"] = exact_mcnemar(baseline, challenger)
        result["metrics"][f"allhit_at_{k}"] = item
    baseline_mrr = np.asarray([reciprocal_rank(baseline_rows[index]) for index in indices])
    challenger_mrr = np.asarray([reciprocal_rank(challenger_rows[index]) for index in indices])
    result["metrics"]["mrr"] = {
        "baseline": float(baseline_mrr.mean()),
        "challenger": float(challenger_mrr.mean()),
        **bootstrap_difference(baseline_mrr, challenger_mrr, samples, seed + 100),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--challenger", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--noninferiority-margin", type=float, default=-0.002)
    args = parser.parse_args()

    baseline = load_jsonl(args.baseline)
    challenger = load_jsonl(args.challenger)
    dataset = load_jsonl(args.dataset)
    if not (len(baseline) == len(challenger) == len(dataset)):
        raise ValueError("row count mismatch")
    for index, (left, right, meta) in enumerate(zip(baseline, challenger, dataset)):
        ids = {str(left.get("id", "")), str(right.get("id", "")), str(meta.get("id", ""))}
        if len(ids) != 1 or left.get("query") != right.get("query"):
            raise ValueError(f"row alignment failure at {index}")

    all_indices = list(range(len(dataset)))
    by_style: dict[str, list[int]] = defaultdict(list)
    by_count: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(dataset):
        by_style[str(row.get("style", "unknown"))].append(index)
        by_count[str(row.get("tool_count", "unknown"))].append(index)
    overall = compare_subset(
        baseline, challenger, all_indices, args.bootstrap_samples, args.seed
    )
    primary = overall["metrics"]["allhit_at_10"]
    promotion = (
        primary["point_difference"] >= 0.0
        and primary["ci95_low"] >= args.noninferiority_margin
    )
    payload = {
        "baseline": str(args.baseline),
        "challenger": str(args.challenger),
        "dataset": str(args.dataset),
        "overall": overall,
        "by_style": {
            key: compare_subset(baseline, challenger, indices, args.bootstrap_samples, args.seed + 1000 + offset)
            for offset, (key, indices) in enumerate(sorted(by_style.items()))
        },
        "by_tool_count": {
            key: compare_subset(baseline, challenger, indices, args.bootstrap_samples, args.seed + 2000 + offset)
            for offset, (key, indices) in enumerate(sorted(by_count.items()))
        },
        "promotion_gate": {
            "primary": "allhit_at_10",
            "point_difference_must_be_nonnegative": True,
            "ci95_low_min": args.noninferiority_margin,
            "passed": promotion,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"AllHit@10 delta={primary['point_difference']:+.6f} "
        f"CI=[{primary['ci95_low']:+.6f},{primary['ci95_high']:+.6f}] promotion={promotion}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
