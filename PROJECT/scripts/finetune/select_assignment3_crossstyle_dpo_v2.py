#!/usr/bin/env python3
"""Select DPO-v2 on sealed dev robustness metrics with SFT guardrails."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ORDERS = ("original", "reverse", "hash_shuffle")
STYLES = ("api_native", "legacy_compact")


def parse_candidate(value: str) -> tuple[str, float, float, Path, dict[str, Path]]:
    parts = value.split("=", 7)
    if len(parts) != 8 or not all(parts):
        raise argparse.ArgumentTypeError(
            "candidate must be NAME=BETA=LR=ADAPTER=ORIGINAL=REVERSE=HASH=LABEL"
        )
    name, beta, learning_rate, adapter, original, reverse, hashed, label = parts
    if label != "sealed_dev_v2":
        raise argparse.ArgumentTypeError("candidate label must be sealed_dev_v2")
    return (
        name,
        float(beta),
        float(learning_rate),
        Path(adapter),
        {
            "original": Path(original),
            "reverse": Path(reverse),
            "hash_shuffle": Path(hashed),
        },
    )


def parse_parent(value: str) -> tuple[str, Path, dict[str, Path]]:
    parts = value.split("=", 5)
    if len(parts) != 6 or not all(parts):
        raise argparse.ArgumentTypeError(
            "parent must be NAME=ADAPTER=ORIGINAL=REVERSE=HASH=LABEL"
        )
    name, adapter, original, reverse, hashed, label = parts
    if label != "sealed_dev_v2":
        raise argparse.ArgumentTypeError("parent label must be sealed_dev_v2")
    return (
        name,
        Path(adapter),
        {
            "original": Path(original),
            "reverse": Path(reverse),
            "hash_shuffle": Path(hashed),
        },
    )


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def summarize_metrics(path: Path, expected_count: int) -> dict[str, Any]:
    metrics = read_json(path)
    if (
        metrics.get("status") != "complete"
        or metrics.get("view") != "rag"
        or int(metrics.get("count", -1)) != expected_count
    ):
        raise ValueError(f"incomplete sealed dev metrics: {path}")
    overall = metrics["overall"]
    buckets = metrics["buckets"]
    style = buckets.get("style", {})
    if set(style) != set(STYLES):
        raise ValueError(f"missing style buckets: {path}")
    return {
        "path": str(path),
        "gold_snapshot_sha256": metrics["gold_snapshot_sha256"],
        "retriever_upper_bound": float(
            metrics["rag_decomposition"]["retriever_upper_bound"]
        ),
        "set_em": float(overall["set_em"]),
        "trace_em": float(overall["trace_em"]),
        "strict_output_valid_rate": float(
            overall["strict_output_valid_rate"]
        ),
        "conditional_set_em": float(
            metrics["rag_decomposition"]["conditional_set_accuracy"]
        ),
        "conditional_trace_em": float(
            metrics["rag_decomposition"]["conditional_trace_accuracy"]
        ),
        "multi_trace_em": float(buckets["single_multi"]["multi"]["trace_em"]),
        "style_trace_em": {
            key: float(style[key]["trace_em"]) for key in STYLES
        },
        "style_set_em": {
            key: float(style[key]["set_em"]) for key in STYLES
        },
    }


def endpoint(
    name: str,
    adapter: Path,
    metrics: dict[str, Path],
    expected_count: int,
    beta: float | None,
    learning_rate: float | None,
) -> dict[str, Any]:
    if not adapter.is_dir():
        raise ValueError(f"missing adapter: {adapter}")
    rows = {
        order: summarize_metrics(path, expected_count)
        for order, path in metrics.items()
    }
    result: dict[str, Any] = {
        "name": name,
        "adapter": str(adapter),
        "orders": rows,
        "robustness_macro": {
            metric: sum(rows[order][metric] for order in ORDERS) / len(ORDERS)
            for metric in (
                "set_em",
                "trace_em",
                "strict_output_valid_rate",
                "conditional_set_em",
                "conditional_trace_em",
            )
        },
        "worst_order_trace_em": min(
            rows[order]["trace_em"] for order in ORDERS
        ),
    }
    if beta is not None:
        result["beta"] = beta
    if learning_rate is not None:
        result["learning_rate"] = learning_rate
    return result


def metric_delta(candidate: dict[str, Any], parent: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "robustness_macro_trace_em": (
            candidate["robustness_macro"]["trace_em"]
            - parent["robustness_macro"]["trace_em"]
        ),
        "worst_order_trace_em": (
            candidate["worst_order_trace_em"] - parent["worst_order_trace_em"]
        ),
        "orders": {},
    }
    for order in ORDERS:
        current = candidate["orders"][order]
        baseline = parent["orders"][order]
        result["orders"][order] = {
            key: current[key] - baseline[key]
            for key in (
                "set_em",
                "trace_em",
                "strict_output_valid_rate",
                "conditional_set_em",
                "conditional_trace_em",
                "multi_trace_em",
            )
        }
        result["orders"][order]["style_trace_em"] = {
            style: current["style_trace_em"][style]
            - baseline["style_trace_em"][style]
            for style in STYLES
        }
    return result


def guardrails(delta: dict[str, Any]) -> dict[str, Any]:
    original = delta["orders"]["original"]
    checks = {
        "original_overall_trace_delta_ge_minus_0p2pp": original["trace_em"]
        >= -0.002,
        "original_multi_trace_delta_ge_minus_0p5pp": original["multi_trace_em"]
        >= -0.005,
        "original_strict_output_delta_ge_minus_0p2pp": original[
            "strict_output_valid_rate"
        ]
        >= -0.002,
        "original_api_trace_delta_ge_minus_0p5pp": original[
            "style_trace_em"
        ]["api_native"]
        >= -0.005,
        "original_legacy_trace_delta_ge_minus_0p5pp": original[
            "style_trace_em"
        ]["legacy_compact"]
        >= -0.005,
    }
    return {"pass": all(checks.values()), "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", type=parse_candidate, required=True)
    parser.add_argument("--parent", type=parse_parent, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=1200)
    args = parser.parse_args()

    parent_name, parent_adapter, parent_metrics = args.parent
    parent = endpoint(
        parent_name,
        parent_adapter,
        parent_metrics,
        args.expected_count,
        None,
        None,
    )
    candidates = []
    for name, beta, learning_rate, adapter, metrics in args.candidate:
        row = endpoint(
            name,
            adapter,
            metrics,
            args.expected_count,
            beta,
            learning_rate,
        )
        row["minus_parent"] = metric_delta(row, parent)
        row["guardrails"] = guardrails(row["minus_parent"])
        candidates.append(row)

    all_endpoints = [parent] + candidates
    gold_hashes = {
        row["orders"][order]["gold_snapshot_sha256"]
        for row in all_endpoints
        for order in ORDERS
    }
    retrieval_bounds = {
        row["orders"][order]["retriever_upper_bound"]
        for row in all_endpoints
        for order in ORDERS
    }
    if len(gold_hashes) != 1 or len(retrieval_bounds) != 1:
        raise ValueError("endpoints do not share one sealed dev/retrieval snapshot")

    ranked = sorted(
        candidates,
        key=lambda row: (
            not row["guardrails"]["pass"],
            -row["robustness_macro"]["trace_em"],
            -row["worst_order_trace_em"],
            -row["orders"]["original"]["trace_em"],
            -row["robustness_macro"]["strict_output_valid_rate"],
            row["learning_rate"],
            row["beta"],
            row["name"],
        ),
    )
    selected = ranked[0]
    output = {
        "selection_version": "assignment3-crossstyle-dpo-v2-selection-v1",
        "status": "complete",
        "selection_split": "sealed_crossstyle_dev_v2",
        "expected_count_per_order": args.expected_count,
        "candidate_orders": list(ORDERS),
        "styles": list(STYLES),
        "retriever_upper_bound": next(iter(retrieval_bounds)),
        "official_original_frozen_sets_used": False,
        "sealed_final_set_used": False,
        "ranking": [
            "guardrail pass before fail",
            "three-order macro TraceEM desc",
            "worst-order TraceEM desc",
            "original-order TraceEM desc",
            "three-order macro strict-output rate desc",
            "learning rate asc, beta asc, name asc",
        ],
        "guardrail_thresholds": {
            "original_overall_trace_delta_min": -0.002,
            "original_multi_trace_delta_min": -0.005,
            "original_each_style_trace_delta_min": -0.005,
            "original_strict_output_delta_min": -0.002,
        },
        "parent_sft": parent,
        "selected_dpo": selected,
        "ranked_dpo_candidates": ranked,
        "deployment_recommendation": (
            "selected_dpo"
            if selected["guardrails"]["pass"]
            else "parent_sft"
        ),
    }
    write_json(args.out, output)
    print(
        f"selected={selected['name']} guardrail={selected['guardrails']['pass']} "
        f"macro_trace={selected['robustness_macro']['trace_em']:.6f} "
        f"delta_parent={selected['minus_parent']['robustness_macro_trace_em']:+.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
