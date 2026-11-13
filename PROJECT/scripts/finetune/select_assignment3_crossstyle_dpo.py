#!/usr/bin/env python3
"""Select a cross-style DPO checkpoint only on the protected development split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import read_json, write_json  # noqa: E402
from finetune.tool_trace_common import sha256_file  # noqa: E402


def parse_candidate(value: str) -> tuple[str, float, Path, Path]:
    parts = value.split("=", 3)
    if len(parts) != 4 or not all(parts):
        raise argparse.ArgumentTypeError(
            "--candidate must be NAME=BETA=ADAPTER=METRICS_JSON"
        )
    try:
        beta = float(parts[1])
    except ValueError as error:
        raise argparse.ArgumentTypeError("candidate beta must be numeric") from error
    return parts[0], beta, Path(parts[2]), Path(parts[3])


def parse_parent(value: str) -> tuple[str, Path, Path]:
    parts = value.split("=", 2)
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError(
            "--parent must be NAME=ADAPTER=METRICS_JSON"
        )
    return parts[0], Path(parts[1]), Path(parts[2])


def metric_row(
    name: str,
    adapter: Path,
    metrics_path: Path,
    expected_count: int,
    beta: float | None,
) -> dict[str, object]:
    if not adapter.is_dir() or not metrics_path.is_file():
        raise SystemExit(f"missing candidate artifact: {name}")
    metrics = read_json(metrics_path)
    if metrics.get("status") != "complete" or metrics.get("count") != expected_count:
        raise SystemExit(f"incomplete dev metrics: {name}")
    if metrics.get("view") != "rag":
        raise SystemExit(f"selection requires the fixed RAG dev view: {name}")
    overall = metrics["overall"]
    decomposition = metrics["rag_decomposition"]
    row: dict[str, object] = {
        "name": name,
        "adapter": str(adapter),
        "adapter_sha256": sha256_file(adapter / "adapter_model.safetensors"),
        "metrics": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
        "count": metrics["count"],
        "gold_snapshot_sha256": metrics["gold_snapshot_sha256"],
        "retriever_upper_bound": float(decomposition["retriever_upper_bound"]),
        "conditional_trace_em": float(
            decomposition["conditional_trace_accuracy"]
        ),
        "conditional_set_em": float(decomposition["conditional_set_accuracy"]),
        "trace_em": float(overall["trace_em"]),
        "set_em": float(overall["set_em"]),
        "strict_output_valid_rate": float(overall["strict_output_valid_rate"]),
    }
    if beta is not None:
        row["beta"] = beta
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        action="append",
        type=parse_candidate,
        required=True,
    )
    parser.add_argument("--parent", type=parse_parent, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=2400)
    parser.add_argument("--expected-retriever-upper-bound", type=float, default=0.95)
    parser.add_argument(
        "--selection-split",
        default="assignment3_api_native_dev_2400",
    )
    args = parser.parse_args()

    parent_name, parent_adapter, parent_metrics = args.parent
    parent = metric_row(
        parent_name,
        parent_adapter,
        parent_metrics,
        args.expected_count,
        None,
    )
    candidates = [
        metric_row(name, adapter, metrics, args.expected_count, beta)
        for name, beta, adapter, metrics in args.candidate
    ]
    rows = [parent] + candidates
    gold_hashes = {str(row["gold_snapshot_sha256"]) for row in rows}
    retrieval_bounds = {float(row["retriever_upper_bound"]) for row in rows}
    if len(gold_hashes) != 1 or "" in gold_hashes:
        raise SystemExit("metrics do not share one immutable development snapshot")
    if retrieval_bounds != {args.expected_retriever_upper_bound}:
        raise SystemExit(f"unexpected retriever upper bounds: {retrieval_bounds}")

    ranked = sorted(
        candidates,
        key=lambda row: (
            -float(row["trace_em"]),
            -float(row["set_em"]),
            -float(row["conditional_trace_em"]),
            -float(row["strict_output_valid_rate"]),
            float(row["beta"]),
            str(row["name"]),
        ),
    )
    selected = ranked[0]
    deltas = {
        metric: float(selected[metric]) - float(parent[metric])
        for metric in [
            "conditional_trace_em",
            "conditional_set_em",
            "trace_em",
            "set_em",
            "strict_output_valid_rate",
        ]
    }
    output = {
        "status": "complete",
        "selection_split": args.selection_split,
        "official_frozen_tests_used": False,
        "expected_count": args.expected_count,
        "retriever_upper_bound": args.expected_retriever_upper_bound,
        "ranking": [
            "overall TraceEM desc",
            "overall SetEM desc",
            "RAG-hit-conditional TraceEM desc",
            "strict output rate desc",
            "beta asc",
            "name asc",
        ],
        "parent": parent,
        "selected": selected,
        "selected_minus_parent": deltas,
        "ranked_dpo_candidates": ranked,
        "deployment_recommendation_guardrail": {
            "dpo_not_worse_than_parent_on_primary_dev_metric": deltas[
                "trace_em"
            ]
            >= 0,
            "note": (
                "The DPO endpoint is still evaluated for the assignment comparison; "
                "the parent SFT remains the deployment recommendation if this guardrail fails."
            ),
        },
    }
    write_json(args.out, output)
    print(
        f"selected={selected['name']} beta={selected['beta']} "
        f"trace_em={selected['trace_em']:.6f} "
        f"delta_parent={deltas['trace_em']:+.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
