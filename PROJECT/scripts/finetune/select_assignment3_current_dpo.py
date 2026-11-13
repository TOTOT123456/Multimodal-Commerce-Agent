#!/usr/bin/env python3
"""Select and confirm current-route DPO endpoints on the train-side dev set."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


STYLES = ("legacy_compact", "api_native")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "complete" or int(value.get("count", -1)) != 1400:
        raise ValueError(f"incomplete DPO development metrics: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def snapshot(rag_path: Path, global_path: Path) -> dict[str, Any]:
    rag = read_json(rag_path)
    global_metrics = read_json(global_path)
    if rag["gold_snapshot_sha256"] != global_metrics["gold_snapshot_sha256"]:
        raise ValueError("RAG/global metrics use different gold snapshots")
    tool_counts = rag["buckets"]["tool_count"]
    styles = rag["buckets"]["style"]
    if not set(STYLES) <= set(styles) or not all(str(value) in tool_counts for value in range(3, 7)):
        raise ValueError("missing style or long-chain development buckets")
    return {
        "rag_metrics": str(rag_path),
        "global_metrics": str(global_path),
        "gold_snapshot_sha256": rag["gold_snapshot_sha256"],
        "rag_trace_em": float(rag["overall"]["trace_em"]),
        "rag_set_em": float(rag["overall"]["set_em"]),
        "strict_output_valid_rate": float(rag["overall"]["strict_output_valid_rate"]),
        "retriever_upper_bound": float(rag["rag_decomposition"]["retriever_upper_bound"]),
        "global_trace_em": float(global_metrics["overall"]["trace_em"]),
        "style_trace_em": {style: float(styles[style]["trace_em"]) for style in STYLES},
        "tool_count_trace_em": {
            str(value): float(tool_counts[str(value)]["trace_em"])
            for value in range(1, 7)
        },
        "long_chain_worst_trace_em": min(
            float(tool_counts[str(value)]["trace_em"]) for value in range(3, 7)
        ),
    }


def parse_grid(value: str) -> tuple[str, float, float, float, Path, Path, Path]:
    parts = value.split("=", 6)
    if len(parts) != 7 or not all(parts):
        raise argparse.ArgumentTypeError(
            "grid candidate must be NAME=LR=BETA=EPOCH=ADAPTER=RAG_METRICS=GLOBAL_METRICS"
        )
    name, lr, beta, epoch, adapter, rag, global_metrics = parts
    return name, float(lr), float(beta), float(epoch), Path(adapter), Path(rag), Path(global_metrics)


def parse_seed(value: str) -> tuple[int, Path, Path]:
    parts = value.split("=", 2)
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError("seed result must be SEED=RAG_METRICS=GLOBAL_METRICS")
    return int(parts[0]), Path(parts[1]), Path(parts[2])


def deltas(row: dict[str, Any], parent: dict[str, Any]) -> dict[str, Any]:
    return {
        "rag_trace_em": row["rag_trace_em"] - parent["rag_trace_em"],
        "rag_set_em": row["rag_set_em"] - parent["rag_set_em"],
        "strict_output_valid_rate": (
            row["strict_output_valid_rate"] - parent["strict_output_valid_rate"]
        ),
        "global_trace_em": row["global_trace_em"] - parent["global_trace_em"],
        "long_chain_worst_trace_em": (
            row["long_chain_worst_trace_em"] - parent["long_chain_worst_trace_em"]
        ),
        "style_trace_em": {
            style: row["style_trace_em"][style] - parent["style_trace_em"][style]
            for style in STYLES
        },
    }


def guard(row: dict[str, Any], parent: dict[str, Any]) -> dict[str, Any]:
    delta = deltas(row, parent)
    checks = {
        "strict_delta_ge_minus_0p2pp": delta["strict_output_valid_rate"] >= -0.002,
        "global_trace_delta_ge_minus_0p5pp": delta["global_trace_em"] >= -0.005,
        "long_chain_worst_delta_ge_minus_1pp": delta["long_chain_worst_trace_em"] >= -0.01,
        **{
            f"{style}_trace_delta_ge_minus_0p5pp": delta["style_trace_em"][style] >= -0.005
            for style in STYLES
        },
    }
    return {"pass": all(checks.values()), "checks": checks, "delta": delta}


def grid(args: argparse.Namespace) -> int:
    parent = snapshot(args.parent_rag, args.parent_global)
    candidates = []
    for name, lr, beta, epoch, adapter, rag, global_metrics in args.candidate:
        if not adapter.is_dir():
            raise ValueError(f"missing adapter: {adapter}")
        row = {
            "name": name,
            "learning_rate": lr,
            "beta": beta,
            "epoch": epoch,
            "adapter": str(adapter),
            **snapshot(rag, global_metrics),
        }
        row["guard"] = guard(row, parent)
        candidates.append(row)
    hashes = {parent["gold_snapshot_sha256"]} | {
        row["gold_snapshot_sha256"] for row in candidates
    }
    if len(hashes) != 1:
        raise ValueError("grid endpoints use different development gold snapshots")
    ranked = sorted(
        candidates,
        key=lambda row: (
            not row["guard"]["pass"],
            -row["rag_trace_em"],
            -row["rag_set_em"],
            -row["long_chain_worst_trace_em"],
            -row["global_trace_em"],
            row["learning_rate"],
            row["beta"],
            row["epoch"],
            row["name"],
        ),
    )
    selected = ranked[0]
    output = {
        "schema_version": 1,
        "status": "complete",
        "stage": "single_seed_grid",
        "model_size": args.model_size,
        "official_frozen_sets_used": False,
        "parent_sft": parent,
        "selected": selected,
        "ranked_candidates": ranked,
        "selection_rule": [
            "guard pass before fail",
            "RAG TraceEM desc",
            "RAG SetEM desc",
            "worst tool-count 3-6 TraceEM desc",
            "Wo-RAG TraceEM desc",
            "learning rate, beta, epoch asc",
        ],
        "advance_to_multiseed": selected["guard"]["pass"],
    }
    write_json(args.out, output)
    print(
        f"selected {selected['name']} trace={selected['rag_trace_em']:.6f} "
        f"delta={selected['guard']['delta']['rag_trace_em']:+.6f} "
        f"guard={selected['guard']['pass']}"
    )
    return 0


def confirm(args: argparse.Namespace) -> int:
    parent = snapshot(args.parent_rag, args.parent_global)
    seeds = []
    for seed, rag, global_metrics in args.seed_result:
        row = {"seed": seed, **snapshot(rag, global_metrics)}
        row["guard"] = guard(row, parent)
        seeds.append(row)
    if len(seeds) != 3 or len({row["seed"] for row in seeds}) != 3:
        raise ValueError("confirmation requires exactly three distinct seeds")
    mean_trace = sum(row["rag_trace_em"] for row in seeds) / len(seeds)
    mean_set = sum(row["rag_set_em"] for row in seeds) / len(seeds)
    worst_trace = min(row["rag_trace_em"] for row in seeds)
    checks = {
        "mean_rag_trace_above_parent": mean_trace > parent["rag_trace_em"],
        "worst_seed_rag_trace_delta_ge_minus_0p2pp": (
            worst_trace - parent["rag_trace_em"] >= -0.002
        ),
        "all_seed_guards_pass": all(row["guard"]["pass"] for row in seeds),
    }
    output = {
        "schema_version": 1,
        "status": "complete",
        "stage": "three_seed_confirmation",
        "model_size": args.model_size,
        "official_frozen_sets_used": False,
        "parent_sft": parent,
        "seeds": sorted(seeds, key=lambda row: row["seed"]),
        "aggregate": {
            "mean_rag_trace_em": mean_trace,
            "mean_rag_set_em": mean_set,
            "worst_seed_rag_trace_em": worst_trace,
            "population_std_rag_trace_em": math.sqrt(
                sum((row["rag_trace_em"] - mean_trace) ** 2 for row in seeds)
                / len(seeds)
            ),
        },
        "promotion_checks": checks,
        "promotion_pass": all(checks.values()),
        "deployment_recommendation": "sft_dpo" if all(checks.values()) else "parent_sft",
    }
    write_json(args.out, output)
    print(
        f"three-seed mean={mean_trace:.6f} worst={worst_trace:.6f} "
        f"promotion={output['promotion_pass']}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["grid", "confirm"], required=True)
    parser.add_argument("--model-size", choices=["4b", "8b"], required=True)
    parser.add_argument("--parent-rag", type=Path, required=True)
    parser.add_argument("--parent-global", type=Path, required=True)
    parser.add_argument("--candidate", action="append", type=parse_grid, default=[])
    parser.add_argument("--seed-result", action="append", type=parse_seed, default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "grid":
        if not args.candidate:
            parser.error("grid mode requires --candidate")
        return grid(args)
    if not args.seed_result:
        parser.error("confirm mode requires --seed-result")
    return confirm(args)


if __name__ == "__main__":
    raise SystemExit(main())
