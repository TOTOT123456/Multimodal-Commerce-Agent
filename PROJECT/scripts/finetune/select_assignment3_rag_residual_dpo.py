#!/usr/bin/env python3
"""Select and confirm residual DPO endpoints on isolated RAG-only panels."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_STYLES = ("legacy_compact", "api_native")
TOOL_COUNTS = tuple(str(value) for value in range(1, 7))


def read_json(path: Path, expected_count: int) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "complete" or int(value.get("count", -1)) != expected_count:
        raise ValueError(f"incomplete metrics or unexpected count: {path}")
    return value


def read_scored(path: Path, expected_count: int) -> dict[str, bool]:
    rows: dict[str, bool] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = str(row["id"])
            if row_id in rows:
                raise ValueError(f"duplicate scored id {row_id}: {path}")
            rows[row_id] = bool(row["trace_em"])
    if len(rows) != expected_count:
        raise ValueError(f"unexpected scored count {len(rows)}: {path}")
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def snapshot(
    metrics_path: Path,
    scored_path: Path,
    expected_count: int,
    styles_required: tuple[str, ...],
) -> dict[str, Any]:
    metrics = read_json(metrics_path, expected_count)
    scored = read_scored(scored_path, expected_count)
    styles = metrics["buckets"]["style"]
    tool_counts = metrics["buckets"]["tool_count"]
    if not set(styles_required) <= set(styles) or not set(TOOL_COUNTS) <= set(tool_counts):
        raise ValueError(f"missing style or tool-count buckets: {metrics_path}")
    return {
        "metrics": str(metrics_path),
        "scored": str(scored_path),
        "gold_snapshot_sha256": str(metrics["gold_snapshot_sha256"]),
        "trace_em": float(metrics["overall"]["trace_em"]),
        "set_em": float(metrics["overall"]["set_em"]),
        "strict_output_valid_rate": float(metrics["overall"]["strict_output_valid_rate"]),
        "style_trace_em": {
            style: float(styles[style]["trace_em"]) for style in styles_required
        },
        "tool_count_trace_em": {
            value: float(tool_counts[value]["trace_em"]) for value in TOOL_COUNTS
        },
        "long_chain_macro_trace_em": sum(
            float(tool_counts[str(value)]["trace_em"]) for value in range(3, 7)
        ) / 4.0,
        "scored_rows": scored,
    }


def paired_test(candidate: dict[str, bool], parent: dict[str, bool]) -> dict[str, Any]:
    if set(candidate) != set(parent):
        raise ValueError("candidate and parent scored IDs differ")
    gains = sum(candidate[key] and not parent[key] for key in parent)
    losses = sum(parent[key] and not candidate[key] for key in parent)
    discordant = gains + losses
    if discordant == 0:
        p_value = 1.0
    else:
        lower = min(gains, losses)
        tail = sum(math.comb(discordant, index) for index in range(lower + 1))
        p_value = min(1.0, 2.0 * tail / (2**discordant))
    return {
        "candidate_gain_parent_wrong": gains,
        "candidate_loss_parent_right": losses,
        "net_gain": gains - losses,
        "discordant": discordant,
        "mcnemar_exact_two_sided_p": p_value,
    }


def public_snapshot(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "scored_rows"}


def guard(
    row: dict[str, Any], parent: dict[str, Any], styles_required: tuple[str, ...]
) -> dict[str, Any]:
    deltas = {
        "trace_em": row["trace_em"] - parent["trace_em"],
        "set_em": row["set_em"] - parent["set_em"],
        "strict_output_valid_rate": (
            row["strict_output_valid_rate"] - parent["strict_output_valid_rate"]
        ),
        "style_trace_em": {
            style: row["style_trace_em"][style] - parent["style_trace_em"][style]
            for style in styles_required
        },
        "tool_count_trace_em": {
            value: row["tool_count_trace_em"][value]
            - parent["tool_count_trace_em"][value]
            for value in TOOL_COUNTS
        },
    }
    checks = {
        "trace_em_above_parent": deltas["trace_em"] > 0,
        "strict_output_delta_ge_minus_0p2pp": deltas["strict_output_valid_rate"] >= -0.002,
        **{
            f"style_{style}_delta_ge_minus_0p5pp": deltas["style_trace_em"][style] >= -0.005
            for style in styles_required
        },
        **{
            f"tool_count_{value}_delta_ge_minus_1pp": deltas["tool_count_trace_em"][value] >= -0.01
            for value in TOOL_COUNTS
        },
    }
    return {"pass": all(checks.values()), "checks": checks, "delta": deltas}


def parse_grid(value: str) -> tuple[str, float, float, float, Path, Path, Path]:
    parts = value.split("=", 6)
    if len(parts) != 7 or not all(parts):
        raise argparse.ArgumentTypeError(
            "candidate must be NAME=LR=BETA=EPOCH=ADAPTER=METRICS=SCORED"
        )
    name, lr, beta, epoch, adapter, metrics, scored = parts
    return name, float(lr), float(beta), float(epoch), Path(adapter), Path(metrics), Path(scored)


def parse_seed(value: str) -> tuple[int, Path, Path]:
    parts = value.split("=", 2)
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError("seed result must be SEED=METRICS=SCORED")
    return int(parts[0]), Path(parts[1]), Path(parts[2])


def run_grid(args: argparse.Namespace) -> int:
    styles_required = tuple(args.styles)
    parent = snapshot(
        args.parent_metrics, args.parent_scored, args.expected_count, styles_required
    )
    candidates = []
    for name, lr, beta, epoch, adapter, metrics, scored in args.candidate:
        if not adapter.is_dir():
            raise ValueError(f"missing adapter: {adapter}")
        internal = snapshot(metrics, scored, args.expected_count, styles_required)
        if internal["gold_snapshot_sha256"] != parent["gold_snapshot_sha256"]:
            raise ValueError(f"gold snapshot differs: {metrics}")
        row = {
            "name": name,
            "learning_rate": lr,
            "beta": beta,
            "epoch": epoch,
            "adapter": str(adapter),
            **public_snapshot(internal),
            "paired_vs_parent": paired_test(internal["scored_rows"], parent["scored_rows"]),
        }
        row["guard"] = guard(row, parent, styles_required)
        candidates.append(row)
    ranked = sorted(
        candidates,
        key=lambda row: (
            not row["guard"]["pass"],
            -row["trace_em"],
            -row["set_em"],
            -row["long_chain_macro_trace_em"],
            row["learning_rate"],
            row["beta"],
            row["epoch"],
            row["name"],
        ),
    )
    output = {
        "schema_version": 1,
        "status": "complete",
        "stage": "rag_only_single_seed_grid",
        "official_frozen_sets_used": False,
        "expected_count": args.expected_count,
        "styles_required": list(styles_required),
        "parent": public_snapshot(parent),
        "selected": ranked[0],
        "advance_to_confirmation": ranked[0]["guard"]["pass"],
        "selection_rule": [
            "all RAG-only guards pass before fail",
            "TraceEM descending",
            "SetEM descending",
            "tool-count 3-6 macro TraceEM descending",
            "learning rate, beta, epoch ascending",
        ],
        "ranked_candidates": ranked,
    }
    write_json(args.out, output)
    selected = ranked[0]
    print(
        f"selected={selected['name']} trace_em={selected['trace_em']:.6f} "
        f"delta={selected['guard']['delta']['trace_em']:+.6f} "
        f"guard={selected['guard']['pass']}"
    )
    return 0


def run_confirm(args: argparse.Namespace) -> int:
    styles_required = tuple(args.styles)
    parent = snapshot(
        args.parent_metrics, args.parent_scored, args.expected_count, styles_required
    )
    seeds = []
    for seed, metrics, scored in args.seed_result:
        internal = snapshot(metrics, scored, args.expected_count, styles_required)
        if internal["gold_snapshot_sha256"] != parent["gold_snapshot_sha256"]:
            raise ValueError(f"gold snapshot differs: {metrics}")
        row = {
            "seed": seed,
            **public_snapshot(internal),
            "paired_vs_parent": paired_test(internal["scored_rows"], parent["scored_rows"]),
        }
        row["guard"] = guard(row, parent, styles_required)
        seeds.append(row)
    if len(seeds) != 3 or len({row["seed"] for row in seeds}) != 3:
        raise ValueError("confirmation requires exactly three distinct seeds")
    mean_trace = sum(row["trace_em"] for row in seeds) / 3
    worst_trace = min(row["trace_em"] for row in seeds)
    checks = {
        "mean_trace_above_parent": mean_trace > parent["trace_em"],
        "worst_seed_delta_ge_minus_0p2pp": worst_trace - parent["trace_em"] >= -0.002,
        "all_rag_only_guards_pass": all(row["guard"]["pass"] for row in seeds),
    }
    output = {
        "schema_version": 1,
        "status": "complete",
        "stage": "rag_only_three_seed_confirmation",
        "official_frozen_sets_used": False,
        "expected_count": args.expected_count,
        "styles_required": list(styles_required),
        "parent": public_snapshot(parent),
        "seeds": sorted(seeds, key=lambda row: row["seed"]),
        "aggregate": {
            "mean_trace_em": mean_trace,
            "worst_seed_trace_em": worst_trace,
            "population_std_trace_em": math.sqrt(
                sum((row["trace_em"] - mean_trace) ** 2 for row in seeds) / 3
            ),
        },
        "promotion_checks": checks,
        "promotion_pass": all(checks.values()),
    }
    write_json(args.out, output)
    print(
        f"three_seed_mean={mean_trace:.6f} worst={worst_trace:.6f} "
        f"promotion={output['promotion_pass']}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("grid", "confirm"), required=True)
    parser.add_argument("--parent-metrics", type=Path, required=True)
    parser.add_argument("--parent-scored", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument(
        "--styles", nargs="+", default=list(DEFAULT_STYLES), choices=DEFAULT_STYLES
    )
    parser.add_argument("--candidate", action="append", type=parse_grid, default=[])
    parser.add_argument("--seed-result", action="append", type=parse_seed, default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "grid":
        if not args.candidate:
            parser.error("grid mode requires at least one --candidate")
        return run_grid(args)
    if not args.seed_result:
        parser.error("confirm mode requires --seed-result")
    return run_confirm(args)


if __name__ == "__main__":
    raise SystemExit(main())
