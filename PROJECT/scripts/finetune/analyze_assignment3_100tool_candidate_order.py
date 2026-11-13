#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measure prediction stability under candidate-presentation reordering."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import load_jsonl, read_json, write_json  # noqa: E402
from finetune.tool_trace_common import sha256_file  # noqa: E402


def parse_variant(value: str) -> tuple[str, Path, Path]:
    parts = value.split("=", 2)
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError("--variant must be NAME=SCORED_JSONL=METRICS_JSON")
    return parts[0], Path(parts[1]), Path(parts[2])


def index(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(path)
    output = {str(row["id"]): row for row in rows}
    if len(output) != len(rows):
        raise ValueError(f"duplicate ids: {path}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-scored", type=Path, required=True)
    parser.add_argument("--original-metrics", type=Path, required=True)
    parser.add_argument("--variant", action="append", type=parse_variant, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    baseline = index(args.original_scored)
    baseline_metrics = read_json(args.original_metrics)
    if baseline_metrics.get("count") != len(baseline):
        raise SystemExit("baseline metrics/scored count mismatch")
    variants = {}
    input_hashes = {
        str(args.original_scored): sha256_file(args.original_scored),
        str(args.original_metrics): sha256_file(args.original_metrics),
    }
    for name, scored_path, metrics_path in args.variant:
        changed = index(scored_path)
        metrics = read_json(metrics_path)
        if set(changed) != set(baseline) or metrics.get("count") != len(changed):
            raise SystemExit(f"variant ids/count mismatch: {name}")
        candidate_set_equal = sum(
            set(baseline[row_id]["candidate_tools"])
            == set(changed[row_id]["candidate_tools"])
            for row_id in baseline
        )
        candidate_order_changed = sum(
            baseline[row_id]["candidate_tools"] != changed[row_id]["candidate_tools"]
            for row_id in baseline
        )
        baseline_trace_correct = [
            row_id for row_id, row in baseline.items() if row["trace_em"]
        ]
        variants[name] = {
            "metrics": {
                key: metrics["overall"][key]
                for key in (
                    "set_em",
                    "trace_em",
                    "strict_output_valid_rate",
                    "keyword_micro_f1",
                )
            },
            "candidate_set_equal_rate": candidate_set_equal / len(baseline),
            "candidate_order_changed_rate": candidate_order_changed / len(baseline),
            "predicted_trace_equal_rate": sum(
                baseline[row_id]["pred_tools"] == changed[row_id]["pred_tools"]
                for row_id in baseline
            )
            / len(baseline),
            "predicted_set_equal_rate": sum(
                set(baseline[row_id]["pred_tools"]) == set(changed[row_id]["pred_tools"])
                for row_id in baseline
            )
            / len(baseline),
            "trace_retention_from_original_correct": (
                sum(changed[row_id]["trace_em"] for row_id in baseline_trace_correct)
                / len(baseline_trace_correct)
                if baseline_trace_correct
                else None
            ),
            "trace_wins": sum(
                not baseline[row_id]["trace_em"] and changed[row_id]["trace_em"]
                for row_id in baseline
            ),
            "trace_losses": sum(
                baseline[row_id]["trace_em"] and not changed[row_id]["trace_em"]
                for row_id in baseline
            ),
        }
        if variants[name]["candidate_set_equal_rate"] != 1.0:
            raise SystemExit(f"candidate set changed in order-only variant: {name}")
        input_hashes[str(scored_path)] = sha256_file(scored_path)
        input_hashes[str(metrics_path)] = sha256_file(metrics_path)
    payload = {
        "status": "complete",
        "rows": len(baseline),
        "baseline_metrics": {
            key: baseline_metrics["overall"][key]
            for key in (
                "set_em",
                "trace_em",
                "strict_output_valid_rate",
                "keyword_micro_f1",
            )
        },
        "variants": variants,
        "inputs": input_hashes,
    }
    write_json(args.out, payload)
    print(f"candidate-order analysis rows={len(baseline)} variants={len(variants)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
