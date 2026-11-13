#!/usr/bin/env python3
"""Reaggregate the frozen 9K system with the canonical Intent endpoint."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parent
ROUTE_CONFIG = PROJECT_ROOT / "configs/retrieval_canonical_route.json"
SYSTEM_EVALUATOR = PROJECT_ROOT / "scripts/finetune/evaluate_tool_trace_system.py"
PREDICTION_ROOT = (
    PROJECT_ROOT
    / "finetune/outputs/preds/tool_trace"
    / "assignment3_current_api9k_dpo_20260802/protected"
)
HISTORICAL_METRIC_ROOT = (
    PROJECT_ROOT
    / "data/tool_call_trace/_aux/assignment3"
    / "assignment3_current_api9k_dpo_20260802/evaluation/protected"
)
CELLS = {
    "4b_sft_scale_selected": (
        PROJECT_ROOT
        / "finetune/outputs/preds/tool_trace/assignment3_api_native_sft_scaling_20260802"
        / "legacy6000/a3-api-native-scale-4b-n9000/rag.jsonl",
        PROJECT_ROOT
        / "data/tool_call_trace/_aux/assignment3/assignment3_api_native_sft_scaling_20260802"
        / "evaluation/system9000/a3-api-native-scale-4b-n9000.system.metrics.json",
    ),
    "4b_sft": (
        PREDICTION_ROOT / "4b/parent-controlled/legacy6000/rag.jsonl",
        HISTORICAL_METRIC_ROOT
        / "4b/parent-controlled/legacy6000/rag.system.metrics.json",
    ),
    "4b_sft_dpo": (
        PREDICTION_ROOT
        / "4b/lr1e6-b005-e10-s20260802/legacy6000/rag.jsonl",
        HISTORICAL_METRIC_ROOT
        / "4b/lr1e6-b005-e10-s20260802/legacy6000/rag.system.metrics.json",
    ),
    "8b_sft": (
        PREDICTION_ROOT / "8b/parent-controlled/legacy6000/rag.jsonl",
        HISTORICAL_METRIC_ROOT
        / "8b/parent-controlled/legacy6000/rag.system.metrics.json",
    ),
    "8b_sft_scale_selected": (
        PROJECT_ROOT
        / "finetune/outputs/preds/tool_trace/assignment3_api_native_sft_scaling_20260802"
        / "legacy6000/a3-api-native-scale-8b-n9000/rag.jsonl",
        PROJECT_ROOT
        / "data/tool_call_trace/_aux/assignment3/assignment3_api_native_sft_scaling_20260802"
        / "evaluation/system9000/a3-api-native-scale-8b-n9000.system.metrics.json",
    ),
    "8b_sft_dpo": (
        PREDICTION_ROOT
        / "8b/lr1e6-b010-e10-s20260802/legacy6000/rag.jsonl",
        HISTORICAL_METRIC_ROOT
        / "8b/lr1e6-b010-e10-s20260802/legacy6000/rag.system.metrics.json",
    ),
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT.resolve()))


def main() -> int:
    route = read_json(ROUTE_CONFIG)
    intent = route.get("intent", {})
    intent_predictions = resolve(str(intent.get("predictions", "")))
    output_root = resolve(
        str(intent.get("system_evaluation", {}).get("artifact_root", ""))
    )
    if route.get("status") != "canonical" or intent.get("model") != "Qwen3-0.6B":
        raise SystemExit("canonical route does not select Qwen3-0.6B Intent")
    if not intent_predictions.is_file():
        raise SystemExit(f"missing canonical Intent predictions: {intent_predictions}")
    try:
        output_root.resolve().relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        raise SystemExit(f"artifact root must stay under PROJECT: {output_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    cell_summaries: dict[str, Any] = {}
    for cell, (tool_predictions, historical_metrics_path) in CELLS.items():
        if not tool_predictions.is_file() or not historical_metrics_path.is_file():
            raise SystemExit(f"missing input for {cell}")
        system_predictions = output_root / f"{cell}.system.jsonl"
        system_metrics = output_root / f"{cell}.system.metrics.json"
        subprocess.run(
            [
                sys.executable,
                str(SYSTEM_EVALUATOR),
                "--intent-predictions",
                str(intent_predictions),
                "--tool-predictions",
                str(tool_predictions),
                "--out",
                str(system_predictions),
                "--metrics-out",
                str(system_metrics),
            ],
            cwd=REPO_ROOT,
            check=True,
        )
        current = read_json(system_metrics)
        historical = read_json(historical_metrics_path)
        if current.get("count") != 9000 or current.get("route_correct_count") != 9000:
            raise SystemExit(f"unexpected canonical routing result for {cell}")
        cell_summaries[cell] = {
            "tool_predictions": relative(tool_predictions),
            "tool_predictions_sha256": sha256_file(tool_predictions),
            "system_predictions": relative(system_predictions),
            "system_predictions_sha256": sha256_file(system_predictions),
            "system_metrics": relative(system_metrics),
            "system_metrics_sha256": sha256_file(system_metrics),
            "full_set_count": int(current["full_pipeline_success_count"]),
            "full_set": float(current["full_pipeline_success"]),
            "full_trace_count": int(current["full_pipeline_trace_success_count"]),
            "full_trace": float(current["full_pipeline_trace_success"]),
            "previous_1p7b_metrics": relative(historical_metrics_path),
            "previous_1p7b_full_set_count": int(
                historical["full_pipeline_success_count"]
            ),
            "previous_1p7b_full_trace_count": int(
                historical["full_pipeline_trace_success_count"]
            ),
            "full_set_count_gain": int(current["full_pipeline_success_count"])
            - int(historical["full_pipeline_success_count"]),
            "full_trace_count_gain": int(
                current["full_pipeline_trace_success_count"]
            )
            - int(historical["full_pipeline_trace_success_count"]),
        }

    if any(
        cell[metric] != 4
        for cell in cell_summaries.values()
        for metric in ("full_set_count_gain", "full_trace_count_gain")
    ):
        raise SystemExit("canonical Intent switch did not produce the expected +4 gain")

    summary = {
        "schema_version": 1,
        "status": "canonical",
        "canonical_revision": route["canonical_revision"],
        "system_revision": route.get("system_revision", route["canonical_revision"]),
        "scope": "static routing plus frozen Planner outputs; not real tool execution",
        "intent": {
            "model": intent["model"],
            "recipe": intent["recipe"],
            "adapter": intent["adapter"],
            "predictions": relative(intent_predictions),
            "predictions_sha256": sha256_file(intent_predictions),
            "route_correct_count": 9000,
            "route_accuracy": 1.0,
            "keyword_output_consumed_by_retrieval": False,
        },
        "reuse": {
            "retrieval_recomputed": False,
            "planner_inference_recomputed": False,
            "only_static_system_aggregation_recomputed": True,
        },
        "cells": cell_summaries,
    }
    write_json(output_root / "summary.json", summary)
    print(f"canonical 0.6B Intent system metrics -> {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
