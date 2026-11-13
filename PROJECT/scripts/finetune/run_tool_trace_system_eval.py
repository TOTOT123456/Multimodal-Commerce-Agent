#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate frozen SFT tool generators behind the frozen intent gate."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import PROJECT_ROOT, read_json  # noqa: E402
from finetune.tool_trace_common import ASSIGNMENT3_DIR  # noqa: E402


REPO_ROOT = PROJECT_ROOT.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "tool_trace_experiments.json"
DEFAULT_INTENT_PREDICTIONS = (
    PROJECT_ROOT
    / "finetune"
    / "outputs"
    / "preds"
    / "qwen3-0.6b-full_lora-context_isolated-best.jsonl"
)
EVALUATOR = PROJECT_ROOT / "scripts" / "finetune" / "evaluate_tool_trace_system.py"
DEFAULT_TOOL_PREDICTIONS_DIR = (
    PROJECT_ROOT / "finetune" / "outputs" / "preds" / "tool_trace" / "system_frozen_6000"
)


def resolve_saved(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--intent-predictions", type=Path, default=DEFAULT_INTENT_PREDICTIONS)
    parser.add_argument(
        "--tool-predictions-dir",
        type=Path,
        default=DEFAULT_TOOL_PREDICTIONS_DIR,
    )
    args = parser.parse_args()
    config = read_json(args.config)
    if config.get("status") != "complete":
        raise SystemExit("tool-trace experiment config must be complete")
    cells = [
        (
            "qwen3_4b_sft",
            args.tool_predictions_dir / "a3-100-sft-4b-rag-frozen6000.jsonl",
        ),
        (
            "qwen3_8b_sft",
            args.tool_predictions_dir / "a3-100-sft-8b-rag-frozen6000.jsonl",
        ),
    ]
    missing = [str(path) for _, path in cells if not path.is_file()]
    if missing:
        raise SystemExit("missing frozen-6000 tool predictions: " + ", ".join(missing))
    for cell_id, predictions in cells:
        output = ASSIGNMENT3_DIR / f"tool_trace_system_{cell_id}.jsonl"
        metrics = ASSIGNMENT3_DIR / f"tool_trace_system_{cell_id}.metrics.json"
        subprocess.run(
            [
                sys.executable,
                str(EVALUATOR),
                "--intent-predictions",
                str(args.intent_predictions),
                "--tool-predictions",
                str(predictions),
                "--out",
                str(output),
                "--metrics-out",
                str(metrics),
            ],
            cwd=REPO_ROOT,
            check=True,
        )
    print("tool-trace frozen-6000 SFT system evaluation complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
