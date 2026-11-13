#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit completed MS-SWIFT LoRA SFT runs and their saved endpoints."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import load_jsonl, read_json, write_json  # noqa: E402
from finetune.tool_trace_common import sha256_file, summary_stats  # noqa: E402


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be NAME=OUTPUT_DIR")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("--run must be NAME=OUTPUT_DIR")
    return name, Path(path)


def nonfinite_paths(value: Any, prefix: str = "") -> list[str]:
    failures: list[str] = []
    if isinstance(value, float) and not math.isfinite(value):
        failures.append(prefix)
    elif isinstance(value, dict):
        for key, nested in value.items():
            failures.extend(nonfinite_paths(nested, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            failures.extend(nonfinite_paths(nested, f"{prefix}[{index}]"))
    return failures


def checkpoint_step(path: Path) -> int:
    return int(path.name.rsplit("-", 1)[1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    results: dict[str, Any] = {}
    failures: list[dict[str, str]] = []
    for name, output_dir in args.run:
        args_path = output_dir / "args.json"
        logging_path = output_dir / "logging.jsonl"
        if not args_path.is_file() or not logging_path.is_file():
            failures.append({"run": name, "reason": "missing_args_or_log"})
            continue
        arguments = read_json(args_path)
        log_rows = load_jsonl(logging_path)
        nonfinite = [
            path
            for row_index, row in enumerate(log_rows)
            for path in nonfinite_paths(row, f"row[{row_index}]")
        ]
        checkpoints = sorted(
            (
                path
                for path in output_dir.glob("checkpoint-*")
                if path.is_dir()
                and (path / "trainer_state.json").is_file()
                and (path / "adapter_model.safetensors").is_file()
            ),
            key=checkpoint_step,
        )
        endpoint = checkpoints[-1] if checkpoints else None
        endpoint_epoch = 0.0
        endpoint_ok = False
        if endpoint:
            trainer_state = read_json(endpoint / "trainer_state.json")
            endpoint_epoch = float(trainer_state.get("epoch", 0.0))
            expected_epoch = float(arguments.get("num_train_epochs", 0.0))
            endpoint_ok = abs(endpoint_epoch - expected_epoch) <= 1e-6
        metric_keys = ["loss", "grad_norm", "learning_rate", "token_acc", "train_loss"]
        series = {
            key: [float(row[key]) for row in log_rows if isinstance(row.get(key), (int, float))]
            for key in metric_keys
        }
        run_ok = (
            arguments.get("tuner_type") == "lora"
            and str(arguments.get("loss_scale", "")).startswith("last_round")
            and not nonfinite
            and bool(series["loss"])
            and bool(series["train_loss"])
            and endpoint_ok
        )
        if not run_ok:
            failures.append({"run": name, "reason": "run_audit_failed"})
        results[name] = {
            "strict_ok": run_ok,
            "output_dir": str(output_dir),
            "args_sha256": sha256_file(args_path),
            "logging_sha256": sha256_file(logging_path),
            "dataset": arguments.get("dataset"),
            "learning_rate": arguments.get("learning_rate"),
            "num_train_epochs": arguments.get("num_train_epochs"),
            "max_length": arguments.get("max_length"),
            "effective_world_size": arguments.get("world_size"),
            "checkpoint_count": len(checkpoints),
            "checkpoints": [str(path) for path in checkpoints],
            "endpoint": str(endpoint) if endpoint else "",
            "endpoint_epoch": endpoint_epoch,
            "endpoint_ok": endpoint_ok,
            "endpoint_adapter_sha256": (
                sha256_file(endpoint / "adapter_model.safetensors") if endpoint else ""
            ),
            "nonfinite_paths": nonfinite[:100],
            "metrics": {
                key: {
                    "summary": summary_stats(values),
                    "first": values[0] if values else None,
                    "last": values[-1] if values else None,
                }
                for key, values in series.items()
                if values
            },
        }

    output = {
        "status": "complete" if not failures else "failed",
        "strict_ok": not failures,
        "run_count": len(results),
        "failures": failures,
        "runs": results,
    }
    write_json(args.out, output)
    print(f"SFT run audit runs={len(results)} strict_ok={output['strict_ok']} -> {args.out}")
    return 0 if output["strict_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
