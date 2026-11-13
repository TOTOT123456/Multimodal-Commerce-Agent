#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit DPO run logs for finite losses, rewards, margins, and endpoints."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import load_jsonl, read_json, write_json  # noqa: E402
from finetune.tool_trace_common import ASSIGNMENT3_DIR, sha256_file, summary_stats  # noqa: E402


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be NAME=OUTPUT_DIR")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("--run must be NAME=OUTPUT_DIR")
    return name, Path(path)


def nonfinite_paths(value: Any, prefix: str = "") -> list[str]:
    failures = []
    if isinstance(value, float) and not math.isfinite(value):
        failures.append(prefix)
    elif isinstance(value, dict):
        for key, nested in value.items():
            failures.extend(nonfinite_paths(nested, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            failures.extend(nonfinite_paths(nested, f"{prefix}[{index}]"))
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=ASSIGNMENT3_DIR / "dpo_training_runs_audit.json",
    )
    args = parser.parse_args()

    results = {}
    failures = []
    for name, output_dir in args.run:
        args_path = output_dir / "args.json"
        log_path = output_dir / "logging.jsonl"
        if not args_path.is_file() or not log_path.is_file():
            failures.append({"run": name, "reason": "missing_args_or_log"})
            continue
        arguments = read_json(args_path)
        expected_adapter_shape = {
            "r": int(arguments.get("lora_rank", -1)),
            "lora_alpha": int(arguments.get("lora_alpha", -1)),
            "lora_dropout": float(arguments.get("lora_dropout", -1.0)),
        }
        log_rows = load_jsonl(log_path)
        nonfinite = [
            path
            for row_index, row in enumerate(log_rows)
            for path in nonfinite_paths(row, f"row[{row_index}]")
        ]
        metric_keys = [
            "loss",
            "rewards/chosen",
            "rewards/rejected",
            "rewards/margins",
            "rewards/accuracies",
            "eval_loss",
            "eval_rewards/chosen",
            "eval_rewards/rejected",
            "eval_rewards/margins",
            "eval_rewards/accuracies",
        ]
        series = {
            key: [float(row[key]) for row in log_rows if isinstance(row.get(key), (int, float))]
            for key in metric_keys
        }
        checkpoints = sorted(
            (
                path
                for path in output_dir.glob("checkpoint-*")
                if path.is_dir() and (path / "trainer_state.json").is_file()
            ),
            key=lambda path: int(path.name.rsplit("-", 1)[1]),
        )
        endpoint = None
        endpoint_ok = False
        adapter_config = None
        adapter_shape_ok = False
        checkpoint_records = []
        for checkpoint in checkpoints:
            checkpoint_config_path = checkpoint / "adapter_config.json"
            checkpoint_config = (
                read_json(checkpoint_config_path) if checkpoint_config_path.is_file() else None
            )
            checkpoint_shape_ok = bool(
                checkpoint_config is not None
                and int(checkpoint_config.get("r", -1))
                == expected_adapter_shape["r"]
                and int(checkpoint_config.get("lora_alpha", -1))
                == expected_adapter_shape["lora_alpha"]
                and abs(
                    float(checkpoint_config.get("lora_dropout", -1.0))
                    - expected_adapter_shape["lora_dropout"]
                )
                <= 1e-12
            )
            checkpoint_records.append(
                {
                    "step": int(checkpoint.name.rsplit("-", 1)[1]),
                    "path": str(checkpoint),
                    "adapter_model_sha256": sha256_file(
                        checkpoint / "adapter_model.safetensors"
                    )
                    if (checkpoint / "adapter_model.safetensors").is_file()
                    else "",
                    "adapter_shape_ok": checkpoint_shape_ok,
                }
            )
        if checkpoints:
            endpoint = checkpoints[-1]
            trainer_state = read_json(endpoint / "trainer_state.json")
            global_step = int(trainer_state.get("global_step", 0))
            max_steps = int(trainer_state.get("max_steps", 0))
            epoch_tolerance = 1.0 / max(1, max_steps)
            endpoint_ok = (
                global_step > 0
                and global_step == max_steps
                and
                abs(
                    float(trainer_state.get("epoch", 0.0))
                    - float(arguments.get("num_train_epochs", 0.0))
                )
                <= epoch_tolerance + 1e-12
            )
            adapter_config_path = endpoint / "adapter_config.json"
            if adapter_config_path.is_file():
                adapter_config = read_json(adapter_config_path)
                adapter_shape_ok = (
                    int(adapter_config.get("r", -1))
                    == expected_adapter_shape["r"]
                    and int(adapter_config.get("lora_alpha", -1))
                    == expected_adapter_shape["lora_alpha"]
                    and abs(
                        float(adapter_config.get("lora_dropout", -1.0))
                        - expected_adapter_shape["lora_dropout"]
                    )
                    <= 1e-12
                )
        all_checkpoint_adapter_shape_ok = bool(checkpoint_records) and all(
            row["adapter_shape_ok"] for row in checkpoint_records
        )
        run_ok = (
            arguments.get("rlhf_type") == "dpo"
            and (arguments.get("loss_type") == "sigmoid" or arguments.get("loss_type") == ["sigmoid"])
            and not nonfinite
            and bool(series["loss"])
            and bool(series["rewards/margins"])
            and endpoint_ok
            and adapter_shape_ok
            and all_checkpoint_adapter_shape_ok
        )
        if not run_ok:
            failures.append({"run": name, "reason": "run_audit_failed"})
        results[name] = {
            "strict_ok": run_ok,
            "output_dir": str(output_dir),
            "args": str(args_path),
            "args_sha256": sha256_file(args_path),
            "logging": str(log_path),
            "logging_sha256": sha256_file(log_path),
            "beta": arguments.get("beta"),
            "loss_type": arguments.get("loss_type"),
            "rpo_alpha": arguments.get("rpo_alpha"),
            "learning_rate": arguments.get("learning_rate"),
            "num_train_epochs": arguments.get("num_train_epochs"),
            "dataset": arguments.get("dataset"),
            "val_dataset": arguments.get("val_dataset"),
            "adapters": arguments.get("adapters"),
            "ref_adapters": arguments.get("ref_adapters"),
            "endpoint": str(endpoint) if endpoint else "",
            "endpoint_ok": endpoint_ok,
            "adapter_shape_ok": adapter_shape_ok,
            "all_checkpoint_adapter_shape_ok": all_checkpoint_adapter_shape_ok,
            "checkpoints": checkpoint_records,
            "adapter_config": {
                "r": adapter_config.get("r"),
                "lora_alpha": adapter_config.get("lora_alpha"),
                "lora_dropout": adapter_config.get("lora_dropout"),
            }
            if adapter_config is not None
            else None,
            "expected_adapter_config": expected_adapter_shape,
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
    print(f"DPO run audit runs={len(results)} strict_ok={output['strict_ok']} -> {args.out}")
    return 0 if output["strict_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
