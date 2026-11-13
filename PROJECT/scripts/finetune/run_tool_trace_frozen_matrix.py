#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execute the preregistered 12-cell frozen tool-trace matrix."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import PROJECT_ROOT, read_json  # noqa: E402
from finetune.tool_trace_common import sha256_file  # noqa: E402


REPO_ROOT = PROJECT_ROOT.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "tool_trace_experiments.json"
RUNNER = PROJECT_ROOT / "scripts" / "finetune" / "run_assignment3_eval.sh"
EVALUATOR = PROJECT_ROOT / "scripts" / "finetune" / "evaluate_tool_trace.py"
MANAGER = PROJECT_ROOT / "scripts" / "finetune" / "manage_tool_trace_experiments.py"
GPU_GROUPS = ["0,1", "2,3", "4,5", "6,7"]


def resolve_saved(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def verify_frozen_config(config: dict[str, Any]) -> None:
    if not config.get("frozen") or config.get("status") != "frozen_pending_execution":
        raise SystemExit("matrix config must be frozen and pending execution")
    snapshot = resolve_saved(config["preregistration_snapshot"])
    if sha256_file(snapshot) != config["preregistration_snapshot_sha256"]:
        raise SystemExit("preregistration snapshot hash mismatch")
    for name, record in config["fixed_inputs"]["hashes"].items():
        path = resolve_saved(record["path"])
        if sha256_file(path) != record["sha256"]:
            raise SystemExit(f"frozen input/code hash changed: {name}: {path}")
    environment_record = config["fixed_inputs"]["hashes"]["environment_manifest"]
    environment = read_json(resolve_saved(environment_record["path"]))
    for path_value, expected_hash in environment["training_scripts"].items():
        path = Path(path_value)
        if sha256_file(path) != expected_hash:
            raise SystemExit(f"recorded assignment-3 code changed after environment freeze: {path}")
    for selection in config["selections"].values():
        path = resolve_saved(selection["path"])
        if sha256_file(path) != selection["sha256"]:
            raise SystemExit(f"selection evidence changed after freeze: {path}")
    for model in config["models"].values():
        for checkpoint_key in ["selected_sft_checkpoint", "final_all_pair_dpo_checkpoint"]:
            checkpoint = model[checkpoint_key]
            root = resolve_saved(checkpoint["path"])
            hashes = {
                name: sha256_file(root / name)
                for name in checkpoint["files"]
            }
            if hashes != checkpoint["files"]:
                raise SystemExit(f"checkpoint changed after freeze: {root}")


def run_wave(
    cells: list[dict[str, Any]],
    batch_size: int,
    max_new_tokens: int,
) -> None:
    processes = []
    for cell, gpu_group in zip(cells, GPU_GROUPS):
        model = resolve_saved(cell["model_path"])
        adapter = resolve_saved(cell["adapter"]) if cell["adapter"] else None
        input_path = resolve_saved(cell["input"])
        output = resolve_saved(cell["predictions"])
        output.parent.mkdir(parents=True, exist_ok=True)
        log_path = output.with_suffix(".infer.log")
        command = [
            "bash",
            str(RUNNER),
            str(model),
            str(adapter) if adapter else "none",
            str(input_path),
            cell["view"],
            str(output),
            cell["run_id"],
        ]
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHON_BIN": sys.executable,
                "GPU_LIST": gpu_group,
                "BATCH_SIZE": str(batch_size),
                "MAX_NEW_TOKENS": str(max_new_tokens),
                "WARMUP_BATCHES": "0",
                "SEED": "20260725",
            }
        )
        log_handle = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((cell, process, log_handle, log_path))
    failures = []
    for cell, process, log_handle, log_path in processes:
        return_code = process.wait()
        log_handle.close()
        if return_code:
            failures.append(f"{cell['cell_id']} (rc={return_code}, log={log_path})")
    if failures:
        raise SystemExit("frozen inference failures: " + "; ".join(failures))


def evaluate_cell(cell: dict[str, Any], gold: Path) -> None:
    command = [
        sys.executable,
        str(EVALUATOR),
        "--predictions",
        str(resolve_saved(cell["predictions"])),
        "--gold",
        str(gold),
        "--metrics-out",
        str(resolve_saved(cell["metrics"])),
        "--scored-out",
        str(resolve_saved(cell["scored_cases"])),
        "--run-id",
        cell["run_id"],
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--complete-config", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    config = read_json(args.config)
    verify_frozen_config(config)
    if config["decoding"] != {
        "enable_thinking": False,
        "do_sample": False,
        "max_new_tokens": 128,
        "guided_or_constrained_decoding": False,
    }:
        raise SystemExit("unexpected frozen decoding configuration")
    cells = config["matrix"]
    if len(cells) != 12 or len({cell["cell_id"] for cell in cells}) != 12:
        raise SystemExit("expected 12 unique matrix cells")
    candidate_k = config["fixed_inputs"]["candidate_k"]
    for cell in cells:
        if cell["view"] == "rag":
            first = next(
                line for line in resolve_saved(cell["input"]).read_text(encoding="utf-8").splitlines() if line
            )
            if json.loads(first)["candidate_k"] != candidate_k:
                raise SystemExit(f"{cell['cell_id']}: input K differs from frozen K")

    for start in range(0, len(cells), len(GPU_GROUPS)):
        wave = cells[start : start + len(GPU_GROUPS)]
        print("running frozen wave:", ", ".join(cell["cell_id"] for cell in wave), flush=True)
        run_wave(wave, args.batch_size, config["decoding"]["max_new_tokens"])
        for cell in wave:
            evaluate_cell(cell, resolve_saved(config["fixed_inputs"]["frozen_test"]))

    if args.complete_config:
        subprocess.run(
            [sys.executable, str(MANAGER), "--config", str(args.config), "complete"],
            cwd=REPO_ROOT,
            check=True,
        )
    print("frozen 12-cell matrix complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
