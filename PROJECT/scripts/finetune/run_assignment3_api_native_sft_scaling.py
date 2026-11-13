#!/usr/bin/env python3
"""Evaluate and select the API-native 9K->54K Assignment3 SFT scale."""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import queue
import subprocess
import threading
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parent
CONFIG_PATH = PROJECT_ROOT / "configs/assignment3_api_native_sft_scaling.json"
ARTIFACT_ROOT = (
    PROJECT_ROOT
    / "data/tool_call_trace/_aux/assignment3"
    / "assignment3_api_native_sft_scaling_20260802"
)
PRED_ROOT = (
    PROJECT_ROOT
    / "finetune/outputs/preds/tool_trace"
    / "assignment3_api_native_sft_scaling_20260802"
)
TRAIN_OUTPUT_ROOT = PROJECT_ROOT / "finetune/outputs"
TRAIN_DATA_ROOT = (
    TRAIN_OUTPUT_ROOT / "tooltrace-data-cache/assignment3_sft_ablation"
)
TRAIN_SCRIPT = PROJECT_ROOT / "scripts/finetune/run_swift_tool_trace_lora.sh"
EVAL_SCRIPT = PROJECT_ROOT / "scripts/finetune/run_assignment3_eval.sh"
EVALUATOR = PROJECT_ROOT / "scripts/finetune/evaluate_tool_trace.py"
SYSTEM_EVALUATOR = PROJECT_ROOT / "scripts/finetune/evaluate_tool_trace_system.py"
INTENT_PREDICTIONS = (
    Path(os.environ["INTENT_PREDS"])
    if os.environ.get("INTENT_PREDS")
    else PROJECT_ROOT
    / "finetune/outputs/preds/qwen3-0.6b-full_lora-context_isolated-best.jsonl"
)
PYTHON_BIN = Path("/root/miniconda/envs/qwen-intent-ft/bin/python")
SELECTION_LOCK = ARTIFACT_ROOT / "selection.lock.json"
DEV_SUMMARY = ARTIFACT_ROOT / "development_summary.json"
CONFIRMATION_SUMMARY = ARTIFACT_ROOT / "confirmation_summary.json"
PROTECTED_SUMMARY = ARTIFACT_ROOT / "protected_summary.json"
RESULTS_SUMMARY = ARTIFACT_ROOT / "results_summary.json"
LOG_LOCK = threading.Lock()
CHECKPOINT_STEPS = {9000: 846, 18000: 1689, 36000: 3375, 54000: 5064}


@dataclasses.dataclass(frozen=True)
class Candidate:
    model: str
    size: int

    @property
    def run_id(self) -> str:
        return f"a3-api-native-scale-{self.model}-n{self.size}"

    @property
    def source_run_id(self) -> str:
        return f"a3-ablate-{self.model}-n{self.size}-lr5em5-e3-r32-d05"

    @property
    def checkpoint(self) -> Path:
        return (
            TRAIN_OUTPUT_ROOT
            / f"tooltrace-{self.source_run_id}"
            / f"checkpoint-{CHECKPOINT_STEPS[self.size]}"
        )

    @property
    def model_path(self) -> Path:
        suffix = "4B" if self.model == "4b" else "8B"
        return PROJECT_ROOT / "models/Qwen" / f"Qwen3-{suffix}"

    @property
    def data_path(self) -> Path:
        return TRAIN_DATA_ROOT / f"sft_train_{self.size}.runtime.jsonl"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT.resolve()))


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def candidates(config: dict[str, Any]) -> list[Candidate]:
    return [
        Candidate(model, int(size))
        for model in config["training"]["models"]
        for size in config["data"]["nested_sizes"]
    ]


def verify_checkpoint(candidate: Candidate, config: dict[str, Any]) -> dict[str, Any]:
    checkpoint = candidate.checkpoint
    required = [
        checkpoint / "adapter_model.safetensors",
        checkpoint / "adapter_config.json",
        checkpoint / "trainer_state.json",
        checkpoint / "args.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"incomplete checkpoint {candidate.run_id}: {missing}")
    arguments = read_json(checkpoint / "args.json")
    state = read_json(checkpoint / "trainer_state.json")
    datasets = arguments.get("dataset", [])
    if isinstance(datasets, str):
        datasets = [datasets]
    training = config["training"]
    checks = {
        "epoch": abs(float(state.get("epoch", -1)) - training["epochs"]) < 1e-9,
        "num_train_epochs": abs(
            float(arguments.get("num_train_epochs", -1)) - training["epochs"]
        )
        < 1e-9,
        "learning_rate": abs(
            float(arguments.get("learning_rate", -1)) - training["learning_rate"]
        )
        < 1e-12,
        "lora_rank": int(arguments.get("lora_rank", -1)) == training["lora_rank"],
        "lora_alpha": int(arguments.get("lora_alpha", -1))
        == training["lora_alpha"],
        "lora_dropout": abs(
            float(arguments.get("lora_dropout", -1)) - training["lora_dropout"]
        )
        < 1e-12,
        "max_length": int(arguments.get("max_length", -1)) == training["max_length"],
        "dataset": len(datasets) == 1
        and Path(datasets[0]).resolve() == candidate.data_path.resolve(),
        "data_rows": sum(1 for _ in candidate.data_path.open(encoding="utf-8"))
        == candidate.size,
    }
    if not all(checks.values()):
        raise RuntimeError(f"checkpoint protocol mismatch {candidate.run_id}: {checks}")
    return {
        "model": candidate.model,
        "size": candidate.size,
        "source_run_id": candidate.source_run_id,
        "checkpoint": relative(checkpoint),
        "data": relative(candidate.data_path),
        "checks": checks,
    }


def metrics_path(candidate: Candidate, split: str, view: str) -> Path:
    return ARTIFACT_ROOT / "evaluation" / split / candidate.run_id / f"{view}.metrics.json"


def scored_path(candidate: Candidate, split: str, view: str) -> Path:
    return ARTIFACT_ROOT / "evaluation" / split / candidate.run_id / f"{view}.scored.jsonl"


def prediction_path(candidate: Candidate, split: str, view: str) -> Path:
    return PRED_ROOT / split / candidate.run_id / f"{view}.jsonl"


def metrics_complete(path: Path, rows: int, view: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = read_json(path)
    except (json.JSONDecodeError, OSError):
        return False
    return (
        payload.get("status") == "complete"
        and payload.get("count") == rows
        and payload.get("view") == view
    )


def run_logged(command: list[str], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{utc_now()}] COMMAND: {' '.join(command)}\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        log.write(f"[{utc_now()}] EXIT: {completed.returncode}\n")
    if completed.returncode:
        raise RuntimeError(f"command failed ({completed.returncode}); see {log_path}")


def evaluate(
    candidate: Candidate,
    split: str,
    view: str,
    split_config: dict[str, Any],
    gpu: str,
) -> Path:
    rows = int(split_config["rows"])
    output_metrics = metrics_path(candidate, split, view)
    if metrics_complete(output_metrics, rows, view):
        return output_metrics
    output_prediction = prediction_path(candidate, split, view)
    output_scored = scored_path(candidate, split, view)
    output_prediction.parent.mkdir(parents=True, exist_ok=True)
    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    input_path = resolve_repo_path(split_config[f"{view}_input"])
    gold_path = resolve_repo_path(split_config["gold"])
    if view == "rag":
        batch = "16" if candidate.model == "4b" else "8"
    else:
        batch = "8" if candidate.model == "4b" else "4"
    run_id = f"{candidate.run_id}-{split}-{view}"
    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": str(PYTHON_BIN),
            "GPU_LIST": gpu,
            "BATCH_SIZE": batch,
        }
    )
    log_path = ARTIFACT_ROOT / "logs" / split / f"{run_id}.log"
    run_logged(
        [
            "bash",
            str(EVAL_SCRIPT),
            str(candidate.model_path),
            str(candidate.checkpoint),
            str(input_path),
            view,
            str(output_prediction),
            run_id,
        ],
        log_path,
        env,
    )
    run_logged(
        [
            str(PYTHON_BIN),
            str(EVALUATOR),
            "--predictions",
            str(output_prediction),
            "--gold",
            str(gold_path),
            "--metrics-out",
            str(output_metrics),
            "--scored-out",
            str(output_scored),
            "--run-id",
            run_id,
        ],
        log_path,
        env,
    )
    if not metrics_complete(output_metrics, rows, view):
        raise RuntimeError(f"incomplete metrics: {output_metrics}")
    return output_metrics


def run_jobs(
    jobs: list[tuple[Candidate, str, str, dict[str, Any]]],
    gpu_ids: list[str],
    gpus_per_job: int = 1,
) -> None:
    pending = [
        job
        for job in jobs
        if not metrics_complete(
            metrics_path(job[0], job[1], job[2]), int(job[3]["rows"]), job[2]
        )
    ]
    pending.sort(
        key=lambda job: (
            int(job[3]["rows"]),
            job[0].model == "8b",
            job[2] == "global",
        ),
        reverse=True,
    )
    if not pending:
        return
    if gpus_per_job <= 0 or len(gpu_ids) < gpus_per_job:
        raise ValueError("invalid gpus_per_job")
    available: queue.Queue[str] = queue.Queue()
    for gpu in gpu_ids:
        available.put(gpu)

    def execute(job: tuple[Candidate, str, str, dict[str, Any]]) -> None:
        candidate, split, view, split_config = job
        allocated = [available.get() for _ in range(gpus_per_job)]
        gpu = ",".join(allocated)
        with LOG_LOCK:
            print(
                f"[{utc_now()}] start {candidate.run_id} {split}/{view} gpu={gpu}",
                flush=True,
            )
        try:
            evaluate(candidate, split, view, split_config, gpu)
        finally:
            for value in allocated:
                available.put(value)
        with LOG_LOCK:
            print(
                f"[{utc_now()}] done  {candidate.run_id} {split}/{view} gpu={gpu}",
                flush=True,
            )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, len(gpu_ids) // gpus_per_job)
    ) as pool:
        futures = [pool.submit(execute, job) for job in pending]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def trace_em(metrics: dict[str, Any]) -> float:
    return float(metrics["overall"]["trace_em"])


def panel_trace(metrics: dict[str, Any], style: str) -> float:
    return float(metrics["buckets"]["style"][style]["trace_em"])


def weighted_trace(metrics: dict[str, Any]) -> float:
    legacy = panel_trace(metrics, "legacy_compact")
    api = panel_trace(metrics, "api_native")
    return (6.0 * legacy + api) / 7.0


def metric_record(candidate: Candidate, split: str, view: str) -> dict[str, Any]:
    path = metrics_path(candidate, split, view)
    payload = read_json(path)
    record: dict[str, Any] = {
        "metrics": relative(path),
        "metrics_sha256": sha256_file(path),
        "set_em": float(payload["overall"]["set_em"]),
        "trace_em": trace_em(payload),
        "strict_output_valid_rate": float(
            payload["overall"]["strict_output_valid_rate"]
        ),
        "keyword_micro_f1": float(payload["overall"]["keyword_micro_f1"]),
    }
    styles = payload.get("buckets", {}).get("style", {})
    if "legacy_compact" in styles and "api_native" in styles:
        record["panel_trace_em"] = {
            "legacy_compact": panel_trace(payload, "legacy_compact"),
            "api_native": panel_trace(payload, "api_native"),
        }
        record["deployment_weighted_trace_em"] = weighted_trace(payload)
    return record


def build_development_summary(
    all_candidates: list[Candidate], config: dict[str, Any], inventory: list[dict[str, Any]]
) -> dict[str, Any]:
    rows = []
    for candidate in all_candidates:
        row = {
            "model": candidate.model,
            "size": candidate.size,
            "source_run_id": candidate.source_run_id,
            "checkpoint": relative(candidate.checkpoint),
            "selection": {
                view: metric_record(candidate, "selection", view)
                for view in config["evaluation"]["views"]
            },
            "strict_combo_isolated": {
                view: metric_record(candidate, "strict_combo_isolated", view)
                for view in config["evaluation"]["views"]
            },
        }
        rows.append(row)
    ranking: dict[str, list[dict[str, Any]]] = {}
    for model in config["training"]["models"]:
        model_rows = [row for row in rows if row["model"] == model]
        model_rows.sort(
            key=lambda row: (
                -row["selection"]["rag"]["deployment_weighted_trace_em"],
                -min(row["selection"]["rag"]["panel_trace_em"].values()),
                -row["strict_combo_isolated"]["rag"]["trace_em"],
                -row["selection"]["global"]["deployment_weighted_trace_em"],
                row["size"],
            )
        )
        ranking[model] = [
            {
                "rank": index,
                "size": row["size"],
                "weighted_rag_trace_em": row["selection"]["rag"][
                    "deployment_weighted_trace_em"
                ],
                "minimum_panel_rag_trace_em": min(
                    row["selection"]["rag"]["panel_trace_em"].values()
                ),
                "strict_combo_rag_trace_em": row["strict_combo_isolated"]["rag"][
                    "trace_em"
                ],
                "weighted_global_trace_em": row["selection"]["global"][
                    "deployment_weighted_trace_em"
                ],
            }
            for index, row in enumerate(model_rows, 1)
        ]
    return {
        "schema_version": 1,
        "task": "assignment3_api_native_sft_scaling_development",
        "status": "complete_before_scale_lock_and_protected_replay",
        "completed_at_utc": utc_now(),
        "config": relative(CONFIG_PATH),
        "config_sha256": sha256_file(CONFIG_PATH),
        "data_audit": config["data"]["audit"],
        "data_audit_sha256": sha256_file(resolve_repo_path(config["data"]["audit"])),
        "checkpoint_inventory": inventory,
        "cells": rows,
        "ranking": ranking,
    }


def freeze_selection(summary: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    selected = {}
    for model, ranking in summary["ranking"].items():
        selected[model] = {
            "winner_size": ranking[0]["size"],
            "runner_up_size": ranking[1]["size"],
            "development_ranking": ranking,
        }
    lock = {
        "schema_version": 1,
        "task": "assignment3_api_native_sft_scale_selection",
        "status": "locked_before_protected_replay",
        "locked_at_utc": utc_now(),
        "selection_rule": config["selection_rule"],
        "development_summary": relative(DEV_SUMMARY),
        "development_summary_sha256": sha256_file(DEV_SUMMARY),
        "selected": selected,
    }
    if SELECTION_LOCK.is_file():
        previous = read_json(SELECTION_LOCK)
        if previous.get("selected") != selected:
            raise RuntimeError("existing selection lock differs from current development ranking")
        lock["locked_at_utc"] = previous.get("locked_at_utc", lock["locked_at_utc"])
    write_json(SELECTION_LOCK, lock)
    return lock


def selected_candidates(lock: dict[str, Any], include_runner_up: bool) -> list[Candidate]:
    result = []
    for model, selected in lock["selected"].items():
        result.append(Candidate(model, int(selected["winner_size"])))
        if include_runner_up:
            result.append(Candidate(model, int(selected["runner_up_size"])))
    return result


def exact_mcnemar_pvalue(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value) for value in range(min(left_only, right_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def paired_record(left: Candidate, right: Candidate, split: str, view: str) -> dict[str, Any]:
    left_rows = {row["id"]: row for row in read_jsonl(scored_path(left, split, view))}
    right_rows = {row["id"]: row for row in read_jsonl(scored_path(right, split, view))}
    if set(left_rows) != set(right_rows):
        raise RuntimeError(f"paired ID mismatch: {left.run_id} vs {right.run_id}")
    left_only = sum(
        bool(left_rows[key]["trace_em"]) and not bool(right_rows[key]["trace_em"])
        for key in left_rows
    )
    right_only = sum(
        bool(right_rows[key]["trace_em"]) and not bool(left_rows[key]["trace_em"])
        for key in left_rows
    )
    return {
        "rows": len(left_rows),
        "winner_correct_runner_up_wrong": left_only,
        "runner_up_correct_winner_wrong": right_only,
        "net_rows": left_only - right_only,
        "exact_mcnemar_pvalue": exact_mcnemar_pvalue(left_only, right_only),
    }


def build_confirmation_summary(lock: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for model, selected in lock["selected"].items():
        winner = Candidate(model, int(selected["winner_size"]))
        runner_up = Candidate(model, int(selected["runner_up_size"]))
        models[model] = {
            "winner_size": winner.size,
            "runner_up_size": runner_up.size,
            "winner": {
                view: metric_record(winner, "sealed_confirmation", view)
                for view in config["evaluation"]["views"]
            },
            "runner_up": {
                view: metric_record(runner_up, "sealed_confirmation", view)
                for view in config["evaluation"]["views"]
            },
            "paired": {
                view: paired_record(winner, runner_up, "sealed_confirmation", view)
                for view in config["evaluation"]["views"]
            },
        }
    return {
        "schema_version": 1,
        "task": "assignment3_api_native_sft_scaling_supplementary_stability_diagnostic",
        "status": "supplementary_only_not_in_canonical_decision_flow",
        "role": "post-lock diagnostic; does not select, promote, or reject a checkpoint",
        "completed_at_utc": utc_now(),
        "selection_lock": relative(SELECTION_LOCK),
        "selection_lock_sha256": sha256_file(SELECTION_LOCK),
        "models": models,
    }


def build_protected_summary(lock: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for model, selected in lock["selected"].items():
        winner = Candidate(model, int(selected["winner_size"]))
        system_metrics_path = (
            ARTIFACT_ROOT
            / "evaluation/system9000"
            / f"{winner.run_id}.system.metrics.json"
        )
        system_metrics = read_json(system_metrics_path)
        results[model] = {
            "selected_size": winner.size,
            "datasets": {
                split: {
                    view: metric_record(winner, split, view)
                    for view in config["evaluation"]["views"]
                }
                for split in config["evaluation"]["protected"]
                if isinstance(config["evaluation"]["protected"][split], dict)
            },
            "static_system_legacy9000": {
                "metrics": relative(system_metrics_path),
                "metrics_sha256": sha256_file(system_metrics_path),
                "full_set": float(system_metrics["full_pipeline_success"]),
                "full_trace": float(system_metrics["full_pipeline_trace_success"]),
            },
        }
    return {
        "schema_version": 1,
        "task": "assignment3_api_native_sft_scaling_protected_replay",
        "status": "complete_after_selection_lock",
        "completed_at_utc": utc_now(),
        "selection_lock": relative(SELECTION_LOCK),
        "selection_lock_sha256": sha256_file(SELECTION_LOCK),
        "results": results,
    }


def evaluate_static_system(candidate: Candidate) -> None:
    root = ARTIFACT_ROOT / "evaluation/system9000"
    metrics = root / f"{candidate.run_id}.system.metrics.json"
    prediction = prediction_path(candidate, "legacy6000", "rag")
    if metrics.is_file():
        existing = read_json(metrics)
        input_hashes = set(existing.get("inputs", {}).values())
        if (
            existing.get("count") == 9000
            and sha256_file(INTENT_PREDICTIONS) in input_hashes
            and sha256_file(prediction) in input_hashes
        ):
            return
    root.mkdir(parents=True, exist_ok=True)
    run_logged(
        [
            str(PYTHON_BIN),
            str(SYSTEM_EVALUATOR),
            "--intent-predictions",
            str(INTENT_PREDICTIONS),
            "--tool-predictions",
            str(prediction),
            "--out",
            str(root / f"{candidate.run_id}.system.jsonl"),
            "--metrics-out",
            str(metrics),
        ],
        ARTIFACT_ROOT / "logs/system9000" / f"{candidate.run_id}.log",
        os.environ.copy(),
    )


def write_results(config: dict[str, Any]) -> None:
    payload = {
        "schema_version": 1,
        "task": "assignment3_api_native_sft_scaling",
        "status": "complete",
        "completed_at_utc": utc_now(),
        "config": {"path": relative(CONFIG_PATH), "sha256": sha256_file(CONFIG_PATH)},
        "data_audit": {
            "path": config["data"]["audit"],
            "sha256": sha256_file(resolve_repo_path(config["data"]["audit"])),
            "strict_ok": bool(read_json(resolve_repo_path(config["data"]["audit"]))["strict_ok"]),
        },
        "development": {
            "path": relative(DEV_SUMMARY),
            "sha256": sha256_file(DEV_SUMMARY),
        },
        "selection": {
            "path": relative(SELECTION_LOCK),
            "sha256": sha256_file(SELECTION_LOCK),
        },
        "protected": {
            "path": relative(PROTECTED_SUMMARY),
            "sha256": sha256_file(PROTECTED_SUMMARY),
        },
        "canonical_flow": ["development", "selection", "protected"],
    }
    if CONFIRMATION_SUMMARY.is_file():
        payload["supplementary_stability_diagnostic"] = {
            "canonical_stage": False,
            "path": relative(CONFIRMATION_SUMMARY),
            "sha256": sha256_file(CONFIRMATION_SUMMARY),
        }
    write_json(RESULTS_SUMMARY, payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["dev", "diagnostic", "confirmation", "protected", "all"],
        default="all",
        help="'all' runs the canonical dev->protected flow; diagnostic/confirmation is optional",
    )
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    args = parser.parse_args()
    gpu_ids = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpu_ids:
        raise SystemExit("at least one GPU is required")
    config = read_json(CONFIG_PATH)
    data_audit = read_json(resolve_repo_path(config["data"]["audit"]))
    if not data_audit.get("strict_ok"):
        raise SystemExit("SFT scale data audit is not strict_ok")
    strict_input_audit = read_json(ARTIFACT_ROOT / "strict_combo_isolated/input_audit.json")
    if not strict_input_audit.get("strict_ok"):
        raise SystemExit("strict-combo current-route input audit is not strict_ok")
    all_candidates = candidates(config)
    inventory = [verify_checkpoint(candidate, config) for candidate in all_candidates]

    if args.stage in {"dev", "all"}:
        dev_jobs = [
            (candidate, split, view, config["evaluation"][split])
            for candidate in all_candidates
            for split in ("selection", "strict_combo_isolated")
            for view in config["evaluation"]["views"]
        ]
        run_jobs(dev_jobs, gpu_ids)
        development = build_development_summary(all_candidates, config, inventory)
        write_json(DEV_SUMMARY, development)
        freeze_selection(development, config)
        print(f"development complete: {relative(DEV_SUMMARY)}", flush=True)

    if args.stage in {"diagnostic", "confirmation", "protected"} and not SELECTION_LOCK.is_file():
        raise SystemExit("selection lock is required before this stage")

    if args.stage in {"diagnostic", "confirmation"}:
        lock = read_json(SELECTION_LOCK)
        confirmation_candidates = selected_candidates(lock, include_runner_up=True)
        split_config = config["supplementary_diagnostics"]["sealed_confirmation_1200"]
        confirmation_jobs = [
            (candidate, "sealed_confirmation", view, split_config)
            for candidate in confirmation_candidates
            for view in config["evaluation"]["views"]
        ]
        run_jobs(confirmation_jobs, gpu_ids)
        write_json(CONFIRMATION_SUMMARY, build_confirmation_summary(lock, config))
        print(
            f"supplementary diagnostic complete: {relative(CONFIRMATION_SUMMARY)}",
            flush=True,
        )

    if args.stage in {"protected", "all"}:
        lock = read_json(SELECTION_LOCK)
        protected_candidates = selected_candidates(lock, include_runner_up=False)
        protected_jobs = [
            (candidate, split, view, split_config)
            for candidate in protected_candidates
            for split, split_config in config["evaluation"]["protected"].items()
            if isinstance(split_config, dict)
            for view in config["evaluation"]["views"]
        ]
        run_jobs(protected_jobs, gpu_ids, gpus_per_job=2)
        for candidate in protected_candidates:
            evaluate_static_system(candidate)
        write_json(PROTECTED_SUMMARY, build_protected_summary(lock, config))
        print(f"protected replay complete: {relative(PROTECTED_SUMMARY)}", flush=True)

    if all(path.is_file() for path in (DEV_SUMMARY, SELECTION_LOCK, PROTECTED_SUMMARY)):
        write_results(config)
        print(f"experiment complete: {relative(RESULTS_SUMMARY)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
