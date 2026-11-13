#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Freeze and later finalize the assignment-3 frozen-test matrix."""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import CATALOG_PATH, PROJECT_ROOT, TOOL_TRACE_DIR, load_jsonl, read_json, write_json  # noqa: E402
from finetune.tool_trace_common import ASSIGNMENT3_DIR, PROMPT_VERSION, sha256_file, sha256_json  # noqa: E402


REPO_ROOT = PROJECT_ROOT.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "tool_trace_experiments.json"
DEFAULT_RANKING = (
    PROJECT_ROOT
    / "data"
    / "rag"
    / "_aux"
    / "reranker_finetune"
    / "bge-reranker-base-e1-g8-bs16_rank_fusion_tuned_hybrid"
    / "preds_rank_fusion_w0p6_k10.jsonl"
)
DEFAULT_PRED_DIR = PROJECT_ROOT / "finetune" / "outputs" / "preds" / "tool_trace"
DEFAULT_SNAPSHOT = ASSIGNMENT3_DIR / "tool_trace_experiments_preregistered.json"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def resolve_saved(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def require_file(path: Path) -> None:
    if not path.is_file():
        raise SystemExit(f"missing file: {path}")


def require_dir(path: Path) -> None:
    if not path.is_dir():
        raise SystemExit(f"missing directory: {path}")


def file_record(path: Path) -> dict[str, str]:
    require_file(path)
    return {"path": relative(path), "sha256": sha256_file(path)}


def checkpoint_record(path: Path) -> dict[str, Any]:
    require_dir(path)
    names = ["adapter_config.json", "adapter_model.safetensors", "args.json", "trainer_state.json"]
    files = {name: sha256_file(path / name) for name in names if (path / name).is_file()}
    if "adapter_config.json" not in files or "adapter_model.safetensors" not in files:
        raise SystemExit(f"not an adapter checkpoint: {path}")
    return {
        "path": relative(path),
        "files": files,
        "identity_sha256": sha256_json(files),
    }


def resolve_argument_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def validate_sft_checkpoint(checkpoint: Path, selection: dict[str, Any]) -> None:
    arguments = read_json(checkpoint / "args.json")
    trainer_state = read_json(checkpoint / "trainer_state.json")
    selected = selection["selected"]
    expected = {
        "tuner_type": "lora",
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "max_length": 8192,
        "torch_dtype": "bfloat16",
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "add_non_thinking_prefix": False,
        "seed": 20260725,
        "data_seed": 20260725,
    }
    mismatches = {
        key: {"actual": arguments.get(key), "expected": value}
        for key, value in expected.items()
        if arguments.get(key) != value
    }
    if mismatches:
        raise SystemExit(f"selected SFT arguments differ from fixed policy: {checkpoint}: {mismatches}")
    if arguments.get("target_modules") != ["all-linear"]:
        raise SystemExit(f"selected SFT target modules differ from fixed policy: {checkpoint}")
    if not str(arguments.get("loss_scale", "")).startswith("last_round"):
        raise SystemExit(f"selected SFT loss scale differs from fixed policy: {checkpoint}")
    if abs(float(arguments.get("learning_rate", 0.0)) - float(selected["learning_rate"])) > 1e-12:
        raise SystemExit(f"selected SFT learning rate mismatch: {checkpoint}")
    if abs(float(trainer_state.get("epoch", 0.0)) - float(selected["epoch"])) > 1e-6:
        raise SystemExit(f"selected SFT checkpoint epoch mismatch: {checkpoint}")
    datasets = arguments.get("dataset", [])
    if isinstance(datasets, str):
        datasets = [datasets]
    if len(datasets) != 1 or Path(datasets[0]).name != "tool_trace_sft_train.runtime.jsonl":
        raise SystemExit(f"selected SFT used unexpected training data: {checkpoint}")


def validate_final_dpo_checkpoint(
    checkpoint: Path,
    selected_sft: Path,
    selection: dict[str, Any],
) -> None:
    arguments = read_json(checkpoint / "args.json")
    trainer_state = read_json(checkpoint / "trainer_state.json")
    selected = selection["selected"]
    if arguments.get("rlhf_type") != "dpo" or arguments.get("tuner_type") != "lora":
        raise SystemExit(f"invalid final DPO mode/tuner: {checkpoint}")
    loss_type = arguments.get("loss_type")
    if not (loss_type == "sigmoid" or loss_type == ["sigmoid"]):
        raise SystemExit(f"invalid final DPO loss type: {checkpoint}: {loss_type}")
    if abs(float(arguments.get("learning_rate", 0.0)) - float(selected["learning_rate"])) > 1e-12:
        raise SystemExit(f"final DPO learning rate differs from dev selection: {checkpoint}")
    if abs(float(arguments.get("beta", 0.0)) - float(selected["beta"])) > 1e-12:
        raise SystemExit(f"final DPO beta differs from dev selection: {checkpoint}")
    if abs(float(arguments.get("num_train_epochs", 0.0)) - float(selected["epoch"])) > 1e-12:
        raise SystemExit(f"final DPO epoch differs from dev selection: {checkpoint}")
    if abs(float(trainer_state.get("epoch", 0.0)) - float(selected["epoch"])) > 1e-6:
        raise SystemExit(f"final DPO checkpoint is not the fixed endpoint: {checkpoint}")
    for key in ["adapters", "ref_adapters"]:
        values = arguments.get(key, [])
        if isinstance(values, str):
            values = [values]
        if len(values) != 1 or resolve_argument_path(values[0]).resolve() != selected_sft.resolve():
            raise SystemExit(f"final DPO {key} does not point to selected SFT: {checkpoint}")
    datasets = arguments.get("dataset", [])
    if isinstance(datasets, str):
        datasets = [datasets]
    if len(datasets) != 1 or Path(datasets[0]).name != "tool_trace_dpo_all.runtime.jsonl":
        raise SystemExit(f"final DPO did not use the all-pair runtime data: {checkpoint}")
    if arguments.get("val_dataset"):
        raise SystemExit(f"final all-pair DPO unexpectedly has validation data: {checkpoint}")


def read_complete_selection(path: Path, expected_stage: str, candidate_k: int) -> dict[str, Any]:
    require_file(path)
    payload = read_json(path)
    if payload.get("status") != "complete" or payload.get("stage") != expected_stage:
        raise SystemExit(f"invalid {expected_stage} selection: {path}")
    if payload.get("selected", {}).get("candidate_k") != candidate_k:
        raise SystemExit(f"selection K differs from frozen K: {path}")
    return {
        "path": relative(path),
        "sha256": sha256_file(path),
        "selected": payload["selected"],
        "dpo_replacement_gate": payload.get("dpo_replacement_gate"),
    }


def assert_label_free(path: Path, expected_view: str) -> None:
    rows = load_jsonl(path)
    if len(rows) != 6000:
        raise SystemExit(f"{path}: expected 6000 label-free rows")
    expected = {"id", "query", "input_hash"}
    if expected_view == "rag":
        expected |= {"candidate_tools", "candidate_k", "ranking_hash"}
    if any(set(row) != expected for row in rows):
        raise SystemExit(f"{path}: label-free field whitelist failed")


def matrix_cells(
    frozen_global: Path,
    frozen_rag: Path,
    checkpoints: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    cells = []
    for model in ["qwen3_4b", "qwen3_8b"]:
        model_path = PROJECT_ROOT / "models" / "Qwen" / ("Qwen3-4B" if model.endswith("4b") else "Qwen3-8B")
        for stage in ["base", "sft", "sft_dpo"]:
            adapter = ""
            if stage == "sft":
                adapter = checkpoints[f"{model}_sft"]["path"]
            elif stage == "sft_dpo":
                adapter = checkpoints[f"{model}_dpo"]["path"]
            stage_id = "dpo" if stage == "sft_dpo" else stage
            for view, input_path in [("global", frozen_global), ("rag", frozen_rag)]:
                cell_id = f"{model}_{stage_id}_{view}"
                prediction = DEFAULT_PRED_DIR / f"frozen-{cell_id}.jsonl"
                cells.append(
                    {
                        "cell_id": cell_id,
                        "model": model,
                        "model_path": relative(model_path),
                        "stage": stage,
                        "view": view,
                        "adapter": adapter,
                        "input": relative(input_path),
                        "run_id": f"frozen-{cell_id}",
                        "predictions": relative(prediction),
                        "metrics": relative(prediction.with_suffix(".metrics.json")),
                        "scored_cases": relative(prediction.with_suffix(".scored.jsonl")),
                        "status": "pending",
                    }
                )
    return cells


def freeze(args: argparse.Namespace) -> int:
    if args.config.exists() and read_json(args.config).get("frozen"):
        raise SystemExit(f"config is already frozen: {args.config}")
    assert_label_free(args.frozen_global_input, "global")
    assert_label_free(args.frozen_rag_input, "rag")
    for audit_path in [args.frozen_input_audit, args.dpo_audit]:
        require_file(audit_path)
        if not read_json(audit_path).get("strict_ok"):
            raise SystemExit(f"strict audit failed: {audit_path}")
    required_audits = {
        "api_generation_audit": (
            ASSIGNMENT3_DIR / "api_native_dev_2400" / "generation_audit.json",
            "strict_ok",
        ),
        "api_isolation_audit": (
            ASSIGNMENT3_DIR / "api_native_dev_2400" / "isolation_audit.json",
            "strict_ok",
        ),
        "sft_data_audit": (ASSIGNMENT3_DIR / "sft_data_audit.json", "strict_ok"),
        "sft_labels_qwen3_4b": (
            ASSIGNMENT3_DIR / "swift_sft_labels_qwen3_4b_smoke.json",
            "strict_ok",
        ),
        "sft_labels_qwen3_8b": (
            ASSIGNMENT3_DIR / "swift_sft_labels_qwen3_8b_smoke.json",
            "strict_ok",
        ),
        "dpo_token_length_audit": (
            ASSIGNMENT3_DIR / "dpo_token_length_audit.json",
            "strict_ok",
        ),
        "dpo_labels_qwen3_4b": (
            ASSIGNMENT3_DIR / "dpo_labels_smoke_qwen3-4b.json",
            "strict_ok",
        ),
        "dpo_labels_qwen3_8b": (
            ASSIGNMENT3_DIR / "dpo_labels_smoke_qwen3-8b.json",
            "strict_ok",
        ),
        "dpo_training_runs_audit": (
            ASSIGNMENT3_DIR / "dpo_training_runs_audit.json",
            "strict_ok",
        ),
    }
    for name, (path, success_field) in required_audits.items():
        require_file(path)
        if not read_json(path).get(success_field):
            raise SystemExit(f"required audit failed: {name}: {path}")

    environment = read_json(args.environment)
    if environment.get("status") != "complete":
        raise SystemExit(f"incomplete environment manifest: {args.environment}")
    models_environment = environment.get("models", {})
    model_specs = {
        "qwen3_4b": ("Qwen/Qwen3-4B", args.sft4, args.dpo4, args.sft4_selection, args.dpo4_selection),
        "qwen3_8b": ("Qwen/Qwen3-8B", args.sft8, args.dpo8, args.sft8_selection, args.dpo8_selection),
    }
    checkpoints: dict[str, dict[str, Any]] = {}
    selections = {}
    models = {}
    for model_key, (model_id, sft_path, dpo_path, sft_selection, dpo_selection) in model_specs.items():
        model_environment = models_environment.get(model_id, {})
        if not model_environment.get("exists"):
            raise SystemExit(f"model missing from environment manifest: {model_id}")
        if not model_environment.get("huggingface_revision"):
            raise SystemExit(f"model revision missing from environment manifest: {model_id}")
        checkpoints[f"{model_key}_sft"] = checkpoint_record(sft_path)
        checkpoints[f"{model_key}_dpo"] = checkpoint_record(dpo_path)
        sft_selection_record = read_complete_selection(
            sft_selection, "sft", args.candidate_k
        )
        dpo_selection_record = read_complete_selection(
            dpo_selection, "dpo", args.candidate_k
        )
        selected_adapter = Path(
            sft_selection_record["selected"]["global_metrics"]["adapter"]
        )
        if not selected_adapter.is_absolute():
            selected_adapter = REPO_ROOT / selected_adapter
        if selected_adapter.resolve() != sft_path.resolve():
            raise SystemExit(
                f"selected SFT adapter mismatch: {selected_adapter} != {sft_path}"
            )
        validate_sft_checkpoint(sft_path, sft_selection_record)
        validate_final_dpo_checkpoint(
            dpo_path,
            sft_path,
            dpo_selection_record,
        )
        selections[f"{model_key}_sft"] = sft_selection_record
        selections[f"{model_key}_dpo"] = dpo_selection_record
        models[model_key] = {
            "model_id": model_id,
            "revision": model_environment.get("huggingface_revision"),
            "identity_hash": model_environment["identity_hash"],
            "selected_sft_checkpoint": checkpoints[f"{model_key}_sft"],
            "final_all_pair_dpo_checkpoint": checkpoints[f"{model_key}_dpo"],
        }

    cells = matrix_cells(args.frozen_global_input, args.frozen_rag_input, checkpoints)
    existing_predictions = [
        resolve_saved(cell["predictions"]) for cell in cells if resolve_saved(cell["predictions"]).exists()
    ]
    existing_frozen_artifacts = sorted(DEFAULT_PRED_DIR.glob("frozen-*"))
    if existing_predictions or existing_frozen_artifacts:
        raise SystemExit(
            "frozen generator artifacts already exist before preregistration: "
            + ", ".join(map(str, sorted(set(existing_predictions + existing_frozen_artifacts))))
        )

    tracked_files = {
        "catalog": CATALOG_PATH,
        "formal_train": TOOL_TRACE_DIR / "train.jsonl",
        "frozen_test": TOOL_TRACE_DIR / "test.jsonl",
        "api_stress_dev": ASSIGNMENT3_DIR / "api_native_dev_2400" / "dev.jsonl",
        "api_stress_ranking": (
            ASSIGNMENT3_DIR / "api_native_dev_2400" / "rag_rankings.jsonl"
        ),
        "intent_frozen": PROJECT_ROOT / "data" / "intent_recognition" / "intent_all.jsonl",
        "frozen_ranking": args.frozen_ranking,
        "frozen_queries_label_free": args.frozen_global_input,
        "frozen_rag_candidates_label_free": args.frozen_rag_input,
        "frozen_label_free_audit": args.frozen_input_audit,
        "dpo_data_audit": args.dpo_audit,
        "effective_reviewed_train": ASSIGNMENT3_DIR / "train.reviewed.jsonl",
        "sft_compact_train": ASSIGNMENT3_DIR / "sft_train_compact.jsonl",
        "dpo_preferences": ASSIGNMENT3_DIR / "dpo_preferences_compact.jsonl",
        **{name: path for name, (path, _) in required_audits.items()},
        "gt_revisions_manifest": ASSIGNMENT3_DIR / "gt_revisions_manifest.json",
        "environment_manifest": args.environment,
        "prompt_and_parser_code": PROJECT_ROOT / "scripts" / "finetune" / "tool_trace_common.py",
        "inference_code": PROJECT_ROOT / "scripts" / "finetune" / "infer_tool_trace.py",
        "evaluation_code": PROJECT_ROOT / "scripts" / "finetune" / "evaluate_tool_trace.py",
    }
    payload = {
        "schema_version": 2,
        "task": "assignment3_tool_trace",
        "status": "frozen_pending_execution",
        "frozen": True,
        "frozen_at_utc": utc_now(),
        "completed_at_utc": None,
        "scope": (
            "固定100工具、API原生query与formal split做query/trace隔离；"
            "legacy formal train/frozen仍有已披露context signature重叠的闭集离线比较"
        ),
        "models": models,
        "selections": selections,
        "fixed_inputs": {
            "catalog": relative(CATALOG_PATH),
            "frozen_test": relative(TOOL_TRACE_DIR / "test.jsonl"),
            "frozen_ranking": relative(args.frozen_ranking),
            "frozen_queries_label_free": relative(args.frozen_global_input),
            "frozen_rag_candidates_label_free": relative(args.frozen_rag_input),
            "candidate_k": args.candidate_k,
            "prompt_version": PROMPT_VERSION,
            "parser_version": "strict-tools-json-v1",
            "hashes": {name: file_record(path) for name, path in tracked_files.items()},
        },
        "decoding": {
            "enable_thinking": False,
            "do_sample": False,
            "max_new_tokens": 128,
            "guided_or_constrained_decoding": False,
        },
        "selection_policy": {
            "development_split": relative(
                ASSIGNMENT3_DIR / "api_native_dev_2400" / "dev.jsonl"
            ),
            "primary": "API-native dev RAG Tool Set Exact Match across joint (LR, epoch, K)",
            "tie_breakers": [
                "API-native dev full-catalog Tool Set Exact Match",
                "API-native dev RAG Trace Exact Match",
                "API-native dev full-catalog Trace Exact Match",
                "minimum then per-view JSON Valid Rate",
                "lower K",
                "lower epoch and compute",
                "for DPO, beta closest to 0.1 then lower beta",
                "lower learning rate as the final deterministic tie-breaker",
            ],
            "frozen_test_used_for_selection": False,
        },
        "matrix": cells,
        "confirmatory_setem_comparisons": [
            {"model": model, "comparison": comparison}
            for model in ["qwen3_4b", "qwen3_8b"]
            for comparison in [
                "base_rag - base_global",
                "sft_global - base_global",
                "sft_rag - base_rag",
                "dpo_global - sft_global",
                "dpo_rag - sft_rag",
            ]
        ],
        "statistics": {
            "paired_bootstrap_seed": 20260725,
            "paired_bootstrap_samples": 10000,
            "confidence_level": 0.95,
            "setem_test": "paired McNemar exact",
            "multiple_testing": "Holm across the 10 confirmatory comparisons",
        },
        "limitations": [
            "Retriever embedding、学习率、epoch及公开reranker曾在同一6000条test上横向比较，Retriever test并非完全untouched。",
            "API-native dev由同一100工具catalog的能力边界生成，不代表新ontology或线上开放世界分布。",
            "full-catalog视图向模型展示全部100个工具，不是闭卷Wo-RAG。",
            "legacy formal train与frozen test存在2764个context signature重叠；API-native dev与两者的完整query和trace id隔离。",
            "GT抽审由模型辅助并经Codex逐项裁决，未取得独立人类复核。",
        ],
        "preregistration_snapshot": relative(args.snapshot),
        "preregistration_snapshot_sha256": None,
    }
    args.snapshot.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.snapshot, payload)
    snapshot_hash = sha256_file(args.snapshot)
    payload["preregistration_snapshot_sha256"] = snapshot_hash
    write_json(args.config, payload)
    print(f"frozen matrix preregistered: {args.config}")
    print(f"snapshot sha256={snapshot_hash}")
    return 0


def complete(args: argparse.Namespace) -> int:
    config = read_json(args.config)
    if not config.get("frozen") or config.get("status") != "frozen_pending_execution":
        raise SystemExit("config is not awaiting frozen-matrix completion")
    snapshot = resolve_saved(config["preregistration_snapshot"])
    if sha256_file(snapshot) != config["preregistration_snapshot_sha256"]:
        raise SystemExit("preregistration snapshot hash mismatch")
    for cell in config["matrix"]:
        prediction = resolve_saved(cell["predictions"])
        metrics_path = resolve_saved(cell["metrics"])
        scored_path = resolve_saved(cell["scored_cases"])
        for path in [prediction, metrics_path, scored_path]:
            require_file(path)
        predictions = load_jsonl(prediction)
        scored = load_jsonl(scored_path)
        metrics = read_json(metrics_path)
        if len(predictions) != 6000 or len(scored) != 6000 or metrics.get("count") != 6000:
            raise SystemExit(f"{cell['cell_id']}: expected 6000 rows")
        if metrics.get("prediction_contains_gold_fields"):
            raise SystemExit(f"{cell['cell_id']}: prediction contains labels")
        if sha256_file(prediction) != metrics.get("predictions_sha256"):
            raise SystemExit(f"{cell['cell_id']}: prediction/metrics hash mismatch")
        cell.update(
            {
                "status": "complete",
                "prediction_sha256": sha256_file(prediction),
                "metrics_sha256": sha256_file(metrics_path),
                "scored_cases_sha256": sha256_file(scored_path),
            }
        )
    config["status"] = "complete"
    config["completed_at_utc"] = utc_now()
    write_json(args.config, config)
    print(f"frozen matrix complete: {args.config}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--candidate-k", type=int, required=True)
    freeze_parser.add_argument("--sft4", type=Path, required=True)
    freeze_parser.add_argument("--dpo4", type=Path, required=True)
    freeze_parser.add_argument("--sft8", type=Path, required=True)
    freeze_parser.add_argument("--dpo8", type=Path, required=True)
    freeze_parser.add_argument("--sft4-selection", type=Path, required=True)
    freeze_parser.add_argument("--dpo4-selection", type=Path, required=True)
    freeze_parser.add_argument("--sft8-selection", type=Path, required=True)
    freeze_parser.add_argument("--dpo8-selection", type=Path, required=True)
    freeze_parser.add_argument("--frozen-global-input", type=Path, required=True)
    freeze_parser.add_argument("--frozen-rag-input", type=Path, required=True)
    freeze_parser.add_argument("--frozen-input-audit", type=Path, required=True)
    freeze_parser.add_argument("--dpo-audit", type=Path, required=True)
    freeze_parser.add_argument("--frozen-ranking", type=Path, default=DEFAULT_RANKING)
    freeze_parser.add_argument(
        "--environment", type=Path, default=ASSIGNMENT3_DIR / "environment_manifest.json"
    )
    freeze_parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    subparsers.add_parser("complete")
    args = parser.parse_args()
    return freeze(args) if args.command == "freeze" else complete(args)


if __name__ == "__main__":
    raise SystemExit(main())
