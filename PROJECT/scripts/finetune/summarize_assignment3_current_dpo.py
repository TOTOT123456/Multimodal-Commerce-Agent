#!/usr/bin/env python3
"""Build the final machine-readable summary for the current-route DPO experiment."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from project_common import load_jsonl, read_json, write_json  # noqa: E402
from finetune.tool_trace_common import sha256_file  # noqa: E402


EXP_ID = "assignment3_current_api9k_dpo_20260802"
EXP_ROOT = PROJECT_ROOT / "data/tool_call_trace/_aux/assignment3" / EXP_ID
CONTROLLED_PARENT_ENDPOINT = "parent-controlled"


def parse_endpoint(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("endpoint must be 4b=NAME or 8b=NAME")
    model, name = value.split("=", 1)
    if model not in {"4b", "8b"} or not name:
        raise argparse.ArgumentTypeError("endpoint must be 4b=NAME or 8b=NAME")
    return model, name


def exact_mcnemar_pvalue(improved: int, regressed: int) -> float:
    discordant = improved + regressed
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value)
        for value in range(min(improved, regressed) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def metric_record(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("status") != "complete":
        raise ValueError(f"incomplete metrics: {path}")
    overall = payload["overall"]
    tool_count = payload.get("buckets", {}).get("tool_count", {})
    error_analysis = payload.get("error_analysis", {})
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "sha256": sha256_file(path),
        "count": int(payload["count"]),
        "set_em": float(overall["set_em"]),
        "trace_em": float(overall["trace_em"]),
        "strict_output_valid_rate": float(overall["strict_output_valid_rate"]),
        "micro_precision": float(overall["micro_precision"]),
        "micro_recall": float(overall["micro_recall"]),
        "micro_f1": float(overall["micro_f1"]),
        "omission_rate": float(overall["omission_rate"]),
        "over_call_rate": float(overall["over_call_rate"]),
        "keyword_micro_f1": float(overall["keyword_micro_f1"]),
        "tool_count_trace_em": {
            str(key): float(value["trace_em"])
            for key, value in tool_count.items()
        },
        "error_counts": error_analysis.get("counts", {}),
        "set_miss_tags": error_analysis.get("set_miss_tags", {}),
    }


def resolve_artifact_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def controlled_inference_pair(sft_metrics_path: Path, dpo_metrics_path: Path) -> dict[str, Any]:
    fields = (
        "input_sha256",
        "catalog_sha256",
        "view",
        "candidate_order",
        "prompt_version",
        "batch_size",
        "max_new_tokens",
        "torch_dtype",
        "attn_implementation",
        "do_sample",
        "temperature",
        "top_p",
        "top_k",
        "samples_per_prompt",
        "seed",
        "warmup_batches",
        "limit",
        "shard_count",
    )

    def load(metrics_path: Path) -> tuple[list[Path], dict[str, Any]]:
        prediction = resolve_artifact_path(str(read_json(metrics_path)["predictions"]))
        first_manifest = prediction.with_name(
            f"{prediction.stem}.shard0.jsonl.manifest.json"
        )
        first_payload = read_json(first_manifest)
        if first_payload.get("status") != "complete":
            raise ValueError(f"incomplete inference manifest: {first_manifest}")
        shard_count = int(first_payload["run_config"]["shard_count"])
        manifests: list[Path] = []
        common_config: dict[str, Any] | None = None
        for shard_index in range(shard_count):
            manifest = prediction.with_name(
                f"{prediction.stem}.shard{shard_index}.jsonl.manifest.json"
            )
            payload = read_json(manifest)
            if payload.get("status") != "complete":
                raise ValueError(f"incomplete inference manifest: {manifest}")
            if int(payload["prediction_rows"]) != int(payload["expected_rows"]):
                raise ValueError(f"incomplete inference rows: {manifest}")
            config = payload["run_config"]
            if int(config["shard_index"]) != shard_index:
                raise ValueError(f"unexpected shard index: {manifest}")
            current = {field: config.get(field) for field in fields}
            if common_config is None:
                common_config = current
            elif current != common_config:
                raise ValueError(f"inconsistent inference shards: {manifest}")
            manifests.append(manifest)
        if common_config is None:
            raise ValueError(f"no inference manifests: {prediction}")
        return manifests, common_config

    sft_manifests, sft_config = load(sft_metrics_path)
    dpo_manifests, dpo_config = load(dpo_metrics_path)
    if sft_config != dpo_config:
        raise ValueError(
            "uncontrolled SFT/DPO inference comparison: "
            f"{sft_manifests[0]} vs {dpo_manifests[0]}"
        )
    return {
        "strict_ok": True,
        "matched_config": sft_config,
        "sft_manifest": str(sft_manifests[0].relative_to(REPO_ROOT)),
        "sft_manifest_sha256": sha256_file(sft_manifests[0]),
        "dpo_manifest": str(dpo_manifests[0].relative_to(REPO_ROOT)),
        "dpo_manifest_sha256": sha256_file(dpo_manifests[0]),
        "sft_shard_manifests": [
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": sha256_file(path),
            }
            for path in sft_manifests
        ],
        "dpo_shard_manifests": [
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": sha256_file(path),
            }
            for path in dpo_manifests
        ],
    }


def paired_boolean_record(sft_path: Path, dpo_path: Path, field: str) -> dict[str, Any]:
    sft = {str(row["id"]): row for row in load_jsonl(sft_path)}
    dpo = {str(row["id"]): row for row in load_jsonl(dpo_path)}
    if set(sft) != set(dpo):
        raise ValueError(f"paired ID mismatch: {sft_path} vs {dpo_path}")
    improved = sum(
        bool(dpo[row_id][field]) and not bool(sft[row_id][field])
        for row_id in sft
    )
    regressed = sum(
        bool(sft[row_id][field]) and not bool(dpo[row_id][field])
        for row_id in sft
    )
    return {
        "rows": len(sft),
        "dpo_correct_sft_wrong": improved,
        "sft_correct_dpo_wrong": regressed,
        "net_rows": improved - regressed,
        "exact_mcnemar_pvalue": exact_mcnemar_pvalue(improved, regressed),
    }


def protected_model(model: str, endpoint: str) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    for split in ("legacy6000", "api1000"):
        views: dict[str, Any] = {}
        for view in ("global", "rag"):
            dpo_dir = EXP_ROOT / "evaluation/protected" / model / endpoint / split
            sft_dir = (
                EXP_ROOT
                / "evaluation/protected"
                / model
                / CONTROLLED_PARENT_ENDPOINT
                / split
            )
            sft_metrics_path = sft_dir / f"{view}.metrics.json"
            dpo_metrics_path = dpo_dir / f"{view}.metrics.json"
            sft_metrics = metric_record(sft_metrics_path)
            dpo_metrics = metric_record(dpo_metrics_path)
            views[view] = {
                "sft": sft_metrics,
                "sft_dpo": dpo_metrics,
                "inference_control": controlled_inference_pair(
                    sft_metrics_path, dpo_metrics_path
                ),
                "delta": {
                    key: dpo_metrics[key] - sft_metrics[key]
                    for key in (
                        "set_em",
                        "trace_em",
                        "strict_output_valid_rate",
                        "micro_precision",
                        "micro_recall",
                        "micro_f1",
                        "omission_rate",
                        "over_call_rate",
                        "keyword_micro_f1",
                    )
                },
                "paired_trace": paired_boolean_record(
                    sft_dir / f"{view}.scored.jsonl",
                    dpo_dir / f"{view}.scored.jsonl",
                    "trace_em",
                ),
                "paired_set": paired_boolean_record(
                    sft_dir / f"{view}.scored.jsonl",
                    dpo_dir / f"{view}.scored.jsonl",
                    "set_em",
                ),
            }
        datasets[split] = views

    sft_system_path = (
        EXP_ROOT
        / "evaluation/protected"
        / model
        / CONTROLLED_PARENT_ENDPOINT
        / "legacy6000/rag.system.metrics.json"
    )
    dpo_system_path = (
        EXP_ROOT
        / "evaluation/protected"
        / model
        / endpoint
        / "legacy6000/rag.system.metrics.json"
    )
    sft_system = read_json(sft_system_path)
    dpo_system = read_json(dpo_system_path)
    sft_system_rows = sft_system_path.with_name("rag.system.jsonl")
    dpo_system_rows = dpo_system_path.with_name("rag.system.jsonl")
    return {
        "endpoint": endpoint,
        "controlled_parent_endpoint": CONTROLLED_PARENT_ENDPOINT,
        "datasets": datasets,
        "static_system_legacy9000": {
            "sft": {
                "path": str(sft_system_path.relative_to(REPO_ROOT)),
                "full_set": float(sft_system["full_pipeline_success"]),
                "full_trace": float(sft_system["full_pipeline_trace_success"]),
            },
            "sft_dpo": {
                "path": str(dpo_system_path.relative_to(REPO_ROOT)),
                "full_set": float(dpo_system["full_pipeline_success"]),
                "full_trace": float(dpo_system["full_pipeline_trace_success"]),
            },
            "delta_full_set": float(dpo_system["full_pipeline_success"])
            - float(sft_system["full_pipeline_success"]),
            "delta_full_trace": float(dpo_system["full_pipeline_trace_success"])
            - float(sft_system["full_pipeline_trace_success"]),
            "paired_full_trace": paired_boolean_record(
                sft_system_rows, dpo_system_rows, "full_pipeline_trace_success"
            ),
            "paired_full_set": paired_boolean_record(
                sft_system_rows, dpo_system_rows, "full_pipeline_success"
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", action="append", type=parse_endpoint, required=True)
    parser.add_argument("--out", type=Path, default=EXP_ROOT / "summary.json")
    args = parser.parse_args()
    endpoints = dict(args.endpoint)
    if set(endpoints) != {"4b", "8b"}:
        parser.error("exactly one 4b endpoint and one 8b endpoint are required")

    plan_audit = read_json(EXP_ROOT / "plan/data_plan_audit.json")
    finalized_inputs_audit = read_json(EXP_ROOT / "data/data_finalize_audit.json")
    if not plan_audit.get("strict_ok") or not finalized_inputs_audit.get("strict_ok"):
        raise ValueError("DPO data isolation audit failed")

    models: dict[str, Any] = {}
    for model in ("4b", "8b"):
        confirmation_path = EXP_ROOT / "selection" / f"{model}_confirmation.json"
        confirmation = read_json(confirmation_path)
        grid = read_json(EXP_ROOT / "selection" / f"{model}_grid.json")
        preference_audit = read_json(
            EXP_ROOT / "preferences" / model / "preference_audit.json"
        )
        token_audit = read_json(
            EXP_ROOT / "preferences" / model / "token_audit.json"
        )
        training_grid_audit = read_json(
            EXP_ROOT / "training" / f"{model}_grid_audit.json"
        )
        training_confirmation_audit = read_json(
            EXP_ROOT / "training" / f"{model}_confirmation_audit.json"
        )
        if not all(
            audit.get("strict_ok")
            for audit in (
                preference_audit,
                token_audit,
                training_grid_audit,
                training_confirmation_audit,
            )
        ):
            raise ValueError(f"DPO audit failed: {model}")
        parent_dev_rag = resolve_artifact_path(grid["parent_sft"]["rag_metrics"])
        selected_dev_rag = resolve_artifact_path(grid["selected"]["rag_metrics"])
        models[model] = {
            "preference_audit": preference_audit,
            "token_audit": token_audit,
            "training_grid_audit": training_grid_audit,
            "training_confirmation_audit": training_confirmation_audit,
            "single_seed_grid": grid,
            "development_selected_seed_paired": {
                "inference_control": controlled_inference_pair(
                    parent_dev_rag, selected_dev_rag
                ),
                "trace": paired_boolean_record(
                    parent_dev_rag.with_name("rag.scored.jsonl"),
                    selected_dev_rag.with_name("rag.scored.jsonl"),
                    "trace_em",
                ),
                "set": paired_boolean_record(
                    parent_dev_rag.with_name("rag.scored.jsonl"),
                    selected_dev_rag.with_name("rag.scored.jsonl"),
                    "set_em",
                ),
            },
            "three_seed_confirmation": confirmation,
            "promotion_pass": bool(confirmation["promotion_pass"]),
            "protected_report_only": protected_model(model, endpoints[model]),
        }

    output = {
        "schema_version": 1,
        "task": "assignment3_current_route_dpo",
        "status": "complete",
        "data_audits": {
            "plan": plan_audit,
            "finalized_inputs": finalized_inputs_audit,
        },
        "selection_boundary": (
            "DPO recipe and promotion are locked on the isolated 1400-row development set; "
            "Legacy 6000/API 1000 are report-only protected replays"
        ),
        "official_frozen_sets_used_for_training_or_selection": False,
        "models": models,
        "canonical_8b_recommendation": (
            "sft_dpo" if models["8b"]["promotion_pass"] else "parent_sft"
        ),
    }
    write_json(args.out, output)
    print(
        "current DPO summary complete "
        f"4b_promote={models['4b']['promotion_pass']} "
        f"8b_promote={models['8b']['promotion_pass']} -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
