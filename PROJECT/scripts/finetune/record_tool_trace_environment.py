#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Record immutable inputs, software, hardware, and model file identities."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import CATALOG_PATH, TOOL_TRACE_DIR, write_json  # noqa: E402
from finetune.tool_trace_common import ASSIGNMENT3_DIR, sha256_file, sha256_json  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
    except FileNotFoundError as exc:
        return {
            "command": command,
            "returncode": 127,
            "stdout": "",
            "stderr": str(exc),
        }
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def model_files(path: Path) -> dict[str, Any]:
    names = [
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "model.safetensors.index.json",
    ]
    hashes = {name: sha256_file(path / name) for name in names if (path / name).exists()}
    shard_names = sorted(file.name for file in path.glob("model-*.safetensors"))
    shard_sizes = {name: (path / name).stat().st_size for name in shard_names}
    revisions = set()
    metadata_dir = path / ".cache" / "huggingface" / "download"
    if metadata_dir.exists():
        for metadata_path in metadata_dir.glob("*.metadata"):
            lines = metadata_path.read_text(encoding="utf-8").splitlines()
            if lines:
                revisions.add(lines[0])
    return {
        "path": str(path),
        "exists": path.exists(),
        "huggingface_revision": next(iter(revisions)) if len(revisions) == 1 else None,
        "metadata_revisions": sorted(revisions),
        "metadata_hashes": hashes,
        "model_shards": shard_sizes,
        "identity_hash": sha256_json({"metadata": hashes, "shards": shard_sizes}),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ASSIGNMENT3_DIR / "environment_manifest.json")
    parser.add_argument("--model-4b", type=Path, default=PROJECT_ROOT / "models" / "Qwen" / "Qwen3-4B")
    parser.add_argument("--model-8b", type=Path, default=PROJECT_ROOT / "models" / "Qwen" / "Qwen3-8B")
    parser.add_argument(
        "--extra-input",
        action="append",
        type=Path,
        default=[],
        help="Additional immutable experiment input; repeat as needed.",
    )
    args = parser.parse_args()

    immutable_paths = [
        CATALOG_PATH,
        TOOL_TRACE_DIR / "train.jsonl",
        TOOL_TRACE_DIR / "test.jsonl",
        PROJECT_ROOT / "data" / "intent_recognition" / "intent_all.jsonl",
        PROJECT_ROOT
        / "data"
        / "rag"
        / "_aux"
        / "reranker_finetune"
        / "bge-reranker-base-e1-g8-bs16_rank_fusion_tuned_hybrid"
        / "preds_rank_fusion_w0p6_k10.jsonl",
        ASSIGNMENT3_DIR / "api_native_dev_2400" / "dev.jsonl",
        ASSIGNMENT3_DIR / "rag_train_rankings.jsonl",
        ASSIGNMENT3_DIR / "api_native_dev_2400" / "rag_rankings.jsonl",
    ]
    immutable_paths.extend(args.extra_input)
    missing = [str(path) for path in immutable_paths if not path.exists()]
    payload = {
        "status": "complete" if not missing and args.model_4b.exists() and args.model_8b.exists() else "incomplete",
        "cwd": str(Path.cwd()),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "platform": platform.platform(),
            "conda_prefix": os.environ.get("CONDA_PREFIX", ""),
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
        },
        "packages": {
            name: package_version(name)
            for name in [
                "torch",
                "transformers",
                "ms-swift",
                "peft",
                "trl",
                "vllm",
                "flash-attn",
                "datasets",
                "accelerate",
                "sentence-transformers",
                "FlagEmbedding",
                "faiss-cpu",
            ]
        },
        "git": {
            "commit": run(["git", "rev-parse", "HEAD"]),
            "status_short": run(["git", "status", "--short"]),
            "describe": run(["git", "describe", "--always", "--dirty"]),
        },
        "cuda": {
            "nvidia_smi": run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,driver_version",
                    "--format=csv,noheader",
                ]
            ),
            "nvcc": run(["nvcc", "--version"]),
        },
        "models": {
            "Qwen/Qwen3-4B": model_files(args.model_4b),
            "Qwen/Qwen3-8B": model_files(args.model_8b),
        },
        "immutable_input_sha256": {
            str(path): sha256_file(path) for path in immutable_paths if path.exists()
        },
        "missing_inputs": missing,
        "training_scripts": {
            str(path): sha256_file(path)
            for path in [
                PROJECT_ROOT / "scripts" / "finetune" / "run_swift_tool_trace_lora.sh",
                PROJECT_ROOT / "scripts" / "finetune" / "run_swift_tool_trace_dpo.sh",
                PROJECT_ROOT / "scripts" / "finetune" / "tool_trace_common.py",
                PROJECT_ROOT / "scripts" / "finetune" / "run_tool_trace_preference_mining.sh",
                PROJECT_ROOT / "scripts" / "finetune" / "prepare_tool_trace_sft_data.py",
                PROJECT_ROOT / "scripts" / "finetune" / "build_dpo_preferences.py",
                PROJECT_ROOT / "scripts" / "finetune" / "audit_tool_trace_dpo_tokens.py",
                PROJECT_ROOT / "scripts" / "finetune" / "audit_tool_trace_dpo_runs.py",
                PROJECT_ROOT / "scripts" / "finetune" / "generate_assignment3_api_native_dev.py",
                PROJECT_ROOT / "scripts" / "finetune" / "build_assignment3_100tool_plan.py",
                PROJECT_ROOT / "scripts" / "finetune" / "audit_assignment3_100tool_api_data.py",
                PROJECT_ROOT / "scripts" / "finetune" / "prepare_assignment3_100tool_sft.py",
                PROJECT_ROOT / "scripts" / "finetune" / "audit_swift_tool_trace_labels.py",
                PROJECT_ROOT / "scripts" / "finetune" / "audit_tool_trace_sft_runs.py",
                PROJECT_ROOT / "scripts" / "finetune" / "audit_tool_trace_dpo_runs.py",
                PROJECT_ROOT / "scripts" / "finetune" / "build_assignment3_order_counterfactual_plan.py",
                PROJECT_ROOT / "scripts" / "finetune" / "generate_assignment3_order_counterfactual.py",
                PROJECT_ROOT / "scripts" / "finetune" / "audit_assignment3_order_counterfactual.py",
                PROJECT_ROOT / "scripts" / "finetune" / "evaluate_order_counterfactual_pairs.py",
                PROJECT_ROOT / "scripts" / "finetune" / "analyze_assignment3_100tool_candidate_order.py",
                PROJECT_ROOT / "scripts" / "finetune" / "select_assignment3_100tool_checkpoint.py",
                PROJECT_ROOT / "scripts" / "finetune" / "run_assignment3_100tool_dev_selection.sh",
                PROJECT_ROOT / "scripts" / "finetune" / "run_assignment3_100tool_eval_stage.sh",
                PROJECT_ROOT / "scripts" / "finetune" / "run_assignment3_100tool_candidate_order_eval.sh",
                PROJECT_ROOT / "scripts" / "finetune" / "run_assignment3_100tool_order_eval.sh",
                PROJECT_ROOT / "scripts" / "finetune" / "run_assignment3_100tool_dpo_mining.sh",
                PROJECT_ROOT / "scripts" / "finetune" / "run_assignment3_100tool_dpo_grid.sh",
                PROJECT_ROOT / "scripts" / "finetune" / "sanitize_tool_trace_eval_inputs.py",
                PROJECT_ROOT / "scripts" / "finetune" / "manage_tool_trace_experiments.py",
                PROJECT_ROOT / "scripts" / "finetune" / "run_tool_trace_frozen_matrix.py",
                PROJECT_ROOT / "scripts" / "finetune" / "build_tool_trace_rag_candidates.py",
                PROJECT_ROOT / "scripts" / "finetune" / "summarize_tool_trace_latency.py",
                PROJECT_ROOT / "scripts" / "finetune" / "infer_tool_trace.py",
                PROJECT_ROOT / "scripts" / "finetune" / "evaluate_tool_trace.py",
                PROJECT_ROOT / "scripts" / "finetune" / "evaluate_tool_trace_system.py",
                PROJECT_ROOT / "scripts" / "finetune" / "run_tool_trace_system_eval.py",
                PROJECT_ROOT / "scripts" / "rag" / "run_retrieval_baselines.py",
                PROJECT_ROOT / "scripts" / "report" / "make_assignment3_100tool_complete_report.py",
                PROJECT_ROOT / "scripts" / "report" / "make_tool_trace_training_report.py",
            ]
        },
    }
    write_json(args.out, payload)
    print(f"environment status={payload['status']} -> {args.out}")
    return 0 if payload["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
