#!/usr/bin/env python3
"""Attach the completed cross-style DPO stage to the traceable SFT summary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXP = (
    ROOT
    / "PROJECT/data/tool_call_trace/_aux/assignment3"
    / "assignment3_crossstyle_rag10"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root-summary",
        type=Path,
        default=DEFAULT_EXP / "final_summary.json",
    )
    parser.add_argument(
        "--dpo-dir",
        type=Path,
        default=DEFAULT_EXP / "dpo",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_EXP / "dpo/final_summary.json",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_complete(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing required artifact: {path}")
    value = read_json(path)
    if value.get("status") != "complete":
        raise ValueError(f"incomplete required artifact: {path}")
    return value


def pp_delta(new: float, old: float) -> float:
    return (new - old) * 100.0


def compact_tool_metrics(
    path: Path,
    parent: dict[str, Any],
) -> dict[str, Any]:
    metrics = require_complete(path)
    overall = metrics["overall"]
    rag = metrics["rag_decomposition"]
    result = {
        "metrics": str(path),
        "metrics_sha256": sha256_file(path),
        "predictions": metrics["predictions"],
        "predictions_sha256": metrics["predictions_sha256"],
        "count": metrics["count"],
        "set_em": overall["set_em"],
        "trace_em": overall["trace_em"],
        "strict_output_valid_rate": overall["strict_output_valid_rate"],
        "retrieval_all_hit_at_10": rag["retriever_upper_bound"],
        "retrieval_success_count": rag["retrieval_success_count"],
        "retrieval_fail_count": rag["retrieval_fail_count"],
        "conditional_set_accuracy": rag["conditional_set_accuracy"],
        "conditional_trace_accuracy": rag["conditional_trace_accuracy"],
        "by_tool_count": {
            key: {
                "count": value["count"],
                "set_em": value["set_em"],
                "trace_em": value["trace_em"],
            }
            for key, value in metrics["buckets"]["tool_count"].items()
        },
        "parent_sft": {
            "set_em": parent["set_em"],
            "trace_em": parent["trace_em"],
            "strict_output_valid_rate": parent["strict_output_valid_rate"],
            "conditional_set_accuracy": parent["conditional_set_accuracy"],
            "conditional_trace_accuracy": parent["conditional_trace_accuracy"],
        },
        "delta_from_parent_sft_percentage_points": {
            "set_em": pp_delta(overall["set_em"], parent["set_em"]),
            "trace_em": pp_delta(overall["trace_em"], parent["trace_em"]),
            "strict_output_valid_rate": pp_delta(
                overall["strict_output_valid_rate"],
                parent["strict_output_valid_rate"],
            ),
            "conditional_set_accuracy": pp_delta(
                rag["conditional_set_accuracy"],
                parent["conditional_set_accuracy"],
            ),
            "conditional_trace_accuracy": pp_delta(
                rag["conditional_trace_accuracy"],
                parent["conditional_trace_accuracy"],
            ),
        },
    }
    return result


def compact_system_metrics(
    path: Path,
    parent: dict[str, Any],
) -> dict[str, Any]:
    metrics = require_complete(path)
    return {
        "metrics": str(path),
        "metrics_sha256": sha256_file(path),
        "count": metrics["count"],
        "route_accuracy": metrics["route_accuracy"],
        "ecommerce_set_em": metrics["ecommerce_set_em"],
        "ecommerce_trace_em": metrics["ecommerce_trace_em"],
        "full_pipeline_success": metrics["full_pipeline_success"],
        "full_pipeline_trace_success": metrics["full_pipeline_trace_success"],
        "full_pipeline_success_count": metrics["full_pipeline_success_count"],
        "full_pipeline_trace_success_count": metrics[
            "full_pipeline_trace_success_count"
        ],
        "delta_from_parent_sft_percentage_points": {
            "ecommerce_set_em": pp_delta(
                metrics["ecommerce_set_em"],
                parent["ecommerce_set_em"],
            ),
            "ecommerce_trace_em": pp_delta(
                metrics["ecommerce_trace_em"],
                parent["ecommerce_trace_em"],
            ),
            "full_pipeline_success": pp_delta(
                metrics["full_pipeline_success"],
                parent["full_pipeline_success"],
            ),
            "full_pipeline_trace_success": pp_delta(
                metrics["full_pipeline_trace_success"],
                parent["full_pipeline_trace_success"],
            ),
        },
        "is_real_tool_execution_success": metrics[
            "is_real_tool_execution_success"
        ],
    }


def main() -> int:
    args = parse_args()
    root_summary = require_complete(args.root_summary)
    preference_audit_path = args.dpo_dir / "preference_audit.json"
    token_audit_path = args.dpo_dir / "token_length_audit.json"
    run_audit_path = args.dpo_dir / "training_runs_audit.json"
    selection_paths = {
        "qwen3_4b": args.dpo_dir / "selection_4b.json",
        "qwen3_8b": args.dpo_dir / "selection_8b.json",
    }
    preference_audit = require_complete(preference_audit_path)
    token_audit = require_complete(token_audit_path)
    run_audit = require_complete(run_audit_path)
    selections = {
        model: require_complete(path)
        for model, path in selection_paths.items()
    }
    if not preference_audit.get("strict_ok"):
        raise ValueError("preference audit is not strict")
    if not token_audit.get("strict_ok") or not run_audit.get("strict_ok"):
        raise ValueError("token or training-run audit is not strict")
    if any(selection["official_frozen_tests_used"] for selection in selections.values()):
        raise ValueError("selection used a frozen test")

    eval_dir = args.dpo_dir / "eval"
    api_parent = root_summary["evaluation"]["api_native_1000"]["models"]
    legacy_parent = root_summary["evaluation"]["legacy_frozen_6000"]["models"]
    system_parent = root_summary["static_system_legacy_9000"]["models"]
    evaluation = {
        "api_native_1000": {
            "independent_protocol": True,
            "models": {
                "qwen3_4b": compact_tool_metrics(
                    eval_dir / "selected-dpo4-api1000.metrics.json",
                    api_parent["qwen3_4b"],
                ),
                "qwen3_8b": compact_tool_metrics(
                    eval_dir / "selected-dpo8-api1000.metrics.json",
                    api_parent["qwen3_8b"],
                ),
            },
        },
        "legacy_frozen_6000": {
            "independent_protocol": True,
            "models": {
                "qwen3_4b": compact_tool_metrics(
                    eval_dir / "selected-dpo4-legacy6000.metrics.json",
                    legacy_parent["qwen3_4b"],
                ),
                "qwen3_8b": compact_tool_metrics(
                    eval_dir / "selected-dpo8-legacy6000.metrics.json",
                    legacy_parent["qwen3_8b"],
                ),
            },
        },
    }
    static_system = {
        "qwen3_4b": compact_system_metrics(
            eval_dir / "selected-dpo4-legacy6000.system.metrics.json",
            system_parent["qwen3_4b"],
        ),
        "qwen3_8b": compact_system_metrics(
            eval_dir / "selected-dpo8-legacy6000.system.metrics.json",
            system_parent["qwen3_8b"],
        ),
    }
    extension = {
        "status": "complete",
        "date": "2026-07-29",
        "experiment_id": "assignment3_crossstyle_rag10_dpo",
        "protocol": {
            "path": "PROJECT/configs/assignment3_crossstyle_dpo.json",
            "sha256": sha256_file(
                ROOT / "PROJECT/configs/assignment3_crossstyle_dpo.json"
            ),
            "frozen_tests_used_for_preference_or_selection": False,
            "frozen_tests_reused_once_after_selection": True,
            "post_frozen_tuning_allowed": False,
        },
        "preference_data": {
            "audit": str(preference_audit_path),
            "audit_sha256": sha256_file(preference_audit_path),
            "strict_ok": preference_audit["strict_ok"],
            **preference_audit["counts"],
            "output_hashes": preference_audit["hashes"]["outputs"],
        },
        "token_audit": {
            "path": str(token_audit_path),
            "sha256": sha256_file(token_audit_path),
            "strict_ok": token_audit["strict_ok"],
            "max_length": 768,
        },
        "training": {
            "run_audit": str(run_audit_path),
            "run_audit_sha256": sha256_file(run_audit_path),
            "runs": run_audit["runs"],
        },
        "development_selection": {
            model: {
                "path": str(selection_paths[model]),
                "sha256": sha256_file(selection_paths[model]),
                **selections[model],
            }
            for model in selections
        },
        "evaluation": evaluation,
        "static_system_legacy_9000": {
            "definition": (
                "chitchat route correctness; ecommerce route correctness plus "
                "tool SetEM or ordered TraceEM"
            ),
            "models": static_system,
            "is_real_tool_execution_success": False,
        },
        "limitations": [
            "The preference set and checkpoint selection exclude both frozen test protocols.",
            "The two frozen sets had already been observed for the parent-SFT report; this DPO extension is a preregistered one-shot reuse, not a pristine first-look test.",
            "No DPO hyperparameter, checkpoint, or data rule may change after these frozen results.",
            "API-native 1k and Legacy frozen 6k are independent and are not pooled.",
            "The scope is a synthetic 100-tool closed set and not live tool execution.",
        ],
    }
    write_json(args.out, extension)
    root_summary["dpo_extension"] = extension
    write_json(args.root_summary, root_summary)
    print(f"DPO final summary -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
