#!/usr/bin/env python3
"""Validate and summarize the current Planner DPO2 endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
EXP = (
    ROOT
    / "PROJECT/data/tool_call_trace/_aux/assignment3"
    / "assignment3_planner_dpo2_20260808"
)
PARENT_EXP = (
    ROOT
    / "PROJECT/data/tool_call_trace/_aux/assignment3"
    / "assignment3_crossstyle_simple_20260808"
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def require(path: Path, *, strict: bool = False) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing required artifact: {path}")
    value = read_json(path)
    if value.get("status") not in (None, "complete"):
        raise ValueError(f"incomplete artifact: {path}")
    if strict and value.get("strict_ok") is not True:
        raise ValueError(f"non-strict artifact: {path}")
    return value


def evidence(path: Path) -> dict[str, str]:
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}


def tool_metrics(path: Path) -> dict[str, Any]:
    value = require(path)
    overall = value["overall"]
    rag = value["rag_decomposition"]
    return {
        **evidence(path),
        "count": value["count"],
        "set_em": overall["set_em"],
        "trace_em": overall["trace_em"],
        "strict_output_valid_rate": overall["strict_output_valid_rate"],
        "retriever_upper_bound": rag["retriever_upper_bound"],
        "conditional_set_em": rag["conditional_set_accuracy"],
        "conditional_trace_em": rag["conditional_trace_accuracy"],
        "predictions": value["predictions"],
        "predictions_sha256": value["predictions_sha256"],
    }


def pp(new: float, old: float) -> float:
    return 100.0 * (new - old)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=EXP / "final_summary.json")
    args = parser.parse_args()

    audit_paths = {
        "preference": EXP / "preferences/preference_audit.json",
        "provenance": EXP / "audits/preference_provenance_audit.json",
        "isolation": EXP / "audits/preference_isolation_audit.json",
        "token_length": EXP / "audits/preference_token_audit.json",
        "vs_dev_leakage": EXP / "audits/preference_vs_dev_leakage.json",
        "vs_frozen_leakage": EXP / "audits/preference_vs_frozen_leakage.json",
        "main_training": EXP / "audits/training_runs_audit.json",
        "extension_training": EXP / "audits/extension_training_runs_audit.json",
    }
    audits = {name: require(path, strict=True) for name, path in audit_paths.items()}

    selection_path = EXP / "selection/dpo_checkpoint_selection.json"
    selection = require(selection_path)
    if selection.get("official_frozen_tests_used") is not False:
        raise ValueError("development selection used a frozen test")
    selected_adapter = Path(str(selection["selected"]["adapter"]))
    if not selected_adapter.is_absolute():
        selected_adapter = ROOT / selected_adapter
    selected_adapter = selected_adapter.resolve()
    if not selected_adapter.is_dir():
        raise ValueError("selected adapter is missing")

    legacy_path = EXP / "evaluation/frozen/legacy6000/selected/rag.metrics.json"
    api_path = EXP / "evaluation/frozen/api1000/selected/rag.metrics.json"
    system_path = EXP / "evaluation/system9000/selected/system.metrics.json"
    legacy = tool_metrics(legacy_path)
    api = tool_metrics(api_path)
    system = require(system_path)
    if legacy["count"] != 6000 or api["count"] != 1000 or system["count"] != 9000:
        raise ValueError("unexpected final evaluation count")
    if system["tool_generator"]["adapters"] != [str(selection["selected"]["adapter"])]:
        raise ValueError("system predictions do not use the selected adapter")

    parent_legacy_path = (
        PARENT_EXP / "evaluation/frozen/legacy6000/dpo_step29/rag.metrics.json"
    )
    parent_api_path = PARENT_EXP / "evaluation/frozen/api1000/dpo_step29/rag.metrics.json"
    parent_system_path = (
        PARENT_EXP / "evaluation/system9000/dpo_step29/system.metrics.json"
    )
    sft_legacy_path = PARENT_EXP / "evaluation/frozen/legacy6000/sft/rag.metrics.json"
    sft_system_path = PARENT_EXP / "evaluation/system9000/sft/system.metrics.json"
    parent_legacy = tool_metrics(parent_legacy_path)
    parent_api = tool_metrics(parent_api_path)
    parent_system = require(parent_system_path)
    sft_legacy = tool_metrics(sft_legacy_path)
    sft_system = require(sft_system_path)

    paired_paths = {
        "dev_vs_parent": EXP / "selection/selected_vs_parent_dev.json",
        "legacy_vs_parent": (
            EXP / "evaluation/frozen/legacy6000/selected_vs_dpo1_parent.json"
        ),
        "legacy_vs_sft": (
            EXP / "evaluation/frozen/legacy6000/selected_vs_sft_parent.json"
        ),
        "api_vs_parent": (
            EXP / "evaluation/frozen/api1000/selected_vs_dpo1_parent.json"
        ),
        "system_vs_parent": EXP / "evaluation/system9000/selected_vs_dpo1.json",
    }
    paired = {name: require(path) for name, path in paired_paths.items()}

    summary = {
        "schema_version": 1,
        "status": "complete",
        "strict_ok": True,
        "date": "2026-08-08",
        "experiment_id": "assignment3_planner_dpo2_20260808",
        "scope": "fixed canonical Top-10 RAG Planner on the synthetic 100-tool closed set",
        "selected_endpoint": {
            **selection["selected"],
            "adapter_sha256": sha256_file(selected_adapter / "adapter_model.safetensors"),
            "selection": evidence(selection_path),
            "candidate_count": len(selection["ranked_dpo_candidates"]),
            "frozen_tests_used_for_selection": False,
        },
        "preference_data": {
            "pairs": audits["preference"]["counts"]["final_pairs"],
            "train_pairs": audits["preference"]["counts"]["train_pairs"],
            "monitor_pairs": audits["preference"]["counts"]["monitor_pairs"],
            "by_tool_count": audits["preference"]["counts"]["by_tool_count"],
            "primary_error_types": audits["preference"]["counts"]["primary_error_types"],
            "represented_tools": audits["preference"]["counts"]["represented_tools"],
            "all_original_candidate_order": (
                audits["preference"]["counts"]["by_candidate_order"]
                == {"original": 4000}
            ),
        },
        "development": {
            "parent_trace_em": selection["parent"]["trace_em"],
            "selected_trace_em": selection["selected"]["trace_em"],
            "delta_percentage_points": pp(
                selection["selected"]["trace_em"], selection["parent"]["trace_em"]
            ),
            "paired": paired["dev_vs_parent"]["paired"]["trace_em"],
        },
        "frozen_evaluation": {
            "legacy6000": {
                "selected": legacy,
                "parent_dpo1": parent_legacy,
                "sft_parent": sft_legacy,
                "delta_from_parent_dpo1_percentage_points": pp(
                    legacy["trace_em"], parent_legacy["trace_em"]
                ),
                "delta_from_sft_percentage_points": pp(
                    legacy["trace_em"], sft_legacy["trace_em"]
                ),
                "paired_vs_parent_dpo1": paired["legacy_vs_parent"]["paired"]["trace_em"],
            },
            "api1000": {
                "selected": api,
                "parent_dpo1": parent_api,
                "delta_from_parent_dpo1_percentage_points": pp(
                    api["trace_em"], parent_api["trace_em"]
                ),
                "paired_vs_parent_dpo1": paired["api_vs_parent"]["paired"]["trace_em"],
            },
            "system9000": {
                **evidence(system_path),
                "full_set": system["full_pipeline_success"],
                "full_trace": system["full_pipeline_trace_success"],
                "full_set_count": system["full_pipeline_success_count"],
                "full_trace_count": system["full_pipeline_trace_success_count"],
                "parent_dpo1_full_trace": parent_system[
                    "full_pipeline_trace_success"
                ],
                "sft_full_trace": sft_system["full_pipeline_trace_success"],
                "delta_from_parent_dpo1_percentage_points": pp(
                    system["full_pipeline_trace_success"],
                    parent_system["full_pipeline_trace_success"],
                ),
                "delta_from_sft_percentage_points": pp(
                    system["full_pipeline_trace_success"],
                    sft_system["full_pipeline_trace_success"],
                ),
                "paired_vs_parent_dpo1": paired["system_vs_parent"]["paired"][
                    "trace_em"
                ],
                "is_real_tool_execution_success": system[
                    "is_real_tool_execution_success"
                ],
            },
        },
        "audits": {
            name: {**evidence(path), "strict_ok": audits[name]["strict_ok"]}
            for name, path in audit_paths.items()
        },
        "limitations": [
            "Frozen Legacy/API data was not used for preference construction or checkpoint selection.",
            "API frozen changed by one net regression (1 improvement, 2 regressions); it was not used to revise the locked endpoint.",
            "Full Trace is a static routing-plus-tool-selection metric, not real tool execution success.",
            "Results apply to the synthetic 100-tool closed set.",
        ],
    }
    write_json(args.out, summary)
    print(
        f"Planner DPO2 final Legacy={legacy['trace_em']:.6f} "
        f"API={api['trace_em']:.6f} "
        f"FullTrace={system['full_pipeline_trace_success']:.6f} -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
