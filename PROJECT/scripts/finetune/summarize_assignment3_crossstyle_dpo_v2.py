#!/usr/bin/env python3
"""Build the traceable Assignment-3 cross-style DPO-v2 final summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import PROJECT_ROOT, read_json, write_json  # noqa: E402
from finetune.tool_trace_common import sha256_file  # noqa: E402


REPO_ROOT = PROJECT_ROOT.parent
CROSSSTYLE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "tool_call_trace"
    / "_aux"
    / "assignment3"
    / "assignment3_crossstyle_rag10"
)
DPO_ROOT = CROSSSTYLE_ROOT / "dpo_v2"
ORDERS = ("original", "reverse", "hash_shuffle")


def repo_path(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def artifact(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": repo_path(path), "sha256": sha256_file(path)}


def require_complete(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict) or value.get("status") != "complete":
        raise ValueError(f"incomplete JSON artifact: {path}")
    return value


def metric_snapshot(path: Path) -> dict[str, Any]:
    value = require_complete(path)
    overall = value["overall"]
    rag = value["rag_decomposition"]
    return {
        **artifact(path),
        "count": int(value["count"]),
        "overall": {
            key: overall[key]
            for key in (
                "set_em",
                "trace_em",
                "strict_output_valid_rate",
                "json_valid_rate",
                "json_schema_valid_rate",
            )
        },
        "rag_decomposition": {
            key: rag[key]
            for key in (
                "retriever_upper_bound",
                "retrieval_success_count",
                "retrieval_fail_count",
                "conditional_set_accuracy",
                "conditional_trace_accuracy",
            )
        },
        "buckets": {
            key: value["buckets"][key]
            for key in ("style", "tool_count", "single_multi")
            if key in value["buckets"]
        },
    }


def prediction_snapshot(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        rows = sum(1 for line in handle if line.strip())
    return {**artifact(path), "rows": rows}


def order_endpoint(
    prediction_dir: Path,
    metrics_dir: Path,
    prefix: str,
) -> dict[str, Any]:
    orders: dict[str, Any] = {}
    for order in ORDERS:
        stem = f"{prefix}-{order}"
        orders[order] = {
            "predictions": prediction_snapshot(prediction_dir / f"{stem}.jsonl"),
            "metrics": metric_snapshot(metrics_dir / f"{stem}.metrics.json"),
        }
    macro_keys = (
        "set_em",
        "trace_em",
        "strict_output_valid_rate",
    )
    macro = {
        key: sum(
            float(orders[order]["metrics"]["overall"][key]) for order in ORDERS
        )
        / len(ORDERS)
        for key in macro_keys
    }
    macro["conditional_set_accuracy"] = sum(
        float(
            orders[order]["metrics"]["rag_decomposition"][
                "conditional_set_accuracy"
            ]
        )
        for order in ORDERS
    ) / len(ORDERS)
    macro["conditional_trace_accuracy"] = sum(
        float(
            orders[order]["metrics"]["rag_decomposition"][
                "conditional_trace_accuracy"
            ]
        )
        for order in ORDERS
    ) / len(ORDERS)
    return {
        "orders": orders,
        "three_order_macro": macro,
        "worst_order_trace_em": min(
            float(orders[order]["metrics"]["overall"]["trace_em"])
            for order in ORDERS
        ),
    }


def endpoint_delta(current: dict[str, Any], parent: dict[str, Any]) -> dict[str, Any]:
    macro = {
        key: float(current["three_order_macro"][key])
        - float(parent["three_order_macro"][key])
        for key in current["three_order_macro"]
    }
    orders: dict[str, Any] = {}
    for order in ORDERS:
        current_metrics = current["orders"][order]["metrics"]
        parent_metrics = parent["orders"][order]["metrics"]
        orders[order] = {
            key: float(current_metrics["overall"][key])
            - float(parent_metrics["overall"][key])
            for key in (
                "set_em",
                "trace_em",
                "strict_output_valid_rate",
            )
        }
        orders[order]["conditional_set_accuracy"] = (
            float(
                current_metrics["rag_decomposition"][
                    "conditional_set_accuracy"
                ]
            )
            - float(
                parent_metrics["rag_decomposition"][
                    "conditional_set_accuracy"
                ]
            )
        )
        orders[order]["conditional_trace_accuracy"] = (
            float(
                current_metrics["rag_decomposition"][
                    "conditional_trace_accuracy"
                ]
            )
            - float(
                parent_metrics["rag_decomposition"][
                    "conditional_trace_accuracy"
                ]
            )
        )
    return {
        "three_order_macro": macro,
        "worst_order_trace_em": float(current["worst_order_trace_em"])
        - float(parent["worst_order_trace_em"]),
        "orders": orders,
    }


def single_metric_comparison(
    parent_path: Path,
    current_path: Path,
) -> dict[str, Any]:
    parent = metric_snapshot(parent_path)
    current = metric_snapshot(current_path)
    return {
        "parent_sft": parent,
        "selected_dpo_v2": current,
        "delta": {
            "set_em": float(current["overall"]["set_em"])
            - float(parent["overall"]["set_em"]),
            "trace_em": float(current["overall"]["trace_em"])
            - float(parent["overall"]["trace_em"]),
            "strict_output_valid_rate": float(
                current["overall"]["strict_output_valid_rate"]
            )
            - float(parent["overall"]["strict_output_valid_rate"]),
            "conditional_set_accuracy": float(
                current["rag_decomposition"]["conditional_set_accuracy"]
            )
            - float(parent["rag_decomposition"]["conditional_set_accuracy"]),
            "conditional_trace_accuracy": float(
                current["rag_decomposition"]["conditional_trace_accuracy"]
            )
            - float(parent["rag_decomposition"]["conditional_trace_accuracy"]),
        },
    }


def system_comparison(parent_path: Path, current_path: Path) -> dict[str, Any]:
    parent = require_complete(parent_path)
    current = require_complete(current_path)
    keys = (
        "route_accuracy",
        "ecommerce_set_em",
        "ecommerce_trace_em",
        "full_pipeline_success",
        "full_pipeline_trace_success",
    )
    return {
        "parent_sft": {
            **artifact(parent_path),
            **{key: parent[key] for key in keys},
        },
        "selected_dpo_v2": {
            **artifact(current_path),
            **{key: current[key] for key in keys},
        },
        "delta": {
            key: float(current[key]) - float(parent[key]) for key in keys
        },
    }


def training_snapshot() -> dict[str, Any]:
    audit_path = DPO_ROOT / "training_runs_audit.json"
    audit = require_complete(audit_path)
    if not audit.get("strict_ok"):
        raise ValueError("DPO-v2 training run audit failed")
    runs: dict[str, Any] = {}
    for name, row in audit["runs"].items():
        log_path = resolve_path(row["logging"])
        log_rows = [
            json.loads(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        train_result = next(
            (
                item
                for item in reversed(log_rows)
                if "train_runtime" in item and "train_loss" in item
            ),
            None,
        )
        if train_result is None:
            raise ValueError(f"missing train result: {name}")
        endpoint = resolve_path(row["endpoint"])
        runs[name] = {
            "strict_ok": row["strict_ok"],
            "beta": row["beta"],
            "learning_rate": row["learning_rate"],
            "epochs": row["num_train_epochs"],
            "endpoint": repo_path(endpoint),
            "adapter_model_sha256": sha256_file(
                endpoint / "adapter_model.safetensors"
            ),
            "train_runtime_seconds": train_result["train_runtime"],
            "train_loss": train_result["train_loss"],
            "global_step": int(train_result["global_step/max_steps"].split("/")[0]),
            "eval_loss": row["metrics"]["eval_loss"]["last"],
            "eval_reward_accuracy": row["metrics"][
                "eval_rewards/accuracies"
            ]["last"],
            "eval_reward_margin": row["metrics"]["eval_rewards/margins"]["last"],
            "args": artifact(resolve_path(row["args"])),
            "logging": artifact(log_path),
        }
    return {"audit": artifact(audit_path), "strict_ok": True, "runs": runs}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DPO_ROOT / "final_summary.json")
    parser.add_argument(
        "--root-summary",
        type=Path,
        default=CROSSSTYLE_ROOT / "final_summary.json",
    )
    parser.add_argument("--update-root-summary", action="store_true")
    args = parser.parse_args()

    config_path = PROJECT_ROOT / "configs" / "assignment3_crossstyle_dpo_v2.json"
    preference_path = DPO_ROOT / "preference_audit.json"
    preference = read_json(preference_path)
    if not preference.get("strict_ok"):
        raise ValueError("DPO-v2 preference audit failed")
    isolation_path = DPO_ROOT / "posthoc_isolation_audit.json"
    isolation = read_json(isolation_path)
    if not isolation.get("strict_ok"):
        raise ValueError("DPO-v2 isolation audit failed")

    selections = {}
    for model in ("4b", "8b"):
        path = DPO_ROOT / f"selection_{model}.json"
        value = require_complete(path)
        if (
            value["selected_dpo"]["name"] != f"dpo2-{model}-lr1e6"
            or not value["selected_dpo"]["guardrails"]["pass"]
        ):
            raise ValueError(f"unexpected selected endpoint: {model}")
        selections[model] = {"artifact": artifact(path), "result": value}

    final_results = {}
    for model in ("4b", "8b"):
        parent = order_endpoint(
            DPO_ROOT / "final_predictions",
            DPO_ROOT / "final_metrics",
            f"final-parent-sft{model[0]}",
        )
        selected = order_endpoint(
            DPO_ROOT / "final_predictions",
            DPO_ROOT / "final_metrics",
            f"final-selected-dpo2-{model}",
        )
        final_results[model] = {
            "parent_sft": parent,
            "selected_dpo_v2": selected,
            "selected_minus_parent": endpoint_delta(selected, parent),
        }

    historical_metrics = DPO_ROOT / "historical_metrics"
    parent_eval = CROSSSTYLE_ROOT / "eval"
    historical = {
        "role": (
            "non-selection regression audit; these frozen results had already "
            "been observed before DPO-v2"
        ),
        "api_native_1000": {},
        "legacy_frozen_6000": {},
        "static_system_legacy_9000": {},
    }
    for model in ("4b", "8b"):
        historical["api_native_1000"][model] = single_metric_comparison(
            parent_eval / f"a3-crossstyle-{model}-api1000.metrics.json",
            historical_metrics
            / f"historical-selected-dpo2-{model}-api1000.metrics.json",
        )
        historical["legacy_frozen_6000"][model] = single_metric_comparison(
            parent_eval / f"a3-crossstyle-{model}-legacy6000.metrics.json",
            historical_metrics
            / f"historical-selected-dpo2-{model}-legacy6000.metrics.json",
        )
        historical["static_system_legacy_9000"][model] = system_comparison(
            parent_eval
            / f"a3-crossstyle-{model}-legacy6000.system.metrics.json",
            historical_metrics
            / f"historical-selected-dpo2-{model}-legacy6000.system.metrics.json",
        )

    plan_audit = DPO_ROOT / "eval_protocol" / "plan_audit.json"
    data_audit = DPO_ROOT / "eval_protocol" / "data_audit.json"
    dev_retrieval = (
        DPO_ROOT / "eval_protocol" / "dev_retrieval_manifest.json"
    )
    final_retrieval = (
        DPO_ROOT / "eval_protocol" / "final_retrieval_manifest.json"
    )
    plan_value = read_json(plan_audit)
    data_value = read_json(data_audit)
    if not plan_value.get("strict_ok") or not data_value.get("strict_ok"):
        raise ValueError("sealed evaluation audit failed")

    summary = {
        "status": "complete",
        "date": "2026-07-29",
        "experiment_id": "assignment3_crossstyle_rag10_dpo_v2",
        "redo_reason": (
            "DPO-v1 used mostly old/early-model negatives and lr=5e-6, then "
            "regressed Legacy frozen 6k; v2 rebuilds evaluation, current-parent "
            "preferences, training, selection, and final evaluation."
        ),
        "scope": {
            "task": "100-tool synthetic closed-set static RAG@10 planning",
            "is_real_tool_execution_success": False,
            "cross_protocol_pooling": False,
            "old_dpo_v1_preserved_as_failed_baseline": True,
            "crossstyle_sft_parents_preserved": True,
        },
        "protocol": {
            "config": artifact(config_path),
            "sealed_plan_audit": artifact(plan_audit),
            "sealed_data_audit": artifact(data_audit),
            "dev_rows": 1200,
            "final_rows": 1200,
            "styles_per_split": {"api_native": 600, "legacy_compact": 600},
            "tool_count_per_style": {str(value): 100 for value in range(1, 7)},
            "source_trace_overlap_across_four_cells": 0,
            "training_use_allowed": False,
            "dev_use": "selection and guardrails only",
            "final_use": "one-shot evaluation only after selection",
            "historical_frozen_use": "post-selection regression audit only",
        },
        "retrieval": {
            "dev": {
                **artifact(dev_retrieval),
                "all_hit_at_10": read_json(dev_retrieval)["metrics"][
                    "allhit_at_10"
                ],
            },
            "final": {
                **artifact(final_retrieval),
                "all_hit_at_10": read_json(final_retrieval)["metrics"][
                    "allhit_at_10"
                ],
            },
        },
        "preference_data": {
            "audit": artifact(preference_path),
            "strict_ok": True,
            "protocol": preference["protocol"],
            "counts": preference["counts"],
            "isolation": preference["isolation"],
            "integrity": preference["integrity"],
            "posthoc_isolation": {
                **artifact(isolation_path),
                "strict_ok": True,
                "counts": isolation["counts"],
                "overlap": isolation["overlap"],
            },
            "token_length_audit": artifact(
                DPO_ROOT / "token_length_audit.json"
            ),
            "label_audits": {
                "4b": artifact(DPO_ROOT / "label_audit_4b.json"),
                "8b": artifact(DPO_ROOT / "label_audit_8b.json"),
            },
        },
        "training": training_snapshot(),
        "selection": {
            "split": "sealed_crossstyle_dev_v2",
            "final_used": False,
            "historical_frozen_used": False,
            "models": selections,
        },
        "sealed_final": {
            "decoder": {
                "view": "rag",
                "candidate_orders": list(ORDERS),
                "do_sample": False,
                "max_new_tokens": 128,
                "seed": 20260725,
            },
            "retrieval_all_hit_at_10": read_json(final_retrieval)["metrics"][
                "allhit_at_10"
            ],
            "models": final_results,
            "post_final_tuning_or_reselection": False,
        },
        "historical_regression_audit": historical,
        "conclusion": {
            "selected_learning_rate_4b": 1e-6,
            "selected_learning_rate_8b": 1e-6,
            "beta": 0.1,
            "sealed_final_original_order_trace_preserved": {
                model: final_results[model]["selected_minus_parent"]["orders"][
                    "original"
                ]["trace_em"]
                == 0.0
                for model in ("4b", "8b")
            },
            "sealed_final_three_order_macro_trace_improved": {
                model: final_results[model]["selected_minus_parent"][
                    "three_order_macro"
                ]["trace_em"]
                > 0.0
                for model in ("4b", "8b")
            },
            "no_further_checkpoint_selection_allowed": True,
        },
        "limitations": [
            "The sealed dev and final are synthetic 100-tool closed-set protocols.",
            "API-native 1k, Legacy frozen 6k, and the new sealed final are independent protocols and are not pooled.",
            "Historical frozen results are regression audits only and cannot trigger another tuning or checkpoint-selection round.",
            "Static Full Set/Trace combines routing and tool selection; it is not real tool execution success.",
        ],
    }
    write_json(args.out, summary)
    if args.update_root_summary:
        root_summary = require_complete(args.root_summary)
        root_summary["dpo_v2_extension"] = summary
        write_json(args.root_summary, root_summary)
    print(
        f"DPO-v2 final summary complete -> {args.out}; "
        f"root_updated={args.update_root_summary}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
