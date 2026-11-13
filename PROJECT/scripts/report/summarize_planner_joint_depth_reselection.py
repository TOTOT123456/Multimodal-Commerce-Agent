#!/usr/bin/env python3
"""Preregister and select an efficiency-aware joint Reranker/Planner depth."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT.parent
RUN_ROOT = (
    PROJECT
    / "data/tool_call_trace/_aux/assignment3/planner_joint_depth_reselection_20260801"
)
PRED_ROOT = (
    PROJECT
    / "finetune/outputs/preds/tool_trace/planner_joint_depth_reselection_20260801"
)
ROUTE_CONFIG = PROJECT / "configs/retrieval_canonical_route.json"
FROZEN_RETRIEVAL = (
    PROJECT / "data/rag/_aux/retrieval_pipeline_reselection/frozen_pipeline.json"
)
MODEL = PROJECT / "models/Qwen/Qwen3-8B"
ADAPTER = (
    PROJECT
    / "finetune/outputs/tooltrace-a3-crossstyle-rag10-sft8-e1-lr5e5-bs16/checkpoint-2429"
)
RERANKER_DEPTHS = (10, 20, 30, 50)
PLANNER_DEPTHS = (5, 10, 20, 30, 50)
PANELS = ("legacy_like", "api_like")
PANEL_WEIGHTS = {"legacy_like": 6.0 / 7.0, "api_like": 1.0 / 7.0}
TRACE_NEAR_BEST = 0.005
MAX_PANEL_TRACE_GAP = 0.01
MAX_SINGLETON_TRACE_GAP = 0.01
MAX_UNSEEN_COMBO_TRACE_GAP = 0.01
MAX_ALLHIT_REGRESSION_FROM_ANCHOR = 0.002
MIN_STRICT = 0.995
CONFIRM_MACRO_MARGIN = 0.005
CONFIRM_PANEL_MARGIN = 0.01
CONFIRM_ALLHIT_MARGIN = 0.002


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def route_ids() -> list[str]:
    return [
        f"rerank{reranker_depth}_planner{planner_depth}"
        for reranker_depth in RERANKER_DEPTHS
        for planner_depth in PLANNER_DEPTHS
        if planner_depth <= reranker_depth
    ]


def implementation_files() -> list[Path]:
    return [
        PROJECT / "scripts/finetune/build_planner_joint_depth_dev.py",
        PROJECT / "scripts/finetune/generate_assignment3_api_native_dev.py",
        PROJECT / "scripts/finetune/sanitize_tool_trace_eval_inputs.py",
        PROJECT / "scripts/finetune/infer_tool_trace.py",
        PROJECT / "scripts/finetune/evaluate_tool_trace.py",
        PROJECT / "scripts/finetune/run_planner_joint_depth_reselection.sh",
        PROJECT / "scripts/finetune/run_planner_joint_depth_protected.sh",
        PROJECT / "scripts/rag/apply_first_stage_route.py",
        PROJECT / "scripts/rag/build_planner_joint_depth_routes.py",
        PROJECT / "scripts/rag/run_retrieval_baselines.py",
        PROJECT / "scripts/rag/run_reranker_baseline.py",
        Path(__file__).resolve(),
    ]


def begin() -> None:
    plan_audit = load_json(RUN_ROOT / "plan_audit.json")
    data_audit = load_json(RUN_ROOT / "data_audit.json")
    if not plan_audit.get("strict_ok") or not data_audit.get("strict_ok"):
        raise ValueError("representative development data did not pass strict audit")
    if data_audit["counts"]["selection"]["rows"] != 2400:
        raise ValueError("selection split must contain 2400 rows")
    if data_audit["counts"]["confirmation"]["rows"] != 1200:
        raise ValueError("confirmation split must contain 1200 rows")
    route = load_json(ROUTE_CONFIG)
    frozen = load_json(FROZEN_RETRIEVAL)
    if (
        route.get("status") != "canonical"
        or frozen.get("status") != "frozen_before_round_final"
        or route["embedding"]["model_sha256"] != frozen["embedding"]["model_sha256"]
        or route["reranker"]["model_sha256"] != frozen["reranker"]["model_sha256"]
    ):
        raise ValueError("canonical model identities do not match frozen retrieval")
    required = [
        RUN_ROOT / "plan_audit.json",
        RUN_ROOT / "data_audit.json",
        RUN_ROOT / "selection.jsonl",
        RUN_ROOT / "confirmation.jsonl",
        ROUTE_CONFIG,
        FROZEN_RETRIEVAL,
        MODEL / "config.json",
        MODEL / "tokenizer.json",
        ADAPTER / "adapter_config.json",
        ADAPTER / "adapter_model.safetensors",
        *implementation_files(),
    ]
    missing = [relative(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"missing protocol inputs: {missing}")
    payload = {
        "schema_version": 1,
        "status": (
            "corrective_preregistered_before_valid_route_search_or_valid_planner_inference"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective": (
            "choose a coherent Reranker-depth/Planner-depth deployment point that "
            "retains near-best quality while reducing context and cross-encoder cost"
        ),
        "known_before_preregistration": {
            "old_combo_disjoint_dev_k10_trace": 0.9333333333333333,
            "old_combo_disjoint_dev_k50_trace": 0.9866666666666667,
            "legacy_k10_trace": 0.8595,
            "legacy_k50_trace": 0.8538333333333333,
            "api_k10_trace": 0.994,
            "api_k50_trace": 0.996,
            "reason": (
                "These results predate this corrective study and are disclosed; "
                "the later aborted metric-plumbing attempt is disclosed separately."
            ),
        },
        "aborted_attempt_disclosure": {
            "path": relative(
                RUN_ROOT / "aborted/group_metric_bug_20260801T1914Z/protocol.json"
            ),
            "protocol_sha256": "40dfc59f2934de8af821681b0fc5049fa6c48e958e397250c2e41b3b39d0bab6",
            "invalid_routes_sha256": "417b67334e7ee67c82bec2430c9931b95b374a28be0b4ce83dcbac6ee06a975d",
            "bug": (
                "subgroup metrics used raw dataset rows containing the Chinese "
                "工具 field while the metric helper expected gold_tools; panel "
                "AllHit was therefore zero and the RRF tie-break was invalid"
            ),
            "observed_before_correction": {
                "overall_allhit_at_10_by_reranker_depth": {
                    "10": 0.99375,
                    "20": 0.9991666666666666,
                    "30": 0.99875,
                    "50": 0.9979166666666667,
                },
                "exact_reranker_p50_ms_per_query": {
                    "10": 23.9481,
                    "20": 47.0914,
                    "30": 70.9828,
                    "50": 116.9624,
                },
                "complete_planner_cells": 0,
            },
            "correction_scope": (
                "only pass aligned gold_tools metric rows into subgroup metrics; "
                "the dataset, candidate grid, thresholds, cost order, models, and "
                "exact-depth reranker predictions are unchanged"
            ),
        },
        "development": {
            "selection": relative(RUN_ROOT / "selection.jsonl"),
            "rows": 2400,
            "panel_weights": PANEL_WEIGHTS,
            "confirmation": relative(RUN_ROOT / "confirmation.jsonl"),
            "confirmation_rows": 1200,
            "confirmation_role": "sealed promotion gate; cannot tune parameters",
            "representativeness": plan_audit["design"],
            "isolation": {
                "plan": plan_audit["isolation"],
                "rendered_queries": data_audit["isolation"],
            },
        },
        "fixed_components": {
            "embedding": route["embedding"],
            "first_stage": route["first_stage"],
            "reranker_model": {
                key: route["reranker"][key]
                for key in (
                    "base_model",
                    "negative_recipe",
                    "learning_rate",
                    "seed",
                    "checkpoint",
                    "model_sha256",
                )
            },
            "planner_model": relative(MODEL),
            "planner_adapter": relative(ADAPTER),
            "planner_training_candidate_k": 10,
            "decoding": "greedy",
        },
        "searched_components": {
            "reranker_depths": list(RERANKER_DEPTHS),
            "planner_depths": list(PLANNER_DEPTHS),
            "constraint": "planner_depth <= reranker_depth",
            "second_stage_source_weights": [
                0.0,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                1.0,
            ],
            "second_stage_rrf_ks": [1, 2, 3, 5, 8, 10, 20, 30, 60, 100],
            "candidate_route_ids": route_ids(),
        },
        "development_selection_rule": {
            "primary_quality": (
                "6:1 Legacy/API deployment-weighted TraceEM on Qwen3-8B SFT"
            ),
            "retrieval_guard": (
                "each panel AllHit may trail coherent rerank10/planner10 anchor "
                f"by at most {MAX_ALLHIT_REGRESSION_FROM_ANCHOR:.3f} absolute"
            ),
            "minimum_overall_strict": MIN_STRICT,
            "trace_near_best_absolute": TRACE_NEAR_BEST,
            "maximum_panel_trace_gap_from_panel_best": MAX_PANEL_TRACE_GAP,
            "maximum_singleton_trace_gap_from_singleton_best": MAX_SINGLETON_TRACE_GAP,
            "maximum_unseen_combo_trace_gap_from_unseen_best": MAX_UNSEEN_COMBO_TRACE_GAP,
            "cost_tiebreak": (
                "smaller Planner depth, then smaller Reranker depth, then "
                "lower deployment-weighted mean input tokens"
            ),
            "fallback_if_no_candidate_is_eligible": "coherent anchor rerank10_planner10",
        },
        "confirmation_gate": {
            "anchor": "rerank10_planner10",
            "selected_deployment_weighted_trace_may_trail_anchor_by_at_most": CONFIRM_MACRO_MARGIN,
            "selected_panel_trace_may_trail_anchor_by_at_most": CONFIRM_PANEL_MARGIN,
            "selected_singleton_and_unseen_trace_may_trail_anchor_by_at_most": CONFIRM_PANEL_MARGIN,
            "selected_panel_allhit_may_trail_anchor_by_at_most": CONFIRM_ALLHIT_MARGIN,
            "on_failure": "retain coherent anchor rerank10_planner10",
            "protected_legacy_api_cannot_change_selection": True,
        },
        "input_hashes": {relative(path): sha256_file(path) for path in required},
    }
    path = RUN_ROOT / "protocol.json"
    if path.exists():
        existing = load_json(path)
        for key in payload:
            if key == "created_at_utc":
                continue
            if existing.get(key) != payload.get(key):
                raise ValueError(f"existing protocol differs at {key}")
        print(f"protocol already frozen: {path}")
        return
    write_json(path, payload)
    print(f"protocol frozen: {path}")


def evaluation_paths(split: str, route_id: str) -> dict[str, Path]:
    stem = f"joint-sft8-{split}-{route_id}"
    return {
        "predictions": PRED_ROOT / split / f"{stem}.jsonl",
        "metrics": RUN_ROOT / split / "eval" / f"{stem}.metrics.json",
        "scored": RUN_ROOT / split / "eval" / f"{stem}.scored.jsonl",
    }


def route_map(split: str) -> tuple[dict[str, Any], Path]:
    path = RUN_ROOT / split / "retrieval/routes.json"
    payload = load_json(path)
    if payload.get("status") != "complete":
        raise ValueError(f"incomplete routes: {path}")
    return {str(row["id"]): row for row in payload["routes"]}, path


def load_cell(split: str, route_id: str, route: dict[str, Any]) -> dict[str, Any]:
    paths = evaluation_paths(split, route_id)
    for path in paths.values():
        if not path.is_file():
            raise ValueError(f"missing inference artifact: {path}")
    metrics = load_json(paths["metrics"])
    scored = load_jsonl(paths["scored"])
    dataset = {
        str(row["id"]): row for row in load_jsonl(RUN_ROOT / f"{split}.jsonl")
    }
    expected = 2400 if split == "selection" else 1200
    if (
        metrics.get("status") != "complete"
        or metrics.get("count") != expected
        or len(scored) != expected
        or metrics.get("predictions_sha256") != sha256_file(paths["predictions"])
        or set(dataset) != {str(row["id"]) for row in scored}
    ):
        raise ValueError(f"invalid evaluation cell: {route_id}")
    panel: dict[str, Any] = {}
    for name in PANELS:
        values = [
            row
            for row in scored
            if dataset[str(row["id"])]["construction"]["panel"] == name
        ]
        panel[name] = {
            "count": len(values),
            "set_em": sum(bool(row["set_em"]) for row in values) / len(values),
            "trace_em": sum(bool(row["trace_em"]) for row in values) / len(values),
            "strict": sum(bool(row["strict_parse_ok"]) for row in values)
            / len(values),
            "mean_input_tokens": statistics.fmean(
                float(row.get("input_tokens", 0)) for row in values
            ),
            "mean_generator_latency_ms": statistics.fmean(
                float(row.get("latency_ms", 0)) for row in values
            ),
            "retriever_allhit": route["by_panel"][name][
                f"allhit_at_{route['planner_depth']}"
            ],
        }

    def planner_buckets(field: str) -> dict[str, Any]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in scored:
            metadata = dataset[str(row["id"])]
            value = metadata.get(field)
            if value is None and isinstance(metadata.get("construction"), dict):
                value = metadata["construction"].get(field)
            grouped.setdefault(str(value), []).append(row)
        return {
            value: {
                "count": len(values),
                "set_em": sum(bool(row["set_em"]) for row in values) / len(values),
                "trace_em": sum(bool(row["trace_em"]) for row in values)
                / len(values),
                "strict": sum(bool(row["strict_parse_ok"]) for row in values)
                / len(values),
            }
            for value, values in sorted(grouped.items())
        }

    by_tool_count = planner_buckets("tool_count")
    by_combo_seen = planner_buckets("combo_seen_bucket")
    overall = metrics["overall"]
    efficiency = metrics["efficiency"]
    reranker_metrics_path = (
        RUN_ROOT
        / split
        / "retrieval"
        / f"reranker_k{route['reranker_depth']}"
        / "metrics.json"
    )
    reranker_metrics = load_json(reranker_metrics_path)
    if int(reranker_metrics.get("candidate_k", 0)) != route["reranker_depth"]:
        raise ValueError(f"invalid exact-depth reranker metrics: {reranker_metrics_path}")
    return {
        "route": route,
        "overall": {
            "set_em": overall["set_em"],
            "trace_em": overall["trace_em"],
            "strict": overall["strict_output_valid_rate"],
            "retriever_allhit": route["overall"][
                f"allhit_at_{route['planner_depth']}"
            ],
            "mean_input_tokens": efficiency["input_tokens"]["mean"],
            "p50_input_tokens": efficiency["input_tokens"]["p50"],
            "p50_generator_latency_ms": efficiency["latency_ms_generator_only"]["p50"],
            "reranker_p50_ms_per_query": reranker_metrics["timing"][
                "rerank_p50_ms_per_query"
            ],
        },
        "panel": panel,
        "by_tool_count": by_tool_count,
        "by_combo_seen_bucket": by_combo_seen,
        "macro_trace_em": sum(panel[name]["trace_em"] for name in PANELS)
        / len(PANELS),
        "deployment_weighted_trace_em": sum(
            PANEL_WEIGHTS[name] * panel[name]["trace_em"] for name in PANELS
        ),
        "deployment_weighted_mean_input_tokens": sum(
            PANEL_WEIGHTS[name] * panel[name]["mean_input_tokens"]
            for name in PANELS
        ),
        "deployment_weighted_mean_generator_latency_ms": sum(
            PANEL_WEIGHTS[name] * panel[name]["mean_generator_latency_ms"]
            for name in PANELS
        ),
        "paths": {
            **{key: relative(path) for key, path in paths.items()},
            "reranker_metrics": relative(reranker_metrics_path),
        },
        "hashes": {
            **{key: sha256_file(path) for key, path in paths.items()},
            "reranker_metrics": sha256_file(reranker_metrics_path),
        },
    }


def select() -> None:
    protocol_path = RUN_ROOT / "protocol.json"
    if not protocol_path.is_file():
        raise ValueError("run --stage begin first")
    routes, routes_path = route_map("selection")
    if set(routes) != set(route_ids()):
        raise ValueError("selection route grid is incomplete")
    cells = {route_id: load_cell("selection", route_id, routes[route_id]) for route_id in route_ids()}
    best_weighted = max(
        cell["deployment_weighted_trace_em"] for cell in cells.values()
    )
    best_panel = {
        panel: max(cell["panel"][panel]["trace_em"] for cell in cells.values())
        for panel in PANELS
    }
    best_singleton = max(
        cell["by_tool_count"]["1"]["trace_em"] for cell in cells.values()
    )
    best_unseen = max(
        cell["by_combo_seen_bucket"]["sft_unseen"]["trace_em"]
        for cell in cells.values()
    )
    anchor = cells["rerank10_planner10"]
    eligibility = {}
    for route_id, cell in cells.items():
        checks = {
            "panel_allhit_anchor_guard": all(
                cell["panel"][panel]["retriever_allhit"]
                >= anchor["panel"][panel]["retriever_allhit"]
                - MAX_ALLHIT_REGRESSION_FROM_ANCHOR
                for panel in PANELS
            ),
            "strict": cell["overall"]["strict"] >= MIN_STRICT,
            "deployment_weighted_trace_near_best": cell[
                "deployment_weighted_trace_em"
            ]
            >= best_weighted - TRACE_NEAR_BEST,
            "panel_trace_guard": all(
                cell["panel"][panel]["trace_em"]
                >= best_panel[panel] - MAX_PANEL_TRACE_GAP
                for panel in PANELS
            ),
            "singleton_trace_guard": cell["by_tool_count"]["1"]["trace_em"]
            >= best_singleton - MAX_SINGLETON_TRACE_GAP,
            "unseen_combo_trace_guard": cell["by_combo_seen_bucket"][
                "sft_unseen"
            ]["trace_em"]
            >= best_unseen - MAX_UNSEEN_COMBO_TRACE_GAP,
        }
        eligibility[route_id] = {**checks, "eligible": all(checks.values())}
    eligible = [route_id for route_id in route_ids() if eligibility[route_id]["eligible"]]
    fallback_used = not eligible
    selected_id = (
        "rerank10_planner10"
        if fallback_used
        else min(
            eligible,
            key=lambda route_id: (
                cells[route_id]["route"]["planner_depth"],
                cells[route_id]["route"]["reranker_depth"],
                cells[route_id]["deployment_weighted_mean_input_tokens"],
            ),
        )
    )
    payload = {
        "schema_version": 1,
        "status": "locked_before_confirmation_inference",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": relative(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "routes": relative(routes_path),
        "routes_sha256": sha256_file(routes_path),
        "cells": cells,
        "best_deployment_weighted_trace_em": best_weighted,
        "best_panel_trace_em": best_panel,
        "best_singleton_trace_em": best_singleton,
        "best_unseen_combo_trace_em": best_unseen,
        "eligibility": eligibility,
        "eligible_route_ids": eligible,
        "fallback_used": fallback_used,
        "selected_route_id": selected_id,
        "selected_route": cells[selected_id]["route"],
    }
    path = RUN_ROOT / "selection.lock.json"
    if path.exists():
        existing = load_json(path)
        for key in (
            "protocol_sha256",
            "routes_sha256",
            "best_deployment_weighted_trace_em",
            "best_panel_trace_em",
            "best_singleton_trace_em",
            "best_unseen_combo_trace_em",
            "eligibility",
            "selected_route_id",
            "selected_route",
        ):
            if existing.get(key) != payload.get(key):
                raise ValueError(f"selection lock differs at {key}")
        print(f"selection already locked: {existing['selected_route_id']}")
        return
    write_json(path, payload)
    print(f"selection locked: {selected_id} -> {path}")


def confirm() -> None:
    lock_path = RUN_ROOT / "selection.lock.json"
    if not lock_path.is_file():
        raise ValueError("run --stage select first")
    lock = load_json(lock_path)
    routes, routes_path = route_map("confirmation")
    selected_id = str(lock["selected_route_id"])
    anchor_id = "rerank10_planner10"
    required = sorted({selected_id, anchor_id})
    cells = {
        route_id: load_cell("confirmation", route_id, routes[route_id])
        for route_id in required
    }
    selected = cells[selected_id]
    anchor = cells[anchor_id]
    checks = {
        "deployment_weighted_trace": selected["deployment_weighted_trace_em"]
        >= anchor["deployment_weighted_trace_em"] - CONFIRM_MACRO_MARGIN,
        "panel_trace": all(
            selected["panel"][panel]["trace_em"]
            >= anchor["panel"][panel]["trace_em"] - CONFIRM_PANEL_MARGIN
            for panel in PANELS
        ),
        "singleton_trace": selected["by_tool_count"]["1"]["trace_em"]
        >= anchor["by_tool_count"]["1"]["trace_em"] - CONFIRM_PANEL_MARGIN,
        "unseen_combo_trace": selected["by_combo_seen_bucket"]["sft_unseen"][
            "trace_em"
        ]
        >= anchor["by_combo_seen_bucket"]["sft_unseen"]["trace_em"]
        - CONFIRM_PANEL_MARGIN,
        "panel_allhit_anchor_guard": all(
            selected["panel"][panel]["retriever_allhit"]
            >= anchor["panel"][panel]["retriever_allhit"] - CONFIRM_ALLHIT_MARGIN
            for panel in PANELS
        ),
    }
    promoted = all(checks.values())
    final_id = selected_id if promoted else anchor_id
    payload = {
        "schema_version": 1,
        "status": "complete_before_protected_reporting",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": relative(RUN_ROOT / "protocol.json"),
        "selection_lock": relative(lock_path),
        "selection_lock_sha256": sha256_file(lock_path),
        "confirmation_routes": relative(routes_path),
        "confirmation_routes_sha256": sha256_file(routes_path),
        "development_selected_route_id": selected_id,
        "confirmation_cells": cells,
        "confirmation_checks": checks,
        "development_selection_promoted": promoted,
        "final_route_id": final_id,
        "final_route": routes[final_id],
        "protected_legacy_api_can_change_final_route": False,
    }
    write_json(RUN_ROOT / "summary.json", payload)
    print(
        json.dumps(
            {
                "development_selected": selected_id,
                "promoted": promoted,
                "final_route": final_id,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def metric_snapshot(path: Path, expected: int) -> dict[str, Any]:
    payload = load_json(path)
    if payload.get("status") != "complete" or payload.get("count") != expected:
        raise ValueError(f"incomplete metrics: {path}")
    return {
        "path": relative(path),
        "sha256": sha256_file(path),
        "count": expected,
        "overall": payload["overall"],
        "rag_decomposition": payload["rag_decomposition"],
        "efficiency": payload["efficiency"],
    }


def paired_transition(left_path: Path, right_path: Path) -> dict[str, Any]:
    left_rows = load_jsonl(left_path)
    right_rows = load_jsonl(right_path)
    left = {str(row["id"]): row for row in left_rows}
    right = {str(row["id"]): row for row in right_rows}
    if len(left) != len(left_rows) or len(right) != len(right_rows) or left.keys() != right.keys():
        raise ValueError("paired scored rows are not aligned")
    result: dict[str, Any] = {"count": len(left)}
    for metric in ("set_em", "trace_em"):
        improved = sum(
            not bool(left[key][metric]) and bool(right[key][metric]) for key in left
        )
        regressed = sum(
            bool(left[key][metric]) and not bool(right[key][metric]) for key in left
        )
        result[metric] = {
            "left": sum(bool(row[metric]) for row in left.values()) / len(left),
            "right": sum(bool(row[metric]) for row in right.values()) / len(right),
            "delta": (improved - regressed) / len(left),
            "improved": improved,
            "regressed": regressed,
        }
    return result


def report() -> None:
    summary_path = RUN_ROOT / "summary.json"
    if not summary_path.is_file():
        raise ValueError("run --stage confirm first")
    summary = load_json(summary_path)
    route_id = str(summary["final_route_id"])
    protected: dict[str, Any] = {}
    pred_root = PRED_ROOT / "protected"
    for split, expected in (("legacy6000", 6000), ("api1000", 1000)):
        retrieval_path = RUN_ROOT / "protected" / split / "retrieval" / "routes.json"
        retrieval = load_json(retrieval_path)
        routes = {str(row["id"]): row for row in retrieval["routes"]}
        if set(routes) != {route_id}:
            raise ValueError(f"protected retrieval route mismatch: {split}")
        cells: dict[str, Any] = {}
        for stage in ("base", "sft"):
            for model in ("4b", "8b"):
                stem = f"joint-final-{stage}-{model}-{split}-{route_id}"
                pred_path = pred_root / f"{stem}.jsonl"
                metrics_path = RUN_ROOT / "protected" / split / "eval" / f"{stem}.metrics.json"
                scored_path = RUN_ROOT / "protected" / split / "eval" / f"{stem}.scored.jsonl"
                if not pred_path.is_file() or len(load_jsonl(scored_path)) != expected:
                    raise ValueError(f"missing protected cell: {stem}")
                cell = metric_snapshot(metrics_path, expected)
                cell.update(
                    {
                        "predictions": relative(pred_path),
                        "predictions_sha256": sha256_file(pred_path),
                        "scored": relative(scored_path),
                        "scored_sha256": sha256_file(scored_path),
                    }
                )
                cells[f"{stage}_{model}"] = cell
        protected[split] = {
            "retrieval": routes[route_id],
            "retrieval_routes": relative(retrieval_path),
            "retrieval_routes_sha256": sha256_file(retrieval_path),
            "cells": cells,
        }
    systems = {}
    for model in ("4b", "8b"):
        stem = f"joint-final-sft-{model}-legacy6000-{route_id}"
        path = RUN_ROOT / "protected/legacy6000/eval" / f"{stem}.system.metrics.json"
        value = load_json(path)
        if value.get("status") != "complete" or value.get("count") != 9000:
            raise ValueError(f"invalid system metrics: {path}")
        systems[model] = {
            "path": relative(path),
            "sha256": sha256_file(path),
            "full_set": value["full_pipeline_success"],
            "full_trace": value["full_pipeline_trace_success"],
        }

    isolated_root = RUN_ROOT / "strict_isolated_diagnostic"
    isolated_ids = sorted({"rerank10_planner10", route_id})
    isolated_cells = {}
    for current_id in isolated_ids:
        stem = f"joint-isolated-sft8-{current_id}"
        metrics_path = isolated_root / "eval" / f"{stem}.metrics.json"
        scored_path = isolated_root / "eval" / f"{stem}.scored.jsonl"
        isolated_cells[current_id] = {
            **metric_snapshot(metrics_path, 1200),
            "scored": relative(scored_path),
            "scored_sha256": sha256_file(scored_path),
        }
    isolated_comparison = None
    if route_id != "rerank10_planner10":
        isolated_comparison = paired_transition(
            ROOT / isolated_cells["rerank10_planner10"]["scored"],
            ROOT / isolated_cells[route_id]["scored"],
        )

    payload = {
        "schema_version": 1,
        "status": "complete_after_locked_route_on_protected_splits",
        "route_id": route_id,
        "selection_summary": relative(summary_path),
        "selection_summary_sha256": sha256_file(summary_path),
        "selection": load_json(RUN_ROOT / "selection.lock.json"),
        "confirmation": summary,
        "protected": protected,
        "static_system_legacy_9000": systems,
        "strict_combo_isolated_diagnostic": {
            "selection_role": "diagnostic_only_known_before_preregistration",
            "cells": isolated_cells,
            "selected_vs_anchor": isolated_comparison,
        },
    }
    path = RUN_ROOT / "final_report.json"
    write_json(path, payload)
    print(f"joint-depth final report: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("begin", "select", "confirm", "report"), required=True
    )
    args = parser.parse_args()
    {"begin": begin, "select": select, "confirm": confirm, "report": report}[
        args.stage
    ]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
