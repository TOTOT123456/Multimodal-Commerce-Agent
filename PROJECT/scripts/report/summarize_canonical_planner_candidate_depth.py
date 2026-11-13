#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Freeze, select, and summarize the current-canonical Planner Top-K study."""
from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT.parent
RUN_ROOT = (
    PROJECT
    / "data/tool_call_trace/_aux/assignment3"
    / "canonical_planner_candidate_depth_isolated_20260801"
)
PRED_ROOT = (
    PROJECT
    / "finetune/outputs/preds/tool_trace"
    / "canonical_planner_candidate_depth_isolated_20260801"
)
DEV_DATASET = RUN_ROOT / "dev.jsonl"
PLAN_AUDIT = RUN_ROOT / "plan_audit.json"
DATA_AUDIT = RUN_ROOT / "data_audit.json"
DEV_RANKING = (
    RUN_ROOT / "retrieval/challenger/preds.jsonl"
)
ROUTE = PROJECT / "configs/retrieval_canonical_route.json"
FROZEN_PIPELINE = (
    PROJECT / "data/rag/_aux/retrieval_pipeline_reselection/frozen_pipeline.json"
)
REPLAY_ROOT = (
    PROJECT
    / "data/tool_call_trace/_aux/assignment3"
    / "retrieval_reselection_planner_replay_20260731"
)
REPLAY_PRED_ROOT = (
    PROJECT
    / "finetune/outputs/preds/tool_trace"
    / "retrieval_reselection_planner_replay_20260731"
)
MODEL_PATHS = {
    "4b": PROJECT / "models/Qwen/Qwen3-4B",
    "8b": PROJECT / "models/Qwen/Qwen3-8B",
}
ADAPTER_PATHS = {
    "4b": PROJECT
    / "finetune/outputs/tooltrace-a3-crossstyle-rag10-sft4-e1-lr5e5-bs16/checkpoint-2429",
    "8b": PROJECT
    / "finetune/outputs/tooltrace-a3-crossstyle-rag10-sft8-e1-lr5e5-bs16/checkpoint-2429",
}
CANDIDATE_KS = (5, 10, 20, 30, 50)
MODELS = ("4b", "8b")
SPLITS = {"legacy6000": 6000, "api1000": 1000}
TRACE_TOLERANCE = 0.002


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def identity_files() -> list[Path]:
    paths: list[Path] = []
    for model in MODELS:
        for base, names in (
            (MODEL_PATHS[model], ("config.json", "tokenizer.json")),
            (
                ADAPTER_PATHS[model],
                ("adapter_config.json", "adapter_model.safetensors", "args.json"),
            ),
        ):
            paths.extend(base / name for name in names)
    return paths


def implementation_files() -> list[Path]:
    return [
        Path(__file__).resolve(),
        PROJECT / "scripts/finetune/run_canonical_planner_candidate_depth.sh",
        PROJECT / "scripts/finetune/sanitize_tool_trace_eval_inputs.py",
        PROJECT / "scripts/finetune/infer_tool_trace.py",
        PROJECT / "scripts/finetune/evaluate_tool_trace.py",
        PROJECT / "scripts/finetune/run_assignment3_eval.sh",
        PROJECT / "scripts/rag/apply_frozen_retrieval_pipeline.py",
        PROJECT / "scripts/rag/run_planner_candidate_depth_isolated_retrieval.sh",
    ]


def protocol_payload() -> dict[str, Any]:
    required = [
        ROUTE,
        FROZEN_PIPELINE,
        DEV_DATASET,
        PLAN_AUDIT,
        DATA_AUDIT,
        DEV_RANKING,
        *identity_files(),
        *implementation_files(),
    ]
    missing = [relative(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"missing protocol inputs: {missing}")
    route = load_json(ROUTE)
    plan_audit = load_json(PLAN_AUDIT)
    data_audit = load_json(DATA_AUDIT)
    ranking_metrics = load_json(DEV_RANKING.parent / "metrics.json")
    ranking_rows = load_jsonl(DEV_RANKING)
    ranking_upper_bounds = {
        str(k): sum(
            bool(row.get("gold_tools"))
            and set(row["gold_tools"]) <= set(row["pred_tools"][:k])
            for row in ranking_rows
        )
        / len(ranking_rows)
        for k in CANDIDATE_KS
    }
    expected_route = {
        "first_candidate_n": route["first_stage"]["candidate_n_per_source"],
        "first_vector_weight": route["first_stage"]["vector_weight"],
        "first_rrf_k": route["first_stage"]["rrf_k"],
        "candidate_k": route["reranker"]["candidate_k"],
        "second_source_weight": route["second_stage"]["source_weight"],
        "second_rrf_k": route["second_stage"]["rrf_k"],
    }
    if (
        route.get("status") != "canonical"
        or route["planner"]["candidate_k"] != 10
        or route["planner"]["candidate_order"] != "retrieval_order"
        or route["planner"]["decoding"] != "greedy"
        or not plan_audit.get("strict_ok")
        or not data_audit.get("strict_ok")
        or data_audit["isolation"]["source_trace_id_overlap_with_sft"] != 0
        or data_audit["isolation"]["tool_combination_overlap_with_sft"] != 0
        or data_audit["isolation"]["exact_query_overlap_with_sft_or_blocked"] != 0
        or ranking_metrics.get("route") != expected_route
        or ranking_metrics.get("count") != 1200
        or ranking_metrics.get("ranking_completion", {}).get("target_depth")
        != max(CANDIDATE_KS)
        or not ranking_metrics.get("ranking_completion", {}).get("prefix_preserved")
        or ranking_metrics["overall"].get("min_output_size", 0) < max(CANDIDATE_KS)
    ):
        raise ValueError("current canonical route and development ranking do not align")
    return {
        "schema_version": 1,
        "status": "preregistered_before_candidate_depth_inference",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Planner input candidate depth under one fixed current-canonical ranking",
        "controlled_variable": "candidate prefix K only",
        "development": {
            "dataset": relative(DEV_DATASET),
            "rows": 1200,
            "construction": (
                "600 SFT-unseen unordered tool combinations, each independently "
                "rendered once as api_native and once as legacy_compact"
            ),
            "tool_counts": [2, 3, 4, 5, 6],
            "per_tool_count_per_style": 120,
            "isolation": {
                "source_trace_id_overlap_with_sft": 0,
                "tool_combination_overlap_with_sft": 0,
                "exact_query_overlap_with_sft_or_blocked": 0,
            },
            "singleton_exclusion": (
                "All 100 singleton tool sets occur in SFT, so strict tool-set "
                "disjointness cannot include tool_count=1."
            ),
            "structural_allhit_upper_bound": {
                "5": 0.8,
                "10": 1.0,
                "20": 1.0,
                "30": 1.0,
                "50": 1.0,
                "reason": (
                    "Exactly 240/1200 rows require six tools, so K=5 cannot "
                    "contain every gold tool for those rows."
                ),
            },
            "ranking": relative(DEV_RANKING),
            "ranking_semantics": (
                "Exact deployment-path easy2 + BM25/vector RRF + v2-m3 Top-10 "
                "+ second RRF ordering; if its union is shorter than 50, append "
                "full-source RRF tail without changing the frozen-route prefix; "
                "every K is a prefix of this same completed ordering"
            ),
            "candidate_ks": list(CANDIDATE_KS),
            "known_retrieval_upper_bounds_before_planner_inference": (
                ranking_upper_bounds
            ),
            "ranking_completion": ranking_metrics["ranking_completion"],
        },
        "fixed_planners": {
            model: {
                "model": relative(MODEL_PATHS[model]),
                "adapter": relative(ADAPTER_PATHS[model]),
            }
            for model in MODELS
        },
        "decoding": {
            "dtype": "bfloat16",
            "greedy": True,
            "candidate_order": "retrieval_order",
            "max_new_tokens": 128,
            "seed": 20260725,
        },
        "selection_rule": {
            "deployment_endpoint": "Qwen3-8B SFT checkpoint-2429",
            "description": (
                "Among K values within 0.002 absolute of the best 8B development "
                "TraceEM, choose the one with the fewest mean input tokens; "
                "retrieval upper bound is reported as an error decomposition, not "
                "double-counted as a second eligibility gate; 4B is diagnostic only."
            ),
            "trace_em_tolerance": TRACE_TOLERANCE,
            "tie_breaker": "fewer mean input tokens, then smaller K",
            "retriever_upper_bound_is_diagnostic": True,
            "four_b_cannot_override": True,
            "protected_splits_cannot_change_development_selection": True,
        },
        "uncertainty_reporting": {
            "unit": "novel unordered tool combination",
            "clusters": 600,
            "rows_per_cluster": 2,
            "styles_per_cluster": ["api_native", "legacy_compact"],
            "method": "paired cluster bootstrap",
            "samples": 10000,
            "confidence_interval": 0.95,
            "selection_uses_preregistered_point_tolerance_not_posthoc_p_values": True,
        },
        "post_selection_evaluation": {
            "applies_only_if_selected_k_differs_from_existing_k10": True,
            "role": "one-shot Legacy/API reporting only",
            "cannot_change_development_selection": True,
            "four_b_is_diagnostic_only": True,
        },
        "input_hashes": {relative(path): sha256_file(path) for path in required},
    }


def begin() -> None:
    expected = protocol_payload()
    path = RUN_ROOT / "protocol.json"
    if path.exists():
        existing = load_json(path)
        for key in (
            "scope",
            "controlled_variable",
            "development",
            "fixed_planners",
            "decoding",
            "selection_rule",
            "uncertainty_reporting",
            "post_selection_evaluation",
        ):
            if existing.get(key) != expected.get(key):
                raise ValueError(f"existing protocol differs at {key}")
        # The canonical JSON gains evidence pointers after a completed study.
        # Its route semantics are validated above; all other preregistered
        # model/data bytes must remain identical on resume.
        route_key = relative(ROUTE)
        existing_hashes = dict(existing.get("input_hashes", {}))
        expected_hashes = dict(expected.get("input_hashes", {}))
        existing_hashes.pop(route_key, None)
        expected_hashes.pop(route_key, None)
        if existing_hashes != expected_hashes:
            raise ValueError("existing protocol model/data hashes differ")
        print(f"resume authorized: {path}")
        return
    write_json(path, expected)
    print(f"preregistered: {path}")


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    overall = metrics["overall"]
    retrieval = metrics["rag_decomposition"]
    efficiency = metrics["efficiency"]
    return {
        "count": metrics["count"],
        "set_em": overall["set_em"],
        "trace_em": overall["trace_em"],
        "strict_output_valid_rate": overall["strict_output_valid_rate"],
        "retriever_upper_bound": retrieval["retriever_upper_bound"],
        "retrieval_success_count": retrieval["retrieval_success_count"],
        "conditional_set_accuracy": retrieval["conditional_set_accuracy"],
        "conditional_trace_accuracy": retrieval["conditional_trace_accuracy"],
        "mean_input_tokens": efficiency["input_tokens"]["mean"],
        "p50_input_tokens": efficiency["input_tokens"]["p50"],
        "mean_output_tokens": efficiency["output_tokens"]["mean"],
        "mean_generator_latency_ms": efficiency["latency_ms_generator_only"]["mean"],
        "p50_generator_latency_ms": efficiency["latency_ms_generator_only"]["p50"],
        "p95_generator_latency_ms": efficiency["latency_ms_generator_only"]["p95"],
        "tool_count": metrics["buckets"]["tool_count"],
        "error_counts": metrics["error_analysis"]["counts"],
    }


def exact_mcnemar_p(improved: int, regressed: int) -> float:
    discordant = improved + regressed
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(improved, regressed) + 1)
    )
    return min(1.0, 2.0 * tail / (2**discordant))


def paired(
    baseline_path: Path, challenger_path: Path, field: str
) -> dict[str, Any]:
    baseline_rows = load_jsonl(baseline_path)
    challenger_rows = load_jsonl(challenger_path)
    baseline = {str(row["id"]): bool(row[field]) for row in baseline_rows}
    challenger = {str(row["id"]): bool(row[field]) for row in challenger_rows}
    if len(baseline) != len(baseline_rows) or baseline.keys() != challenger.keys():
        raise ValueError(f"paired rows do not align for {field}")
    improved = sum(not baseline[key] and challenger[key] for key in baseline)
    regressed = sum(baseline[key] and not challenger[key] for key in baseline)
    count = len(baseline)
    return {
        "field": field,
        "count": count,
        "baseline": sum(baseline.values()) / count,
        "challenger": sum(challenger.values()) / count,
        "delta": (improved - regressed) / count,
        "improved": improved,
        "regressed": regressed,
        "unchanged": count - improved - regressed,
        "exact_mcnemar_p": exact_mcnemar_p(improved, regressed),
    }


def clustered_paired(
    baseline_path: Path,
    challenger_path: Path,
    field: str,
    seed: int,
) -> dict[str, Any]:
    baseline_rows = load_jsonl(baseline_path)
    challenger_rows = load_jsonl(challenger_path)
    baseline = {str(row["id"]): bool(row[field]) for row in baseline_rows}
    challenger = {str(row["id"]): bool(row[field]) for row in challenger_rows}
    if len(baseline) != len(baseline_rows) or baseline.keys() != challenger.keys():
        raise ValueError(f"clustered paired rows do not align for {field}")
    metadata = {str(row["id"]): row for row in load_jsonl(DEV_DATASET)}
    if set(metadata) != set(baseline):
        raise ValueError("development metadata does not align with scored rows")
    cluster_ids: dict[str, list[str]] = {}
    for trace_id, row in metadata.items():
        combo_id = str(row["construction"]["novel_combo_id"])
        cluster_ids.setdefault(combo_id, []).append(trace_id)
    if len(cluster_ids) != 600 or any(len(values) != 2 for values in cluster_ids.values()):
        raise ValueError("expected 600 paired two-style development clusters")
    deltas = [
        sum(
            int(challenger[trace_id]) - int(baseline[trace_id])
            for trace_id in trace_ids
        )
        / len(trace_ids)
        for trace_ids in cluster_ids.values()
    ]
    observed = sum(deltas) / len(deltas)
    rng = random.Random(seed)
    bootstrap = sorted(
        sum(rng.choices(deltas, k=len(deltas))) / len(deltas)
        for _ in range(10000)
    )
    lower = bootstrap[int(0.025 * len(bootstrap))]
    upper = bootstrap[int(0.975 * len(bootstrap)) - 1]
    return {
        "field": field,
        "clusters": len(deltas),
        "rows_per_cluster": 2,
        "delta": observed,
        "bootstrap_samples": len(bootstrap),
        "ci95": [lower, upper],
    }


def dev_paths(model: str, candidate_k: int) -> dict[str, Path]:
    stem = f"canonical-isolated-sft-{model}-dev1200-k{candidate_k}"
    return {
        "input": RUN_ROOT / "inputs" / f"dev_rag_k{candidate_k}_label_free.jsonl",
        "audit": RUN_ROOT / "inputs" / f"dev_k{candidate_k}_audit.json",
        "predictions": PRED_ROOT / "dev" / f"{stem}.jsonl",
        "manifest": PRED_ROOT / "dev" / f"{stem}.shard0.jsonl.manifest.json",
        "metrics": RUN_ROOT / "dev" / f"{stem}.metrics.json",
        "scored": RUN_ROOT / "dev" / f"{stem}.scored.jsonl",
    }


def load_dev_cell(model: str, candidate_k: int) -> dict[str, Any]:
    paths = dev_paths(model, candidate_k)
    missing = [relative(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ValueError(f"incomplete development cell: {missing}")
    audit = load_json(paths["audit"])
    metrics = load_json(paths["metrics"])
    manifest = load_json(paths["manifest"])
    scored = load_jsonl(paths["scored"])
    if (
        not audit.get("strict_ok")
        or audit.get("candidate_k") != candidate_k
        or audit.get("rag_rows") != 1200
        or metrics.get("status") != "complete"
        or metrics.get("count") != 1200
        or metrics.get("view") != "rag"
        or len(scored) != 1200
        or metrics.get("predictions_sha256") != sha256_file(paths["predictions"])
        or manifest["run_config"]["input_sha256"] != sha256_file(paths["input"])
        or manifest["run_config"]["candidate_order"] != "original"
        or manifest["run_config"]["model_identity"]["adapter"]
        != str(ADAPTER_PATHS[model])
    ):
        raise ValueError(f"invalid development cell: model={model} K={candidate_k}")
    return {
        **compact_metrics(metrics),
        "batch_size": manifest["run_config"]["batch_size"],
        "peak_gpu_memory_gib": manifest["peak_gpu_memory_gib"],
        "paths": {name: relative(path) for name, path in paths.items()},
        "hashes": {name: sha256_file(path) for name, path in paths.items()},
    }


def development_results() -> dict[str, Any]:
    cells = {
        model: {str(k): load_dev_cell(model, k) for k in CANDIDATE_KS}
        for model in MODELS
    }
    best_trace = max(cells["8b"][str(k)]["trace_em"] for k in CANDIDATE_KS)
    best_upper = max(
        cells["8b"][str(k)]["retriever_upper_bound"] for k in CANDIDATE_KS
    )
    eligibility = {}
    for k in CANDIDATE_KS:
        row = cells["8b"][str(k)]
        eligibility[str(k)] = {
            "trace_within_tolerance": row["trace_em"] >= best_trace - TRACE_TOLERANCE,
            "retriever_upper_bound": row["retriever_upper_bound"],
            "retriever_gap_from_best": best_upper - row["retriever_upper_bound"],
        }
        eligibility[str(k)]["eligible"] = eligibility[str(k)][
            "trace_within_tolerance"
        ]
    eligible = [k for k in CANDIDATE_KS if eligibility[str(k)]["eligible"]]
    if not eligible:
        raise ValueError("selection rule produced no eligible candidate")
    selected_k = min(
        eligible,
        key=lambda k: (cells["8b"][str(k)]["mean_input_tokens"], k),
    )
    paired_vs_k10 = {
        model: {
            str(k): paired(
                dev_paths(model, 10)["scored"],
                dev_paths(model, k)["scored"],
                "trace_em",
            )
            for k in CANDIDATE_KS
            if k != 10
        }
        for model in MODELS
    }
    clustered_vs_k10 = {
        model: {
            str(k): clustered_paired(
                dev_paths(model, 10)["scored"],
                dev_paths(model, k)["scored"],
                "trace_em",
                20260801 + MODELS.index(model) * 100 + k,
            )
            for k in CANDIDATE_KS
            if k != 10
        }
        for model in MODELS
    }
    return {
        "cells": cells,
        "best_8b_trace_em": best_trace,
        "best_retriever_upper_bound": best_upper,
        "eligibility": eligibility,
        "selected_k": selected_k,
        "paired_trace_vs_k10": paired_vs_k10,
        "paired_trace_vs_k10_clustered_by_combo": clustered_vs_k10,
    }


def select() -> None:
    protocol = RUN_ROOT / "protocol.json"
    if not protocol.is_file():
        raise ValueError("run --stage begin before selection")
    development = development_results()
    metric_paths = [
        dev_paths(model, k)["metrics"] for model in MODELS for k in CANDIDATE_KS
    ]
    payload = {
        "schema_version": 1,
        "status": "locked_before_protected_split_challenger_inference",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": relative(protocol),
        "protocol_sha256": sha256_file(protocol),
        "selected_k": development["selected_k"],
        "selection_rule": load_json(protocol)["selection_rule"],
        "development": development,
        "development_metric_hashes": {
            relative(path): sha256_file(path) for path in metric_paths
        },
    }
    path = RUN_ROOT / "selection.lock.json"
    if path.exists():
        existing = load_json(path)
        for key in (
            "protocol_sha256",
            "selected_k",
            "selection_rule",
            "development_metric_hashes",
        ):
            if existing.get(key) != payload.get(key):
                raise ValueError(f"existing selection lock differs at {key}")
        print(f"selection already locked: K={existing['selected_k']}")
        return
    write_json(path, payload)
    print(f"development selection locked: K={payload['selected_k']} -> {path}")


def baseline_paths(model: str, split: str) -> dict[str, Path]:
    stem = f"reselection-crossstyle-{model}-{split}"
    paths = {
        "predictions": REPLAY_PRED_ROOT / f"{stem}.jsonl",
        "metrics": REPLAY_ROOT / "eval" / f"{stem}.metrics.json",
        "scored": REPLAY_ROOT / "eval" / f"{stem}.scored.jsonl",
    }
    if split == "legacy6000":
        paths.update(
            {
                "system": REPLAY_ROOT / "eval" / f"{stem}.system.jsonl",
                "system_metrics": REPLAY_ROOT
                / "eval"
                / f"{stem}.system.metrics.json",
            }
        )
    return paths


def challenger_paths(model: str, split: str, candidate_k: int) -> dict[str, Path]:
    stem = f"canonical-sft-{model}-{split}-k{candidate_k}"
    paths = {
        "input": RUN_ROOT
        / "inputs"
        / f"{split}_rag_k{candidate_k}_label_free.jsonl",
        "audit": RUN_ROOT / "inputs" / f"{split}_k{candidate_k}_audit.json",
        "ranking": RUN_ROOT / "protected_rankings" / split / "preds.jsonl",
        "ranking_metrics": RUN_ROOT
        / "protected_rankings"
        / split
        / "metrics.json",
        "predictions": PRED_ROOT / "protected" / f"{stem}.jsonl",
        "manifest": PRED_ROOT
        / "protected"
        / f"{stem}.shard0.jsonl.manifest.json",
        "metrics": RUN_ROOT / "protected" / f"{stem}.metrics.json",
        "scored": RUN_ROOT / "protected" / f"{stem}.scored.jsonl",
    }
    if split == "legacy6000":
        paths.update(
            {
                "system": RUN_ROOT / "protected" / f"{stem}.system.jsonl",
                "system_metrics": RUN_ROOT
                / "protected"
                / f"{stem}.system.metrics.json",
            }
        )
    return paths


def load_protected(
    paths: dict[str, Path], expected: int, candidate_k: int | None = None
) -> dict[str, Any]:
    missing = [relative(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ValueError(f"incomplete protected cell: {missing}")
    metrics = load_json(paths["metrics"])
    scored = load_jsonl(paths["scored"])
    if (
        metrics.get("status") != "complete"
        or metrics.get("count") != expected
        or len(scored) != expected
        or metrics.get("predictions_sha256") != sha256_file(paths["predictions"])
    ):
        raise ValueError(f"invalid protected metrics: {paths['metrics']}")
    if "audit" in paths:
        audit = load_json(paths["audit"])
        manifest = load_json(paths["manifest"])
        ranking_metrics = load_json(paths["ranking_metrics"])
        if (
            not audit.get("strict_ok")
            or audit.get("rag_rows") != expected
            or (
                candidate_k is not None
                and audit.get("candidate_k") != candidate_k
            )
            or manifest["run_config"]["input_sha256"] != sha256_file(paths["input"])
            or manifest["run_config"]["candidate_order"] != "original"
            or ranking_metrics.get("count") != expected
            or ranking_metrics.get("ranking_completion", {}).get("target_depth")
            != max(CANDIDATE_KS)
            or not ranking_metrics.get("ranking_completion", {}).get(
                "prefix_preserved"
            )
        ):
            raise ValueError(f"invalid protected input evidence: {paths['input']}")
    result = {
        "metrics": compact_metrics(metrics),
        "paths": {name: relative(path) for name, path in paths.items()},
        "hashes": {name: sha256_file(path) for name, path in paths.items()},
    }
    if "system_metrics" in paths:
        system = load_json(paths["system_metrics"])
        if system.get("status") != "complete" or system.get("count") != 9000:
            raise ValueError(f"invalid system metrics: {paths['system_metrics']}")
        result["system_metrics"] = {
            "full_set": system["full_pipeline_success"],
            "full_trace": system["full_pipeline_trace_success"],
        }
    return result


def complete() -> None:
    selection_path = RUN_ROOT / "selection.lock.json"
    if not selection_path.is_file():
        raise ValueError("run --stage select before completion")
    selection = load_json(selection_path)
    selected_k = int(selection["selected_k"])
    baseline = {
        model: {
            split: load_protected(baseline_paths(model, split), count)
            for split, count in SPLITS.items()
        }
        for model in MODELS
    }
    protected: dict[str, Any]
    if selected_k == 10:
        protected = {
            "mode": "reuse_existing_current_canonical_k10",
            "baseline_k10": baseline,
            "challenger_not_needed": True,
        }
    else:
        challenger = {
            model: {
                split: load_protected(
                    challenger_paths(model, split, selected_k), count, selected_k
                )
                for split, count in SPLITS.items()
            }
            for model in MODELS
        }
        comparisons: dict[str, Any] = {}
        for model in MODELS:
            comparisons[model] = {}
            for split in SPLITS:
                base_paths = baseline_paths(model, split)
                new_paths = challenger_paths(model, split, selected_k)
                comparisons[model][split] = {
                    field: paired(base_paths["scored"], new_paths["scored"], field)
                    for field in ("set_em", "trace_em", "strict_parse_ok")
                }

        protected = {
            "mode": "one_shot_selected_k_vs_existing_k10_for_reporting_only",
            "baseline_k10": baseline,
            "challenger": challenger,
            "paired_comparisons": comparisons,
        }
    final_k = selected_k
    decision = f"select_k{selected_k}_on_sft_combo_disjoint_development"
    payload = {
        "schema_version": 1,
        "status": "complete_no_post_protected_split_tuning",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "current-canonical Planner input candidate depth on SFT-combo-disjoint development",
        "protocol": relative(RUN_ROOT / "protocol.json"),
        "protocol_sha256": sha256_file(RUN_ROOT / "protocol.json"),
        "selection_lock": relative(selection_path),
        "selection_lock_sha256": sha256_file(selection_path),
        "development": selection["development"],
        "development_selected_k": selected_k,
        "protected_evaluation": protected,
        "post_selection_evaluation": {
            "applicable": selected_k != 10,
            "role": "reporting_only",
            "changed_selection": False,
        },
        "decision": decision,
        "final_planner_candidate_k": final_k,
    }
    write_json(RUN_ROOT / "summary.json", payload)
    print(
        json.dumps(
            {
                "development_selected_k": selected_k,
                "decision": decision,
                "final_planner_candidate_k": final_k,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("begin", "select", "complete"), required=True)
    args = parser.parse_args()
    {"begin": begin, "select": select, "complete": complete}[args.stage]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
