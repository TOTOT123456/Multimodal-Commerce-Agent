#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify and summarize the current canonical Base/SFT Planner matrix."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import PROJECT_ROOT, load_jsonl, read_json, write_json  # noqa: E402
from finetune.tool_trace_common import sha256_file  # noqa: E402


STAGES = ("base", "sft")
MODELS = ("4b", "8b")
VIEWS = ("global", "rag")
SPLITS = {"legacy6000": 6000, "api1000": 1000}

MATRIX_ROOT = (
    PROJECT_ROOT
    / "data"
    / "tool_call_trace"
    / "_aux"
    / "assignment3"
    / "canonical_base_sft_matrix_20260801"
)
PRED_ROOT = (
    PROJECT_ROOT
    / "finetune"
    / "outputs"
    / "preds"
    / "tool_trace"
    / "canonical_base_sft_matrix_20260801"
)
REPLAY_ROOT = (
    PROJECT_ROOT
    / "data"
    / "tool_call_trace"
    / "_aux"
    / "assignment3"
    / "retrieval_reselection_planner_replay_20260731"
)
REPLAY_PRED_ROOT = (
    PROJECT_ROOT
    / "finetune"
    / "outputs"
    / "preds"
    / "tool_trace"
    / "retrieval_reselection_planner_replay_20260731"
)
HISTORICAL_PRED_ROOT = (
    PROJECT_ROOT
    / "finetune"
    / "outputs"
    / "preds"
    / "tool_trace"
    / "assignment3_100tool_test"
)
DEPTH_ROOT = (
    PROJECT_ROOT
    / "data/tool_call_trace/_aux/assignment3"
    / "canonical_planner_candidate_depth_isolated_20260801"
)
DEPTH_PRED_ROOT = (
    PROJECT_ROOT
    / "finetune/outputs/preds/tool_trace"
    / "canonical_planner_candidate_depth_isolated_20260801"
)
ROUTE_PATH = PROJECT_ROOT / "configs" / "retrieval_canonical_route.json"
JOINT_ROOT = (
    PROJECT_ROOT
    / "data/tool_call_trace/_aux/assignment3/planner_joint_depth_reselection_20260801"
)
JOINT_PRED_ROOT = (
    PROJECT_ROOT
    / "finetune/outputs/preds/tool_trace/planner_joint_depth_reselection_20260801/protected"
)
SECOND_STAGE_ROOT = (
    PROJECT_ROOT
    / "data/tool_call_trace/_aux/assignment3/r20p10_second_rrf_optimization_20260801"
)
SECOND_STAGE_PRED_ROOT = (
    PROJECT_ROOT
    / "finetune/outputs/preds/tool_trace/r20p10_second_rrf_optimization_20260801"
)


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.parent.resolve()))
    except ValueError:
        return str(path)


def exact_mcnemar_p(improved: int, regressed: int) -> float:
    discordant = improved + regressed
    if discordant == 0:
        return 1.0
    tail = min(improved, regressed)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / (
        2**discordant
    )
    return min(1.0, 2.0 * probability)


def paths_for(
    split: str,
    stage: str,
    model: str,
    view: str,
    candidate_k: int,
    joint_route_id: str | None,
    promoted_route_id: str | None,
) -> tuple[Path, Path, Path, str]:
    if split == "api1000" and stage == "base" and view == "global":
        stem = f"a3-100-base-{model}-global"
        return (
            HISTORICAL_PRED_ROOT / f"{stem}.jsonl",
            HISTORICAL_PRED_ROOT / f"{stem}.metrics.json",
            HISTORICAL_PRED_ROOT / f"{stem}.scored.jsonl",
            "reused_route_independent",
        )
    if view == "rag" and promoted_route_id is not None:
        stem = f"{stage}{model.removesuffix('b')}_{promoted_route_id}"
        return (
            SECOND_STAGE_PRED_ROOT / "protected_after_promotion" / split / f"{stem}.jsonl",
            SECOND_STAGE_ROOT / "protected_after_promotion" / split / "eval" / f"{stem}.metrics.json",
            SECOND_STAGE_ROOT / "protected_after_promotion" / split / "eval" / f"{stem}.scored.jsonl",
            "new_post_second_stage_promotion_inference",
        )
    if view == "rag" and joint_route_id is not None:
        stem = f"joint-final-{stage}-{model}-{split}-{joint_route_id}"
        return (
            JOINT_PRED_ROOT / f"{stem}.jsonl",
            JOINT_ROOT / "protected" / split / "eval" / f"{stem}.metrics.json",
            JOINT_ROOT / "protected" / split / "eval" / f"{stem}.scored.jsonl",
            "new_joint_depth_current_canonical_inference",
        )
    if view == "rag" and candidate_k != 10:
        if stage == "sft":
            stem = f"canonical-sft-{model}-{split}-k{candidate_k}"
        else:
            stem = f"canonical-base-{model}-{split}-rag-k{candidate_k}"
        return (
            DEPTH_PRED_ROOT / "protected" / f"{stem}.jsonl",
            DEPTH_ROOT / "protected" / f"{stem}.metrics.json",
            DEPTH_ROOT / "protected" / f"{stem}.scored.jsonl",
            "new_current_canonical_candidate_depth_inference",
        )
    if stage == "sft" and view == "rag":
        stem = f"reselection-crossstyle-{model}-{split}"
        return (
            REPLAY_PRED_ROOT / f"{stem}.jsonl",
            REPLAY_ROOT / "eval" / f"{stem}.metrics.json",
            REPLAY_ROOT / "eval" / f"{stem}.scored.jsonl",
            "reused_current_canonical_replay",
        )
    stem = f"canonical-{stage}-{model}-{split}-{view}"
    return (
        PRED_ROOT / f"{stem}.jsonl",
        MATRIX_ROOT / "eval" / f"{stem}.metrics.json",
        MATRIX_ROOT / "eval" / f"{stem}.scored.jsonl",
        "new_inference",
    )


def load_cell(
    split: str,
    stage: str,
    model: str,
    view: str,
    candidate_k: int,
    joint_route_id: str | None,
    promoted_route_id: str | None,
) -> dict[str, Any]:
    predictions, metrics_path, scored_path, provenance = paths_for(
        split,
        stage,
        model,
        view,
        candidate_k,
        joint_route_id,
        promoted_route_id,
    )
    for path in (predictions, metrics_path, scored_path):
        if not path.is_file():
            raise ValueError(f"missing matrix artifact: {path}")
    metrics = read_json(metrics_path)
    scored = load_jsonl(scored_path)
    expected = SPLITS[split]
    if (
        metrics.get("status") != "complete"
        or metrics.get("count") != expected
        or len(scored) != expected
        or metrics.get("predictions_sha256") != sha256_file(predictions)
    ):
        raise ValueError(f"incomplete or inconsistent matrix cell: {metrics_path}")
    if metrics.get("view") != view:
        raise ValueError(f"view mismatch in {metrics_path}")
    overall = metrics["overall"]
    return {
        "split": split,
        "stage": stage,
        "model": model,
        "view": view,
        "provenance": provenance,
        "predictions": relative(predictions),
        "predictions_sha256": sha256_file(predictions),
        "metrics": relative(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
        "scored": relative(scored_path),
        "scored_sha256": sha256_file(scored_path),
        "gold_snapshot_sha256": metrics["gold_snapshot_sha256"],
        "count": expected,
        "set_em": overall["set_em"],
        "trace_em": overall["trace_em"],
        "strict_output_valid_rate": overall["strict_output_valid_rate"],
        "keyword_exact_accuracy": overall["keyword_exact_accuracy"],
        "keyword_micro_f1": overall["keyword_micro_f1"],
        "retriever_upper_bound": metrics["rag_decomposition"][
            "retriever_upper_bound"
        ],
        "conditional_set_accuracy": metrics["rag_decomposition"][
            "conditional_set_accuracy"
        ],
        "conditional_trace_accuracy": metrics["rag_decomposition"][
            "conditional_trace_accuracy"
        ],
    }


def paired(left_path: str, right_path: str, metric: str) -> dict[str, Any]:
    left_rows = load_jsonl(PROJECT_ROOT.parent / left_path)
    right_rows = load_jsonl(PROJECT_ROOT.parent / right_path)
    left = {row["id"]: row for row in left_rows}
    right = {row["id"]: row for row in right_rows}
    if len(left) != len(left_rows) or len(right) != len(right_rows) or left.keys() != right.keys():
        raise ValueError("paired matrix rows are not aligned")
    improved = sum(not bool(left[key][metric]) and bool(right[key][metric]) for key in left)
    regressed = sum(bool(left[key][metric]) and not bool(right[key][metric]) for key in left)
    delta = (improved - regressed) / len(left)
    return {
        "metric": metric,
        "count": len(left),
        "left": sum(bool(row[metric]) for row in left.values()) / len(left),
        "right": sum(bool(row[metric]) for row in right.values()) / len(right),
        "delta": delta,
        "improved": improved,
        "regressed": regressed,
        "exact_mcnemar_p": exact_mcnemar_p(improved, regressed),
    }


def main() -> int:
    route_path = ROUTE_PATH
    route = read_json(route_path)
    candidate_k = int(route["planner"]["candidate_k"])
    expected_joint_route_id = (
        f"rerank{int(route['reranker']['candidate_k'])}_planner{candidate_k}"
    )
    joint_report_path = JOINT_ROOT / "final_report.json"
    joint_route_id = None
    promoted_route_id = None
    if route.get("canonical_revision") == "r20p10_w040_k008":
        promoted_route_id = "r20p10_w040_k008"
    if joint_report_path.is_file():
        joint_report = read_json(joint_report_path)
        if (
            joint_report.get("status")
            == "complete_after_locked_route_on_protected_splits"
            and joint_report.get("route_id") == expected_joint_route_id
        ):
            joint_route_id = expected_joint_route_id
    if (
        route.get("status") != "canonical"
        or route["second_stage"]["output_k"] != 10
        or candidate_k not in {5, 10, 20, 30, 50}
        or route["planner"]["decoding"] != "greedy"
    ):
        raise ValueError("retrieval_canonical_route.json is not the expected route")

    cells: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for split in SPLITS:
        cells[split] = {}
        for stage in STAGES:
            cells[split][stage] = {}
            for model in MODELS:
                cells[split][stage][model] = {}
                for view in VIEWS:
                    cells[split][stage][model][view] = load_cell(
                        split,
                        stage,
                        model,
                        view,
                        candidate_k,
                        joint_route_id,
                        promoted_route_id,
                    )

    comparisons: dict[str, Any] = {}
    for split in SPLITS:
        comparisons[split] = {"rag_vs_wo_rag": {}, "sft_vs_base": {}}
        for stage in STAGES:
            comparisons[split]["rag_vs_wo_rag"][stage] = {}
            for model in MODELS:
                left = cells[split][stage][model]["global"]
                right = cells[split][stage][model]["rag"]
                comparisons[split]["rag_vs_wo_rag"][stage][model] = {
                    metric: paired(left["scored"], right["scored"], metric)
                    for metric in ("set_em", "trace_em")
                }
        for view in VIEWS:
            comparisons[split]["sft_vs_base"][view] = {}
            for model in MODELS:
                left = cells[split]["base"][model][view]
                right = cells[split]["sft"][model][view]
                comparisons[split]["sft_vs_base"][view][model] = {
                    metric: paired(left["scored"], right["scored"], metric)
                    for metric in ("set_em", "trace_em")
                }

    gold_hashes = {
        split: sorted(
            {
                cells[split][stage][model][view]["gold_snapshot_sha256"]
                for stage in STAGES
                for model in MODELS
                for view in VIEWS
            }
        )
        for split in SPLITS
    }
    if any(len(values) != 1 for values in gold_hashes.values()):
        raise ValueError("matrix cells do not share one gold snapshot per split")

    summary = {
        "schema_version": 1,
        "status": "complete",
        "scope": (
            "Base/SFT x Qwen3-4B/8B x Wo-RAG(full catalog)/current "
            f"canonical RAG@{candidate_k}; DPO excluded"
        ),
        "route": {
            "path": relative(route_path),
            "sha256": sha256_file(route_path),
            "embedding": route["embedding"],
            "first_stage": route["first_stage"],
            "reranker": route["reranker"],
            "second_stage": route["second_stage"],
            "planner_interface": {
                "candidate_k": route["planner"]["candidate_k"],
                "candidate_order": route["planner"]["candidate_order"],
                "decoding": route["planner"]["decoding"],
            },
        },
        "definitions": {
            "wo_rag": "global full-catalog prompt containing all 100 tools",
            "rag": (
                f"current canonical retrieval-ordered Top-{candidate_k} prompt"
            ),
            "base": "unadapted local Qwen3 base checkpoint",
            "sft": "cross-style RAG@10 SFT checkpoint-2429",
        },
        "split_counts": SPLITS,
        "gold_snapshot_sha256": {
            split: values[0] for split, values in gold_hashes.items()
        },
        "cells": cells,
        "paired_comparisons": comparisons,
        "reuse_policy": {
            "api_base_wo_rag": "reused because query input, model, catalog, prompt, greedy decoding, and gold snapshot are identical and this view is retrieval-independent",
            "rag_cells": (
                "rerun after the locked R20/P10 second-stage promotion"
                if promoted_route_id is not None
                else "rerun under the locked coherent joint Reranker/Planner route"
                if joint_route_id is not None
                else (
                    "reused from the completed K=10 replay"
                    if candidate_k == 10
                    else "rerun at the earlier isolated-development candidate depth"
                )
            ),
            "all_other_cells": "new or unchanged route-independent inference under this matrix run",
        },
    }
    MATRIX_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(MATRIX_ROOT / "summary.json", summary)
    print(json.dumps({
        split: {
            stage: {
                model: {
                    view: {
                        "set_em": cells[split][stage][model][view]["set_em"],
                        "trace_em": cells[split][stage][model][view]["trace_em"],
                        "strict": cells[split][stage][model][view]["strict_output_valid_rate"],
                    }
                    for view in VIEWS
                }
                for model in MODELS
            }
            for stage in STAGES
        }
        for split in SPLITS
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
