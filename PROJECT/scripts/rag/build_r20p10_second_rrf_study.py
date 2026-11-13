#!/usr/bin/env python3
"""Prepare and lock a finite R20/P10 second-stage RRF study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from search_retrieval_pipeline import (  # noqa: E402
    assert_aligned,
    load_jsonl,
    metrics,
    second_rrf,
    write_json,
    write_jsonl,
)


SOURCE_WEIGHTS = tuple(index / 10 for index in range(11))
RRF_KS = (1, 2, 3, 5, 8, 10, 20, 30, 60, 100)
PANEL_WEIGHTS = {"legacy_like": 6.0 / 7.0, "api_like": 1.0 / 7.0}
STYLE_TO_PANEL = {"legacy_compact": "legacy_like", "api_native": "api_like"}
KNOWN_COMPARATORS = {
    "current_r20p10": {
        "reranker_depth": 20,
        "planner_depth": 10,
        "source_weight": 0.4,
        "rrf_k": 2,
        "selection_weighted_trace_em": 0.9935714285714285,
    },
    "raw_best_r30p20": {
        "reranker_depth": 30,
        "planner_depth": 20,
        "source_weight": 0.1,
        "rrf_k": 8,
        "selection_weighted_trace_em": 0.9964285714285714,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def gold_tools(row: dict[str, Any]) -> list[str]:
    value = row.get("工具", row.get("gold_tools", []))
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item) for item in value]


def ranking_hash(rankings: list[list[str]], depth: int) -> str:
    digest = hashlib.sha256()
    for ranking in rankings:
        digest.update("\x1f".join(ranking[:depth]).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def route_id(source_weight: float, rrf_k: int) -> str:
    return f"r20p10_w{round(source_weight * 100):03d}_k{rrf_k:03d}"


def aligned_inputs(
    dataset_path: Path, source_path: Path, reranker_path: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    dataset = load_jsonl(dataset_path)
    source = load_jsonl(source_path)
    reranker = load_jsonl(reranker_path)
    assert_aligned(source, reranker, "source", "reranker")
    by_id = {str(row["id"]): row for row in dataset}
    if len(by_id) != len(dataset) or set(by_id) != {str(row["id"]) for row in source}:
        raise ValueError("dataset/source ids are not exactly aligned")
    ordered = [by_id[str(row["id"])] for row in source]
    for meta, source_row, reranker_row in zip(ordered, source, reranker):
        query = str(meta.get("问题", meta.get("query", "")))
        if query != source_row.get("query") or query != reranker_row.get("query"):
            raise ValueError(f"query mismatch for {meta.get('id')}")
        if len(source_row.get("pred_tools", [])) < 20:
            raise ValueError(f"source depth below 20 for {meta.get('id')}")
        if int(reranker_row.get("candidate_count", 0)) != 20:
            raise ValueError(f"reranker artifact is not exact Top-20 for {meta.get('id')}")
        if set(reranker_row.get("pred_tools", [])) != set(source_row["pred_tools"][:20]):
            raise ValueError(f"reranker universe mismatch for {meta.get('id')}")
    return ordered, source, reranker


def score_route(
    dataset: list[dict[str, Any]], rankings: list[list[str]]
) -> dict[str, Any]:
    metric_rows = [
        {"id": row["id"], "gold_tools": gold_tools(row)} for row in dataset
    ]
    indices: dict[str, list[int]] = defaultdict(list)
    resolved_panels: list[str | None] = []
    for row in dataset:
        construction = row.get("construction")
        value = construction.get("panel") if isinstance(construction, dict) else None
        if value is not None and value not in PANEL_WEIGHTS:
            raise ValueError(f"row {row.get('id')} has an unsupported panel: {value}")
        resolved_panels.append(str(value) if value is not None else None)
    if any(panel is None for panel in resolved_panels) and not all(
        panel is None for panel in resolved_panels
    ):
        raise ValueError("panel metadata is only partially populated")
    for index, panel in enumerate(resolved_panels):
        indices[panel or "overall"].append(index)
    by_panel = {}
    for panel, selected in sorted(indices.items()):
        by_panel[panel] = {
            "count": len(selected),
            **metrics(
                [metric_rows[index] for index in selected],
                [rankings[index] for index in selected],
                [6, 10, 20],
            ),
        }
    overall = metrics(metric_rows, rankings, [6, 10, 20])
    if set(PANEL_WEIGHTS) <= set(by_panel):
        weighted = {
            name: sum(
                PANEL_WEIGHTS[panel] * float(by_panel[panel][name])
                for panel in PANEL_WEIGHTS
            )
            for name in ("allhit_at_6", "allhit_at_10", "allhit_at_20", "mrr")
        }
        weighting = "6:1 legacy_like/api_like"
    else:
        weighted = {
            name: float(overall[name])
            for name in ("allhit_at_6", "allhit_at_10", "allhit_at_20", "mrr")
        }
        weighting = "overall because both deployment panels are not present"
    return {
        "overall": overall,
        "by_panel": by_panel,
        "deployment_weighted": weighted,
        "deployment_weighting": weighting,
    }


def prepare(args: argparse.Namespace) -> int:
    dataset, source, reranker = aligned_inputs(
        args.dataset, args.source_preds, args.reranker20_preds
    )
    variants = []
    seen_prefixes: set[str] = set()
    for source_weight in SOURCE_WEIGHTS:
        for rrf_k in RRF_KS:
            rankings = [
                second_rrf(
                    source_row["pred_tools"][:20],
                    reranker_row["pred_tools"],
                    source_weight,
                    rrf_k,
                )
                for source_row, reranker_row in zip(source, reranker)
            ]
            prefix_hash = ranking_hash(rankings, 10)
            route_metrics = score_route(dataset, rankings)
            perfect_panel_allhit = all(
                math.isclose(
                    float(route_metrics["by_panel"][panel]["allhit_at_10"]),
                    1.0,
                    abs_tol=1e-12,
                )
                for panel in PANEL_WEIGHTS
            )
            if not perfect_panel_allhit or prefix_hash in seen_prefixes:
                continue
            seen_prefixes.add(prefix_hash)
            identifier = route_id(source_weight, rrf_k)
            ranking_path = args.out_dir / "selection" / "rankings" / f"{identifier}.jsonl"
            write_jsonl(
                ranking_path,
                [
                    {
                        "id": row["id"],
                        "query": str(row.get("问题", row.get("query", ""))),
                        "gold_tools": gold_tools(row),
                        "pred_tools": ranking,
                        "source_tools": source_row["pred_tools"][:20],
                        "reranked_tools": reranker_row["pred_tools"],
                    }
                    for row, source_row, reranker_row, ranking in zip(
                        dataset, source, reranker, rankings
                    )
                ],
            )
            variants.append(
                {
                    "id": identifier,
                    "reranker_depth": 20,
                    "planner_depth": 10,
                    "second_stage": {
                        "source_weight": source_weight,
                        "rrf_k": rrf_k,
                    },
                    "retrieval": route_metrics,
                    "top10_ranking_sha256": prefix_hash,
                    "rankings": str(ranking_path),
                    "rankings_sha256": sha256_file(ranking_path),
                }
            )
    variants.sort(
        key=lambda row: (
            float(row["second_stage"]["source_weight"]),
            int(row["second_stage"]["rrf_k"]),
        )
    )
    if len(variants) != 11:
        raise ValueError(f"expected exactly 11 preregistered eligible variants, got {len(variants)}")
    routes_path = args.out_dir / "selection" / "routes.json"
    write_json(
        routes_path,
        {
            "schema_version": 1,
            "status": "complete_before_variant_planner_inference",
            "eligibility": "unique Top-10 ranking and AllHit@10=1.0 in both open selection panels",
            "grid": {
                "source_weights": list(SOURCE_WEIGHTS),
                "rrf_ks": list(RRF_KS),
                "raw_combinations": len(SOURCE_WEIGHTS) * len(RRF_KS),
                "eligible_unique_variants": len(variants),
            },
            "variants": variants,
        },
    )
    protocol = {
        "schema_version": 1,
        "status": "preregistered_before_new_variant_planner_inference",
        "created_at_utc": utc_now(),
        "objective": (
            "test whether the previously retrieval-only R20/P10 residual-RRF choice can "
            "be improved at fixed model weights and fixed R20/P10 compute"
        ),
        "disclosure": {
            "current_r20p10_and_r30p20_planner_results_were_known": True,
            "known_comparators": KNOWN_COMPARATORS,
            "no_other_eligible_variant_has_been_run_through_the_planner": True,
        },
        "candidate_rule": {
            "finite_grid": "11 source weights x 10 RRF constants",
            "eligibility": (
                "retain each unique Top-10 ranking whose AllHit@10 is exactly 1.0 "
                "in both open selection panels"
            ),
            "eligible_count": len(variants),
            "eligible_ids": [row["id"] for row in variants],
            "no_early_stopping": True,
        },
        "selection_rule": {
            "primary": "6:1 legacy_like/api_like weighted Planner TraceEM",
            "tie_breaks": [
                "minimum panel TraceEM",
                "overall strict output valid rate",
                "SFT-unseen TraceEM",
                "smaller source weight",
                "smaller RRF k",
            ],
            "selection_is_tuning_only": True,
        },
        "new_confirmation_gate": {
            "must_be_locked_before_confirmation_retrieval_or_planner_inference": True,
            "comparators": ["current_r20p10", "raw_best_r30p20"],
            "raw_best_claim": (
                "allowed only if selected R20/P10 weighted TraceEM is no lower than "
                "R30/P20 on the untouched confirmation snapshot"
            ),
            "reasonable_deployment_rule": (
                "independent of raw-best claim: R20/P10 may remain preferred when within "
                "0.5 percentage points because R30/P20 costs 58.35% more Planner tokens "
                "and 50.74% more reranker p50 latency on the fixed hardware benchmark"
            ),
            "protected_legacy_api_and_test6000_cannot_tune_or_select": True,
            "report_all_outcomes": True,
        },
        "inputs": {
            str(path): sha256_file(path)
            for path in (
                args.dataset,
                args.source_preds,
                args.reranker20_preds,
            )
        },
        "routes": str(routes_path),
        "routes_sha256": sha256_file(routes_path),
    }
    write_json(args.out_dir / "protocol.json", protocol)
    print(f"R20/P10 eligible variants={len(variants)} -> {args.out_dir / 'protocol.json'}")
    return 0


def metric_cell(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise ValueError(f"incomplete metric file: {path}")
    style = payload["buckets"]["style"]
    panel_trace = {
        panel: float(style[style_name]["trace_em"])
        for style_name, panel in STYLE_TO_PANEL.items()
    }
    combo = payload["buckets"]["combo"]
    return {
        "metrics": str(path),
        "metrics_sha256": sha256_file(path),
        "overall_trace_em": float(payload["overall"]["trace_em"]),
        "overall_set_em": float(payload["overall"]["set_em"]),
        "overall_strict": float(payload["overall"]["strict_output_valid_rate"]),
        "panel_trace_em": panel_trace,
        "minimum_panel_trace_em": min(panel_trace.values()),
        "deployment_weighted_trace_em": sum(
            PANEL_WEIGHTS[panel] * panel_trace[panel] for panel in PANEL_WEIGHTS
        ),
        "sft_unseen_trace_em": float(combo["sft_unseen"]["trace_em"]),
        "mean_input_tokens": float(
            payload.get("efficiency", {}).get("input_tokens", {}).get("mean", 0.0)
        ),
    }


def select(args: argparse.Namespace) -> int:
    protocol_path = args.out_dir / "protocol.json"
    routes_path = args.out_dir / "selection" / "routes.json"
    if not protocol_path.exists() or not routes_path.exists():
        raise ValueError("prepare stage has not completed")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "preregistered_before_new_variant_planner_inference":
        raise ValueError("unexpected protocol status")
    routes = json.loads(routes_path.read_text(encoding="utf-8"))["variants"]
    cells = []
    for route in routes:
        metric_path = args.metrics_dir / f"{route['id']}.metrics.json"
        if not metric_path.exists():
            raise ValueError(f"missing preregistered metric: {metric_path}")
        cells.append({"route": route, **metric_cell(metric_path)})
    ranked = sorted(
        cells,
        key=lambda row: (
            -row["deployment_weighted_trace_em"],
            -row["minimum_panel_trace_em"],
            -row["overall_strict"],
            -row["sft_unseen_trace_em"],
            float(row["route"]["second_stage"]["source_weight"]),
            int(row["route"]["second_stage"]["rrf_k"]),
        ),
    )
    selected = ranked[0]
    lock = {
        "schema_version": 1,
        "status": "locked_before_new_confirmation_retrieval_or_planner_inference",
        "locked_at_utc": utc_now(),
        "protocol": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "routes": str(routes_path),
        "routes_sha256": sha256_file(routes_path),
        "selected_route_id": selected["route"]["id"],
        "selected_route": selected["route"],
        "selected_metrics": {
            key: value for key, value in selected.items() if key != "route"
        },
        "ranked_candidates": [
            {
                "rank": index,
                "id": row["route"]["id"],
                "second_stage": row["route"]["second_stage"],
                **{key: value for key, value in row.items() if key != "route"},
            }
            for index, row in enumerate(ranked, 1)
        ],
        "known_comparators": KNOWN_COMPARATORS,
        "protected_results_opened": False,
    }
    write_json(args.out_dir / "selection.lock.json", lock)
    print(
        f"selected={lock['selected_route_id']} "
        f"weighted_trace={selected['deployment_weighted_trace_em']:.9f}"
    )
    return 0


def validate_exact_reranker(
    dataset: list[dict[str, Any]],
    source: list[dict[str, Any]],
    reranker_path: Path,
    depth: int,
) -> list[dict[str, Any]]:
    reranker = load_jsonl(reranker_path)
    assert_aligned(source, reranker, "source", f"reranker{depth}")
    for meta, source_row, reranker_row in zip(dataset, source, reranker):
        query = str(meta.get("问题", meta.get("query", "")))
        if query != reranker_row.get("query"):
            raise ValueError(f"reranker{depth} query mismatch for {meta.get('id')}")
        if int(reranker_row.get("candidate_count", 0)) != depth:
            raise ValueError(f"reranker artifact is not exact Top-{depth}")
        if set(reranker_row.get("pred_tools", [])) != set(source_row["pred_tools"][:depth]):
            raise ValueError(f"reranker{depth} universe mismatch for {meta.get('id')}")
    return reranker


def apply_confirmation(args: argparse.Namespace) -> int:
    lock_path = args.out_dir / "selection.lock.json"
    if not lock_path.exists():
        raise ValueError("selection must be locked before confirmation application")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "locked_before_new_confirmation_retrieval_or_planner_inference":
        raise ValueError("unexpected selection lock status")
    dataset, source, reranker20 = aligned_inputs(
        args.dataset, args.source_preds, args.reranker20_preds
    )
    reranker30 = validate_exact_reranker(
        dataset, source, args.reranker30_preds, 30
    )
    selected_stage = lock["selected_route"]["second_stage"]
    specifications = [
        {
            "id": "selected_r20p10",
            "selection_route_id": lock["selected_route_id"],
            "reranker_depth": 20,
            "planner_depth": 10,
            "source_weight": float(selected_stage["source_weight"]),
            "rrf_k": int(selected_stage["rrf_k"]),
            "reranker": reranker20,
        },
        {
            "id": "current_r20p10",
            "reranker_depth": 20,
            "planner_depth": 10,
            "source_weight": 0.4,
            "rrf_k": 2,
            "reranker": reranker20,
        },
        {
            "id": "raw_best_r30p20",
            "reranker_depth": 30,
            "planner_depth": 20,
            "source_weight": 0.1,
            "rrf_k": 8,
            "reranker": reranker30,
        },
    ]
    confirmation_dir = args.confirmation_dir or (args.out_dir / "confirmation")
    route_rows = []
    for specification in specifications:
        depth = int(specification["reranker_depth"])
        rankings = [
            second_rrf(
                source_row["pred_tools"][:depth],
                reranker_row["pred_tools"],
                float(specification["source_weight"]),
                int(specification["rrf_k"]),
            )
            for source_row, reranker_row in zip(source, specification["reranker"])
        ]
        ranking_path = (
            confirmation_dir
            / "rankings"
            / f"{specification['id']}.jsonl"
        )
        write_jsonl(
            ranking_path,
            [
                {
                    "id": row["id"],
                    "query": str(row.get("问题", row.get("query", ""))),
                    "gold_tools": gold_tools(row),
                    "pred_tools": ranking,
                    "source_tools": source_row["pred_tools"][:depth],
                    "reranked_tools": reranker_row["pred_tools"],
                }
                for row, source_row, reranker_row, ranking in zip(
                    dataset, source, specification["reranker"], rankings
                )
            ],
        )
        route_rows.append(
            {
                key: value
                for key, value in specification.items()
                if key != "reranker"
            }
            | {
                "retrieval": score_route(dataset, rankings),
                "planner_prefix_outside_reranker_count": sum(
                    not set(ranking[: int(specification["planner_depth"])])
                    <= set(reranker_row["pred_tools"])
                    for ranking, reranker_row in zip(
                        rankings, specification["reranker"]
                    )
                ),
                "rankings": str(ranking_path),
                "rankings_sha256": sha256_file(ranking_path),
            }
        )
    if any(row["planner_prefix_outside_reranker_count"] for row in route_rows):
        raise ValueError("confirmation Planner prefix contains an unreranked tool")
    payload = {
        "schema_version": 1,
        "status": "complete_after_selection_lock",
        "selection_lock": str(lock_path),
        "selection_lock_sha256": sha256_file(lock_path),
        "inputs": {
            str(path): sha256_file(path)
            for path in (
                args.dataset,
                args.source_preds,
                args.reranker20_preds,
                args.reranker30_preds,
            )
        },
        "routes": route_rows,
    }
    write_json(confirmation_dir / "routes.json", payload)
    print(
        "confirmation routes="
        + ",".join(row["id"] for row in route_rows)
    )
    return 0


def apply_selected_route(args: argparse.Namespace) -> int:
    lock_path = args.out_dir / "selection.lock.json"
    if not lock_path.exists():
        raise ValueError("selection must be locked before applying the selected route")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    dataset, source, reranker = aligned_inputs(
        args.dataset, args.source_preds, args.reranker20_preds
    )
    stage = lock["selected_route"]["second_stage"]
    rankings = [
        second_rrf(
            source_row["pred_tools"][:20],
            reranker_row["pred_tools"],
            float(stage["source_weight"]),
            int(stage["rrf_k"]),
        )
        for source_row, reranker_row in zip(source, reranker)
    ]
    write_jsonl(
        args.rankings_out,
        [
            {
                "id": row["id"],
                "query": str(row.get("问题", row.get("query", ""))),
                "gold_tools": gold_tools(row),
                "pred_tools": ranking,
                "source_tools": source_row["pred_tools"][:20],
                "reranked_tools": reranker_row["pred_tools"],
            }
            for row, source_row, reranker_row, ranking in zip(
                dataset, source, reranker, rankings
            )
        ],
    )
    sidecar = {
        "schema_version": 1,
        "status": "complete_after_selection_lock",
        "selection_lock": str(lock_path),
        "selection_lock_sha256": sha256_file(lock_path),
        "route_id": lock["selected_route_id"],
        "reranker_depth": 20,
        "planner_depth": 10,
        "second_stage": stage,
        "retrieval": score_route(dataset, rankings),
        "inputs": {
            str(path): sha256_file(path)
            for path in (args.dataset, args.source_preds, args.reranker20_preds)
        },
        "rankings": str(args.rankings_out),
        "rankings_sha256": sha256_file(args.rankings_out),
    }
    sidecar_path = args.rankings_out.with_suffix(".metrics.json")
    write_json(sidecar_path, sidecar)
    print(
        f"applied selected {lock['selected_route_id']} -> {args.rankings_out}"
    )
    return 0


def exact_mcnemar_pvalue(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = min(left_only, right_only)
    probability = sum(
        math.comb(discordant, index) for index in range(tail + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * probability)


def paired_trace(left_path: Path, right_path: Path) -> dict[str, Any]:
    left = load_jsonl(left_path)
    right = load_jsonl(right_path)
    assert_aligned(left, right, "left_scored", "right_scored")
    left_only = sum(
        bool(left_row["trace_em"]) and not bool(right_row["trace_em"])
        for left_row, right_row in zip(left, right)
    )
    right_only = sum(
        bool(right_row["trace_em"]) and not bool(left_row["trace_em"])
        for left_row, right_row in zip(left, right)
    )
    return {
        "rows": len(left),
        "left_correct_right_wrong": left_only,
        "right_correct_left_wrong": right_only,
        "net_rows": left_only - right_only,
        "exact_mcnemar_pvalue": exact_mcnemar_pvalue(left_only, right_only),
        "left_scored_sha256": sha256_file(left_path),
        "right_scored_sha256": sha256_file(right_path),
    }


def paired_boolean(
    left_path: Path, right_path: Path, key: str
) -> dict[str, Any]:
    left = load_jsonl(left_path)
    right = load_jsonl(right_path)
    assert_aligned(left, right, "left_rows", "right_rows")
    left_only = sum(
        bool(left_row[key]) and not bool(right_row[key])
        for left_row, right_row in zip(left, right)
    )
    right_only = sum(
        bool(right_row[key]) and not bool(left_row[key])
        for left_row, right_row in zip(left, right)
    )
    return {
        "rows": len(left),
        "left_correct_right_wrong": left_only,
        "right_correct_left_wrong": right_only,
        "net_rows": left_only - right_only,
        "exact_mcnemar_pvalue": exact_mcnemar_pvalue(left_only, right_only),
        "left_sha256": sha256_file(left_path),
        "right_sha256": sha256_file(right_path),
    }


def summarize_confirmation(args: argparse.Namespace) -> int:
    lock_path = args.out_dir / "selection.lock.json"
    confirmation_dir = args.confirmation_dir or (args.out_dir / "confirmation")
    routes_path = confirmation_dir / "routes.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    routes = json.loads(routes_path.read_text(encoding="utf-8"))
    cells = {}
    for route in routes["routes"]:
        identifier = route["id"]
        metric_path = args.metrics_dir / f"{identifier}.metrics.json"
        scored_path = args.metrics_dir / f"{identifier}.scored.jsonl"
        if not metric_path.exists() or not scored_path.exists():
            raise ValueError(f"missing confirmation output for {identifier}")
        cells[identifier] = {
            "route": {key: value for key, value in route.items() if key != "retrieval"},
            "retrieval": route["retrieval"],
            **metric_cell(metric_path),
            "scored": str(scored_path),
            "scored_sha256": sha256_file(scored_path),
        }
    selected = cells["selected_r20p10"]
    current = cells["current_r20p10"]
    raw_best = cells["raw_best_r30p20"]
    raw_best_supported = (
        selected["deployment_weighted_trace_em"]
        >= raw_best["deployment_weighted_trace_em"]
    )
    selected_panel_guard = all(
        selected["panel_trace_em"][panel]
        >= current["panel_trace_em"][panel] - 0.005
        for panel in PANEL_WEIGHTS
    )
    selected_variant_promoted = (
        selected["deployment_weighted_trace_em"]
        >= current["deployment_weighted_trace_em"]
        and selected_panel_guard
    )
    selected_scored = Path(selected["scored"])
    report = {
        "schema_version": 1,
        "status": "complete_no_protected_split_tuning",
        "selection_lock": str(lock_path),
        "selection_lock_sha256": sha256_file(lock_path),
        "confirmation_routes": str(routes_path),
        "confirmation_routes_sha256": sha256_file(routes_path),
        "cells": cells,
        "paired": {
            "selected_vs_current": paired_trace(
                selected_scored, Path(current["scored"])
            ),
            "selected_vs_raw_best_r30p20": paired_trace(
                selected_scored, Path(raw_best["scored"])
            ),
        },
        "gates": {
            "selected_variant_no_worse_than_current_weighted": (
                selected["deployment_weighted_trace_em"]
                >= current["deployment_weighted_trace_em"]
            ),
            "selected_variant_panel_guard": selected_panel_guard,
            "selected_variant_promoted": selected_variant_promoted,
            "r20p10_raw_best_supported_on_confirmation": raw_best_supported,
        },
        "decision": {
            "r20p10_remains_reasonable_compute_quality_choice": (
                selected["deployment_weighted_trace_em"]
                >= raw_best["deployment_weighted_trace_em"] - 0.005
            ),
            "raw_best_claim_allowed": raw_best_supported,
            "second_stage": (
                lock["selected_route"]["second_stage"]
                if selected_variant_promoted
                else KNOWN_COMPARATORS["current_r20p10"]
            ),
            "protected_legacy_api_results_used_for_decision": False,
        },
    }
    write_json(confirmation_dir / "final_report.json", report)
    print(
        f"confirmation selected={selected['deployment_weighted_trace_em']:.9f} "
        f"current={current['deployment_weighted_trace_em']:.9f} "
        f"r30p20={raw_best['deployment_weighted_trace_em']:.9f} "
        f"raw_best_supported={raw_best_supported}"
    )
    return 0


def protected_metric(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise ValueError(f"incomplete protected metric: {path}")
    return {
        "trace_em": float(payload["overall"]["trace_em"]),
        "set_em": float(payload["overall"]["set_em"]),
        "strict": float(payload["overall"]["strict_output_valid_rate"]),
        "mean_input_tokens": float(payload["efficiency"]["input_tokens"]["mean"]),
        "path": str(path),
        "sha256": sha256_file(path),
    }


def summarize_protected(args: argparse.Namespace) -> int:
    if args.baseline_protected_dir is None:
        raise ValueError("summarize-protected requires --baseline-protected-dir")
    promoted_root = args.out_dir / "protected_after_promotion"
    baseline_root = args.baseline_protected_dir
    confirmation_path = args.out_dir / "sealed_confirmation" / "final_report.json"
    confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
    if not confirmation["gates"]["selected_variant_promoted"]:
        raise ValueError("the selected variant did not pass the sealed promotion gate")

    splits = {}
    for split in ("legacy6000", "api1000"):
        promoted_eval = promoted_root / split / "eval"
        baseline_eval = baseline_root / split / "eval"
        promoted_metric_path = promoted_eval / "sft8_r20p10_w040_k008.metrics.json"
        baseline_stem = f"joint-final-sft-8b-{split}-rerank20_planner10"
        baseline_metric_path = baseline_eval / f"{baseline_stem}.metrics.json"
        promoted_scored = promoted_eval / "sft8_r20p10_w040_k008.scored.jsonl"
        baseline_scored = baseline_eval / f"{baseline_stem}.scored.jsonl"
        promoted_route_path = (
            promoted_root
            / split
            / "retrieval/rankings/r20p10_w040_k008.metrics.json"
        )
        baseline_routes_path = baseline_root / split / "retrieval/routes.json"
        promoted_route = json.loads(promoted_route_path.read_text(encoding="utf-8"))
        baseline_route = json.loads(baseline_routes_path.read_text(encoding="utf-8"))["routes"][0]
        promoted_metric = protected_metric(promoted_metric_path)
        baseline_metric = protected_metric(baseline_metric_path)
        splits[split] = {
            "retrieval": {
                "baseline_allhit_at_10": float(baseline_route["overall"]["allhit_at_10"]),
                "promoted_allhit_at_10": float(
                    promoted_route["retrieval"]["overall"]["allhit_at_10"]
                ),
                "delta": float(
                    promoted_route["retrieval"]["overall"]["allhit_at_10"]
                )
                - float(baseline_route["overall"]["allhit_at_10"]),
                "promoted_route": str(promoted_route_path),
                "promoted_route_sha256": sha256_file(promoted_route_path),
                "baseline_routes": str(baseline_routes_path),
                "baseline_routes_sha256": sha256_file(baseline_routes_path),
            },
            "planner": {
                "baseline": baseline_metric,
                "promoted": promoted_metric,
                "trace_em_delta": promoted_metric["trace_em"] - baseline_metric["trace_em"],
                "paired_promoted_vs_baseline": paired_trace(
                    promoted_scored, baseline_scored
                ),
            },
        }

    round_promoted_path = (
        promoted_root
        / "round_final/retrieval/rankings/r20p10_w040_k008.metrics.json"
    )
    round_baseline_path = baseline_root / "round_final/retrieval/routes.json"
    round_promoted = json.loads(round_promoted_path.read_text(encoding="utf-8"))
    round_baseline = json.loads(round_baseline_path.read_text(encoding="utf-8"))["routes"][0]

    promoted_system_path = (
        promoted_root / "legacy6000/eval/sft8_r20p10_w040_k008.system.metrics.json"
    )
    promoted_system_rows = (
        promoted_root / "legacy6000/eval/sft8_r20p10_w040_k008.system.jsonl"
    )
    baseline_system_path = (
        baseline_root
        / "legacy6000/eval/joint-final-sft-8b-legacy6000-rerank20_planner10.system.metrics.json"
    )
    baseline_system_rows = (
        baseline_root
        / "legacy6000/eval/joint-final-sft-8b-legacy6000-rerank20_planner10.system.jsonl"
    )
    promoted_system = json.loads(promoted_system_path.read_text(encoding="utf-8"))
    baseline_system = json.loads(baseline_system_path.read_text(encoding="utf-8"))

    payload = {
        "schema_version": 1,
        "status": "complete_lock_after_promotion_reporting_only",
        "protected_splits_used_for_selection": False,
        "selection_lock": str(args.out_dir / "selection.lock.json"),
        "selection_lock_sha256": sha256_file(args.out_dir / "selection.lock.json"),
        "sealed_confirmation": str(confirmation_path),
        "sealed_confirmation_sha256": sha256_file(confirmation_path),
        "splits": splits,
        "round_final_retrieval": {
            "baseline_allhit_at_10": float(round_baseline["overall"]["allhit_at_10"]),
            "promoted_allhit_at_10": float(
                round_promoted["retrieval"]["overall"]["allhit_at_10"]
            ),
            "delta": float(round_promoted["retrieval"]["overall"]["allhit_at_10"])
            - float(round_baseline["overall"]["allhit_at_10"]),
            "promoted_route": str(round_promoted_path),
            "promoted_route_sha256": sha256_file(round_promoted_path),
        },
        "static_9000": {
            "baseline_full_trace": float(
                baseline_system["full_pipeline_trace_success"]
            ),
            "promoted_full_trace": float(
                promoted_system["full_pipeline_trace_success"]
            ),
            "delta": float(promoted_system["full_pipeline_trace_success"])
            - float(baseline_system["full_pipeline_trace_success"]),
            "paired_promoted_vs_baseline": paired_boolean(
                promoted_system_rows,
                baseline_system_rows,
                "full_pipeline_trace_success",
            ),
            "promoted_metrics": str(promoted_system_path),
            "promoted_metrics_sha256": sha256_file(promoted_system_path),
            "baseline_metrics": str(baseline_system_path),
            "baseline_metrics_sha256": sha256_file(baseline_system_path),
        },
        "interpretation": (
            "These protected results are a lock-after-promotion report. They cannot "
            "reopen the second-stage variant or depth selection."
        ),
    }
    write_json(promoted_root / "summary.json", payload)
    print(
        "protected promoted legacy_trace="
        f"{splits['legacy6000']['planner']['promoted']['trace_em']:.9f} "
        "api_trace="
        f"{splits['api1000']['planner']['promoted']['trace_em']:.9f} "
        "full_trace="
        f"{payload['static_9000']['promoted_full_trace']:.9f}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "prepare",
            "select",
            "apply-selected-route",
            "apply-confirmation",
            "summarize-confirmation",
            "summarize-protected",
        ),
        required=True,
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--source-preds", type=Path)
    parser.add_argument("--reranker20-preds", type=Path)
    parser.add_argument("--reranker30-preds", type=Path)
    parser.add_argument("--metrics-dir", type=Path)
    parser.add_argument("--rankings-out", type=Path)
    parser.add_argument(
        "--confirmation-dir",
        type=Path,
        help="Confirmation artifact root; defaults to OUT_DIR/confirmation.",
    )
    parser.add_argument(
        "--baseline-protected-dir",
        type=Path,
        help="Prior protected-report root used only by summarize-protected.",
    )
    args = parser.parse_args()
    if args.stage == "prepare":
        missing = [
            name
            for name, value in (
                ("--dataset", args.dataset),
                ("--source-preds", args.source_preds),
                ("--reranker20-preds", args.reranker20_preds),
            )
            if value is None
        ]
        if missing:
            raise SystemExit("prepare requires " + ", ".join(missing))
        return prepare(args)
    if args.stage == "select":
        if args.metrics_dir is None:
            raise SystemExit("select requires --metrics-dir")
        return select(args)
    if args.stage == "apply-confirmation":
        missing = [
            name
            for name, value in (
                ("--dataset", args.dataset),
                ("--source-preds", args.source_preds),
                ("--reranker20-preds", args.reranker20_preds),
                ("--reranker30-preds", args.reranker30_preds),
            )
            if value is None
        ]
        if missing:
            raise SystemExit("apply-confirmation requires " + ", ".join(missing))
        return apply_confirmation(args)
    if args.stage == "apply-selected-route":
        missing = [
            name
            for name, value in (
                ("--dataset", args.dataset),
                ("--source-preds", args.source_preds),
                ("--reranker20-preds", args.reranker20_preds),
                ("--rankings-out", args.rankings_out),
            )
            if value is None
        ]
        if missing:
            raise SystemExit("apply-selected-route requires " + ", ".join(missing))
        return apply_selected_route(args)
    if args.stage == "summarize-protected":
        return summarize_protected(args)
    if args.metrics_dir is None:
        raise SystemExit("summarize-confirmation requires --metrics-dir")
    return summarize_confirmation(args)


if __name__ == "__main__":
    raise SystemExit(main())
