#!/usr/bin/env python3
"""Build depth-aligned retrieval routes for joint Reranker/Planner selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from search_retrieval_pipeline import (  # noqa: E402
    assert_aligned,
    load_json,
    load_jsonl,
    metrics,
    second_rrf,
    weighted_rrf,
    write_json,
    write_jsonl,
)


RERANKER_DEPTHS = (10, 20, 30, 50)
PLANNER_DEPTHS = (5, 10, 20, 30, 50)
SOURCE_WEIGHTS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
RRF_KS = (1, 2, 3, 5, 8, 10, 20, 30, 60, 100)
PANEL_WEIGHTS = {"legacy_like": 6.0 / 7.0, "api_like": 1.0 / 7.0}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_depth_path(value: str) -> tuple[int, Path]:
    raw_depth, separator, raw_path = value.partition("=")
    if not separator or not raw_depth.isdigit() or not raw_path:
        raise argparse.ArgumentTypeError("expected DEPTH=PATH")
    depth = int(raw_depth)
    if depth not in RERANKER_DEPTHS:
        raise argparse.ArgumentTypeError(
            f"reranker depth must be one of {RERANKER_DEPTHS}"
        )
    return depth, Path(raw_path)


def subgroup_metrics(
    rows: list[dict[str, Any]],
    rankings: list[list[str]],
    field: str,
    metric_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    metric_rows = rows if metric_rows is None else metric_rows
    if len(rows) != len(rankings) or len(metric_rows) != len(rankings):
        raise ValueError("subgroup rows and rankings differ")
    indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        value = row.get(field)
        if value is None and isinstance(row.get("construction"), dict):
            value = row["construction"].get(field)
        indices[str(value)].append(index)
    return {
        value: {
            "count": len(selected),
            **metrics(
                [metric_rows[index] for index in selected],
                [rankings[index] for index in selected],
        [5, 6, 10, 20, 30, 50],
            ),
        }
        for value, selected in sorted(indices.items())
    }


def deployment_weighted_metrics(by_panel: dict[str, Any]) -> dict[str, float] | None:
    if not set(PANEL_WEIGHTS) <= set(by_panel):
        return None
    names = [
        "allhit_at_5",
        "allhit_at_6",
        "allhit_at_10",
        "allhit_at_20",
        "allhit_at_30",
        "allhit_at_50",
        "mrr",
    ]
    return {
        name: sum(PANEL_WEIGHTS[panel] * float(by_panel[panel][name]) for panel in PANEL_WEIGHTS)
        for name in names
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--vector-preds", type=Path, required=True)
    parser.add_argument("--bm25-preds", type=Path, required=True)
    parser.add_argument(
        "--reranker-preds",
        type=Path,
        help=(
            "Backward-compatible shared score file. Prefer exact-depth files via "
            "--reranker-preds-at-depth for deployment-faithful evaluation."
        ),
    )
    parser.add_argument(
        "--reranker-preds-at-depth",
        action="append",
        type=parse_depth_path,
        default=[],
        metavar="DEPTH=PATH",
        help="Reranker output produced with candidate-k exactly DEPTH; repeat by depth.",
    )
    parser.add_argument("--route-json", type=Path, required=True)
    parser.add_argument(
        "--frozen-route-specs",
        type=Path,
        help="Apply depth-pair RRF parameters selected on another split.",
    )
    parser.add_argument(
        "--route-id",
        action="append",
        help="Build only the named depth pair; repeat as needed.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    all_route_ids = {
        f"rerank{reranker_depth}_planner{planner_depth}"
        for reranker_depth in RERANKER_DEPTHS
        for planner_depth in PLANNER_DEPTHS
        if planner_depth <= reranker_depth
    }
    active_route_ids = set(args.route_id or all_route_ids)
    if not active_route_ids or not active_route_ids <= all_route_ids:
        raise ValueError(f"invalid route ids: {sorted(active_route_ids - all_route_ids)}")
    required_reranker_depths = sorted(
        {
            int(route_id.split("_", 1)[0].removeprefix("rerank"))
            for route_id in active_route_ids
        }
    )
    required_reranker_depth = max(required_reranker_depths)
    explicit_reranker_paths: dict[int, Path] = {}
    for depth, path in args.reranker_preds_at_depth:
        if depth in explicit_reranker_paths:
            raise ValueError(f"duplicate reranker predictions for depth {depth}")
        explicit_reranker_paths[depth] = path
    reranker_paths = dict(explicit_reranker_paths)
    if args.reranker_preds is not None:
        for depth in required_reranker_depths:
            reranker_paths.setdefault(depth, args.reranker_preds)
    missing_depths = [
        depth for depth in required_reranker_depths if depth not in reranker_paths
    ]
    if missing_depths:
        raise ValueError(
            "missing reranker predictions for depths "
            f"{missing_depths}; pass --reranker-preds-at-depth DEPTH=PATH"
        )

    dataset = load_jsonl(args.dataset)
    vector = load_jsonl(args.vector_preds)
    bm25 = load_jsonl(args.bm25_preds)
    assert_aligned(vector, bm25, "vector", "bm25")
    if len(dataset) != len(vector):
        raise ValueError("dataset and retrieval rows differ")
    metadata = {str(row["id"]): row for row in dataset}
    if set(metadata) != {str(row["id"]) for row in vector}:
        raise ValueError("dataset and retrieval ids differ")

    route_payload = load_json(args.route_json)
    first = route_payload["first_stage"]
    frozen_source_rankings = [
        weighted_rrf(
            vector_row,
            bm25_row,
            int(first["candidate_n_per_source"]),
            float(first["vector_weight"]),
            int(first["rrf_k"]),
        )
        for vector_row, bm25_row in zip(vector, bm25)
    ]
    full_source_rankings = [
        weighted_rrf(
            vector_row,
            bm25_row,
            min(len(vector_row["pred_tools"]), len(bm25_row["pred_tools"])),
            float(first["vector_weight"]),
            int(first["rrf_k"]),
        )
        for vector_row, bm25_row in zip(vector, bm25)
    ]
    source_rankings = []
    completed_rows = 0
    completed_tools = 0
    for frozen_source, full_source in zip(frozen_source_rankings, full_source_rankings):
        completed = list(frozen_source)
        if len(completed) < required_reranker_depth:
            seen = set(completed)
            appended = [tool for tool in full_source if tool not in seen]
            needed = required_reranker_depth - len(completed)
            completed.extend(appended[:needed])
            completed_rows += 1
            completed_tools += min(needed, len(appended))
        if len(completed) < required_reranker_depth:
            raise ValueError(
                f"cannot complete first-stage source to {required_reranker_depth} tools"
            )
        source_rankings.append(completed)
    score_maps_by_depth: dict[int, list[dict[str, float]]] = {}
    reranker_scoring: dict[str, Any] = {}
    for depth in required_reranker_depths:
        path = reranker_paths[depth]
        reranker = load_jsonl(path)
        assert_aligned(vector, reranker, "vector", f"reranker@{depth}")
        exact_depth_file = depth in explicit_reranker_paths
        score_maps = []
        for index, (row, source) in enumerate(zip(reranker, source_rankings)):
            tools = [str(value) for value in row.get("candidate_tools", [])]
            scores = row.get("scores", [])
            if len(tools) != len(scores) or len(tools) < depth:
                raise ValueError(
                    f"reranker@{depth} row {index} lacks {depth} aligned scores"
                )
            if exact_depth_file and len(tools) != depth:
                raise ValueError(
                    f"reranker@{depth} row {index} scored {len(tools)} tools; "
                    "exact-depth evaluation requires exactly DEPTH tools"
                )
            if tools[:depth] != source[:depth]:
                raise ValueError(
                    f"reranker@{depth} row {index} candidates do not match "
                    "the frozen first-stage prefix"
                )
            score_maps.append(dict(zip(tools[:depth], map(float, scores[:depth]))))
        score_maps_by_depth[depth] = score_maps
        reranker_scoring[str(depth)] = {
            "predictions": str(path),
            "sha256": sha256_file(path),
            "exact_candidate_depth": exact_depth_file,
        }

    rows_for_metrics = [
        {
            "id": row["id"],
            "query": row["query"],
            "gold_tools": row.get("gold_tools", []),
        }
        for row in vector
    ]
    frozen_specs = None
    if args.frozen_route_specs:
        frozen_payload = load_json(args.frozen_route_specs)
        frozen_specs = {str(row["id"]): row for row in frozen_payload["routes"]}
    selected_routes: list[dict[str, Any]] = []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for reranker_depth in RERANKER_DEPTHS:
        if not any(
            route_id.startswith(f"rerank{reranker_depth}_")
            for route_id in active_route_ids
        ):
            continue
        reranked = []
        for source, score_map in zip(
            source_rankings, score_maps_by_depth[reranker_depth]
        ):
            source_prefix = source[:reranker_depth]
            missing = [tool for tool in source_prefix if tool not in score_map]
            if missing:
                raise ValueError(f"missing reranker scores: {missing[:3]}")
            source_pos = {tool: index for index, tool in enumerate(source_prefix)}
            reranked.append(
                sorted(
                    source_prefix,
                    key=lambda tool: (-score_map[tool], source_pos[tool], tool),
                )
            )
        for planner_depth in PLANNER_DEPTHS:
            if planner_depth > reranker_depth:
                continue
            candidates = []
            stem = f"rerank{reranker_depth}_planner{planner_depth}"
            if stem not in active_route_ids:
                continue
            if frozen_specs is None:
                parameter_pairs = [
                    (source_weight, rrf_k)
                    for source_weight in SOURCE_WEIGHTS
                    for rrf_k in RRF_KS
                ]
            else:
                frozen = frozen_specs.get(stem)
                if frozen is None:
                    raise ValueError(f"frozen route specs lack {stem}")
                parameter_pairs = [
                    (
                        float(frozen["second_stage"]["source_weight"]),
                        int(frozen["second_stage"]["rrf_k"]),
                    )
                ]
            for source_weight, rrf_k in parameter_pairs:
                rankings = [
                    second_rrf(
                        source[:reranker_depth],
                        rerank,
                        source_weight,
                        rrf_k,
                    )
                    for source, rerank in zip(source_rankings, reranked)
                ]
                overall = metrics(
                    rows_for_metrics, rankings, [5, 6, 10, 20, 30, 50]
                )
                by_panel = subgroup_metrics(
                    dataset, rankings, "panel", rows_for_metrics
                )
                weighted = deployment_weighted_metrics(by_panel)
                candidates.append(
                    {
                        "source_weight": source_weight,
                        "rrf_k": rrf_k,
                        "rankings": rankings,
                        "overall": overall,
                        "by_panel": by_panel,
                        "deployment_weighted": weighted,
                    }
                )
            selected = max(
                candidates,
                key=lambda item: (
                    (item["deployment_weighted"] or item["overall"])[
                        f"allhit_at_{planner_depth}"
                    ],
                    min(
                        (
                            panel[f"allhit_at_{planner_depth}"]
                            for panel in item["by_panel"].values()
                        ),
                        default=item["overall"][f"allhit_at_{planner_depth}"],
                    ),
                    (item["deployment_weighted"] or item["overall"])["allhit_at_6"],
                    (item["deployment_weighted"] or item["overall"])["mrr"],
                    -item["source_weight"],
                    -item["rrf_k"],
                ),
            )
            rankings = selected.pop("rankings")
            top_prefix_outside_reranker = sum(
                not set(ranking[:planner_depth]) <= set(rerank)
                for ranking, rerank in zip(rankings, reranked)
            )
            if top_prefix_outside_reranker:
                raise ValueError("Planner prefix contains an unreranked tool")
            ranking_path = args.out_dir / "rankings" / f"{stem}.jsonl"
            write_jsonl(
                ranking_path,
                [
                    {
                        "id": row["id"],
                        "query": row["query"],
                        "gold_tools": row.get("gold_tools", []),
                        "pred_tools": ranking,
                        "source_tools": source,
                        "reranked_tools": rerank,
                    }
                    for row, source, rerank, ranking in zip(
                        rows_for_metrics, source_rankings, reranked, rankings
                    )
                ],
            )
            selected_routes.append(
                {
                    "id": stem,
                    "reranker_depth": reranker_depth,
                    "planner_depth": planner_depth,
                    "first_stage": first,
                    "second_stage": {
                        "source_weight": selected["source_weight"],
                        "rrf_k": selected["rrf_k"],
                        "candidate_universe": "first_stage_prefix_at_reranker_depth",
                    },
                    "overall": selected["overall"],
                    "deployment_weighted": selected["deployment_weighted"],
                    "by_panel": selected["by_panel"],
                    "by_tool_count": subgroup_metrics(
                        dataset, rankings, "tool_count", rows_for_metrics
                    ),
                    "planner_prefix_outside_reranker_count": top_prefix_outside_reranker,
                    "rankings": str(ranking_path),
                    "rankings_sha256": sha256_file(ranking_path),
                }
            )

    payload = {
        "schema_version": 1,
        "status": "complete",
        "scope": "fixed first stage; joint reranker depth, residual RRF, and planner depth",
        "invariant": "planner_depth <= reranker_depth and every displayed tool has a cross-encoder score",
        "reranker_scoring": reranker_scoring,
        "first_stage_completion": {
            "target_depth": required_reranker_depth,
            "policy": "preserve frozen N=40 source prefix, then append full-source RRF tail before reranking",
            "rows_extended": completed_rows,
            "tools_appended": completed_tools,
        },
        "second_stage_selection": (
            "apply frozen development route parameters"
            if frozen_specs is not None
            else "for each depth pair maximize 6:1 Legacy/API weighted "
            "AllHit@PlannerK, then minimum-panel AllHit, weighted AllHit@6, "
            "weighted MRR, lower source weight, and lower RRF k"
        ),
        "panel_weights": PANEL_WEIGHTS,
        "inputs": {
            str(path): sha256_file(path)
            for path in dict.fromkeys(
                (
                    args.dataset,
                    args.vector_preds,
                    args.bm25_preds,
                    *[reranker_paths[depth] for depth in required_reranker_depths],
                    args.route_json,
                    *([args.frozen_route_specs] if args.frozen_route_specs else []),
                )
            )
        },
        "grid": {
            "reranker_depths": list(RERANKER_DEPTHS),
            "planner_depths": list(PLANNER_DEPTHS),
            "active_route_ids": sorted(active_route_ids),
            "source_weights": list(SOURCE_WEIGHTS),
            "rrf_ks": list(RRF_KS),
        },
        "routes": selected_routes,
    }
    write_json(args.out_dir / "routes.json", payload)
    print(f"joint depth routes={len(selected_routes)} -> {args.out_dir / 'routes.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
