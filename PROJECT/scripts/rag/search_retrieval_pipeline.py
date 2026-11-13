#!/usr/bin/env python3
"""Search the two-stage retrieval pipeline after the Embedding checkpoint is frozen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_aligned(left: list[dict], right: list[dict], left_name: str, right_name: str) -> None:
    if len(left) != len(right):
        raise ValueError(f"row count mismatch: {left_name}={len(left)} {right_name}={len(right)}")
    for index, (a, b) in enumerate(zip(left, right)):
        if str(a.get("id", "")) != str(b.get("id", "")):
            raise ValueError(f"id mismatch at {index}: {a.get('id')} != {b.get('id')}")
        if a.get("query") != b.get("query"):
            raise ValueError(f"query mismatch at {index}")


def allhit_at(rows: list[dict], rankings: list[list[str]], k: int) -> float:
    return sum(
        int(bool(row.get("gold_tools")) and set(row["gold_tools"]) <= set(ranking[:k]))
        for row, ranking in zip(rows, rankings)
    ) / max(1, len(rows))


def mrr(rows: list[dict], rankings: list[list[str]]) -> float:
    total = 0.0
    for row, ranking in zip(rows, rankings):
        gold = set(row.get("gold_tools", []))
        total += next((1.0 / rank for rank, tool in enumerate(ranking, 1) if tool in gold), 0.0)
    return total / max(1, len(rows))


def metrics(rows: list[dict], rankings: list[list[str]], ks: list[int]) -> dict[str, float]:
    result = {f"allhit_at_{k}": allhit_at(rows, rankings, k) for k in ks}
    result["mrr"] = mrr(rows, rankings)
    return result


def weighted_rrf(
    vector: dict,
    bm25: dict,
    candidate_n: int,
    vector_weight: float,
    rrf_k: int,
) -> list[str]:
    vector_pos = {tool: rank for rank, tool in enumerate(vector["pred_tools"][:candidate_n], 1)}
    bm25_pos = {tool: rank for rank, tool in enumerate(bm25["pred_tools"][:candidate_n], 1)}
    missing = 9999
    candidates = set(vector_pos) | set(bm25_pos)
    return sorted(
        candidates,
        key=lambda tool: (
            -(
                vector_weight / (rrf_k + vector_pos.get(tool, missing))
                + (1.0 - vector_weight) / (rrf_k + bm25_pos.get(tool, missing))
            ),
            tool,
        ),
    )


def second_rrf(source: list[str], reranked: list[str], source_weight: float, rrf_k: int) -> list[str]:
    source_pos = {tool: rank for rank, tool in enumerate(source, 1)}
    reranker_pos = {tool: rank for rank, tool in enumerate(reranked, 1)}
    missing = 9999
    return sorted(
        set(source_pos) | set(reranker_pos),
        key=lambda tool: (
            -(
                source_weight / (rrf_k + source_pos.get(tool, missing))
                + (1.0 - source_weight) / (rrf_k + reranker_pos.get(tool, missing))
            ),
            source_pos.get(tool, missing),
            reranker_pos.get(tool, missing),
            tool,
        ),
    )


def pareto_front(rows: list[dict], metric_names: list[str]) -> list[dict]:
    result = []
    for candidate in rows:
        dominated = False
        for other in rows:
            if other is candidate:
                continue
            no_worse = all(other[name] >= candidate[name] for name in metric_names)
            strictly_better = any(other[name] > candidate[name] for name in metric_names)
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            result.append(candidate)
    return result


def rank_sum(row: dict, rows: list[dict], metric_names: list[str]) -> int:
    total = 0
    for name in metric_names:
        ordered = sorted({float(value[name]) for value in rows}, reverse=True)
        total += ordered.index(float(row[name]))
    return total


def first_stage_search(
    vector_rows: list[dict],
    bm25_rows: list[dict],
    candidate_ns: list[int],
    vector_weights: list[float],
    rrf_ks: list[int],
    candidate_ks: list[int],
    top_per_depth: int,
    frontier_cap: int,
) -> tuple[list[dict], list[dict], list[dict], dict, dict[tuple, list[list[str]]]]:
    all_ks = sorted(set(candidate_ks + [6, 10, 20, 30, 40, 50]))
    grid = []
    rankings_by_key = {}
    for candidate_n in candidate_ns:
        for vector_weight in vector_weights:
            for rrf_k in rrf_ks:
                rankings = [
                    weighted_rrf(vector, bm25, candidate_n, vector_weight, rrf_k)
                    for vector, bm25 in zip(vector_rows, bm25_rows)
                ]
                key = (candidate_n, vector_weight, rrf_k)
                rankings_by_key[key] = rankings
                digest = hashlib.sha256()
                for ranking in rankings:
                    digest.update("\x1f".join(ranking).encode())
                    digest.update(b"\n")
                grid.append(
                    {
                        "candidate_n": candidate_n,
                        "vector_weight": vector_weight,
                        "rrf_k": rrf_k,
                        "mean_output_size": statistics.fmean(map(len, rankings)),
                        "ranking_sha256": digest.hexdigest(),
                        **metrics(vector_rows, rankings, all_ks),
                    }
                )
    unique_by_ranking = {}
    for row in grid:
        fingerprint = row["ranking_sha256"]
        previous = unique_by_ranking.get(fingerprint)
        preference = (
            row["candidate_n"],
            -row["vector_weight"],
            row["rrf_k"],
        )
        if previous is None or preference < (
            previous["candidate_n"],
            -previous["vector_weight"],
            previous["rrf_k"],
        ):
            unique_by_ranking[fingerprint] = row
    unique_grid = list(unique_by_ranking.values())
    metric_names = [f"allhit_at_{k}" for k in candidate_ks] + ["mrr"]
    frontier = pareto_front(unique_grid, metric_names)
    required = set()
    for k in candidate_ks:
        top = sorted(
            unique_grid,
            key=lambda row: (
                row[f"allhit_at_{k}"],
                row["allhit_at_10"],
                row["mrr"],
                -row["candidate_n"],
                row["vector_weight"],
                -row["rrf_k"],
            ),
            reverse=True,
        )[:top_per_depth]
        required.update((row["candidate_n"], row["vector_weight"], row["rrf_k"]) for row in top)
    candidates = {
        (row["candidate_n"], row["vector_weight"], row["rrf_k"]): row
        for row in frontier
    }
    candidates.update(
        {
            (row["candidate_n"], row["vector_weight"], row["rrf_k"]): row
            for row in unique_grid
            if (row["candidate_n"], row["vector_weight"], row["rrf_k"]) in required
        }
    )
    retained = list(candidates.values())
    if len(retained) > frontier_cap:
        retained.sort(
            key=lambda row: (
                rank_sum(row, unique_grid, metric_names),
                -row["allhit_at_50"],
                -row["allhit_at_20"],
                -row["allhit_at_10"],
                row["candidate_n"],
                -row["vector_weight"],
                row["rrf_k"],
            )
        )
        retained = retained[:frontier_cap]
    retained.sort(key=lambda row: (row["candidate_n"], row["vector_weight"], row["rrf_k"]))
    mining = max(
        unique_grid,
        key=lambda row: (
            row["allhit_at_50"],
            row["allhit_at_20"],
            row["allhit_at_10"],
            row["mrr"],
            -row["candidate_n"],
            row["vector_weight"],
            -row["rrf_k"],
        ),
    )
    return grid, unique_grid, retained, mining, rankings_by_key


def parse_reranker_spec(value: str) -> dict[str, Any]:
    parts = value.split("|", 2)
    if len(parts) != 3:
        raise ValueError("--reranker-spec must be GROUP|SEED|PREDS_JSONL")
    group, seed, path = parts
    return {"group": group, "seed": int(seed), "path": Path(path)}


def parameter_count(model_dir: Path) -> int:
    manifest_path = model_dir / "training_manifest.json"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        value = manifest.get("parameter_counts", {}).get("total")
        if value is not None:
            return int(value)
    weights = model_dir / "model.safetensors"
    if not weights.exists():
        raise FileNotFoundError(weights)
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("safetensors is required to count reranker parameters") from exc
    with safe_open(weights, framework="pt", device="cpu") as handle:
        return sum(
            math.prod(handle.get_slice(key).get_shape()) for key in handle.keys()
        )


def score_maps(
    source_rows: list[dict], path: Path
) -> tuple[list[dict[str, float]], float, int]:
    rows = load_jsonl(path)
    assert_aligned(source_rows, rows, "source", str(path))
    result = []
    for row in rows:
        tools = row.get("candidate_tools", [])
        scores = row.get("scores", [])
        if len(tools) != len(scores) or len(tools) < 100:
            raise ValueError(f"reranker file must score all 100 tools: {path}")
        result.append({str(tool): float(score) for tool, score in zip(tools, scores)})
    metric_path = path.parent / "metrics.json"
    latency = 0.0
    params = 0
    if metric_path.exists():
        metric_payload = load_json(metric_path)
        latency = float(metric_payload.get("timing", {}).get("rerank_p50_ms_per_query", 0.0))
        params = parameter_count(Path(str(metric_payload["model"])))
    return result, latency, params


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate_metric(rows: list[dict], name: str) -> dict[str, float]:
    values = [float(row[name]) for row in rows]
    return {
        "mean": statistics.fmean(values),
        "population_std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def rank_arrays(
    rows: list[dict],
    source_rankings: list[list[str]],
    reranked: list[list[str]],
    tool_index: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = (len(rows), len(tool_index))
    source_positions = np.full(shape, 9999, dtype=np.int32)
    reranker_positions = np.full(shape, 9999, dtype=np.int32)
    gold_mask = np.zeros(shape, dtype=bool)
    for row_index, (row, source, reranker) in enumerate(zip(rows, source_rankings, reranked)):
        for rank, tool in enumerate(source, 1):
            source_positions[row_index, tool_index[tool]] = rank
        for rank, tool in enumerate(reranker, 1):
            reranker_positions[row_index, tool_index[tool]] = rank
        for tool in row.get("gold_tools", []):
            gold_mask[row_index, tool_index[tool]] = True
    return source_positions, reranker_positions, gold_mask


def prepare_gold_rank_context(
    source_positions: np.ndarray,
    reranker_positions: np.ndarray,
    gold_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    gold_counts = gold_mask.sum(axis=1)
    if np.any(gold_counts == 0):
        raise ValueError("every query must contain at least one gold tool")
    max_gold = int(gold_counts.max())
    gold_indices = np.argsort(~gold_mask, axis=1, kind="stable")[:, :max_gold]
    gold_valid = np.take_along_axis(gold_mask, gold_indices, axis=1)
    present = source_positions < 9999
    gold_present = np.take_along_axis(present, gold_indices, axis=1)
    gold_source = np.take_along_axis(source_positions, gold_indices, axis=1)
    gold_reranker = np.take_along_axis(reranker_positions, gold_indices, axis=1)

    candidate_source = source_positions[:, None, :]
    candidate_reranker = reranker_positions[:, None, :]
    source_tie = candidate_source == gold_source[:, :, None]
    tie_break_better = candidate_source < gold_source[:, :, None]
    tie_break_better |= source_tie & (
        candidate_reranker < gold_reranker[:, :, None]
    )
    reranker_tie = source_tie & (
        candidate_reranker == gold_reranker[:, :, None]
    )
    candidate_tools = np.arange(source_positions.shape[1], dtype=np.int32)[
        None, None, :
    ]
    tie_break_better |= reranker_tie & (
        candidate_tools < gold_indices[:, :, None]
    )
    return {
        "gold_indices": gold_indices,
        "gold_valid": gold_valid,
        "gold_present": gold_present,
        "tie_break_better": tie_break_better,
    }


def array_metrics(
    source_positions: np.ndarray,
    reranker_positions: np.ndarray,
    gold_mask: np.ndarray,
    source_weight: float,
    rrf_k: int,
    ks: list[int],
    gold_rank_context: dict[str, np.ndarray] | None = None,
) -> dict[str, float]:
    present = source_positions < 9999
    scores = (
        source_weight / (rrf_k + source_positions.astype(np.float64))
        + (1.0 - source_weight) / (rrf_k + reranker_positions.astype(np.float64))
    )
    scores[~present] = -np.inf
    if gold_rank_context is None:
        gold_rank_context = prepare_gold_rank_context(
            source_positions, reranker_positions, gold_mask
        )
    gold_indices = gold_rank_context["gold_indices"]
    gold_valid = gold_rank_context["gold_valid"]
    gold_present = gold_rank_context["gold_present"]
    gold_scores = np.take_along_axis(scores, gold_indices, axis=1)

    candidate_scores = scores[:, None, :]
    better = candidate_scores > gold_scores[:, :, None]
    score_tie = candidate_scores == gold_scores[:, :, None]
    better |= score_tie & gold_rank_context["tie_break_better"]
    better &= present[:, None, :]

    gold_ranks = better.sum(axis=2, dtype=np.int32)
    gold_ranks[~(gold_valid & gold_present)] = 9999
    worst_gold = np.where(gold_valid, gold_ranks, -1).max(axis=1)
    best_gold = np.where(gold_valid, gold_ranks, 9999).min(axis=1)
    result = {f"allhit_at_{k}": float(np.mean(worst_gold < k)) for k in ks}
    result["mrr"] = float(np.mean(np.where(best_gold < 9999, 1.0 / (best_gold + 1), 0.0)))
    return result


def retain_near_best(
    rows: list[dict], accessor: Callable[[dict], float], tolerance: float, higher: bool = True
) -> list[dict]:
    target = (max if higher else min)(accessor(row) for row in rows)
    if higher:
        return [row for row in rows if accessor(row) >= target - tolerance]
    return [row for row in rows if accessor(row) <= target + tolerance]


def select_aggregate_route(
    rows: list[dict], tolerance: float, compare_latency: bool = False
) -> tuple[dict, list[dict]]:
    active = rows
    trace = []
    criteria = [
        ("mean_allhit_at_10", lambda row: row["aggregate"]["allhit_at_10"]["mean"], True),
        ("mean_allhit_at_6", lambda row: row["aggregate"]["allhit_at_6"]["mean"], True),
        ("mean_mrr", lambda row: row["aggregate"]["mrr"]["mean"], True),
        ("candidate_k", lambda row: float(row["route"]["candidate_k"]), False),
    ]
    if compare_latency:
        criteria.extend(
            [
                ("estimated_rerank_p50", lambda row: row["estimated_rerank_p50_ms"], False),
                ("parameter_count", lambda row: float(row["parameter_count"]), False),
            ]
        )
    for name, accessor, higher in criteria:
        if len(active) <= 1:
            break
        active = retain_near_best(active, accessor, tolerance if higher else 0.0, higher)
        trace.append({"criterion": name, "remaining": len(active)})
    if len(active) > 1:
        active.sort(
            key=lambda row: (
                row["route"]["first_candidate_n"],
                -row["route"]["first_vector_weight"],
                row["route"]["first_rrf_k"],
                -row["route"]["second_source_weight"],
                row["route"]["second_rrf_k"],
            )
        )
    return active[0], trace


def route_key(route: dict) -> tuple:
    return (
        route["first_candidate_n"],
        route["first_vector_weight"],
        route["first_rrf_k"],
        route["candidate_k"],
        route["second_source_weight"],
        route["second_rrf_k"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vector-preds", type=Path, required=True)
    parser.add_argument("--bm25-preds", type=Path, required=True)
    parser.add_argument("--reranker-spec", action="append", default=[])
    parser.add_argument("--candidate-ns", nargs="+", type=int, default=[10, 20, 30, 40, 50, 75, 100])
    parser.add_argument("--first-vector-weights", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.99, 1.0])
    parser.add_argument("--first-rrf-ks", nargs="+", type=int, default=[1, 2, 3, 5, 8, 10, 20, 30, 60, 100])
    parser.add_argument("--candidate-ks", nargs="+", type=int, default=[10, 20, 30, 40, 50])
    parser.add_argument("--top-per-depth", type=int, default=5)
    parser.add_argument("--frontier-cap", type=int, default=24)
    parser.add_argument("--second-source-weights", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
    parser.add_argument("--second-rrf-ks", nargs="+", type=int, default=[1, 2, 3, 5, 8, 10, 20, 30, 60, 100])
    parser.add_argument("--selection-tolerance", type=float, default=0.001)
    parser.add_argument(
        "--compare-group-latency",
        action="store_true",
        help=(
            "Use measured reranker latency when choosing between groups. Leave disabled for "
            "learning-rate, seed, or negative-recipe groups of the same architecture."
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    vector_rows = load_jsonl(args.vector_preds)
    bm25_rows = load_jsonl(args.bm25_preds)
    assert_aligned(vector_rows, bm25_rows, "vector", "bm25")
    required_depth = max(args.candidate_ns)
    if any(len(row.get("pred_tools", [])) < required_depth for row in vector_rows + bm25_rows):
        raise ValueError(f"first-stage inputs must contain at least {required_depth} tools per query")

    grid, unique_grid, frontier, mining, rankings_by_key = first_stage_search(
        vector_rows,
        bm25_rows,
        args.candidate_ns,
        args.first_vector_weights,
        args.first_rrf_ks,
        args.candidate_ks,
        args.top_per_depth,
        args.frontier_cap,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    mining_key = (mining["candidate_n"], mining["vector_weight"], mining["rrf_k"])
    mining_rows = [
        {
            "id": row.get("id", ""),
            "query": row["query"],
            "gold_tools": row.get("gold_tools", []),
            "pred_tools": ranking,
        }
        for row, ranking in zip(vector_rows, rankings_by_key[mining_key])
    ]
    write_jsonl(args.out_dir / "mining_source_preds.jsonl", mining_rows)
    first_payload = {
        "vector_preds": str(args.vector_preds),
        "vector_preds_sha256": sha256_file(args.vector_preds),
        "bm25_preds": str(args.bm25_preds),
        "bm25_preds_sha256": sha256_file(args.bm25_preds),
        "grid_size": len(grid),
        "unique_ranking_count": len(unique_grid),
        "frontier_size_before_cap": len(pareto_front(unique_grid, [f"allhit_at_{k}" for k in args.candidate_ks] + ["mrr"])),
        "retained_size": len(frontier),
        "retained": frontier,
        "mining_source_selected": mining,
        "grid": grid,
    }
    write_json(args.out_dir / "first_stage_search.json", first_payload)
    print(f"first-stage grid={len(grid)} retained={len(frontier)}")

    specs = [parse_reranker_spec(value) for value in args.reranker_spec]
    if not specs:
        return 0
    seed_results = []
    latency_by_spec = {}
    tool_ids = sorted({tool for row in vector_rows for tool in row.get("pred_tools", [])})
    if len(tool_ids) != 100:
        raise ValueError(f"expected exactly 100 tools, found {len(tool_ids)}")
    tool_index = {tool: index for index, tool in enumerate(tool_ids)}
    for spec in specs:
        maps, full100_latency, params = score_maps(vector_rows, spec["path"])
        latency_by_spec[(spec["group"], spec["seed"])] = full100_latency
        for first in frontier:
            first_key = (first["candidate_n"], first["vector_weight"], first["rrf_k"])
            source_rankings = rankings_by_key[first_key]
            for candidate_k in args.candidate_ks:
                reranked = [
                    sorted(
                        source[:candidate_k],
                        key=lambda tool: (-score_map[tool], source.index(tool), tool),
                    )
                    for source, score_map in zip(source_rankings, maps)
                ]
                source_positions, reranker_positions, gold_mask = rank_arrays(
                    vector_rows,
                    source_rankings,
                    reranked,
                    tool_index,
                )
                gold_rank_context = prepare_gold_rank_context(
                    source_positions, reranker_positions, gold_mask
                )
                for source_weight in args.second_source_weights:
                    for second_k in args.second_rrf_ks:
                        route = {
                            "first_candidate_n": first["candidate_n"],
                            "first_vector_weight": first["vector_weight"],
                            "first_rrf_k": first["rrf_k"],
                            "candidate_k": candidate_k,
                            "second_source_weight": source_weight,
                            "second_rrf_k": second_k,
                        }
                        seed_results.append(
                            {
                                "group": spec["group"],
                                "seed": spec["seed"],
                                "route": route,
                                "full100_rerank_p50_ms": full100_latency,
                                "estimated_rerank_p50_ms": full100_latency * candidate_k / 100.0,
                                "parameter_count": params,
                                **array_metrics(
                                    source_positions,
                                    reranker_positions,
                                    gold_mask,
                                    source_weight,
                                    second_k,
                                    [6, 10, 20, 30, 40, 50],
                                    gold_rank_context,
                                ),
                            }
                        )

    grouped: dict[tuple[str, tuple], list[dict]] = defaultdict(list)
    for row in seed_results:
        grouped[(row["group"], route_key(row["route"]))].append(row)
    aggregate_rows = []
    for (group, _), rows in grouped.items():
        aggregate_rows.append(
            {
                "group": group,
                "route": rows[0]["route"],
                "seeds": [row["seed"] for row in rows],
                "estimated_rerank_p50_ms": statistics.fmean(row["estimated_rerank_p50_ms"] for row in rows),
                "parameter_count": int(statistics.fmean(row["parameter_count"] for row in rows)),
                "aggregate": {
                    name: aggregate_metric(rows, name)
                    for name in ("allhit_at_6", "allhit_at_10", "allhit_at_20", "allhit_at_30", "allhit_at_40", "allhit_at_50", "mrr")
                },
                "seed_metrics": rows,
            }
        )
    selected_by_group = []
    traces = {}
    for group in sorted({row["group"] for row in aggregate_rows}):
        candidates = [row for row in aggregate_rows if row["group"] == group]
        selected, trace = select_aggregate_route(
            candidates, args.selection_tolerance, compare_latency=False
        )
        selected_by_group.append(selected)
        traces[group] = trace
    selected_group, group_trace = select_aggregate_route(
        selected_by_group,
        args.selection_tolerance,
        compare_latency=args.compare_group_latency,
    )
    selected_seed = max(
        selected_group["seed_metrics"],
        key=lambda row: (row["allhit_at_10"], row["allhit_at_6"], row["mrr"], -row["seed"]),
    )
    selected_first_key = (
        selected_group["route"]["first_candidate_n"],
        selected_group["route"]["first_vector_weight"],
        selected_group["route"]["first_rrf_k"],
    )
    selected_source = rankings_by_key[selected_first_key]
    selected_spec = next(
        spec
        for spec in specs
        if spec["group"] == selected_group["group"] and spec["seed"] == selected_seed["seed"]
    )
    selected_maps, _, _ = score_maps(vector_rows, selected_spec["path"])
    selected_positions = [
        {tool: index for index, tool in enumerate(source)} for source in selected_source
    ]
    selected_reranker = [
        sorted(
            source[: selected_group["route"]["candidate_k"]],
            key=lambda tool: (-score_map[tool], position[tool], tool),
        )
        for source, score_map, position in zip(selected_source, selected_maps, selected_positions)
    ]
    final_rankings = [
        second_rrf(
            source,
            rerank,
            selected_group["route"]["second_source_weight"],
            selected_group["route"]["second_rrf_k"],
        )
        for source, rerank in zip(selected_source, selected_reranker)
    ]
    write_jsonl(
        args.out_dir / "selected_dev_preds.jsonl",
        [
            {
                "id": row.get("id", ""),
                "query": row["query"],
                "gold_tools": row.get("gold_tools", []),
                "pred_tools": ranking,
            }
            for row, ranking in zip(vector_rows, final_rankings)
        ],
    )
    payload = {
        "reranker_specs": [
            {
                "group": spec["group"],
                "seed": spec["seed"],
                "path": str(spec["path"]),
                "sha256": sha256_file(spec["path"]),
            }
            for spec in specs
        ],
        "vector_preds_sha256": sha256_file(args.vector_preds),
        "bm25_preds_sha256": sha256_file(args.bm25_preds),
        "selection_tolerance": args.selection_tolerance,
        "compare_group_latency": args.compare_group_latency,
        "selected_by_group": selected_by_group,
        "within_group_selection_traces": traces,
        "group_selection_trace": group_trace,
        "selected": {
            "group": selected_group["group"],
            "route": selected_group["route"],
            "aggregate": selected_group["aggregate"],
            "deployment_seed": selected_seed["seed"],
            "deployment_seed_metrics": selected_seed,
        },
        "aggregate_route_count": len(aggregate_rows),
    }
    write_json(args.out_dir / "pipeline_search.json", payload)
    print(
        f"selected group={selected_group['group']} seed={selected_seed['seed']} "
        f"AllHit@10={selected_seed['allhit_at_10']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
