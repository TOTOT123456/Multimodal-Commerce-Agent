#!/usr/bin/env python3
"""在隔离的多轮 JSONL 上运行 canonical Agent，并断点续写预测与指标。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from PROJECT.agent.factory import (  # noqa: E402
    DEFAULT_CONFIG,
    RuntimeDevices,
    build_agent,
    config_hash,
    load_agent_config,
)
from PROJECT.agent.orchestrator import AgentPolicy  # noqa: E402
from PROJECT.agent.types import (  # noqa: E402
    IntentResult,
    PlanEvent,
    PlannerResult,
    RetrievalResult,
    normalize_goal_text,
)
from PROJECT.scripts.project_common import (  # noqa: E402
    catalog_by_id,
    load_catalog,
    load_jsonl,
    write_json,
    write_jsonl,
)


EXPECTED_FIELDS = {
    "id",
    "conversation_id",
    "turn_id",
    "query",
    "gold_tools",
    "gold_resolved_goals",
    "gold_cancelled_goals",
    "gold_reference_turn_ids",
    "context_dependency",
    "scenario",
    "memory_distance",
    "source_trace_id",
    "expression_cluster",
    "context_skeleton",
    "eval_split",
}
CONTEXT_DEPENDENCIES = {"none", "recent", "long"}
SCENARIOS = {
    "self_contained",
    "reference",
    "inheritance",
    "cancellation",
    "correction",
    "reorder",
    "long_recall",
    "task_switch",
    "irrelevant_history",
    "summary_pressure",
    "failure_fallback",
    "session_isolation",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_backend_cache(path: Path | None) -> dict[str, dict[str, Any]] | None:
    if path is None:
        return None
    result: dict[str, dict[str, Any]] = {"intent": {}, "retrieval": {}, "planner": {}}
    for row in load_jsonl(path):
        kind = str(row["kind"])
        key = str(row["key"])
        payload = row["payload"]
        if kind == "intent":
            value = IntentResult(
                intent_type=str(payload["intent_type"]),
                keywords=str(payload.get("keywords", "")),
                raw_output=str(payload.get("raw_output", "")),
                latency_ms=float(payload.get("latency_ms", 0.0)),
            )
        elif kind == "retrieval":
            value = RetrievalResult(
                candidate_tools=[str(item) for item in payload["candidate_tools"]],
                source_tools=[str(item) for item in payload.get("source_tools", [])],
                reranked_tools=[str(item) for item in payload.get("reranked_tools", [])],
                revision=str(payload.get("revision", "")),
                route_hash=str(payload.get("route_hash", "")),
                latency_ms=float(payload.get("latency_ms", 0.0)),
                diagnostics=dict(payload.get("diagnostics", {})),
            )
        elif kind == "planner":
            value = PlannerResult(
                tools=[str(item) for item in payload["tools"]],
                strict_ok=bool(payload["strict_ok"]),
                strict_error=str(payload["strict_error"]),
                raw_output=str(payload["raw_output"]),
                input_tokens=int(payload["input_tokens"]),
                output_tokens=int(payload["output_tokens"]),
                latency_ms=float(payload["latency_ms"]),
                prompt_version=str(payload["prompt_version"]),
                prompt_hash=str(payload["prompt_hash"]),
                model_revision=str(payload["model_revision"]),
            )
        else:
            raise ValueError(f"unknown backend cache kind: {kind}")
        if key in result[kind]:
            raise ValueError(f"duplicate backend cache key: {kind}/{key}")
        result[kind][key] = value
    return result


def normalized_query(value: str) -> str:
    return re.sub(r"[\W_]+", "", value.lower(), flags=re.UNICODE)


def validate_dataset(
    rows: list[dict[str, Any]],
    split: str,
    catalog_ids: set[str],
    protected_paths: list[Path],
    require_tool_combination_isolation: bool = True,
    allow_duplicate_unscored_queries: bool = False,
    allow_duplicate_scored_queries: bool = False,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("multi-turn evaluation dataset is empty")
    seen_ids: set[str] = set()
    seen_conversations: set[str] = set()
    closed_conversations: set[str] = set()
    expected_turn: dict[str, int] = defaultdict(lambda: 1)
    current_conversation = ""
    queries: set[str] = set()
    duplicate_unscored_queries = 0
    duplicate_scored_queries = 0
    source_ids: set[str] = set()
    clusters: set[str] = set()
    skeletons: set[str] = set()
    ordered_tool_combinations: set[tuple[str, ...]] = set()
    unordered_tool_combinations: set[tuple[str, ...]] = set()
    for index, row in enumerate(rows):
        if set(row) != EXPECTED_FIELDS:
            raise ValueError(
                f"row {index} field whitelist failed: "
                f"missing={sorted(EXPECTED_FIELDS - set(row))} "
                f"extra={sorted(set(row) - EXPECTED_FIELDS)}"
            )
        row_id = str(row["id"])
        conversation = str(row["conversation_id"])
        turn_id = int(row["turn_id"])
        if not row_id or row_id in seen_ids:
            raise ValueError(f"row {index} has empty or duplicate id")
        if not conversation or turn_id != expected_turn[conversation]:
            raise ValueError(f"{row_id}: conversation turns must start at 1 and be contiguous")
        if conversation != current_conversation:
            if conversation in closed_conversations:
                raise ValueError(f"{row_id}: conversations must occupy contiguous file blocks")
            if current_conversation:
                closed_conversations.add(current_conversation)
            current_conversation = conversation
        if row["eval_split"] != split:
            raise ValueError(f"{row_id}: eval_split differs from --split {split}")
        if row["context_dependency"] not in CONTEXT_DEPENDENCIES:
            raise ValueError(f"{row_id}: invalid context_dependency")
        if row["scenario"] not in SCENARIOS:
            raise ValueError(f"{row_id}: invalid scenario")
        memory_distance = row["memory_distance"]
        if memory_distance is not None and (
            isinstance(memory_distance, bool)
            or not isinstance(memory_distance, int)
            or memory_distance < 0
        ):
            raise ValueError(f"{row_id}: memory_distance must be null or non-negative int")
        for field in ("gold_resolved_goals", "gold_cancelled_goals"):
            if not isinstance(row[field], list) or any(
                not isinstance(value, str) or not value.strip() for value in row[field]
            ):
                raise ValueError(f"{row_id}: invalid {field}")
        if not str(row["source_trace_id"]).strip() or not str(
            row["expression_cluster"]
        ).strip() or not str(row["context_skeleton"]).strip():
            raise ValueError(f"{row_id}: source and expression provenance must not be empty")
        tools = [str(value) for value in row["gold_tools"]]
        if len(tools) != len(set(tools)) or not set(tools) <= catalog_ids:
            raise ValueError(f"{row_id}: invalid gold tools")
        if len(tools) >= 2:
            ordered_tool_combinations.add(tuple(tools))
            unordered_tool_combinations.add(tuple(sorted(tools)))
        references = [int(value) for value in row["gold_reference_turn_ids"]]
        if any(value <= 0 or value >= turn_id for value in references):
            raise ValueError(f"{row_id}: invalid gold reference turn")
        query_key = normalized_query(str(row["query"]))
        if not query_key:
            raise ValueError(f"{row_id}: empty normalized query")
        if query_key in queries:
            if allow_duplicate_scored_queries and memory_distance is not None:
                duplicate_scored_queries += 1
            elif allow_duplicate_unscored_queries and memory_distance is None:
                duplicate_unscored_queries += 1
            else:
                raise ValueError(f"{row_id}: duplicate normalized query")
        seen_ids.add(row_id)
        seen_conversations.add(conversation)
        queries.add(query_key)
        source_ids.add(str(row["source_trace_id"]))
        clusters.add(str(row["expression_cluster"]))
        skeletons.add(str(row["context_skeleton"]))
        expected_turn[conversation] += 1

    overlap = {
        "query": set(),
        "source_trace_id": set(),
        "expression_cluster": set(),
        "context_skeleton": set(),
        "ordered_tool_combination": set(),
        "unordered_tool_combination": set(),
    }
    protected_rows = 0
    for path in protected_paths:
        for row in load_jsonl(path):
            protected_rows += 1
            query = str(row.get("问题", row.get("query", "")))
            query_key = normalized_query(query)
            if query_key in queries:
                overlap["query"].add(query_key)
            if str(row.get("id", "")) in source_ids:
                overlap["source_trace_id"].add(str(row.get("id", "")))
            protected_clusters = row.get("expression_cluster_ids", [])
            if isinstance(protected_clusters, dict):
                protected_clusters = protected_clusters.values()
            for cluster in protected_clusters:
                if str(cluster) in clusters:
                    overlap["expression_cluster"].add(str(cluster))
            protected_skeleton = str(
                row.get("context_skeleton", row.get("template_family", ""))
            )
            if protected_skeleton in skeletons:
                overlap["context_skeleton"].add(protected_skeleton)
            protected_tools = row.get("工具", row.get("gold_tools", []))
            if isinstance(protected_tools, str):
                protected_tools = [
                    value.strip() for value in protected_tools.split(",") if value.strip()
                ]
            protected_combination = tuple(str(value) for value in protected_tools)
            if len(protected_combination) >= 2:
                if protected_combination in ordered_tool_combinations:
                    overlap["ordered_tool_combination"].add(protected_combination)
                unordered = tuple(sorted(protected_combination))
                if unordered in unordered_tool_combinations:
                    overlap["unordered_tool_combination"].add(unordered)
    content_overlap = {
        key: overlap[key]
        for key in (
            "query",
            "source_trace_id",
            "expression_cluster",
            "context_skeleton",
        )
    }
    combination_overlap = {
        key: overlap[key]
        for key in ("ordered_tool_combination", "unordered_tool_combination")
    }
    strict_failures = list(content_overlap.values())
    if require_tool_combination_isolation:
        strict_failures.extend(combination_overlap.values())
    audit = {
        "rows": len(rows),
        "conversations": len(seen_conversations),
        "split": split,
        "protected_rows": protected_rows,
        "query_overlap": len(overlap["query"]),
        "source_trace_id_overlap": len(overlap["source_trace_id"]),
        "expression_cluster_overlap": len(overlap["expression_cluster"]),
        "context_skeleton_overlap": len(overlap["context_skeleton"]),
        "ordered_tool_combination_overlap": len(overlap["ordered_tool_combination"]),
        "unordered_tool_combination_overlap": len(overlap["unordered_tool_combination"]),
        "tool_combination_isolation_required": require_tool_combination_isolation,
        "duplicate_unscored_queries_allowed": allow_duplicate_unscored_queries,
        "duplicate_unscored_query_occurrences": duplicate_unscored_queries,
        "duplicate_scored_queries_allowed": allow_duplicate_scored_queries,
        "duplicate_scored_query_occurrences": duplicate_scored_queries,
        "unique_normalized_queries": len(queries),
        "strict_ok": not any(strict_failures),
    }
    if not audit["strict_ok"]:
        raise ValueError(f"multi-turn isolation audit failed: {audit}")
    return audit


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def stats(values: list[int | float]) -> dict[str, float | int]:
    numeric = [float(value) for value in values]
    return {
        "count": len(numeric),
        "mean": sum(numeric) / len(numeric) if numeric else 0.0,
        "p50": percentile(numeric, 0.50),
        "p95": percentile(numeric, 0.95),
        "p99": percentile(numeric, 0.99),
        "max": max(numeric, default=0.0),
    }


def prediction_row(row: dict[str, Any], event: PlanEvent) -> dict[str, Any]:
    gold_tools = [str(value) for value in row["gold_tools"]]
    pred_tools = list(event.planned_tools)
    gold_set = set(gold_tools)
    pred_set = set(pred_tools)
    predicted_goals = [normalize_goal_text(goal.text) for goal in event.resolved_goals]
    predicted_cancelled = [normalize_goal_text(goal.text) for goal in event.cancelled_goals]
    gold_goals = [normalize_goal_text(str(value)) for value in row["gold_resolved_goals"]]
    gold_cancelled = [normalize_goal_text(str(value)) for value in row["gold_cancelled_goals"]]
    return {
        "id": row["id"],
        "conversation_id": row["conversation_id"],
        "turn_id": row["turn_id"],
        "query": row["query"],
        "gold_tools": gold_tools,
        "pred_tools": pred_tools,
        "trace_exact": pred_tools == gold_tools,
        "set_exact": pred_set == gold_set,
        "strict_output": event.strict_output,
        "status": event.status,
        "context_dependency": row["context_dependency"],
        "pred_context_mode": event.context_mode,
        "context_mode_exact": event.context_mode == row["context_dependency"],
        "scenario": row["scenario"],
        "memory_distance": row["memory_distance"],
        "gold_reference_turn_ids": row["gold_reference_turn_ids"],
        "pred_reference_turn_ids": event.reference_turn_ids,
        "reference_exact": event.reference_turn_ids == row["gold_reference_turn_ids"],
        "goal_trace_exact": predicted_goals == gold_goals,
        "goal_set_exact": set(predicted_goals) == set(gold_goals),
        "cancellation_set_exact": set(predicted_cancelled) == set(gold_cancelled),
        "extra_tools": sorted(pred_set - gold_set),
        "missing_tools": sorted(gold_set - pred_set),
        "input_tokens": event.input_tokens,
        "output_tokens": event.output_tokens,
        "latency_ms": event.latency_ms,
        "resolved_query": event.resolved_query,
        "warnings": event.warnings,
        "event": event.to_dict(),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    total_predicted = sum(len(row["pred_tools"]) for row in rows)
    total_gold = sum(len(row["gold_tools"]) for row in rows)
    latency = [
        float(row["latency_ms"].get("total_before_commit", sum(row["latency_ms"].values())))
        for row in rows
    ]
    return {
        "count": count,
        "trace_em": sum(row["trace_exact"] for row in rows) / max(1, count),
        "set_em": sum(row["set_exact"] for row in rows) / max(1, count),
        "strict_output_rate": sum(row["strict_output"] for row in rows) / max(1, count),
        "context_mode_accuracy": sum(row["context_mode_exact"] for row in rows) / max(1, count),
        "reference_exact": sum(row["reference_exact"] for row in rows) / max(1, count),
        "goal_trace_exact": sum(row["goal_trace_exact"] for row in rows) / max(1, count),
        "goal_set_exact": sum(row["goal_set_exact"] for row in rows) / max(1, count),
        "cancellation_set_exact": sum(row["cancellation_set_exact"] for row in rows)
        / max(1, count),
        "extra_tool_rate": sum(len(row["extra_tools"]) for row in rows)
        / max(1, total_predicted),
        "omission_rate": sum(len(row["missing_tools"]) for row in rows) / max(1, total_gold),
        "status_counts": dict(Counter(str(row["status"]) for row in rows)),
        "input_tokens": stats([row["input_tokens"] for row in rows]),
        "output_tokens": stats([row["output_tokens"] for row in rows]),
        "latency_ms_before_commit": stats(latency),
    }


def grouped_metrics(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {key: aggregate(value) for key, value in sorted(groups.items())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--protected",
        type=Path,
        action="append",
        default=[
            REPO_ROOT / "PROJECT/data/tool_call_trace/train.jsonl",
            REPO_ROOT / "PROJECT/data/tool_call_trace/test.jsonl",
        ],
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--allow-tool-combination-overlap",
        action="store_true",
        help=(
            "允许标签层面的工具组合与保护集重合；query、trace id、表达簇和上下文骨架仍必须隔离"
        ),
    )
    parser.add_argument(
        "--allow-duplicate-unscored-queries",
        action="store_true",
        help=(
            "允许不同会话复用自然干扰请求；计分的历史引用 query 仍必须全局唯一。"
        ),
    )
    parser.add_argument(
        "--allow-duplicate-scored-queries",
        action="store_true",
        help=(
            "允许自然短确认在隔离会话中重复；完整会话与来源计划仍需独立，"
            "且该开关不会放宽保护集重合审计。"
        ),
    )
    parser.add_argument("--intent-device", default="cuda:0")
    parser.add_argument("--embedding-device", default="cuda:0")
    parser.add_argument("--reranker-device", default="cuda:1")
    parser.add_argument("--planner-device", default="cuda:2")
    parser.add_argument("--intent-base-model", default="")
    parser.add_argument("--planner-base-model", default="")
    parser.add_argument("--backend-cache", type=Path, default=None)
    parser.add_argument(
        "--require-backend-cache-hit",
        action="store_true",
        help="Fail the turn instead of loading a live backend when a cache entry is missing.",
    )
    parser.add_argument(
        "--fixed-ecommerce-intent",
        action="store_true",
        help="Only for audited all-ecommerce protocols; retrieval and Planner still use real cached outputs.",
    )
    parser.add_argument("--architecture", choices=list("ABCDE"), default="E")
    parser.add_argument("--recent-event-limit", type=int, choices=[2, 3, 4], default=2)
    parser.add_argument("--long-memory-top-k", type=int, choices=[1, 2, 3], default=2)
    parser.add_argument(
        "--target-input-tokens",
        type=int,
        choices=[640, 768, 1024, 2048, 4096],
        default=640,
    )
    parser.add_argument(
        "--hard-input-tokens",
        type=int,
        default=None,
        help="Optional evaluation-only override; must be at least the target budget.",
    )
    parser.add_argument(
        "--direct-history-max-extra-tokens",
        type=int,
        default=None,
        help=(
            "Architecture B only: permit the first raw-history variant to exceed "
            "the normal resolved-query delta by this many tokens."
        ),
    )
    parser.add_argument(
        "--catalog-alias-guard",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--single-tool-goal-normalization",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--recent-over-generic-long",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--natural-request-boundary-guard",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--whole-turn-reference-replay",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--standalone-multigoal-guard",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--whole-turn-plan-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--semantic-reference-strategy",
        choices=["off", "raw_bm25", "anchor_bm25", "anchor_bm25_position"],
        default=None,
    )
    parser.add_argument("--allow-reproduced-adapter", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    catalog = load_catalog(REPO_ROOT / "PROJECT/configs/tool_catalog.jsonl")
    audit = validate_dataset(
        rows,
        args.split,
        set(catalog_by_id(catalog)),
        args.protected,
        require_tool_combination_isolation=not args.allow_tool_combination_overlap,
        allow_duplicate_unscored_queries=args.allow_duplicate_unscored_queries,
        allow_duplicate_scored_queries=args.allow_duplicate_scored_queries,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    metrics_path = args.output_dir / "metrics.json"
    manifest_path = args.output_dir / "manifest.json"
    db_path = args.output_dir / "runtime.sqlite3"
    agent_config = load_agent_config(args.config)
    memory_config = agent_config["memory"]
    token_config = agent_config["token_budget"]
    planning_config = agent_config["planning"]
    policy = AgentPolicy(
        context_architecture=args.architecture,
        recent_event_limit=args.recent_event_limit,
        long_memory_top_k=args.long_memory_top_k,
        long_memory_timeout_ms=int(memory_config["long_memory_timeout_ms"]),
        target_input_tokens=args.target_input_tokens,
        hard_input_tokens=(
            int(token_config["hard_input_tokens"])
            if args.hard_input_tokens is None
            else args.hard_input_tokens
        ),
        max_resolved_extra_tokens=int(token_config["max_resolved_extra_tokens"]),
        direct_history_max_extra_tokens=args.direct_history_max_extra_tokens,
        max_query_characters=int(token_config["max_query_characters"]),
        catalog_alias_guard=(
            bool(planning_config["catalog_alias_guard"])
            if args.catalog_alias_guard is None
            else args.catalog_alias_guard
        ),
        single_tool_goal_normalization=(
            bool(planning_config["single_tool_goal_normalization"])
            if args.single_tool_goal_normalization is None
            else args.single_tool_goal_normalization
        ),
        recent_over_generic_long=(
            bool(planning_config["recent_over_generic_long"])
            if args.recent_over_generic_long is None
            else args.recent_over_generic_long
        ),
        natural_request_boundary_guard=(
            bool(planning_config["natural_request_boundary_guard"])
            if args.natural_request_boundary_guard is None
            else args.natural_request_boundary_guard
        ),
        whole_turn_reference_replay=(
            bool(planning_config.get("whole_turn_reference_replay", False))
            if args.whole_turn_reference_replay is None
            else args.whole_turn_reference_replay
        ),
        standalone_multigoal_guard=(
            bool(planning_config.get("standalone_multigoal_guard", False))
            if args.standalone_multigoal_guard is None
            else args.standalone_multigoal_guard
        ),
        whole_turn_plan_cache=(
            bool(planning_config.get("whole_turn_plan_cache", False))
            if args.whole_turn_plan_cache is None
            else args.whole_turn_plan_cache
        ),
        semantic_reference_strategy=(
            str(planning_config.get("semantic_reference_strategy", "off"))
            if args.semantic_reference_strategy is None
            else args.semantic_reference_strategy
        ),
    )
    run_config = {
        "run_id": args.run_id,
        "input": str(args.input.resolve()),
        "input_sha256": sha256_file(args.input),
        "split": args.split,
        "agent_config": str(args.config.resolve()),
        "agent_config_hash": config_hash(agent_config),
        "devices": {
            "intent": args.intent_device,
            "embedding": args.embedding_device,
            "reranker": args.reranker_device,
            "planner": args.planner_device,
        },
        "policy": {
            "architecture": args.architecture,
            "recent_event_limit": args.recent_event_limit,
            "long_memory_top_k": args.long_memory_top_k,
            "target_input_tokens": args.target_input_tokens,
            "hard_input_tokens": policy.hard_input_tokens,
            "direct_history_max_extra_tokens": policy.direct_history_max_extra_tokens,
            "catalog_alias_guard": policy.catalog_alias_guard,
            "single_tool_goal_normalization": policy.single_tool_goal_normalization,
            "recent_over_generic_long": policy.recent_over_generic_long,
            "natural_request_boundary_guard": policy.natural_request_boundary_guard,
            "whole_turn_reference_replay": policy.whole_turn_reference_replay,
            "standalone_multigoal_guard": policy.standalone_multigoal_guard,
            "whole_turn_plan_cache": policy.whole_turn_plan_cache,
            "semantic_reference_strategy": policy.semantic_reference_strategy,
        },
        "allow_reproduced_adapter": args.allow_reproduced_adapter,
        "allow_tool_combination_overlap": args.allow_tool_combination_overlap,
        "backend_cache": (
            None
            if args.backend_cache is None
            else {
                "path": str(args.backend_cache.resolve()),
                "sha256": sha256_file(args.backend_cache),
            }
        ),
        "require_backend_cache_hit": args.require_backend_cache_hit,
        "fixed_ecommerce_intent": args.fixed_ecommerce_intent,
    }
    run_hash = hashlib.sha256(
        json.dumps(run_config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest.get("run_hash") != run_hash:
            raise SystemExit("output directory belongs to another evaluation run")
    else:
        write_json(
            manifest_path,
            {
                "status": "running",
                "run_config": run_config,
                "run_hash": run_hash,
                "dataset_audit": audit,
            },
        )

    existing = load_jsonl(predictions_path) if predictions_path.exists() else []
    if [row["id"] for row in existing] != [row["id"] for row in rows[: len(existing)]]:
        raise SystemExit("existing predictions are not an exact input prefix")
    output = list(existing)
    devices = RuntimeDevices(
        intent=args.intent_device,
        embedding=args.embedding_device,
        reranker=args.reranker_device,
        planner=args.planner_device,
    )
    agent = build_agent(
        config_path=args.config,
        db_path=db_path,
        devices=devices,
        intent_base_model=args.intent_base_model or None,
        planner_base_model=args.planner_base_model or None,
        policy_override=policy,
        verify_adapter_hash=False if args.allow_reproduced_adapter else None,
        backend_cache=load_backend_cache(args.backend_cache),
        require_backend_cache_hit=args.require_backend_cache_hit,
        fixed_ecommerce_intent=args.fixed_ecommerce_intent,
    )
    try:
        for saved in output:
            saved_event = PlanEvent.from_dict(saved["event"])
            if agent.store.event(saved_event.conversation_id, saved_event.turn_id) is not None:
                continue
            snapshot = agent.store.snapshot(
                saved_event.conversation_id,
                recent_limit=agent.policy.recent_event_limit + 1,
            )
            if saved_event.turn_id != snapshot.state.last_turn_id + 1:
                raise RuntimeError("cannot reconstruct runtime DB from saved prediction prefix")
            restored_state = agent.state_reducer.reduce(snapshot.state, saved_event)
            restored_summary, restored_memories = agent.memory_reducer.reduce(
                snapshot.summary,
                snapshot.recent_events + [saved_event],
                newest_turn_id=saved_event.turn_id,
            )
            agent.store.commit_turn(
                event=saved_event,
                state=restored_state,
                summary=restored_summary,
                new_memories=restored_memories,
                expected_version=snapshot.state.version,
            )
        for row in rows[len(output) :]:
            session_id = f"{args.run_id}:{row['conversation_id']}"
            committed = agent.store.event(session_id, int(row["turn_id"]))
            if committed is not None:
                if committed.raw_query != row["query"]:
                    raise RuntimeError("runtime DB is ahead with a different query")
                event = committed
            else:
                result = agent.run_turn(session_id, str(row["query"]))
                event = result.event
            prediction = prediction_row(row, event)
            output.append(prediction)
            with predictions_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(prediction, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
            print(f"evaluated {len(output)}/{len(rows)}", flush=True)
    finally:
        agent.close()

    metrics = {
        "run_hash": run_hash,
        "dataset_audit": audit,
        "overall": aggregate(output),
        "by_context_dependency": grouped_metrics(output, "context_dependency"),
        "by_scenario": grouped_metrics(output, "scenario"),
        "by_memory_distance": grouped_metrics(output, "memory_distance"),
    }
    write_json(metrics_path, metrics)
    write_json(
        manifest_path,
        {
            "status": "complete",
            "run_config": run_config,
            "run_hash": run_hash,
            "dataset_audit": audit,
            "prediction_rows": len(output),
            "predictions_sha256": sha256_file(predictions_path),
            "metrics_sha256": sha256_file(metrics_path),
        },
    )
    print(
        f"multi-turn {args.split} TraceEM={metrics['overall']['trace_em']:.6f} "
        f"rows={len(output)} -> {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
