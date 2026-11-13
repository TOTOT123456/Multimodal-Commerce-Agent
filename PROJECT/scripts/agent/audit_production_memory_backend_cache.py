#!/usr/bin/env python3
"""离线遍历短会话候选，导出真实评测尚缺的唯一后端输入。"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from PROJECT.agent.backends import CachedPlannerBackend, query_cache_key  # noqa: E402
from PROJECT.agent.factory import config_hash, load_agent_config  # noqa: E402
from PROJECT.agent.memory import DeterministicMemoryReducer, SQLiteMemoryStore  # noqa: E402
from PROJECT.agent.orchestrator import AgentPolicy, ToolPlanningAgent  # noqa: E402
from PROJECT.agent.types import IntentResult, PlannerResult, RetrievalResult  # noqa: E402
from PROJECT.scripts.agent.evaluate_multiturn_agent import load_backend_cache  # noqa: E402
from PROJECT.scripts.agent.run_production_memory_evaluation import variants  # noqa: E402
from PROJECT.scripts.finetune.tool_trace_common import sha256_json  # noqa: E402
from PROJECT.scripts.project_common import (  # noqa: E402
    catalog_by_id,
    load_catalog,
    load_jsonl,
    write_json,
    write_jsonl,
)


DEFAULT_ROOT = (
    REPO_ROOT / "PROJECT/data/agent/_aux/natural_short_memory_evaluation"
)


class AlwaysEcommerce:
    def classify(self, query: str) -> IntentResult:
        return IntentResult(intent_type="电商类")


class RecordingRetriever:
    def __init__(
        self,
        cache: dict[str, RetrievalResult],
        missing: dict[str, str],
    ):
        self.cache = cache
        self.missing = missing

    def retrieve(self, query: str) -> RetrievalResult:
        key = query_cache_key(query)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        self.missing[key] = query
        return RetrievalResult(candidate_tools=[])


class RecordingPlanner:
    def __init__(
        self,
        cache: dict[str, PlannerResult],
        missing: dict[str, tuple[str, list[str]]],
    ):
        self.cache = cache
        self.missing = missing

    def count_input_tokens(self, query: str, candidate_tools: list[str]) -> int:
        cached = self.cache.get(CachedPlannerBackend.key(query, candidate_tools))
        return int(cached.input_tokens) if cached is not None else 500

    def cached_input_tokens(
        self,
        query: str,
        candidate_tools: list[str],
        *,
        allow_missing: bool = False,
    ) -> int:
        return self.count_input_tokens(query, candidate_tools)

    def plan(self, query: str, candidate_tools: list[str]) -> PlannerResult:
        key = CachedPlannerBackend.key(query, candidate_tools)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        self.missing[key] = (query, list(candidate_tools))
        return PlannerResult(
            tools=[],
            strict_ok=True,
            strict_error="",
            raw_output='{"tools":[]}',
            input_tokens=500,
            output_tokens=8,
            latency_ms=0.0,
            prompt_version="tool-trace-v1",
            prompt_hash="missing-cache-audit",
            model_revision="missing-cache-audit",
        )


def policy_for(variant: object, runtime: dict) -> AgentPolicy:
    architecture = str(getattr(variant, "architecture"))
    direct = architecture == "B"
    return AgentPolicy(
        context_architecture=architecture,
        recent_event_limit=2,
        long_memory_top_k=2,
        long_memory_timeout_ms=int(runtime["memory"]["long_memory_timeout_ms"]),
        target_input_tokens=4096 if direct else 640,
        hard_input_tokens=4096 if direct else 1024,
        max_resolved_extra_tokens=int(
            runtime["token_budget"]["max_resolved_extra_tokens"]
        ),
        direct_history_max_extra_tokens=4096 if direct else None,
        max_query_characters=int(runtime["token_budget"]["max_query_characters"]),
        catalog_alias_guard=bool(getattr(variant, "catalog_alias_guard")),
        single_tool_goal_normalization=bool(
            getattr(variant, "single_tool_goal_normalization")
        ),
        recent_over_generic_long=bool(getattr(variant, "recent_over_generic_long")),
        natural_request_boundary_guard=bool(
            getattr(variant, "natural_request_boundary_guard")
        ),
        whole_turn_reference_replay=bool(
            getattr(variant, "whole_turn_reference_replay")
        ),
        standalone_multigoal_guard=bool(
            getattr(variant, "standalone_multigoal_guard")
        ),
        whole_turn_plan_cache=bool(getattr(variant, "whole_turn_plan_cache")),
        semantic_reference_strategy=str(
            getattr(variant, "semantic_reference_strategy")
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--labels", default="")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    cache = (
        load_backend_cache(args.cache)
        if args.cache is not None and args.cache.is_file()
        else {"intent": {}, "retrieval": {}, "planner": {}}
    )
    assert cache is not None
    selected = variants()
    requested = {value for value in args.labels.split(",") if value}
    if requested:
        unknown = requested - {variant.label for variant in selected}
        if unknown:
            raise SystemExit(f"unknown labels: {sorted(unknown)}")
        selected = [variant for variant in selected if variant.label in requested]

    runtime = load_agent_config(REPO_ROOT / "PROJECT/configs/agent_runtime.json")
    by_id = catalog_by_id(
        load_catalog(REPO_ROOT / "PROJECT/configs/tool_catalog.jsonl")
    )
    rows = load_jsonl(args.root / f"{args.split}.jsonl")
    missing_retrieval: dict[str, str] = {}
    missing_planner: dict[str, tuple[str, list[str]]] = {}
    per_variant = []
    for variant in selected:
        policy = policy_for(variant, runtime)
        agent = ToolPlanningAgent(
            store=SQLiteMemoryStore(":memory:", ttl_seconds=2592000),
            intent_backend=AlwaysEcommerce(),
            retriever=RecordingRetriever(cache["retrieval"], missing_retrieval),
            planner=RecordingPlanner(cache["planner"], missing_planner),
            catalog_by_id=by_id,
            policy=policy,
            agent_config_hash=config_hash(
                {"runtime": runtime, "policy_override": asdict(policy)}
            ),
            route_revision=str(runtime["required_canonical_revision"]),
            route_hash=str(runtime["canonical_route_sha256"]),
            memory_reducer=DeterministicMemoryReducer(
                by_id,
                recent_event_limit=2,
                max_summary_goals=int(runtime["memory"]["max_summary_goals"]),
                memory_ttl_seconds=int(runtime["memory"]["session_ttl_seconds"]),
            ),
        )
        before_retrieval = len(missing_retrieval)
        before_planner = len(missing_planner)
        try:
            for row in rows:
                agent.run_turn(
                    f"cache-audit:{args.split}:{variant.label}:{row['conversation_id']}",
                    str(row["query"]),
                )
        finally:
            agent.close()
        per_variant.append(
            {
                "label": variant.label,
                "new_missing_retrieval": len(missing_retrieval) - before_retrieval,
                "new_missing_planner": len(missing_planner) - before_planner,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    retrieval_rows = [
        {"id": key, "query": query}
        for key, query in sorted(missing_retrieval.items())
    ]
    planner_rows = []
    for key, (query, candidates) in sorted(missing_planner.items()):
        payload = {
            "id": key,
            "query": query,
            "candidate_tools": candidates,
            "candidate_k": len(candidates),
            "ranking_hash": sha256_json(candidates),
        }
        payload["input_hash"] = sha256_json(payload)
        planner_rows.append(payload)
    write_jsonl(args.output_dir / "missing_retrieval.jsonl", retrieval_rows)
    write_jsonl(args.output_dir / "missing_planner_inputs.jsonl", planner_rows)
    write_json(
        args.output_dir / "summary.json",
        {
            "status": "complete",
            "split": args.split,
            "labels": [variant.label for variant in selected],
            "cache": None if args.cache is None else str(args.cache.resolve()),
            "missing_retrieval": len(retrieval_rows),
            "missing_planner": len(planner_rows),
            "per_variant": per_variant,
        },
    )
    print(json.dumps({"retrieval": len(retrieval_rows), "planner": len(planner_rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
