#!/usr/bin/env python3
"""对多轮运行时的超时、预算、非法输出与会话隔离做可机读故障注入。"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from PROJECT.agent.factory import config_hash, load_agent_config  # noqa: E402
from PROJECT.agent.memory import DeterministicMemoryReducer, SQLiteMemoryStore  # noqa: E402
from PROJECT.agent.orchestrator import AgentPolicy, ToolPlanningAgent  # noqa: E402
from PROJECT.agent.types import (  # noqa: E402
    IntentResult,
    PlannerResult,
    RetrievalResult,
)
from PROJECT.scripts.project_common import (  # noqa: E402
    catalog_by_id,
    load_catalog,
    write_json,
)

PROJECT_ROOT = REPO_ROOT / "PROJECT"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data/agent/_aux/natural_short_memory_evaluation/validation/failure_guards.json"
)


class FakeIntent:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def classify(self, query: str) -> IntentResult:
        self.queries.append(query)
        return IntentResult(intent_type="电商类")


class FakeRetriever:
    def __init__(self, candidates: list[str]) -> None:
        self.candidates = candidates
        self.queries: list[str] = []

    def retrieve(self, query: str) -> RetrievalResult:
        self.queries.append(query)
        return RetrievalResult(candidate_tools=list(self.candidates), revision="fault-test")


class FakePlanner:
    def __init__(self, input_tokens: int = 100, strict_ok: bool = True) -> None:
        self.input_tokens = input_tokens
        self.strict_ok = strict_ok
        self.plan_queries: list[str] = []

    def count_input_tokens(self, query: str, candidate_tools: list[str]) -> int:
        return self.input_tokens

    def plan(self, query: str, candidate_tools: list[str]) -> PlannerResult:
        self.plan_queries.append(query)
        tools = []
        if "物流" in query:
            tools = ["get_order_logistics"]
        elif "库存" in query:
            tools = ["get_sku_stock"]
        elif "优惠" in query:
            tools = ["calc_discount"]
        elif "订单详情" in query:
            tools = ["get_order_detail"]
        return PlannerResult(
            tools=tools if self.strict_ok else ["get_order_logistics"],
            strict_ok=self.strict_ok,
            strict_error="" if self.strict_ok else "invalid_json",
            raw_output=(json.dumps({"tools": tools}) if self.strict_ok else "not-json"),
            input_tokens=self.input_tokens,
            output_tokens=5,
            latency_ms=0.0,
            prompt_version="fault-test",
            prompt_hash="fault-test",
            model_revision="fault-test",
        )


def build(
    by_id: dict[str, dict[str, Any]],
    policy: AgentPolicy,
    planner: FakePlanner,
) -> tuple[ToolPlanningAgent, FakeIntent, FakeRetriever]:
    intent = FakeIntent()
    candidates = [
        "get_order_logistics",
        "get_sku_stock",
        "calc_discount",
        "get_order_detail",
    ] + [tool for tool in by_id if tool not in {
        "get_order_logistics", "get_sku_stock", "calc_discount", "get_order_detail"
    }][:6]
    retriever = FakeRetriever(candidates)
    agent = ToolPlanningAgent(
        store=SQLiteMemoryStore(":memory:", ttl_seconds=3600),
        intent_backend=intent,
        retriever=retriever,
        planner=planner,
        catalog_by_id=by_id,
        policy=policy,
        agent_config_hash="fault-test",
        route_revision="fault-test",
        route_hash="fault-test",
        memory_reducer=DeterministicMemoryReducer(
            by_id,
            recent_event_limit=policy.recent_event_limit,
            max_summary_goals=20,
            memory_ttl_seconds=3600,
        ),
    )
    return agent, intent, retriever


def selected_policy(runtime: dict[str, Any], **overrides: Any) -> AgentPolicy:
    memory = runtime["memory"]
    token = runtime["token_budget"]
    planning = runtime["planning"]
    values = {
        "context_architecture": str(memory["context_architecture"]),
        "recent_event_limit": int(memory["recent_event_limit"]),
        "long_memory_top_k": int(memory["long_memory_top_k"]),
        "long_memory_timeout_ms": int(memory["long_memory_timeout_ms"]),
        "target_input_tokens": int(token["target_input_tokens"]),
        "hard_input_tokens": int(token["hard_input_tokens"]),
        "max_resolved_extra_tokens": int(token["max_resolved_extra_tokens"]),
        "max_query_characters": int(token["max_query_characters"]),
        "catalog_alias_guard": bool(planning["catalog_alias_guard"]),
        "single_tool_goal_normalization": bool(
            planning["single_tool_goal_normalization"]
        ),
        "recent_over_generic_long": bool(planning["recent_over_generic_long"]),
        "natural_request_boundary_guard": bool(
            planning["natural_request_boundary_guard"]
        ),
        "whole_turn_reference_replay": bool(
            planning.get("whole_turn_reference_replay", False)
        ),
        "standalone_multigoal_guard": bool(
            planning.get("standalone_multigoal_guard", False)
        ),
        "whole_turn_plan_cache": bool(planning.get("whole_turn_plan_cache", False)),
    }
    values.update(overrides)
    return AgentPolicy(**values)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    by_id = catalog_by_id(load_catalog(PROJECT_ROOT / "configs/tool_catalog.jsonl"))
    runtime = load_agent_config()
    canonical_policy = selected_policy(runtime)
    cases: dict[str, dict[str, Any]] = {}

    planner = FakePlanner()
    agent, intent, retriever = build(by_id, canonical_policy, planner)
    try:
        result = agent.run_turn("empty", "继续刚才的两个操作")
        cases["unresolved_reference_guard"] = {
            "status": result.status,
            "tools": result.tools,
            "strict_output": result.strict_output,
            "intent_calls": len(intent.queries),
            "retrieval_calls": len(retriever.queries),
            "planner_calls": len(planner.plan_queries),
            "passed": result.status == "unresolved_reference"
            and result.strict_output
            and not result.tools
            and not intent.queries
            and not retriever.queries
            and not planner.plan_queries,
        }
    finally:
        agent.close()

    planner = FakePlanner()
    agent, _, _ = build(by_id, canonical_policy, planner)
    try:
        agent.run_turn("business-mutation", "查订单详情")
        query = "查我之前提交的退款进度；如果购物车里有失效商品就删掉。"
        result = agent.run_turn("business-mutation", query)
        cases["business_mutation_not_meta_cancellation"] = {
            "context_mode": result.context_mode,
            "resolved_query": result.resolved_query,
            "carried_goals": [goal.text for goal in result.event.carried_goals],
            "cancelled_goals": [goal.text for goal in result.event.cancelled_goals],
            "warnings": result.warnings,
            "passed": result.context_mode == "none"
            and result.resolved_query == query
            and not result.event.carried_goals
            and not result.event.cancelled_goals
            and "standalone_multigoal_context_guard" in result.warnings,
        }
    finally:
        agent.close()

    planner = FakePlanner()
    agent, intent, retriever = build(by_id, canonical_policy, planner)
    try:
        first = agent.run_turn("plan-cache", "查订单物流")
        repeated = agent.run_turn("plan-cache", "继续刚才的全部任务")
        cases["validated_whole_turn_plan_cache"] = {
            "source_tools": first.tools,
            "repeated_tools": repeated.tools,
            "prompt_version": repeated.event.prompt_version,
            "input_tokens": repeated.event.input_tokens,
            "intent_calls": len(intent.queries),
            "retrieval_calls": len(retriever.queries),
            "planner_calls": len(planner.plan_queries),
            "passed": repeated.tools == first.tools
            and repeated.event.prompt_version == "whole-turn-plan-replay-v1"
            and repeated.event.input_tokens == 0
            and len(intent.queries) == 1
            and len(retriever.queries) == 1
            and len(planner.plan_queries) == 1,
        }
    finally:
        agent.close()

    planner = FakePlanner()
    agent, intent, retriever = build(by_id, canonical_policy, planner)
    try:
        first = agent.run_turn("cache-invalidation", "查订单物流")
        agent.agent_config_hash = "fault-test-changed"
        repeated = agent.run_turn("cache-invalidation", "继续刚才的全部任务")
        cases["whole_turn_plan_cache_hash_invalidation"] = {
            "source_tools": first.tools,
            "repeated_tools": repeated.tools,
            "prompt_version": repeated.event.prompt_version,
            "intent_calls": len(intent.queries),
            "retrieval_calls": len(retriever.queries),
            "planner_calls": len(planner.plan_queries),
            "warnings": repeated.warnings,
            "passed": repeated.tools == first.tools
            and repeated.event.prompt_version == "fault-test"
            and "whole_turn_plan_cache_applied" not in repeated.warnings
            and "whole_turn_reference_replayed" in repeated.warnings
            and len(intent.queries) == 2
            and len(retriever.queries) == 2
            and len(planner.plan_queries) == 2,
        }
    finally:
        agent.close()

    planner = FakePlanner(input_tokens=200)
    policy = selected_policy(runtime, target_input_tokens=120, hard_input_tokens=150)
    agent, _, _ = build(by_id, policy, planner)
    try:
        result = agent.run_turn("budget", "查库存")
        cases["hard_token_budget"] = {
            "status": result.status,
            "planner_calls": len(planner.plan_queries),
            "passed": result.status == "budget_rejected" and not planner.plan_queries,
        }
    finally:
        agent.close()

    planner = FakePlanner(strict_ok=False)
    agent, _, _ = build(by_id, canonical_policy, planner)
    try:
        result = agent.run_turn("invalid", "查订单物流")
        persisted = agent.store.snapshot("invalid", 2).state.last_planned_tools
        cases["invalid_planner_output"] = {
            "status": result.status,
            "exposed_tools": result.tools,
            "persisted_tools": persisted,
            "passed": result.status == "planner_invalid"
            and not result.tools
            and not persisted,
        }
    finally:
        agent.close()

    planner = FakePlanner()
    agent, _, _ = build(by_id, canonical_policy, planner)
    try:
        agent.run_turn("session-a", "查订单物流")
        result = agent.run_turn("session-b", "继续刚才的操作")
        cases["session_isolation"] = {
            "session_b_status": result.status,
            "session_b_tools": result.tools,
            "session_b_memory_rows": len(agent.store.memory_records("session-b")),
            "passed": result.status == "unresolved_reference"
            and not result.tools
            and not agent.store.memory_records("session-b"),
        }
    finally:
        agent.close()

    planner = FakePlanner()
    timeout_policy = selected_policy(runtime, long_memory_timeout_ms=1)
    agent, _, _ = build(by_id, timeout_policy, planner)
    try:
        agent.run_turn("timeout", "查订单物流")
        agent.run_turn("timeout", "查库存")
        agent.run_turn("timeout", "计算优惠")
        original = agent.long_memory.search

        def slow_search(*values: Any, **kwargs: Any) -> list[Any]:
            time.sleep(0.02)
            return original(*values, **kwargs)

        agent.long_memory.search = slow_search  # type: ignore[method-assign]
        started = time.perf_counter()
        result = agent.run_turn("timeout", "回到第1轮那个方案")
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        cases["long_memory_timeout"] = {
            "status": result.status,
            "tools": result.tools,
            "warnings": result.warnings,
            "run_turn_elapsed_ms": elapsed_ms,
            "passed": result.status == "unresolved_reference"
            and not result.tools
            and "long_memory_timeout_fallback" in result.warnings,
        }
    finally:
        agent.close()

    result = {
        "status": "complete",
        "canonical_runtime_config_hash": config_hash(runtime),
        "selected_config_id": runtime["selection_status"]["selected_config_id"],
        "cases": cases,
        "passed": sum(row["passed"] for row in cases.values()),
        "total": len(cases),
        "strict_ok": all(row["passed"] for row in cases.values()),
    }
    if not result["strict_ok"]:
        raise SystemExit(f"failure guard audit failed: {result}")
    write_json(args.output, result)
    print(f"failure guards passed {result['passed']}/{result['total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
