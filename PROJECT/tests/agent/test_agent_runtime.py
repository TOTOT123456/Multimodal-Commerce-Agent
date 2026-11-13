from __future__ import annotations

import json
import stat
import tempfile
import time
import unittest
from pathlib import Path

from PROJECT.agent.backends import CanonicalRoute
from PROJECT.agent.factory import PROJECT_ROOT, load_agent_config
from PROJECT.agent.memory import (
    ConcurrentTurnError,
    DeterministicMemoryReducer,
    SQLiteMemoryStore,
)
from PROJECT.agent.reference_selection import (
    is_whole_turn_reference,
    reference_search_text,
    is_implicit_confirmation,
)
from PROJECT.agent.orchestrator import AgentPolicy, ToolPlanningAgent
from PROJECT.agent.resolver import is_cancellation_clause
from PROJECT.agent.types import (
    CompactToolState,
    Goal,
    IntentResult,
    MemoryRecord,
    PlanEvent,
    PlannerResult,
    RetrievalResult,
    ToolRollingSummary,
    utc_now,
)
from PROJECT.scripts.project_common import catalog_by_id, load_catalog
from PROJECT.scripts.agent.evaluate_multiturn_agent import validate_dataset


class FakeIntent:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def classify(self, query: str) -> IntentResult:
        self.queries.append(query)
        intent = "闲聊类" if "天气" in query else "电商类"
        return IntentResult(intent_type=intent, latency_ms=1.0)


class FakeRetriever:
    def __init__(self, candidates: list[str]) -> None:
        self.candidates = candidates
        self.queries: list[str] = []

    def retrieve(self, query: str) -> RetrievalResult:
        self.queries.append(query)
        return RetrievalResult(
            candidate_tools=list(self.candidates),
            source_tools=list(self.candidates),
            reranked_tools=list(self.candidates),
            revision="test-route",
            route_hash="route-hash",
            latency_ms=2.0,
        )


class FakePlanner:
    def __init__(
        self,
        candidates: list[str],
        fixed_input_tokens: int | None = None,
        strict_ok: bool = True,
        inference_token_delta: int = 0,
    ) -> None:
        self.candidates = candidates
        self.fixed_input_tokens = fixed_input_tokens
        self.strict_ok = strict_ok
        self.inference_token_delta = inference_token_delta
        self.plan_queries: list[str] = []

    def count_input_tokens(self, query: str, candidate_tools: list[str]) -> int:
        self.assert_candidates(candidate_tools)
        return self.fixed_input_tokens or 100 + len(query)

    def assert_candidates(self, candidate_tools: list[str]) -> None:
        if candidate_tools != self.candidates:
            raise AssertionError("candidate order changed")

    def plan(self, query: str, candidate_tools: list[str]) -> PlannerResult:
        self.assert_candidates(candidate_tools)
        self.plan_queries.append(query)
        selected: list[tuple[int, str]] = []
        if "物流" in query and "不再需要" not in query:
            selected.append((query.index("物流"), "get_order_logistics"))
        delivery_cues = [value for value in ("什么时候到", "配送时效", "到达时间") if value in query]
        if delivery_cues:
            selected.append((min(query.index(value) for value in delivery_cues), "estimate_delivery_time"))
        if "库存" in query:
            selected.append((query.index("库存"), "get_sku_stock"))
        if "优惠" in query:
            selected.append((query.index("优惠"), "calc_discount"))
        tools = list(dict.fromkeys(tool for _, tool in sorted(selected)))
        raw = json.dumps({"tools": tools}, ensure_ascii=False, separators=(",", ":"))
        return PlannerResult(
            tools=tools if self.strict_ok else [],
            strict_ok=self.strict_ok,
            strict_error="" if self.strict_ok else "invalid_json",
            raw_output=raw if self.strict_ok else "not-json",
            input_tokens=(
                self.count_input_tokens(query, candidate_tools) + self.inference_token_delta
            ),
            output_tokens=8,
            latency_ms=3.0,
            prompt_version="tool-trace-v1",
            prompt_hash="prompt-hash",
            model_revision="model-hash",
        )


class AgentRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        catalog = load_catalog(PROJECT_ROOT / "configs/tool_catalog.jsonl")
        cls.catalog = catalog_by_id(catalog)
        preferred = [
            "get_order_logistics",
            "estimate_delivery_time",
            "get_sku_stock",
            "calc_discount",
            "get_order_detail",
            "get_available_coupons",
            "apply_coupon",
            "get_cart_summary",
            "cancel_order",
            "get_order_list",
        ]
        assert set(preferred) <= set(cls.catalog)
        cls.candidates = preferred

    def make_agent(
        self,
        policy: AgentPolicy | None = None,
        fixed_input_tokens: int | None = None,
        strict_ok: bool = True,
        inference_token_delta: int = 0,
    ) -> tuple[ToolPlanningAgent, FakeIntent, FakeRetriever, FakePlanner]:
        store = SQLiteMemoryStore(":memory:", ttl_seconds=3600)
        intent = FakeIntent()
        retriever = FakeRetriever(self.candidates)
        planner = FakePlanner(
            self.candidates,
            fixed_input_tokens=fixed_input_tokens,
            strict_ok=strict_ok,
            inference_token_delta=inference_token_delta,
        )
        selected_policy = policy or AgentPolicy()
        agent = ToolPlanningAgent(
            store=store,
            intent_backend=intent,
            retriever=retriever,
            planner=planner,
            catalog_by_id=self.catalog,
            policy=selected_policy,
            agent_config_hash="agent-hash",
            route_revision="test-route",
            route_hash="route-hash",
            memory_reducer=DeterministicMemoryReducer(
                self.catalog,
                recent_event_limit=selected_policy.recent_event_limit,
                max_summary_goals=20,
                memory_ttl_seconds=3600,
            ),
        )
        return agent, intent, retriever, planner

    def test_context_independent_query_is_unchanged(self) -> None:
        agent, _, retriever, planner = self.make_agent()
        try:
            agent.run_turn("s", "先查订单物流，再估计什么时候到。")
            result = agent.run_turn("s", "帮我查一下库存")
            self.assertEqual(result.context_mode, "none")
            self.assertEqual(result.resolved_query, "帮我查一下库存")
            self.assertEqual(retriever.queries[-1], "帮我查一下库存")
            self.assertEqual(planner.plan_queries[-1], "帮我查一下库存")
            self.assertEqual(result.tools, ["get_sku_stock"])
            self.assertEqual(agent._session_locks, {})
        finally:
            agent.close()

    def test_reference_cleanup_does_not_corrupt_city_name(self) -> None:
        agent, _, _, _ = self.make_agent()
        try:
            result = agent.run_turn("s", "查询成都的配送时效")
            self.assertEqual(result.context_mode, "none")
            self.assertEqual(result.resolved_query, "查询成都的配送时效")
            self.assertTrue(any("成都" in goal.text for goal in result.event.current_goals))
        finally:
            agent.close()

    def test_cancel_order_is_a_tool_goal_not_meta_cancellation(self) -> None:
        agent, _, _, _ = self.make_agent()
        try:
            result = agent.run_turn("s", "请取消这笔订单")
            self.assertEqual(result.event.context_mode, "none")
            self.assertEqual(
                [goal.text for goal in result.event.current_goals], ["取消这笔订单"]
            )
        finally:
            agent.close()

    def test_business_cancellation_inside_a_long_request_is_not_meta_cancellation(self) -> None:
        agent, _, _, _ = self.make_agent()
        try:
            agent.run_turn("s", "查订单物流")
            query = (
                "帮我查退款进度；把还没完成的退货申请取消；"
                "如果还能操作就重新提交退货退款。"
            )
            result = agent.run_turn("s", query)
            self.assertEqual(result.resolved_query, query)
            self.assertEqual(result.event.carried_goals, [])
            self.assertEqual(result.event.cancelled_goals, [])
            self.assertTrue(
                any("退货申请取消" in goal.text for goal in result.event.current_goals)
            )
        finally:
            agent.close()

    def test_cancel_the_order_phrase_remains_a_current_tool_goal(self) -> None:
        agent, _, _, _ = self.make_agent()
        try:
            agent.run_turn("s", "查库存")
            query = "不能合并的话就把还允许取消的那单取消掉。"
            result = agent.run_turn("s", query)
            self.assertEqual(result.resolved_query, query)
            self.assertEqual(result.event.carried_goals, [])
            self.assertEqual(result.event.cancelled_goals, [])
            self.assertTrue(result.event.current_goals)
        finally:
            agent.close()

    def test_cart_item_deletion_does_not_cancel_a_previous_goal(self) -> None:
        agent, _, _, _ = self.make_agent()
        try:
            agent.run_turn("s", "查订单详情")
            query = "如果购物车里那个失效的滤芯不能买了就删掉。"
            result = agent.run_turn("s", query)
            self.assertEqual(result.resolved_query, query)
            self.assertEqual(result.event.carried_goals, [])
            self.assertEqual(result.event.cancelled_goals, [])
            self.assertEqual(
                [goal.text for goal in result.event.current_goals],
                ["如果购物车里那个失效的滤芯不能买了就删掉"],
            )
        finally:
            agent.close()

    def test_tool_goal_suffix_named_plan_is_preserved(self) -> None:
        agent, _, _, _ = self.make_agent()
        try:
            result = agent.run_turn("s", "请查看这件商品的优惠方案")
            self.assertEqual(
                [goal.text for goal in result.event.current_goals],
                ["查看这件商品的优惠方案"],
            )
        finally:
            agent.close()

    def test_catalog_alias_guard_bypasses_planner_only_for_complete_exact_suffix(self) -> None:
        policy = AgentPolicy(catalog_alias_guard=True)
        agent, intent, retriever, planner = self.make_agent(policy=policy)
        try:
            result = agent.run_turn("s", "请核实这件商品的库存")
            self.assertEqual(result.tools, ["get_sku_stock"])
            self.assertEqual(result.event.prompt_version, "catalog-alias-direct-v2")
            self.assertEqual(result.event.input_tokens, 0)
            self.assertIn("catalog_alias_direct_applied", result.warnings)
            self.assertEqual(intent.queries, [])
            self.assertEqual(retriever.queries, [])

            retained = agent.run_turn("u", "只保留确认这件商品的库存")
            self.assertEqual(retained.tools, ["get_sku_stock"])
            self.assertEqual(retained.event.prompt_version, "catalog-alias-direct-v2")
            self.assertEqual(planner.plan_queries, [])

            fallback = agent.run_turn("t", "请核实库存情况")
            self.assertEqual(fallback.tools, ["get_sku_stock"])
            self.assertEqual(fallback.event.prompt_version, "tool-trace-v1")
            self.assertEqual(planner.plan_queries, ["请核实库存情况"])
        finally:
            agent.close()

    def test_catalog_alias_guard_requires_every_goal_to_match(self) -> None:
        policy = AgentPolicy(catalog_alias_guard=True)
        agent, _, _, planner = self.make_agent(policy=policy)
        try:
            query = "先核实这件商品的库存；再估计什么时候到。"
            result = agent.run_turn("s", query)
            self.assertEqual(
                result.tools,
                ["get_sku_stock", "estimate_delivery_time"],
            )
            self.assertEqual(result.event.prompt_version, "tool-trace-v1")
            self.assertEqual(planner.plan_queries, [query])
        finally:
            agent.close()

    def test_price_history_tool_goal_does_not_trigger_long_memory(self) -> None:
        agent, _, _, _ = self.make_agent()
        try:
            result = agent.run_turn("s", "请查这件商品的价格历史")
            self.assertEqual(result.context_mode, "none")
        finally:
            agent.close()

    def test_first_turn_business_time_reference_keeps_raw_query(self) -> None:
        agent, _, _, _ = self.make_agent()
        try:
            query = "之前有些商品会限购，帮我确认这件能不能买"
            result = agent.run_turn("s", query)
            self.assertEqual(result.context_mode, "none")
            self.assertEqual(result.resolved_query, query)
        finally:
            agent.close()

    def test_word_separately_is_not_a_cancellation_signal(self) -> None:
        agent, _, _, _ = self.make_agent()
        try:
            query = "帮我比较满减和会员价分别组合后哪个最低"
            result = agent.run_turn("s", query)
            self.assertEqual(result.context_mode, "none")
            self.assertEqual(result.resolved_query, query)
        finally:
            agent.close()

    def test_difference_and_mop_substrings_are_not_meta_cancellations(self) -> None:
        agent, _, _, _ = self.make_agent()
        try:
            difference = agent.run_turn(
                "difference", "帮我比较两款商品的价格和服务差别"
            )
            self.assertEqual(difference.context_mode, "none")
            self.assertNotEqual(difference.status, "unresolved_reference")

            mop = agent.run_turn(
                "mop", "请核实这件蒸汽拖把静音款的取消退款"
            )
            self.assertEqual(mop.context_mode, "none")
            self.assertEqual(
                [goal.text for goal in mop.event.current_goals],
                ["核实这件蒸汽拖把静音款的取消退款"],
            )
        finally:
            agent.close()

    def test_natural_request_boundary_guard_has_a_reproducible_disabled_policy(self) -> None:
        cart_delete = "如果购物车里有失效商品就删掉"
        purchase_question = "帮我看看这件商品要不要买"
        self.assertFalse(is_cancellation_clause(cart_delete))
        self.assertFalse(is_cancellation_clause(purchase_question))
        self.assertTrue(
            is_cancellation_clause(
                cart_delete, natural_request_boundary_guard=False
            )
        )
        self.assertTrue(
            is_cancellation_clause(
                purchase_question, natural_request_boundary_guard=False
            )
        )

    def test_recent_marker_wins_over_generic_previous_word(self) -> None:
        agent, _, _, _ = self.make_agent(
            policy=AgentPolicy(recent_over_generic_long=True)
        )
        try:
            agent.run_turn("s", "查订单物流")
            result = agent.run_turn("s", "同样上一轮的之前方案照旧")
            self.assertEqual(result.context_mode, "recent")
            self.assertEqual(result.event.reference_turn_ids, [1])
            self.assertEqual(result.tools, ["get_order_logistics"])
        finally:
            agent.close()

    def test_multigoal_standalone_guard_ignores_business_history_words(self) -> None:
        policy = AgentPolicy(
            recent_over_generic_long=True,
            standalone_multigoal_guard=True,
        )
        agent, _, _, _ = self.make_agent(policy=policy)
        try:
            agent.run_turn("s", "查订单详情")
            query = "查我之前提交的退款进度；再看这件商品的库存。"
            result = agent.run_turn("s", query)
            self.assertEqual(result.context_mode, "none")
            self.assertEqual(result.event.reference_turn_ids, [])
            self.assertEqual(result.event.carried_goals, [])
            self.assertEqual(result.resolved_query, query)
            self.assertIn("standalone_multigoal_context_guard", result.warnings)
        finally:
            agent.close()

    def test_multigoal_standalone_guard_keeps_short_object_reference(self) -> None:
        policy = AgentPolicy(standalone_multigoal_guard=True)
        agent, _, _, _ = self.make_agent(policy=policy)
        try:
            agent.run_turn("s", "查订单物流")
            result = agent.run_turn("s", "查询那个商品的库存")
            self.assertEqual(result.context_mode, "recent")
            self.assertNotIn("standalone_multigoal_context_guard", result.warnings)
        finally:
            agent.close()

    def test_single_tool_goal_normalization_preserves_full_natural_query(self) -> None:
        policy = AgentPolicy(single_tool_goal_normalization=True)
        agent, _, _, _ = self.make_agent(policy=policy)
        try:
            query = "购物车商品可不可售，有没有限购，库存够不够"
            result = agent.run_turn("s", query)
            self.assertEqual(result.tools, ["get_sku_stock"])
            self.assertEqual(
                [goal.text for goal in result.event.resolved_goals],
                [query],
            )
            self.assertIn("single_tool_goal_normalized", result.warnings)
        finally:
            agent.close()

    def test_whole_turn_reference_replays_the_exact_planner_query(self) -> None:
        policy = AgentPolicy(
            whole_turn_reference_replay=True,
            max_resolved_extra_tokens=0,
        )
        agent, _, _, planner = self.make_agent(policy=policy)
        try:
            query = "先查这个订单的物流，再估计什么时候到。"
            first = agent.run_turn("s", query)
            self.assertEqual(first.tools, ["get_order_logistics", "estimate_delivery_time"])
            repeated = agent.run_turn("s", "继续刚才的全部任务")
            self.assertEqual(repeated.resolved_query, query)
            self.assertEqual(repeated.tools, first.tools)
            self.assertEqual(planner.plan_queries[-1], query)
            self.assertIn("whole_turn_reference_replayed", repeated.warnings)
            self.assertIn("trusted_replay_extra_budget_exempt", repeated.warnings)
        finally:
            agent.close()

    def test_whole_turn_reference_replays_compacted_long_memory(self) -> None:
        policy = AgentPolicy(
            whole_turn_reference_replay=True,
            recent_event_limit=2,
        )
        agent, _, _, _ = self.make_agent(policy=policy)
        try:
            query = "先查这个订单的物流，再估计什么时候到。"
            first = agent.run_turn("s", query)
            agent.run_turn("s", "查库存")
            agent.run_turn("s", "计算优惠")
            agent.run_turn("s", "查订单详情")
            repeated = agent.run_turn("s", "回到第1轮的全部任务")
            self.assertEqual(repeated.event.reference_turn_ids, [1])
            self.assertEqual(repeated.resolved_query, query)
            self.assertEqual(repeated.tools, first.tools)
            self.assertIn("whole_turn_reference_replayed", repeated.warnings)
        finally:
            agent.close()

    def test_semantic_reference_strategy_recovers_named_old_turn(self) -> None:
        policy = AgentPolicy(
            recent_over_generic_long=True,
            natural_request_boundary_guard=True,
            whole_turn_reference_replay=True,
            semantic_reference_strategy="anchor_bm25",
        )
        agent, _, _, _ = self.make_agent(policy=policy)
        try:
            agent.run_turn("s", "查蓝色羽绒服库存")
            agent.run_turn("s", "查订单物流")
            agent.run_turn("s", "计算优惠")
            result = agent.run_turn(
                "s", "最早提到蓝色羽绒服的那套事情，照原样再办一次。"
            )
            self.assertEqual(result.context_mode, "long")
            self.assertEqual(result.event.reference_turn_ids, [1])
            self.assertEqual(result.event.current_goals, [])
            self.assertEqual(result.resolved_query, "查蓝色羽绒服库存")
        finally:
            agent.close()

    def test_natural_confirmation_replays_the_previous_whole_turn(self) -> None:
        policy = AgentPolicy(
            whole_turn_reference_replay=True,
            whole_turn_plan_cache=True,
            semantic_reference_strategy="anchor_bm25",
        )
        agent, _, _, planner = self.make_agent(policy=policy)
        try:
            source = "先查订单物流，再估计什么时候到。"
            first = agent.run_turn("s", source)
            repeated = agent.run_turn("s", "行，就按这个处理吧")
            self.assertEqual(repeated.event.reference_turn_ids, [1])
            self.assertEqual(repeated.resolved_query, source)
            self.assertEqual(repeated.tools, first.tools)
            self.assertIn("whole_turn_plan_cache_applied", repeated.warnings)
            self.assertEqual(len(planner.plan_queries), 1)
        finally:
            agent.close()

    def test_natural_plural_scope_recovers_a_named_older_turn(self) -> None:
        policy = AgentPolicy(
            recent_over_generic_long=True,
            natural_request_boundary_guard=True,
            whole_turn_reference_replay=True,
            semantic_reference_strategy="anchor_bm25",
        )
        agent, _, _, _ = self.make_agent(policy=policy)
        try:
            source = "先查蓝色羽绒服库存，再推荐合适尺码。"
            agent.run_turn("s", source)
            agent.run_turn("s", "查订单物流")
            agent.run_turn("s", "计算购物车优惠")
            repeated = agent.run_turn(
                "s", "前面说的蓝色羽绒服那几件事接着帮我办吧"
            )
            self.assertEqual(repeated.event.reference_turn_ids, [1])
            self.assertEqual(repeated.resolved_query, source)
        finally:
            agent.close()

    def test_semantic_reference_strategy_keeps_complete_new_request_standalone(self) -> None:
        policy = AgentPolicy(
            recent_over_generic_long=True,
            natural_request_boundary_guard=True,
            whole_turn_reference_replay=True,
            standalone_multigoal_guard=True,
            semantic_reference_strategy="anchor_bm25_position",
        )
        agent, _, _, _ = self.make_agent(policy=policy)
        try:
            agent.run_turn("s", "查订单物流")
            query = "我之前买的羽绒服，帮我查一下黑色 L 码库存。"
            result = agent.run_turn("s", query)
            self.assertEqual(result.event.reference_turn_ids, [])
            self.assertEqual(result.event.carried_goals, [])
            self.assertEqual(result.resolved_query, query)
        finally:
            agent.close()

    def test_semantic_reference_position_selects_turn_between_two_anchors(self) -> None:
        policy = AgentPolicy(
            recent_over_generic_long=True,
            natural_request_boundary_guard=True,
            whole_turn_reference_replay=True,
            semantic_reference_strategy="anchor_bm25_position",
        )
        agent, _, _, _ = self.make_agent(policy=policy)
        try:
            agent.run_turn("s", "查询售后进度")
            agent.run_turn("s", "查询先用后付还款时间")
            agent.run_turn("s", "查蓝色羽绒服库存")
            result = agent.run_turn(
                "s", "夹在售后那条和蓝色羽绒服那条中间的请求，原样重做一遍。"
            )
            self.assertEqual(result.event.reference_turn_ids, [2])
            self.assertEqual(result.resolved_query, "查询先用后付还款时间")
        finally:
            agent.close()

    def test_whole_turn_reference_detector_requires_history_and_replay(self) -> None:
        self.assertTrue(
            is_whole_turn_reference("之前那笔订单的全部事情照原样再办一次")
        )
        self.assertTrue(
            is_whole_turn_reference("夹在前后两条中间的请求，原样重做一遍")
        )
        self.assertFalse(
            is_whole_turn_reference("我之前买的羽绒服，帮我查黑色 L 码库存")
        )
        self.assertTrue(is_whole_turn_reference("行，就按这个处理吧"))
        self.assertTrue(is_implicit_confirmation("行，就按这个处理吧"))
        self.assertTrue(is_implicit_confirmation("可以，先按刚才那样办吧"))
        self.assertTrue(
            is_implicit_confirmation("行，婴儿车那个订单就按刚才说的办吧")
        )
        self.assertTrue(is_implicit_confirmation("行，按这个帮我处理吧"))
        self.assertTrue(
            is_implicit_confirmation("行，按刚才说的那单羽绒服继续处理吧")
        )
        self.assertTrue(
            is_whole_turn_reference("刚才说的那几件事继续帮我办吧")
        )
        self.assertTrue(is_whole_turn_reference("耳机那单，前面说的都接着办吧"))
        self.assertTrue(
            is_whole_turn_reference("露营帐篷和睡袋那几件事先继续看下")
        )
        self.assertTrue(is_whole_turn_reference("刚才聊的那几项一起处理吧"))
        self.assertTrue(is_implicit_confirmation("可以，就这么改吧"))
        self.assertTrue(is_implicit_confirmation("好，就这么定了"))
        self.assertTrue(
            is_whole_turn_reference("那条黑色直筒牛仔裤，前面说的那些先帮我记上吧")
        )
        self.assertTrue(
            is_whole_turn_reference("小熊空气炸锅那单，刚才说的那些还按着办吧")
        )
        self.assertTrue(is_whole_turn_reference("刚才那单价格的事继续帮我看下吧"))
        self.assertFalse(
            is_whole_turn_reference("那个耳机订单继续帮我查一下物流")
        )
        self.assertEqual(
            reference_search_text("最早提到蓝色 M 码那单的整套需求，照原样再办一遍"),
            "蓝色 M 码",
        )

    def test_earliest_reference_reads_an_uncompacted_turn_in_n4(self) -> None:
        policy = AgentPolicy(
            whole_turn_reference_replay=True,
            whole_turn_plan_cache=True,
            recent_event_limit=4,
        )
        agent, intent, retriever, planner = self.make_agent(policy=policy)
        try:
            query = "先查这个订单的物流，再估计什么时候到。"
            first = agent.run_turn("s", query)
            agent.run_turn("s", "查库存")
            agent.run_turn("s", "计算优惠")
            agent.run_turn("s", "查订单详情")
            repeated = agent.run_turn("s", "回到最开始的全部任务")
            self.assertEqual(repeated.event.reference_turn_ids, [1])
            self.assertEqual(repeated.resolved_query, query)
            self.assertEqual(repeated.tools, first.tools)
            self.assertEqual(repeated.event.prompt_version, "whole-turn-plan-replay-v1")
            self.assertEqual(len(intent.queries), 4)
            self.assertEqual(len(retriever.queries), 4)
            self.assertEqual(len(planner.plan_queries), 4)
        finally:
            agent.close()

    def test_whole_turn_plan_cache_bypasses_all_model_backends(self) -> None:
        policy = AgentPolicy(
            whole_turn_reference_replay=True,
            whole_turn_plan_cache=True,
        )
        agent, intent, retriever, planner = self.make_agent(policy=policy)
        try:
            query = "先查这个订单的物流，再估计什么时候到。"
            first = agent.run_turn("s", query)
            repeated = agent.run_turn("s", "继续刚才的全部任务")
            self.assertEqual(repeated.tools, first.tools)
            self.assertEqual(repeated.resolved_query, query)
            self.assertEqual(repeated.event.prompt_version, "whole-turn-plan-replay-v1")
            self.assertEqual(repeated.event.input_tokens, 0)
            self.assertEqual(len(intent.queries), 1)
            self.assertEqual(len(retriever.queries), 1)
            self.assertEqual(len(planner.plan_queries), 1)
            self.assertIn("whole_turn_plan_cache_applied", repeated.warnings)
        finally:
            agent.close()

    def test_whole_turn_plan_cache_falls_back_after_runtime_hash_change(self) -> None:
        policy = AgentPolicy(
            whole_turn_reference_replay=True,
            whole_turn_plan_cache=True,
        )
        agent, intent, retriever, planner = self.make_agent(policy=policy)
        try:
            query = "先查这个订单的物流，再估计什么时候到。"
            agent.run_turn("s", query)
            agent.agent_config_hash = "new-runtime-hash"
            repeated = agent.run_turn("s", "继续刚才的全部任务")
            self.assertEqual(repeated.resolved_query, query)
            self.assertEqual(repeated.event.prompt_version, "tool-trace-v1")
            self.assertNotIn("whole_turn_plan_cache_applied", repeated.warnings)
            self.assertEqual(len(intent.queries), 2)
            self.assertEqual(len(retriever.queries), 2)
            self.assertEqual(len(planner.plan_queries), 2)
        finally:
            agent.close()

    def test_first_turn_concrete_delete_is_not_blocked_as_unresolved(self) -> None:
        agent, _, _, _ = self.make_agent()
        try:
            query = "刚才购物车里那件衣服不要了，帮我删掉"
            result = agent.run_turn("s", query)
            self.assertEqual(result.context_mode, "none")
            self.assertEqual(result.resolved_query, query)
            self.assertNotEqual(result.status, "unresolved_reference")
        finally:
            agent.close()

    def test_recent_cancellation_removes_old_goal(self) -> None:
        agent, _, _, _ = self.make_agent()
        try:
            first = agent.run_turn("s", "先查这个订单的物流，再估计什么时候到。")
            self.assertEqual(
                first.tools,
                ["get_order_logistics", "estimate_delivery_time"],
            )
            second = agent.run_turn("s", "物流不用查了，只看什么时候到。")
            self.assertEqual(second.context_mode, "recent")
            self.assertEqual(second.tools, ["estimate_delivery_time"])
            self.assertIn(1, second.event.reference_turn_ids)
            self.assertTrue(any("物流" in goal.text for goal in second.event.cancelled_goals))
            self.assertFalse(any(tool == "get_order_logistics" for tool in second.tools))
            self.assertTrue(
                all(goal.source_turn_ids == [2] for goal in second.event.current_goals)
            )
            self.assertTrue(
                any(goal.source_turn_ids == [1, 2] for goal in second.event.resolved_goals)
            )
        finally:
            agent.close()

    def test_reference_clause_order_is_preserved_before_new_goal(self) -> None:
        agent, _, _, _ = self.make_agent()
        try:
            agent.run_turn("s", "查订单物流")
            result = agent.run_turn("s", "继续刚才的；再查库存")
            self.assertEqual(
                result.tools,
                ["get_order_logistics", "get_sku_stock"],
            )
            self.assertLess(result.resolved_query.index("物流"), result.resolved_query.index("库存"))
        finally:
            agent.close()

    def test_architecture_a_never_reads_history(self) -> None:
        agent, _, _, _ = self.make_agent(policy=AgentPolicy(context_architecture="A"))
        try:
            agent.run_turn("s", "查订单物流")
            result = agent.run_turn("s", "继续刚才的")
            self.assertEqual(result.context_mode, "none")
            self.assertEqual(result.resolved_query, "继续刚才的")
            self.assertEqual(result.tools, [])
        finally:
            agent.close()

    def test_architecture_b_directly_concatenates_recent_queries(self) -> None:
        agent, _, _, _ = self.make_agent(policy=AgentPolicy(context_architecture="B"))
        try:
            agent.run_turn("s", "查订单物流")
            result = agent.run_turn("s", "继续刚才的")
            self.assertIn("历史用户请求", result.resolved_query)
            self.assertIn("查订单物流", result.resolved_query)
            self.assertEqual(result.tools, ["get_order_logistics"])
        finally:
            agent.close()

    def test_architecture_b_explicit_budget_override_does_not_fall_back(self) -> None:
        policy = AgentPolicy(
            context_architecture="B",
            target_input_tokens=300,
            hard_input_tokens=500,
            max_resolved_extra_tokens=2,
            direct_history_max_extra_tokens=180,
        )
        agent, _, _, _ = self.make_agent(policy=policy)
        try:
            agent.run_turn("s", "先查订单物流，再估计什么时候到")
            result = agent.run_turn("s", "继续刚才两个操作")
            self.assertIn("历史用户请求", result.resolved_query)
            self.assertIn("查订单物流", result.resolved_query)
            self.assertNotIn("resolved_query_budget_fallback_variant:1", result.warnings)
            self.assertIn("direct_history_concat_forced", result.warnings)
        finally:
            agent.close()

    def test_architecture_b_forced_baseline_always_concatenates_recent_queries(self) -> None:
        policy = AgentPolicy(
            context_architecture="B",
            target_input_tokens=300,
            hard_input_tokens=500,
            direct_history_max_extra_tokens=180,
        )
        agent, _, _, _ = self.make_agent(policy=policy)
        try:
            agent.run_turn("s", "查订单物流")
            result = agent.run_turn("s", "今天有什么促销")
            self.assertIn("历史用户请求", result.resolved_query)
            self.assertIn("查订单物流", result.resolved_query)
            self.assertIn("direct_history_concat_forced", result.warnings)
        finally:
            agent.close()

    def test_architecture_c_cannot_recover_compacted_turn(self) -> None:
        policy = AgentPolicy(context_architecture="C", recent_event_limit=3)
        agent, _, _, _ = self.make_agent(policy=policy)
        try:
            agent.run_turn("s", "查订单物流")
            agent.run_turn("s", "查库存")
            agent.run_turn("s", "计算优惠")
            agent.run_turn("s", "查订单详情")
            result = agent.run_turn("s", "回到第一轮那个方案")
            self.assertEqual(result.event.reference_turn_ids, [])
            self.assertEqual(result.tools, [])
        finally:
            agent.close()

    def test_architecture_d_recovers_compacted_turn_from_summary(self) -> None:
        policy = AgentPolicy(context_architecture="D", recent_event_limit=3)
        agent, _, _, _ = self.make_agent(policy=policy)
        try:
            agent.run_turn("s", "查订单物流")
            agent.run_turn("s", "查库存")
            agent.run_turn("s", "计算优惠")
            agent.run_turn("s", "查订单详情")
            result = agent.run_turn("s", "回到第一轮那个方案")
            self.assertEqual(result.event.reference_turn_ids, [1])
            self.assertEqual(result.tools, ["get_order_logistics"])
        finally:
            agent.close()

    def test_object_reference_does_not_carry_old_tools(self) -> None:
        agent, _, _, _ = self.make_agent()
        try:
            agent.run_turn("s", "查订单物流")
            result = agent.run_turn("s", "查询那个商品的库存")
            self.assertEqual(result.context_mode, "recent")
            self.assertEqual(result.tools, ["get_sku_stock"])
            self.assertNotIn("get_order_logistics", result.tools)
            self.assertEqual(result.event.carried_goals, [])
        finally:
            agent.close()

    def test_chitchat_breaks_implicit_recent_reference(self) -> None:
        agent, _, _, _ = self.make_agent()
        try:
            agent.run_turn("s", "查订单物流")
            chat = agent.run_turn("s", "今天天气怎么样")
            self.assertEqual(chat.status, "chitchat")
            continued = agent.run_turn("s", "继续刚才的")
            self.assertEqual(continued.event.reference_turn_ids, [])
            self.assertEqual(continued.tools, [])
            state = agent.store.snapshot("s", 4).state
            self.assertEqual(state.last_planned_tools, [])
        finally:
            agent.close()

    def test_long_exact_turn_retrieval_after_compaction(self) -> None:
        agent, _, _, _ = self.make_agent()
        try:
            agent.run_turn("s", "先查订单物流，再估计什么时候到")
            agent.run_turn("s", "查库存")
            agent.run_turn("s", "计算优惠")
            agent.run_turn("s", "查订单详情")
            records = agent.store.memory_records("s")
            self.assertEqual([record.source_turn_id for record in records], [2, 1])
            result = agent.run_turn("s", "回到第一轮那个方案")
            self.assertEqual(result.context_mode, "long")
            self.assertEqual(result.event.reference_turn_ids, [1])
            self.assertEqual(
                result.tools,
                ["get_order_logistics", "estimate_delivery_time"],
            )
            summary = agent.store.snapshot("s", 4).summary
            self.assertGreaterEqual(summary.covered_through_turn, 2)
        finally:
            agent.close()

    def test_sessions_never_share_memory(self) -> None:
        agent, _, _, _ = self.make_agent()
        try:
            agent.run_turn("alice", "查订单物流")
            result = agent.run_turn("bob", "继续刚才的")
            self.assertEqual(result.event.reference_turn_ids, [])
            self.assertEqual(result.tools, [])
            self.assertEqual(agent.store.all_events("alice")[0].conversation_id, "alice")
            self.assertEqual(agent.store.all_events("bob")[0].conversation_id, "bob")
        finally:
            agent.close()

    def test_budget_falls_back_before_planner(self) -> None:
        policy = AgentPolicy(
            recent_event_limit=3,
            long_memory_top_k=2,
            long_memory_timeout_ms=100,
            target_input_tokens=120,
            hard_input_tokens=150,
            max_resolved_extra_tokens=2,
        )
        agent, _, _, planner = self.make_agent(policy=policy)
        try:
            agent.run_turn("s", "先查订单物流，再估计什么时候到")
            result = agent.run_turn("s", "继续刚才两个操作")
            self.assertIn("resolved_query_budget_fallback_variant:1", result.warnings)
            self.assertEqual(result.resolved_query, "继续刚才两个操作")
            self.assertLessEqual(result.event.input_tokens, policy.hard_input_tokens)
            self.assertEqual(planner.plan_queries[-1], result.resolved_query)
        finally:
            agent.close()

    def test_hard_budget_rejects_without_calling_planner(self) -> None:
        policy = AgentPolicy(target_input_tokens=120, hard_input_tokens=150)
        agent, _, _, planner = self.make_agent(policy=policy, fixed_input_tokens=200)
        try:
            result = agent.run_turn("s", "查库存")
            self.assertEqual(result.status, "budget_rejected")
            self.assertEqual(planner.plan_queries, [])
            self.assertIn("planner_input_hard_budget_rejected", result.warnings)
            self.assertEqual(len(agent.store.all_events("s")), 1)
        finally:
            agent.close()

    def test_query_character_limit_rejects_before_backends(self) -> None:
        policy = AgentPolicy(max_query_characters=8)
        agent, intent, retriever, planner = self.make_agent(policy=policy)
        try:
            with self.assertRaises(ValueError):
                agent.run_turn("s", "这是一条明显超过限制的请求")
            self.assertEqual(intent.queries, [])
            self.assertEqual(retriever.queries, [])
            self.assertEqual(planner.plan_queries, [])
            self.assertFalse(agent.store.has_session("s"))
        finally:
            agent.close()

    def test_invalid_planner_output_is_not_persisted_as_tools(self) -> None:
        agent, _, _, _ = self.make_agent(strict_ok=False)
        try:
            result = agent.run_turn("s", "查库存")
            self.assertEqual(result.status, "planner_invalid")
            self.assertEqual(result.tools, [])
            self.assertEqual(agent.store.snapshot("s", 4).state.last_planned_tools, [])
            self.assertTrue(any("planner_strict_error" in value for value in result.warnings))
        finally:
            agent.close()

    def test_post_plan_validation_error_does_not_persist_tools(self) -> None:
        agent, _, _, _ = self.make_agent(inference_token_delta=1)
        try:
            result = agent.run_turn("s", "查库存")
            self.assertEqual(result.status, "backend_error")
            self.assertFalse(result.strict_output)
            self.assertEqual(result.tools, [])
            self.assertEqual(agent.store.snapshot("s", 4).state.last_planned_tools, [])
        finally:
            agent.close()

    def test_long_memory_timeout_falls_back_without_blocking_turn(self) -> None:
        policy = AgentPolicy(long_memory_timeout_ms=1)
        agent, _, _, _ = self.make_agent(policy=policy)
        original_search = agent.long_memory.search

        def slow_search(*args: object, **kwargs: object) -> list[object]:
            time.sleep(0.02)
            return original_search(*args, **kwargs)

        agent.long_memory.search = slow_search  # type: ignore[method-assign]
        try:
            result = agent.run_turn("s", "回到第一轮那个方案")
            self.assertEqual(result.status, "unresolved_reference")
            self.assertEqual(result.tools, [])
            self.assertIn("long_memory_timeout_fallback", result.warnings)
            self.assertEqual(result.event.reference_turn_ids, [])
        finally:
            agent.close()

    def test_unresolved_reference_is_safe_and_preserves_previous_state(self) -> None:
        agent, intent, retriever, planner = self.make_agent()
        try:
            agent.run_turn("s", "查订单物流")
            before = agent.store.snapshot("s", 4).state
            intent_calls = len(intent.queries)
            retrieval_calls = len(retriever.queries)
            planner_calls = len(planner.plan_queries)
            result = agent.run_turn("s", "回到第99轮那个方案")
            after = agent.store.snapshot("s", 4).state
            self.assertEqual(result.status, "unresolved_reference")
            self.assertTrue(result.strict_output)
            self.assertEqual(result.tools, [])
            self.assertEqual(len(intent.queries), intent_calls)
            self.assertEqual(len(retriever.queries), retrieval_calls)
            self.assertEqual(len(planner.plan_queries), planner_calls)
            self.assertEqual(
                [goal.text for goal in after.last_request_goals],
                [goal.text for goal in before.last_request_goals],
            )
            self.assertEqual(after.last_planned_tools, before.last_planned_tools)
        finally:
            agent.close()


class MemoryStoreTest(unittest.TestCase):
    def test_file_database_uses_private_permissions_and_schema_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.sqlite3"
            store = SQLiteMemoryStore(path, ttl_seconds=3600)
            try:
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
                version = store._connection.execute("PRAGMA user_version").fetchone()[0]
                self.assertEqual(version, 1)
            finally:
                store.close()

    def test_compare_and_swap_rejects_stale_commit(self) -> None:
        store = SQLiteMemoryStore(":memory:", ttl_seconds=3600)
        try:
            snapshot = store.snapshot("s", 3)
            event = PlanEvent(
                conversation_id="s",
                turn_id=1,
                created_at=utc_now(),
                raw_query="q",
                resolved_query="q",
                context_mode="none",
                current_goals=[],
                carried_goals=[],
                cancelled_goals=[],
                resolved_goals=[],
                reference_turn_ids=[],
                intent_type="电商类",
                candidate_tools=[],
                planned_tools=[],
                strict_output=True,
                status="ok",
            )
            state = CompactToolState(
                conversation_id="s",
                version=1,
                last_turn_id=1,
            )
            summary = ToolRollingSummary(conversation_id="s")
            store.commit_turn(event, state, summary, [], expected_version=0)
            with self.assertRaises(ConcurrentTurnError):
                store.commit_turn(event, state, summary, [], expected_version=0)
        finally:
            store.close()

    def test_delete_session_cascades_events_and_memory(self) -> None:
        store = SQLiteMemoryStore(":memory:", ttl_seconds=3600)
        try:
            self.assertFalse(store.has_session("s"))
            store.snapshot("s", 3)
            self.assertTrue(store.has_session("s"))
            goal = Goal(text="查库存", source_turn_ids=[1])
            event = PlanEvent(
                conversation_id="s",
                turn_id=1,
                created_at=utc_now(),
                raw_query="查库存",
                resolved_query="查库存",
                context_mode="none",
                current_goals=[goal],
                carried_goals=[],
                cancelled_goals=[],
                resolved_goals=[goal],
                reference_turn_ids=[],
                intent_type="电商类",
                candidate_tools=["get_sku_stock"],
                planned_tools=["get_sku_stock"],
                strict_output=True,
                status="ok",
            )
            state = CompactToolState(
                conversation_id="s",
                version=1,
                last_turn_id=1,
                last_request_goals=[goal],
                last_planned_tools=["get_sku_stock"],
            )
            memory = MemoryRecord(
                conversation_id="s",
                source_turn_id=1,
                text="查库存",
                goals=[goal],
                planned_tools=["get_sku_stock"],
                domains=["inventory"],
                created_at=event.created_at,
            )
            store.commit_turn(
                event,
                state,
                ToolRollingSummary(conversation_id="s"),
                [memory],
                expected_version=0,
            )
            self.assertEqual(len(store.all_events("s")), 1)
            self.assertEqual(len(store.memory_records("s")), 1)
            self.assertTrue(store.delete_session("s"))
            self.assertFalse(store.has_session("s"))
            self.assertEqual(store.all_events("s"), [])
            self.assertEqual(store.memory_records("s"), [])
            recreated = store.snapshot("s", 3)
            self.assertEqual(recreated.state.version, 0)
            self.assertEqual(recreated.recent_events, [])
        finally:
            store.close()

    def test_expired_session_is_hidden_and_purged(self) -> None:
        store = SQLiteMemoryStore(":memory:", ttl_seconds=3600)
        try:
            store.snapshot("expired", 3)
            store._connection.execute(
                "UPDATE sessions SET expires_at = ? WHERE conversation_id = ?",
                ("2000-01-01T00:00:00+00:00", "expired"),
            )
            self.assertFalse(store.has_session("expired"))
            self.assertEqual(store.purge_expired(), 1)
            self.assertEqual(store.all_events("expired"), [])
        finally:
            store.close()


class CanonicalContractTest(unittest.TestCase):
    def test_agent_config_locks_current_route(self) -> None:
        config_path = PROJECT_ROOT / "configs/agent_runtime.json"
        config = load_agent_config(config_path)
        self.assertEqual(config["schema_version"], 3)
        self.assertEqual(config["status"], "canonical")
        self.assertEqual(config["memory"]["context_architecture"], "E")
        self.assertEqual(config["memory"]["recent_event_limit"], 2)
        self.assertEqual(config["memory"]["long_memory_top_k"], 2)
        self.assertEqual(config["token_budget"]["target_input_tokens"], 640)
        self.assertEqual(config["selection_status"]["selected_config_id"], "E3-n2-k2-t640")
        self.assertTrue(config["planning"]["catalog_alias_guard"])
        self.assertTrue(config["planning"]["recent_over_generic_long"])
        self.assertTrue(config["planning"]["natural_request_boundary_guard"])
        self.assertTrue(config["planning"]["single_tool_goal_normalization"])
        self.assertTrue(config["planning"]["whole_turn_reference_replay"])
        self.assertTrue(config["planning"]["standalone_multigoal_guard"])
        self.assertTrue(config["planning"]["whole_turn_plan_cache"])
        self.assertFalse(config["model_artifacts"]["verify_adapter_hash"])
        route_path = Path(__file__).resolve().parents[3] / config["canonical_route"]
        route = CanonicalRoute.load(
            PROJECT_ROOT,
            route_path,
            required_revision=config["required_canonical_revision"],
        )
        self.assertEqual(route.payload["schema_version"], 5)
        self.assertEqual(route.sha256, config["canonical_route_sha256"])
        self.assertEqual(route.payload["first_stage"]["candidate_n_per_source"], 40)
        self.assertEqual(route.payload["reranker"]["candidate_k"], 20)
        self.assertEqual(route.payload["second_stage"]["output_k"], 10)
        self.assertEqual(route.payload["planner"]["candidate_k"], 10)
        self.assertEqual(route.payload["planner"]["max_new_tokens"], 96)
        self.assertEqual(route.payload["planner"]["prompt_version"], "tool-trace-v1")
        self.assertEqual(route.payload["intent"]["max_new_tokens"], 48)
        self.assertEqual(route.payload["catalog"]["path"], "PROJECT/configs/tool_catalog.jsonl")
        self.assertEqual(
            route.payload["embedding"]["query_instruction"],
            "为电商导购助手检索最相关的工具描述：",
        )


class MultiTurnDatasetAuditTest(unittest.TestCase):
    def test_expression_cluster_dict_values_are_protected(self) -> None:
        row = {
            "id": "eval-1",
            "conversation_id": "conversation-1",
            "turn_id": 1,
            "query": "一条全新的请求",
            "gold_tools": ["get_sku_stock"],
            "gold_resolved_goals": ["查库存"],
            "gold_cancelled_goals": [],
            "gold_reference_turn_ids": [],
            "context_dependency": "none",
            "scenario": "self_contained",
            "memory_distance": None,
            "source_trace_id": "new-source",
            "expression_cluster": "protected-cluster",
            "context_skeleton": "new-skeleton",
            "eval_split": "dev",
        }
        protected = {
            "id": "protected-source",
            "问题": "另一条请求",
            "expression_cluster_ids": {"get_sku_stock": "protected-cluster"},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protected.jsonl"
            path.write_text(json.dumps(protected, ensure_ascii=False) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_dataset([row], "dev", {"get_sku_stock"}, [path])


if __name__ == "__main__":
    unittest.main()
