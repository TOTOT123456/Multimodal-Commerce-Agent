"""多轮工具规划的 Context Orchestrator。"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

from .backends import IntentBackend, PlannerBackend, ToolRetriever
from .memory import (
    DeterministicMemoryReducer,
    LexicalLongTermMemory,
    SQLiteMemoryStore,
    bm25_scores,
)
from .resolver import (
    ContextRouter,
    DeterministicResolver,
    DirectHistoryResolver,
    ToolStateReducer,
    explicit_turn_references,
    is_first_turn_standalone,
)
from .types import (
    AgentTurnResult,
    Goal,
    IntentResult,
    MemoryHit,
    MemoryRecord,
    PlanEvent,
    PlannerResult,
    RetrievalResult,
    ToolRollingSummary,
    utc_now,
)


@dataclass(frozen=True, slots=True)
class AgentPolicy:
    context_architecture: str = "E"
    recent_event_limit: int = 2
    long_memory_top_k: int = 2
    long_memory_timeout_ms: int = 100
    target_input_tokens: int = 640
    hard_input_tokens: int = 1024
    max_resolved_extra_tokens: int = 128
    direct_history_max_extra_tokens: int | None = None
    max_query_characters: int = 4096
    catalog_alias_guard: bool = False
    single_tool_goal_normalization: bool = False
    recent_over_generic_long: bool = False
    natural_request_boundary_guard: bool = True
    whole_turn_reference_replay: bool = False
    standalone_multigoal_guard: bool = False
    whole_turn_plan_cache: bool = False
    semantic_reference_strategy: str = "off"

    def __post_init__(self) -> None:
        if self.context_architecture not in {"A", "B", "C", "D", "E"}:
            raise ValueError("context_architecture must be one of A/B/C/D/E")
        if not 2 <= self.recent_event_limit <= 4:
            raise ValueError("recent_event_limit must be in [2, 4]")
        if not 1 <= self.long_memory_top_k <= 3:
            raise ValueError("long_memory_top_k must be in [1, 3]")
        if self.long_memory_timeout_ms <= 0:
            raise ValueError("long_memory_timeout_ms must be positive")
        if self.target_input_tokens <= 0:
            raise ValueError("target_input_tokens must be positive")
        if self.hard_input_tokens < self.target_input_tokens:
            raise ValueError("hard_input_tokens must not be smaller than target")
        if self.max_resolved_extra_tokens < 0:
            raise ValueError("max_resolved_extra_tokens must be non-negative")
        if (
            self.direct_history_max_extra_tokens is not None
            and self.direct_history_max_extra_tokens < 0
        ):
            raise ValueError("direct_history_max_extra_tokens must be non-negative")
        if self.max_query_characters <= 0:
            raise ValueError("max_query_characters must be positive")
        if self.semantic_reference_strategy not in {
            "off",
            "raw_bm25",
            "anchor_bm25",
            "anchor_bm25_position",
        }:
            raise ValueError("unknown semantic reference strategy")


@dataclass(slots=True)
class _SelectedInput:
    query: str
    retrieval: RetrievalResult
    input_tokens: int
    raw_input_tokens: int
    variant_index: int


@dataclass(slots=True)
class _SessionLockEntry:
    lock: Any
    users: int = 0


class ToolPlanningAgent:
    """串起解析、Intent、canonical RAG、Planner 与原子状态提交。"""

    def __init__(
        self,
        store: SQLiteMemoryStore,
        intent_backend: IntentBackend,
        retriever: ToolRetriever,
        planner: PlannerBackend,
        catalog_by_id: dict[str, dict[str, Any]],
        policy: AgentPolicy,
        agent_config_hash: str,
        route_revision: str,
        route_hash: str,
        memory_reducer: DeterministicMemoryReducer | None = None,
    ):
        self.store = store
        self.intent_backend = intent_backend
        self.retriever = retriever
        self.planner = planner
        self.catalog_by_id = catalog_by_id
        self.policy = policy
        self.agent_config_hash = agent_config_hash
        self.route_revision = route_revision
        self.route_hash = route_hash
        self.router = ContextRouter(
            prefer_recent_over_generic_long=policy.recent_over_generic_long,
            natural_request_boundary_guard=policy.natural_request_boundary_guard,
            semantic_reference_strategy=policy.semantic_reference_strategy,
        )
        self.resolver = DeterministicResolver(
            natural_request_boundary_guard=policy.natural_request_boundary_guard,
            semantic_reference_strategy=policy.semantic_reference_strategy,
        )
        self.direct_history_resolver = DirectHistoryResolver(
            natural_request_boundary_guard=policy.natural_request_boundary_guard
        )
        self.state_reducer = ToolStateReducer(catalog_by_id)
        self.memory_reducer = memory_reducer or DeterministicMemoryReducer(
            catalog_by_id,
            recent_event_limit=policy.recent_event_limit,
        )
        self.long_memory = LexicalLongTermMemory(
            store, reference_strategy=policy.semantic_reference_strategy
        )
        self._session_locks: dict[str, _SessionLockEntry] = {}
        self._session_locks_guard = threading.Lock()
        self._memory_executor = ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="tool-memory",
        )

    @staticmethod
    def _normalize_catalog_alias(value: str) -> str:
        return re.sub(r"[\s，,。；;！!？?：:\-_/]+", "", value).lower()

    def _catalog_alias_tools(
        self,
        goals: list[Goal],
        candidate_tools: list[str],
    ) -> list[str] | None:
        """仅对“的 + 完整能力名”结尾的高精度目标做全 catalog 直达。"""

        if not goals:
            return None
        selected: list[str] = []
        for goal in goals:
            normalized_goal = self._normalize_catalog_alias(goal.text)
            structured_goal = normalized_goal.removeprefix("只保留")
            if not structured_goal.startswith(("核实", "确认")) or "这件" not in structured_goal[:18]:
                return None
            matches: list[tuple[int, int, str]] = []
            for position, tool in enumerate(candidate_tools):
                anchor = self._normalize_catalog_alias(
                    str(self.catalog_by_id[tool]["keyword_label"])
                )
                if len(anchor) >= 2 and structured_goal.endswith(f"的{anchor}"):
                    matches.append((len(anchor), position, tool))
            if not matches:
                return None
            longest = max(length for length, _, _ in matches)
            winners = [tool for length, _, tool in matches if length == longest]
            if len(winners) != 1 or winners[0] in selected:
                return None
            selected.append(winners[0])
        return selected

    def _catalog_alias_result(self, tools: list[str]) -> PlannerResult:
        raw_output = json.dumps(
            {"tools": tools},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return PlannerResult(
            tools=tools,
            strict_ok=True,
            strict_error="",
            raw_output=raw_output,
            input_tokens=0,
            output_tokens=0,
            latency_ms=0.0,
            prompt_version="catalog-alias-direct-v2",
            prompt_hash=hashlib.sha256(
                b"full-catalog-structured-terminal-possessive-keyword-label-optional-retain-v2"
            ).hexdigest(),
            model_revision=f"catalog:{self.route_hash}",
        )

    @staticmethod
    def _cached_plan_result(tools: list[str], source_event: PlanEvent) -> PlannerResult:
        return PlannerResult(
            tools=list(tools),
            strict_ok=True,
            strict_error="",
            raw_output=json.dumps(
                {"tools": tools}, ensure_ascii=False, separators=(",", ":")
            ),
            input_tokens=0,
            output_tokens=0,
            latency_ms=0.0,
            prompt_version="whole-turn-plan-replay-v1",
            prompt_hash=hashlib.sha256(
                b"validated-same-runtime-whole-turn-plan-replay-v1"
            ).hexdigest(),
            model_revision=source_event.model_revision,
        )

    @contextmanager
    def _locked_session(self, conversation_id: str) -> Iterator[None]:
        with self._session_locks_guard:
            entry = self._session_locks.get(conversation_id)
            if entry is None:
                entry = _SessionLockEntry(lock=threading.RLock())
                self._session_locks[conversation_id] = entry
            entry.users += 1
        entry.lock.acquire()
        try:
            yield
        finally:
            entry.lock.release()
            with self._session_locks_guard:
                entry.users -= 1
                if entry.users == 0 and self._session_locks.get(conversation_id) is entry:
                    del self._session_locks[conversation_id]

    def _retrieve_memory(
        self,
        conversation_id: str,
        query: str,
    ) -> tuple[list[Any], list[str], float]:
        warnings: list[str] = []
        started = time.perf_counter()
        future = self._memory_executor.submit(
            self.long_memory.search,
            conversation_id,
            query,
            self.policy.long_memory_top_k,
            explicit_turn_references(query),
        )
        try:
            hits = future.result(timeout=self.policy.long_memory_timeout_ms / 1000.0)
        except FutureTimeoutError:
            future.cancel()
            hits = []
            warnings.append("long_memory_timeout_fallback")
        except Exception as exc:  # 降级必须可观察，但不能中断 Planner。
            hits = []
            warnings.append(f"long_memory_error_fallback:{type(exc).__name__}")
        return hits, warnings, (time.perf_counter() - started) * 1000.0

    def _summary_hits(self, summary: ToolRollingSummary, query: str) -> list[MemoryHit]:
        by_turn: dict[int, list[Goal]] = {}
        for goal in summary.historical_goals:
            for source_turn in goal.source_turn_ids:
                by_turn.setdefault(source_turn, []).append(
                    Goal(
                        text=goal.text,
                        source_turn_ids=[source_turn],
                        origin="retrieved",
                        order=goal.order,
                    )
                )
        if not by_turn:
            return []
        explicit = explicit_turn_references(query)
        selected_turns: list[int]
        if explicit:
            selected_turns = [turn for turn in explicit if turn in by_turn]
        elif any(signal in query for signal in ("最开始", "一开始", "最早")):
            selected_turns = [min(by_turn)]
        else:
            turns = sorted(by_turn)
            texts = ["；".join(goal.text for goal in by_turn[turn]) for turn in turns]
            scores = bm25_scores(query, texts)
            selected_turns = [
                turn
                for turn, score in sorted(
                    zip(turns, scores), key=lambda item: (-item[1], -item[0])
                )
                if score > 0.0
            ]
        hits = []
        for turn in selected_turns[: self.policy.long_memory_top_k]:
            goals = by_turn[turn]
            hits.append(
                MemoryHit(
                    record=MemoryRecord(
                        conversation_id=summary.conversation_id,
                        source_turn_id=turn,
                        text="；".join(goal.text for goal in goals),
                        goals=goals,
                        planned_tools=[],
                        domains=[],
                        created_at=summary.updated_at,
                    ),
                    score=1.0,
                    match_reason="rolling_summary",
                )
            )
        return hits

    def _select_input(
        self,
        raw_query: str,
        variants: list[str],
        warnings: list[str],
        trusted_extra_variants: set[int] | None = None,
    ) -> _SelectedInput | None:
        trusted_extra_variants = trusted_extra_variants or set()
        hard_fallback: _SelectedInput | None = None
        for variant_index, candidate_query in enumerate(variants):
            retrieval = self.retriever.retrieve(candidate_query)
            cached_tokens = getattr(self.planner, "cached_input_tokens", None)
            if callable(cached_tokens):
                raw_input_tokens = cached_tokens(
                    raw_query,
                    retrieval.candidate_tools,
                    allow_missing=True,
                )
                input_tokens = cached_tokens(
                    candidate_query,
                    retrieval.candidate_tools,
                )
            else:
                raw_input_tokens = self.planner.count_input_tokens(
                    raw_query, retrieval.candidate_tools
                )
                input_tokens = self.planner.count_input_tokens(
                    candidate_query, retrieval.candidate_tools
                )
            within_hard = input_tokens <= self.policy.hard_input_tokens
            allowed_extra_tokens = self.policy.max_resolved_extra_tokens
            if (
                self.policy.context_architecture == "B"
                and variant_index == 0
                and self.policy.direct_history_max_extra_tokens is not None
            ):
                allowed_extra_tokens = self.policy.direct_history_max_extra_tokens
            exceeds_extra = input_tokens - raw_input_tokens > allowed_extra_tokens
            within_extra = variant_index in trusted_extra_variants or not exceeds_extra
            if within_hard and within_extra and input_tokens <= self.policy.target_input_tokens:
                if variant_index in trusted_extra_variants and exceeds_extra:
                    warnings.append("trusted_replay_extra_budget_exempt")
                if variant_index:
                    warnings.append(f"resolved_query_budget_fallback_variant:{variant_index}")
                return _SelectedInput(
                    query=candidate_query,
                    retrieval=retrieval,
                    input_tokens=input_tokens,
                    raw_input_tokens=raw_input_tokens,
                    variant_index=variant_index,
                )
            if within_hard and within_extra and hard_fallback is None:
                hard_fallback = _SelectedInput(
                    query=candidate_query,
                    retrieval=retrieval,
                    input_tokens=input_tokens,
                    raw_input_tokens=raw_input_tokens,
                    variant_index=variant_index,
                )
        if hard_fallback is not None:
            warnings.append("planner_input_above_target")
            if hard_fallback.variant_index in trusted_extra_variants:
                warnings.append("trusted_replay_extra_budget_exempt")
            if hard_fallback.variant_index:
                warnings.append(
                    f"resolved_query_budget_fallback_variant:{hard_fallback.variant_index}"
                )
            return hard_fallback
        warnings.append("planner_input_hard_budget_rejected")
        return None

    def run_turn(self, conversation_id: str, query: str) -> AgentTurnResult:
        conversation_id = conversation_id.strip()
        query = query.strip()
        if not conversation_id:
            raise ValueError("conversation id must not be empty")
        if len(conversation_id) > 256:
            raise ValueError("conversation id is too long")
        if not query:
            raise ValueError("query must not be empty")
        if len(query) > self.policy.max_query_characters:
            raise ValueError("query exceeds the configured character limit")

        with self._locked_session(conversation_id):
            total_started = time.perf_counter()
            snapshot = self.store.snapshot(
                conversation_id,
                recent_limit=self.policy.recent_event_limit + 1,
            )
            turn_id = snapshot.state.last_turn_id + 1
            mode = "none" if self.policy.context_architecture == "A" else self.router.route(query)
            warnings: list[str] = []
            memory_hits = []
            memory_ms = 0.0
            if mode == "long" and self.policy.context_architecture == "E":
                memory_hits, memory_warnings, memory_ms = self._retrieve_memory(
                    conversation_id, query
                )
                warnings.extend(memory_warnings)
            elif mode == "long" and self.policy.context_architecture == "D":
                memory_started = time.perf_counter()
                memory_hits = self._summary_hits(snapshot.summary, query)
                memory_ms = (time.perf_counter() - memory_started) * 1000.0

            resolution_started = time.perf_counter()
            recent_events = snapshot.recent_events[-self.policy.recent_event_limit :]
            if self.policy.context_architecture == "B":
                force_direct_history = (
                    self.policy.direct_history_max_extra_tokens is not None
                    and bool(recent_events)
                )
                resolved = self.direct_history_resolver.resolve(
                    query=query,
                    turn_id=turn_id,
                    mode="recent" if force_direct_history and mode == "none" else mode,
                    recent_events=recent_events,
                )
                if force_direct_history and resolved.query_variants:
                    resolved.query_variants = resolved.query_variants[:1]
                    warnings.append("direct_history_concat_forced")
            else:
                resolved = self.resolver.resolve(
                    query=query,
                    turn_id=turn_id,
                    mode=mode,
                    state=snapshot.state,
                    recent_events=recent_events,
                    memory_hits=memory_hits,
                )
            if (
                self.policy.standalone_multigoal_guard
                and mode != "none"
                and not explicit_turn_references(query)
                and len(resolved.interpretation.current_goals) >= 2
                and not resolved.interpretation.carried_goals
                and not resolved.interpretation.cancelled_goals
            ):
                mode = "none"
                resolved = self.resolver.resolve(
                    query=query,
                    turn_id=turn_id,
                    mode="none",
                    state=snapshot.state,
                    recent_events=[],
                    memory_hits=[],
                )
                warnings.append("standalone_multigoal_context_guard")
            trusted_extra_variants: set[int] = set()
            cached_replay_event: PlanEvent | None = None
            if (
                (self.policy.whole_turn_reference_replay or self.policy.whole_turn_plan_cache)
                and len(resolved.interpretation.reference_turn_ids) == 1
                and not resolved.interpretation.current_goals
                and not resolved.interpretation.cancelled_goals
                and resolved.interpretation.carried_goals
            ):
                reference_turn = resolved.interpretation.reference_turn_ids[0]
                replay_event = next(
                    (
                        event
                        for event in recent_events
                        if event.turn_id == reference_turn
                    ),
                    None,
                )
                if replay_event is None:
                    replay_event = self.store.event(conversation_id, reference_turn)
                replay_query = (
                    replay_event.resolved_query
                    if replay_event is not None and replay_event.status == "ok"
                    else ""
                )
                if not replay_query:
                    replay_record = next(
                        (
                            hit.record
                            for hit in memory_hits
                            if hit.record.source_turn_id == reference_turn
                        ),
                        None,
                    )
                    replay_query = (
                        replay_record.text
                        if replay_record is not None and replay_record.planned_tools
                        else ""
                    )
                if replay_query:
                    resolved.query_variants = list(
                        dict.fromkeys([replay_query, *resolved.query_variants])
                    )
                    trusted_extra_variants.add(0)
                    warnings.append("whole_turn_reference_replayed")
                if (
                    self.policy.whole_turn_plan_cache
                    and replay_event is not None
                    and replay_event.status == "ok"
                    and replay_event.strict_output
                    and replay_event.planned_tools
                    and replay_event.route_hash == self.route_hash
                    and replay_event.retrieval_revision == self.route_revision
                    and replay_event.agent_config_hash == self.agent_config_hash
                ):
                    cached_replay_event = replay_event
            first_turn_standalone = (
                turn_id == 1
                and mode != "none"
                and is_first_turn_standalone(
                    query, resolved.interpretation.current_goals
                )
            )
            if first_turn_standalone:
                mode = "none"
                if self.policy.context_architecture == "B":
                    resolved = self.direct_history_resolver.resolve(
                        query=query,
                        turn_id=turn_id,
                        mode=mode,
                        recent_events=[],
                    )
                else:
                    resolved = self.resolver.resolve(
                        query=query,
                        turn_id=turn_id,
                        mode=mode,
                        state=snapshot.state,
                        recent_events=[],
                        memory_hits=[],
                    )
                warnings.append("first_turn_context_signal_treated_as_standalone")
            warnings.extend(resolved.interpretation.notes)
            resolution_ms = (time.perf_counter() - resolution_started) * 1000.0

            intent = IntentResult(intent_type="")
            retrieval: RetrievalResult | None = None
            planner = PlannerResult(
                tools=[],
                strict_ok=False,
                strict_error="not_called",
                raw_output="",
                input_tokens=0,
                output_tokens=0,
                latency_ms=0.0,
                prompt_version="",
                prompt_hash="",
                model_revision="",
            )
            status = "backend_error"
            final_query = resolved.query_variants[0]
            input_tokens = 0
            retrieval_ms = 0.0
            intent_ms = 0.0
            unresolved_reference_guard = resolved.interpretation.unresolved_reference
            cached_replay_tools = (
                list(cached_replay_event.planned_tools)
                if cached_replay_event is not None
                else None
            )
            direct_alias_tools = (
                self._catalog_alias_tools(
                    resolved.interpretation.resolved_goals,
                    sorted(self.catalog_by_id),
                )
                if self.policy.catalog_alias_guard and not unresolved_reference_guard
                else None
            )

            try:
                intent = (
                    IntentResult(intent_type="电商类", raw_output="whole_turn_plan_cache")
                    if cached_replay_tools is not None
                    else (
                        IntentResult(intent_type="电商类", raw_output="catalog_alias_direct")
                        if direct_alias_tools is not None
                        else (
                            IntentResult(intent_type="闲聊类", raw_output="context_guard")
                            if unresolved_reference_guard
                            else self.intent_backend.classify(final_query)
                        )
                    )
                )
                intent_ms = intent.latency_ms
                if cached_replay_tools is not None:
                    assert cached_replay_event is not None
                    planner = self._cached_plan_result(
                        cached_replay_tools, cached_replay_event
                    )
                    warnings.append("whole_turn_plan_cache_applied")
                    status = "ok"
                elif direct_alias_tools is not None:
                    planner = self._catalog_alias_result(direct_alias_tools)
                    warnings.append("catalog_alias_direct_applied")
                    status = "ok"
                elif intent.intent_type == "闲聊类":
                    status = "chitchat"
                    planner = PlannerResult(
                        tools=[],
                        strict_ok=True,
                        strict_error="",
                        raw_output='{"tools":[]}',
                        input_tokens=0,
                        output_tokens=0,
                        latency_ms=0.0,
                        prompt_version="intent_gate",
                        prompt_hash="",
                        model_revision="",
                    )
                else:
                    selection_started = time.perf_counter()
                    selected = self._select_input(
                        query,
                        resolved.query_variants,
                        warnings,
                        trusted_extra_variants=trusted_extra_variants,
                    )
                    retrieval_ms = (time.perf_counter() - selection_started) * 1000.0
                    if selected is None:
                        status = "budget_rejected"
                    else:
                        final_query = selected.query
                        retrieval = selected.retrieval
                        input_tokens = selected.input_tokens
                        if selected.variant_index:
                            intent = self.intent_backend.classify(final_query)
                            intent_ms += intent.latency_ms
                        if intent.intent_type == "闲聊类":
                            status = "chitchat"
                            planner = PlannerResult(
                                tools=[],
                                strict_ok=True,
                                strict_error="",
                                raw_output='{"tools":[]}',
                                input_tokens=0,
                                output_tokens=0,
                                latency_ms=0.0,
                                prompt_version="intent_gate",
                                prompt_hash="",
                                model_revision="",
                            )
                        else:
                            planner = self.planner.plan(
                                final_query, retrieval.candidate_tools
                            )
                            if planner.input_tokens != input_tokens:
                                raise RuntimeError(
                                    "Planner token count changed between budget check and inference"
                                )
                            if planner.input_tokens > self.policy.hard_input_tokens:
                                raise RuntimeError("Planner exceeded the hard input budget")
                            if planner.strict_ok:
                                status = "ok"
                            else:
                                status = "planner_invalid"
                                warnings.append(f"planner_strict_error:{planner.strict_error}")
                if unresolved_reference_guard:
                    planner.prompt_version = "unresolved_reference_guard"
            except Exception as exc:
                status = "backend_error"
                warnings.append(f"backend_error:{type(exc).__name__}:{exc}")

            candidate_tools = (
                list(planner.tools)
                if planner.prompt_version
                in {"catalog-alias-direct-v2", "whole-turn-plan-replay-v1"}
                else (retrieval.candidate_tools if retrieval else [])
            )
            accepted_output = status in {"ok", "chitchat"} and planner.strict_ok
            planned_tools = planner.tools if status == "ok" and accepted_output else []
            if (
                self.policy.single_tool_goal_normalization
                and mode == "none"
                and len(planned_tools) == 1
                and planner.prompt_version != "catalog-alias-direct-v2"
                and not resolved.interpretation.carried_goals
                and not resolved.interpretation.cancelled_goals
                and not resolved.interpretation.reference_turn_ids
            ):
                normalized_goal = Goal(
                    text=query,
                    source_turn_ids=[turn_id],
                    origin="current",
                    order=0,
                )
                before = [goal.key for goal in resolved.interpretation.resolved_goals]
                if before != [normalized_goal.key]:
                    warnings.append("single_tool_goal_normalized")
                resolved.interpretation.current_goals = [normalized_goal]
                resolved.interpretation.resolved_goals = [normalized_goal]
            strict_output = accepted_output
            event_status = "unresolved_reference" if unresolved_reference_guard else status
            event_intent_type = "" if unresolved_reference_guard else intent.intent_type
            event = PlanEvent(
                conversation_id=conversation_id,
                turn_id=turn_id,
                created_at=utc_now(),
                raw_query=query,
                resolved_query=final_query,
                context_mode=mode,
                current_goals=resolved.interpretation.current_goals,
                carried_goals=resolved.interpretation.carried_goals,
                cancelled_goals=resolved.interpretation.cancelled_goals,
                resolved_goals=resolved.interpretation.resolved_goals,
                reference_turn_ids=resolved.interpretation.reference_turn_ids,
                intent_type=event_intent_type,
                candidate_tools=candidate_tools,
                planned_tools=planned_tools,
                strict_output=strict_output,
                status=event_status,
                raw_planner_output=planner.raw_output,
                prompt_version=planner.prompt_version,
                prompt_hash=planner.prompt_hash,
                model_revision=planner.model_revision,
                retrieval_revision=(
                    retrieval.revision if retrieval and retrieval.revision else self.route_revision
                ),
                route_hash=(
                    retrieval.route_hash if retrieval and retrieval.route_hash else self.route_hash
                ),
                agent_config_hash=self.agent_config_hash,
                input_tokens=(
                    0
                    if planner.prompt_version
                    in {"catalog-alias-direct-v2", "whole-turn-plan-replay-v1"}
                    else (planner.input_tokens if planner.input_tokens else input_tokens)
                ),
                output_tokens=planner.output_tokens,
                latency_ms={
                    "memory": memory_ms,
                    "resolution": resolution_ms,
                    "intent": intent_ms,
                    "retrieval_and_budget": retrieval_ms,
                    "planner": planner.latency_ms,
                    "total_before_commit": (time.perf_counter() - total_started) * 1000.0,
                },
                warnings=list(dict.fromkeys(warnings)),
            )
            new_state = self.state_reducer.reduce(snapshot.state, event)
            summary, memories = self.memory_reducer.reduce(
                snapshot.summary,
                snapshot.recent_events + [event],
                newest_turn_id=turn_id,
            )
            self.store.commit_turn(
                event=event,
                state=new_state,
                summary=summary,
                new_memories=memories,
                expected_version=snapshot.state.version,
            )
            return AgentTurnResult(
                conversation_id=conversation_id,
                turn_id=turn_id,
                status=event.status,
                tools=planned_tools,
                resolved_query=final_query,
                context_mode=mode,
                strict_output=strict_output,
                warnings=event.warnings,
                event=event,
            )

    def close(self) -> None:
        self._memory_executor.shutdown(wait=True, cancel_futures=True)
        self.store.close()
