"""上下文触发、目标解析和确定性 query 改写。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .types import (
    CompactToolState,
    Goal,
    MemoryHit,
    PlanEvent,
    TurnInterpretation,
    normalize_goal_text,
    unique_goals,
    utc_now,
)
from .reference_selection import is_whole_turn_reference


RECENT_SIGNAL = re.compile(
    r"刚才|上一轮|上一步|继续|接着来|还是|照旧|同样|一样|那个|前面|改成|换成"
)
LONG_SIGNAL = re.compile(
    r"之前|上次|早些时候|最开始|很久前|"
    r"历史(?:上|里|记录中|任务中)|第\s*[^\s，。；]+\s*轮"
)
STRONG_LONG_SIGNAL = re.compile(
    r"上次|早些时候|最开始|很久前|"
    r"历史(?:上|里|记录中|任务中)|第\s*[^\s，。；]+\s*轮"
)
NEGATIVE_BIE_SIGNAL = (
    r"别(?:再)?(?=(?:查|看|做|处理|执行|继续|买|卖|用|要|帮|给|选|加|删|改|退|换|"
    r"取消|保留|调用|查询|查看|安排|提交|申请|创建|核实|确认))"
)
CANCEL_SIGNAL = re.compile(
    rf"不用|(?<!要)不要|不再|取消|{NEGATIVE_BIE_SIGNAL}|无需|去掉|删掉|撤销"
)
LEGACY_CANCEL_SIGNAL = re.compile(
    r"不用|不要|不再|取消|(?<!分)别(?:再)?|无需|去掉|删掉|撤销"
)
CANCEL_ALL_SIGNAL = re.compile(r"都不用|全部取消|全都不要|都取消|全部去掉")
REFERENCE_WORDS = re.compile(
    r"刚才|上一轮|上一步|继续|接着做|接着|还是|照旧|同样|一样|那个|前面|之前|上次|"
    r"早些时候|最开始|很久前|历史(?:上|里)?|第\s*[^\s，。；]+\s*轮|"
    r"这(?:个|些)|那(?:个|些)|操作|方案|任务|诉求|步骤|两个|三个|全部|"
    r"回到|恢复|沿用|接着做"
)
CLAUSE_SEPARATOR = re.compile(
    r"[，,。；;！!？?\n]+|(?<=.)(?:然后|接着|随后|之后|并且|同时|另外|以及|再帮我|再给我)"
)
POLITE_PREFIX = re.compile(r"^(?:请|麻烦|劳驾|帮我|给我|替我|再|然后|接着|顺便)+")
ORDER_PREFIX = re.compile(r"^(?:先|再|最后|然后|接着|并)")
TURN_REFERENCE = re.compile(r"第\s*(\d+|[一二三四五六七八九十]+)\s*轮")
CORRECTION_PREFIX = re.compile(
    r"^(?:(?:刚才|上一轮|上一步|前面)(?:的)?)?(?:改成|换成)"
)
LEADING_CONTEXT = re.compile(
    r"^(?:(?:继续|接着|还是|照旧|同样|一样|沿用|回到|恢复|"
    r"刚才|上一轮|上一步|前面|之前|上次|早些时候|最开始|很久前)"
    r"(?:的)?)+"
)
META_CANCEL_SIGNAL = re.compile(
    rf"不用|(?<!要)不要|不再|{NEGATIVE_BIE_SIGNAL}|无需|去掉|删掉|撤销|"
    r"(?:全部|全都|都)取消|"
    r"(?:^|请|麻烦|帮我|我要|我想|给我|替我|然后|再)(?:把|将).+取消(?:掉|了|吧)?|"
    r"取消(?:掉|了|吧)$"
)
LEGACY_META_CANCEL_SIGNAL = re.compile(
    r"不用|不要|不再|(?<!分)别(?:再)?|无需|去掉|删掉|撤销|"
    r"(?:全部|全都|都)取消|"
    r"(?:把|将).+取消(?:掉|了|吧)?|"
    r"取消(?:掉|了|吧)$"
)
STANDALONE_MUTATION_SIGNAL = re.compile(r"删掉|删除|移除|清空")
BUSINESS_CANCELLATION_TARGET = re.compile(
    r"(?:取消|撤销|不想退).{0,24}(?:订单|这单|那单|退款|退货|换货|售后单|申请|预约)|"
    r"(?:订单|这单|那单|退款申请|退货申请|换货申请|售后申请|预约|退款).{0,40}"
    r"(?:取消|撤销|不想退)"
)
BUSINESS_CART_DELETION_TARGET = re.compile(
    r"(?:购物车|加购清单).{0,32}(?:删掉|删除|移除|清理)|"
    r"(?:删掉|删除|移除|清理).{0,16}(?:购物车|加购清单)"
)
META_CANCELLATION_ARTIFACT = re.compile(
    r"任务|诉求|操作|步骤|方案|(?:查询|查看|核实|确认)(?:请求|任务|操作)"
)
META_CANCEL_ACTION = re.compile(
    rf"(?:不用|(?<!要)不要|不再|无需|{NEGATIVE_BIE_SIGNAL}).{{0,8}}"
    r"(?:查|看|做|处理|执行|调用|查询|查看|核实|确认)|"
    r"(?:取消|撤销|删掉|删除|移除|去掉).{0,8}"
    r"(?:查询|查看|核实|确认|任务|诉求|操作|步骤|方案)"
)
META_CANCELLED_ARTIFACT = re.compile(
    r"(?:查询|查看|核实|确认|任务|诉求|操作|步骤|方案).{0,12}"
    r"(?:取消|撤销|删掉|删除|移除|去掉)"
)

CHINESE_NUMERALS = {
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}

OBJECT_TYPES = (
    "订单",
    "商品",
    "购物车",
    "优惠券",
    "物流",
    "退款",
    "售后",
    "发票",
    "会员",
    "支付",
    "地址",
    "店铺",
    "客服",
)
EARLIEST_REFERENCE_SIGNALS = ("最开始", "一开始", "最早")


def parse_chinese_number(value: str) -> int | None:
    if value.isdigit():
        return int(value)
    if value in CHINESE_NUMERALS:
        return CHINESE_NUMERALS[value]
    if value.startswith("十") and len(value) == 2 and value[1] in CHINESE_NUMERALS:
        return 10 + CHINESE_NUMERALS[value[1]]
    if value.endswith("十") and len(value) == 2 and value[0] in CHINESE_NUMERALS:
        return CHINESE_NUMERALS[value[0]] * 10
    if len(value) == 3 and value[1] == "十":
        left = CHINESE_NUMERALS.get(value[0])
        right = CHINESE_NUMERALS.get(value[2])
        if left and right:
            return left * 10 + right
    return None


def explicit_turn_references(query: str) -> list[int]:
    result = []
    for match in TURN_REFERENCE.finditer(query):
        value = parse_chinese_number(match.group(1))
        if value and value > 0:
            result.append(value)
    return sorted(set(result))


def is_cancellation_clause(
    clause: str, *, natural_request_boundary_guard: bool = True
) -> bool:
    """区分“取消上一个诉求”与“取消订单/取消退款”工具目标。"""

    if not natural_request_boundary_guard:
        if LEGACY_META_CANCEL_SIGNAL.search(clause):
            return True
        if "取消" not in clause:
            return False
        if CORRECTION_PREFIX.search(clause):
            return False
        return bool(
            TURN_REFERENCE.search(clause)
            or RECENT_SIGNAL.search(clause)
            or LONG_SIGNAL.search(clause)
            or re.search(r"第[一二三四五六七八九十\d]+个|最后一个|前两个", clause)
        )
    if (
        (
            BUSINESS_CANCELLATION_TARGET.search(clause)
            or BUSINESS_CART_DELETION_TARGET.search(clause)
        )
        and not META_CANCELLED_ARTIFACT.search(clause)
    ):
        return False
    if not CANCEL_SIGNAL.search(clause) and not STANDALONE_MUTATION_SIGNAL.search(clause):
        return False
    if CORRECTION_PREFIX.search(clause):
        return False
    if CANCEL_ALL_SIGNAL.search(clause):
        return True
    if re.search(r"不用了(?:吧)?$", clause) and "要不要" not in clause:
        return True
    has_meta_target = bool(
        META_CANCELLATION_ARTIFACT.search(clause)
        or META_CANCEL_ACTION.search(clause)
    )
    has_history_target = bool(
        TURN_REFERENCE.search(clause)
        or RECENT_SIGNAL.search(clause)
        or LONG_SIGNAL.search(clause)
        or re.search(r"第[一二三四五六七八九十\d]+个|最后一个|前两个", clause)
    )
    return has_meta_target or has_history_target


def is_first_turn_standalone(query: str, current_goals: list[Goal]) -> bool:
    """无会话历史时，业务时间词或具体删除动作不应触发多轮继承。"""

    return bool(current_goals) or bool(STANDALONE_MUTATION_SIGNAL.search(query))


class ContextRouter:
    """只根据当前 query 决定是否允许读取历史。"""

    def __init__(
        self,
        prefer_recent_over_generic_long: bool = False,
        natural_request_boundary_guard: bool = True,
        semantic_reference_strategy: str = "off",
    ):
        self.prefer_recent_over_generic_long = prefer_recent_over_generic_long
        self.natural_request_boundary_guard = natural_request_boundary_guard
        self.semantic_reference_strategy = semantic_reference_strategy

    def route(self, query: str) -> str:
        if not query.strip():
            raise ValueError("query must not be empty")
        if explicit_turn_references(query):
            return "long"
        if (
            self.semantic_reference_strategy != "off"
            and is_whole_turn_reference(query)
        ):
            return "long"
        has_recent = bool(RECENT_SIGNAL.search(query)) or any(
            is_cancellation_clause(
                clause,
                natural_request_boundary_guard=self.natural_request_boundary_guard,
            )
            for clause in split_clauses(query)
        )
        if self.prefer_recent_over_generic_long:
            if STRONG_LONG_SIGNAL.search(query):
                return "long"
            if has_recent:
                return "recent"
            if LONG_SIGNAL.search(query):
                return "long"
            return "none"
        if LONG_SIGNAL.search(query):
            return "long"
        if has_recent:
            return "recent"
        return "none"


def split_clauses(query: str) -> list[str]:
    clauses = []
    for value in CLAUSE_SEPARATOR.split(query):
        value = value.strip(" \t：:。.!！?？；;，,")
        if value:
            clauses.append(value)
    return clauses or [query.strip()]


def clean_goal_clause(clause: str) -> str:
    raw_value = clause.strip()
    value = POLITE_PREFIX.sub("", raw_value)
    reference_probes = []
    for probe in (raw_value, value):
        probe = TURN_REFERENCE.sub("", probe)
        probe = REFERENCE_WORDS.sub("", probe)
        probe = CORRECTION_PREFIX.sub("", probe)
        probe = probe.strip(" 的地得：:。.!！?？；;，,")
        probe = re.sub(r"^(?:和|与|及|以及)+$", "", probe)
        reference_probes.append(probe)
    if any(not normalize_goal_text(probe) for probe in reference_probes):
        return ""
    value = CORRECTION_PREFIX.sub("", value).strip()
    value = LEADING_CONTEXT.sub("", value).strip()
    value = TURN_REFERENCE.sub("", value).strip()
    value = re.sub(r"^(?:那个|这个)(?:的)?", "", value).strip()
    value = ORDER_PREFIX.sub("", value).strip()
    return value.strip(" 的地得：:。.!！?？；;，,")


def cancellation_subject(
    clause: str, *, natural_request_boundary_guard: bool = True
) -> str:
    signal = CANCEL_SIGNAL if natural_request_boundary_guard else LEGACY_CANCEL_SIGNAL
    value = signal.sub("", clause)
    value = re.sub(r"(?:了|啦|吧|掉)$", "", value)
    value = re.sub(r"^(?:查询|查一下|查看|看看|看|执行|做)", "", value)
    value = re.sub(r"(?:查询|查看|执行|操作)$", "", value)
    value = REFERENCE_WORDS.sub("", value)
    return value.strip(" 的地得：:。.!！?？；;，,")


def ngrams(value: str, n: int = 2) -> set[str]:
    normalized = normalize_goal_text(value)
    if len(normalized) <= n:
        return {normalized} if normalized else set()
    return {normalized[index : index + n] for index in range(len(normalized) - n + 1)}


def goal_similarity(left: str, right: str) -> float:
    a = normalize_goal_text(left)
    b = normalize_goal_text(right)
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return min(len(a), len(b)) / max(len(a), len(b)) + 0.35
    a_grams = ngrams(a)
    b_grams = ngrams(b)
    union = a_grams | b_grams
    jaccard = len(a_grams & b_grams) / len(union) if union else 0.0
    longest = 0
    previous = [0] * (len(b) + 1)
    for left_character in a:
        current = [0] * (len(b) + 1)
        for index, right_character in enumerate(b, 1):
            if left_character == right_character:
                current[index] = previous[index - 1] + 1
                longest = max(longest, current[index])
        previous = current
    substring = longest / max(len(a), len(b))
    if longest >= 4:
        substring += 0.15
    return max(jaccard, substring)


def fuzzy_unique(goals: list[Goal], threshold: float = 0.72) -> list[Goal]:
    result: list[Goal] = []
    for goal in unique_goals(goals):
        duplicate = next(
            (existing for existing in result if goal_similarity(goal.text, existing.text) >= threshold),
            None,
        )
        if duplicate is None:
            result.append(goal)
        else:
            duplicate.source_turn_ids = sorted(
                set(duplicate.source_turn_ids) | set(goal.source_turn_ids)
            )
    for index, goal in enumerate(result):
        goal.order = index
    return result


def clone_goal(goal: Goal, origin: str) -> Goal:
    return Goal(
        text=goal.text,
        source_turn_ids=list(goal.source_turn_ids),
        origin=origin,
        order=goal.order,
    )


def ordinal_targets(clause: str, goals: list[Goal]) -> list[Goal]:
    if not goals:
        return []
    if "第一个" in clause or "第一项" in clause:
        return goals[:1]
    if "第二个" in clause or "第二项" in clause:
        return goals[1:2]
    if "第三个" in clause or "第三项" in clause:
        return goals[2:3]
    if "最后一个" in clause or "最后一项" in clause:
        return goals[-1:]
    if "前两个" in clause or "前两项" in clause:
        return goals[:2]
    return []


def select_cancelled(
    clause: str,
    reference_goals: list[Goal],
    *,
    natural_request_boundary_guard: bool = True,
) -> list[Goal]:
    if CANCEL_ALL_SIGNAL.search(clause):
        return reference_goals
    ordinal = ordinal_targets(clause, reference_goals)
    if ordinal:
        return ordinal
    subject = cancellation_subject(
        clause,
        natural_request_boundary_guard=natural_request_boundary_guard,
    )
    if not subject:
        return []
    scored = sorted(
        ((goal_similarity(subject, goal.text), goal) for goal in reference_goals),
        key=lambda item: (-item[0], item[1].order),
    )
    if not scored or scored[0][0] < 0.18:
        return []
    best = scored[0][0]
    return [goal for score, goal in scored if score >= max(0.18, best - 0.08)]


@dataclass(slots=True)
class ResolvedTurn:
    interpretation: TurnInterpretation
    query_variants: list[str]


class DirectHistoryResolver:
    """对照组 B：不解析状态，只把近期 user query 直接拼到当前请求前。"""

    def __init__(self, natural_request_boundary_guard: bool = True):
        self.natural_request_boundary_guard = natural_request_boundary_guard

    def resolve(
        self,
        query: str,
        turn_id: int,
        mode: str,
        recent_events: list[PlanEvent],
    ) -> ResolvedTurn:
        current = []
        for clause in split_clauses(query):
            if is_cancellation_clause(
                clause,
                natural_request_boundary_guard=self.natural_request_boundary_guard,
            ):
                continue
            cleaned = clean_goal_clause(clause)
            if cleaned and normalize_goal_text(cleaned):
                current.append(
                    Goal(
                        text=cleaned,
                        source_turn_ids=[turn_id],
                        origin="current",
                        order=len(current),
                    )
                )
        current = fuzzy_unique(current)
        if mode == "none" or not recent_events:
            return ResolvedTurn(
                interpretation=TurnInterpretation(
                    context_mode=mode,
                    current_goals=current,
                    resolved_goals=current,
                ),
                query_variants=[query.strip()],
            )
        history = "\n".join(
            f"第{event.turn_id}轮用户请求：{event.raw_query}" for event in recent_events
        )
        concatenated = f"历史用户请求：\n{history}\n当前用户请求：{query.strip()}"
        return ResolvedTurn(
            interpretation=TurnInterpretation(
                context_mode=mode,
                current_goals=current,
                resolved_goals=current,
                reference_turn_ids=[event.turn_id for event in recent_events],
            ),
            query_variants=[concatenated, query.strip()],
        )


class DeterministicResolver:
    """不调用第二个 LLM，只合并用户可追溯的目标文本。"""

    def __init__(
        self,
        natural_request_boundary_guard: bool = True,
        semantic_reference_strategy: str = "off",
    ):
        self.natural_request_boundary_guard = natural_request_boundary_guard
        self.semantic_reference_strategy = semantic_reference_strategy

    def _reference_goals(
        self,
        query: str,
        mode: str,
        state: CompactToolState,
        recent_events: list[PlanEvent],
        memory_hits: list[MemoryHit],
    ) -> tuple[list[Goal], list[int]]:
        explicit = explicit_turn_references(query)
        if explicit:
            by_turn = {event.turn_id: event for event in recent_events}
            goals: list[Goal] = []
            turns: list[int] = []
            hit_by_turn = {hit.record.source_turn_id: hit.record for hit in memory_hits}
            for turn_id in explicit:
                if turn_id in by_turn:
                    goals.extend(clone_goal(goal, "retrieved") for goal in by_turn[turn_id].resolved_goals)
                    turns.append(turn_id)
                elif turn_id in hit_by_turn:
                    goals.extend(clone_goal(goal, "retrieved") for goal in hit_by_turn[turn_id].goals)
                    turns.append(turn_id)
            return fuzzy_unique(goals), sorted(set(turns))

        if (
            self.semantic_reference_strategy == "off"
            and mode == "long"
            and any(
            signal in query for signal in EARLIEST_REFERENCE_SIGNALS
            )
        ):
            candidates: list[tuple[int, list[Goal]]] = [
                (event.turn_id, event.resolved_goals)
                for event in recent_events
                if event.resolved_goals
            ]
            candidates.extend(
                (hit.record.source_turn_id, hit.record.goals)
                for hit in memory_hits
                if hit.record.goals
            )
            if candidates:
                source_turn, source_goals = min(candidates, key=lambda item: item[0])
                return (
                    [clone_goal(goal, "retrieved") for goal in source_goals],
                    [source_turn],
                )

        if mode == "recent":
            if not state.last_request_goals:
                return [], []
            return (
                [clone_goal(goal, "carried") for goal in state.last_request_goals],
                [state.last_turn_id],
            )

        if mode == "long" and memory_hits:
            hit = memory_hits[0]
            return (
                [clone_goal(goal, "retrieved") for goal in hit.record.goals],
                [hit.record.source_turn_id],
            )
        if mode == "long" and state.last_request_goals:
            return (
                [clone_goal(goal, "carried") for goal in state.last_request_goals],
                [state.last_turn_id],
            )
        return [], []

    def resolve(
        self,
        query: str,
        turn_id: int,
        mode: str,
        state: CompactToolState,
        recent_events: list[PlanEvent],
        memory_hits: list[MemoryHit],
    ) -> ResolvedTurn:
        if mode not in {"none", "recent", "long"}:
            raise ValueError(f"unknown context mode: {mode}")
        if not query.strip():
            raise ValueError("query must not be empty")

        clauses = split_clauses(query)
        whole_reference = (
            self.semantic_reference_strategy != "off"
            and is_whole_turn_reference(query)
        )
        current_goals: list[Goal] = []
        cancel_clauses: list[str] = []
        has_reference_only_clause = False
        reference_precedes_current = False
        for clause in clauses:
            if whole_reference:
                has_reference_only_clause = True
                reference_precedes_current = True
                continue
            if is_cancellation_clause(
                clause,
                natural_request_boundary_guard=self.natural_request_boundary_guard,
            ):
                cancel_clauses.append(clause)
                continue
            cleaned = clean_goal_clause(clause)
            if cleaned and normalize_goal_text(cleaned):
                current_goals.append(
                    Goal(
                        text=cleaned,
                        source_turn_ids=[turn_id],
                        origin="current",
                        order=len(current_goals),
                    )
                )
            elif RECENT_SIGNAL.search(clause) or LONG_SIGNAL.search(clause):
                has_reference_only_clause = True
                if not current_goals:
                    reference_precedes_current = True
        current_goals = fuzzy_unique(current_goals)

        if mode == "none":
            interpretation = TurnInterpretation(
                context_mode="none",
                current_goals=current_goals,
                resolved_goals=current_goals,
            )
            return ResolvedTurn(interpretation=interpretation, query_variants=[query.strip()])

        reference_goals, reference_turns = self._reference_goals(
            query, mode, state, recent_events, memory_hits
        )
        cancelled: list[Goal] = []
        for clause in cancel_clauses:
            cancelled.extend(
                select_cancelled(
                    clause,
                    reference_goals,
                    natural_request_boundary_guard=self.natural_request_boundary_guard,
                )
            )
        cancelled = fuzzy_unique([clone_goal(goal, goal.origin) for goal in cancelled])
        cancelled_keys = {goal.key for goal in cancelled}
        remaining = [goal for goal in reference_goals if goal.key not in cancelled_keys]

        has_reference_signal = bool(
            explicit_turn_references(query)
            or RECENT_SIGNAL.search(query)
            or LONG_SIGNAL.search(query)
            or whole_reference
        )
        should_carry = bool(cancel_clauses) or has_reference_only_clause
        should_carry = should_carry or (not current_goals and has_reference_signal)
        carried = [clone_goal(goal, goal.origin) for goal in remaining] if should_carry else []
        ordered_goals = (
            carried + current_goals if reference_precedes_current else current_goals + carried
        )
        resolved = fuzzy_unique(ordered_goals)
        unresolved = not reference_goals and not current_goals
        notes = []
        if unresolved:
            notes.append("reference_not_resolved")
        if cancel_clauses and not cancelled:
            notes.append("cancellation_target_not_resolved")

        interpretation = TurnInterpretation(
            context_mode=mode,
            current_goals=current_goals,
            carried_goals=carried,
            cancelled_goals=cancelled,
            resolved_goals=resolved,
            reference_turn_ids=reference_turns,
            unresolved_reference=unresolved,
            notes=notes,
        )
        variants = self._query_variants(query, interpretation)
        return ResolvedTurn(interpretation=interpretation, query_variants=variants)

    def _query_variants(
        self, query: str, interpretation: TurnInterpretation
    ) -> list[str]:
        if interpretation.context_mode == "none" or interpretation.unresolved_reference:
            return [query.strip()]
        if (
            self.natural_request_boundary_guard
            and not interpretation.carried_goals
            and not interpretation.cancelled_goals
        ):
            return [query.strip()]
        active = "；".join(goal.text for goal in interpretation.resolved_goals)
        cancelled = "；".join(goal.text for goal in interpretation.cancelled_goals)
        variants = []
        if active and cancelled:
            variants.append(f"{active}；不再需要：{cancelled}。")
        elif active:
            variants.append(active)
        elif cancelled:
            variants.append(f"先前诉求已全部取消，本轮不需要调用相关工具：{cancelled}。")
        if active:
            variants.append(active)
        variants.append(query.strip())
        return list(dict.fromkeys(value for value in variants if value.strip()))


class ToolStateReducer:
    """由用户目标和规划结果确定性更新状态；Planner 结果不生成用户目标。"""

    def __init__(self, catalog_by_id: dict[str, dict[str, Any]], max_cancelled_goals: int = 50):
        self.catalog_by_id = catalog_by_id
        self.max_cancelled_goals = max_cancelled_goals

    def reduce(
        self,
        previous: CompactToolState,
        event: PlanEvent,
    ) -> CompactToolState:
        if event.status == "unresolved_reference":
            return CompactToolState(
                conversation_id=previous.conversation_id,
                version=previous.version + 1,
                last_turn_id=event.turn_id,
                last_request_goals=[clone_goal(goal, goal.origin) for goal in previous.last_request_goals],
                explicitly_carried_goals=[
                    clone_goal(goal, goal.origin) for goal in previous.explicitly_carried_goals
                ],
                cancelled_goals=[clone_goal(goal, goal.origin) for goal in previous.cancelled_goals],
                last_planned_tools=list(previous.last_planned_tools),
                active_domains=list(previous.active_domains),
                active_object_types=list(previous.active_object_types),
                updated_at=utc_now(),
            )
        if event.intent_type == "闲聊类":
            last_goals: list[Goal] = []
            carried: list[Goal] = []
            planned: list[str] = []
            domains: list[str] = []
            objects: list[str] = []
        else:
            last_goals = [clone_goal(goal, goal.origin) for goal in event.resolved_goals]
            carried = [clone_goal(goal, goal.origin) for goal in event.carried_goals]
            planned = list(event.planned_tools)
            domains = list(
                dict.fromkeys(
                    str(self.catalog_by_id[tool]["domain"])
                    for tool in planned
                    if tool in self.catalog_by_id
                )
            )
            objects = [value for value in OBJECT_TYPES if value in event.resolved_query]

        cancellations = fuzzy_unique(
            [clone_goal(goal, goal.origin) for goal in previous.cancelled_goals]
            + [clone_goal(goal, goal.origin) for goal in event.cancelled_goals]
        )[-self.max_cancelled_goals :]
        return CompactToolState(
            conversation_id=previous.conversation_id,
            version=previous.version + 1,
            last_turn_id=event.turn_id,
            last_request_goals=last_goals,
            explicitly_carried_goals=carried,
            cancelled_goals=cancellations,
            last_planned_tools=planned,
            active_domains=domains,
            active_object_types=objects,
            updated_at=utc_now(),
        )
