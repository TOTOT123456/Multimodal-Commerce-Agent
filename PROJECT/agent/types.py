"""Agent 运行时使用的结构化数据契约。"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


CONTEXT_MODES = {"none", "recent", "long"}
GOAL_ORIGINS = {"current", "carried", "retrieved"}
TURN_STATUSES = {
    "ok",
    "chitchat",
    "planner_invalid",
    "budget_rejected",
    "unresolved_reference",
    "backend_error",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_goal_text(value: str) -> str:
    return re.sub(r"[\W_]+", "", value.lower(), flags=re.UNICODE)


@dataclass(slots=True)
class Goal:
    text: str
    source_turn_ids: list[int]
    origin: str = "current"
    order: int = 0

    def __post_init__(self) -> None:
        self.text = self.text.strip()
        self.source_turn_ids = sorted(set(int(value) for value in self.source_turn_ids))
        if not self.text:
            raise ValueError("goal text must not be empty")
        if not self.source_turn_ids:
            raise ValueError("goal must retain at least one source turn")
        if any(value <= 0 for value in self.source_turn_ids):
            raise ValueError("goal source turn ids must be positive")
        if self.origin not in GOAL_ORIGINS:
            raise ValueError(f"unknown goal origin: {self.origin}")
        if self.order < 0:
            raise ValueError("goal order must be non-negative")

    @property
    def key(self) -> str:
        return normalize_goal_text(self.text)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Goal":
        return cls(
            text=str(payload["text"]),
            source_turn_ids=[int(value) for value in payload.get("source_turn_ids", [])],
            origin=str(payload.get("origin", "current")),
            order=int(payload.get("order", 0)),
        )


def unique_goals(goals: list[Goal]) -> list[Goal]:
    result: list[Goal] = []
    positions: dict[str, int] = {}
    for source in goals:
        goal = Goal(
            text=source.text,
            source_turn_ids=list(source.source_turn_ids),
            origin=source.origin,
            order=source.order,
        )
        key = goal.key
        if not key:
            continue
        if key not in positions:
            positions[key] = len(result)
            result.append(goal)
            continue
        existing = result[positions[key]]
        existing.source_turn_ids = sorted(
            set(existing.source_turn_ids) | set(goal.source_turn_ids)
        )
    for index, goal in enumerate(result):
        goal.order = index
    return result


@dataclass(slots=True)
class TurnInterpretation:
    context_mode: str
    current_goals: list[Goal] = field(default_factory=list)
    carried_goals: list[Goal] = field(default_factory=list)
    cancelled_goals: list[Goal] = field(default_factory=list)
    resolved_goals: list[Goal] = field(default_factory=list)
    reference_turn_ids: list[int] = field(default_factory=list)
    unresolved_reference: bool = False
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.context_mode not in CONTEXT_MODES:
            raise ValueError(f"unknown context mode: {self.context_mode}")
        self.reference_turn_ids = sorted(set(self.reference_turn_ids))


@dataclass(slots=True)
class CompactToolState:
    conversation_id: str
    version: int = 0
    last_turn_id: int = 0
    last_request_goals: list[Goal] = field(default_factory=list)
    explicitly_carried_goals: list[Goal] = field(default_factory=list)
    cancelled_goals: list[Goal] = field(default_factory=list)
    last_planned_tools: list[str] = field(default_factory=list)
    active_domains: list[str] = field(default_factory=list)
    active_object_types: list[str] = field(default_factory=list)
    updated_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        self.conversation_id = self.conversation_id.strip()
        if not self.conversation_id:
            raise ValueError("conversation id must not be empty")
        if self.version < 0 or self.last_turn_id < 0:
            raise ValueError("state versions and turn ids must be non-negative")
        self.last_planned_tools = list(dict.fromkeys(self.last_planned_tools))
        self.active_domains = list(dict.fromkeys(self.active_domains))
        self.active_object_types = list(dict.fromkeys(self.active_object_types))

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "version": self.version,
            "last_turn_id": self.last_turn_id,
            "last_request_goals": [goal.to_dict() for goal in self.last_request_goals],
            "explicitly_carried_goals": [
                goal.to_dict() for goal in self.explicitly_carried_goals
            ],
            "cancelled_goals": [goal.to_dict() for goal in self.cancelled_goals],
            "last_planned_tools": self.last_planned_tools,
            "active_domains": self.active_domains,
            "active_object_types": self.active_object_types,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CompactToolState":
        return cls(
            conversation_id=str(payload["conversation_id"]),
            version=int(payload.get("version", 0)),
            last_turn_id=int(payload.get("last_turn_id", 0)),
            last_request_goals=[
                Goal.from_dict(value) for value in payload.get("last_request_goals", [])
            ],
            explicitly_carried_goals=[
                Goal.from_dict(value)
                for value in payload.get("explicitly_carried_goals", [])
            ],
            cancelled_goals=[
                Goal.from_dict(value) for value in payload.get("cancelled_goals", [])
            ],
            last_planned_tools=[str(value) for value in payload.get("last_planned_tools", [])],
            active_domains=[str(value) for value in payload.get("active_domains", [])],
            active_object_types=[
                str(value) for value in payload.get("active_object_types", [])
            ],
            updated_at=str(payload.get("updated_at", utc_now())),
        )


@dataclass(slots=True)
class ToolRollingSummary:
    conversation_id: str
    version: int = 0
    covered_through_turn: int = 0
    historical_goals: list[Goal] = field(default_factory=list)
    cancelled_goals: list[Goal] = field(default_factory=list)
    updated_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        self.conversation_id = self.conversation_id.strip()
        if not self.conversation_id:
            raise ValueError("conversation id must not be empty")
        if self.version < 0 or self.covered_through_turn < 0:
            raise ValueError("summary versions and turn ids must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "version": self.version,
            "covered_through_turn": self.covered_through_turn,
            "historical_goals": [goal.to_dict() for goal in self.historical_goals],
            "cancelled_goals": [goal.to_dict() for goal in self.cancelled_goals],
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ToolRollingSummary":
        return cls(
            conversation_id=str(payload["conversation_id"]),
            version=int(payload.get("version", 0)),
            covered_through_turn=int(payload.get("covered_through_turn", 0)),
            historical_goals=[
                Goal.from_dict(value) for value in payload.get("historical_goals", [])
            ],
            cancelled_goals=[
                Goal.from_dict(value) for value in payload.get("cancelled_goals", [])
            ],
            updated_at=str(payload.get("updated_at", utc_now())),
        )


@dataclass(slots=True)
class PlanEvent:
    conversation_id: str
    turn_id: int
    created_at: str
    raw_query: str
    resolved_query: str
    context_mode: str
    current_goals: list[Goal]
    carried_goals: list[Goal]
    cancelled_goals: list[Goal]
    resolved_goals: list[Goal]
    reference_turn_ids: list[int]
    intent_type: str
    candidate_tools: list[str]
    planned_tools: list[str]
    strict_output: bool
    status: str
    raw_planner_output: str = ""
    prompt_version: str = ""
    prompt_hash: str = ""
    model_revision: str = ""
    retrieval_revision: str = ""
    route_hash: str = ""
    agent_config_hash: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.conversation_id = self.conversation_id.strip()
        self.raw_query = self.raw_query.strip()
        self.resolved_query = self.resolved_query.strip()
        self.reference_turn_ids = sorted(set(int(value) for value in self.reference_turn_ids))
        if not self.conversation_id:
            raise ValueError("conversation id must not be empty")
        if not self.raw_query or not self.resolved_query:
            raise ValueError("raw and resolved queries must not be empty")
        if self.turn_id <= 0:
            raise ValueError("turn id must be positive")
        if self.context_mode not in CONTEXT_MODES:
            raise ValueError(f"unknown context mode: {self.context_mode}")
        if self.status not in TURN_STATUSES:
            raise ValueError(f"unknown turn status: {self.status}")
        if self.intent_type not in {"", "电商类", "闲聊类"}:
            raise ValueError(f"unknown intent type: {self.intent_type}")
        if len(self.candidate_tools) != len(set(self.candidate_tools)):
            raise ValueError("candidate tools must be unique")
        if len(self.planned_tools) != len(set(self.planned_tools)):
            raise ValueError("planned tools must be unique")
        if self.strict_output and not set(self.planned_tools) <= set(self.candidate_tools):
            raise ValueError("strict planned tools must stay inside candidate tools")
        if any(value <= 0 or value >= self.turn_id for value in self.reference_turn_ids):
            raise ValueError("reference turns must precede the current turn")
        goal_sources = [
            source_turn
            for goals in (
                self.current_goals,
                self.carried_goals,
                self.cancelled_goals,
                self.resolved_goals,
            )
            for goal in goals
            for source_turn in goal.source_turn_ids
        ]
        if any(value > self.turn_id for value in goal_sources):
            raise ValueError("goal provenance cannot point to a future turn")
        if self.status == "ok" and (not self.strict_output or self.intent_type != "电商类"):
            raise ValueError("ok events require accepted strict ecommerce output")
        if self.status == "chitchat" and (
            not self.strict_output or self.intent_type != "闲聊类" or self.planned_tools
        ):
            raise ValueError("chitchat events require an accepted empty plan")
        if self.status == "unresolved_reference" and (
            not self.strict_output or self.intent_type or self.planned_tools
        ):
            raise ValueError("unresolved references require an accepted empty plan")
        if self.status in {"planner_invalid", "budget_rejected", "backend_error"} and (
            self.strict_output or self.planned_tools
        ):
            raise ValueError("failed events cannot expose an accepted plan")
        if self.input_tokens < 0 or self.output_tokens < 0:
            raise ValueError("token counts must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PlanEvent":
        goal_fields = {
            name: [Goal.from_dict(value) for value in payload.get(name, [])]
            for name in (
                "current_goals",
                "carried_goals",
                "cancelled_goals",
                "resolved_goals",
            )
        }
        return cls(
            conversation_id=str(payload["conversation_id"]),
            turn_id=int(payload["turn_id"]),
            created_at=str(payload["created_at"]),
            raw_query=str(payload["raw_query"]),
            resolved_query=str(payload["resolved_query"]),
            context_mode=str(payload["context_mode"]),
            reference_turn_ids=[int(value) for value in payload.get("reference_turn_ids", [])],
            intent_type=str(payload.get("intent_type", "")),
            candidate_tools=[str(value) for value in payload.get("candidate_tools", [])],
            planned_tools=[str(value) for value in payload.get("planned_tools", [])],
            strict_output=bool(payload.get("strict_output", False)),
            status=str(payload["status"]),
            raw_planner_output=str(payload.get("raw_planner_output", "")),
            prompt_version=str(payload.get("prompt_version", "")),
            prompt_hash=str(payload.get("prompt_hash", "")),
            model_revision=str(payload.get("model_revision", "")),
            retrieval_revision=str(payload.get("retrieval_revision", "")),
            route_hash=str(payload.get("route_hash", "")),
            agent_config_hash=str(payload.get("agent_config_hash", "")),
            input_tokens=int(payload.get("input_tokens", 0)),
            output_tokens=int(payload.get("output_tokens", 0)),
            latency_ms={
                str(key): float(value) for key, value in payload.get("latency_ms", {}).items()
            },
            warnings=[str(value) for value in payload.get("warnings", [])],
            **goal_fields,
        )


@dataclass(slots=True)
class MemoryRecord:
    conversation_id: str
    source_turn_id: int
    text: str
    goals: list[Goal]
    planned_tools: list[str]
    domains: list[str]
    created_at: str
    expires_at: str | None = None

    def __post_init__(self) -> None:
        self.conversation_id = self.conversation_id.strip()
        self.text = self.text.strip()
        self.planned_tools = list(dict.fromkeys(self.planned_tools))
        self.domains = list(dict.fromkeys(self.domains))
        if not self.conversation_id or not self.text:
            raise ValueError("memory conversation and text must not be empty")
        if self.source_turn_id <= 0:
            raise ValueError("memory source turn must be positive")
        if any(
            source_turn > self.source_turn_id
            for goal in self.goals
            for source_turn in goal.source_turn_ids
        ):
            raise ValueError("memory goal provenance cannot point to a future turn")

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "source_turn_id": self.source_turn_id,
            "text": self.text,
            "goals": [goal.to_dict() for goal in self.goals],
            "planned_tools": self.planned_tools,
            "domains": self.domains,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MemoryRecord":
        return cls(
            conversation_id=str(payload["conversation_id"]),
            source_turn_id=int(payload["source_turn_id"]),
            text=str(payload["text"]),
            goals=[Goal.from_dict(value) for value in payload.get("goals", [])],
            planned_tools=[str(value) for value in payload.get("planned_tools", [])],
            domains=[str(value) for value in payload.get("domains", [])],
            created_at=str(payload["created_at"]),
            expires_at=(str(payload["expires_at"]) if payload.get("expires_at") else None),
        )


@dataclass(slots=True)
class MemoryHit:
    record: MemoryRecord
    score: float
    match_reason: str


@dataclass(slots=True)
class SessionSnapshot:
    state: CompactToolState
    summary: ToolRollingSummary
    recent_events: list[PlanEvent]


@dataclass(slots=True)
class IntentResult:
    intent_type: str
    keywords: str = ""
    raw_output: str = ""
    latency_ms: float = 0.0


@dataclass(slots=True)
class RetrievalResult:
    candidate_tools: list[str]
    source_tools: list[str] = field(default_factory=list)
    reranked_tools: list[str] = field(default_factory=list)
    revision: str = ""
    route_hash: str = ""
    latency_ms: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PlannerResult:
    tools: list[str]
    strict_ok: bool
    strict_error: str
    raw_output: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    prompt_version: str
    prompt_hash: str
    model_revision: str


@dataclass(slots=True)
class AgentTurnResult:
    conversation_id: str
    turn_id: int
    status: str
    tools: list[str]
    resolved_query: str
    context_mode: str
    strict_output: bool
    warnings: list[str]
    event: PlanEvent

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "turn_id": self.turn_id,
            "status": self.status,
            "tools": self.tools,
            "resolved_query": self.resolved_query,
            "context_mode": self.context_mode,
            "strict_output": self.strict_output,
            "warnings": self.warnings,
            "event": self.event.to_dict(),
        }
