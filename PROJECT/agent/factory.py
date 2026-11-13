"""从机器可读配置构造 canonical Agent 运行时。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PROJECT.scripts.project_common import catalog_by_id, load_catalog

from .backends import (
    CachedIntentBackend,
    CachedPlannerBackend,
    CachedToolRetriever,
    CanonicalIntentBackend,
    CanonicalPlannerBackend,
    CanonicalRoute,
    CanonicalToolRetriever,
    FixedEcommerceIntentBackend,
    resolve_project_reference,
)
from .memory import DeterministicMemoryReducer, SQLiteMemoryStore
from .orchestrator import AgentPolicy, ToolPlanningAgent


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT / "PROJECT"
DEFAULT_CONFIG = PROJECT_ROOT / "configs/agent_runtime.json"
DEFAULT_DB = PROJECT_ROOT / "data/agent/runtime/agent_memory.sqlite3"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def config_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def resolve_config_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if value.startswith("PROJECT/"):
        return REPO_ROOT / value
    return PROJECT_ROOT / value


def load_agent_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 3:
        raise ValueError("unsupported Agent runtime config schema")
    if payload.get("status") != "canonical":
        raise ValueError("Agent runtime config is not canonical")
    required = {
        "canonical_route",
        "canonical_route_sha256",
        "memory",
        "token_budget",
        "planning",
        "model_artifacts",
        "selection_status",
    }
    if not required <= set(payload):
        raise ValueError(f"Agent runtime config lacks: {sorted(required - set(payload))}")
    memory = payload["memory"]
    token = payload["token_budget"]
    planning = payload["planning"]
    if not all(
        bool(planning.get(key))
        for key in (
            "catalog_alias_guard",
            "recent_over_generic_long",
            "natural_request_boundary_guard",
            "single_tool_goal_normalization",
            "whole_turn_reference_replay",
            "standalone_multigoal_guard",
            "whole_turn_plan_cache",
        )
    ):
        raise ValueError("Agent planning guards are not fully enabled")
    selected_id = (
        f"{memory['context_architecture']}3-n{int(memory['recent_event_limit'])}"
        f"-k{int(memory['long_memory_top_k'])}-t{int(token['target_input_tokens'])}"
    )
    selection = payload["selection_status"]
    expected_status = "locked_on_natural_short_memory_evaluation"
    if selection.get("status") != expected_status:
        raise ValueError("Agent runtime selection is not complete")
    if selection.get("selected_config_id") != selected_id:
        raise ValueError("Agent runtime parameters differ from the selected configuration")
    return payload


@dataclass(frozen=True, slots=True)
class RuntimeDevices:
    intent: str = "cuda:0"
    embedding: str = "cuda:0"
    reranker: str = "cuda:1"
    planner: str = "cuda:2"


def build_agent(
    config_path: Path = DEFAULT_CONFIG,
    db_path: str | Path = DEFAULT_DB,
    devices: RuntimeDevices = RuntimeDevices(),
    intent_base_model: str | None = None,
    planner_base_model: str | None = None,
    policy_override: AgentPolicy | None = None,
    verify_adapter_hash: bool | None = None,
    backend_cache: dict[str, dict[str, Any]] | None = None,
    require_backend_cache_hit: bool = False,
    fixed_ecommerce_intent: bool = False,
) -> ToolPlanningAgent:
    payload = load_agent_config(config_path)
    route_path = resolve_config_path(str(payload["canonical_route"]))
    route = CanonicalRoute.load(
        project_root=PROJECT_ROOT,
        path=route_path,
        required_revision=str(payload["required_canonical_revision"]),
    )
    if route.sha256 != payload["canonical_route_sha256"]:
        raise ValueError(
            "canonical route content changed; review and update agent_runtime.json explicitly"
        )
    token = payload["token_budget"]
    if verify_adapter_hash is None:
        verify_adapter_hash = bool(payload["model_artifacts"]["verify_adapter_hash"])
    if int(token["planner_output_reserve"]) != int(route.payload["planner"]["max_new_tokens"]):
        raise ValueError("Agent output reserve differs from canonical Planner max_new_tokens")
    memory = payload["memory"]
    policy = policy_override or AgentPolicy(
        context_architecture=str(memory["context_architecture"]),
        recent_event_limit=int(memory["recent_event_limit"]),
        long_memory_top_k=int(memory["long_memory_top_k"]),
        long_memory_timeout_ms=int(memory["long_memory_timeout_ms"]),
        target_input_tokens=int(token["target_input_tokens"]),
        hard_input_tokens=int(token["hard_input_tokens"]),
        max_resolved_extra_tokens=int(token["max_resolved_extra_tokens"]),
        max_query_characters=int(token["max_query_characters"]),
        catalog_alias_guard=bool(
            payload.get("planning", {}).get("catalog_alias_guard", False)
        ),
        single_tool_goal_normalization=bool(
            payload.get("planning", {}).get("single_tool_goal_normalization", False)
        ),
        recent_over_generic_long=bool(
            payload.get("planning", {}).get("recent_over_generic_long", False)
        ),
        natural_request_boundary_guard=bool(
            payload.get("planning", {}).get("natural_request_boundary_guard", False)
        ),
        whole_turn_reference_replay=bool(
            payload.get("planning", {}).get("whole_turn_reference_replay", False)
        ),
        standalone_multigoal_guard=bool(
            payload.get("planning", {}).get("standalone_multigoal_guard", False)
        ),
        whole_turn_plan_cache=bool(
            payload.get("planning", {}).get("whole_turn_plan_cache", False)
        ),
        semantic_reference_strategy=str(
            payload.get("planning", {}).get("semantic_reference_strategy", "off")
        ),
    )
    catalog = load_catalog(
        Path(resolve_project_reference(PROJECT_ROOT, str(route.payload["catalog"]["path"])))
    )
    by_id = catalog_by_id(catalog)
    if len(catalog) != 100 or len(by_id) != 100:
        raise ValueError("canonical Agent requires 100 unique catalog tools")
    store = SQLiteMemoryStore(
        db_path,
        ttl_seconds=int(memory["session_ttl_seconds"]),
    )
    reducer = DeterministicMemoryReducer(
        by_id,
        recent_event_limit=policy.recent_event_limit,
        max_summary_goals=int(memory["max_summary_goals"]),
        memory_ttl_seconds=int(memory["session_ttl_seconds"]),
    )
    intent_backend = CanonicalIntentBackend(
        route,
        device=devices.intent,
        base_model=intent_base_model,
        verify_adapter_hash=verify_adapter_hash,
    )
    if fixed_ecommerce_intent:
        intent_backend = FixedEcommerceIntentBackend()
    retriever = CanonicalToolRetriever(
        route,
        embedding_device=devices.embedding,
        reranker_device=devices.reranker,
    )
    planner = CanonicalPlannerBackend(
        route,
        by_id,
        device=devices.planner,
        base_model=planner_base_model,
        verify_adapter_hash=verify_adapter_hash,
    )
    if backend_cache is not None:
        if not fixed_ecommerce_intent:
            intent_backend = CachedIntentBackend(
                intent_backend,
                backend_cache.get("intent", {}),
                require_hit=require_backend_cache_hit,
            )
        retriever = CachedToolRetriever(
            retriever,
            backend_cache.get("retrieval", {}),
            require_hit=require_backend_cache_hit,
        )
        planner = CachedPlannerBackend(
            planner,
            backend_cache.get("planner", {}),
            require_hit=require_backend_cache_hit,
        )
    return ToolPlanningAgent(
        store=store,
        intent_backend=intent_backend,
        retriever=retriever,
        planner=planner,
        catalog_by_id=by_id,
        policy=policy,
        agent_config_hash=config_hash(
            payload
            if policy_override is None
            else {"runtime": payload, "policy_override": asdict(policy)}
        ),
        route_revision=route.revision,
        route_hash=route.sha256,
        memory_reducer=reducer,
    )
