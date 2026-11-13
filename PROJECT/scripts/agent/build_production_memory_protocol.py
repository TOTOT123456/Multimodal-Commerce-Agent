#!/usr/bin/env python3
"""从无 query 计划构造短会话、自然承接的生产近似记忆协议。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from PROJECT.scripts.agent.evaluate_multiturn_agent import (  # noqa: E402
    normalized_query,
    validate_dataset,
)
from PROJECT.scripts.project_common import (  # noqa: E402
    catalog_by_id,
    load_catalog,
    load_jsonl,
    write_json,
    write_jsonl,
)


PROJECT_ROOT = REPO_ROOT / "PROJECT"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/agent/_aux/natural_short_memory_evaluation"
CASES_PER_SPLIT = 1000
TARGET_KINDS = ("natural_replay", "standalone_switch")
TARGET_KIND_WEIGHTS = (0.60, 0.40)
REPLAY_DISTANCES = (1, 2, 3, 4, 5)
REPLAY_DISTANCE_WEIGHTS = (0.55, 0.25, 0.12, 0.06, 0.02)
STANDALONE_HISTORY_DEPTHS = (1, 2, 3)
STANDALONE_HISTORY_WEIGHTS = (0.65, 0.25, 0.10)
TOOL_COUNTS = (1, 2, 3, 4, 5, 6)
# 与 24K SFT 轨迹的工具数边际分布一致，避免对单工具任务过采样。
TOOL_COUNT_WEIGHTS = (0.375, 0.086, 0.161, 0.128, 0.125, 0.125)
PROTECTED = [
    PROJECT_ROOT / "data/tool_call_trace/train.jsonl",
    PROJECT_ROOT / "data/tool_call_trace/test.jsonl",
    PROJECT_ROOT / "data/intent_recognition/intent_all.jsonl",
    PROJECT_ROOT
    / "data/tool_call_trace/_aux/assignment3/api_native_dev_2400/dev.jsonl",
    PROJECT_ROOT
    / "data/tool_call_trace/_aux/assignment3/assignment3_100tool_api/test_api/test.jsonl",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def query_of(row: dict[str, Any]) -> str:
    return str(row.get("问题", row.get("query", ""))).strip()


def tools_of(row: dict[str, Any]) -> list[str]:
    value = row.get("工具", row.get("gold_tools", row.get("tools", [])))
    if isinstance(value, list):
        return [str(item) for item in value]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def depends_on_closure(
    tool_id: str, by_id: dict[str, dict[str, Any]]
) -> list[str]:
    output: list[str] = []

    def visit(current: str) -> None:
        for dependency in by_id[current].get("depends_on", []):
            dependency = str(dependency)
            if dependency in by_id and dependency not in output:
                visit(dependency)
                output.append(dependency)

    visit(tool_id)
    return output


def ordered_plan(
    anchor: str,
    tool_count: int,
    by_id: dict[str, dict[str, Any]],
    exposure: Counter[str],
    rng: random.Random,
    forbidden: set[tuple[str, ...]] | None = None,
) -> list[str]:
    forbidden = forbidden or set()
    anchors = list(by_id)
    anchors.sort(
        key=lambda value: (
            0 if value == anchor else 1,
            exposure[value],
            hashlib.sha256(f"{rng.random()}:{value}".encode()).hexdigest(),
        )
    )
    for candidate_anchor in anchors:
        plan = depends_on_closure(candidate_anchor, by_id) + [candidate_anchor]
        plan = list(dict.fromkeys(plan))[-tool_count:]
        domains = {str(by_id[value]["domain"]) for value in plan}
        while len(plan) < tool_count:
            candidates = [value for value in by_id if value not in plan]
            candidates.sort(
                key=lambda value: (
                    exposure[value],
                    0 if str(by_id[value]["domain"]) in domains else 1,
                    hashlib.sha256(f"{rng.random()}:{value}".encode()).hexdigest(),
                )
            )
            chosen = candidates[0]
            prerequisites = [
                value
                for value in depends_on_closure(chosen, by_id)
                if value not in plan
            ]
            remaining = tool_count - len(plan)
            if len(prerequisites) + 1 <= remaining:
                plan.extend(prerequisites)
            if len(plan) < tool_count:
                plan.append(chosen)
            domains.add(str(by_id[chosen]["domain"]))
        plan = list(dict.fromkeys(plan))[:tool_count]
        if len(plan) == tool_count and tuple(plan) not in forbidden:
            return plan
    raise RuntimeError("unable to construct an allowed ordered plan")


def reference_family(distance: int, rng: random.Random) -> str:
    if distance == 1:
        return rng.choices(
            ("implicit_confirmation", "recent_resume"), weights=(0.60, 0.40), k=1
        )[0]
    if distance == 2:
        return rng.choices(
            ("recent_resume", "object_anchor"), weights=(0.60, 0.40), k=1
        )[0]
    return "object_anchor"


def sample_schedule(role: str, seed: int, count: int) -> list[dict[str, Any]]:
    """独立抽样，并只设置统计功效下限，不强制人为等分。"""

    for attempt in range(1000):
        rng = random.Random(seed + attempt * 1009)
        rows = []
        for index in range(count):
            target_kind = rng.choices(
                TARGET_KINDS, weights=TARGET_KIND_WEIGHTS, k=1
            )[0]
            if target_kind == "natural_replay":
                history_gap = rng.choices(
                    REPLAY_DISTANCES, weights=REPLAY_DISTANCE_WEIGHTS, k=1
                )[0]
                family = reference_family(history_gap, rng)
            else:
                history_gap = rng.choices(
                    STANDALONE_HISTORY_DEPTHS,
                    weights=STANDALONE_HISTORY_WEIGHTS,
                    k=1,
                )[0]
                family = "standalone_switch"
            source_turn = rng.choices((1, 2), weights=(0.80, 0.20), k=1)[0]
            rows.append(
                {
                    "case_id": f"prodmem_{role}_{index:05d}",
                    "case_index": index,
                    "dataset_role": role,
                    "target_kind": target_kind,
                    "reference_family": family,
                    "source_turn": source_turn,
                    "history_gap": history_gap,
                    "target_turn": source_turn + history_gap,
                    "source_plan_id": f"prodmem_{role}_source_{index:05d}",
                    "target_plan_id": (
                        None
                        if target_kind == "natural_replay"
                        else f"prodmem_{role}_target_{index:05d}"
                    ),
                }
            )
        kinds = Counter(row["target_kind"] for row in rows)
        replay_distances = Counter(
            row["history_gap"]
            for row in rows
            if row["target_kind"] == "natural_replay"
        )
        if (
            kinds["natural_replay"] >= 550
            and kinds["standalone_switch"] >= 350
            and all(replay_distances[value] >= 8 for value in REPLAY_DISTANCES)
        ):
            return rows
    raise RuntimeError("unable to sample a schedule satisfying power floors")


def build_query_free_plans(
    role: str,
    schedule: list[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    exposure: Counter[str] = Counter()
    tools = sorted(by_id)
    write_tools = [
        value for value in tools if str(by_id[value].get("operation_type")) == "write"
    ]
    if not write_tools:
        raise RuntimeError("catalog contains no write tools for implicit confirmations")
    rng.shuffle(tools)
    output = []
    for row in schedule:
        source_count = rng.choices(
            TOOL_COUNTS, weights=TOOL_COUNT_WEIGHTS, k=1
        )[0]
        if row["reference_family"] == "implicit_confirmation":
            source_anchor_pool = write_tools
        else:
            source_anchor_pool = tools
        anchor = min(
            source_anchor_pool,
            key=lambda value: (
                exposure[value],
                hashlib.sha256(
                    f"source:{seed}:{row['case_index']}:{value}".encode()
                ).hexdigest(),
            ),
        )
        source_tools = ordered_plan(anchor, source_count, by_id, exposure, rng)
        if row["reference_family"] == "implicit_confirmation" and not any(
            str(by_id[value].get("operation_type")) == "write"
            for value in source_tools
        ):
            raise RuntimeError("implicit confirmation source lacks a write operation")
        exposure.update(source_tools)
        output.append(
            {
                "id": row["source_plan_id"],
                "tools": source_tools,
                "tool_count": len(source_tools),
                "domain": "+".join(
                    sorted({str(by_id[value]["domain"]) for value in source_tools})
                ),
                "combo_seen_bucket": "natural_short_memory_source",
                "construction": {
                    "type": "query_free_natural_short_memory",
                    "dataset_role": role,
                    "pool_role": "source",
                    "case_index": row["case_index"],
                    "target_kind": row["target_kind"],
                    "reference_family": row["reference_family"],
                    "generation_constraints": (
                        "消息中必须自然出现一个明确、具体、后续可复指的业务对象或线索，"
                        "例如具体商品类型、退款单、默认地址或尺码档案；不能只写这个商品、"
                        "这单或这件东西。该线索只是背景，不能增加目标能力之外的新动作。"
                        if row["reference_family"] == "object_anchor"
                        else ""
                    ),
                },
            }
        )
        if row["target_plan_id"] is None:
            continue
        target_count = rng.choices(
            TOOL_COUNTS, weights=TOOL_COUNT_WEIGHTS, k=1
        )[0]
        target_pool = set(tools)
        target_anchor = min(
            target_pool,
            key=lambda value: (
                exposure[value],
                hashlib.sha256(
                    f"target:{seed}:{row['case_index']}:{value}".encode()
                ).hexdigest(),
            ),
        )
        target_tools = ordered_plan(
            target_anchor,
            target_count,
            by_id,
            exposure,
            rng,
            forbidden={tuple(source_tools)},
        )
        exposure.update(target_tools)
        output.append(
            {
                "id": row["target_plan_id"],
                "tools": target_tools,
                "tool_count": len(target_tools),
                "domain": "+".join(
                    sorted({str(by_id[value]["domain"]) for value in target_tools})
                ),
                "combo_seen_bucket": "natural_short_memory_control",
                "construction": {
                    "type": "query_free_natural_short_memory",
                    "dataset_role": role,
                    "pool_role": "target",
                    "case_index": row["case_index"],
                    "target_kind": row["target_kind"],
                },
            }
        )
    return output


def plans_stage(args: argparse.Namespace) -> int:
    if args.cases_per_split < 800:
        raise SystemExit("--cases-per-split must be at least 800")
    by_id = catalog_by_id(load_catalog(PROJECT_ROOT / "configs/tool_catalog.jsonl"))
    if len(by_id) != 100:
        raise SystemExit("expected exactly 100 catalog tools")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    artifacts = {}
    for role, role_seed in (("dev", args.seed), ("test", args.seed + 100003)):
        schedule = sample_schedule(role, role_seed, args.cases_per_split)
        plans = build_query_free_plans(role, schedule, by_id, role_seed + 70001)
        schedule_path = args.output_dir / f"schedule_{role}.jsonl"
        plan_path = args.output_dir / f"source_generation/{role}_plan.jsonl"
        write_jsonl(schedule_path, schedule)
        write_jsonl(plan_path, plans)
        counts[role] = {
            "cases": len(schedule),
            "query_free_plans": len(plans),
            "target_kind": dict(sorted(Counter(row["target_kind"] for row in schedule).items())),
            "replay_distance": dict(
                sorted(
                    Counter(
                        row["history_gap"]
                        for row in schedule
                        if row["target_kind"] == "natural_replay"
                    ).items()
                )
            ),
            "reference_family": dict(
                sorted(
                    Counter(
                        row["reference_family"]
                        for row in schedule
                        if row["target_kind"] == "natural_replay"
                    ).items()
                )
            ),
            "source_turn": dict(sorted(Counter(row["source_turn"] for row in schedule).items())),
            "max_target_turn": max(row["target_turn"] for row in schedule),
            "tool_count": dict(sorted(Counter(row["tool_count"] for row in plans).items())),
        }
        artifacts[role] = {
            "schedule": str(schedule_path.relative_to(REPO_ROOT)),
            "schedule_sha256": sha256_file(schedule_path),
            "query_free_plan": str(plan_path.relative_to(REPO_ROOT)),
            "query_free_plan_sha256": sha256_file(plan_path),
        }
    design = {
        "schema_version": 1,
        "status": "locked_before_formal_query_generation",
        "scope": "short production-like ecommerce conversations with natural recent carry-over",
        "seed": args.seed,
        "cases_per_split": args.cases_per_split,
        "hard_constraints": {
            "conversation_turns": "2..7",
            "history_gap": "1..5 for replay; 1..3 for controls",
            "numbered_turn_reference_allowed": False,
            "evaluation_language_allowed": False,
            "reference_examples_allowed": [
                "行，就这么办吧",
                "刚才说的那几件事接着办吧",
                "那个耳机订单前面谈的几件事继续处理吧",
            ],
            "banned_reference_shapes": [
                "第N轮",
                "完整原样重做",
                "步骤和顺序都别变",
                "调用之前的任务",
            ],
            "generic_target_queries_may_repeat": True,
            "whole_replay_label_clarity": (
                "multi-action non-confirmation references must use a natural collective "
                "cue such as 那些事/几件事/那几项/前面说的都照办; single-action "
                "sources may use a natural singular 的事 cue; numbered-turn and "
                "evaluation wording remain forbidden"
            ),
            "semantic_rejection_sampling": (
                "before any candidate evaluation, generator+independent-judge failures "
                "may be excluded under one split-invariant rule: at least 98% of each "
                "scheduled split must pass; rejected cases and reasons remain auditable"
            ),
            "isolation_unit": "full conversation history plus query-free source plan",
            "anti_inheritance_control": (
                "a complete standalone new request after unrelated short history; no "
                "synthetic same-entity follow-up is claimed without parameter labels"
            ),
            "dependent_distractors": (
                "tool- and domain-disjoint from the referenced source to keep the "
                "referent identifiable without numbered-turn language"
            ),
        },
        "sampling": {
            "target_kinds": dict(zip(TARGET_KINDS, TARGET_KIND_WEIGHTS)),
            "replay_distance": dict(zip(REPLAY_DISTANCES, REPLAY_DISTANCE_WEIGHTS)),
            "standalone_history_depth": dict(
                zip(STANDALONE_HISTORY_DEPTHS, STANDALONE_HISTORY_WEIGHTS)
            ),
            "source_turn": {"1": 0.80, "2": 0.20},
            "source_and_standalone_tool_count": dict(
                zip(TOOL_COUNTS, TOOL_COUNT_WEIGHTS)
            ),
            "ambiguous_same_domain_distractors_for_dependent_targets": 0.0,
            "tool_count_source": "PROJECT/data/tool_call_trace/train.jsonl marginal distribution",
            "note": "stratified for measurement, not claimed as measured production traffic prevalence",
        },
        "metrics": {
            "replay": "ordered tool-list exact match on natural_replay targets",
            "no_inherit": "ordered tool-list exact match on standalone_switch targets",
            "ordinary": "ordered tool-list exact match on unscored history turns",
        },
        "selection_rule": {
            "non_inferiority_margin_pp_vs_stateless_on_no_inherit": 1.0,
            "rank_eligible_by": [
                "higher natural_replay TraceEM",
                "higher standalone_switch TraceEM",
                "lower p95 input tokens",
                "earlier candidate label",
            ],
            "confirmation_candidates": (
                "all preregistered cumulative candidates, executed once only after "
                "the selected label is locked on development"
            ),
        },
        "counts": counts,
        "catalog": "PROJECT/configs/tool_catalog.jsonl",
        "catalog_sha256": sha256_file(PROJECT_ROOT / "configs/tool_catalog.jsonl"),
        "artifacts": artifacts,
    }
    write_json(args.output_dir / "design.lock.json", design)
    print(json.dumps(counts, ensure_ascii=False))
    return 0


def choose_distractors(
    source: dict[str, Any],
    pool: list[dict[str, Any]],
    count: int,
    seed: int,
    require_disjoint: bool,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    if count <= 0:
        return [], Counter()
    source_tools = set(tools_of(source))
    source_domains = set(str(source.get("domain", "")).split("+"))
    candidates = [row for row in pool if row["id"] != source["id"]]
    disjoint = [
        row
        for row in candidates
        if not (source_tools & set(tools_of(row)))
        and not (source_domains & set(str(row.get("domain", "")).split("+")))
    ]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    rng.shuffle(disjoint)
    if require_disjoint:
        candidates = disjoint
    output = []
    selected: set[str] = set()
    kinds: Counter[str] = Counter()
    for _ in range(count):
        chosen = next(
            (row for row in candidates if str(row["id"]) not in selected), None
        )
        if chosen is None:
            raise RuntimeError(f"not enough distractors for {source['id']}")
        selected.add(str(chosen["id"]))
        output.append(chosen)
        kinds[
            "domain_and_tool_disjoint" if require_disjoint else "random"
        ] += 1
    return output, kinds


def generated_sources(output_dir: Path, role: str) -> list[dict[str, Any]]:
    filename = "dev.jsonl" if role == "dev" else "test.jsonl"
    return load_jsonl(output_dir / f"source_generation/{role}/{filename}")


def target_plans_stage(args: argparse.Namespace) -> int:
    for role, role_seed in (("dev", args.seed), ("test", args.seed + 100003)):
        schedule = load_jsonl(args.output_dir / f"schedule_{role}.jsonl")
        sources = generated_sources(args.output_dir, role)
        source_by_id = {str(row["id"]): row for row in sources}
        source_pool = [
            row
            for row in sources
            if (row.get("construction") or {}).get("pool_role") == "source"
        ]
        expected_ids = {
            str(row["source_plan_id"]) for row in schedule
        } | {
            str(row["target_plan_id"])
            for row in schedule
            if row["target_plan_id"] is not None
        }
        if set(source_by_id) != expected_ids:
            raise SystemExit(f"{role}: generated source ids do not match locked plans")
        output = []
        for row in schedule:
            source = source_by_id[str(row["source_plan_id"])]
            distractor_count = int(row["target_turn"]) - 2
            distractors, distractor_types = choose_distractors(
                source,
                source_pool,
                distractor_count,
                role_seed + int(row["case_index"]) * 101,
                require_disjoint=row["target_kind"] == "natural_replay",
            )
            history = []
            cursor = 0
            for turn_id in range(1, int(row["target_turn"])):
                current = source if turn_id == int(row["source_turn"]) else distractors[cursor]
                if turn_id != int(row["source_turn"]):
                    cursor += 1
                history.append(
                    {
                        "turn_id": turn_id,
                        "id": str(current["id"]),
                        "query": query_of(current),
                        "tools": tools_of(current),
                        "domain": str(current.get("domain", "")),
                        "is_source": turn_id == int(row["source_turn"]),
                    }
                )
            target = (
                None
                if row["target_plan_id"] is None
                else source_by_id[str(row["target_plan_id"])]
            )
            output.append(
                {
                    **row,
                    "source_query": query_of(source),
                    "source_tools": tools_of(source),
                    "source_domain": str(source.get("domain", "")),
                    "target_base_query": "" if target is None else query_of(target),
                    "target_tools": [] if target is None else tools_of(target),
                    "target_domain": "" if target is None else str(target.get("domain", "")),
                    "history_turns": history,
                    "distractor_type": dict(sorted(distractor_types.items())),
                }
            )
        path = args.output_dir / f"target_generation/{role}/plan.jsonl"
        write_jsonl(path, output)
        print(f"{role}: wrote {len(output)} target plans")
    return 0


def build_split(
    role: str,
    plans: list[dict[str, Any]],
    targets: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    split = "dev" if role == "dev" else "test"
    output = []
    for case_index, plan in enumerate(plans):
        conversation = f"natural-short-memory-{role}-{case_index:05d}"
        for history in plan["history_turns"]:
            turn_id = int(history["turn_id"])
            output.append(
                {
                    "id": f"natural_short_memory_{role}_{len(output):07d}",
                    "conversation_id": conversation,
                    "turn_id": turn_id,
                    "query": str(history["query"]),
                    "gold_tools": [str(value) for value in history["tools"]],
                    "gold_resolved_goals": [str(history["query"])],
                    "gold_cancelled_goals": [],
                    "gold_reference_turn_ids": [],
                    "context_dependency": "none",
                    "scenario": "self_contained" if turn_id == 1 else "task_switch",
                    "memory_distance": None,
                    "history_gap": None,
                    "is_scored_target": False,
                    "target_kind": "history",
                    "reference_family": "none",
                    "source_trace_id": f"natural-short-memory:{role}:history:{history['id']}:{case_index}:{turn_id}",
                    "expression_cluster": f"natural_short_memory_{role}_history_{case_index}_{turn_id}",
                    "context_skeleton": f"natural_short_memory_{role}_history_{case_index}_{turn_id}",
                    "eval_split": split,
                }
            )
        accepted = targets[str(plan["case_id"])]
        target_kind = str(plan["target_kind"])
        if target_kind == "natural_replay":
            gold_tools = [str(value) for value in plan["source_tools"]]
            goals = [str(plan["source_query"])]
            references = [int(plan["source_turn"])]
            dependency = "recent"
            memory_distance: int | None = int(plan["history_gap"])
        elif target_kind == "contextual_new_action":
            gold_tools = [str(value) for value in plan["target_tools"]]
            goals = [str(plan["target_base_query"])]
            references = [int(plan["source_turn"])]
            dependency = "recent"
            memory_distance = int(plan["history_gap"])
        else:
            gold_tools = [str(value) for value in plan["target_tools"]]
            goals = [str(accepted["query"])]
            references = []
            dependency = "none"
            memory_distance = None
        output.append(
            {
                "id": f"natural_short_memory_{role}_{len(output):07d}",
                "conversation_id": conversation,
                "turn_id": int(plan["target_turn"]),
                "query": str(accepted["query"]),
                "gold_tools": gold_tools,
                "gold_resolved_goals": goals,
                "gold_cancelled_goals": [],
                "gold_reference_turn_ids": references,
                "context_dependency": dependency,
                "scenario": {
                    "natural_replay": "reference",
                    "contextual_new_action": "inheritance",
                    "standalone_switch": "task_switch",
                }[target_kind],
                "memory_distance": memory_distance,
                "history_gap": int(plan["history_gap"]),
                "is_scored_target": True,
                "target_kind": target_kind,
                "reference_family": str(plan["reference_family"]),
                "source_trace_id": f"natural-short-memory:{role}:target:{plan['case_id']}",
                "expression_cluster": f"natural_short_memory_{role}_{target_kind}_{case_index}",
                "context_skeleton": (
                    f"natural_short_memory_{role}_{target_kind}_{plan['reference_family']}_"
                    f"gap{plan['history_gap']}_{case_index}"
                ),
                "eval_split": split,
            }
        )
    return output


def assemble_stage(args: argparse.Namespace) -> int:
    catalog_ids = set(
        catalog_by_id(load_catalog(PROJECT_ROOT / "configs/tool_catalog.jsonl"))
    )
    protected = [path for path in PROTECTED if path.exists()] + list(
        args.additional_protected
    )
    split_rows = {}
    plans_by_role = {}
    target_audits = {}
    source_audits = {}
    audits = {}
    for role in ("dev", "test"):
        plans = load_jsonl(args.output_dir / f"target_generation/{role}/plan.jsonl")
        accepted_path = args.output_dir / f"target_generation/{role}/accepted.jsonl"
        accepted = {str(row["case_id"]): row for row in load_jsonl(accepted_path)}
        planned_ids = {str(row["case_id"]) for row in plans}
        if len(plans) != args.cases_per_split or not set(accepted) <= planned_ids:
            raise SystemExit(f"{role}: target generation ids do not match locked plans")
        target_audit_path = args.output_dir / f"target_generation/{role}/audit.json"
        target_audit = json.loads(target_audit_path.read_text(encoding="utf-8"))
        if target_audit.get("strict_ok") is not True:
            raise SystemExit(f"{role}: target generation audit failed")
        missing_ids = planned_ids - set(accepted)
        if set(map(str, target_audit.get("missing", []))) != missing_ids:
            raise SystemExit(f"{role}: target exclusion audit does not match accepted rows")
        source_filename = "dev.jsonl" if role == "dev" else "test.jsonl"
        source_path = args.output_dir / f"source_generation/{role}/{source_filename}"
        source_audit_path = source_path.parent / "generation_audit.json"
        source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
        if source_audit.get("strict_ok") is not True:
            raise SystemExit(f"{role}: source generation audit failed")
        accepted_plans = [
            plan for plan in plans if str(plan["case_id"]) in accepted
        ]
        rows_with_metadata = build_split(role, accepted_plans, accepted)
        metadata_fields = (
            "history_gap",
            "is_scored_target",
            "target_kind",
            "reference_family",
        )
        metadata = [
            {
                "id": str(row["id"]),
                **{field: row[field] for field in metadata_fields},
            }
            for row in rows_with_metadata
        ]
        rows = [
            {key: value for key, value in row.items() if key not in metadata_fields}
            for row in rows_with_metadata
        ]
        audits[role] = validate_dataset(
            rows,
            role,
            catalog_ids,
            protected,
            require_tool_combination_isolation=False,
            allow_duplicate_unscored_queries=True,
            allow_duplicate_scored_queries=True,
        )
        split_rows[role] = rows
        plans_by_role[role] = accepted_plans
        write_jsonl(args.output_dir / f"target_metadata_{role}.jsonl", metadata)
        target_audits[role] = target_audit
        source_audits[role] = source_audit
    metadata_by_role = {
        role: {
            str(row["id"]): row
            for row in load_jsonl(args.output_dir / f"target_metadata_{role}.jsonl")
        }
        for role in ("dev", "test")
    }
    dev_targets = {
        normalized_query(row["query"])
        for row in split_rows["dev"]
        if metadata_by_role["dev"][str(row["id"])]["is_scored_target"]
    }
    test_targets = {
        normalized_query(row["query"])
        for row in split_rows["test"]
        if metadata_by_role["test"][str(row["id"])]["is_scored_target"]
    }
    dev_queries = {normalized_query(row["query"]) for row in split_rows["dev"]}
    test_queries = {normalized_query(row["query"]) for row in split_rows["test"]}
    if dev_targets & test_targets:
        raise SystemExit("dev/test scored target query overlap")
    if dev_queries & test_queries:
        raise SystemExit("dev/test normalized query overlap")
    history_queries = {
        role: {
            normalized_query(row["query"])
            for row in split_rows[role]
            if not metadata_by_role[role][str(row["id"])]["is_scored_target"]
        }
        for role in ("dev", "test")
    }
    if history_queries["dev"] & history_queries["test"]:
        raise SystemExit("dev/test source or distractor query overlap")

    def conversation_signatures(role: str) -> set[str]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in split_rows[role]:
            grouped.setdefault(str(row["conversation_id"]), []).append(row)
        return {
            hashlib.sha256(
                json.dumps(
                    [
                        {
                            "query": normalized_query(str(row["query"])),
                            "gold_tools": row["gold_tools"],
                            "target_kind": metadata_by_role[role][str(row["id"])][
                                "target_kind"
                            ],
                        }
                        for row in rows
                    ],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            for rows in grouped.values()
        }

    full_conversation_overlap = len(
        conversation_signatures("dev") & conversation_signatures("test")
    )
    if full_conversation_overlap:
        raise SystemExit("dev/test full conversation overlap")
    for role, rows in split_rows.items():
        write_jsonl(args.output_dir / f"{role}.jsonl", rows)
    counts = {}
    for role, rows in split_rows.items():
        targets = [
            row
            for row in rows
            if metadata_by_role[role][str(row["id"])]["is_scored_target"]
        ]
        replays = [
            row
            for row in targets
            if metadata_by_role[role][str(row["id"])]["target_kind"]
            == "natural_replay"
        ]
        counts[role] = {
            "scheduled_conversations": len(plans_by_role[role])
            + len(target_audits[role].get("missing", [])),
            "conversations": len(targets),
            "audited_exclusions": len(target_audits[role].get("missing", [])),
            "executed_rows": len(rows),
            "conversation_length": {
                "min": min(row["turn_id"] for row in targets),
                "mean": sum(int(row["turn_id"]) for row in targets) / len(targets),
                "max": max(row["turn_id"] for row in targets),
            },
            "target_kind": dict(
                sorted(
                    Counter(
                        metadata_by_role[role][str(row["id"])]["target_kind"]
                        for row in targets
                    ).items()
                )
            ),
            "replay_distance": dict(
                sorted(
                    Counter(
                        metadata_by_role[role][str(row["id"])]["history_gap"]
                        for row in replays
                    ).items()
                )
            ),
            "reference_family": dict(
                sorted(
                    Counter(
                        metadata_by_role[role][str(row["id"])]["reference_family"]
                        for row in replays
                    ).items()
                )
            ),
            "replay_tool_count": dict(sorted(Counter(len(row["gold_tools"]) for row in replays).items())),
        }
    protocol = {
        "schema_version": 1,
        "status": "built_before_candidate_evaluation",
        "design_lock_sha256": sha256_file(args.output_dir / "design.lock.json"),
        "scope": "natural short ecommerce sessions with replay and anti-inheritance controls",
        "counts": counts,
        "audits": audits,
        "generation_quality": {
            role: {
                "source": {
                    "strict_ok": source_audits[role]["strict_ok"],
                    "rows": source_audits[role]["accepted_count"],
                    "generator_model": source_audits[role]["current_run_generator_model"],
                    "judge_model": source_audits[role]["current_run_judge_model"],
                },
                "targets": target_audits[role],
            }
            for role in ("dev", "test")
        },
        "isolation": {
            "dev_test_normalized_query_overlap": len(dev_queries & test_queries),
            "dev_test_scored_target_query_overlap": len(dev_targets & test_targets),
            "dev_test_source_or_distractor_query_overlap": 0,
            "dev_test_full_conversation_overlap": full_conversation_overlap,
            "generic_target_query_overlap_allowed": True,
            "generic_overlap_reason": (
                "short confirmations such as 行就这么办 are naturally reusable; "
                "gold labels remain bound to isolated histories"
            ),
            "query_free_tool_plans": True,
            "external_agent_evaluation_samples_reused": False,
            "frozen_data_used_as_generation_source": False,
            "test_candidate_outputs_opened_before_lock": False,
        },
        "artifacts": {
            role: {
                "dataset": str((args.output_dir / f"{role}.jsonl").relative_to(REPO_ROOT)),
                "dataset_sha256": sha256_file(args.output_dir / f"{role}.jsonl"),
                "target_plan_sha256": sha256_file(
                    args.output_dir / f"target_generation/{role}/plan.jsonl"
                ),
                "accepted_target_sha256": sha256_file(
                    args.output_dir / f"target_generation/{role}/accepted.jsonl"
                ),
                "target_metadata_sha256": sha256_file(
                    args.output_dir / f"target_metadata_{role}.jsonl"
                ),
            }
            for role in ("dev", "test")
        },
    }
    write_json(args.output_dir / "protocol.json", protocol)
    preregistration = {
        "status": "locked_before_candidate_evaluation",
        "protocol_sha256": sha256_file(args.output_dir / "protocol.json"),
        "implementation_sha256": {
            str(path.relative_to(REPO_ROOT)): sha256_file(path)
            for path in (
                PROJECT_ROOT / "agent/reference_selection.py",
                PROJECT_ROOT / "agent/memory.py",
                PROJECT_ROOT / "agent/resolver.py",
                PROJECT_ROOT / "agent/orchestrator.py",
                PROJECT_ROOT / "scripts/agent/evaluate_multiturn_agent.py",
                PROJECT_ROOT / "scripts/agent/run_production_memory_evaluation.py",
            )
        },
        "candidate_labels": [
            "a_stateless",
            "b_direct_concat",
            "c_structured_recent",
            "d_rule_summary",
            "e_long_memory",
            "f_route_precedence",
            "g_request_boundary",
            "h_single_tool_normalization",
            "i_catalog_alias",
            "j_whole_turn_recovery",
            "k_standalone_guard",
            "l_final_cache",
        ],
        "selection_split": "dev",
        "unopened_split": "test",
            "metrics": {
                "replay": "ordered TraceEM on natural_replay targets",
                "no_inherit": "ordered TraceEM on standalone_switch targets",
        },
        "selection_rule": {
            "eligibility": "no_inherit TraceEM may be at most 1.0pp below a_stateless",
            "ranking": [
                "higher replay TraceEM",
                "higher standalone_switch TraceEM",
                "lower p95 input tokens",
                "earlier candidate label",
            ],
        },
        "required_slices": [
            "target_kind",
            "history_gap",
            "reference_family",
            "gold_tool_count",
        ],
        "test_labels_after_lock": [
            "a_stateless",
            "b_direct_concat",
            "c_structured_recent",
            "d_rule_summary",
            "e_long_memory",
            "f_route_precedence",
            "g_request_boundary",
            "h_single_tool_normalization",
            "i_catalog_alias",
            "j_whole_turn_recovery",
            "k_standalone_guard",
            "l_final_cache",
        ],
        "test_use": "one-shot reporting after the selected label is locked; never used for tuning",
    }
    write_json(args.output_dir / "preregistration.json", preregistration)
    print(json.dumps(counts, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("plans", "target-plans", "assemble"), required=True
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=2026081501)
    parser.add_argument("--cases-per-split", type=int, default=CASES_PER_SPLIT)
    parser.add_argument(
        "--additional-protected", type=Path, action="append", default=[]
    )
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.additional_protected = [path.resolve() for path in args.additional_protected]
    if args.stage == "plans":
        return plans_stage(args)
    if args.stage == "target-plans":
        return target_plans_stage(args)
    return assemble_stage(args)


if __name__ == "__main__":
    raise SystemExit(main())
