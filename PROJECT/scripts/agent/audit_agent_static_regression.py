#!/usr/bin/env python3
"""校验多轮层的首轮恒等性与自然短会话评测证据。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from PROJECT.agent.orchestrator import ToolPlanningAgent  # noqa: E402
from PROJECT.agent.resolver import (  # noqa: E402
    ContextRouter,
    DeterministicResolver,
    explicit_turn_references,
    is_first_turn_standalone,
)
from PROJECT.agent.types import CompactToolState  # noqa: E402
from PROJECT.scripts.agent.evaluate_multiturn_agent import sha256_file  # noqa: E402
from PROJECT.scripts.finetune.tool_trace_common import sha256_json  # noqa: E402
from PROJECT.scripts.project_common import (  # noqa: E402
    catalog_by_id,
    load_catalog,
    load_jsonl,
    write_json,
)

PROJECT_ROOT = REPO_ROOT / "PROJECT"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data/agent/_aux/natural_short_memory_evaluation/validation/static_regression.json"
)
METRICS = {
    "legacy6000": PROJECT_ROOT
    / "data/tool_call_trace/_aux/assignment3/assignment3_planner_dpo2_20260808/"
    "evaluation/frozen/legacy6000/selected/rag.metrics.json",
    "api1000": PROJECT_ROOT
    / "data/tool_call_trace/_aux/assignment3/assignment3_planner_dpo2_20260808/"
    "evaluation/frozen/api1000/selected/rag.metrics.json",
}


def audit_split(metrics_path: Path, runtime: dict, by_id: dict) -> dict:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    prediction_path = REPO_ROOT / str(metrics["predictions"])
    if sha256_file(prediction_path) != str(metrics["predictions_sha256"]):
        raise ValueError(f"canonical prediction hash differs: {prediction_path}")
    rows = load_jsonl(prediction_path)
    counts: Counter[str] = Counter()
    changed = []
    router = ContextRouter(
        prefer_recent_over_generic_long=bool(
            runtime["planning"]["recent_over_generic_long"]
        ),
        natural_request_boundary_guard=bool(
            runtime["planning"]["natural_request_boundary_guard"]
        ),
    )
    resolver = DeterministicResolver(
        natural_request_boundary_guard=bool(
            runtime["planning"]["natural_request_boundary_guard"]
        )
    )
    alias_matcher = object.__new__(ToolPlanningAgent)
    alias_matcher.catalog_by_id = by_id
    for row in rows:
        query = str(row["query"])
        mode = router.route(query)
        resolution = resolver.resolve(
            query,
            turn_id=1,
            mode=mode,
            state=CompactToolState(conversation_id="static-audit"),
            recent_events=[],
            memory_hits=[],
        )
        counts[f"initial_mode_{mode}"] += 1
        if mode != "none" and is_first_turn_standalone(
            query, resolution.interpretation.current_goals
        ):
            counts["first_turn_standalone_fallback"] += 1
            mode = "none"
            resolution = resolver.resolve(
                query,
                turn_id=1,
                mode=mode,
                state=CompactToolState(conversation_id="static-audit"),
                recent_events=[],
                memory_hits=[],
            )
        if (
            runtime["planning"]["standalone_multigoal_guard"]
            and mode != "none"
            and not explicit_turn_references(query)
            and len(resolution.interpretation.current_goals) >= 2
            and not resolution.interpretation.carried_goals
            and not resolution.interpretation.cancelled_goals
        ):
            counts["standalone_multigoal_fallback"] += 1
            mode = "none"
            resolution = resolver.resolve(
                query,
                turn_id=1,
                mode=mode,
                state=CompactToolState(conversation_id="static-audit"),
                recent_events=[],
                memory_hits=[],
            )
        final_query = resolution.query_variants[0]
        counts["raw_query_identity"] += final_query == query
        counts["unresolved_reference"] += (
            resolution.interpretation.unresolved_reference
        )
        direct = alias_matcher._catalog_alias_tools(
            resolution.interpretation.resolved_goals,
            sorted(by_id),
        )
        counts["catalog_alias_direct_activation"] += direct is not None
        if final_query != query:
            changed.append(
                {
                    "id": row["id"],
                    "query": query,
                    "final_query": final_query,
                    "mode": mode,
                }
            )
    strict_ok = (
        counts["raw_query_identity"] == len(rows)
        and counts["unresolved_reference"] == 0
        and counts["catalog_alias_direct_activation"] == 0
        and not changed
    )
    if not strict_ok:
        raise SystemExit(f"static transformation regression failed: {changed[:3]}")
    return {
        "rows": len(rows),
        "initial_modes": {
            mode: counts[f"initial_mode_{mode}"]
            for mode in ("none", "recent", "long")
        },
        "first_turn_standalone_fallback": counts[
            "first_turn_standalone_fallback"
        ],
        "standalone_multigoal_fallback": counts["standalone_multigoal_fallback"],
        "raw_query_identity_count": counts["raw_query_identity"],
        "raw_query_identity_rate": counts["raw_query_identity"] / len(rows),
        "unresolved_reference_count": counts["unresolved_reference"],
        "catalog_alias_direct_activation_count": counts[
            "catalog_alias_direct_activation"
        ],
        "canonical_metrics": metrics["overall"],
        "canonical_rag_decomposition": metrics["rag_decomposition"],
        "evidence": {
            "metrics": str(metrics_path.relative_to(REPO_ROOT)),
            "metrics_sha256": sha256_file(metrics_path),
            "predictions": str(prediction_path.relative_to(REPO_ROOT)),
            "predictions_sha256": sha256_file(prediction_path),
        },
        "strict_ok": strict_ok,
    }


def audit_natural_short_memory(runtime: dict) -> dict:
    selected = runtime["selection_status"]
    protocol_path = REPO_ROOT / str(selected["protocol"])
    results_path = REPO_ROOT / str(selected["results"])
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    results = json.loads(results_path.read_text(encoding="utf-8"))
    label = str(selected["selected_label"])
    development = next(
        row for row in results["selection"]["candidate_summaries"]
        if row["label"] == label
    )
    test = next(
        row for row in results["confirmation"]["candidate_summaries"]
        if row["label"] == label
    )
    expected_dev = selected["development"]
    expected_test = selected["test"]
    checks = {
        "protocol_complete": protocol["status"] == "built_before_candidate_evaluation",
        "results_complete": results["status"] == "complete",
        "selected_label": results["selection"]["selected_label"] == label,
        "formal_runs": results["execution"]["formal_runs"] == 24,
        "prediction_rows": results["execution"]["prediction_rows"] == 67536,
        "backend_errors": results["execution"]["aggregate_status_counts"].get(
            "backend_error", 0
        ) == 0,
        "development_mixed": development["targets"]["correct"]
        == expected_dev["mixed_correct"],
        "development_replay": development["replay"]["correct"]
        == expected_dev["replay_correct"],
        "development_no_inherit": development["no_inherit"]["correct"]
        == expected_dev["no_inherit_correct"],
        "test_mixed": test["targets"]["correct"] == expected_test["mixed_correct"],
        "test_replay": test["replay"]["correct"] == expected_test["replay_correct"],
        "test_no_inherit": test["no_inherit"]["correct"]
        == expected_test["no_inherit_correct"],
        "test_planner_calls": test["targets"]["planner_model_calls"]
        == expected_test["planner_calls"],
    }
    strict_ok = all(checks.values())
    if not strict_ok:
        raise SystemExit(f"natural short memory evidence differs: {checks}")
    return {
        "selected_label": label,
        "development": development,
        "test": test,
        "checks": checks,
        "evidence": {
            "protocol": str(protocol_path.relative_to(REPO_ROOT)),
            "protocol_sha256": sha256_file(protocol_path),
            "results": str(results_path.relative_to(REPO_ROOT)),
            "results_sha256": sha256_file(results_path),
        },
        "strict_ok": strict_ok,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    runtime = json.loads(
        (PROJECT_ROOT / "configs/agent_runtime.json").read_text(encoding="utf-8")
    )
    by_id = catalog_by_id(load_catalog(PROJECT_ROOT / "configs/tool_catalog.jsonl"))
    splits = {
        name: audit_split(path, runtime, by_id) for name, path in METRICS.items()
    }
    memory_evaluation = audit_natural_short_memory(runtime)
    result = {
        "status": "complete",
        "canonical_runtime_config_hash": sha256_json(runtime),
        "selected_config_id": runtime["selection_status"]["selected_config_id"],
        "method": (
            "Execute the selected first-turn resolver and exact catalog guard for "
            "every frozen query; downstream predictions remain reusable only when "
            "the final query is byte-identical and no direct guard activates."
        ),
        "model_inference_required": False,
        "splits": splits,
        "natural_short_memory_evaluation": memory_evaluation,
        "strict_ok": all(row["strict_ok"] for row in splits.values())
        and memory_evaluation["strict_ok"],
    }
    write_json(args.output, result)
    print(
        "static regression identity: "
        + ", ".join(
            f"{name}={row['raw_query_identity_count']}/{row['rows']}"
            for name, row in splits.items()
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
