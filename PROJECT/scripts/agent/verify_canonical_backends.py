#!/usr/bin/env python3
"""抽样校验 Intent、RAG 与 Planner 后端没有接口漂移。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from PROJECT.agent.backends import (  # noqa: E402
    CanonicalIntentBackend,
    CanonicalPlannerBackend,
    CanonicalRoute,
    CanonicalToolRetriever,
    load_jsonl,
    resolve_project_reference,
    sha256_file,
)
from PROJECT.agent.factory import (  # noqa: E402
    DEFAULT_CONFIG,
    PROJECT_ROOT,
    load_agent_config,
    resolve_config_path,
)
from PROJECT.agent.types import utc_now  # noqa: E402
from PROJECT.scripts.project_common import catalog_by_id, load_catalog  # noqa: E402


DEFAULT_RETRIEVAL_REFERENCE = (
    PROJECT_ROOT
    / "data/rag/_aux/current_canonical_dataset_retrieval/"
    "planner_dpo2_dev_20260808/final/preds.jsonl"
)
DEFAULT_INTENT_REFERENCE = (
    PROJECT_ROOT
    / "finetune/outputs/preds/qwen3-0.6b-full_lora-context_isolated-best.jsonl"
)
DEFAULT_PLANNER_REFERENCE = (
    PROJECT_ROOT
    / "data/tool_call_trace/_aux/assignment3/"
    "assignment3_planner_dpo2_20260808/evaluation/dev/"
    "ext_dpo_5_s113/rag.scored.jsonl"
)


def mismatch(row_id: str, field: str, expected: Any, actual: Any) -> dict[str, Any]:
    return {
        "id": row_id,
        "field": field,
        "expected": expected,
        "actual": actual,
    }


def verify_retrieval(
    backend: CanonicalToolRetriever,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    latencies: list[float] = []
    for row in rows:
        row_id = str(row["id"])
        result = backend.retrieve(str(row["query"]))
        latencies.append(result.latency_ms)
        expected = {
            "source_tools": [str(value) for value in row["source_tools"]],
            "reranked_tools": [str(value) for value in row["reranked_tools"]],
            "candidate_tools": [str(value) for value in row["pred_tools"][:10]],
        }
        actual = {
            "source_tools": result.source_tools,
            "reranked_tools": result.reranked_tools,
            "candidate_tools": result.candidate_tools,
        }
        for field in expected:
            if expected[field] != actual[field]:
                failures.append(mismatch(row_id, field, expected[field], actual[field]))
    return {
        "checked": len(rows),
        "exact": len(rows) - len({value["id"] for value in failures}),
        "mismatch_count": len(failures),
        "latency_ms_mean": sum(latencies) / len(latencies) if latencies else 0.0,
        "mismatches": failures,
    }


def verify_intent(
    backend: CanonicalIntentBackend,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    latencies: list[float] = []
    for row in rows:
        row_id = str(row["id"])
        result = backend.classify(str(row["问题"]))
        latencies.append(result.latency_ms)
        expected = {
            "intent_type": str(row["预测类型"]),
            "keywords": str(row["预测关键词"]),
        }
        actual = {
            "intent_type": result.intent_type,
            "keywords": result.keywords,
        }
        for field in expected:
            if expected[field] != actual[field]:
                failures.append(mismatch(row_id, field, expected[field], actual[field]))
    return {
        "checked": len(rows),
        "exact": len(rows) - len({value["id"] for value in failures}),
        "mismatch_count": len(failures),
        "latency_ms_mean": sum(latencies) / len(latencies) if latencies else 0.0,
        "mismatches": failures,
    }


def verify_planner(
    backend: CanonicalPlannerBackend,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    latencies: list[float] = []
    for row in rows:
        row_id = str(row["id"])
        query = str(row["query"])
        candidates = [str(value) for value in row["candidate_tools"]]
        counted_tokens = backend.count_input_tokens(query, candidates)
        result = backend.plan(query, candidates)
        latencies.append(result.latency_ms)
        expected = {
            "counted_input_tokens": int(row["input_tokens"]),
            "inference_input_tokens": int(row["input_tokens"]),
            "tools": [str(value) for value in row["pred_tools"]],
            "raw_output": str(row["raw_output"]),
            "strict_ok": bool(row["strict_parse_ok"]),
            "strict_error": str(row["strict_error"]),
            "prompt_hash": str(row["prompt_hash"]),
        }
        actual = {
            "counted_input_tokens": counted_tokens,
            "inference_input_tokens": result.input_tokens,
            "tools": result.tools,
            "raw_output": result.raw_output,
            "strict_ok": result.strict_ok,
            "strict_error": result.strict_error,
            "prompt_hash": result.prompt_hash,
        }
        for field in expected:
            if expected[field] != actual[field]:
                failures.append(mismatch(row_id, field, expected[field], actual[field]))
    return {
        "checked": len(rows),
        "exact": len(rows) - len({value["id"] for value in failures}),
        "mismatch_count": len(failures),
        "latency_ms_mean": sum(latencies) / len(latencies) if latencies else 0.0,
        "mismatches": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--intent-reference", type=Path, default=DEFAULT_INTENT_REFERENCE)
    parser.add_argument("--intent-limit", type=int, default=20)
    parser.add_argument(
        "--retrieval-reference", type=Path, default=DEFAULT_RETRIEVAL_REFERENCE
    )
    parser.add_argument("--retrieval-limit", type=int, default=20)
    parser.add_argument("--planner-reference", type=Path, default=DEFAULT_PLANNER_REFERENCE)
    parser.add_argument("--planner-limit", type=int, default=1)
    parser.add_argument("--intent-device", default="cuda:0")
    parser.add_argument("--embedding-device", default="cuda:0")
    parser.add_argument("--reranker-device", default="cuda:1")
    parser.add_argument("--planner-device", default="cuda:2")
    parser.add_argument("--planner-base-model", default="")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.intent_limit < 0 or args.retrieval_limit < 0 or args.planner_limit < 0:
        parser.error("sample limits must be non-negative")

    config = load_agent_config(args.config)
    route_path = resolve_config_path(str(config["canonical_route"]))
    route = CanonicalRoute.load(
        PROJECT_ROOT,
        route_path,
        required_revision=str(config["required_canonical_revision"]),
    )
    if route.sha256 != str(config["canonical_route_sha256"]):
        raise SystemExit("Agent config and canonical route hash differ")

    intent_rows = load_jsonl(args.intent_reference)[: args.intent_limit]
    retrieval_rows = load_jsonl(args.retrieval_reference)[: args.retrieval_limit]
    planner_rows = load_jsonl(args.planner_reference)[: args.planner_limit]
    if len(intent_rows) != args.intent_limit:
        raise SystemExit("Intent reference has fewer rows than requested")
    if len(retrieval_rows) != args.retrieval_limit:
        raise SystemExit("retrieval reference has fewer rows than requested")
    if len(planner_rows) != args.planner_limit:
        raise SystemExit("Planner reference has fewer rows than requested")

    catalog = catalog_by_id(
        load_catalog(
            Path(
                resolve_project_reference(
                    PROJECT_ROOT, str(route.payload["catalog"]["path"])
                )
            )
        )
    )
    intent = CanonicalIntentBackend(route, device=args.intent_device)
    retriever = CanonicalToolRetriever(
        route,
        embedding_device=args.embedding_device,
        reranker_device=args.reranker_device,
    )
    planner = CanonicalPlannerBackend(
        route,
        catalog,
        device=args.planner_device,
        base_model=args.planner_base_model or None,
    )
    intent_result = verify_intent(intent, intent_rows)
    retrieval_result = verify_retrieval(retriever, retrieval_rows)
    planner_result = verify_planner(planner, planner_rows)
    payload = {
        "schema_version": 1,
        "kind": "engineering_backend_parity_check",
        "generated_at": utc_now(),
        "canonical_revision": route.revision,
        "canonical_route_sha256": route.sha256,
        "devices": {
            "intent": args.intent_device,
            "embedding": args.embedding_device,
            "reranker": args.reranker_device,
            "planner": args.planner_device,
        },
        "references": {
            "intent": {
                "path": str(args.intent_reference.relative_to(REPO_ROOT)),
                "sha256": sha256_file(args.intent_reference),
            },
            "retrieval": {
                "path": str(args.retrieval_reference.relative_to(REPO_ROOT)),
                "sha256": sha256_file(args.retrieval_reference),
            },
            "planner": {
                "path": str(args.planner_reference.relative_to(REPO_ROOT)),
                "sha256": sha256_file(args.planner_reference),
            },
        },
        "intent": intent_result,
        "retrieval": retrieval_result,
        "planner": planner_result,
        "strict_ok": (
            intent_result["mismatch_count"] == 0
            and retrieval_result["mismatch_count"] == 0
            and planner_result["mismatch_count"] == 0
        ),
        "scope_note": (
            "少量历史样本的在线实现复现检查，不是新的模型质量或延迟评测；"
            "耗时只用于发现异常。"
        ),
    }
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if payload["strict_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
