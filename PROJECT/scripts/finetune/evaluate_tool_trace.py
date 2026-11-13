#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate label-free tool-trace predictions against an authoritative split."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import CATALOG_PATH, catalog_by_id, load_catalog, load_jsonl, split_tools, write_json, write_jsonl  # noqa: E402
from finetune.tool_trace_common import percentile, sha256_file, strict_parse_tools, summary_stats  # noqa: E402


ORDER_MARKERS = [
    "先",
    "再",
    "然后",
    "最后",
    "接着",
    "随后",
    "第一步",
    "第二步",
    "按顺序",
    "依次",
    "收尾",
    "起初",
    "接下来",
    "再之后",
]


def order_identifiable(query: str, gold: list[str], by_id: dict[str, dict[str, Any]]) -> tuple[bool, list[str]]:
    if len(gold) <= 1:
        return False, []
    evidence = [marker for marker in ORDER_MARKERS if marker in query]
    gold_set = set(gold)
    for tool_id in gold:
        for dependency in by_id[tool_id].get("depends_on", []):
            if dependency in gold_set:
                evidence.append(f"depends_on:{dependency}->{tool_id}")
    return bool(evidence), evidence


def finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def json_schema_valid(text: str) -> bool:
    """Check only the exact JSON shape, without validating tool IDs."""
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return False
    return (
        isinstance(payload, dict)
        and set(payload) == {"tools"}
        and isinstance(payload["tools"], list)
        and all(isinstance(tool_id, str) for tool_id in payload["tools"])
    )


def score_subset(cases: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(cases)
    if not count:
        return {"count": 0}
    tp = sum(len(set(case["pred_tools"]) & set(case["gold_tools"])) for case in cases)
    fp = sum(len(set(case["pred_tools"]) - set(case["gold_tools"])) for case in cases)
    fn = sum(len(set(case["gold_tools"]) - set(case["pred_tools"])) for case in cases)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "count": count,
        "set_em": sum(case["set_em"] for case in cases) / count,
        "trace_em": sum(case["trace_em"] for case in cases) / count,
        "json_schema_valid_rate": sum(
            case["json_schema_valid"] for case in cases
        ) / count,
        "json_valid_rate": sum(case["strict_parse_ok"] for case in cases) / count,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
        "omission_rate": sum(case["omission"] for case in cases) / count,
        "over_call_rate": sum(case["extra"] for case in cases) / count,
    }


def bucket_metrics(
    cases: list[dict[str, Any]],
    key_fn: Callable[[dict[str, Any]], list[str]],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        for key in key_fn(case):
            grouped[str(key)].append(case)
    return {key: score_subset(rows) for key, rows in sorted(grouped.items())}


def macro_tool_f1(cases: list[dict[str, Any]], catalog_ids: list[str]) -> tuple[float, dict[str, dict[str, float | int]]]:
    metrics = {}
    for tool_id in catalog_ids:
        tp = sum(tool_id in case["gold_tools"] and tool_id in case["pred_tools"] for case in cases)
        fp = sum(tool_id not in case["gold_tools"] and tool_id in case["pred_tools"] for case in cases)
        fn = sum(tool_id in case["gold_tools"] and tool_id not in case["pred_tools"] for case in cases)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics[tool_id] = {
            "support": tp + fn,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return sum(row["f1"] for row in metrics.values()) / max(1, len(metrics)), metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    parser.add_argument("--metrics-out", type=Path)
    parser.add_argument("--scored-out", type=Path)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--run-id", default="")
    args = parser.parse_args()

    catalog = load_catalog(args.catalog)
    by_id = catalog_by_id(catalog)
    catalog_ids = list(by_id)
    catalog_set = set(catalog_ids)
    gold_rows = load_jsonl(args.gold)
    gold_by_id = {row["id"]: row for row in gold_rows}
    if len(gold_by_id) != len(gold_rows):
        raise SystemExit("duplicate ids in gold split")
    prediction_rows = [
        row
        for row in load_jsonl(args.predictions)
        if int(row.get("sample_index", 0)) == args.sample_index
    ]
    pred_by_id = {row.get("id"): row for row in prediction_rows}
    if len(pred_by_id) != len(prediction_rows):
        raise SystemExit("duplicate prediction ids for selected sample index")
    if set(pred_by_id) != set(gold_by_id):
        missing = sorted(set(gold_by_id) - set(pred_by_id))[:10]
        extra = sorted(set(pred_by_id) - set(gold_by_id))[:10]
        raise SystemExit(f"prediction/gold id mismatch missing={missing} extra={extra}")

    cases = []
    for gold_row in gold_rows:
        prediction = pred_by_id[gold_row["id"]]
        if prediction.get("query") != gold_row["问题"]:
            raise SystemExit(f"query mismatch: {gold_row['id']}")
        view = prediction.get("view")
        candidates = prediction.get("candidate_tools", [])
        candidate_set = set(candidates) if view == "rag" else None
        parsed, strict_ok, strict_error = strict_parse_tools(
            prediction.get("raw_output", ""),
            catalog_set,
            candidate_set,
        )
        gold_tools = split_tools(gold_row["工具"])
        gold_set = set(gold_tools)
        pred_set = set(parsed)
        candidate_recall = gold_set <= set(candidates) if view == "rag" else True
        schema_valid = json_schema_valid(prediction.get("raw_output", ""))
        set_em = strict_ok and pred_set == gold_set
        trace_em = strict_ok and parsed == gold_tools
        identifiable, order_evidence = order_identifiable(gold_row["问题"], gold_tools, by_id)
        omission = bool(gold_set - pred_set)
        extra = bool(pred_set - gold_set)
        unknown = [tool_id for tool_id in parsed if tool_id not in catalog_set]
        escaped = [tool_id for tool_id in parsed if view == "rag" and tool_id not in set(candidates)]
        if not strict_ok:
            error_bucket = "format_miss"
        elif not candidate_recall:
            error_bucket = "retrieval_miss"
        elif not set_em:
            error_bucket = "set_miss"
        elif identifiable and not trace_em:
            error_bucket = "pure_order_miss"
        else:
            error_bucket = "correct"
        case = {
            "id": gold_row["id"],
            "query": gold_row["问题"],
            "gold_tools": gold_tools,
            "pred_tools": parsed,
            "raw_output": prediction.get("raw_output", ""),
            "strict_parse_ok": strict_ok,
            "json_schema_valid": schema_valid,
            "strict_error": strict_error,
            "set_em": set_em,
            "trace_em": trace_em,
            "view": view,
            "candidate_k": prediction.get("candidate_k", len(candidates)),
            "candidate_tools": candidates,
            "candidate_recall": candidate_recall,
            "omission": omission,
            "extra": extra,
            "unknown_tools": unknown,
            "candidate_escape_tools": escaped,
            "order_identifiable": identifiable,
            "order_evidence": order_evidence,
            "unidentifiable_order_difference": set_em and not trace_em and not identifiable,
            "error_bucket": error_bucket,
            "tool_count": int(gold_row.get("tool_count", len(gold_tools))),
            "combo_seen_bucket": gold_row.get("combo_seen_bucket", ""),
            "style": gold_row.get("style", ""),
            "domains": [value for value in str(gold_row.get("domain", "")).split(",") if value],
            "input_tokens": prediction.get("input_tokens", 0),
            "output_tokens": prediction.get("output_tokens", 0),
            "latency_ms": prediction.get("latency_ms", 0.0),
            "prompt_build_latency_ms": prediction.get("prompt_build_latency_ms", 0.0),
            "run_id": prediction.get("run_id", args.run_id),
            "prompt_hash": prediction.get("prompt_hash", ""),
            "input_hash": prediction.get("input_hash", ""),
        }
        cases.append(case)

    overall = score_subset(cases)
    # The PDF rubric calls this “工具关键词匹配准确率”.  Under the
    # canonical tool-ID output protocol, exact keyword-set accuracy is SetEM;
    # micro-F1 is retained for partial keyword overlap.
    overall["keyword_exact_accuracy"] = overall["set_em"]
    overall["keyword_micro_f1"] = overall["micro_f1"]
    overall["strict_output_valid_rate"] = overall["json_valid_rate"]
    macro_f1, per_tool = macro_tool_f1(cases, catalog_ids)
    overall["macro_per_tool_f1"] = macro_f1
    overall["unknown_tool_row_rate"] = sum(bool(case["unknown_tools"]) for case in cases) / len(cases)
    overall["candidate_escape_row_rate"] = sum(bool(case["candidate_escape_tools"]) for case in cases) / len(cases)
    identifiable_set_correct = [
        case for case in cases if case["set_em"] and case["order_identifiable"]
    ]
    overall["identifiable_order_accuracy"] = (
        sum(case["trace_em"] for case in identifiable_set_correct) / len(identifiable_set_correct)
        if identifiable_set_correct
        else None
    )
    overall["identifiable_set_correct_count"] = len(identifiable_set_correct)
    overall["unidentifiable_order_difference_rate"] = sum(
        case["unidentifiable_order_difference"] for case in cases
    ) / len(cases)

    retrieval_success = [case for case in cases if case["candidate_recall"]]
    retrieval_fail = [case for case in cases if not case["candidate_recall"]]
    rag_decomposition = {
        "retriever_upper_bound": sum(case["candidate_recall"] for case in cases) / len(cases),
        "retrieval_success_count": len(retrieval_success),
        "retrieval_fail_count": len(retrieval_fail),
        "conditional_set_accuracy": (
            sum(case["set_em"] for case in retrieval_success) / len(retrieval_success)
            if retrieval_success
            else None
        ),
        "conditional_trace_accuracy": (
            sum(case["trace_em"] for case in retrieval_success) / len(retrieval_success)
            if retrieval_success
            else None
        ),
    }
    error_counts = Counter(case["error_bucket"] for case in cases)
    format_counts = Counter(case["strict_error"] or "valid" for case in cases)
    error_analysis = {
        "mutually_exclusive_priority": [
            "format_miss",
            "retrieval_miss",
            "set_miss",
            "pure_order_miss",
            "correct",
        ],
        "counts": dict(error_counts),
        "rates": {key: value / len(cases) for key, value in error_counts.items()},
        "format_types": dict(format_counts),
        "set_miss_tags": {
            "omission": sum(case["error_bucket"] == "set_miss" and case["omission"] for case in cases),
            "extra": sum(case["error_bucket"] == "set_miss" and case["extra"] for case in cases),
            "substitution": sum(
                case["error_bucket"] == "set_miss" and case["omission"] and case["extra"]
                for case in cases
            ),
        },
    }
    buckets = {
        "single_multi": bucket_metrics(
            cases, lambda case: ["single" if case["tool_count"] == 1 else "multi"]
        ),
        "tool_count": bucket_metrics(cases, lambda case: [str(case["tool_count"])]),
        "combo": bucket_metrics(cases, lambda case: [case["combo_seen_bucket"]]),
        "style": bucket_metrics(
            cases, lambda case: [case["style"]] if case["style"] else []
        ),
        "domain": bucket_metrics(cases, lambda case: case["domains"]),
        "tool": bucket_metrics(cases, lambda case: case["gold_tools"]),
        "candidate_recall": bucket_metrics(
            cases, lambda case: ["success" if case["candidate_recall"] else "fail"]
        ),
        "format": bucket_metrics(
            cases, lambda case: [case["strict_error"] or "valid"]
        ),
    }
    efficiency = {
        "input_tokens": summary_stats(case["input_tokens"] for case in cases),
        "output_tokens": summary_stats(case["output_tokens"] for case in cases),
        "latency_ms_generator_only": summary_stats(case["latency_ms"] for case in cases),
        "latency_ms_prompt_build": summary_stats(
            case["prompt_build_latency_ms"] for case in cases
        ),
        "tokens_per_second": finite(
            sum(case["output_tokens"] for case in cases)
            / (sum(case["latency_ms"] for case in cases) / 1000.0)
            if sum(case["latency_ms"] for case in cases)
            else float("nan")
        ),
    }
    metrics = {
        "status": "complete",
        "run_id": args.run_id or (cases[0]["run_id"] if cases else ""),
        "predictions": str(args.predictions),
        "predictions_sha256": sha256_file(args.predictions),
        "gold": str(args.gold),
        "gold_snapshot_sha256": sha256_file(args.gold),
        "catalog": str(args.catalog),
        "catalog_snapshot_sha256": sha256_file(args.catalog),
        "count": len(cases),
        "sample_index": args.sample_index,
        "view": cases[0]["view"] if cases else "",
        "metric_definitions": {
            "keyword_exact_accuracy": "工具关键词集合完全匹配；等价于当前规范输出下的 SetEM",
            "keyword_micro_f1": "按工具关键词微平均计算的部分匹配 F1",
            "json_schema_valid_rate": (
                "exact JSON object with only a string-array tools field; "
                "tool IDs are not validated"
            ),
            "strict_output_valid_rate": (
                "JSON schema valid, unique IDs, catalog-valid IDs, and for RAG "
                "all IDs remain inside the candidate set"
            ),
            "json_valid_rate": "legacy alias of strict_output_valid_rate",
        },
        "overall": overall,
        "rag_decomposition": rag_decomposition,
        "error_analysis": error_analysis,
        "buckets": buckets,
        "per_tool": per_tool,
        "efficiency": efficiency,
        "prediction_contains_gold_fields": any(
            any("gold" in key.lower() or "工具" in key or "label" in key.lower() for key in row)
            for row in prediction_rows
        ),
    }
    if metrics["prediction_contains_gold_fields"]:
        raise SystemExit("prediction file contains forbidden gold/label fields")
    args.metrics_out = args.metrics_out or args.predictions.with_suffix(".metrics.json")
    args.scored_out = args.scored_out or args.predictions.with_suffix(".scored.jsonl")
    write_json(args.metrics_out, metrics)
    write_jsonl(args.scored_out, cases)
    print(
        f"{metrics['run_id']} N={len(cases)} SetEM={overall['set_em']:.6f} "
        f"TraceEM={overall['trace_em']:.6f} JSON={overall['json_valid_rate']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
