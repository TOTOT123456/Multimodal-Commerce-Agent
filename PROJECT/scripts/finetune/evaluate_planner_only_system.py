#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate a tool-only Planner on frozen 9K without an Intent or chat response path."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import (  # noqa: E402
    CATALOG_PATH,
    INTENT_DIR,
    load_catalog,
    load_jsonl,
    split_tools,
    write_json,
    write_jsonl,
)
from finetune.tool_trace_common import sha256_file, strict_parse_tools  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intent-gold", type=Path, default=INTENT_DIR / "intent_all.jsonl")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--ecommerce-predictions", type=Path)
    parser.add_argument("--chitchat-predictions", type=Path)
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metrics-out", type=Path, required=True)
    args = parser.parse_args()

    gold_rows = load_jsonl(args.intent_gold)
    gold_by_id = {str(row["id"]): row for row in gold_rows}
    if len(gold_by_id) != len(gold_rows):
        raise SystemExit("duplicate ids in gold")
    if args.predictions:
        if args.ecommerce_predictions or args.chitchat_predictions:
            raise SystemExit("use either --predictions or the two split prediction files")
        prediction_rows = [
            row
            for row in load_jsonl(args.predictions)
            if int(row.get("sample_index", 0)) == 0
        ]
        pred_by_id = {str(row["id"]): row for row in prediction_rows}
        if len(pred_by_id) != len(prediction_rows) or set(gold_by_id) != set(pred_by_id):
            missing = sorted(set(gold_by_id) - set(pred_by_id))[:10]
            extra = sorted(set(pred_by_id) - set(gold_by_id))[:10]
            raise SystemExit(f"prediction/gold id mismatch missing={missing} extra={extra}")
        prediction_for = lambda gold: pred_by_id[str(gold["id"])]
        prediction_inputs = {str(args.predictions): sha256_file(args.predictions)}
    else:
        if not args.ecommerce_predictions or not args.chitchat_predictions:
            raise SystemExit("split mode requires both ecommerce and chitchat predictions")
        ecommerce_rows = load_jsonl(args.ecommerce_predictions)
        chitchat_rows = load_jsonl(args.chitchat_predictions)
        ecommerce_by_id = {str(row["id"]): row for row in ecommerce_rows}
        chitchat_by_id = {str(row["id"]): row for row in chitchat_rows}
        expected_ecommerce = {
            str(row["trace_id"]) for row in gold_rows if row["类型"] == "电商类"
        }
        expected_chitchat = {
            str(row["id"]) for row in gold_rows if row["类型"] == "闲聊类"
        }
        if set(ecommerce_by_id) != expected_ecommerce or set(chitchat_by_id) != expected_chitchat:
            raise SystemExit("split prediction ids do not match their frozen subsets")
        prediction_for = lambda gold: (
            ecommerce_by_id[str(gold["trace_id"])]
            if gold["类型"] == "电商类"
            else chitchat_by_id[str(gold["id"])]
        )
        prediction_inputs = {
            str(args.ecommerce_predictions): sha256_file(args.ecommerce_predictions),
            str(args.chitchat_predictions): sha256_file(args.chitchat_predictions),
        }

    catalog_ids = {str(row["tool_id"]) for row in load_catalog(args.catalog)}
    cases: list[dict[str, Any]] = []
    chitchat_count = 0
    chitchat_correct = 0
    ecommerce_count = 0
    ecommerce_set_correct = 0
    ecommerce_trace_correct = 0
    strict_count = 0
    chitchat_empty_tool_count = 0
    diagnostic_route_correct_count = 0

    for gold in gold_rows:
        prediction = prediction_for(gold)
        if prediction.get("query") != gold["问题"]:
            raise SystemExit(f"query mismatch: {gold['id']}")
        candidates = prediction.get("candidate_tools", [])
        candidate_ids = set(candidates) if prediction.get("view") == "rag" else None
        parsed, strict_ok, strict_error = strict_parse_tools(
            prediction.get("raw_output", ""), catalog_ids, candidate_ids
        )
        strict_count += int(strict_ok)

        is_ecommerce = gold["类型"] == "电商类"
        gold_tools = split_tools(gold.get("工具")) if is_ecommerce else []
        diagnostic_inferred_route = "闲聊类" if strict_ok and not parsed else "电商类"
        diagnostic_route_correct = diagnostic_inferred_route == gold["类型"]
        diagnostic_route_correct_count += int(diagnostic_route_correct)

        if is_ecommerce:
            set_correct = strict_ok and set(parsed) == set(gold_tools)
            trace_correct = strict_ok and parsed == gold_tools
            ecommerce_count += 1
            ecommerce_set_correct += int(set_correct)
            ecommerce_trace_correct += int(trace_correct)
        else:
            # A tool-only Planner has no explicit route or chat response head.  An
            # empty tool list is therefore a useful abstention diagnostic, not a
            # successful response to a chitchat request.
            set_correct = False
            trace_correct = False
            chitchat_count += 1
            chitchat_empty_tool_count += int(strict_ok and not parsed)

        cases.append(
            {
                "id": gold["id"],
                "query": gold["问题"],
                "gold_route": gold["类型"],
                "pred_route": None,
                "route_correct": None,
                "explicit_route_supported": False,
                "chitchat_response_supported": False,
                "diagnostic_inferred_route": diagnostic_inferred_route,
                "diagnostic_inferred_route_correct": diagnostic_route_correct,
                "gold_tools": gold_tools,
                "pred_tools": parsed,
                "strict_parse_ok": strict_ok,
                "strict_error": strict_error,
                "tool_set_correct": set_correct,
                "tool_trace_correct": trace_correct,
                "full_pipeline_success": set_correct,
                "full_pipeline_trace_success": trace_correct,
            }
        )

    total = len(cases)
    full_set_count = sum(bool(row["full_pipeline_success"]) for row in cases)
    full_trace_count = sum(bool(row["full_pipeline_trace_success"]) for row in cases)
    write_jsonl(args.out, cases)
    metrics = {
        "status": "complete",
        "scope": "tool_only_planner_without_intent_or_chitchat_response_path",
        "is_real_tool_execution_success": False,
        "intent_model_used": False,
        "explicit_route_supported": False,
        "chitchat_response_supported": False,
        "scoring_contract": (
            "all chitchat requests fail because the system has no chitchat response path; "
            "strict empty tool output is diagnostic only"
        ),
        "diagnostic_route_definition": (
            "strict empty tool list is inferred as chitchat only for abstention diagnostics; "
            "every non-empty or invalid output is inferred as ecommerce"
        ),
        "count": total,
        "route_accuracy": None,
        "route_correct_count": None,
        "diagnostic_inferred_route_accuracy": diagnostic_route_correct_count / total,
        "diagnostic_inferred_route_correct_count": diagnostic_route_correct_count,
        "chitchat_count": chitchat_count,
        "chitchat_correct": chitchat_correct,
        "chitchat_accuracy": chitchat_correct / chitchat_count,
        "chitchat_empty_tool_count": chitchat_empty_tool_count,
        "chitchat_empty_tool_rate": chitchat_empty_tool_count / chitchat_count,
        "chitchat_tool_activation_or_invalid_count": chitchat_count - chitchat_empty_tool_count,
        "chitchat_tool_activation_or_invalid_rate": 1.0 - chitchat_empty_tool_count / chitchat_count,
        "ecommerce_count": ecommerce_count,
        "ecommerce_set_correct": ecommerce_set_correct,
        "ecommerce_set_em": ecommerce_set_correct / ecommerce_count,
        "ecommerce_trace_correct": ecommerce_trace_correct,
        "ecommerce_trace_em": ecommerce_trace_correct / ecommerce_count,
        "strict_output_valid_count": strict_count,
        "strict_output_valid_rate": strict_count / total,
        "full_pipeline_success_count": full_set_count,
        "full_pipeline_success": full_set_count / total,
        "full_pipeline_trace_success_count": full_trace_count,
        "full_pipeline_trace_success": full_trace_count / total,
        "inputs": {
            str(args.intent_gold): sha256_file(args.intent_gold),
            str(args.catalog): sha256_file(args.catalog),
            **prediction_inputs,
        },
        "predictions": str(args.out),
        "predictions_sha256": sha256_file(args.out),
    }
    write_json(args.metrics_out, metrics)
    print(
        f"planner-only diagnostic_route={metrics['diagnostic_inferred_route_accuracy']:.6f} "
        f"chitchat_contract={metrics['chitchat_accuracy']:.6f} "
        f"chitchat_empty={metrics['chitchat_empty_tool_rate']:.6f} "
        f"ecommerce_TraceEM={metrics['ecommerce_trace_em']:.6f} "
        f"full_trace={metrics['full_pipeline_trace_success']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
