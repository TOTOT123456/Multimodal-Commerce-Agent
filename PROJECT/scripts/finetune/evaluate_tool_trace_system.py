#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate the intent gate plus final frozen tool generator as one system."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import INTENT_DIR, load_jsonl, split_tools, write_json, write_jsonl  # noqa: E402
from finetune.tool_trace_common import sha256_file, strict_parse_tools  # noqa: E402
from project_common import CATALOG_PATH, load_catalog  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intent-gold", type=Path, default=INTENT_DIR / "intent_all.jsonl")
    parser.add_argument("--intent-predictions", type=Path, required=True)
    parser.add_argument("--tool-predictions", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metrics-out", type=Path, required=True)
    args = parser.parse_args()

    intent_gold = load_jsonl(args.intent_gold)
    intent_pred = load_jsonl(args.intent_predictions)
    tool_pred = load_jsonl(args.tool_predictions)
    gold_by_id = {row["id"]: row for row in intent_gold}
    intent_by_id = {row["id"]: row for row in intent_pred}
    if set(gold_by_id) != set(intent_by_id):
        raise SystemExit("intent prediction ids do not match intent frozen set")
    tool_by_trace = {row["id"]: row for row in tool_pred}
    if len(tool_by_trace) != len(tool_pred):
        raise SystemExit("duplicate tool prediction ids")
    catalog_ids = {row["tool_id"] for row in load_catalog(args.catalog)}

    output = []
    route_correct_count = 0
    false_activation = 0
    chitchat_count = 0
    chitchat_route_correct = 0
    ecommerce_count = 0
    ecommerce_set_correct = 0
    ecommerce_trace_correct = 0
    full_success = 0
    full_trace_success = 0
    missing_downstream = []
    for gold in intent_gold:
        route = intent_by_id[gold["id"]].get("预测类型", "")
        route_correct = route == gold["类型"]
        route_correct_count += int(route_correct)
        is_ecommerce = gold["类型"] == "电商类"
        false_activation += int(not is_ecommerce and route == "电商类")
        if not is_ecommerce:
            chitchat_count += 1
            chitchat_route_correct += int(route_correct)
        downstream: dict[str, Any] | None = None
        set_correct = None
        trace_correct = None
        if route == "电商类" and is_ecommerce:
            trace_id = gold.get("trace_id", "")
            downstream = tool_by_trace.get(trace_id)
            if not downstream:
                missing_downstream.append({"intent_id": gold["id"], "trace_id": trace_id})
            else:
                candidates = downstream.get("candidate_tools", [])
                parsed, valid, _ = strict_parse_tools(
                    downstream.get("raw_output", ""),
                    catalog_ids,
                    set(candidates) if downstream.get("view") == "rag" else None,
                )
                gold_tools = split_tools(gold.get("工具"))
                set_correct = valid and set(parsed) == set(gold_tools)
                trace_correct = valid and parsed == gold_tools
                ecommerce_set_correct += int(set_correct)
                ecommerce_trace_correct += int(trace_correct)
        if is_ecommerce:
            ecommerce_count += 1
            case_success = route_correct and bool(set_correct)
            case_trace_success = route_correct and bool(trace_correct)
        else:
            case_success = route_correct
            case_trace_success = route_correct
        full_success += int(case_success)
        full_trace_success += int(case_trace_success)
        output.append(
            {
                "id": gold["id"],
                "query": gold["问题"],
                "gold_route": gold["类型"],
                "pred_route": route,
                "route_correct": route_correct,
                "trace_id": gold.get("trace_id", ""),
                "gold_tools": split_tools(gold.get("工具")) if is_ecommerce else [],
                "tool_raw_output": downstream.get("raw_output", "") if downstream else "",
                "pred_tools": downstream.get("pred_tools", []) if downstream else [],
                "tool_set_correct": set_correct,
                "tool_trace_correct": trace_correct,
                "full_pipeline_success": case_success,
                "full_pipeline_trace_success": case_trace_success,
            }
        )
    if missing_downstream:
        raise SystemExit(f"missing downstream outputs for routed ecommerce cases: {missing_downstream[:10]}")
    write_jsonl(args.out, output)
    metrics = {
        "status": "complete",
        "scope": "static_routing_plus_tool_trace_selection",
        "is_real_tool_execution_success": False,
        "count": len(output),
        "route_accuracy": route_correct_count / len(output),
        "route_correct_count": route_correct_count,
        "chitchat_count": chitchat_count,
        "chitchat_route_correct": chitchat_route_correct,
        "chitchat_route_accuracy": chitchat_route_correct / max(1, chitchat_count),
        "false_tool_activation": false_activation
        / max(1, chitchat_count),
        "ecommerce_count": ecommerce_count,
        "ecommerce_set_correct": ecommerce_set_correct,
        "ecommerce_set_em": ecommerce_set_correct / ecommerce_count,
        "ecommerce_trace_correct": ecommerce_trace_correct,
        "ecommerce_trace_em": ecommerce_trace_correct / ecommerce_count,
        "full_pipeline_success_count": full_success,
        "full_pipeline_success": full_success / len(output),
        "full_pipeline_trace_success_count": full_trace_success,
        "full_pipeline_trace_success": full_trace_success / len(output),
        "definitions": {
            "full_pipeline_success": (
                "chitchat: route correct; ecommerce: route correct and tool SetEM"
            ),
            "full_pipeline_trace_success": (
                "chitchat: route correct; ecommerce: route correct and ordered tool TraceEM"
            ),
        },
        "routing_model_runs": sorted(
            {str(row.get("run_name", "")) for row in intent_pred if row.get("run_name")}
        ),
        "tool_generator": {
            "run_ids": sorted({str(row.get("run_id", "")) for row in tool_pred}),
            "models": sorted({str(row.get("model", "")) for row in tool_pred}),
            "adapters": sorted({str(row.get("adapter", "")) for row in tool_pred}),
            "views": sorted({str(row.get("view", "")) for row in tool_pred}),
            "candidate_ks": sorted({int(row.get("candidate_k", 0)) for row in tool_pred}),
        },
        "inputs": {
            str(args.intent_gold): sha256_file(args.intent_gold),
            str(args.intent_predictions): sha256_file(args.intent_predictions),
            str(args.tool_predictions): sha256_file(args.tool_predictions),
        },
        "predictions": str(args.out),
        "predictions_sha256": sha256_file(args.out),
    }
    write_json(args.metrics_out, metrics)
    print(
        f"system route={metrics['route_accuracy']:.6f} false_activation={metrics['false_tool_activation']:.6f} "
        f"ecommerce_SetEM={metrics['ecommerce_set_em']:.6f} "
        f"ecommerce_TraceEM={metrics['ecommerce_trace_em']:.6f} "
        f"full_set={metrics['full_pipeline_success']:.6f} "
        f"full_trace={metrics['full_pipeline_trace_success']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
