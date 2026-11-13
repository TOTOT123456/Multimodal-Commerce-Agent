#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit the standalone API-native Assignment3 evaluation split."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import (  # noqa: E402
    catalog_by_id,
    load_catalog,
    load_jsonl,
    normalize_question,
    split_tools,
    write_json,
)
from dataset.generate_diverse_tool_trace import catalog_term_index, safe_text  # noqa: E402
from finetune.tool_trace_common import sha256_file  # noqa: E402


INTERNAL_RE = re.compile(
    r"tool[_ -]?id|工具\s*id|catalog|rag|api|mcp|候选工具|调用工具|内部字段|实现逻辑",
    re.I,
)


def normalized_queries(rows: list[dict[str, Any]]) -> set[str]:
    return {normalize_question(str(row["问题"])) for row in rows}


def rule_metrics(
    rows: list[dict[str, Any]],
    catalog: list[dict[str, Any]],
    signal_builder: Callable[[dict[str, Any]], list[str]],
) -> dict[str, Any]:
    set_hits = 0
    trace_hits = 0
    all_gold_hits = 0
    any_prediction = 0
    predicted_counts = []
    for row in rows:
        query = normalize_question(str(row["问题"]))
        matched: list[tuple[int, int, str]] = []
        for catalog_index, tool in enumerate(catalog):
            positions = [
                query.find(signal)
                for signal in signal_builder(tool)
                if signal and signal in query
            ]
            if positions:
                matched.append((min(positions), catalog_index, str(tool["tool_id"])))
        prediction = [
            tool_id for _, _, tool_id in sorted(matched, key=lambda item: (item[0], item[1]))
        ]
        gold = split_tools(row["工具"])
        set_hits += set(prediction) == set(gold)
        trace_hits += prediction == gold
        all_gold_hits += set(gold) <= set(prediction)
        any_prediction += bool(prediction)
        predicted_counts.append(len(prediction))
    count = max(1, len(rows))
    return {
        "rows": len(rows),
        "set_em": set_hits / count,
        "trace_em": trace_hits / count,
        "all_gold_covered": all_gold_hits / count,
        "rows_with_any_prediction": any_prediction / count,
        "mean_predicted_tools": sum(predicted_counts) / count,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--target-plan", type=Path, required=True)
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--generation-audit", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--generation-meta", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=2400)
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    plan = load_jsonl(args.target_plan)
    accepted = load_jsonl(args.accepted)
    train = load_jsonl(args.train)
    test = load_jsonl(args.test)
    meta = load_jsonl(args.generation_meta)
    catalog = load_catalog(args.catalog)
    by_id = catalog_by_id(catalog)
    generation_audit = json.loads(args.generation_audit.read_text(encoding="utf-8"))

    ids = [str(row["id"]) for row in rows]
    questions = [normalize_question(str(row["问题"])) for row in rows]
    plan_by_id = {str(row["id"]): row for row in plan}
    accepted_by_id = {str(row["id"]): row for row in accepted}
    alignment_errors = []
    forbidden_provenance_fields = []
    for row in rows:
        trace_id = str(row["id"])
        target = plan_by_id.get(trace_id)
        accepted_row = accepted_by_id.get(trace_id)
        reasons = []
        if "parent_id" in row:
            forbidden_provenance_fields.append(trace_id)
        if target is None:
            reasons.append("missing_target_plan")
        else:
            target_tools = [str(value) for value in target.get("tools", [])]
            if split_tools(row["工具"]) != target_tools:
                reasons.append("tool_trace_mismatch")
            if int(row["tool_count"]) != len(target_tools):
                reasons.append("tool_count_mismatch")
            if row.get("combo_seen_bucket") != target.get("combo_seen_bucket"):
                reasons.append("combo_bucket_mismatch")
        if accepted_row is None:
            reasons.append("missing_accepted_record")
        else:
            if "parent_id" in accepted_row:
                forbidden_provenance_fields.append(trace_id)
            if str(accepted_row.get("question")) != str(row["问题"]):
                reasons.append("accepted_question_mismatch")
            if accepted_row.get("tools") != split_tools(row["工具"]):
                reasons.append("accepted_tools_mismatch")
            if (accepted_row.get("judgment") or {}).get("verdict") != "pass":
                reasons.append("accepted_judgment_not_pass")
        if reasons:
            alignment_errors.append({"id": trace_id, "reasons": reasons})

    query_sets = {
        "api_native": set(questions),
        "train": normalized_queries(train),
        "frozen_test": normalized_queries(test),
    }
    query_overlap = {
        name: len(query_sets["api_native"] & values)
        for name, values in query_sets.items()
        if name != "api_native"
    }
    id_sets = {
        "api_native": set(ids),
        "train": {str(row["id"]) for row in train},
        "frozen_test": {str(row["id"]) for row in test},
    }
    trace_id_overlap = {
        name: len(id_sets["api_native"] & values)
        for name, values in id_sets.items()
        if name != "api_native"
    }

    expression_phrases: dict[str, set[str]] = {"train": set(), "test": set()}
    for meta_row in meta:
        split = str(meta_row.get("split", ""))
        if split not in expression_phrases:
            continue
        for phrase in (meta_row.get("expression_phrases") or {}).values():
            normalized = normalize_question(str(phrase))
            if normalized:
                expression_phrases[split].add(normalized)
    expression_phrase_overlap = {}
    for split, phrases in expression_phrases.items():
        hit_rows = sum(any(phrase in question for phrase in phrases) for question in questions)
        expression_phrase_overlap[split] = {
            "unique_reference_phrases": len(phrases),
            "api_rows_containing_full_reference_phrase": hit_rows,
            "rate": hit_rows / max(1, len(rows)),
        }

    internal_term_rows = [
        str(row["id"]) for row in rows if INTERNAL_RE.search(str(row["问题"]))
    ]
    tool_id_rows = [
        str(row["id"])
        for row in rows
        if any(str(tool_id).casefold() in str(row["问题"]).casefold() for tool_id in by_id)
    ]
    terms = catalog_term_index(catalog)
    cleaned_descriptions = {
        str(tool["tool_id"]): normalize_question(safe_text(str(tool["description"]), terms))
        for tool in catalog
    }
    copied_description_rows = [
        str(row["id"])
        for row, question in zip(rows, questions)
        if any(value and value in question for value in cleaned_descriptions.values())
    ]

    def full_description_signals(tool: dict[str, Any]) -> list[str]:
        return [cleaned_descriptions[str(tool["tool_id"])]]

    def catalog_surface_signals(tool: dict[str, Any]) -> list[str]:
        values = [
            str(tool.get("name_zh", "")),
            str(tool.get("keyword_label", "")),
            *[str(value) for value in tool.get("positive_phrases", [])],
        ]
        return sorted(
            {normalize_question(value) for value in values if value},
            key=len,
            reverse=True,
        )

    tool_counts = Counter(tool_id for row in rows for tool_id in split_tools(row["工具"]))
    payload = {
        "audit_version": "assignment3-api-native-isolation-audit-v2",
        "strict_ok": False,
        "counts": {
            "rows": len(rows),
            "unique_ids": len(set(ids)),
            "unique_normalized_queries": len(set(questions)),
            "tool_count": dict(sorted(Counter(int(row["tool_count"]) for row in rows).items())),
            "combo_seen_bucket": dict(
                sorted(Counter(str(row["combo_seen_bucket"]) for row in rows).items())
            ),
            "missing_tools": sorted(set(by_id) - set(tool_counts)),
            "per_tool_min": min(tool_counts.values()) if tool_counts else 0,
            "per_tool_max": max(tool_counts.values()) if tool_counts else 0,
        },
        "standalone_alignment": {
            "target_plan_rows": len(plan),
            "accepted_rows": len(accepted),
            "forbidden_provenance_fields": sorted(set(forbidden_provenance_fields)),
            "errors": alignment_errors,
        },
        "query_overlap": query_overlap,
        "trace_id_overlap": trace_id_overlap,
        "expression_phrase_substring_overlap": expression_phrase_overlap,
        "generation_provenance": {
            "query_template_used": bool(generation_audit.get("query_templates_used")),
            "target_plan_contains_queries": bool(
                generation_audit.get("target_plan_contains_queries")
            ),
            "api_generator": "free-form API generation; no expression cluster selected",
            "training_use_allowed": False,
        },
        "internal_leakage": {
            "internal_term_rows": internal_term_rows,
            "literal_tool_id_rows": tool_id_rows,
            "cleaned_full_catalog_description_rows": copied_description_rows,
        },
        "literal_rule_diagnostics": {
            "cleaned_full_description": rule_metrics(
                rows, catalog, full_description_signals
            ),
            "catalog_surface_terms": rule_metrics(rows, catalog, catalog_surface_signals),
        },
        "generation_audit_strict_ok": bool(generation_audit.get("strict_ok")),
        "training_use_allowed": False,
        "hashes": {
            str(path): sha256_file(path)
            for path in (
                args.input,
                args.target_plan,
                args.accepted,
                args.generation_audit,
                args.train,
                args.test,
                args.generation_meta,
                args.catalog,
            )
        },
    }
    payload["strict_ok"] = (
        len(rows) == args.expected_count
        and len(plan) == len(rows)
        and len(accepted) == len(rows)
        and len(set(ids)) == len(rows)
        and len(set(questions)) == len(rows)
        and not alignment_errors
        and not forbidden_provenance_fields
        and not any(query_overlap.values())
        and not any(trace_id_overlap.values())
        and not internal_term_rows
        and not tool_id_rows
        and not copied_description_rows
        and not payload["counts"]["missing_tools"]
        and generation_audit.get("strict_ok") is True
        and generation_audit.get("query_templates_used") is False
        and generation_audit.get("target_plan_contains_queries") is False
    )
    write_json(args.output, payload)
    print(
        f"api-native audit rows={len(rows)} strict_ok={payload['strict_ok']} "
        f"query_overlap={sum(query_overlap.values())} "
        f"cleaned_description_hits={len(copied_description_rows)}"
    )
    return 0 if payload["strict_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
