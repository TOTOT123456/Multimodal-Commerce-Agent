#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit catalog-derived datasets for coverage, leakage and overlap."""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import (
    CATALOG_PATH,
    INTENT_DIR,
    PROJECT_ROOT,
    TOOL_TRACE_DIR,
    catalog_by_id,
    domain_counts,
    has_tool_id_leak,
    load_catalog,
    load_jsonl,
    normalize_question,
    split_tools,
    tool_counts,
    write_json,
)


def duplicate_count(rows: list[dict]) -> int:
    keys = [normalize_question(str(row.get("问题", ""))) for row in rows]
    return len(keys) - len(set(keys))


def render_markdown(payload: dict) -> str:
    lines = [
        "# Dataset Audit",
        "",
        f"- status: {'PASS' if payload['ok'] else 'FAIL'}",
        f"- catalog_tools: {payload['catalog_tools']}",
        f"- train_rows: {payload['tool_trace']['train_rows']}",
        f"- test_rows: {payload['tool_trace']['test_rows']}",
        f"- intent_rows: {payload['intent']['intent_rows']}",
        "",
        "## Coverage",
        "",
        f"- missing_train_tools: {len(payload['tool_trace']['missing_train_tools'])}",
        f"- missing_test_tools: {len(payload['tool_trace']['missing_test_tools'])}",
        f"- train_test_overlap: {payload['tool_trace']['train_test_overlap']}",
        f"- leaked_tool_id_rows: {payload['tool_trace']['leaked_tool_id_rows']}",
        "",
        "## Intent",
        "",
    ]
    for key, value in sorted(payload["intent"].get("by_type", {}).items()):
        lines.append(f"- {key}: {value}")
    if payload["errors"]:
        lines += ["", "## Errors", ""]
        lines.extend(f"- {item}" for item in payload["errors"])
    if payload["warnings"]:
        lines += ["", "## Warnings", ""]
        lines.extend(f"- {item}" for item in payload["warnings"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    parser.add_argument("--train", type=Path, default=TOOL_TRACE_DIR / "train.jsonl")
    parser.add_argument("--test", type=Path, default=TOOL_TRACE_DIR / "test.jsonl")
    parser.add_argument("--intent", type=Path, default=INTENT_DIR / "intent_all.jsonl")
    parser.add_argument("--chitchat", type=Path, default=INTENT_DIR / "chitchat.jsonl")
    parser.add_argument("--json-out", type=Path, default=PROJECT_ROOT / "reports" / "dataset_audit.json")
    parser.add_argument("--md-out", type=Path, default=None, help="Optional Markdown summary output path.")
    parser.add_argument("--strict-min-test-per-tool", type=int, default=0)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()

    catalog = load_catalog(args.catalog)
    by_id = catalog_by_id(catalog)
    tool_ids = set(by_id)
    train = load_jsonl(args.train)
    test = load_jsonl(args.test)
    intent = load_jsonl(args.intent)
    chitchat = load_jsonl(args.chitchat)

    train_keys = {normalize_question(row["问题"]) for row in train}
    test_keys = {normalize_question(row["问题"]) for row in test}
    intent_keys = {normalize_question(row["问题"]) for row in intent}
    chitchat_keys = {normalize_question(row["问题"]) for row in chitchat}
    train_counts = tool_counts(train)
    test_counts = tool_counts(test)
    leaked_rows = [row.get("id", "") for row in train + test if has_tool_id_leak(str(row.get("问题", "")), tool_ids)]
    invalid_tools = sorted({t for row in train + test for t in split_tools(row.get("工具")) if t not in tool_ids})

    errors: list[str] = []
    warnings: list[str] = []
    if train_keys & test_keys:
        errors.append(f"train/test normalized overlap: {len(train_keys & test_keys)}")
    if train_keys & intent_keys:
        errors.append(f"train/intent_all normalized overlap: {len(train_keys & intent_keys)}")
    if train_keys & chitchat_keys:
        errors.append(f"train/chitchat normalized overlap: {len(train_keys & chitchat_keys)}")
    if invalid_tools:
        errors.append(f"invalid tool ids in traces: {invalid_tools}")
    missing_train = sorted(tool_ids - set(train_counts))
    missing_test = sorted(tool_ids - set(test_counts))
    if missing_train:
        errors.append(f"missing train tool coverage: {missing_train[:10]} ... total={len(missing_train)}")
    if missing_test:
        errors.append(f"missing test tool coverage: {missing_test[:10]} ... total={len(missing_test)}")
    if leaked_rows:
        errors.append(f"question leaks english tool_id in {len(leaked_rows)} rows")
    if duplicate_count(train):
        warnings.append(f"train duplicate normalized questions: {duplicate_count(train)}")
    if duplicate_count(test):
        warnings.append(f"test duplicate normalized questions: {duplicate_count(test)}")
    if args.strict_min_test_per_tool:
        below = {tool: n for tool, n in test_counts.items() if n < args.strict_min_test_per_tool}
        missing_below = {tool: 0 for tool in tool_ids - set(test_counts)}
        below.update(missing_below)
        if below:
            warnings.append(f"{len(below)} tools below strict_min_test_per_tool={args.strict_min_test_per_tool}")

    payload = {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "catalog_tools": len(catalog),
        "tool_trace": {
            "train_rows": len(train),
            "test_rows": len(test),
            "train_test_overlap": len(train_keys & test_keys),
            "train_intent_overlap": len(train_keys & intent_keys),
            "train_chitchat_overlap": len(train_keys & chitchat_keys),
            "leaked_tool_id_rows": len(leaked_rows),
            "invalid_tools": invalid_tools,
            "missing_train_tools": missing_train,
            "missing_test_tools": missing_test,
            "train_by_tool": dict(sorted(train_counts.items())),
            "test_by_tool": dict(sorted(test_counts.items())),
            "train_by_domain": dict(sorted(domain_counts(train, by_id).items())),
            "test_by_domain": dict(sorted(domain_counts(test, by_id).items())),
            "train_duplicates": duplicate_count(train),
            "test_duplicates": duplicate_count(test),
        },
        "intent": {
            "intent_rows": len(intent),
            "chitchat_rows": len(chitchat),
            "by_type": dict(Counter(row.get("类型", "") for row in intent)),
            "empty_keyword_ecommerce": sum(1 for row in intent if row.get("类型") == "电商类" and not row.get("关键词")),
            "duplicates": duplicate_count(intent),
        },
    }
    write_json(args.json_out, payload)
    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(render_markdown(payload), encoding="utf-8")
    print(f"dataset audit ok={payload['ok']} errors={len(errors)} -> {args.json_out}")
    return 0 if (payload["ok"] or args.no_fail) else 1


if __name__ == "__main__":
    raise SystemExit(main())
