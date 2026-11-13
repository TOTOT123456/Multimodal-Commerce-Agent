#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare the scaled API-native Assignment3 SFT runtime dataset."""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import (  # noqa: E402
    CATALOG_PATH,
    catalog_by_id,
    load_catalog,
    load_jsonl,
    split_tools,
    write_json,
    write_jsonl,
)
from finetune.prepare_tool_trace_sft_data import (  # noqa: E402
    apply_template_batch,
    tokenizer_identity,
)
from finetune.tool_trace_common import (  # noqa: E402
    PROMPT_VERSION,
    canonical_target,
    prompt_hash,
    render_messages,
    sha256_file,
    sha256_json,
    summary_stats,
)


def candidate_tools(row: dict[str, Any]) -> list[str]:
    construction = row.get("construction")
    if not isinstance(construction, dict):
        raise SystemExit(f"missing construction metadata: {row.get('id')}")
    candidates = construction.get("candidate_tools")
    if not isinstance(candidates, list):
        raise SystemExit(f"missing candidate tools: {row.get('id')}")
    return [str(value) for value in candidates]


def build_records(
    rows: list[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    records = []
    for row in rows:
        gold = split_tools(row["工具"])
        candidates = candidate_tools(row)
        if (
            len(candidates) != len(set(candidates))
            or not set(candidates) <= set(by_id)
            or not set(gold) <= set(candidates)
        ):
            raise SystemExit(f"invalid candidates: {row['id']}")
        target = canonical_target(gold)
        messages = render_messages(row["问题"], candidates, by_id, "rag")
        records.append(
            {
                "trace_id": row["id"],
                "query": row["问题"],
                "gold_tools": gold,
                "target": target,
                "source_split": "assignment3_api_native_train",
                "view": "rag",
                "candidate_tools": candidates,
                "candidate_source": row["construction"]["type"],
                "candidate_k": len(candidates),
                "prompt_version": PROMPT_VERSION,
                "prompt_hash": prompt_hash(messages),
                "tool_count": int(row["tool_count"]),
                "domain": row["domain"],
                "combo_seen_bucket": row["combo_seen_bucket"],
                "construction": row["construction"],
            }
        )
    return records


def runtime_row(
    record: dict[str, Any],
    by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    messages = render_messages(
        record["query"],
        record["candidate_tools"],
        by_id,
        "rag",
        assistant=record["target"],
    )
    return {
        "messages": messages,
        "meta": {
            "trace_id": record["trace_id"],
            "source_split": record["source_split"],
            "view": record["view"],
            "candidate_source": record["candidate_source"],
            "candidate_k": record["candidate_k"],
            "prompt_hash": record["prompt_hash"],
            "record_hash": sha256_json(record),
            "construction_type": record["construction"]["type"],
        },
    }


def measure(
    tokenizer: Any,
    records: list[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    input_lengths = []
    total_lengths = []
    target_lengths = []
    target_preserved = 0
    for start in range(0, len(records), 256):
        chunk = records[start : start + 256]
        prompt_messages = [
            render_messages(row["query"], row["candidate_tools"], by_id, "rag")
            for row in chunk
        ]
        full_messages = [
            messages + [{"role": "assistant", "content": row["target"]}]
            for messages, row in zip(prompt_messages, chunk)
        ]
        inputs = apply_template_batch(tokenizer, prompt_messages, True)
        totals = apply_template_batch(tokenizer, full_messages, False)
        targets = tokenizer(
            [row["target"] for row in chunk],
            add_special_tokens=False,
        )["input_ids"]
        for row, input_ids, total_ids, target_ids in zip(chunk, inputs, totals, targets):
            input_lengths.append(len(input_ids))
            total_lengths.append(len(total_ids))
            target_lengths.append(len(target_ids))
            tail = tokenizer.decode(total_ids[-max(128, len(target_ids) + 32) :])
            target_preserved += int(row["target"] in tail)
    return {
        "input_tokens": summary_stats(input_lengths),
        "total_tokens": summary_stats(total_lengths),
        "target_tokens": summary_stats(target_lengths),
        "max_total_tokens": max(total_lengths),
        "target_preserved_rate": target_preserved / len(records),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    parser.add_argument("--runtime-out", type=Path, required=True)
    parser.add_argument("--compact-out", type=Path, required=True)
    parser.add_argument("--stats-out", type=Path, required=True)
    parser.add_argument("--audit-out", type=Path, required=True)
    parser.add_argument("--smoke-out", type=Path, required=True)
    parser.add_argument("--tokenizer-models", nargs="+", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=54000)
    parser.add_argument("--length-candidates", nargs="+", type=int, default=[2048, 4096, 8192])
    parser.add_argument("--safety-margin", type=int, default=64)
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    plans = load_jsonl(args.plan)
    if len(rows) != args.expected_count or len(plans) != args.expected_count:
        raise SystemExit("scaled SFT row count mismatch")
    plan_by_id = {row["id"]: row for row in plans}
    if len(plan_by_id) != len(plans) or {row["id"] for row in rows} != set(plan_by_id):
        raise SystemExit("SFT data/plan id mismatch")
    for row in rows:
        plan = plan_by_id[row["id"]]
        if row.get("construction") != plan.get("construction"):
            raise SystemExit(f"construction mismatch: {row['id']}")

    catalog = load_catalog(args.catalog)
    by_id = catalog_by_id(catalog)
    records = build_records(rows, by_id)
    runtime = [runtime_row(row, by_id) for row in records]
    smoke_records = sorted(
        records,
        key=lambda row: (
            -sum(
                len(message["content"])
                for message in render_messages(
                    row["query"],
                    row["candidate_tools"],
                    by_id,
                    "rag",
                    assistant=row["target"],
                )
            ),
            row["trace_id"],
        ),
    )[:16]
    smoke_runtime = [runtime_row(row, by_id) for row in smoke_records]
    write_jsonl(args.compact_out, records)
    write_jsonl(args.runtime_out, runtime)
    write_jsonl(args.smoke_out, smoke_runtime)

    tokenizer_stats = {}
    selected_lengths = []
    for model_path in args.tokenizer_models:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        measured = measure(tokenizer, records, by_id)
        selected = next(
            (
                value
                for value in args.length_candidates
                if measured["max_total_tokens"] + args.safety_margin <= value
            ),
            None,
        )
        if selected is None:
            raise SystemExit(f"{model_path}: no safe max length")
        measured.update(
            {
                "identity": tokenizer_identity(tokenizer, model_path),
                "selected_max_length": selected,
                "safety_margin": args.safety_margin,
                "truncated_records": 0,
            }
        )
        tokenizer_stats[str(model_path)] = measured
        selected_lengths.append(selected)
    stats = {
        "status": "complete",
        "prompt_version": PROMPT_VERSION,
        "records": len(records),
        "tokenizers": tokenizer_stats,
    }
    write_json(args.stats_out, stats)

    type_counts = Counter(row["construction"]["type"] for row in records)
    candidate_counts = Counter(len(row["candidate_tools"]) for row in records)
    prompt_matches = sum(
        prompt_hash(
            render_messages(row["query"], row["candidate_tools"], by_id, "rag")
        )
        == row["prompt_hash"]
        for row in records
    )
    runtime_matches = sum(
        row["meta"]["trace_id"] == record["trace_id"]
        and row["meta"]["record_hash"] == sha256_json(record)
        for row, record in zip(runtime, records)
    )
    audit = {
        "audit_version": "assignment3-100tool-api-sft-runtime-v1",
        "strict_ok": (
            len(records) == args.expected_count
            and len({row["trace_id"] for row in records}) == len(records)
            and all(set(row["gold_tools"]) <= set(row["candidate_tools"]) for row in records)
            and all(len(row["candidate_tools"]) <= 6 for row in records)
            and prompt_matches == len(records)
            and runtime_matches == len(records)
            and all(
                value["target_preserved_rate"] == 1.0
                and value["truncated_records"] == 0
                for value in tokenizer_stats.values()
            )
        ),
        "semantic_rows": len(rows),
        "runtime_rows": len(runtime),
        "smoke_rows": len(smoke_runtime),
        "smoke_selection": "16 longest rendered-message character lengths",
        "view_distribution": {"rag": len(runtime), "global": 0},
        "construction_counts": dict(sorted(type_counts.items())),
        "candidate_count_distribution": dict(sorted(candidate_counts.items())),
        "gold_candidate_coverage_rate": sum(
            set(row["gold_tools"]) <= set(row["candidate_tools"]) for row in records
        )
        / len(records),
        "full_catalog_training_rows": 0,
        "prompt_hash_match_rate": prompt_matches / len(records),
        "runtime_record_hash_match_rate": runtime_matches / len(records),
        "inputs": {
            str(args.input): sha256_file(args.input),
            str(args.plan): sha256_file(args.plan),
            str(args.catalog): sha256_file(args.catalog),
        },
        "outputs": {
            str(args.runtime_out): sha256_file(args.runtime_out),
            str(args.compact_out): sha256_file(args.compact_out),
            str(args.smoke_out): sha256_file(args.smoke_out),
            str(args.stats_out): sha256_file(args.stats_out),
        },
    }
    write_json(args.audit_out, audit)
    print(
        f"Assignment3 100-tool SFT rows={len(runtime)} "
        f"max_length={max(selected_lengths)} strict_ok={audit['strict_ok']}"
    )
    return 0 if audit["strict_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
