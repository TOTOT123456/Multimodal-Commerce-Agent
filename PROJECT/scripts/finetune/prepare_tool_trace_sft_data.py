#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build compact and runtime SFT views for assignment-3 tool tracing."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import (  # noqa: E402
    CATALOG_PATH,
    TOOL_TRACE_DIR,
    catalog_by_id,
    load_catalog,
    load_jsonl,
    split_tools,
    write_json,
    write_jsonl,
)
from finetune.tool_trace_common import (  # noqa: E402
    ASSIGNMENT3_DIR,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    canonical_target,
    deterministic_shuffle,
    prompt_hash,
    render_messages,
    sha256_file,
    sha256_json,
    stable_seed,
    summary_stats,
)


DEFAULT_RUNTIME = (
    Path(__file__).resolve().parents[2]
    / "finetune"
    / "outputs"
    / "tooltrace-data-cache"
    / "runs"
    / "tool_trace_sft_train.runtime.jsonl"
)
DEFAULT_SMOKE_RUNTIME = DEFAULT_RUNTIME.with_name("tool_trace_sft_smoke.runtime.jsonl")
DEFAULT_MODEL_4B = Path(__file__).resolve().parents[2] / "models" / "Qwen" / "Qwen3-4B"
DEFAULT_MODEL_8B = Path(__file__).resolve().parents[2] / "models" / "Qwen" / "Qwen3-8B"


def aligned_rankings(
    source: Path,
    train_rows: list[dict[str, Any]],
    catalog_ids: set[str],
) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(source)
    if len(rows) != len(train_rows):
        raise SystemExit(f"{source}: expected {len(train_rows)} rows, got {len(rows)}")
    by_id = {row.get("id"): row for row in rows}
    if len(by_id) != len(rows):
        raise SystemExit(f"{source}: duplicate ranking ids")
    for train in train_rows:
        ranking = by_id.get(train["id"])
        if not ranking:
            raise SystemExit(f"{source}: missing {train['id']}")
        if ranking.get("query") != train["问题"]:
            raise SystemExit(f"{source}: query mismatch {train['id']}")
        if ranking.get("gold_tools") != split_tools(train["工具"]):
            raise SystemExit(f"{source}: gold mismatch {train['id']}")
        tools = ranking.get("pred_tools")
        if not isinstance(tools, list) or len(tools) != len(set(tools)) or not set(tools) <= catalog_ids:
            raise SystemExit(f"{source}: invalid ranking {train['id']}")
    return by_id


def inject_gold(
    ranking: list[str],
    gold: list[str],
    k: int,
    catalog_ids: set[str],
) -> tuple[list[str], list[str], dict[str, int]]:
    original = [tool_id for tool_id in ranking if tool_id in catalog_ids]
    original = list(dict.fromkeys(original))
    candidates = original[:k]
    injected = [tool_id for tool_id in gold if tool_id not in candidates]
    for tool_id in injected:
        replace_at = next(
            (index for index in range(len(candidates) - 1, -1, -1) if candidates[index] not in gold),
            None,
        )
        if replace_at is None:
            candidates.append(tool_id)
        else:
            candidates[replace_at] = tool_id
    candidates = list(dict.fromkeys(candidates))
    for tool_id in gold:
        if tool_id not in candidates:
            candidates.append(tool_id)
    for tool_id in original:
        if len(candidates) >= k:
            break
        if tool_id not in candidates:
            candidates.append(tool_id)
    if len(candidates) > k:
        removable = [tool_id for tool_id in reversed(candidates) if tool_id not in gold]
        while len(candidates) > k and removable:
            candidates.remove(removable.pop(0))
    ranks = {tool_id: original.index(tool_id) + 1 if tool_id in original else -1 for tool_id in candidates}
    if not set(gold) <= set(candidates):
        raise AssertionError("gold injection failed")
    return candidates, injected, ranks


def hard_candidates(
    trace_id: str,
    gold: list[str],
    k: int,
    catalog: list[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
) -> list[str]:
    values = list(gold)
    for tool_id in gold:
        values.extend(by_id[tool_id].get("confusable_tools", []))
    gold_domains = {by_id[tool_id]["domain"] for tool_id in gold}
    values.extend(
        tool["tool_id"]
        for tool in catalog
        if tool["domain"] in gold_domains and tool["tool_id"] not in gold
    )
    values.extend(tool["tool_id"] for tool in catalog)
    values = list(dict.fromkeys(values))
    negative = deterministic_shuffle([tool_id for tool_id in values if tool_id not in gold], trace_id, "hard-negative")
    candidates = list(gold) + negative
    return candidates[:k]


def build_rag_record(
    row: dict[str, Any],
    catalog: list[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
    final_ranking: dict[str, Any],
    untuned_ranking: dict[str, Any],
) -> dict[str, Any]:
    trace_id = row["id"]
    gold = split_tools(row["工具"])
    k = [10, 20, 30][stable_seed(trace_id, "candidate-k") % 3]
    source_bucket = stable_seed(trace_id, "candidate-source") % 4
    if source_bucket < 2:
        source = "task_tuned_final"
        ranking = final_ranking["pred_tools"]
    elif source_bucket == 2:
        source = "untuned_bge_large_bm25_rrf"
        ranking = untuned_ranking["pred_tools"]
    else:
        source = "gold_confusable_hard_negative"
        ranking = hard_candidates(trace_id, gold, k, catalog, by_id)
    candidates, injected, original_ranks = inject_gold(ranking, gold, k, set(by_id))
    if not any(tool_id not in gold for tool_id in candidates):
        extra = next(tool["tool_id"] for tool in catalog if tool["tool_id"] not in gold)
        if len(candidates) >= k:
            candidates[-1] = extra
        else:
            candidates.append(extra)
    shuffled = stable_seed(trace_id, "candidate-order") % 2 == 1
    if shuffled:
        candidates = deterministic_shuffle(candidates, trace_id, "candidate-order-values")
    target = canonical_target(gold)
    messages = render_messages(row["问题"], candidates, by_id, "rag")
    return {
        "trace_id": trace_id,
        "query": row["问题"],
        "gold_tools": gold,
        "target": target,
        "source_split": "train",
        "view": "rag",
        "candidate_tools": candidates,
        "candidate_source": source,
        "candidate_k": len(candidates),
        "candidate_order_shuffled": shuffled,
        "gold_injected": bool(injected),
        "injected_tools": injected,
        "candidate_original_ranks": original_ranks,
        "prompt_version": PROMPT_VERSION,
        "prompt_hash": prompt_hash(messages),
        "tool_count": int(row["tool_count"]),
        "domain": row["domain"],
        "combo_seen_bucket": row["combo_seen_bucket"],
    }


def build_global_record(
    row: dict[str, Any],
    catalog: list[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    trace_id = row["id"]
    candidates = [tool["tool_id"] for tool in catalog]
    shuffled = stable_seed(trace_id, "global-order") % 2 == 1
    if shuffled:
        candidates = deterministic_shuffle(candidates, trace_id, "global-order-values")
    gold = split_tools(row["工具"])
    messages = render_messages(row["问题"], candidates, by_id, "global")
    return {
        "trace_id": trace_id,
        "query": row["问题"],
        "gold_tools": gold,
        "target": canonical_target(gold),
        "source_split": "train",
        "view": "global",
        "candidate_tools": candidates,
        "candidate_source": "catalog_snapshot",
        "candidate_k": len(candidates),
        "candidate_order_shuffled": shuffled,
        "gold_injected": False,
        "injected_tools": [],
        "candidate_original_ranks": {},
        "prompt_version": PROMPT_VERSION,
        "prompt_hash": prompt_hash(messages),
        "tool_count": int(row["tool_count"]),
        "domain": row["domain"],
        "combo_seen_bucket": row["combo_seen_bucket"],
    }


def tokenizer_identity(tokenizer: Any, model_path: Path) -> dict[str, Any]:
    files = {}
    for name in ["tokenizer.json", "tokenizer_config.json", "config.json", "generation_config.json"]:
        path = model_path / name
        if path.exists():
            files[name] = sha256_file(path)
    return {
        "model_path": str(model_path),
        "tokenizer_class": tokenizer.__class__.__name__,
        "model_max_length": tokenizer.model_max_length,
        "files": files,
        "identity_hash": sha256_json(files),
    }


def apply_template(tokenizer: Any, messages: list[dict[str, str]], add_generation_prompt: bool) -> list[int]:
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": False,
    }
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking")
        return tokenizer.apply_chat_template(messages, **kwargs)


def apply_template_batch(
    tokenizer: Any,
    conversations: list[list[dict[str, str]]],
    add_generation_prompt: bool,
) -> list[list[int]]:
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": False,
    }
    try:
        return tokenizer.apply_chat_template(conversations, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking")
        return tokenizer.apply_chat_template(conversations, **kwargs)


def measure_records(
    tokenizer: Any,
    records: list[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
) -> dict[str, dict[str, int]]:
    lengths: dict[str, dict[str, int]] = {}
    batch_size = 128
    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        prompt_messages = [
            render_messages(record["query"], record["candidate_tools"], by_id, record["view"])
            for record in chunk
        ]
        full_messages = [
            messages + [{"role": "assistant", "content": record["target"]}]
            for messages, record in zip(prompt_messages, chunk)
        ]
        input_rows = apply_template_batch(tokenizer, prompt_messages, True)
        full_rows = apply_template_batch(tokenizer, full_messages, False)
        target_rows = tokenizer(
            [record["target"] for record in chunk],
            add_special_tokens=False,
        )["input_ids"]
        for record, input_ids, full_ids, target_ids in zip(chunk, input_rows, full_rows, target_rows):
            # The target and any template-inserted assistant prefix are at the
            # tail; decoding every 100-tool prompt is unnecessary.
            tail_size = max(256, len(target_ids) + 32)
            decoded_full = tokenizer.decode(full_ids[-tail_size:], skip_special_tokens=False)
            lengths[f"{record['trace_id']}:{record['view']}"] = {
                "input_tokens": len(input_ids),
                "total_tokens": len(full_ids),
                "target_tokens": len(target_ids),
                "assistant_preserved": int(record["target"] in decoded_full),
                # The stock Hugging Face Qwen3 template inserts an empty
                # non-thinking prefix for completed assistant messages.  It is
                # not part of our target and MS-SWIFT is configured with
                # add_non_thinking_prefix=false; the separate Swift-label audit
                # verifies what is actually supervised.  Here, fail only if
                # the source target itself contains thinking tags.
                "unexpected_think": int(
                    "<think>" in record["target"] or "</think>" in record["target"]
                ),
                "hf_template_empty_think_prefix": int(
                    "<think>\n\n</think>" in decoded_full
                ),
            }
        processed = start + len(chunk)
        if processed % 5120 == 0 or processed == len(records):
            print(f"tokenized {processed}/{len(records)}", flush=True)
    return lengths


def coverage_labels(record: dict[str, Any], by_id: dict[str, dict[str, Any]]) -> set[str]:
    labels = {
        f"tool_count:{record['tool_count']}",
        f"kind:{'single' if record['tool_count'] == 1 else 'multi'}",
        f"combo:{record['combo_seen_bucket']}",
    }
    labels.update(f"tool:{tool_id}" for tool_id in record["gold_tools"])
    labels.update(f"domain:{by_id[tool_id]['domain']}" for tool_id in record["gold_tools"])
    return labels


def select_global_records(
    records: list[dict[str, Any]],
    target_n: int,
    by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    ordered = sorted(records, key=lambda record: stable_seed(record["trace_id"], "global-subset"))
    selected = ordered[:target_n]
    all_labels = set().union(*(coverage_labels(record, by_id) for record in records))
    covered = set().union(*(coverage_labels(record, by_id) for record in selected)) if selected else set()
    selected_ids = {record["trace_id"] for record in selected}
    while missing := all_labels - covered:
        best = max(
            (record for record in ordered if record["trace_id"] not in selected_ids),
            key=lambda record: (
                len(coverage_labels(record, by_id) & missing),
                -stable_seed(record["trace_id"], "coverage"),
            ),
        )
        selected.append(best)
        selected_ids.add(best["trace_id"])
        covered.update(coverage_labels(best, by_id))
    return selected


def runtime_row(record: dict[str, Any], by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    messages = render_messages(
        record["query"],
        record["candidate_tools"],
        by_id,
        record["view"],
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
            "gold_injected": record["gold_injected"],
            "prompt_hash": record["prompt_hash"],
            "compact_record_hash": sha256_json(record),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    parser.add_argument("--train", type=Path, default=TOOL_TRACE_DIR / "train.jsonl")
    parser.add_argument("--final-rankings", type=Path, default=ASSIGNMENT3_DIR / "rag_train_rankings.jsonl")
    parser.add_argument("--untuned-rankings", type=Path, default=ASSIGNMENT3_DIR / "rag_train_untuned_rankings.jsonl")
    parser.add_argument("--out", type=Path, default=ASSIGNMENT3_DIR / "sft_train_compact.jsonl")
    parser.add_argument("--global-ids-out", type=Path, default=ASSIGNMENT3_DIR / "global_view_ids.json")
    parser.add_argument("--length-stats-out", type=Path, default=ASSIGNMENT3_DIR / "prompt_length_stats.json")
    parser.add_argument("--audit-out", type=Path, default=ASSIGNMENT3_DIR / "sft_data_audit.json")
    parser.add_argument("--runtime-out", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--smoke-runtime-out", type=Path, default=DEFAULT_SMOKE_RUNTIME)
    parser.add_argument("--tokenizer-models", nargs="+", type=Path, default=[DEFAULT_MODEL_4B, DEFAULT_MODEL_8B])
    parser.add_argument("--global-token-share", type=float, default=0.40)
    parser.add_argument("--length-candidates", nargs="+", type=int, default=[8192, 12288, 16384])
    parser.add_argument("--safety-margin", type=int, default=64)
    parser.add_argument(
        "--expected-train-count",
        type=int,
        default=24000,
        help="Expected semantic training rows; set to the protocol-defined scaled count.",
    )
    args = parser.parse_args()

    if not 0.0 < args.global_token_share < 1.0:
        raise SystemExit("--global-token-share must be between zero and one")
    catalog = load_catalog(args.catalog)
    by_id = catalog_by_id(catalog)
    train_rows = load_jsonl(args.train)
    if len(train_rows) != args.expected_train_count:
        raise SystemExit(
            f"expected {args.expected_train_count} formal train rows, got {len(train_rows)}"
        )
    final_by_id = aligned_rankings(args.final_rankings, train_rows, set(by_id))
    untuned_by_id = aligned_rankings(args.untuned_rankings, train_rows, set(by_id))

    rag_records = [
        build_rag_record(row, catalog, by_id, final_by_id[row["id"]], untuned_by_id[row["id"]])
        for row in train_rows
    ]
    global_records = [build_global_record(row, catalog, by_id) for row in train_rows]

    from transformers import AutoTokenizer

    tokenizers = []
    tokenizer_stats = {}
    primary_lengths: dict[str, dict[str, int]] | None = None
    all_preliminary = rag_records + global_records
    for model_path in args.tokenizer_models:
        if not model_path.exists():
            raise SystemExit(f"missing tokenizer model: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        lengths = measure_records(tokenizer, all_preliminary, by_id)
        identity = tokenizer_identity(tokenizer, model_path)
        tokenizers.append((model_path, tokenizer, lengths, identity))
        if primary_lengths is None:
            primary_lengths = lengths
        tokenizer_stats[str(model_path)] = {"identity": identity}
    assert primary_lengths is not None

    mean_rag = sum(primary_lengths[f"{record['trace_id']}:rag"]["input_tokens"] for record in rag_records) / len(rag_records)
    mean_global = sum(
        primary_lengths[f"{record['trace_id']}:global"]["input_tokens"] for record in global_records
    ) / len(global_records)
    raw_n_global = math.floor(
        args.global_token_share
        / (1.0 - args.global_token_share)
        * len(rag_records)
        * mean_rag
        / mean_global
    )
    raw_n_global = min(len(global_records), raw_n_global)
    selected_global = select_global_records(global_records, raw_n_global, by_id)
    selected_global_ids = {record["trace_id"] for record in selected_global}
    compact = rag_records + selected_global

    for model_path, _, lengths, identity in tokenizers:
        view_payload = {}
        for view, records in [("rag", rag_records), ("global", selected_global)]:
            view_payload[view] = {
                "input_tokens": summary_stats(
                    lengths[f"{record['trace_id']}:{view}"]["input_tokens"] for record in records
                ),
                "total_tokens": summary_stats(
                    lengths[f"{record['trace_id']}:{view}"]["total_tokens"] for record in records
                ),
                "target_tokens": summary_stats(
                    lengths[f"{record['trace_id']}:{view}"]["target_tokens"] for record in records
                ),
                "assistant_not_preserved": sum(
                    not lengths[f"{record['trace_id']}:{view}"]["assistant_preserved"] for record in records
                ),
                "unexpected_think": sum(
                    lengths[f"{record['trace_id']}:{view}"]["unexpected_think"] for record in records
                ),
                "hf_template_empty_think_prefix": sum(
                    lengths[f"{record['trace_id']}:{view}"]["hf_template_empty_think_prefix"]
                    for record in records
                ),
            }
        max_total = max(
            lengths[f"{record['trace_id']}:{record['view']}"]["total_tokens"] for record in compact
        )
        selected_max_length = next(
            (
                candidate
                for candidate in args.length_candidates
                if max_total + args.safety_margin <= candidate
            ),
            None,
        )
        if selected_max_length is None:
            raise SystemExit(
                f"{model_path}: max total tokens {max_total} plus margin {args.safety_margin} "
                f"exceeds {args.length_candidates}"
            )
        tokenizer_stats[str(model_path)].update(
            {
                "views": view_payload,
                "max_total_tokens": max_total,
                "safety_margin": args.safety_margin,
                "selected_max_length": selected_max_length,
                "truncated_records": sum(
                    lengths[f"{record['trace_id']}:{record['view']}"]["total_tokens"] > selected_max_length
                    for record in compact
                ),
            }
        )

    primary_input_rag = sum(primary_lengths[f"{record['trace_id']}:rag"]["input_tokens"] for record in rag_records)
    primary_input_global = sum(
        primary_lengths[f"{record['trace_id']}:global"]["input_tokens"] for record in selected_global
    )
    length_payload = {
        "status": "complete",
        "prompt_version": PROMPT_VERSION,
        "catalog_fields": ["tool_id", "domain", "operation_type", "description"],
        "global_token_share_target": args.global_token_share,
        "mean_input_tokens_preselection": {"rag": mean_rag, "global": mean_global},
        "raw_n_global_formula": raw_n_global,
        "selected_n_global": len(selected_global),
        "rag_records": len(rag_records),
        "total_records": len(compact),
        "actual_global_input_token_share": primary_input_global / (primary_input_rag + primary_input_global),
        "tokenizers": tokenizer_stats,
    }
    write_json(args.length_stats_out, length_payload)

    write_jsonl(args.out, compact)
    write_json(
        args.global_ids_out,
        {
            "seed": 20260725,
            "target_global_input_token_share": args.global_token_share,
            "raw_n_global_formula": raw_n_global,
            "selected_n_global": len(selected_global),
            "trace_ids": [record["trace_id"] for record in selected_global],
            "shuffled_order_trace_ids": [
                record["trace_id"] for record in selected_global if record["candidate_order_shuffled"]
            ],
        },
    )
    runtime = [runtime_row(record, by_id) for record in compact]
    write_jsonl(args.runtime_out, runtime)
    write_jsonl(args.smoke_runtime_out, runtime[:10])

    source_counts = Counter(record["candidate_source"] for record in rag_records)
    k_counts = Counter(record["candidate_k"] for record in rag_records)
    assistant_parse_ok = 0
    assistant_equals_gold = 0
    unknown_tools = 0
    for record in compact:
        payload = json.loads(record["target"])
        assistant_parse_ok += int(isinstance(payload, dict) and isinstance(payload.get("tools"), list))
        assistant_equals_gold += int(payload.get("tools") == record["gold_tools"])
        unknown_tools += sum(tool_id not in by_id for tool_id in payload.get("tools", []))
    prompt_hash_matches = sum(
        runtime_item["meta"]["prompt_hash"] == record["prompt_hash"]
        for runtime_item, record in zip(runtime, compact)
    )
    compact_hash_matches = sum(
        runtime_item["meta"]["compact_record_hash"] == sha256_json(record)
        for runtime_item, record in zip(runtime, compact)
    )
    audit = {
        "strict_ok": True,
        "formal_train_rows": len(train_rows),
        "rag_records": len(rag_records),
        "global_records": len(selected_global),
        "compact_records": len(compact),
        "unique_rag_trace_ids": len({record["trace_id"] for record in rag_records}),
        "rag_views_per_formal_trace_min": min(Counter(record["trace_id"] for record in rag_records).values()),
        "rag_views_per_formal_trace_max": max(Counter(record["trace_id"] for record in rag_records).values()),
        "global_ids_match_manifest": {record["trace_id"] for record in selected_global} == selected_global_ids,
        "assistant_json_parse_rate": assistant_parse_ok / len(compact),
        "assistant_equals_source_gt_rate": assistant_equals_gold / len(compact),
        "unknown_assistant_tools": unknown_tools,
        "rag_gold_coverage_rate": sum(
            set(record["gold_tools"]) <= set(record["candidate_tools"]) for record in rag_records
        )
        / len(rag_records),
        "rag_gold_injection_rate": sum(record["gold_injected"] for record in rag_records) / len(rag_records),
        "rag_candidate_source_counts": dict(sorted(source_counts.items())),
        "rag_candidate_k_counts": dict(sorted(k_counts.items())),
        "rag_candidate_shuffle_rate": sum(record["candidate_order_shuffled"] for record in rag_records)
        / len(rag_records),
        "global_order_shuffle_rate": sum(record["candidate_order_shuffled"] for record in selected_global)
        / max(1, len(selected_global)),
        "runtime_prompt_hash_alignment_rate": prompt_hash_matches / len(compact),
        "runtime_compact_hash_alignment_rate": compact_hash_matches / len(compact),
        "tokenizer_zero_truncation": all(
            payload["truncated_records"] == 0 for payload in tokenizer_stats.values()
        ),
        "tokenizer_assistant_preserved": all(
            view["assistant_not_preserved"] == 0
            for payload in tokenizer_stats.values()
            for view in payload["views"].values()
        ),
        "tokenizer_unexpected_think": sum(
            view["unexpected_think"]
            for payload in tokenizer_stats.values()
            for view in payload["views"].values()
        ),
        "inputs": {
            str(args.catalog): sha256_file(args.catalog),
            str(args.train): sha256_file(args.train),
            str(args.final_rankings): sha256_file(args.final_rankings),
            str(args.untuned_rankings): sha256_file(args.untuned_rankings),
        },
        "outputs": {
            str(args.out): sha256_file(args.out),
            str(args.runtime_out): sha256_file(args.runtime_out),
            str(args.smoke_runtime_out): sha256_file(args.smoke_runtime_out),
            str(args.global_ids_out): sha256_file(args.global_ids_out),
            str(args.length_stats_out): sha256_file(args.length_stats_out),
        },
        "system_prompt_sha256": sha256_json(SYSTEM_PROMPT),
        "prompt_version": PROMPT_VERSION,
    }
    audit["strict_ok"] = (
        len(rag_records) == len(train_rows)
        and audit["unique_rag_trace_ids"] == len(train_rows)
        and audit["rag_views_per_formal_trace_min"] == 1
        and audit["rag_views_per_formal_trace_max"] == 1
        and audit["assistant_json_parse_rate"] == 1.0
        and audit["assistant_equals_source_gt_rate"] == 1.0
        and audit["unknown_assistant_tools"] == 0
        and audit["rag_gold_coverage_rate"] == 1.0
        and audit["runtime_prompt_hash_alignment_rate"] == 1.0
        and audit["runtime_compact_hash_alignment_rate"] == 1.0
        and audit["tokenizer_zero_truncation"]
        and audit["tokenizer_assistant_preserved"]
        and audit["tokenizer_unexpected_think"] == 0
    )
    write_json(args.audit_out, audit)
    print(
        f"SFT compact={len(compact)} rag={len(rag_records)} global={len(selected_global)} "
        f"global_token_share={length_payload['actual_global_input_token_share']:.4f} strict_ok={audit['strict_ok']}"
    )
    return 0 if audit["strict_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
