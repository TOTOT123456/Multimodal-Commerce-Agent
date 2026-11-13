#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build quality-gated DPO pairs from real train-side model errors."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import CATALOG_PATH, TOOL_TRACE_DIR, catalog_by_id, load_catalog, load_jsonl, normalize_question, split_tools, write_json, write_jsonl  # noqa: E402
from finetune.tool_trace_common import (  # noqa: E402
    ASSIGNMENT3_DIR,
    canonical_target,
    deterministic_shuffle,
    prompt_hash,
    render_messages,
    sha256_file,
    sha256_json,
    stable_seed,
    strict_parse_tools,
    summary_stats,
)


DEFAULT_RUNTIME_DIR = (
    Path(__file__).resolve().parents[2]
    / "finetune"
    / "outputs"
    / "tooltrace-data-cache"
    / "runs"
)


def parse_source_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("prediction source must be NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("prediction source must be NAME=PATH")
    return name, Path(path)


def error_types(predicted: list[str], gold: list[str]) -> list[str]:
    pred_set = set(predicted)
    gold_set = set(gold)
    errors = []
    if gold_set - pred_set:
        errors.append("omission")
    if pred_set - gold_set:
        errors.append("extra")
    if (gold_set - pred_set) and (pred_set - gold_set):
        errors.append("substitution")
    if (
        pred_set == gold_set
        and len(predicted) == len(gold)
        and predicted != gold
    ):
        errors.append("order-only")
    return errors


def order_identifiable(query: str, tools: list[str], by_id: dict[str, dict[str, Any]]) -> tuple[bool, list[str]]:
    markers = ["先", "再", "然后", "最后", "接着", "随后", "按顺序", "依次", "起初", "接下来", "收尾"]
    evidence = [marker for marker in markers if marker in query]
    tool_set = set(tools)
    for tool_id in tools:
        for dependency in by_id[tool_id].get("depends_on", []):
            if dependency in tool_set:
                evidence.append(f"depends_on:{dependency}->{tool_id}")
    return bool(evidence), evidence


def synthetic_rejected(
    trace_id: str,
    gold: list[str],
    candidates: list[str],
    query: str,
    by_id: dict[str, dict[str, Any]],
) -> tuple[str, list[str], bool, list[str]] | None:
    identifiable, evidence = order_identifiable(query, gold, by_id)
    modes = ["omission", "substitution", "extra", "order-only"]
    for mode in deterministic_shuffle(modes, trace_id, "synthetic-mode"):
        wrong = list(gold)
        if mode == "omission" and len(wrong) > 1:
            wrong.pop(stable_seed(trace_id, mode) % len(wrong))
        elif mode == "substitution":
            index = stable_seed(trace_id, mode) % len(wrong)
            confusable = [
                tool_id
                for tool_id in by_id[wrong[index]].get("confusable_tools", [])
                if tool_id in candidates and tool_id not in gold
            ]
            if not confusable:
                continue
            wrong[index] = confusable[0]
        elif mode == "extra":
            extras = [tool_id for tool_id in candidates if tool_id not in gold]
            if not extras:
                continue
            wrong.append(extras[stable_seed(trace_id, mode) % len(extras)])
        elif mode == "order-only" and len(wrong) > 1 and identifiable:
            wrong[0], wrong[1] = wrong[1], wrong[0]
        else:
            continue
        return canonical_target(wrong), [mode], identifiable, evidence
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", action="append", type=parse_source_arg, required=True)
    parser.add_argument("--core-source-pattern", default=r"(selected_sft|early_sft)")
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    parser.add_argument("--train", type=Path, default=TOOL_TRACE_DIR / "train.jsonl")
    parser.add_argument(
        "--dev",
        type=Path,
        default=ASSIGNMENT3_DIR / "api_native_dev_2400" / "dev.jsonl",
    )
    parser.add_argument("--frozen", type=Path, default=TOOL_TRACE_DIR / "test.jsonl")
    parser.add_argument(
        "--extra-blocked",
        action="append",
        type=Path,
        default=[],
        help="Additional evaluation splits forbidden from preference construction.",
    )
    parser.add_argument("--out", type=Path, default=ASSIGNMENT3_DIR / "dpo_preferences_compact.jsonl")
    parser.add_argument("--audit-out", type=Path, default=ASSIGNMENT3_DIR / "dpo_data_audit.json")
    parser.add_argument("--runtime-all-out", type=Path, default=DEFAULT_RUNTIME_DIR / "tool_trace_dpo_all.runtime.jsonl")
    parser.add_argument("--runtime-train-out", type=Path, default=DEFAULT_RUNTIME_DIR / "tool_trace_dpo_train.runtime.jsonl")
    parser.add_argument("--runtime-monitor-out", type=Path, default=DEFAULT_RUNTIME_DIR / "tool_trace_dpo_monitor.runtime.jsonl")
    parser.add_argument("--min-pairs", type=int, default=4000)
    parser.add_argument("--max-pairs", type=int, default=8000)
    parser.add_argument("--max-per-query", type=int, default=2)
    parser.add_argument("--min-core-ratio", type=float, default=0.70)
    parser.add_argument("--allow-synthetic", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    catalog = load_catalog(args.catalog)
    by_id = catalog_by_id(catalog)
    catalog_ids = set(by_id)
    train = load_jsonl(args.train)
    train_by_id = {row["id"]: row for row in train}
    if len(train_by_id) != len(train):
        raise SystemExit("duplicate formal train ids")
    dev = load_jsonl(args.dev)
    frozen = load_jsonl(args.frozen)
    extra_blocked = [
        row
        for path in args.extra_blocked
        for row in load_jsonl(path)
    ]
    blocked_ids = {row["id"] for row in dev + frozen + extra_blocked}
    blocked_queries = {
        normalize_question(row["问题"]) for row in dev + frozen + extra_blocked
    }
    core_pattern = re.compile(args.core_source_pattern)

    real_pool = []
    input_hashes = {}
    for source_name, path in args.predictions:
        if not path.exists():
            raise SystemExit(f"missing prediction source: {path}")
        prediction_hash = sha256_file(path)
        input_hashes[f"{source_name}:{path}"] = prediction_hash
        for prediction in load_jsonl(path):
            trace_id = prediction.get("id")
            source_row = train_by_id.get(trace_id)
            if not source_row:
                raise SystemExit(f"{source_name}: prediction is not formal train: {trace_id}")
            if trace_id in blocked_ids or normalize_question(prediction.get("query", "")) in blocked_queries:
                raise SystemExit(f"{source_name}: dev/frozen leakage: {trace_id}")
            if prediction.get("query") != source_row["问题"]:
                raise SystemExit(f"{source_name}: query mismatch: {trace_id}")
            view = prediction.get("view")
            if view not in {"global", "rag"}:
                raise SystemExit(f"{source_name}: invalid view: {view}")
            candidates = prediction.get("candidate_tools", [])
            if (
                not isinstance(candidates, list)
                or len(candidates) != len(set(candidates))
                or not set(candidates) <= catalog_ids
            ):
                raise SystemExit(f"{source_name}: invalid candidates: {trace_id}")
            messages = render_messages(source_row["问题"], candidates, by_id, view)
            expected_prompt_hash = prompt_hash(messages)
            if prediction.get("prompt_hash") != expected_prompt_hash:
                raise SystemExit(f"{source_name}: prompt hash mismatch: {trace_id}")
            gold = split_tools(source_row["工具"])
            parsed, valid, parse_error = strict_parse_tools(
                prediction.get("raw_output", ""),
                catalog_ids,
                set(candidates) if view == "rag" else None,
            )
            rejected = prediction.get("raw_output", "")
            if not rejected or rejected == canonical_target(gold):
                continue
            if valid:
                errors = error_types(parsed, gold)
            else:
                errors = [parse_error or "format"]
                errors.extend(error for error in error_types(parsed, gold) if error not in errors)
            if valid and set(parsed) == set(gold):
                # Order-only pairs are accepted only with explicit sequence evidence.
                identifiable, order_evidence = order_identifiable(source_row["问题"], gold, by_id)
                if parsed == gold or not identifiable:
                    continue
            else:
                identifiable, order_evidence = order_identifiable(source_row["问题"], gold, by_id)
            is_core = bool(core_pattern.search(source_name)) and valid and set(parsed) != set(gold)
            real_pool.append(
                {
                    "trace_id": trace_id,
                    "query": source_row["问题"],
                    "gold_tools": gold,
                    "view": view,
                    "candidate_tools": candidates,
                    "candidate_k": len(candidates),
                    "messages": messages,
                    "chosen": canonical_target(gold),
                    "rejected": rejected,
                    "rejected_tools": parsed,
                    "rejected_source": source_name,
                    "prediction_path": str(path),
                    "prediction_sha256": prediction_hash,
                    "sample_index": prediction.get("sample_index", 0),
                    "strict_rejected": valid,
                    "parse_error": parse_error,
                    "error_types": errors,
                    "core_hard_error": is_core,
                    "synthetic": False,
                    "prompt_hash": expected_prompt_hash,
                    "candidate_hash": sha256_json(candidates),
                    "order_identifiable": identifiable,
                    "order_evidence": order_evidence,
                    "tool_count": len(gold),
                    "domains": source_row["domain"].split(","),
                    "combo_seen_bucket": source_row["combo_seen_bucket"],
                }
            )

    def source_priority(source_name: str) -> int:
        if "selected_sft" in source_name:
            return 0
        if "early_sft" in source_name:
            return 1
        return 2

    # Prefer genuine endpoint-SFT failures because they are the hardest and
    # most deployment-relevant negatives. Earlier checkpoints only fill a
    # documented shortfall; non-core/format failures remain capped below.
    real_pool.sort(
        key=lambda row: (
            not row["core_hard_error"],
            source_priority(row["rejected_source"]),
            stable_seed(
                row["trace_id"],
                row["rejected_source"],
                row["sample_index"],
                "dpo-pool",
            ),
        )
    )
    selected = []
    per_query = Counter()
    selected_objects: set[int] = set()

    def add_item(item: dict[str, Any]) -> None:
        selected.append(item)
        per_query[item["trace_id"]] += 1
        selected_objects.add(id(item))

    # Cover every tool represented by a real core error before filling the pool.
    uncovered_tools = set(catalog_ids)
    for item in (row for row in real_pool if row["core_hard_error"]):
        if len(selected) >= args.max_pairs:
            break
        if per_query[item["trace_id"]] >= args.max_per_query:
            continue
        newly_covered = set(item["gold_tools"]) & uncovered_tools
        if not newly_covered:
            continue
        add_item(item)
        uncovered_tools -= set(item["gold_tools"])

    # If a tool has no core error, use a real supplementary error when the 30% cap permits it.
    for item in (row for row in real_pool if not row["core_hard_error"]):
        if len(selected) >= args.max_pairs or not uncovered_tools:
            break
        if per_query[item["trace_id"]] >= args.max_per_query:
            continue
        if not (set(item["gold_tools"]) & uncovered_tools):
            continue
        core_count = sum(row["core_hard_error"] for row in selected)
        max_supplement = int(core_count * (1.0 - args.min_core_ratio) / args.min_core_ratio)
        if sum(not row["core_hard_error"] for row in selected) >= max_supplement:
            continue
        add_item(item)
        uncovered_tools -= set(item["gold_tools"])

    for item in real_pool:
        if len(selected) >= args.max_pairs:
            break
        if id(item) in selected_objects:
            continue
        if per_query[item["trace_id"]] >= args.max_per_query:
            continue
        if not item["core_hard_error"]:
            core_count = sum(row["core_hard_error"] for row in selected)
            max_supplement = int(core_count * (1.0 - args.min_core_ratio) / args.min_core_ratio)
            if sum(not row["core_hard_error"] for row in selected) >= max_supplement:
                continue
        add_item(item)

    if args.allow_synthetic and len(selected) < args.min_pairs:
        core_count = sum(row["core_hard_error"] for row in selected)
        supplement_cap = int(core_count * (1.0 - args.min_core_ratio) / args.min_core_ratio)
        current_supplement = sum(not row["core_hard_error"] for row in selected)
        candidates_for_synthetic = sorted(
            train,
            key=lambda row: stable_seed(row["id"], "synthetic-preference"),
        )
        template_by_id = {}
        for item in real_pool:
            template_by_id.setdefault(item["trace_id"], item)
        for source_row in candidates_for_synthetic:
            if len(selected) >= min(args.min_pairs, args.max_pairs) or current_supplement >= supplement_cap:
                break
            if per_query[source_row["id"]] >= args.max_per_query:
                continue
            template = template_by_id.get(source_row["id"])
            if not template:
                continue
            generated = synthetic_rejected(
                source_row["id"],
                template["gold_tools"],
                template["candidate_tools"],
                source_row["问题"],
                by_id,
            )
            if not generated:
                continue
            rejected, errors, identifiable, evidence = generated
            synthetic = dict(template)
            synthetic.update(
                {
                    "rejected": rejected,
                    "rejected_tools": json.loads(rejected)["tools"],
                    "rejected_source": "rule_targeted_coverage",
                    "prediction_path": "",
                    "prediction_sha256": "",
                    "sample_index": -1,
                    "strict_rejected": True,
                    "parse_error": "",
                    "error_types": errors,
                    "core_hard_error": False,
                    "synthetic": True,
                    "order_identifiable": identifiable,
                    "order_evidence": evidence,
                }
            )
            selected.append(synthetic)
            per_query[source_row["id"]] += 1
            current_supplement += 1

    compact = []
    for item in selected:
        messages = item["messages"] + [{"role": "assistant", "content": item["chosen"]}]
        meta = {
            "trace_id": item["trace_id"],
            "view": item["view"],
            "candidate_tools": item["candidate_tools"],
            "candidate_k": item["candidate_k"],
            "rejected_source": item["rejected_source"],
            "prediction_path": item["prediction_path"],
            "prediction_sha256": item["prediction_sha256"],
            "sample_index": item["sample_index"],
            "error_types": item["error_types"],
            "synthetic": item["synthetic"],
            "core_hard_error": item["core_hard_error"],
            "prompt_hash": item["prompt_hash"],
            "candidate_hash": item["candidate_hash"],
            "order_identifiable": item["order_identifiable"],
            "order_evidence": item["order_evidence"],
            "tool_count": item["tool_count"],
            "domains": item["domains"],
            "combo_seen_bucket": item["combo_seen_bucket"],
        }
        compact.append(
            {
                "messages": messages,
                "rejected_response": item["rejected"],
                "meta": meta,
            }
        )
    write_jsonl(args.out, compact)
    monitor_trace_ids = {
        trace_id
        for trace_id in per_query
        if stable_seed(trace_id, "dpo-monitor-split") % 10 == 0
    }
    runtime_train = [row for row in compact if row["meta"]["trace_id"] not in monitor_trace_ids]
    runtime_monitor = [row for row in compact if row["meta"]["trace_id"] in monitor_trace_ids]
    write_jsonl(args.runtime_all_out, compact)
    write_jsonl(args.runtime_train_out, runtime_train)
    write_jsonl(args.runtime_monitor_out, runtime_monitor)

    core_count = sum(row["meta"]["core_hard_error"] for row in compact)
    synthetic_count = sum(row["meta"]["synthetic"] for row in compact)
    chosen_lengths = [len(row["messages"][-1]["content"]) for row in compact]
    rejected_lengths = [len(row["rejected_response"]) for row in compact]
    source_counts = Counter(row["meta"]["rejected_source"] for row in compact)
    error_counts = Counter(error for row in compact for error in row["meta"]["error_types"])
    chosen_parse_ok = 0
    chosen_gt_equal = 0
    rejected_equal = 0
    normalized_chosen_rejected_equal = 0
    prompt_matches = 0
    candidate_matches = 0
    for row in compact:
        meta = row["meta"]
        source_row = train_by_id[meta["trace_id"]]
        gold = split_tools(source_row["工具"])
        chosen = json.loads(row["messages"][-1]["content"])
        chosen_parse_ok += int(isinstance(chosen, dict) and isinstance(chosen.get("tools"), list))
        chosen_gt_equal += int(chosen.get("tools") == gold)
        rejected_equal += int(row["messages"][-1]["content"] == row["rejected_response"])
        rejected_tools, rejected_valid, _ = strict_parse_tools(
            row["rejected_response"],
            catalog_ids,
            set(meta["candidate_tools"]) if meta["view"] == "rag" else None,
        )
        normalized_chosen_rejected_equal += int(
            rejected_valid and canonical_target(rejected_tools) == canonical_target(gold)
        )
        prompt_messages = row["messages"][:-1]
        prompt_matches += int(prompt_hash(prompt_messages) == meta["prompt_hash"])
        candidate_matches += int(sha256_json(meta["candidate_tools"]) == meta["candidate_hash"])
    train_trace_ids = {row["meta"]["trace_id"] for row in runtime_train}
    monitor_ids = {row["meta"]["trace_id"] for row in runtime_monitor}
    audit = {
        "strict_ok": True,
        "quality_limited_exception": len(compact) < args.min_pairs,
        "requested_range": [args.min_pairs, args.max_pairs],
        "final_pairs": len(compact),
        "core_hard_errors": core_count,
        "core_hard_error_ratio": core_count / max(1, len(compact)),
        "supplement_ratio": (len(compact) - core_count) / max(1, len(compact)),
        "synthetic_pairs": synthetic_count,
        "max_pairs_per_query": max(per_query.values(), default=0),
        "chosen_parse_rate": chosen_parse_ok / max(1, len(compact)),
        "chosen_equals_gt_rate": chosen_gt_equal / max(1, len(compact)),
        "chosen_unknown_tools": sum(
            tool_id not in catalog_ids
            for row in compact
            for tool_id in json.loads(row["messages"][-1]["content"])["tools"]
        ),
        "chosen_rejected_equal": rejected_equal,
        "normalized_chosen_rejected_equal": normalized_chosen_rejected_equal,
        "prompt_hash_match_rate": prompt_matches / max(1, len(compact)),
        "candidate_hash_match_rate": candidate_matches / max(1, len(compact)),
        "dev_trace_overlap": len({row["meta"]["trace_id"] for row in compact} & {row["id"] for row in dev}),
        "frozen_trace_overlap": len({row["meta"]["trace_id"] for row in compact} & {row["id"] for row in frozen}),
        "dev_query_overlap": len(
            {
                normalize_question(train_by_id[row["meta"]["trace_id"]]["问题"])
                for row in compact
            }
            & {normalize_question(row["问题"]) for row in dev}
        ),
        "frozen_query_overlap": len(
            {
                normalize_question(train_by_id[row["meta"]["trace_id"]]["问题"])
                for row in compact
            }
            & {normalize_question(row["问题"]) for row in frozen}
        ),
        "extra_blocked_trace_overlap": len(
            {row["meta"]["trace_id"] for row in compact}
            & {row["id"] for row in extra_blocked}
        ),
        "extra_blocked_query_overlap": len(
            {
                normalize_question(train_by_id[row["meta"]["trace_id"]]["问题"])
                for row in compact
            }
            & {normalize_question(row["问题"]) for row in extra_blocked}
        ),
        "train_monitor_trace_overlap": len(train_trace_ids & monitor_ids),
        "train_pairs": len(runtime_train),
        "monitor_pairs": len(runtime_monitor),
        "unique_trace_ids": len({row["meta"]["trace_id"] for row in compact}),
        "rejected_source_counts": dict(source_counts),
        "error_type_counts": dict(error_counts),
        "tool_coverage": len(
            {
                tool_id
                for row in compact
                for tool_id in json.loads(row["messages"][-1]["content"])["tools"]
            }
        ),
        "missing_tools": sorted(
            catalog_ids
            - {
                tool_id
                for row in compact
                for tool_id in json.loads(row["messages"][-1]["content"])["tools"]
            }
        ),
        "core_pool_tool_coverage": len(
            {tool_id for row in real_pool if row["core_hard_error"] for tool_id in row["gold_tools"]}
        ),
        "tool_count_distribution": dict(Counter(row["meta"]["tool_count"] for row in compact)),
        "single_multi_distribution": dict(
            Counter(
                "single" if row["meta"]["tool_count"] == 1 else "multi"
                for row in compact
            )
        ),
        "combo_distribution": dict(
            Counter(row["meta"]["combo_seen_bucket"] for row in compact)
        ),
        "domain_distribution": dict(Counter(domain for row in compact for domain in row["meta"]["domains"])),
        "view_distribution": dict(Counter(row["meta"]["view"] for row in compact)),
        "chosen_length_chars": summary_stats(chosen_lengths),
        "rejected_length_chars": summary_stats(rejected_lengths),
        "length_delta_chars": summary_stats(
            rejected - chosen for rejected, chosen in zip(rejected_lengths, chosen_lengths)
        ),
        "order_only_without_evidence": sum(
            "order-only" in row["meta"]["error_types"] and not row["meta"]["order_identifiable"]
            for row in compact
        ),
        "input_prediction_hashes": input_hashes,
        "output_hashes": {
            str(args.out): sha256_file(args.out),
            str(args.runtime_all_out): sha256_file(args.runtime_all_out),
            str(args.runtime_train_out): sha256_file(args.runtime_train_out),
            str(args.runtime_monitor_out): sha256_file(args.runtime_monitor_out),
        },
        "raw_pool": {
            "total": len(real_pool),
            "core": sum(row["core_hard_error"] for row in real_pool),
            "supplement": sum(not row["core_hard_error"] for row in real_pool),
        },
        "selection_policy": {
            "priority": ["selected_sft", "early_sft", "other_real"],
            "synthetic_allowed": args.allow_synthetic,
            "min_core_ratio": args.min_core_ratio,
            "max_per_query": args.max_per_query,
        },
    }
    audit["strict_ok"] = (
        bool(compact)
        and len(compact) <= args.max_pairs
        and audit["core_hard_error_ratio"] >= args.min_core_ratio
        and audit["supplement_ratio"] <= 1.0 - args.min_core_ratio + 1e-12
        and audit["max_pairs_per_query"] <= args.max_per_query
        and audit["chosen_parse_rate"] == 1.0
        and audit["chosen_equals_gt_rate"] == 1.0
        and audit["chosen_unknown_tools"] == 0
        and audit["chosen_rejected_equal"] == 0
        and audit["normalized_chosen_rejected_equal"] == 0
        and audit["prompt_hash_match_rate"] == 1.0
        and audit["candidate_hash_match_rate"] == 1.0
        and audit["dev_trace_overlap"] == 0
        and audit["frozen_trace_overlap"] == 0
        and audit["dev_query_overlap"] == 0
        and audit["frozen_query_overlap"] == 0
        and audit["extra_blocked_trace_overlap"] == 0
        and audit["extra_blocked_query_overlap"] == 0
        and audit["train_monitor_trace_overlap"] == 0
        and audit["order_only_without_evidence"] == 0
        and audit["tool_coverage"] >= audit["core_pool_tool_coverage"]
    )
    write_json(args.audit_out, audit)
    print(
        f"DPO pairs={len(compact)} core={core_count} ({audit['core_hard_error_ratio']:.3f}) "
        f"synthetic={synthetic_count} strict_ok={audit['strict_ok']}"
    )
    return 0 if audit["strict_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
