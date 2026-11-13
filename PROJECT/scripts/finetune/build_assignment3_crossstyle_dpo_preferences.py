#!/usr/bin/env python3
"""Build balanced cross-style DPO pairs from real train-side generations."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import (  # noqa: E402
    CATALOG_PATH,
    catalog_by_id,
    load_catalog,
    normalize_question,
)
from finetune.tool_trace_common import (  # noqa: E402
    canonical_target,
    prompt_hash,
    render_messages,
    sha256_file,
    sha256_json,
    stable_seed,
    strict_parse_tools,
    summary_stats,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXP = (
    ROOT
    / "PROJECT/data/tool_call_trace/_aux/assignment3"
    / "assignment3_crossstyle_rag10/dpo"
)
DEFAULT_RUNTIME = (
    ROOT / "PROJECT/finetune/outputs/tooltrace-data-cache/runs"
)
SOURCE_SPLITS = (
    "assignment3_api_native_train",
    "legacy_train",
)
TOOL_COUNT_WEIGHTS = {
    1: 0.10,
    2: 0.15,
    3: 0.20,
    4: 0.20,
    5: 0.20,
    6: 0.15,
}


def parse_source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("prediction source must be NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("prediction source must be NAME=PATH")
    return name, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compact",
        type=Path,
        default=ROOT
        / "PROJECT/data/tool_call_trace/_aux/assignment3"
        / "assignment3_crossstyle_rag10/sft_compact.jsonl",
    )
    parser.add_argument("--predictions", action="append", type=parse_source, required=True)
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    parser.add_argument(
        "--legacy-frozen",
        type=Path,
        default=ROOT / "PROJECT/data/tool_call_trace/test.jsonl",
    )
    parser.add_argument(
        "--api-frozen",
        type=Path,
        default=ROOT
        / "PROJECT/data/tool_call_trace/_aux/assignment3"
        / "assignment3_100tool_api/test_api/test.jsonl",
    )
    parser.add_argument(
        "--selection-dev",
        type=Path,
        default=ROOT
        / "PROJECT/data/tool_call_trace/_aux/assignment3"
        / "api_native_dev_2400/dev.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_EXP / "preferences_compact.jsonl",
    )
    parser.add_argument(
        "--audit-out",
        type=Path,
        default=DEFAULT_EXP / "preference_audit.json",
    )
    parser.add_argument(
        "--runtime-all-out",
        type=Path,
        default=DEFAULT_RUNTIME / "assignment3_crossstyle_dpo_all.runtime.jsonl",
    )
    parser.add_argument(
        "--runtime-train-out",
        type=Path,
        default=DEFAULT_RUNTIME / "assignment3_crossstyle_dpo_train.runtime.jsonl",
    )
    parser.add_argument(
        "--runtime-monitor-out",
        type=Path,
        default=DEFAULT_RUNTIME / "assignment3_crossstyle_dpo_monitor.runtime.jsonl",
    )
    parser.add_argument("--pairs", type=int, default=5000)
    parser.add_argument("--max-per-query", type=int, default=2)
    parser.add_argument("--min-semantic-ratio", type=float, default=0.90)
    parser.add_argument("--monitor-percent", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20_260_729)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(row)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    temporary.replace(path)


def quotas(total: int) -> dict[int, int]:
    raw = {
        tool_count: total * weight
        for tool_count, weight in TOOL_COUNT_WEIGHTS.items()
    }
    result = {tool_count: int(value) for tool_count, value in raw.items()}
    remaining = total - sum(result.values())
    ranking = sorted(
        raw,
        key=lambda tool_count: (
            -(raw[tool_count] - result[tool_count]),
            tool_count,
        ),
    )
    for tool_count in ranking[:remaining]:
        result[tool_count] += 1
    return result


def error_types(predicted: list[str], gold: list[str]) -> list[str]:
    predicted_set = set(predicted)
    gold_set = set(gold)
    errors: list[str] = []
    if gold_set - predicted_set:
        errors.append("omission")
    if predicted_set - gold_set:
        errors.append("extra")
    if (gold_set - predicted_set) and (predicted_set - gold_set):
        errors.append("substitution")
    if (
        predicted_set == gold_set
        and len(predicted) == len(gold)
        and predicted != gold
    ):
        errors.append("order-only")
    return errors


def order_identifiable(
    query: str,
    tools: list[str],
    catalog: dict[str, dict[str, Any]],
) -> tuple[bool, list[str]]:
    markers = [
        "先",
        "再",
        "然后",
        "最后",
        "接着",
        "随后",
        "按顺序",
        "依次",
        "起初",
        "接下来",
        "收尾",
    ]
    evidence = [marker for marker in markers if marker in query]
    tool_set = set(tools)
    for tool_id in tools:
        for dependency in catalog[tool_id].get("depends_on", []):
            if dependency in tool_set:
                evidence.append(f"depends_on:{dependency}->{tool_id}")
    return bool(evidence), evidence


def source_priority(name: str) -> int:
    lowered = name.lower()
    if "current" in lowered or "selected" in lowered:
        return 0
    if "previous" in lowered:
        return 1
    if "early" in lowered:
        return 2
    return 3


def blocked_index(paths: list[Path]) -> tuple[set[str], set[str]]:
    ids: set[str] = set()
    queries: set[str] = set()
    for path in paths:
        for row in read_jsonl(path):
            ids.add(str(row["id"]))
            queries.add(normalize_question(str(row["问题"])))
    return ids, queries


def main() -> int:
    args = parse_args()
    if args.pairs <= 0 or args.pairs % len(SOURCE_SPLITS):
        raise ValueError("--pairs must be positive and divisible by two")
    if not 0 < args.monitor_percent < 100:
        raise ValueError("--monitor-percent must be between 1 and 99")

    catalog_rows = load_catalog(args.catalog)
    catalog = catalog_by_id(catalog_rows)
    catalog_ids = set(catalog)
    compact_rows = read_jsonl(args.compact)
    compact: dict[str, dict[str, Any]] = {}
    for row in compact_rows:
        trace_id = str(row["trace_id"])
        if trace_id in compact:
            raise ValueError(f"duplicate compact trace id: {trace_id}")
        if row["source_split"] not in SOURCE_SPLITS:
            raise ValueError(f"unexpected source split: {row['source_split']}")
        if row.get("view") != "rag" or int(row.get("candidate_k", 0)) != 10:
            raise ValueError(f"{trace_id}: expected RAG Top-10")
        if not set(row["gold_tools"]).issubset(row["candidate_tools"]):
            raise ValueError(f"{trace_id}: gold missing from candidates")
        compact[trace_id] = row

    pool: list[dict[str, Any]] = []
    input_hashes: dict[str, str] = {}
    duplicate_generation_keys = 0
    generation_keys: set[tuple[str, str]] = set()
    for source_name, prediction_path in args.predictions:
        if not prediction_path.is_file():
            raise ValueError(f"missing predictions: {prediction_path}")
        prediction_sha = sha256_file(prediction_path)
        input_hashes[f"{source_name}:{prediction_path}"] = prediction_sha
        for prediction in read_jsonl(prediction_path):
            trace_id = str(prediction.get("id", ""))
            source_row = compact.get(trace_id)
            if source_row is None:
                raise ValueError(
                    f"{source_name}: prediction is not in cross-style train: {trace_id}"
                )
            if prediction.get("query") != source_row["query"]:
                raise ValueError(f"{source_name}: query mismatch: {trace_id}")
            if prediction.get("view") != "rag":
                raise ValueError(f"{source_name}: non-RAG prediction: {trace_id}")
            candidates = prediction.get("candidate_tools")
            if candidates != source_row["candidate_tools"]:
                raise ValueError(f"{source_name}: candidate mismatch: {trace_id}")
            messages = render_messages(
                source_row["query"],
                candidates,
                catalog,
                "rag",
            )
            expected_prompt_hash = prompt_hash(messages)
            if prediction.get("prompt_hash") != expected_prompt_hash:
                raise ValueError(f"{source_name}: prompt hash mismatch: {trace_id}")

            rejected = prediction.get("raw_output", "")
            if not isinstance(rejected, str) or not rejected:
                continue
            generation_key = (trace_id, rejected)
            if generation_key in generation_keys:
                duplicate_generation_keys += 1
                continue
            generation_keys.add(generation_key)

            gold = list(source_row["gold_tools"])
            predicted, strict_valid, parse_error = strict_parse_tools(
                rejected,
                catalog_ids,
                set(candidates),
            )
            semantic_errors = error_types(predicted, gold)
            if strict_valid and predicted == gold:
                continue
            if strict_valid and set(predicted) == set(gold):
                identifiable, evidence = order_identifiable(
                    source_row["query"],
                    gold,
                    catalog,
                )
                if not identifiable:
                    continue
            else:
                identifiable, evidence = order_identifiable(
                    source_row["query"],
                    gold,
                    catalog,
                )
            errors = list(semantic_errors)
            if not strict_valid:
                errors.insert(0, parse_error or "format")
            semantic_pair = bool(semantic_errors)
            if not errors:
                continue
            pool.append(
                {
                    "trace_id": trace_id,
                    "source_split": source_row["source_split"],
                    "query": source_row["query"],
                    "gold_tools": gold,
                    "candidate_tools": candidates,
                    "tool_count": int(source_row["tool_count"]),
                    "domain": source_row.get("domain"),
                    "combo_seen_bucket": source_row.get("combo_seen_bucket"),
                    "messages": messages,
                    "chosen": canonical_target(gold),
                    "rejected": rejected,
                    "rejected_tools": predicted,
                    "strict_rejected": strict_valid,
                    "parse_error": parse_error,
                    "error_types": errors,
                    "semantic_pair": semantic_pair,
                    "order_identifiable": identifiable,
                    "order_evidence": evidence,
                    "rejected_source": source_name,
                    "prediction_path": str(prediction_path),
                    "prediction_sha256": prediction_sha,
                    "sample_index": int(prediction.get("sample_index", 0)),
                    "prompt_hash": expected_prompt_hash,
                    "candidate_hash": sha256_json(candidates),
                    "record_hash": source_row.get("record_hash", ""),
                }
            )

    def pool_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
        return (
            not row["semantic_pair"],
            source_priority(row["rejected_source"]),
            stable_seed(
                args.seed,
                row["source_split"],
                row["tool_count"],
                row["trace_id"],
                row["rejected_source"],
                row["sample_index"],
                row["rejected"],
            ),
        )

    buckets: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in pool:
        buckets[(row["source_split"], row["tool_count"])].append(row)
    for rows in buckets.values():
        rows.sort(key=pool_sort_key)

    per_source_target = args.pairs // len(SOURCE_SPLITS)
    tool_targets = quotas(per_source_target)
    selected: list[dict[str, Any]] = []
    selected_keys: set[int] = set()
    per_query: Counter[str] = Counter()

    def add_from(rows: list[dict[str, Any]], target: int) -> int:
        added = 0
        for row in rows:
            if added >= target:
                break
            if id(row) in selected_keys:
                continue
            if per_query[row["trace_id"]] >= args.max_per_query:
                continue
            selected.append(row)
            selected_keys.add(id(row))
            per_query[row["trace_id"]] += 1
            added += 1
        return added

    source_shortfalls: dict[str, int] = {}
    for source_split in SOURCE_SPLITS:
        source_before = len(selected)
        for tool_count, target in sorted(tool_targets.items()):
            add_from(buckets[(source_split, tool_count)], target)
        source_selected = len(selected) - source_before
        if source_selected < per_source_target:
            remainder = sorted(
                (
                    row
                    for row in pool
                    if row["source_split"] == source_split
                    and id(row) not in selected_keys
                ),
                key=pool_sort_key,
            )
            add_from(remainder, per_source_target - source_selected)
        source_shortfalls[source_split] = (
            per_source_target - (len(selected) - source_before)
        )

    if any(source_shortfalls.values()):
        write_json(
            args.audit_out,
            {
                "status": "failed",
                "strict_ok": False,
                "reason": "insufficient_real_error_pairs",
                "requested_pairs": args.pairs,
                "source_shortfalls": source_shortfalls,
                "raw_pool_rows": len(pool),
                "raw_pool_by_source": dict(
                    Counter(row["source_split"] for row in pool)
                ),
                "raw_pool_by_negative_source": dict(
                    Counter(row["rejected_source"] for row in pool)
                ),
            },
        )
        raise SystemExit(
            f"insufficient real errors for balanced selection: {source_shortfalls}"
        )

    selected.sort(
        key=lambda row: stable_seed(
            args.seed,
            "final-preference-order",
            row["trace_id"],
            row["rejected_source"],
            row["sample_index"],
            row["rejected"],
        )
    )
    runtime_all = []
    for row in selected:
        meta = {
            "trace_id": row["trace_id"],
            "source_split": row["source_split"],
            "view": "rag",
            "candidate_tools": row["candidate_tools"],
            "candidate_k": 10,
            "rejected_source": row["rejected_source"],
            "prediction_path": row["prediction_path"],
            "prediction_sha256": row["prediction_sha256"],
            "sample_index": row["sample_index"],
            "error_types": row["error_types"],
            "semantic_pair": row["semantic_pair"],
            "strict_rejected": row["strict_rejected"],
            "prompt_hash": row["prompt_hash"],
            "candidate_hash": row["candidate_hash"],
            "source_record_hash": row["record_hash"],
            "order_identifiable": row["order_identifiable"],
            "order_evidence": row["order_evidence"],
            "tool_count": row["tool_count"],
            "domain": row["domain"],
            "combo_seen_bucket": row["combo_seen_bucket"],
            "synthetic": False,
        }
        runtime_all.append(
            {
                "messages": row["messages"]
                + [{"role": "assistant", "content": row["chosen"]}],
                "rejected_response": row["rejected"],
                "meta": meta,
            }
        )
    write_jsonl(args.out, runtime_all)
    write_jsonl(args.runtime_all_out, runtime_all)

    monitor_ids = {
        trace_id
        for trace_id in per_query
        if stable_seed(args.seed, "dpo-monitor", trace_id) % 100
        < args.monitor_percent
    }
    runtime_train = [
        row for row in runtime_all if row["meta"]["trace_id"] not in monitor_ids
    ]
    runtime_monitor = [
        row for row in runtime_all if row["meta"]["trace_id"] in monitor_ids
    ]
    write_jsonl(args.runtime_train_out, runtime_train)
    write_jsonl(args.runtime_monitor_out, runtime_monitor)

    blocked_paths = [args.legacy_frozen, args.api_frozen, args.selection_dev]
    blocked_ids, blocked_queries = blocked_index(blocked_paths)
    selected_ids = {row["meta"]["trace_id"] for row in runtime_all}
    selected_queries = {
        normalize_question(compact[trace_id]["query"]) for trace_id in selected_ids
    }
    source_trace_ids = {
        str(compact[trace_id].get("construction", {}).get("source_trace_id"))
        for trace_id in selected_ids
        if compact[trace_id].get("construction", {}).get("source_trace_id")
    }
    leakage = {
        "blocked_id_overlap": len(selected_ids & blocked_ids),
        "blocked_query_overlap": len(selected_queries & blocked_queries),
        "api_source_trace_blocked_id_overlap": len(source_trace_ids & blocked_ids),
    }
    train_ids = {row["meta"]["trace_id"] for row in runtime_train}
    monitor_trace_ids = {row["meta"]["trace_id"] for row in runtime_monitor}
    semantic_count = sum(row["meta"]["semantic_pair"] for row in runtime_all)
    chosen_lengths = [len(row["messages"][-1]["content"]) for row in runtime_all]
    rejected_lengths = [len(row["rejected_response"]) for row in runtime_all]
    represented_tools = {
        tool_id
        for row in runtime_all
        for tool_id in json.loads(row["messages"][-1]["content"])["tools"]
    }
    source_counts = Counter(row["meta"]["source_split"] for row in runtime_all)
    prompt_matches = sum(
        prompt_hash(row["messages"][:-1]) == row["meta"]["prompt_hash"]
        for row in runtime_all
    )
    candidate_matches = sum(
        sha256_json(row["meta"]["candidate_tools"])
        == row["meta"]["candidate_hash"]
        for row in runtime_all
    )
    normalized_equal = 0
    for row in runtime_all:
        rejected_tools, valid, _ = strict_parse_tools(
            row["rejected_response"],
            catalog_ids,
            set(row["meta"]["candidate_tools"]),
        )
        chosen_tools = json.loads(row["messages"][-1]["content"])["tools"]
        normalized_equal += int(valid and rejected_tools == chosen_tools)

    audit = {
        "status": "complete",
        "strict_ok": True,
        "protocol": {
            "pairs": args.pairs,
            "per_source_target": per_source_target,
            "tool_count_weights": TOOL_COUNT_WEIGHTS,
            "target_tool_count_quota_per_source": tool_targets,
            "max_per_query": args.max_per_query,
            "min_semantic_ratio": args.min_semantic_ratio,
            "monitor_percent": args.monitor_percent,
            "real_generations_only": True,
            "synthetic_negatives_allowed": False,
            "selection_priority": [
                "semantic_error_before_format_only",
                "current_sft_before_previous_sft_before_early_or_other_models",
                "deterministic_seeded_order",
            ],
        },
        "counts": {
            "final_pairs": len(runtime_all),
            "train_pairs": len(runtime_train),
            "monitor_pairs": len(runtime_monitor),
            "unique_trace_ids": len(selected_ids),
            "max_pairs_per_query": max(per_query.values(), default=0),
            "semantic_pairs": semantic_count,
            "semantic_pair_ratio": semantic_count / max(1, len(runtime_all)),
            "synthetic_pairs": 0,
            "by_source": dict(sorted(source_counts.items())),
            "by_negative_source": dict(
                sorted(
                    Counter(
                        row["meta"]["rejected_source"] for row in runtime_all
                    ).items()
                )
            ),
            "by_source_and_tool_count": dict(
                sorted(
                    Counter(
                        f"{row['meta']['source_split']}:{row['meta']['tool_count']}"
                        for row in runtime_all
                    ).items()
                )
            ),
            "error_types": dict(
                sorted(
                    Counter(
                        error
                        for row in runtime_all
                        for error in row["meta"]["error_types"]
                    ).items()
                )
            ),
            "strict_rejected": sum(
                row["meta"]["strict_rejected"] for row in runtime_all
            ),
            "tool_coverage": len(represented_tools),
            "missing_tools": sorted(catalog_ids - represented_tools),
        },
        "raw_pool": {
            "rows": len(pool),
            "semantic_rows": sum(row["semantic_pair"] for row in pool),
            "by_source": dict(
                sorted(Counter(row["source_split"] for row in pool).items())
            ),
            "by_negative_source": dict(
                sorted(Counter(row["rejected_source"] for row in pool).items())
            ),
            "duplicate_generation_keys_dropped": duplicate_generation_keys,
        },
        "quality": {
            "chosen_parse_rate": 1.0,
            "chosen_equals_training_gold_rate": 1.0,
            "chosen_rejected_normalized_equal": normalized_equal,
            "prompt_hash_match_rate": prompt_matches / max(1, len(runtime_all)),
            "candidate_hash_match_rate": candidate_matches
            / max(1, len(runtime_all)),
            "order_only_without_evidence": sum(
                "order-only" in row["meta"]["error_types"]
                and not row["meta"]["order_identifiable"]
                for row in runtime_all
            ),
            "chosen_length_chars": summary_stats(chosen_lengths),
            "rejected_length_chars": summary_stats(rejected_lengths),
            "length_delta_chars": summary_stats(
                rejected - chosen
                for rejected, chosen in zip(rejected_lengths, chosen_lengths)
            ),
        },
        "isolation": {
            **leakage,
            "train_monitor_trace_overlap": len(train_ids & monitor_trace_ids),
            "frozen_or_dev_rows_used_for_pair_construction": 0,
            "blocked_paths": [str(path) for path in blocked_paths],
        },
        "hashes": {
            "inputs": {
                str(args.compact): sha256_file(args.compact),
                str(args.catalog): sha256_file(args.catalog),
                **input_hashes,
                **{str(path): sha256_file(path) for path in blocked_paths},
            },
            "outputs": {
                str(args.out): sha256_file(args.out),
                str(args.runtime_all_out): sha256_file(args.runtime_all_out),
                str(args.runtime_train_out): sha256_file(args.runtime_train_out),
                str(args.runtime_monitor_out): sha256_file(
                    args.runtime_monitor_out
                ),
            },
        },
    }
    audit["strict_ok"] = (
        len(runtime_all) == args.pairs
        and len(runtime_train) > 0
        and len(runtime_monitor) > 0
        and all(
            source_counts[source] == per_source_target
            for source in SOURCE_SPLITS
        )
        and audit["counts"]["semantic_pair_ratio"] >= args.min_semantic_ratio
        and audit["counts"]["synthetic_pairs"] == 0
        and audit["counts"]["max_pairs_per_query"] <= args.max_per_query
        and audit["counts"]["tool_coverage"] == len(catalog_ids)
        and normalized_equal == 0
        and audit["quality"]["prompt_hash_match_rate"] == 1.0
        and audit["quality"]["candidate_hash_match_rate"] == 1.0
        and audit["quality"]["order_only_without_evidence"] == 0
        and not any(leakage.values())
        and audit["isolation"]["train_monitor_trace_overlap"] == 0
    )
    audit["status"] = "complete" if audit["strict_ok"] else "failed"
    write_json(args.audit_out, audit)
    print(
        f"DPO pairs={len(runtime_all)} "
        f"semantic={audit['counts']['semantic_pair_ratio']:.3f} "
        f"api={source_counts['assignment3_api_native_train']} "
        f"legacy={source_counts['legacy_train']} "
        f"strict_ok={audit['strict_ok']}"
    )
    return 0 if audit["strict_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
