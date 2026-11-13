#!/usr/bin/env python3
"""Build DPO-v2 pairs only from strict semantic errors of audited parent checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Hashable, Iterable

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


DEFAULT_SOURCE_SPLITS = ("assignment3_api_native_train", "legacy_train")
TOOL_COUNT_WEIGHTS = {1: 0.10, 2: 0.15, 3: 0.20, 4: 0.20, 5: 0.20, 6: 0.15}
ERROR_TYPES = ("substitution", "omission_only", "extra_only", "order_only")


def parse_source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("prediction source must be NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("prediction source must be NAME=PATH")
    return name, Path(path)


def parse_tool_count_weight(value: str) -> tuple[int, float]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("tool-count weight must be COUNT=WEIGHT")
    raw_count, raw_weight = value.split("=", 1)
    try:
        tool_count = int(raw_count)
        weight = float(raw_weight)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "tool-count weight must be COUNT=WEIGHT"
        ) from exc
    if tool_count not in range(1, 7) or weight <= 0:
        raise argparse.ArgumentTypeError(
            "tool count must be 1..6 and weight must be positive"
        )
    return tool_count, weight


def parse_error_type_weight(value: str) -> tuple[str, float]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("error-type weight must be TYPE=WEIGHT")
    error_type, raw_weight = value.split("=", 1)
    try:
        weight = float(raw_weight)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "error-type weight must be TYPE=WEIGHT"
        ) from exc
    if error_type not in ERROR_TYPES or weight <= 0:
        raise argparse.ArgumentTypeError(
            f"error type must be one of {','.join(ERROR_TYPES)} and weight positive"
        )
    return error_type, weight


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


def quotas(
    total: int, weights: dict[Hashable, float]
) -> dict[Hashable, int]:
    weight_sum = sum(weights.values())
    raw = {
        tool_count: total * weight / weight_sum
        for tool_count, weight in weights.items()
    }
    result = {tool_count: int(value) for tool_count, value in raw.items()}
    remainder = total - sum(result.values())
    order = sorted(
        raw,
        key=lambda tool_count: (
            -(raw[tool_count] - result[tool_count]),
            tool_count,
        ),
    )
    for tool_count in order[:remainder]:
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
    if predicted_set == gold_set and predicted != gold:
        errors.append("order-only")
    return errors


def primary_error_type(errors: list[str]) -> str:
    if "substitution" in errors:
        return "substitution"
    if "omission" in errors:
        return "omission_only"
    if "extra" in errors:
        return "extra_only"
    if "order-only" in errors:
        return "order_only"
    raise ValueError(f"unsupported semantic error combination: {errors}")


def order_identifiable(
    query: str,
    tools: list[str],
    catalog: dict[str, dict[str, Any]],
) -> tuple[bool, list[str]]:
    markers = (
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
    )
    evidence = [marker for marker in markers if marker in query]
    tool_set = set(tools)
    for tool_id in tools:
        for dependency in catalog[tool_id].get("depends_on", []):
            if dependency in tool_set:
                evidence.append(f"depends_on:{dependency}->{tool_id}")
    return bool(evidence), evidence


def error_priority(errors: list[str]) -> int:
    if "substitution" in errors:
        return 0
    if "omission" in errors and "extra" in errors:
        return 1
    if "omission" in errors:
        return 2
    if "extra" in errors:
        return 3
    return 4


def protected_source_ids(paths: list[Path]) -> set[str]:
    result: set[str] = set()
    for path in paths:
        for row in read_jsonl(path):
            row_id = row.get("id")
            if row_id:
                result.add(str(row_id))
            construction = row.get("construction")
            if isinstance(construction, dict) and construction.get("source_trace_id"):
                result.add(str(construction["source_trace_id"]))
            elif not row_id:
                raise ValueError(
                    f"{path}: missing both id and construction.source_trace_id"
                )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument("--compact-extra", action="append", type=Path, default=[])
    parser.add_argument("--predictions", action="append", type=parse_source, required=True)
    parser.add_argument("--allowed-adapter", action="append", type=Path, required=True)
    parser.add_argument(
        "--protected-source-plan",
        action="append",
        type=Path,
        required=True,
    )
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit-out", type=Path, required=True)
    parser.add_argument("--runtime-train-out", type=Path, required=True)
    parser.add_argument("--runtime-monitor-out", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=2_000)
    parser.add_argument(
        "--source-split",
        action="append",
        default=[],
        help="Eligible compact source_split; defaults to API-native plus Legacy train.",
    )
    parser.add_argument(
        "--min-per-style",
        type=int,
        default=400,
        help="Minimum pair count for each requested source split; residual errors fill the remainder.",
    )
    parser.add_argument("--monitor-percent", type=int, default=10)
    parser.add_argument(
        "--exact-monitor-percent",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Select exactly the requested percentage by stable trace-id order.",
    )
    parser.add_argument(
        "--one-pair-per-source",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also deduplicate construction.source_trace_id across selected pairs.",
    )
    parser.add_argument("--max-pairs-per-query", type=int, default=1)
    parser.add_argument("--max-pairs-per-source", type=int, default=1)
    parser.add_argument(
        "--prefer-original-order",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Prioritize real errors generated with the deployed candidate order before order-perturbation errors.",
    )
    parser.add_argument(
        "--tool-count-weight",
        action="append",
        type=parse_tool_count_weight,
        default=[],
        help=(
            "Optional COUNT=WEIGHT entries. When supplied, only those tool "
            "counts are eligible and weights are normalized before quotas."
        ),
    )
    parser.add_argument(
        "--error-type-weight",
        action="append",
        type=parse_error_type_weight,
        default=[],
        help=(
            "Optional TYPE=WEIGHT entries for stratification inside each "
            "tool-count quota. Missing supply is filled by other real errors."
        ),
    )
    parser.add_argument("--seed", type=int, default=20_260_729)
    args = parser.parse_args()
    source_splits = tuple(args.source_split or DEFAULT_SOURCE_SPLITS)
    if args.tool_count_weight:
        tool_count_weights = dict(args.tool_count_weight)
        if len(tool_count_weights) != len(args.tool_count_weight):
            raise SystemExit("duplicate --tool-count-weight count")
    else:
        tool_count_weights = dict(TOOL_COUNT_WEIGHTS)
    if args.error_type_weight:
        error_type_weights = dict(args.error_type_weight)
        if len(error_type_weights) != len(args.error_type_weight):
            raise SystemExit("duplicate --error-type-weight type")
    else:
        error_type_weights = {}
    if not source_splits or len(source_splits) != len(set(source_splits)):
        raise SystemExit("--source-split values must be non-empty and unique")
    if not 2_000 <= args.pairs <= 5_000 or args.pairs % 2:
        raise SystemExit("--pairs must be an even value in the PDF range 2000..5000")
    if args.min_per_style <= 0 or len(source_splits) * args.min_per_style > args.pairs:
        raise SystemExit("--min-per-style must be positive and fit all source splits")
    if not 0 < args.monitor_percent < 100:
        raise SystemExit("--monitor-percent must be between 1 and 99")
    if args.max_pairs_per_query < 1 or args.max_pairs_per_source < 1:
        raise SystemExit("maximum pair counts must be positive")
    if args.one_pair_per_source and args.max_pairs_per_source != 1:
        raise SystemExit(
            "--one-pair-per-source conflicts with --max-pairs-per-source != 1"
        )

    catalog_rows = load_catalog(args.catalog)
    catalog = catalog_by_id(catalog_rows)
    catalog_ids = set(catalog)
    compact_paths = [args.compact, *args.compact_extra]
    compact_rows = [row for path in compact_paths for row in read_jsonl(path)]
    compact = {str(row["trace_id"]): row for row in compact_rows}
    if len(compact) != len(compact_rows):
        raise ValueError("duplicate compact trace ids")

    allowed_adapters = {str(path.resolve()) for path in args.allowed_adapter}
    if any(not path.is_dir() for path in args.allowed_adapter):
        raise ValueError("one or more allowed parent adapters are missing")
    protected_sources = protected_source_ids(args.protected_source_plan)

    pool: list[dict[str, Any]] = []
    generation_keys: set[tuple[str, str, str]] = set()
    prediction_hashes: dict[str, str] = {}
    rejected_non_parent = 0
    excluded_protected_generation_rows = 0
    for source_name, prediction_path in args.predictions:
        if not prediction_path.is_file():
            raise ValueError(f"missing predictions: {prediction_path}")
        prediction_hashes[f"{source_name}:{prediction_path}"] = sha256_file(
            prediction_path
        )
        for prediction in read_jsonl(prediction_path):
            trace_id = str(prediction.get("id", ""))
            source = compact.get(trace_id)
            if source is None:
                raise ValueError(f"{source_name}: unknown train trace {trace_id}")
            adapter = str(Path(str(prediction.get("adapter", ""))).resolve())
            if adapter not in allowed_adapters:
                rejected_non_parent += 1
                continue
            if prediction.get("query") != source["query"]:
                raise ValueError(f"{source_name}:{trace_id}: query mismatch")
            candidates = prediction.get("candidate_tools")
            if (
                prediction.get("view") != "rag"
                or not isinstance(candidates, list)
                or len(candidates) != 10
                or len(candidates) != len(set(candidates))
                or set(candidates) != set(source["candidate_tools"])
            ):
                raise ValueError(f"{source_name}:{trace_id}: candidate protocol mismatch")
            messages = render_messages(source["query"], candidates, catalog, "rag")
            expected_prompt_hash = prompt_hash(messages)
            if prediction.get("prompt_hash") != expected_prompt_hash:
                raise ValueError(f"{source_name}:{trace_id}: prompt hash mismatch")
            rejected = prediction.get("raw_output")
            if not isinstance(rejected, str) or not rejected:
                continue
            generation_key = (trace_id, expected_prompt_hash, rejected)
            if generation_key in generation_keys:
                continue
            generation_keys.add(generation_key)

            predicted, strict_valid, parse_error = strict_parse_tools(
                rejected,
                catalog_ids,
                set(candidates),
            )
            if not strict_valid:
                continue
            gold = list(source["gold_tools"])
            if not set(gold) <= set(candidates):
                continue
            errors = error_types(predicted, gold)
            if not errors:
                continue
            identifiable, evidence = order_identifiable(
                str(source["query"]),
                gold,
                catalog,
            )
            if errors == ["order-only"] and not identifiable:
                continue
            source_trace_id = str(
                source.get("construction", {}).get("source_trace_id", trace_id)
            )
            if trace_id in protected_sources or source_trace_id in protected_sources:
                excluded_protected_generation_rows += 1
                continue
            if int(source["tool_count"]) not in tool_count_weights:
                continue
            pool.append(
                {
                    "trace_id": trace_id,
                    "source_split": str(source["source_split"]),
                    "source_trace_id": source_trace_id,
                    "query": source["query"],
                    "gold_tools": gold,
                    "candidate_tools": candidates,
                    "tool_count": int(source["tool_count"]),
                    "domain": source.get("domain"),
                    "combo_seen_bucket": source.get("combo_seen_bucket"),
                    "messages": messages,
                    "chosen": canonical_target(gold),
                    "rejected": rejected,
                    "rejected_tools": predicted,
                    "error_types": errors,
                    "primary_error_type": primary_error_type(errors),
                    "order_identifiable": identifiable,
                    "order_evidence": evidence,
                    "rejected_source": source_name,
                    "prediction_path": str(prediction_path),
                    "prediction_sha256": prediction_hashes[
                        f"{source_name}:{prediction_path}"
                    ],
                    "sample_index": int(prediction.get("sample_index", 0)),
                    "prompt_hash": expected_prompt_hash,
                    "candidate_hash": sha256_json(candidates),
                    "candidate_order_variant": str(
                        prediction.get("candidate_order_variant", "original")
                    ),
                    "parent_adapter": str(prediction.get("adapter", "")),
                    "record_hash": source.get("record_hash", ""),
                }
            )

    def sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
        return (
            0
            if not args.prefer_original_order
            or row["candidate_order_variant"] == "original"
            else 1,
            error_priority(row["error_types"]),
            stable_seed(
                args.seed,
                row["source_split"],
                row["tool_count"],
                row["candidate_order_variant"],
                row["trace_id"],
                row["rejected_source"],
                row["sample_index"],
                row["rejected"],
            ),
        )

    buckets: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    error_buckets: dict[
        tuple[str, int, str], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in pool:
        if row["source_split"] not in source_splits:
            raise ValueError(f"unexpected source split {row['source_split']}")
        buckets[(row["source_split"], row["tool_count"])].append(row)
        error_buckets[
            (
                row["source_split"],
                row["tool_count"],
                row["primary_error_type"],
            )
        ].append(row)
    for rows in buckets.values():
        rows.sort(key=sort_key)
    for rows in error_buckets.values():
        rows.sort(key=sort_key)

    selected: list[dict[str, Any]] = []
    selected_objects: set[int] = set()
    selected_queries: set[str] = set()
    selected_sources: set[str] = set()
    selected_query_counts: Counter[str] = Counter()
    selected_source_counts: Counter[str] = Counter()

    def add(
        rows: list[dict[str, Any]],
        target: int,
        *,
        allow_repeated_query_or_source: bool = False,
    ) -> int:
        added = 0
        for row in rows:
            if added >= target:
                break
            if id(row) in selected_objects:
                continue
            query_count = selected_query_counts[row["trace_id"]]
            source_count = selected_source_counts[row["source_trace_id"]]
            if not allow_repeated_query_or_source and (
                query_count or source_count
            ):
                continue
            if (
                query_count >= args.max_pairs_per_query
                or source_count >= args.max_pairs_per_source
            ):
                continue
            selected.append(row)
            selected_objects.add(id(row))
            selected_queries.add(row["trace_id"])
            selected_sources.add(row["source_trace_id"])
            selected_query_counts[row["trace_id"]] += 1
            selected_source_counts[row["source_trace_id"]] += 1
            added += 1
        return added

    target_by_count = quotas(args.min_per_style, tool_count_weights)
    shortfalls: dict[str, int] = {}
    for source_split in source_splits:
        before = len(selected)
        for tool_count, target in sorted(target_by_count.items()):
            count_before = len(selected)
            if error_type_weights:
                target_by_error = quotas(target, error_type_weights)
                for error_type, error_target in sorted(target_by_error.items()):
                    add(
                        error_buckets[(source_split, tool_count, error_type)],
                        error_target,
                    )
            count_current = len(selected) - count_before
            if count_current < target:
                add(
                    buckets[(source_split, tool_count)],
                    target - count_current,
                )
        current = len(selected) - before
        if current < args.min_per_style:
            remainder = sorted(
                (
                    row
                    for row in pool
                    if row["source_split"] == source_split
                    and id(row) not in selected_objects
                    and row["trace_id"] not in selected_queries
                    and (
                        not args.one_pair_per_source
                        or row["source_trace_id"] not in selected_sources
                    )
                ),
                key=sort_key,
            )
            add(remainder, args.min_per_style - current)
        current = len(selected) - before
        if current < args.min_per_style:
            repeated_remainder = sorted(
                (
                    row
                    for row in pool
                    if row["source_split"] == source_split
                    and id(row) not in selected_objects
                ),
                key=sort_key,
            )
            add(
                repeated_remainder,
                args.min_per_style - current,
                allow_repeated_query_or_source=True,
            )
        shortfalls[source_split] = args.min_per_style - (len(selected) - before)

    if any(shortfalls.values()):
        write_json(
            args.audit_out,
            {
                "status": "failed",
                "strict_ok": False,
                "reason": "insufficient_current_parent_strict_semantic_errors",
                "requested_pairs": args.pairs,
                "shortfalls": shortfalls,
                "raw_pool_rows": len(pool),
                "raw_pool_unique_queries": len(
                    {row["trace_id"] for row in pool}
                ),
                "raw_pool_by_source": dict(
                    Counter(row["source_split"] for row in pool)
                ),
                "raw_pool_by_generator": dict(
                    Counter(row["rejected_source"] for row in pool)
                ),
                "raw_pool_by_candidate_order": dict(
                    Counter(row["candidate_order_variant"] for row in pool)
                ),
            },
        )
        raise SystemExit(f"insufficient real current-parent errors: {shortfalls}")

    remainder = sorted(
        (
            row
            for row in pool
            if id(row) not in selected_objects
            and row["trace_id"] not in selected_queries
            and (
                not args.one_pair_per_source
                or row["source_trace_id"] not in selected_sources
            )
        ),
        key=sort_key,
    )
    add(remainder, args.pairs - len(selected))
    if len(selected) < args.pairs:
        repeated_remainder = sorted(
            (row for row in pool if id(row) not in selected_objects),
            key=sort_key,
        )
        add(
            repeated_remainder,
            args.pairs - len(selected),
            allow_repeated_query_or_source=True,
        )
    if len(selected) < args.pairs:
        write_json(
            args.audit_out,
            {
                "status": "failed",
                "strict_ok": False,
                "reason": "insufficient_unique_current_parent_strict_semantic_errors",
                "requested_pairs": args.pairs,
                "selected_pairs": len(selected),
                "raw_pool_rows": len(pool),
                "raw_pool_unique_queries": len(
                    {row["trace_id"] for row in pool}
                ),
            },
        )
        raise SystemExit(
            f"insufficient unique real current-parent errors: {len(selected)}/{args.pairs}"
        )

    selected.sort(
        key=lambda row: stable_seed(
            args.seed,
            "final-order",
            row["trace_id"],
            row["rejected_source"],
            row["sample_index"],
        )
    )
    runtime_all = []
    for row in selected:
        runtime_all.append(
            {
                "messages": row["messages"]
                + [{"role": "assistant", "content": row["chosen"]}],
                "rejected_response": row["rejected"],
                "meta": {
                    "trace_id": row["trace_id"],
                    "source_trace_id": row["source_trace_id"],
                    "source_split": row["source_split"],
                    "view": "rag",
                    "candidate_tools": row["candidate_tools"],
                    "candidate_k": 10,
                    "candidate_order_variant": row["candidate_order_variant"],
                    "rejected_source": row["rejected_source"],
                    "parent_adapter": row["parent_adapter"],
                    "prediction_path": row["prediction_path"],
                    "prediction_sha256": row["prediction_sha256"],
                    "sample_index": row["sample_index"],
                    "error_types": row["error_types"],
                    "primary_error_type": row["primary_error_type"],
                    "semantic_pair": True,
                    "strict_rejected": True,
                    "prompt_hash": row["prompt_hash"],
                    "candidate_hash": row["candidate_hash"],
                    "source_record_hash": row["record_hash"],
                    "order_identifiable": row["order_identifiable"],
                    "order_evidence": row["order_evidence"],
                    "tool_count": row["tool_count"],
                    "domain": row["domain"],
                    "combo_seen_bucket": row["combo_seen_bucket"],
                    "synthetic": False,
                },
            }
        )
    write_jsonl(args.out, runtime_all)
    if args.exact_monitor_percent:
        monitor_count = round(len(runtime_all) * args.monitor_percent / 100)
        trace_counts = Counter(row["meta"]["trace_id"] for row in runtime_all)
        singleton_ids = sorted(
            (trace_id for trace_id, count in trace_counts.items() if count == 1),
            key=lambda trace_id: stable_seed(args.seed, "monitor", trace_id),
        )
        if len(singleton_ids) < monitor_count:
            raise ValueError("not enough singleton queries for an exact monitor split")
        monitor_ids = set(singleton_ids[:monitor_count])
    else:
        monitor_ids = {
            row["meta"]["trace_id"]
            for row in runtime_all
            if stable_seed(args.seed, "monitor", row["meta"]["trace_id"]) % 100
            < args.monitor_percent
        }
    runtime_train = [
        row for row in runtime_all if row["meta"]["trace_id"] not in monitor_ids
    ]
    runtime_monitor = [
        row for row in runtime_all if row["meta"]["trace_id"] in monitor_ids
    ]
    if not runtime_train or not runtime_monitor:
        raise ValueError("empty train or monitor preference split")
    write_jsonl(args.runtime_train_out, runtime_train)
    write_jsonl(args.runtime_monitor_out, runtime_monitor)

    represented_tools = {
        tool_id
        for row in runtime_all
        for tool_id in json.loads(row["messages"][-1]["content"])["tools"]
    }
    prompt_matches = sum(
        prompt_hash(row["messages"][:-1]) == row["meta"]["prompt_hash"]
        for row in runtime_all
    )
    candidate_matches = sum(
        sha256_json(row["meta"]["candidate_tools"])
        == row["meta"]["candidate_hash"]
        for row in runtime_all
    )
    train_ids = {row["meta"]["trace_id"] for row in runtime_train}
    monitor_trace_ids = {
        row["meta"]["trace_id"] for row in runtime_monitor
    }
    train_source_ids = {
        row["meta"]["source_trace_id"] for row in runtime_train
    }
    monitor_source_ids = {
        row["meta"]["source_trace_id"] for row in runtime_monitor
    }
    selected_source_ids = {
        row["meta"]["source_trace_id"] for row in runtime_all
    }
    strict_ok = (
        len(runtime_all) == args.pairs
        and max(selected_query_counts.values(), default=0)
        <= args.max_pairs_per_query
        and max(selected_source_counts.values(), default=0)
        <= args.max_pairs_per_source
        and len(represented_tools) == 100
        and prompt_matches == args.pairs
        and candidate_matches == args.pairs
        and not (train_ids & monitor_trace_ids)
        and not (train_source_ids & monitor_source_ids)
        and not (selected_source_ids & protected_sources)
        and all(
            sum(
                row["meta"]["source_split"] == source_split
                for row in runtime_all
            )
            >= args.min_per_style
            for source_split in source_splits
        )
    )
    audit = {
        "audit_version": "assignment3-crossstyle-dpo-v2-preference-v1",
        "status": "complete" if strict_ok else "failed",
        "strict_ok": strict_ok,
        "protocol": {
            "pdf_pair_range": [2000, 5000],
            "pairs": args.pairs,
            "minimum_pairs_per_style": args.min_per_style,
            "source_splits": list(source_splits),
            "tool_count_weights": tool_count_weights,
            "tool_count_quota_per_style": target_by_count,
            "error_type_weights_within_tool_count": error_type_weights,
            "remaining_pairs": "filled by the natural train-side residual-error distribution",
            "max_pairs_per_query": args.max_pairs_per_query,
            "max_pairs_per_source_trace": args.max_pairs_per_source,
            "diversity_policy": "select unique queries and sources first; use bounded repeats only when the real-error pool cannot reach the requested target",
            "allowed_parent_adapters_only": True,
            "strict_semantic_errors_only": True,
            "format_only_pairs_allowed": False,
            "synthetic_or_rule_generated_rejected_allowed": False,
            "candidate_order_changes_display_only_not_candidate_set": True,
            "prefer_deployed_original_candidate_order": args.prefer_original_order,
            "monitor_split_percent_by_trace_id": args.monitor_percent,
            "exact_monitor_count": args.exact_monitor_percent,
        },
        "counts": {
            "raw_current_parent_semantic_pool": len(pool),
            "final_pairs": len(runtime_all),
            "train_pairs": len(runtime_train),
            "monitor_pairs": len(runtime_monitor),
            "unique_trace_ids": len(selected_queries),
            "unique_source_trace_ids": len(selected_sources),
            "repeated_query_pairs": args.pairs - len(selected_queries),
            "repeated_source_pairs": args.pairs - len(selected_sources),
            "max_observed_pairs_per_query": max(
                selected_query_counts.values(), default=0
            ),
            "max_observed_pairs_per_source": max(
                selected_source_counts.values(), default=0
            ),
            "represented_tools": len(represented_tools),
            "by_style": dict(
                sorted(
                    Counter(
                        row["meta"]["source_split"] for row in runtime_all
                    ).items()
                )
            ),
            "by_tool_count": dict(
                sorted(
                    Counter(
                        str(row["meta"]["tool_count"]) for row in runtime_all
                    ).items()
                )
            ),
            "by_generator": dict(
                sorted(
                    Counter(
                        row["meta"]["rejected_source"] for row in runtime_all
                    ).items()
                )
            ),
            "by_parent_adapter": dict(
                sorted(
                    Counter(
                        row["meta"]["parent_adapter"] for row in runtime_all
                    ).items()
                )
            ),
            "by_candidate_order": dict(
                sorted(
                    Counter(
                        row["meta"]["candidate_order_variant"]
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
            "primary_error_types": dict(
                sorted(
                    Counter(
                        row["meta"]["primary_error_type"] for row in runtime_all
                    ).items()
                )
            ),
            "strict_rejected": sum(
                row["meta"]["strict_rejected"] for row in runtime_all
            ),
            "semantic_pairs": sum(
                row["meta"]["semantic_pair"] for row in runtime_all
            ),
            "synthetic_pairs": sum(
                row["meta"]["synthetic"] for row in runtime_all
            ),
            "rejected_non_parent_generation_rows": rejected_non_parent,
            "excluded_protected_generation_rows": excluded_protected_generation_rows,
        },
        "isolation": {
            "protected_source_trace_ids": len(protected_sources),
            "selected_source_trace_overlap": len(
                selected_source_ids & protected_sources
            ),
            "train_monitor_trace_overlap": len(train_ids & monitor_trace_ids),
            "train_monitor_source_trace_overlap": len(
                train_source_ids & monitor_source_ids
            ),
            "new_eval_query_text_read_by_builder": False,
            "new_eval_gold_trajectory_used_by_builder": False,
        },
        "integrity": {
            "prompt_hash_matches": prompt_matches,
            "candidate_hash_matches": candidate_matches,
            "chosen_rejected_exact_equal": sum(
                row["messages"][-1]["content"] == row["rejected_response"]
                for row in runtime_all
            ),
            "chosen_character_length": summary_stats(
                len(row["messages"][-1]["content"]) for row in runtime_all
            ),
            "rejected_character_length": summary_stats(
                len(row["rejected_response"]) for row in runtime_all
            ),
        },
        "inputs": {
            **{str(path): sha256_file(path) for path in compact_paths},
            str(args.catalog): sha256_file(args.catalog),
            **prediction_hashes,
            **{
                str(path): sha256_file(path)
                for path in args.protected_source_plan
            },
            **{
                f"allowed_adapter:{path}": sha256_file(
                    path / "adapter_model.safetensors"
                )
                for path in args.allowed_adapter
            },
        },
        "outputs": {
            "preferences": {
                "path": str(args.out),
                "sha256": sha256_file(args.out),
            },
            "runtime_train": {
                "path": str(args.runtime_train_out),
                "sha256": sha256_file(args.runtime_train_out),
            },
            "runtime_monitor": {
                "path": str(args.runtime_monitor_out),
                "sha256": sha256_file(args.runtime_monitor_out),
            },
        },
    }
    write_json(args.audit_out, audit)
    print(
        f"DPO-v2 preferences={len(runtime_all)} train={len(runtime_train)} "
        f"monitor={len(runtime_monitor)} tools={len(represented_tools)} "
        f"strict_ok={strict_ok}"
    )
    return 0 if strict_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
