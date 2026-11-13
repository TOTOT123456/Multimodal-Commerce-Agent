#!/usr/bin/env python3
"""Prepare source-isolated train-side data for the current API-9K DPO redo."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import (  # noqa: E402
    PROJECT_ROOT,
    TOOL_TRACE_DIR,
    load_jsonl,
    normalize_question,
    split_tools,
    write_json,
    write_jsonl,
)
from finetune.tool_trace_common import (  # noqa: E402
    sha256_file,
    sha256_json,
    stable_seed,
)


EXP_ROOT = (
    PROJECT_ROOT
    / "data/tool_call_trace/_aux/assignment3/assignment3_current_api9k_dpo_20260802"
)
API_ROOT = (
    PROJECT_ROOT
    / "data/tool_call_trace/_aux/assignment3/assignment3_100tool_api"
)
SFT_9K = (
    PROJECT_ROOT
    / "finetune/outputs/tooltrace-data-cache/assignment3_sft_ablation"
    / "sft_train_9000.runtime.jsonl"
)
LEGACY_DEV_COUNTS = {1: 600, 2: 96, 3: 136, 4: 118, 5: 131, 6: 119}
API_DEV_COUNTS = {1: 34, 2: 34, 3: 33, 4: 33, 5: 33, 6: 33}
API_MINING_CONSTRUCTION_COUNTS = {
    "single_intent": 1333,
    "multi_candidate_order_shuffled": 1778,
    "multi_same_type_distractors": 889,
}


def source_id(row: dict[str, Any]) -> str:
    construction = row.get("construction")
    if isinstance(construction, dict) and construction.get("source_trace_id"):
        return str(construction["source_trace_id"])
    return str(row["id"])


def tools(row: dict[str, Any]) -> list[str]:
    if isinstance(row.get("gold_tools"), list):
        return [str(value) for value in row["gold_tools"]]
    return split_tools(str(row["工具"]))


def copy_row(row: dict[str, Any], style: str, split: str) -> dict[str, Any]:
    result = dict(row)
    result["style"] = style
    result["source_split"] = split
    construction = dict(result.get("construction") or {})
    construction.setdefault("source_trace_id", source_id(row))
    construction.setdefault("type", "legacy_train_source")
    result["construction"] = construction
    return result


def choose_balanced(
    rows: list[dict[str, Any]],
    counts: dict[Any, int],
    bucket,
    blocked_sources: set[str],
    seed: int,
    label: str,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_sources = set(blocked_sources)
    exposure: Counter[str] = Counter()
    for bucket_value, target in counts.items():
        by_source: dict[str, dict[str, Any]] = {}
        for row in rows:
            row_source = source_id(row)
            if bucket(row) != bucket_value or row_source in used_sources:
                continue
            previous = by_source.get(row_source)
            if previous is None or stable_seed(seed, label, row["id"]) < stable_seed(
                seed, label, previous["id"]
            ):
                by_source[row_source] = row
        pool = sorted(
            by_source.values(),
            key=lambda row: stable_seed(seed, label, bucket_value, row["id"]),
        )
        for index in range(target):
            if index >= len(pool):
                raise ValueError(f"insufficient {label} rows for {bucket_value}: {index}/{target}")
            window_end = min(len(pool), index + 256)
            row = min(
                pool[index:window_end],
                key=lambda item: (
                    max(exposure[tool] + 1 for tool in tools(item)),
                    sum((exposure[tool] + 1) ** 2 for tool in tools(item)),
                    stable_seed(seed, label, bucket_value, index, item["id"]),
                ),
            )
            selected_index = pool.index(row, index, window_end)
            pool[index], pool[selected_index] = pool[selected_index], pool[index]
            selected.append(row)
            used_sources.add(source_id(row))
            exposure.update(tools(row))
    return selected


def proportional_counts(rows: list[dict[str, Any]], total: int) -> dict[int, int]:
    distribution = Counter(int(row["tool_count"]) for row in rows)
    raw = {key: total * value / len(rows) for key, value in distribution.items()}
    result = {key: int(value) for key, value in raw.items()}
    remainder = total - sum(result.values())
    order = sorted(raw, key=lambda key: (-(raw[key] - result[key]), key))
    for key in order[:remainder]:
        result[key] += 1
    return dict(sorted(result.items()))


def normalized_queries(rows: list[dict[str, Any]]) -> set[str]:
    return {
        normalize_question(str(row.get("问题", row.get("query", ""))))
        for row in rows
    }


def plan(args: argparse.Namespace) -> int:
    api_rows = load_jsonl(args.api_train)
    api_by_id = {str(row["id"]): row for row in api_rows}
    legacy_rows = load_jsonl(args.legacy_train)
    sft_rows = load_jsonl(args.sft_9k)
    sft_ids = [str(row["meta"]["trace_id"]) for row in sft_rows]
    if len(api_rows) != 54_000 or len(legacy_rows) != 24_000 or len(sft_ids) != 9_000:
        raise ValueError("unexpected established train row count")
    if len(api_by_id) != len(api_rows) or len(set(sft_ids)) != len(sft_ids):
        raise ValueError("duplicate API or SFT trace id")
    if not set(sft_ids) <= set(api_by_id):
        raise ValueError("SFT-9K ids are not a subset of API-native 54K")

    frozen_rows = [
        row
        for path in args.blocked_dataset
        for row in load_jsonl(path)
    ]
    frozen_ids = {str(row["id"]) for row in frozen_rows}
    frozen_queries = normalized_queries(frozen_rows)
    all_train_ids = {str(row["id"]) for row in api_rows + legacy_rows}
    if all_train_ids & frozen_ids:
        raise ValueError("train/frozen trace id overlap")
    if normalized_queries(api_rows + legacy_rows) & frozen_queries:
        raise ValueError("train/frozen normalized query overlap")

    sft_source_ids = {source_id(api_by_id[trace_id]) for trace_id in sft_ids}
    dev_legacy = choose_balanced(
        legacy_rows,
        LEGACY_DEV_COUNTS,
        lambda row: int(row["tool_count"]),
        sft_source_ids,
        args.seed,
        "dev-legacy",
    )
    dev_sources = sft_source_ids | {source_id(row) for row in dev_legacy}
    dev_api = choose_balanced(
        api_rows,
        API_DEV_COUNTS,
        lambda row: int(row["tool_count"]),
        dev_sources,
        args.seed,
        "dev-api",
    )
    dev_rows = [
        copy_row(row, "legacy_compact", "legacy_train") for row in dev_legacy
    ] + [
        copy_row(row, "api_native", "assignment3_api_native_train")
        for row in dev_api
    ]
    dev_source_ids = {source_id(row) for row in dev_rows}

    current_sft_rows = [
        copy_row(api_by_id[trace_id], "api_native", "assignment3_api_native_train")
        for trace_id in sft_ids
    ]
    expansion_blocked = sft_source_ids | dev_source_ids
    api_expansion = choose_balanced(
        api_rows,
        API_MINING_CONSTRUCTION_COUNTS,
        lambda row: str(row["construction"]["type"]),
        expansion_blocked,
        args.seed,
        "mining-api-expansion",
    )
    mining_sources = expansion_blocked | {source_id(row) for row in api_expansion}
    legacy_counts = proportional_counts(legacy_rows, 4000)
    legacy_mining = choose_balanced(
        legacy_rows,
        legacy_counts,
        lambda row: int(row["tool_count"]),
        mining_sources,
        args.seed,
        "mining-legacy",
    )
    mining_rows = current_sft_rows + [
        copy_row(row, "api_native", "assignment3_api_native_train")
        for row in api_expansion
    ] + [
        copy_row(row, "legacy_compact", "legacy_train")
        for row in legacy_mining
    ]

    if {str(row["id"]) for row in dev_rows} & {str(row["id"]) for row in mining_rows}:
        raise ValueError("dev/mining trace overlap")
    mining_source_ids = {source_id(row) for row in mining_rows}
    if dev_source_ids & mining_source_ids:
        raise ValueError("dev/mining source trace overlap")
    if normalized_queries(dev_rows) & normalized_queries(mining_rows):
        raise ValueError("dev/mining normalized query overlap")

    retrieval_rows = dev_rows + mining_rows
    if len({str(row["id"]) for row in retrieval_rows}) != len(retrieval_rows):
        raise ValueError("duplicate retrieval id")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "plan/dev_gold.jsonl", dev_rows)
    write_jsonl(args.output_dir / "plan/mining_source.jsonl", mining_rows)
    write_jsonl(args.output_dir / "plan/retrieval_dataset.jsonl", retrieval_rows)
    audit = {
        "schema_version": 1,
        "status": "complete",
        "strict_ok": True,
        "seed": args.seed,
        "training_use_allowed": {
            "dev": False,
            "mining": True,
            "frozen": False,
        },
        "counts": {
            "dev": len(dev_rows),
            "dev_by_style": dict(Counter(row["style"] for row in dev_rows)),
            "dev_by_tool_count": dict(Counter(str(row["tool_count"]) for row in dev_rows)),
            "mining": len(mining_rows),
            "mining_current_sft_queries": len(current_sft_rows),
            "mining_api_expansion": len(api_expansion),
            "mining_legacy": len(legacy_mining),
            "mining_legacy_tool_count_targets": legacy_counts,
            "retrieval_total": len(retrieval_rows),
        },
        "isolation": {
            "sft_unique_source_trace_ids": len(sft_source_ids),
            "dev_unique_source_trace_ids": len(dev_source_ids),
            "mining_unique_source_trace_ids": len(mining_source_ids),
            "dev_sft_source_overlap": len(dev_source_ids & sft_source_ids),
            "dev_mining_source_overlap": len(dev_source_ids & mining_source_ids),
            "dev_mining_trace_overlap": 0,
            "dev_mining_query_overlap": 0,
            "train_frozen_trace_overlap": 0,
            "train_frozen_query_overlap": 0,
        },
        "inputs": {
            str(args.api_train): sha256_file(args.api_train),
            str(args.legacy_train): sha256_file(args.legacy_train),
            str(args.sft_9k): sha256_file(args.sft_9k),
            **{str(path): sha256_file(path) for path in args.blocked_dataset},
        },
        "outputs": {
            str(path): sha256_file(path)
            for path in (
                args.output_dir / "plan/dev_gold.jsonl",
                args.output_dir / "plan/mining_source.jsonl",
                args.output_dir / "plan/retrieval_dataset.jsonl",
            )
        },
    }
    write_json(args.output_dir / "plan/data_plan_audit.json", audit)
    print(f"DPO plan dev={len(dev_rows)} mining={len(mining_rows)} strict_ok=True")
    return 0


def merge_reranker(args: argparse.Namespace) -> int:
    source_rows = load_jsonl(args.first_stage_preds)
    shards = [load_jsonl(path) for path in args.reranker_shard]
    merged = {str(row["id"]): row for rows in shards for row in rows}
    if sum(map(len, shards)) != len(merged):
        raise ValueError("duplicate reranker shard ids")
    source_ids = [str(row["id"]) for row in source_rows]
    if set(source_ids) != set(merged):
        raise ValueError("reranker shards do not cover first-stage rows")
    aligned = [merged[trace_id] for trace_id in source_ids]
    for source, reranked in zip(source_rows, aligned):
        if source["query"] != reranked["query"]:
            raise ValueError(f"reranker query mismatch: {source['id']}")
    output = args.merged_reranker_out or (
        args.output_dir / "retrieval/reranker_merged/preds.jsonl"
    )
    write_jsonl(output, aligned)
    manifest = {
        "status": "complete",
        "strict_ok": True,
        "rows": len(aligned),
        "source": str(args.first_stage_preds),
        "source_sha256": sha256_file(args.first_stage_preds),
        "shards": {str(path): sha256_file(path) for path in args.reranker_shard},
        "output": str(output),
        "output_sha256": sha256_file(output),
    }
    write_json(output.parent / "manifest.json", manifest)
    print(f"merged reranker shards={len(shards)} rows={len(aligned)}")
    return 0


def plan_expansion(args: argparse.Namespace) -> int:
    api_rows = load_jsonl(args.api_train)
    dev_rows = load_jsonl(args.output_dir / "plan/dev_gold.jsonl")
    mining_rows = load_jsonl(args.output_dir / "plan/mining_source.jsonl")
    blocked_sources = {source_id(row) for row in dev_rows + mining_rows}
    by_source: dict[str, dict[str, Any]] = {}
    for row in api_rows:
        row_source = source_id(row)
        if row_source in blocked_sources:
            continue
        previous = by_source.get(row_source)
        if previous is None or stable_seed(args.seed, "api-expansion", row["id"]) < stable_seed(
            args.seed, "api-expansion", previous["id"]
        ):
            by_source[row_source] = row
    expansion = [
        copy_row(row, "api_native", "assignment3_api_native_train")
        for row in sorted(
            by_source.values(),
            key=lambda row: stable_seed(args.seed, "api-expansion-order", row["id"]),
        )
    ]
    if not expansion:
        raise ValueError("empty adaptive API expansion")
    expansion_sources = {source_id(row) for row in expansion}
    if expansion_sources & blocked_sources:
        raise ValueError("adaptive expansion source overlap")
    path = args.output_dir / "expansion/plan/retrieval_dataset.jsonl"
    write_jsonl(path, expansion)
    audit = {
        "schema_version": 1,
        "status": "complete",
        "strict_ok": True,
        "reason": "initial real-error pool did not meet the preregistered 400-pair API-style floor",
        "selection": "one deterministic unused API-native query per remaining source_trace_id",
        "rows": len(expansion),
        "unique_source_trace_ids": len(expansion_sources),
        "overlap_with_dev_or_initial_mining": 0,
        "tool_count": dict(Counter(str(row["tool_count"]) for row in expansion)),
        "input": {str(args.api_train): sha256_file(args.api_train)},
        "output": {str(path): sha256_file(path)},
    }
    write_json(args.output_dir / "expansion/plan/audit.json", audit)
    print(f"adaptive API expansion rows={len(expansion)} sources={len(expansion_sources)}")
    return 0


def plan_legacy_expansion(args: argparse.Namespace) -> int:
    legacy_rows = load_jsonl(args.legacy_train)
    dev_rows = load_jsonl(args.output_dir / "plan/dev_gold.jsonl")
    mining_rows = load_jsonl(args.output_dir / "plan/mining_source.jsonl")
    api_expansion_rows = load_jsonl(
        args.output_dir / "expansion/plan/retrieval_dataset.jsonl"
    )
    blocked_sources = {
        source_id(row) for row in dev_rows + mining_rows + api_expansion_rows
    }
    by_source: dict[str, dict[str, Any]] = {}
    for row in legacy_rows:
        row_source = source_id(row)
        if row_source in blocked_sources:
            continue
        previous = by_source.get(row_source)
        if previous is None or stable_seed(
            args.seed, "legacy-expansion", row["id"]
        ) < stable_seed(args.seed, "legacy-expansion", previous["id"]):
            by_source[row_source] = row
    selected = sorted(
        by_source.values(),
        key=lambda row: stable_seed(
            args.seed, "legacy-expansion-order", row["id"]
        ),
    )
    expansion = [
        copy_row(row, "legacy_compact", "legacy_train") for row in selected
    ]
    expansion_sources = {source_id(row) for row in expansion}
    if len(expansion_sources) != len(expansion) or not expansion:
        raise ValueError("legacy expansion must contain one row per source")
    if expansion_sources & blocked_sources:
        raise ValueError("legacy expansion source overlap")
    path = args.output_dir / "expansion_legacy/plan/retrieval_dataset.jsonl"
    write_jsonl(path, expansion)
    audit = {
        "schema_version": 1,
        "status": "complete",
        "strict_ok": True,
        "reason": "API-style floor was addressed, but the real-error pool remained below the PDF minimum of 2000 pairs",
        "selection": "all remaining deterministic unused Legacy sources, one query per source_trace_id",
        "rows": len(expansion),
        "unique_source_trace_ids": len(expansion_sources),
        "overlap_with_dev_initial_mining_or_api_expansion": 0,
        "tool_count": dict(Counter(str(row["tool_count"]) for row in expansion)),
        "input": {str(args.legacy_train): sha256_file(args.legacy_train)},
        "output": {str(path): sha256_file(path)},
    }
    write_json(args.output_dir / "expansion_legacy/plan/audit.json", audit)
    print(
        f"adaptive Legacy expansion rows={len(expansion)} "
        f"sources={len(expansion_sources)}"
    )
    return 0


def finalize_expansion(args: argparse.Namespace) -> int:
    source_rows = load_jsonl(args.output_dir / "expansion/plan/retrieval_dataset.jsonl")
    ranking_rows = load_jsonl(args.canonical_preds)
    rankings = {str(row["id"]): row for row in ranking_rows}
    if set(rankings) != {str(row["id"]) for row in source_rows}:
        raise ValueError("adaptive expansion rankings do not match source rows")
    compact = []
    inputs = []
    misses = 0
    for row in source_rows:
        candidates = list(rankings[str(row["id"])]["pred_tools"][:10])
        if len(candidates) != 10 or len(set(candidates)) != 10:
            raise ValueError(f"invalid expansion Top-10: {row['id']}")
        if not set(tools(row)) <= set(candidates):
            misses += 1
            continue
        compact.append(
            {
                "trace_id": str(row["id"]),
                "query": str(row["问题"]),
                "gold_tools": tools(row),
                "source_split": "assignment3_api_native_train",
                "candidate_tools": candidates,
                "candidate_k": 10,
                "tool_count": int(row["tool_count"]),
                "domain": row.get("domain", ""),
                "combo_seen_bucket": row.get("combo_seen_bucket", ""),
                "construction": row["construction"],
                "record_hash": sha256_json(row),
            }
        )
        inputs.append(label_free_row(row, candidates))
    data_dir = args.output_dir / "expansion/data"
    write_jsonl(data_dir / "mining_api_compact.jsonl", compact)
    write_jsonl(data_dir / "mining_api_rag_label_free.jsonl", inputs)
    supplemental_sample_inputs = load_jsonl(
        args.output_dir / "data/mining_legacy_rag_label_free.jsonl"
    ) + inputs
    write_jsonl(
        data_dir / "supplemental_sample_rag_label_free.jsonl",
        supplemental_sample_inputs,
    )
    initial = load_jsonl(args.output_dir / "data/mining_compact.jsonl")
    combined = initial + compact
    if len({row["trace_id"] for row in combined}) != len(combined):
        raise ValueError("duplicate compact id after adaptive expansion")
    write_jsonl(data_dir / "mining_compact_expanded.jsonl", combined)
    audit = {
        "schema_version": 1,
        "status": "complete",
        "strict_ok": True,
        "planned": len(source_rows),
        "eligible": len(compact),
        "retrieval_miss_excluded": misses,
        "combined_compact_rows": len(combined),
        "canonical_predictions": {
            "path": str(args.canonical_preds),
            "sha256": sha256_file(args.canonical_preds),
        },
        "outputs": {
            str(path): sha256_file(path)
            for path in (
                data_dir / "mining_api_compact.jsonl",
                data_dir / "mining_api_rag_label_free.jsonl",
                data_dir / "supplemental_sample_rag_label_free.jsonl",
                data_dir / "mining_compact_expanded.jsonl",
            )
        },
    }
    write_json(data_dir / "audit.json", audit)
    print(f"adaptive API eligible={len(compact)}/{len(source_rows)} combined={len(combined)}")
    return 0


def finalize_legacy_expansion(args: argparse.Namespace) -> int:
    source_rows = load_jsonl(
        args.output_dir / "expansion_legacy/plan/retrieval_dataset.jsonl"
    )
    ranking_rows = load_jsonl(args.canonical_preds)
    rankings = {str(row["id"]): row for row in ranking_rows}
    if set(rankings) != {str(row["id"]) for row in source_rows}:
        raise ValueError("adaptive Legacy expansion rankings do not match source rows")
    compact = []
    inputs = []
    misses = 0
    for row in source_rows:
        candidates = list(rankings[str(row["id"])]["pred_tools"][:10])
        if len(candidates) != 10 or len(set(candidates)) != 10:
            raise ValueError(f"invalid Legacy expansion Top-10: {row['id']}")
        if not set(tools(row)) <= set(candidates):
            misses += 1
            continue
        compact.append(
            {
                "trace_id": str(row["id"]),
                "query": str(row["问题"]),
                "gold_tools": tools(row),
                "source_split": "legacy_train",
                "candidate_tools": candidates,
                "candidate_k": 10,
                "tool_count": int(row["tool_count"]),
                "domain": row.get("domain", ""),
                "combo_seen_bucket": row.get("combo_seen_bucket", ""),
                "construction": row["construction"],
                "record_hash": sha256_json(row),
            }
        )
        inputs.append(label_free_row(row, candidates))
    data_dir = args.output_dir / "expansion_legacy/data"
    write_jsonl(data_dir / "mining_legacy_compact.jsonl", compact)
    write_jsonl(data_dir / "mining_legacy_rag_label_free.jsonl", inputs)
    all_mining_inputs = (
        load_jsonl(args.output_dir / "data/mining_rag_label_free.jsonl")
        + load_jsonl(args.output_dir / "expansion/data/mining_api_rag_label_free.jsonl")
        + inputs
    )
    write_jsonl(data_dir / "mining_all_rag_label_free.jsonl", all_mining_inputs)
    previous = load_jsonl(
        args.output_dir / "expansion/data/mining_compact_expanded.jsonl"
    )
    combined = previous + compact
    if len({row["trace_id"] for row in combined}) != len(combined):
        raise ValueError("duplicate compact id after adaptive Legacy expansion")
    write_jsonl(data_dir / "mining_compact_expanded.jsonl", combined)
    audit = {
        "schema_version": 1,
        "status": "complete",
        "strict_ok": True,
        "planned": len(source_rows),
        "eligible": len(compact),
        "retrieval_miss_excluded": misses,
        "combined_compact_rows": len(combined),
        "canonical_predictions": {
            "path": str(args.canonical_preds),
            "sha256": sha256_file(args.canonical_preds),
        },
        "outputs": {
            str(path): sha256_file(path)
            for path in (
                data_dir / "mining_legacy_compact.jsonl",
                data_dir / "mining_legacy_rag_label_free.jsonl",
                data_dir / "mining_all_rag_label_free.jsonl",
                data_dir / "mining_compact_expanded.jsonl",
            )
        },
    }
    write_json(data_dir / "audit.json", audit)
    print(
        f"adaptive Legacy eligible={len(compact)}/{len(source_rows)} "
        f"combined={len(combined)}"
    )
    return 0


def label_free_row(row: dict[str, Any], candidates: list[str] | None) -> dict[str, Any]:
    payload: dict[str, Any] = {"id": str(row["id"]), "query": str(row["问题"])}
    if candidates is not None:
        payload.update(
            {
                "candidate_tools": candidates,
                "candidate_k": len(candidates),
                "ranking_hash": sha256_json(candidates),
            }
        )
    payload["input_hash"] = sha256_json(payload)
    return payload


def finalize(args: argparse.Namespace) -> int:
    dev_rows = load_jsonl(args.output_dir / "plan/dev_gold.jsonl")
    mining_rows = load_jsonl(args.output_dir / "plan/mining_source.jsonl")
    ranking_rows = load_jsonl(args.canonical_preds)
    rankings = {str(row["id"]): row for row in ranking_rows}
    expected = {str(row["id"]) for row in dev_rows + mining_rows}
    if set(rankings) != expected or len(rankings) != len(ranking_rows):
        raise ValueError("canonical ranking ids do not match the DPO plan")

    def top10(row: dict[str, Any]) -> list[str]:
        ranking = list(rankings[str(row["id"])]["pred_tools"][:10])
        if len(ranking) != 10 or len(set(ranking)) != 10:
            raise ValueError(f"invalid canonical Top-10: {row['id']}")
        return ranking

    dev_rag = [label_free_row(row, top10(row)) for row in dev_rows]
    dev_global = [label_free_row(row, None) for row in dev_rows]
    eligible = [row for row in mining_rows if set(tools(row)) <= set(top10(row))]
    eligible_ids = {str(row["id"]) for row in eligible}
    retrieval_misses = [
        row for row in mining_rows if str(row["id"]) not in eligible_ids
    ]
    compact = []
    mining_inputs = []
    for row in eligible:
        candidates = top10(row)
        compact.append(
            {
                "trace_id": str(row["id"]),
                "query": str(row["问题"]),
                "gold_tools": tools(row),
                "source_split": str(row["source_split"]),
                "candidate_tools": candidates,
                "candidate_k": 10,
                "tool_count": int(row["tool_count"]),
                "domain": row.get("domain", ""),
                "combo_seen_bucket": row.get("combo_seen_bucket", ""),
                "construction": row["construction"],
                "record_hash": sha256_json(row),
            }
        )
        mining_inputs.append(label_free_row(row, candidates))

    final_dir = args.output_dir / "data"
    write_jsonl(final_dir / "dev_rag_label_free.jsonl", dev_rag)
    write_jsonl(final_dir / "dev_global_label_free.jsonl", dev_global)
    write_jsonl(final_dir / "mining_compact.jsonl", compact)
    write_jsonl(final_dir / "mining_rag_label_free.jsonl", mining_inputs)
    write_jsonl(
        final_dir / "mining_api_rag_label_free.jsonl",
        [
            input_row
            for source_row, input_row in zip(compact, mining_inputs)
            if source_row["source_split"] == "assignment3_api_native_train"
        ],
    )
    write_jsonl(
        final_dir / "mining_legacy_rag_label_free.jsonl",
        [
            input_row
            for source_row, input_row in zip(compact, mining_inputs)
            if source_row["source_split"] == "legacy_train"
        ],
    )
    dev_hit = sum(set(tools(row)) <= set(top10(row)) for row in dev_rows)
    audit = {
        "schema_version": 1,
        "status": "complete",
        "strict_ok": True,
        "fixed_candidate_interface": "current canonical R20/P10 retrieval order",
        "counts": {
            "dev": len(dev_rows),
            "dev_allhit_at_10": dev_hit / len(dev_rows),
            "mining_planned": len(mining_rows),
            "mining_eligible": len(eligible),
            "mining_retrieval_miss_excluded": len(retrieval_misses),
            "eligible_by_style": dict(Counter(row["source_split"] for row in compact)),
            "eligible_by_tool_count": dict(Counter(str(row["tool_count"]) for row in compact)),
        },
        "rules": {
            "retrieval_misses_can_enter_dpo": False,
            "dev_can_enter_dpo": False,
            "label_free_field_whitelist_rag": [
                "id",
                "query",
                "candidate_tools",
                "candidate_k",
                "ranking_hash",
                "input_hash",
            ],
            "label_free_contains_gold": False,
        },
        "inputs": {
            str(args.canonical_preds): sha256_file(args.canonical_preds),
            str(args.output_dir / "plan/dev_gold.jsonl"): sha256_file(
                args.output_dir / "plan/dev_gold.jsonl"
            ),
            str(args.output_dir / "plan/mining_source.jsonl"): sha256_file(
                args.output_dir / "plan/mining_source.jsonl"
            ),
        },
        "outputs": {
            str(path): sha256_file(path)
            for path in (
                final_dir / "dev_rag_label_free.jsonl",
                final_dir / "dev_global_label_free.jsonl",
                final_dir / "mining_compact.jsonl",
                final_dir / "mining_rag_label_free.jsonl",
                final_dir / "mining_api_rag_label_free.jsonl",
                final_dir / "mining_legacy_rag_label_free.jsonl",
            )
        },
    }
    write_json(final_dir / "data_finalize_audit.json", audit)
    print(
        f"DPO data dev={len(dev_rows)} dev_AllHit@10={dev_hit / len(dev_rows):.6f} "
        f"mining_eligible={len(eligible)}/{len(mining_rows)}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=[
            "plan",
            "merge-reranker",
            "finalize",
            "plan-expansion",
            "finalize-expansion",
            "plan-legacy-expansion",
            "finalize-legacy-expansion",
        ],
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, default=EXP_ROOT)
    parser.add_argument("--api-train", type=Path, default=API_ROOT / "train_api/train.jsonl")
    parser.add_argument("--legacy-train", type=Path, default=TOOL_TRACE_DIR / "train.jsonl")
    parser.add_argument("--sft-9k", type=Path, default=SFT_9K)
    parser.add_argument(
        "--blocked-dataset",
        action="append",
        type=Path,
        default=[TOOL_TRACE_DIR / "test.jsonl", API_ROOT / "test_api/test.jsonl"],
    )
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--first-stage-preds", type=Path)
    parser.add_argument("--reranker-shard", action="append", type=Path, default=[])
    parser.add_argument("--merged-reranker-out", type=Path)
    parser.add_argument("--canonical-preds", type=Path)
    args = parser.parse_args()
    if args.stage == "plan":
        return plan(args)
    if args.stage == "plan-expansion":
        return plan_expansion(args)
    if args.stage == "plan-legacy-expansion":
        return plan_legacy_expansion(args)
    if args.stage == "merge-reranker":
        if not args.first_stage_preds or not args.reranker_shard:
            parser.error("merge-reranker requires --first-stage-preds and --reranker-shard")
        return merge_reranker(args)
    if args.stage == "finalize-expansion":
        if not args.canonical_preds:
            parser.error("finalize-expansion requires --canonical-preds")
        return finalize_expansion(args)
    if args.stage == "finalize-legacy-expansion":
        if not args.canonical_preds:
            parser.error("finalize-legacy-expansion requires --canonical-preds")
        return finalize_legacy_expansion(args)
    if not args.canonical_preds:
        parser.error("finalize requires --canonical-preds")
    return finalize(args)


if __name__ == "__main__":
    raise SystemExit(main())
