#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build 100-tool embedding train/test datasets with typed negatives."""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import (
    CATALOG_PATH,
    RAG_DIR,
    TOOL_TRACE_DIR,
    catalog_by_id,
    compact_tool_doc,
    load_catalog,
    load_jsonl,
    split_tools,
    write_json,
    write_jsonl,
)


def unique(items: list[str]) -> list[str]:
    out = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out


def sample_from(pool: list[str], blocked: set[str], n: int, rng: random.Random) -> list[str]:
    values = [item for item in unique(pool) if item not in blocked]
    rng.shuffle(values)
    return values[:n]


def random_fill(all_ids: list[str], blocked: set[str], n: int, rng: random.Random) -> list[str]:
    return sample_from(all_ids, blocked, n, rng)


def same_domain_candidates(tool_id: str, catalog: list[dict[str, Any]], by_id: dict[str, dict[str, Any]]) -> list[str]:
    if tool_id not in by_id:
        return []
    domain = by_id[tool_id]["domain"]
    confusable = [x for x in by_id[tool_id].get("confusable_tools", []) if x in by_id]
    same_domain = [row["tool_id"] for row in catalog if row["domain"] == domain and row["tool_id"] != tool_id]
    return unique(confusable + same_domain)


def cross_domain_candidates(gold: list[str], catalog: list[dict[str, Any]], by_id: dict[str, dict[str, Any]]) -> list[str]:
    gold_domains = {by_id[tool]["domain"] for tool in gold if tool in by_id}
    return [row["tool_id"] for row in catalog if row["domain"] not in gold_domains]


def single_negatives(
    gold: list[str],
    catalog: list[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
    all_ids: list[str],
    rng: random.Random,
    total: int,
) -> list[tuple[str, str]]:
    tool_id = gold[0]
    blocked = set(gold)
    same_n = total // 2
    cross_n = total - same_n
    same = sample_from(same_domain_candidates(tool_id, catalog, by_id), blocked, same_n, rng)
    blocked.update(same)
    cross = sample_from(cross_domain_candidates(gold, catalog, by_id), blocked, cross_n, rng)
    blocked.update(cross)
    if len(same) < same_n:
        same += random_fill(all_ids, blocked, same_n - len(same), rng)
        blocked.update(same)
    if len(cross) < cross_n:
        cross += random_fill(all_ids, blocked, cross_n - len(cross), rng)
    return [(tool, "single_same_domain") for tool in same[:same_n]] + [(tool, "single_cross_domain") for tool in cross[:cross_n]]


def multi_negatives(
    gold: list[str],
    catalog: list[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
    all_ids: list[str],
    rng: random.Random,
    total: int,
) -> list[tuple[str, str]]:
    blocked = set(gold)
    partial_n = total // 2
    add_n = total - partial_n
    partial_pool: list[str] = []
    for tool_id in gold:
        partial_pool.extend(same_domain_candidates(tool_id, catalog, by_id))
    partial = sample_from(partial_pool, blocked, partial_n, rng)
    blocked.update(partial)
    add = sample_from(cross_domain_candidates(gold, catalog, by_id), blocked, add_n, rng)
    blocked.update(add)
    if len(partial) < partial_n:
        partial += random_fill(all_ids, blocked, partial_n - len(partial), rng)
        blocked.update(partial)
    if len(add) < add_n:
        add += random_fill(all_ids, blocked, add_n - len(add), rng)
    return [(tool, "multi_partial_replacement") for tool in partial[:partial_n]] + [(tool, "multi_add_irrelevant") for tool in add[:add_n]]


def make_pair(
    row: dict[str, Any],
    tool_id: str,
    label: int,
    by_id: dict[str, dict[str, Any]],
    split: str,
    negative_type: str = "",
) -> dict[str, Any]:
    tool = by_id[tool_id]
    gold = split_tools(row.get("工具"))
    return {
        "query": row["问题"],
        "tool_id": tool_id,
        "label": label,
        "negative_type": negative_type,
        "document": compact_tool_doc(tool),
        "trace_id": row.get("id", ""),
        "split": split,
        "source": row.get("source", ""),
        "gold_tools": gold,
        "tool_count": len(gold),
        "domain": tool.get("domain", ""),
    }


def build_split(
    rows: list[dict[str, Any]],
    catalog: list[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
    rng: random.Random,
    split: str,
    single_neg_per_query: int,
    multi_neg_per_query: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_ids = [row["tool_id"] for row in catalog]
    pairs: list[dict[str, Any]] = []
    grouped: list[dict[str, Any]] = []
    for row in rows:
        gold = [tool for tool in split_tools(row.get("工具")) if tool in by_id]
        if not gold:
            continue
        negatives = (
            single_negatives(gold, catalog, by_id, all_ids, rng, single_neg_per_query)
            if len(gold) == 1
            else multi_negatives(gold, catalog, by_id, all_ids, rng, multi_neg_per_query)
        )
        for tool_id in gold:
            pairs.append(make_pair(row, tool_id, 1, by_id, split))
        for tool_id, neg_type in negatives:
            pairs.append(make_pair(row, tool_id, 0, by_id, split, neg_type))
        grouped.append(
            {
                "query": row["问题"],
                "trace_id": row.get("id", ""),
                "split": split,
                "tool_count": len(gold),
                "pos_tool_ids": gold,
                "pos": [compact_tool_doc(by_id[tool]) for tool in gold],
                "neg_tool_ids": [tool for tool, _ in negatives],
                "neg": [compact_tool_doc(by_id[tool]) for tool, _ in negatives],
                "negative_types": [neg_type for _, neg_type in negatives],
            }
        )
    return pairs, grouped


def summarize(pairs: list[dict[str, Any]], grouped: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(row["label"] for row in pairs)
    negative_types = Counter(row.get("negative_type", "") for row in pairs if row["label"] == 0)
    tool_counts = Counter(row["tool_count"] for row in grouped)
    return {
        "queries": len(grouped),
        "pairs": len(pairs),
        "positive_pairs": labels.get(1, 0),
        "negative_pairs": labels.get(0, 0),
        "negative_to_positive_ratio": round(labels.get(0, 0) / max(1, labels.get(1, 0)), 4),
        "negative_types": dict(sorted(negative_types.items())),
        "query_tool_count_distribution": dict(sorted(tool_counts.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    parser.add_argument("--train-trace", type=Path, default=TOOL_TRACE_DIR / "train.jsonl")
    parser.add_argument("--test-trace", type=Path, default=TOOL_TRACE_DIR / "test.jsonl")
    parser.add_argument("--train-out", type=Path, default=RAG_DIR / "embedding_train.jsonl")
    parser.add_argument("--test-out", type=Path, default=RAG_DIR / "embedding_test.jsonl")
    parser.add_argument("--grouped-train-out", type=Path, default=RAG_DIR / "embedding_grouped_train.jsonl")
    parser.add_argument("--grouped-test-out", type=Path, default=RAG_DIR / "embedding_grouped_test.jsonl")
    parser.add_argument("--stats-out", type=Path, default=RAG_DIR / "_aux" / "embedding_dataset_stats.json")
    parser.add_argument("--single-neg-per-query", type=int, default=4)
    parser.add_argument("--multi-neg-per-query", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260706)
    args = parser.parse_args()

    catalog = load_catalog(args.catalog)
    by_id = catalog_by_id(catalog)
    train_rng = random.Random(args.seed)
    test_rng = random.Random(args.seed + 1)
    train_pairs, train_grouped = build_split(
        load_jsonl(args.train_trace),
        catalog,
        by_id,
        train_rng,
        "train",
        args.single_neg_per_query,
        args.multi_neg_per_query,
    )
    test_pairs, test_grouped = build_split(
        load_jsonl(args.test_trace),
        catalog,
        by_id,
        test_rng,
        "test",
        args.single_neg_per_query,
        args.multi_neg_per_query,
    )
    write_jsonl(args.train_out, train_pairs)
    write_jsonl(args.test_out, test_pairs)
    write_jsonl(args.grouped_train_out, train_grouped)
    write_jsonl(args.grouped_test_out, test_grouped)
    stats = {
        "policy": {
            "catalog_tools": len(catalog),
            "seed": args.seed,
            "single_neg_per_query": args.single_neg_per_query,
            "single_negative_mix": "same_domain:cross_domain = 1:1",
            "multi_neg_per_query": args.multi_neg_per_query,
            "multi_negative_mix": "partial_replacement:add_irrelevant = 1:1",
            "scale_note": "100 工具版保留 24k/6k 轨迹切分中的全部正样本工具对，并按 query 级别抽取类型化负样本。",
        },
        "train": summarize(train_pairs, train_grouped),
        "test": summarize(test_pairs, test_grouped),
        "outputs": {
            "train_pairs": str(args.train_out),
            "test_pairs": str(args.test_out),
            "train_grouped": str(args.grouped_train_out),
            "test_grouped": str(args.grouped_test_out),
        },
    }
    write_json(args.stats_out, stats)
    print(
        "embedding dataset "
        f"train_pairs={len(train_pairs)} test_pairs={len(test_pairs)} "
        f"grouped_train={len(train_grouped)} grouped_test={len(test_grouped)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
