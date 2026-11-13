#!/usr/bin/env python3
"""Finalize a fresh Legacy-generator development panel with strict isolation audits."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import normalize_question  # noqa: E402
from finetune.tool_trace_common import sha256_file  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def query(row: dict[str, Any]) -> str:
    return normalize_question(str(row.get("问题") or row.get("query") or ""))


def source_id(row: dict[str, Any]) -> str:
    construction = row.get("construction") or {}
    meta = row.get("meta") or {}
    return str(
        construction.get("source_trace_id")
        or meta.get("source_trace_id")
        or row.get("id")
        or ""
    )


def cluster_pairs(rows: Iterable[dict[str, Any]]) -> set[tuple[str, str]]:
    result: set[tuple[str, str]] = set()
    for row in rows:
        for tool, value in (row.get("expression_cluster_ids") or {}).items():
            result.add((str(tool), str(value)))
    return result


def phrase_pairs(rows: Iterable[dict[str, Any]]) -> set[tuple[str, str]]:
    result: set[tuple[str, str]] = set()
    for row in rows:
        for tool, value in (row.get("expression_phrases") or {}).items():
            result.add((str(tool), normalize_question(str(value))))
    return result


def stable_key(seed: int, row_id: str) -> str:
    return hashlib.sha256(f"{seed}:{row_id}".encode()).hexdigest()


def split_tools(row: dict[str, Any]) -> list[str]:
    return [tool.strip() for tool in str(row["工具"]).split(",") if tool.strip()]


def context_signature(meta_row: dict[str, Any]) -> str:
    text = str(meta_row.get("query", ""))
    phrases = sorted(
        (meta_row.get("expression_phrases") or {}).values(),
        key=lambda value: len(str(value)),
        reverse=True,
    )
    for phrase in phrases:
        if phrase:
            text = text.replace(str(phrase), "{PHRASE}")
    return normalize_question(text)


def select_balanced_dev(
    rows: list[dict[str, Any]],
    seed: int,
    raw_meta_by_id: dict[str, dict[str, Any]],
    canonical_train_meta: list[dict[str, Any]],
    all_tools: list[str],
) -> list[dict[str, Any]]:
    if len(rows) == 1200:
        return rows
    train_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for meta in canonical_train_meta:
        train_counts[context_signature(meta)].update(str(tool) for tool in meta["tools"])

    def context_hit(row: dict[str, Any]) -> bool:
        meta = raw_meta_by_id[str(row["id"])]
        counts = train_counts.get(context_signature(meta), Counter())
        ranked = [
            tool for tool, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        ]
        ranked.extend(tool for tool in all_tools if tool not in counts)
        return set(split_tools(row)) <= set(ranked[:10])

    def selection_key(row: dict[str, Any]) -> tuple[bool, str]:
        return context_hit(row), stable_key(seed, str(row["id"]))

    singles: dict[str, list[dict[str, Any]]] = defaultdict(list)
    multis: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if int(row["tool_count"]) == 1:
            singles[split_tools(row)[0]].append(row)
        else:
            multis[int(row["tool_count"])].append(row)
    if len(singles) != 100 or any(len(values) < 6 for values in singles.values()):
        raise ValueError("raw development pool cannot provide six singles per tool")
    if any(len(multis[value]) < 120 for value in range(2, 7)):
        raise ValueError("raw development pool cannot provide 120 rows per multi length")
    selected = [
        row
        for tool in sorted(singles)
        for row in sorted(
            singles[tool], key=selection_key
        )[:6]
    ]
    for value in range(2, 7):
        selected.extend(
            sorted(
                multis[value], key=selection_key
            )[:120]
        )
    return sorted(
        selected, key=lambda item: stable_key(seed, "final:" + str(item["id"]))
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dev", type=Path, required=True)
    parser.add_argument("--raw-generation-meta", type=Path, required=True)
    parser.add_argument("--canonical-generation-meta", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--canonical-train", type=Path, required=True)
    parser.add_argument("--frozen-test", type=Path, required=True)
    parser.add_argument("--preferences", type=Path, required=True)
    parser.add_argument("--blocked-dataset", action="append", type=Path, default=[])
    parser.add_argument("--prefix", default="residual_dev2_")
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--generator-seed", type=int, default=20260803)
    parser.add_argument("--context-salt", type=int, default=400009)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit-meta-out", type=Path, required=True)
    parser.add_argument("--audit-out", type=Path, required=True)
    args = parser.parse_args()

    raw_pool = read_jsonl(args.raw_dev)
    raw_meta = read_jsonl(args.raw_generation_meta)
    canonical_meta = read_jsonl(args.canonical_generation_meta)
    catalog = read_jsonl(args.catalog)
    train = read_jsonl(args.canonical_train)
    frozen = read_jsonl(args.frozen_test)
    preferences = read_jsonl(args.preferences)
    blocked = [row for path in args.blocked_dataset for row in read_jsonl(path)]
    train_queries_early = {query(row) for row in train}
    frozen_queries_early = {query(row) for row in frozen}
    blocked_queries_early = {query(row) for row in blocked}
    ineligible_queries = (
        train_queries_early | frozen_queries_early | blocked_queries_early
    )
    eligible_raw_pool = [
        row for row in raw_pool if query(row) not in ineligible_queries
    ]
    train_ids = {str(row["id"]) for row in train}
    canonical_train_meta = [
        row
        for row in canonical_meta
        if row.get("split") == "train" and str(row.get("trace_id")) in train_ids
    ]
    if len(canonical_train_meta) != len(train):
        raise ValueError("canonical metadata does not cover canonical train rows")
    raw_meta_by_id = {
        str(row["trace_id"]): row for row in raw_meta if row.get("split") == "test"
    }
    raw_dev = select_balanced_dev(
        eligible_raw_pool,
        args.seed,
        raw_meta_by_id,
        canonical_train_meta,
        [str(row["tool_id"]) for row in catalog],
    )
    if len(raw_dev) != 1200:
        raise ValueError(f"expected 1200 selected development rows, got {len(raw_dev)}")

    raw_dev_ids = {str(row["id"]) for row in raw_dev}
    new_meta_by_id = {
        str(row["trace_id"]): row
        for row in raw_meta
        if row.get("split") == "test" and str(row["trace_id"]) in raw_dev_ids
    }
    if len(new_meta_by_id) != len(raw_dev):
        raise ValueError("new generation metadata does not cover raw development rows")

    output: list[dict[str, Any]] = []
    prefixed_test_meta: list[dict[str, Any]] = []
    for row in raw_dev:
        old_id = str(row["id"])
        new_id = args.prefix + old_id
        copied = dict(row)
        copied["id"] = new_id
        copied["source"] = "assignment3_context_isolated_legacy_generator_dev"
        copied["style"] = "legacy_compact"
        copied["source_split"] = "residual_dpo_dev"
        copied["generation_split"] = "dev"
        copied["construction"] = {
            "source_trace_id": new_id,
            "type": "fresh_context_isolated_legacy_generator_dev",
            "raw_trace_id": old_id,
        }
        output.append(copied)
        meta = dict(new_meta_by_id[old_id])
        meta["trace_id"] = new_id
        meta["split"] = "test"
        prefixed_test_meta.append(meta)

    write_jsonl(args.out, output)
    write_jsonl(args.audit_meta_out, canonical_train_meta + prefixed_test_meta)

    selected_ids = {str(row["id"]) for row in output}
    selected_queries = {query(row) for row in output}
    selected_sources = {source_id(row) for row in output}
    train_queries = {query(row) for row in train}
    frozen_queries = {query(row) for row in frozen}
    blocked_queries = {query(row) for row in blocked}
    pref_sources = {source_id(row) for row in preferences}
    train_clusters = cluster_pairs(canonical_train_meta)
    dev_clusters = cluster_pairs(prefixed_test_meta)
    train_phrases = phrase_pairs(canonical_train_meta)
    dev_phrases = phrase_pairs(prefixed_test_meta)
    represented_tools = {tool for row in output for tool in split_tools(row)}
    checks = {
        "row_count_1200": len(output) == 1200,
        "unique_id": len(selected_ids) == len(output),
        "unique_query": len(selected_queries) == len(output),
        "source_disjoint_from_preferences": not (selected_sources & pref_sources),
        "query_disjoint_from_canonical_train": not (selected_queries & train_queries),
        "query_disjoint_from_frozen_test": not (selected_queries & frozen_queries),
        "query_disjoint_from_other_dev": not (selected_queries & blocked_queries),
        "expression_cluster_disjoint_from_train": not (train_clusters & dev_clusters),
        "expression_phrase_disjoint_from_train": not (train_phrases & dev_phrases),
        "all_100_tools_represented": len(represented_tools) == 100,
    }
    audit = {
        "schema_version": 1,
        "status": "complete" if all(checks.values()) else "failed",
        "strict_ok": all(checks.values()),
        "role": "primary checkpoint-selection development set; never SFT/DPO training data",
        "frozen_labels_used": False,
        "construction": {
            "style": "legacy_compact",
            "context_salt": args.context_salt,
            "generator_seed": args.generator_seed,
            "single_rows": 600,
            "multi_rows": 600,
            "raw_pool_rows": len(raw_pool),
            "eligible_raw_pool_rows": len(eligible_raw_pool),
            "sampling_seed": args.seed,
            "note": "fresh queries from the repository generator; balanced 1-6 tool depths; context-skeleton shortcuts excluded; final IDs are namespaced",
        },
        "counts": {
            "rows": len(output),
            "by_tool_count": dict(
                sorted(Counter(str(row["tool_count"]) for row in output).items())
            ),
            "represented_tools": len(represented_tools),
        },
        "overlaps": {
            "preference_source": len(selected_sources & pref_sources),
            "canonical_train_query": len(selected_queries & train_queries),
            "frozen_test_query": len(selected_queries & frozen_queries),
            "blocked_dev_query": len(selected_queries & blocked_queries),
            "train_expression_cluster": len(train_clusters & dev_clusters),
            "train_expression_phrase": len(train_phrases & dev_phrases),
        },
        "checks": checks,
        "inputs": {
            str(path): sha256_file(path)
            for path in (
                args.raw_dev,
                args.raw_generation_meta,
                args.canonical_generation_meta,
                args.catalog,
                args.canonical_train,
                args.frozen_test,
                args.preferences,
                *args.blocked_dataset,
            )
        },
        "outputs": {
            str(args.out): sha256_file(args.out),
            str(args.audit_meta_out): sha256_file(args.audit_meta_out),
        },
    }
    args.audit_out.parent.mkdir(parents=True, exist_ok=True)
    args.audit_out.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"legacy generator dev rows={len(output)} tools={len(represented_tools)} "
        f"strict_ok={audit['strict_ok']}"
    )
    return 0 if audit["strict_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
