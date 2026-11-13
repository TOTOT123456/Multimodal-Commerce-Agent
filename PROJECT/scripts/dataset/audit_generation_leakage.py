#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit train/test leakage for generated 100-tool trace datasets."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import (
    CATALOG_PATH,
    RAG_DIR,
    TOOL_CORPUS_PATH,
    TOOL_TRACE_DIR,
    load_catalog,
    load_jsonl,
    normalize_question,
    split_tools,
    write_json,
)
from rag.run_retrieval_baselines import BM25, build_tokenizer, evaluate_rankings, rank_from_scores


DEFAULT_OUT = TOOL_TRACE_DIR / "_aux" / "generation_leakage_audit.json"


def pct(num: int, den: int) -> float:
    return round(num / max(1, den), 6)


def catalog_terms(catalog: list[dict[str, Any]]) -> dict[str, list[tuple[str, str, str]]]:
    out: dict[str, list[tuple[str, str, str]]] = {}
    for tool in catalog:
        rows = []
        for kind, field in [("tool_id", "tool_id"), ("name_zh", "name_zh"), ("keyword_label", "keyword_label")]:
            raw = str(tool.get(field, ""))
            normalized = normalize_question(raw)
            if normalized:
                rows.append((kind, raw, normalized))
        for phrase in tool.get("positive_phrases", []):
            normalized = normalize_question(str(phrase))
            if normalized:
                rows.append(("positive_phrase", str(phrase), normalized))
        out[tool["tool_id"]] = rows
    return out


def trigger_overlap(rows: list[dict[str, Any]], terms: dict[str, list[tuple[str, str, str]]]) -> dict[str, Any]:
    totals = Counter()
    examples = []
    by_kind = Counter()
    for row in rows:
        qn = normalize_question(row.get("问题", ""))
        gold = split_tools(row.get("工具"))
        hits = []
        for tool_id in gold:
            for kind, raw, term in terms.get(tool_id, []):
                if term and term in qn:
                    hits.append({"tool_id": tool_id, "kind": kind, "term": raw})
                    by_kind[kind] += 1
        totals["rows"] += 1
        totals["rows_with_any_gold_trigger"] += int(bool(hits))
        totals["gold_instances"] += len(gold)
        hit_tools = {hit["tool_id"] for hit in hits}
        totals["gold_instances_with_trigger"] += len(hit_tools)
        if hits and len(examples) < 20:
            examples.append({"id": row.get("id"), "query": row.get("问题"), "hits": hits[:10]})
    return {
        "rows": totals["rows"],
        "rows_with_any_gold_trigger": totals["rows_with_any_gold_trigger"],
        "rows_with_any_gold_trigger_rate": pct(totals["rows_with_any_gold_trigger"], totals["rows"]),
        "gold_instances": totals["gold_instances"],
        "gold_instances_with_trigger": totals["gold_instances_with_trigger"],
        "gold_instances_with_trigger_rate": pct(totals["gold_instances_with_trigger"], totals["gold_instances"]),
        "by_kind_hits": dict(sorted(by_kind.items())),
        "examples": examples,
    }


def meta_by_trace(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("trace_id")): row for row in load_jsonl(path)}


def expression_cluster_overlap(train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]], meta: dict[str, dict[str, Any]]) -> dict[str, Any]:
    train_clusters = set()
    test_clusters = set()
    train_tool_phrases = set()
    test_tool_phrases = set()
    test_seen_instances = 0
    test_instances = 0

    for row in train_rows:
        m = meta.get(str(row.get("id")), {})
        for tool_id, cluster_id in (m.get("expression_cluster_ids") or {}).items():
            train_clusters.add((tool_id, cluster_id))
        for tool_id, phrase in (m.get("expression_phrases") or {}).items():
            train_tool_phrases.add((tool_id, normalize_question(str(phrase))))

    for row in test_rows:
        m = meta.get(str(row.get("id")), {})
        phrases = m.get("expression_phrases") or {}
        for tool_id, cluster_id in (m.get("expression_cluster_ids") or {}).items():
            test_clusters.add((tool_id, cluster_id))
        for tool_id, phrase in phrases.items():
            test_instances += 1
            norm = normalize_question(str(phrase))
            test_tool_phrases.add((tool_id, norm))
            test_seen_instances += int((tool_id, norm) in train_tool_phrases)

    return {
        "train_unique_tool_clusters": len(train_clusters),
        "test_unique_tool_clusters": len(test_clusters),
        "test_tool_clusters_seen_in_train": len(test_clusters & train_clusters),
        "test_tool_clusters_seen_in_train_rate": pct(len(test_clusters & train_clusters), len(test_clusters)),
        "train_unique_tool_phrases": len(train_tool_phrases),
        "test_unique_tool_phrases": len(test_tool_phrases),
        "test_tool_phrases_seen_in_train": len(test_tool_phrases & train_tool_phrases),
        "test_tool_phrase_seen_in_train_rate": pct(len(test_tool_phrases & train_tool_phrases), len(test_tool_phrases)),
        "test_gold_instances": test_instances,
        "test_gold_instances_phrase_seen_in_train": test_seen_instances,
        "test_gold_instances_phrase_seen_in_train_rate": pct(test_seen_instances, test_instances),
    }


def train_phrase_rankings(catalog: list[dict[str, Any]], test_rows: list[dict[str, Any]], train_meta: list[dict[str, Any]]) -> list[list[str]]:
    all_tools = [tool["tool_id"] for tool in catalog]
    phrases: dict[str, set[str]] = defaultdict(set)
    for row in train_meta:
        if row.get("split") != "train":
            continue
        for tool_id, phrase in (row.get("expression_phrases") or {}).items():
            normalized = normalize_question(str(phrase))
            if normalized:
                phrases[tool_id].add(normalized)

    rankings = []
    for row in test_rows:
        qn = normalize_question(row.get("问题", ""))
        scored = []
        for tool_id in all_tools:
            score = 0
            for phrase in phrases.get(tool_id, set()):
                if phrase and phrase in qn:
                    score += 1 + len(phrase) / 1000.0
            scored.append((tool_id, score))
        scored.sort(key=lambda item: (-item[1], item[0]))
        rankings.append([tool_id for tool_id, _ in scored])
    return rankings


def catalog_trigger_rankings(catalog: list[dict[str, Any]], terms: dict[str, list[tuple[str, str, str]]], test_rows: list[dict[str, Any]]) -> list[list[str]]:
    rankings = []
    weights = {"positive_phrase": 5, "keyword_label": 3, "name_zh": 3, "tool_id": 10}
    for row in test_rows:
        qn = normalize_question(row.get("问题", ""))
        scored = []
        for tool in catalog:
            score = 0
            for kind, _, term in terms.get(tool["tool_id"], []):
                if term and term in qn:
                    score += 100 + weights.get(kind, 1) * len(term)
            scored.append((tool["tool_id"], score))
        scored.sort(key=lambda item: (-item[1], item[0]))
        rankings.append([tool_id for tool_id, _ in scored])
    return rankings


def ngrams(text: str, n: int) -> set[str]:
    normalized = normalize_question(text)
    if len(normalized) < n:
        return {normalized} if normalized else set()
    return {normalized[idx : idx + n] for idx in range(len(normalized) - n + 1)}


def max_ngram_jaccard(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    n: int,
    max_df: int = 500,
    top_candidates: int = 300,
) -> dict[str, Any]:
    train_sets = [ngrams(row.get("问题", ""), n) for row in train_rows]
    inverted: dict[str, set[int]] = defaultdict(set)
    for idx, grams in enumerate(train_sets):
        for gram in grams:
            inverted[gram].add(idx)

    values = []
    argmax = []
    for row in test_rows:
        grams = ngrams(row.get("问题", ""), n)
        candidate_counts: Counter[int] = Counter()
        for gram in grams:
            ids = inverted.get(gram, set())
            if len(ids) > max_df:
                continue
            candidate_counts.update(ids)
        best = 0.0
        best_id = ""
        for idx, _ in candidate_counts.most_common(top_candidates):
            other = train_sets[idx]
            union = len(grams | other)
            score = len(grams & other) / union if union else 0.0
            if score > best:
                best = score
                best_id = str(train_rows[idx].get("id", ""))
        values.append(best)
        if len(argmax) < 20:
            argmax.append({"test_id": row.get("id"), "max_jaccard": round(best, 6), "train_id": best_id})
    return {
        "n": n,
        "method": "candidate-capped exact Jaccard over train rows sharing informative n-grams",
        "max_df": max_df,
        "top_candidates": top_candidates,
        "count": len(values),
        "mean": round(statistics.mean(values), 6) if values else 0.0,
        "p50": round(statistics.median(values), 6) if values else 0.0,
        "p95": round(sorted(values)[int(0.95 * (len(values) - 1))], 6) if values else 0.0,
        "max": round(max(values), 6) if values else 0.0,
        "examples": argmax,
    }


def combination_overlap(train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]]) -> dict[str, Any]:
    train_all = Counter(tuple(split_tools(row.get("工具"))) for row in train_rows)
    test_all = Counter(tuple(split_tools(row.get("工具"))) for row in test_rows)
    train_multi = Counter(tuple(sorted(split_tools(row.get("工具")))) for row in train_rows if len(split_tools(row.get("工具"))) > 1)
    test_multi = Counter(tuple(sorted(split_tools(row.get("工具")))) for row in test_rows if len(split_tools(row.get("工具"))) > 1)
    multi_seen_rows = sum(count for combo, count in test_multi.items() if combo in train_multi)
    return {
        "train_unique_ordered_combinations": len(train_all),
        "test_unique_ordered_combinations": len(test_all),
        "test_unique_ordered_combinations_seen_in_train": sum(1 for combo in test_all if combo in train_all),
        "test_rows_ordered_combination_seen_in_train": sum(count for combo, count in test_all.items() if combo in train_all),
        "train_unique_multi_tool_sets": len(train_multi),
        "test_unique_multi_tool_sets": len(test_multi),
        "test_unique_multi_tool_sets_seen_in_train": sum(1 for combo in test_multi if combo in train_multi),
        "test_multi_rows_with_tool_set_seen_in_train": multi_seen_rows,
        "test_multi_rows_with_tool_set_seen_in_train_rate": pct(multi_seen_rows, sum(test_multi.values())),
        "test_multi_rows_with_tool_set_unseen_in_train": sum(test_multi.values()) - multi_seen_rows,
    }


def context_signature(meta_row: dict[str, Any]) -> str:
    query = str(meta_row.get("query", ""))
    phrases = sorted((meta_row.get("expression_phrases") or {}).values(), key=lambda value: len(str(value)), reverse=True)
    for phrase in phrases:
        text = str(phrase)
        if text:
            query = query.replace(text, "{PHRASE}")
    return normalize_question(query)


def context_bucket(row: dict[str, Any], meta: dict[str, dict[str, Any]]) -> str:
    meta_row = meta.get(str(row.get("id")), {})
    if int(row.get("tool_count", meta_row.get("tool_count", 0)) or 0) == 1:
        return "single"
    return str(row.get("combo_seen_bucket") or meta_row.get("combo_seen_bucket") or "multi")


def context_skeleton_overlap(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    meta: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    train_by_signature: dict[str, list[dict[str, Any]]] = defaultdict(list)
    test_by_signature: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in train_rows:
        meta_row = meta.get(str(row.get("id")), {})
        train_by_signature[context_signature(meta_row)].append(row)
    for row in test_rows:
        meta_row = meta.get(str(row.get("id")), {})
        test_by_signature[context_signature(meta_row)].append(row)

    overlap = set(train_by_signature) & set(test_by_signature)
    totals = Counter()
    by_bucket: dict[str, Counter] = defaultdict(Counter)
    examples = []
    for signature in sorted(overlap):
        train_toolsets = [tuple(split_tools(row.get("工具"))) for row in train_by_signature[signature]]
        train_tool_union = {tool for tools in train_toolsets for tool in tools}
        for row in test_by_signature[signature]:
            test_tools = tuple(split_tools(row.get("工具")))
            bucket = context_bucket(row, meta)
            totals["test_rows_with_context_seen_in_train"] += 1
            totals["test_rows_with_exact_tool_sequence_seen"] += int(test_tools in train_toolsets)
            totals["test_rows_with_all_tools_seen_under_context"] += int(set(test_tools) <= train_tool_union)
            totals["test_rows_with_any_tool_seen_under_context"] += int(bool(set(test_tools) & train_tool_union))
            by_bucket[bucket]["rows"] += 1
            by_bucket[bucket]["exact_tool_sequence_seen"] += int(test_tools in train_toolsets)
            by_bucket[bucket]["all_tools_seen_under_context"] += int(set(test_tools) <= train_tool_union)
            by_bucket[bucket]["any_tool_seen_under_context"] += int(bool(set(test_tools) & train_tool_union))
        if len(examples) < 12:
            examples.append(
                {
                    "signature": signature,
                    "train": [
                        {"id": row.get("id"), "tools": split_tools(row.get("工具"))}
                        for row in train_by_signature[signature][:3]
                    ],
                    "test": [
                        {"id": row.get("id"), "tools": split_tools(row.get("工具"))}
                        for row in test_by_signature[signature][:3]
                    ],
                }
            )

    bucket_payload = {}
    for bucket, counts in sorted(by_bucket.items()):
        rows = counts["rows"]
        bucket_payload[bucket] = {
            "test_rows_with_context_seen_in_train": rows,
            "exact_tool_sequence_seen_rate": pct(counts["exact_tool_sequence_seen"], rows),
            "all_tools_seen_under_context_rate": pct(counts["all_tools_seen_under_context"], rows),
            "any_tool_seen_under_context_rate": pct(counts["any_tool_seen_under_context"], rows),
        }

    return {
        "train_unique_context_signatures": len(train_by_signature),
        "test_unique_context_signatures": len(test_by_signature),
        "context_signatures_seen_in_train": len(overlap),
        "test_rows_with_context_seen_in_train": totals["test_rows_with_context_seen_in_train"],
        "test_rows_with_context_seen_in_train_rate": pct(totals["test_rows_with_context_seen_in_train"], len(test_rows)),
        "test_rows_with_exact_tool_sequence_seen": totals["test_rows_with_exact_tool_sequence_seen"],
        "test_rows_with_exact_tool_sequence_seen_rate": pct(totals["test_rows_with_exact_tool_sequence_seen"], len(test_rows)),
        "test_rows_with_all_tools_seen_under_context": totals["test_rows_with_all_tools_seen_under_context"],
        "test_rows_with_all_tools_seen_under_context_rate": pct(totals["test_rows_with_all_tools_seen_under_context"], len(test_rows)),
        "test_rows_with_any_tool_seen_under_context": totals["test_rows_with_any_tool_seen_under_context"],
        "test_rows_with_any_tool_seen_under_context_rate": pct(totals["test_rows_with_any_tool_seen_under_context"], len(test_rows)),
        "by_bucket": bucket_payload,
        "examples": examples,
    }


def context_skeleton_rankings(
    catalog: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    meta_rows: list[dict[str, Any]],
) -> list[list[str]]:
    all_tools = [tool["tool_id"] for tool in catalog]
    tool_counts_by_signature: dict[str, Counter[str]] = defaultdict(Counter)
    meta_lookup = {str(row.get("trace_id")): row for row in meta_rows}
    for row in meta_rows:
        if row.get("split") != "train":
            continue
        tool_counts_by_signature[context_signature(row)].update(row.get("tools") or [])

    rankings = []
    for row in test_rows:
        signature = context_signature(meta_lookup.get(str(row.get("id")), {}))
        counts = tool_counts_by_signature.get(signature, Counter())
        ranked = [tool_id for tool_id, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]
        ranked.extend(tool_id for tool_id in all_tools if tool_id not in counts)
        rankings.append(ranked)
    return rankings


def evaluate_rankings_by_bucket(
    test_rows: list[dict[str, Any]],
    rankings: list[list[str]],
    meta: dict[str, dict[str, Any]],
    k_values: list[int],
) -> dict[str, Any]:
    row_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ranking_groups: dict[str, list[list[str]]] = defaultdict(list)
    for row, ranking in zip(test_rows, rankings):
        bucket = context_bucket(row, meta)
        row_groups[bucket].append(row)
        ranking_groups[bucket].append(ranking)
    return {
        bucket: evaluate_rankings(row_groups[bucket], ranking_groups[bucket], k_values)
        for bucket in sorted(row_groups)
    }


def bm25_ablation(corpus_path: Path, test_rows: list[dict[str, Any]], k_values: list[int], tokenizer_mode: str) -> dict[str, Any]:
    corpus = load_jsonl(corpus_path)
    if not corpus:
        return {"available": False, "reason": f"missing or empty corpus: {corpus_path}"}
    tokenize = build_tokenizer(tokenizer_mode)
    fields = {
        "description_only": lambda row: row.get("description", ""),
        "full_document": lambda row: row.get("document", row.get("text", "")),
    }
    out = {"available": True}
    for name, fn in fields.items():
        bm25 = BM25([tokenize(fn(row)) for row in corpus])
        rankings = []
        for row in test_rows:
            ranked, _ = rank_from_scores(bm25.scores(tokenize(row.get("问题", ""))), corpus, max(k_values))
            rankings.append(ranked)
        out[name] = evaluate_rankings(test_rows, rankings, k_values)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    parser.add_argument("--train", type=Path, default=TOOL_TRACE_DIR / "train.jsonl")
    parser.add_argument("--test", type=Path, default=TOOL_TRACE_DIR / "test.jsonl")
    parser.add_argument("--generation-meta", type=Path, default=TOOL_TRACE_DIR / "_aux" / "generation_meta.jsonl")
    parser.add_argument("--corpus", type=Path, default=TOOL_CORPUS_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--k-values", nargs="+", type=int, default=[6, 7, 8, 9, 10, 15, 20, 30])
    parser.add_argument("--tokenizer", choices=["auto", "jieba", "fallback"], default="auto")
    args = parser.parse_args()

    catalog = load_catalog(args.catalog)
    terms = catalog_terms(catalog)
    train_rows = load_jsonl(args.train)
    test_rows = load_jsonl(args.test)
    meta_rows = load_jsonl(args.generation_meta)
    meta = meta_by_trace(args.generation_meta)

    train_keys = {normalize_question(row.get("问题", "")) for row in train_rows}
    test_keys = {normalize_question(row.get("问题", "")) for row in test_rows}
    train_ids = {str(row.get("id")) for row in train_rows}
    test_ids = {str(row.get("id")) for row in test_rows}

    catalog_rankings = catalog_trigger_rankings(catalog, terms, test_rows)
    phrase_rankings = train_phrase_rankings(catalog, test_rows, meta_rows)
    context_rankings = context_skeleton_rankings(catalog, test_rows, meta_rows)
    context_metrics = evaluate_rankings(test_rows, context_rankings, args.k_values)
    payload = {
        "train": str(args.train),
        "test": str(args.test),
        "generation_meta": str(args.generation_meta),
        "k_values": args.k_values,
        "exact_normalized_query_overlap": len(train_keys & test_keys),
        "trace_id_overlap": len(train_ids & test_ids),
        "catalog_trigger_overlap": {
            "train": trigger_overlap(train_rows, terms),
            "test": trigger_overlap(test_rows, terms),
        },
        "expression_cluster_overlap": expression_cluster_overlap(train_rows, test_rows, meta),
        "context_skeleton_overlap": context_skeleton_overlap(train_rows, test_rows, meta),
        "tool_combinations": combination_overlap(train_rows, test_rows),
        "ngram_overlap": {
            "char_4gram": max_ngram_jaccard(train_rows, test_rows, 4),
            "char_5gram": max_ngram_jaccard(train_rows, test_rows, 5),
        },
        "rule_retrievers": {
            "catalog_trigger_matcher": evaluate_rankings(test_rows, catalog_rankings, args.k_values),
            "train_expression_phrase_matcher": evaluate_rankings(test_rows, phrase_rankings, args.k_values),
            "context_skeleton_matcher": context_metrics,
            "context_skeleton_matcher_by_bucket": evaluate_rankings_by_bucket(test_rows, context_rankings, meta, args.k_values),
            "bm25": bm25_ablation(args.corpus, test_rows, args.k_values, args.tokenizer),
        },
    }
    payload["ok"] = (
        payload["exact_normalized_query_overlap"] == 0
        and payload["trace_id_overlap"] == 0
        and payload["catalog_trigger_overlap"]["test"]["rows_with_any_gold_trigger"] == 0
        and payload["expression_cluster_overlap"]["test_tool_clusters_seen_in_train"] == 0
        and payload["expression_cluster_overlap"]["test_tool_phrases_seen_in_train"] == 0
        and payload["tool_combinations"]["test_multi_rows_with_tool_set_unseen_in_train"] > 0
    )
    payload["strict_ok"] = (
        payload["ok"]
        and context_metrics.get("@10", {}).get("AllHit@K", 1.0) < 0.1
        and payload["rule_retrievers"]["context_skeleton_matcher_by_bucket"]
        .get("single", {})
        .get("@10", {})
        .get("AllHit@K", 1.0)
        < 0.2
    )
    payload["risk_flags"] = []
    if not payload["strict_ok"]:
        payload["risk_flags"].append(
            "context_skeleton_overlap: train/test expression phrases are isolated, but non-intent context skeletons can still predict part of the test labels."
        )
    write_json(args.out, payload)
    print(
        "generation leakage audit "
        f"ok={payload['ok']} query_overlap={payload['exact_normalized_query_overlap']} "
        f"catalog_test_trigger_rows={payload['catalog_trigger_overlap']['test']['rows_with_any_gold_trigger']} "
        f"phrase_matcher_AllHit@10={payload['rule_retrievers']['train_expression_phrase_matcher']['@10']['AllHit@K']} "
        f"context_skeleton_AllHit@10={payload['rule_retrievers']['context_skeleton_matcher']['@10']['AllHit@K']} "
        f"strict_ok={payload['strict_ok']} "
        f"-> {args.out}"
    )
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
