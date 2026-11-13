#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit whether the 100-tool RAG datasets are too lexical/template-heavy."""
from __future__ import annotations

import argparse
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


DEFAULT_OUTPUT = RAG_DIR / "_aux" / "dataset_ease_audit.json"


def pct(numerator: int, denominator: int) -> float:
    return round(numerator / max(1, denominator), 6)


def term_index(catalog: list[dict[str, Any]]) -> dict[str, list[tuple[str, str, str]]]:
    out: dict[str, list[tuple[str, str, str]]] = {}
    for tool in catalog:
        terms: list[tuple[str, str, str]] = []
        for kind, field in [("keyword_label", "keyword_label"), ("name_zh", "name_zh")]:
            value = str(tool.get(field, ""))
            normalized = normalize_question(value)
            if normalized:
                terms.append((kind, value, normalized))
        for phrase in tool.get("positive_phrases", []):
            normalized = normalize_question(str(phrase))
            if normalized:
                terms.append(("positive_phrase", str(phrase), normalized))
        out[tool["tool_id"]] = terms
    return out


def matching_kinds(query: str, terms: list[tuple[str, str, str]]) -> list[str]:
    normalized_query = normalize_question(query)
    return [kind for kind, _, normalized in terms if normalized and normalized in normalized_query]


def exact_overlap(rows: list[dict[str, Any]], terms: dict[str, list[tuple[str, str, str]]]) -> dict[str, Any]:
    by_tool_count: dict[int, Counter[str]] = defaultdict(Counter)
    totals = Counter()
    for row in rows:
        gold = [tool for tool in split_tools(row.get("工具")) if tool in terms]
        per_tool_hits = [matching_kinds(row["问题"], terms[tool]) for tool in gold]
        all_exact = bool(gold) and all(per_tool_hits)
        any_exact = any(per_tool_hits)
        kinds = {kind for hits in per_tool_hits for kind in hits}

        totals["queries"] += 1
        totals["any_gold_exact"] += int(any_exact)
        totals["all_gold_exact"] += int(all_exact)
        totals["keyword_any"] += int("keyword_label" in kinds)
        totals["name_any"] += int("name_zh" in kinds)
        totals["positive_phrase_any"] += int("positive_phrase" in kinds)

        bucket = by_tool_count[len(gold)]
        bucket["queries"] += 1
        bucket["any_gold_exact"] += int(any_exact)
        bucket["all_gold_exact"] += int(all_exact)

    return {
        "queries": totals["queries"],
        "any_gold_exact": totals["any_gold_exact"],
        "any_gold_exact_rate": pct(totals["any_gold_exact"], totals["queries"]),
        "all_gold_exact": totals["all_gold_exact"],
        "all_gold_exact_rate": pct(totals["all_gold_exact"], totals["queries"]),
        "keyword_any_rate": pct(totals["keyword_any"], totals["queries"]),
        "name_any_rate": pct(totals["name_any"], totals["queries"]),
        "positive_phrase_any_rate": pct(totals["positive_phrase_any"], totals["queries"]),
        "by_tool_count": {
            str(k): {
                "queries": v["queries"],
                "any_gold_exact_rate": pct(v["any_gold_exact"], v["queries"]),
                "all_gold_exact_rate": pct(v["all_gold_exact"], v["queries"]),
            }
            for k, v in sorted(by_tool_count.items())
        },
    }


def exact_term_rankings(
    catalog: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    terms: dict[str, list[tuple[str, str, str]]],
    k_values: list[int],
) -> dict[str, Any]:
    rankings: list[list[str]] = []
    ambiguous_top = 0
    zero_top = 0
    weights = {"positive_phrase": 5, "keyword_label": 3, "name_zh": 3}
    for row in rows:
        query = normalize_question(row["问题"])
        scored = []
        for tool in catalog:
            score = 0
            hits = 0
            for kind, _, normalized in terms[tool["tool_id"]]:
                if normalized and normalized in query:
                    score += 100 + weights.get(kind, 1) * len(normalized)
                    hits += 1
            if hits:
                score += hits
            scored.append((tool["tool_id"], score))
        scored.sort(key=lambda item: (-item[1], item[0]))
        zero_top += int(scored[0][1] == 0)
        ambiguous_top += int(sum(1 for _, score in scored if score == scored[0][1]) > 1)
        rankings.append([tool_id for tool_id, _ in scored])

    return {
        "description": "Exact catalog-term matcher using only keyword_label/name_zh/positive_phrases.",
        "zero_top_score": zero_top,
        "ambiguous_top_score": ambiguous_top,
        "metrics": evaluate_rankings(rows, rankings, k_values),
    }


def signature(row: dict[str, Any], terms: dict[str, list[tuple[str, str, str]]], include_phrase: bool) -> tuple:
    query = normalize_question(row["问题"])
    items = []
    for tool_id in split_tools(row.get("工具")):
        if not include_phrase:
            items.append((tool_id, ()))
            continue
        phrases = tuple(
            sorted(raw for kind, raw, normalized in terms.get(tool_id, []) if kind == "positive_phrase" and normalized in query)
        )
        items.append((tool_id, phrases))
    return tuple(items)


def signature_overlap(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    terms: dict[str, list[tuple[str, str, str]]],
) -> dict[str, Any]:
    out = {}
    for include_phrase in [False, True]:
        train_counts = Counter(signature(row, terms, include_phrase) for row in train_rows)
        test_counts = Counter(signature(row, terms, include_phrase) for row in test_rows)
        seen = set(train_counts) & set(test_counts)
        key = "tool_chain_and_phrase" if include_phrase else "tool_chain_only"
        out[key] = {
            "train_unique_signatures": len(train_counts),
            "test_unique_signatures": len(test_counts),
            "test_unique_seen_in_train": len(seen),
            "test_unique_seen_in_train_rate": pct(len(seen), len(test_counts)),
            "test_rows_seen_in_train": sum(count for sig, count in test_counts.items() if sig in seen),
            "test_rows_seen_in_train_rate": pct(sum(count for sig, count in test_counts.items() if sig in seen), len(test_rows)),
            "max_train_repeat": max(train_counts.values()) if train_counts else 0,
            "max_test_repeat": max(test_counts.values()) if test_counts else 0,
        }
    return out


def embedding_pair_overlap(path: Path, terms: dict[str, list[tuple[str, str, str]]]) -> dict[str, Any]:
    rows = load_jsonl(path)
    by_label = defaultdict(Counter)
    by_negative_type = defaultdict(Counter)
    for row in rows:
        label = int(row.get("label", 0))
        has_exact = bool(matching_kinds(row.get("query", ""), terms.get(row.get("tool_id", ""), [])))
        by_label[label]["total"] += 1
        by_label[label]["exact"] += int(has_exact)
        neg_type = row.get("negative_type") or ("positive" if label == 1 else "unknown_negative")
        by_negative_type[neg_type]["total"] += 1
        by_negative_type[neg_type]["exact"] += int(has_exact)
    return {
        "rows": len(rows),
        "by_label": {
            str(label): {
                "total": counts["total"],
                "candidate_exact_term_in_query": counts["exact"],
                "candidate_exact_term_in_query_rate": pct(counts["exact"], counts["total"]),
            }
            for label, counts in sorted(by_label.items())
        },
        "by_negative_type": {
            kind: {
                "total": counts["total"],
                "candidate_exact_term_in_query": counts["exact"],
                "candidate_exact_term_in_query_rate": pct(counts["exact"], counts["total"]),
            }
            for kind, counts in sorted(by_negative_type.items())
        },
    }


def bm25_field_ablation(
    corpus: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    k_values: list[int],
    tokenizer_mode: str,
) -> dict[str, Any]:
    fields = {
        "description_only": lambda tool: tool.get("description", ""),
        "name_description": lambda tool: "\n".join([tool.get("name_zh", ""), tool.get("description", "")]),
        "no_positive_phrases": lambda tool: "\n".join(
            [
                tool.get("name_zh", ""),
                tool.get("description", ""),
                tool.get("enhanced_desc", ""),
                tool.get("negative_boundary", ""),
            ]
        ),
        "full_document": lambda tool: tool.get("document", ""),
    }
    tokenize = build_tokenizer(tokenizer_mode)
    out = {}
    for name, text_fn in fields.items():
        bm25 = BM25([tokenize(text_fn(tool)) for tool in corpus])
        rankings = []
        for row in rows:
            tools, _ = rank_from_scores(bm25.scores(tokenize(row["问题"])), corpus, max(k_values))
            rankings.append(tools)
        out[name] = evaluate_rankings(rows, rankings, k_values)
    return out


def prediction_tool_count_breakdown(path: Path, test_rows: list[dict[str, Any]], k_values: list[int]) -> dict[str, Any]:
    if not path.exists():
        return {}
    source_by_id = {str(row.get("id")): row for row in test_rows}
    counters: dict[int, dict[int, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    for row in load_jsonl(path):
        base = source_by_id.get(str(row.get("id")), {})
        gold = set(row.get("gold_tools") or split_tools(base.get("工具")))
        pred = row.get("pred_tools", [])
        count = len(gold) or int(base.get("tool_count") or 0)
        for k in k_values:
            bucket = counters[count][k]
            bucket["total"] += 1
            bucket["all_hit"] += int(bool(gold) and gold <= set(pred[:k]))
    return {
        str(count): {
            f"AllHit@{k}": pct(counter["all_hit"], counter["total"])
            for k, counter in sorted(per_k.items())
            if counter["total"]
        }
        for count, per_k in sorted(counters.items())
    }


def build_verdict(
    exact: dict[str, Any],
    matcher: dict[str, Any],
    signatures: dict[str, Any],
    embedding_overlap: dict[str, Any],
) -> list[str]:
    matcher10 = matcher["metrics"].get("@10", {}).get("AllHit@K", 0.0)
    tool_chain_seen = signatures.get("tool_chain_only", {}).get("test_unique_seen_in_train_rate", 0.0)
    pos_exact = (
        embedding_overlap.get("test", {})
        .get("by_label", {})
        .get("1", {})
        .get("candidate_exact_term_in_query_rate", 0.0)
    )
    neg_exact = (
        embedding_overlap.get("test", {})
        .get("by_label", {})
        .get("0", {})
        .get("candidate_exact_term_in_query_rate", 0.0)
    )

    verdict = []
    if exact.get("all_gold_exact_rate", 0.0) == 0:
        verdict.append("gold 工具精确触发词覆盖率为 0，当前 query 未直接复用 catalog 触发词。")
    else:
        verdict.append(
            f"gold 工具仍有精确触发词覆盖，all_gold_exact_rate={exact.get('all_gold_exact_rate', 0.0):.4f}。"
        )

    if matcher10 < 0.1:
        verdict.append(f"朴素精确词匹配器 AllHit@10={matcher10:.4f}，显式词面规则已无法解决该测试集。")
    else:
        verdict.append(f"朴素精确词匹配器 AllHit@10={matcher10:.4f}，仍需警惕词面线索偏强。")

    verdict.append(
        f"工具链组合 train/test 复用率为 {tool_chain_seen:.4f}；这是闭集组合分布限制，不等同于 query 文本泄漏。"
    )

    if max(pos_exact, neg_exact) < 0.001:
        verdict.append(
            f"embedding pair 候选工具精确触发词命中率接近 0（pos={pos_exact:.4f}, neg={neg_exact:.4f}），负样本不能再靠触发词有无简单区分。"
        )
    else:
        verdict.append(
            f"embedding pair 仍存在少量候选工具精确触发词命中（pos={pos_exact:.4f}, neg={neg_exact:.4f}），需结合字段消融解读。"
        )
    return verdict


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    parser.add_argument("--corpus", type=Path, default=TOOL_CORPUS_PATH)
    parser.add_argument("--train-trace", type=Path, default=TOOL_TRACE_DIR / "train.jsonl")
    parser.add_argument("--test-trace", type=Path, default=TOOL_TRACE_DIR / "test.jsonl")
    parser.add_argument("--embedding-train", type=Path, default=RAG_DIR / "embedding_train.jsonl")
    parser.add_argument("--embedding-test", type=Path, default=RAG_DIR / "embedding_test.jsonl")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tokenizer", choices=["auto", "jieba", "fallback"], default="auto")
    parser.add_argument("--ks", nargs="+", type=int, default=[6, 10, 20, 30])
    args = parser.parse_args()

    catalog = load_catalog(args.catalog)
    corpus = load_jsonl(args.corpus)
    train_rows = load_jsonl(args.train_trace)
    test_rows = load_jsonl(args.test_trace)
    terms = term_index(catalog)

    prediction_files = {
        "bm25": RAG_DIR / "_aux" / "rag_baseline" / "preds" / "bm25.jsonl",
        "vector_bge_large": RAG_DIR / "_aux" / "rag_baseline" / "preds" / "vector_bge-large.jsonl",
        "hybrid_bge_large_rrf": RAG_DIR / "_aux" / "rag_baseline" / "preds" / "hybrid_bge-large_rrf.jsonl",
        "reranker_bge_base": RAG_DIR / "_aux" / "reranker_baseline" / "preds.jsonl",
    }
    breakdown_ks = [k for k in [6, 10, 20, 30] if k <= max(args.ks)]
    exact_stats = exact_overlap(test_rows, terms)
    matcher_stats = exact_term_rankings(catalog, test_rows, terms, args.ks)
    signature_stats = signature_overlap(train_rows, test_rows, terms)
    embedding_overlap = {
        "train": embedding_pair_overlap(args.embedding_train, terms),
        "test": embedding_pair_overlap(args.embedding_test, terms),
    }
    field_ablation = bm25_field_ablation(corpus, test_rows, args.ks, args.tokenizer)
    prediction_breakdown = {
        name: prediction_tool_count_breakdown(path, test_rows, breakdown_ks)
        for name, path in prediction_files.items()
    }

    audit = {
        "catalog": str(args.catalog),
        "corpus": str(args.corpus),
        "train_trace": str(args.train_trace),
        "test_trace": str(args.test_trace),
        "num_tools": len(catalog),
        "num_train_queries": len(train_rows),
        "num_test_queries": len(test_rows),
        "k_values": args.ks,
        "exact_gold_term_overlap": exact_stats,
        "exact_term_matcher": matcher_stats,
        "train_test_signature_overlap": signature_stats,
        "embedding_pair_exact_overlap": embedding_overlap,
        "bm25_field_ablation": field_ablation,
        "existing_prediction_allhit_by_tool_count": prediction_breakdown,
        "verdict": build_verdict(exact_stats, matcher_stats, signature_stats, embedding_overlap),
    }
    write_json(args.output, audit)
    exact = audit["exact_gold_term_overlap"]
    matcher = audit["exact_term_matcher"]["metrics"]
    print(
        "dataset ease audit "
        f"all_gold_exact_rate={exact['all_gold_exact_rate']:.4f} "
        f"exact_matcher_AllHit@10={matcher['@10']['AllHit@K']:.4f} "
        f"-> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
