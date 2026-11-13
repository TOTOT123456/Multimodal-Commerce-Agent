#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit the scaled API-native Assignment3 train/test datasets."""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import load_jsonl, normalize_question, split_tools, write_json  # noqa: E402
from finetune.tool_trace_common import sha256_file  # noqa: E402


INTERNAL_RE = re.compile(
    r"(?i)(?:tool[_-]?[a-z0-9]|tool_id|catalog|候选工具|工具目录|MCP|RAG|API)"
)
QUOTED_ENTITY_RE = re.compile(
    r"“[^”]{1,80}”|‘[^’]{1,80}’|《[^》]{1,80}》|\"[^\"]{1,80}\"|'[^']{1,80}'"
)
NUMBER_OR_CODE_RE = re.compile(
    r"(?i)(?:[a-z]+[-_]?\d[\w-]*|\d[\w.-]*|[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,})"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def index(rows: list[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    result = {str(row["id"]): row for row in rows}
    if len(result) != len(rows):
        raise SystemExit(f"duplicate ids in {name}")
    return result


def normalized_queries(rows: list[dict[str, Any]]) -> set[str]:
    return {normalize_question(str(row["问题"])) for row in rows}


def tool_sequence(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(split_tools(row["工具"]))


def char_ngrams(text: str, n: int) -> set[str]:
    if len(text) < n:
        return {text} if text else set()
    return {text[index : index + n] for index in range(len(text) - n + 1)}


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * pct / 100.0
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def nearest_jaccard(
    train_values: list[str],
    test_values: list[str],
    train_ids: list[str],
    test_ids: list[str],
    n: int = 4,
    max_df: int = 800,
    top_candidates: int = 250,
) -> dict[str, Any]:
    """Candidate-capped exact Jaccard for surface-near-duplicate auditing."""
    gram_to_id: dict[str, int] = {}
    train_sets: list[set[int]] = []
    inverted: dict[int, list[int]] = defaultdict(list)
    for train_index, value in enumerate(train_values):
        ids = {
            gram_to_id.setdefault(gram, len(gram_to_id))
            for gram in char_ngrams(value, n)
        }
        train_sets.append(ids)
        for gram_id in ids:
            inverted[gram_id].append(train_index)

    scores: list[float] = []
    nearest: list[dict[str, Any]] = []
    for test_id, value in zip(test_ids, test_values):
        test_set = {
            gram_to_id[gram]
            for gram in char_ngrams(value, n)
            if gram in gram_to_id
        }
        candidates: Counter[int] = Counter()
        for gram_id in test_set:
            posting = inverted[gram_id]
            if len(posting) <= max_df:
                candidates.update(posting)
        best_score = 0.0
        best_index = -1
        for train_index, _ in candidates.most_common(top_candidates):
            train_set = train_sets[train_index]
            union = len(test_set | train_set)
            score = len(test_set & train_set) / union if union else 0.0
            if score > best_score:
                best_score = score
                best_index = train_index
        scores.append(best_score)
        nearest.append(
            {
                "test_id": test_id,
                "train_id": train_ids[best_index] if best_index >= 0 else "",
                "jaccard": round(best_score, 6),
            }
        )
    nearest.sort(key=lambda row: (-row["jaccard"], row["test_id"]))
    return {
        "method": (
            f"character_{n}gram_candidate_capped_exact_jaccard;"
            f"max_df={max_df};top_candidates={top_candidates}"
        ),
        "count": len(scores),
        "mean": sum(scores) / max(1, len(scores)),
        "p50": percentile(scores, 50),
        "p95": percentile(scores, 95),
        "p99": percentile(scores, 99),
        "max": max(scores, default=0.0),
        "rows_at_or_above_0_80": sum(score >= 0.80 for score in scores),
        "rows_at_or_above_0_90": sum(score >= 0.90 for score in scores),
        "top_examples": nearest[:20],
    }


def expression_proxy_signature(query: str) -> str:
    """Remove obvious variable entities while retaining the request phrasing."""
    value = QUOTED_ENTITY_RE.sub("实体", query)
    value = NUMBER_OR_CODE_RE.sub("数值", value)
    return normalize_question(value)


def mine_intent_terms(
    train_rows: list[dict[str, Any]],
    min_support: int = 20,
    min_precision: float = 0.55,
    terms_per_tool: int = 80,
) -> dict[str, list[str]]:
    """Mine label-discriminative phrases used only for leakage auditing."""
    global_counts: Counter[str] = Counter()
    per_tool: dict[str, Counter[str]] = defaultdict(Counter)
    for row in train_rows:
        normalized = normalize_question(str(row["问题"]))
        grams = char_ngrams(normalized, 3) | char_ngrams(normalized, 4)
        global_counts.update(grams)
        for tool_id in split_tools(row["工具"]):
            per_tool[tool_id].update(grams)
    result: dict[str, list[str]] = {}
    for tool_id, counts in per_tool.items():
        ranked = []
        for term, support in counts.items():
            precision = support / global_counts[term]
            if support < min_support or precision < min_precision:
                continue
            score = precision * math.log1p(support) * len(term)
            ranked.append((term, score, precision, support))
        ranked.sort(key=lambda item: (-item[1], -item[2], -item[3], item[0]))
        selected: list[str] = []
        for term, _, _, _ in ranked:
            if any(term in existing for existing in selected):
                continue
            selected.append(term)
            if len(selected) >= terms_per_tool:
                break
        result[tool_id] = sorted(selected, key=lambda value: (-len(value), value))
    return result


def context_proxy_signature(
    row: dict[str, Any],
    intent_terms: dict[str, list[str]],
) -> str:
    """Mask mined intent cues, leaving a reproducible context-skeleton proxy."""
    value = normalize_question(str(row["问题"]))
    terms = {
        term
        for tool_id in split_tools(row["工具"])
        for term in intent_terms.get(tool_id, [])
    }
    for term in sorted(terms, key=lambda item: (-len(item), item)):
        value = value.replace(term, "意图")
    return re.sub(r"(?:意图)+", "意图", value)


def signature_overlap(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    train_signatures: list[str],
    test_signatures: list[str],
) -> dict[str, Any]:
    train_by_signature: dict[str, list[tuple[str, ...]]] = defaultdict(list)
    for row, signature in zip(train_rows, train_signatures):
        train_by_signature[signature].append(tool_sequence(row))
    rows_seen = 0
    same_sequence = 0
    same_set = 0
    examples = []
    for row, signature in zip(test_rows, test_signatures):
        train_sequences = train_by_signature.get(signature, [])
        if not train_sequences:
            continue
        rows_seen += 1
        sequence = tool_sequence(row)
        sequence_seen = sequence in train_sequences
        set_seen = any(set(sequence) == set(value) for value in train_sequences)
        same_sequence += int(sequence_seen)
        same_set += int(set_seen)
        if len(examples) < 20:
            examples.append(
                {
                    "test_id": row["id"],
                    "signature": signature,
                    "test_tools": list(sequence),
                    "train_tool_sequences": [list(value) for value in train_sequences[:5]],
                    "same_sequence": sequence_seen,
                }
            )
    return {
        "train_unique_signatures": len(train_by_signature),
        "test_unique_signatures": len(set(test_signatures)),
        "test_rows_signature_seen_in_train": rows_seen,
        "test_rows_signature_seen_in_train_rate": rows_seen / max(1, len(test_rows)),
        "test_rows_same_tool_sequence_seen_under_signature": same_sequence,
        "test_rows_same_tool_set_seen_under_signature": same_set,
        "examples": examples,
    }


def distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tools = Counter(tool_id for row in rows for tool_id in split_tools(row["工具"]))
    domains = Counter(
        domain
        for row in rows
        for domain in str(row.get("domain", "")).split(",")
        if domain
    )
    return {
        "rows": len(rows),
        "tool_count": dict(sorted(Counter(int(row["tool_count"]) for row in rows).items())),
        "construction": dict(
            sorted(
                Counter(
                    row.get("construction", {}).get("type", "<missing>")
                    for row in rows
                ).items()
            )
        ),
        "domain": dict(sorted(domains.items())),
        "tool_coverage": len(tools),
        "tool_occurrence_min": min(tools.values(), default=0),
        "tool_occurrence_max": max(tools.values(), default=0),
    }


def target_overlap(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Describe closed-taxonomy target reuse without treating it as text leakage."""
    train_sequences = {tool_sequence(row) for row in train_rows}
    train_sets = {tuple(sorted(tool_sequence(row))) for row in train_rows}
    buckets: dict[str, dict[str, int | float]] = {}
    for tool_count in sorted({len(tool_sequence(row)) for row in test_rows}):
        subset = [
            row for row in test_rows if len(tool_sequence(row)) == tool_count
        ]
        sequence_seen = sum(tool_sequence(row) in train_sequences for row in subset)
        set_seen = sum(
            tuple(sorted(tool_sequence(row))) in train_sets for row in subset
        )
        buckets[str(tool_count)] = {
            "rows": len(subset),
            "ordered_sequence_seen": sequence_seen,
            "ordered_sequence_unseen": len(subset) - sequence_seen,
            "tool_set_seen": set_seen,
            "tool_set_unseen": len(subset) - set_seen,
        }
    ordered_seen = sum(
        tool_sequence(row) in train_sequences for row in test_rows
    )
    set_seen = sum(
        tuple(sorted(tool_sequence(row))) in train_sets for row in test_rows
    )
    return {
        "interpretation": (
            "Target reuse is expected in a fixed 100-tool closed taxonomy and is "
            "not query leakage. It must nevertheless be disclosed because it can "
            "make trajectory prediction easier."
        ),
        "train_unique_ordered_sequences": len(train_sequences),
        "train_unique_tool_sets": len(train_sets),
        "test_rows": len(test_rows),
        "test_ordered_sequence_seen": ordered_seen,
        "test_ordered_sequence_seen_rate": ordered_seen / max(1, len(test_rows)),
        "test_tool_set_seen": set_seen,
        "test_tool_set_seen_rate": set_seen / max(1, len(test_rows)),
        "by_tool_count": buckets,
    }


def validate_split(
    role: str,
    rows: list[dict[str, Any]],
    plan_rows: list[dict[str, Any]],
    accepted_rows: list[dict[str, Any]],
    generation_audit: dict[str, Any],
    expected_count: int,
) -> dict[str, Any]:
    data = index(rows, f"{role} data")
    plans = index(plan_rows, f"{role} plan")
    accepted = index(accepted_rows, f"{role} accepted")
    alignment_errors = []
    for trace_id, row in data.items():
        plan = plans.get(trace_id)
        generated = accepted.get(trace_id)
        if plan is None or generated is None:
            alignment_errors.append(f"{trace_id}:missing_plan_or_accepted")
            continue
        if split_tools(row["工具"]) != plan["tools"]:
            alignment_errors.append(f"{trace_id}:tool_mismatch")
        if row["问题"] != generated["question"]:
            alignment_errors.append(f"{trace_id}:query_mismatch")
        if row.get("construction") != plan.get("construction"):
            alignment_errors.append(f"{trace_id}:construction_mismatch")
    normalized = [normalize_question(str(row["问题"])) for row in rows]
    internal_rows = [row["id"] for row in rows if INTERNAL_RE.search(str(row["问题"]))]
    type_counts = Counter(
        row.get("construction", {}).get("type", "<missing>") for row in rows
    )
    blocked_protection = generation_audit.get("blocked_query_protection", {})
    blocked_protection_ok = role != "train" or (
        isinstance(blocked_protection.get("files"), dict)
        and len(blocked_protection["files"]) >= 4
        and blocked_protection.get("unique_normalized_queries", 0) > 0
        and blocked_protection.get("final_overlap_count") == 0
    )
    strict_ok = (
        len(rows) == expected_count
        and set(data) == set(plans) == set(accepted)
        and len(normalized) == len(set(normalized))
        and not alignment_errors
        and not internal_rows
        and generation_audit.get("strict_ok") is True
        and generation_audit.get("dataset_role") == role
        and generation_audit.get("query_templates_used") is False
        and generation_audit.get("target_plan_contains_queries") is False
        and generation_audit.get("training_use_allowed") == (role == "train")
        and blocked_protection_ok
    )
    return {
        "strict_ok": strict_ok,
        "rows": len(rows),
        "unique_ids": len(data),
        "unique_normalized_queries": len(set(normalized)),
        "construction_counts": dict(sorted(type_counts.items())),
        "alignment_error_count": len(alignment_errors),
        "alignment_errors": alignment_errors[:100],
        "internal_leak_count": len(internal_rows),
        "internal_leak_rows": internal_rows[:100],
        "generation_audit_strict_ok": generation_audit.get("strict_ok"),
        "blocked_query_protection_ok": blocked_protection_ok,
        "blocked_query_protection": blocked_protection,
        "training_use_allowed": role == "train",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--train-plan", type=Path, required=True)
    parser.add_argument("--train-accepted", type=Path, required=True)
    parser.add_argument("--train-generation-audit", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--test-plan", type=Path, required=True)
    parser.add_argument("--test-accepted", type=Path, required=True)
    parser.add_argument("--test-generation-audit", type=Path, required=True)
    parser.add_argument("--formal-train", type=Path, required=True)
    parser.add_argument("--frozen-test", type=Path, required=True)
    parser.add_argument("--api-dev", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-train", type=int, default=54000)
    parser.add_argument("--expected-test", type=int, default=1000)
    args = parser.parse_args()

    train = load_jsonl(args.train)
    test = load_jsonl(args.test)
    formal_train = load_jsonl(args.formal_train)
    frozen_test = load_jsonl(args.frozen_test)
    api_dev = load_jsonl(args.api_dev)
    train_plan = load_jsonl(args.train_plan)
    test_plan = load_jsonl(args.test_plan)
    train_result = validate_split(
        "train",
        train,
        train_plan,
        load_jsonl(args.train_accepted),
        load_json(args.train_generation_audit),
        args.expected_train,
    )
    test_result = validate_split(
        "test",
        test,
        test_plan,
        load_jsonl(args.test_accepted),
        load_json(args.test_generation_audit),
        args.expected_test,
    )
    train_queries = normalized_queries(train)
    test_queries = normalized_queries(test)
    overlap = {
        "train_vs_test": len(train_queries & test_queries),
        "train_vs_formal_train": len(train_queries & normalized_queries(formal_train)),
        "train_vs_frozen_test": len(train_queries & normalized_queries(frozen_test)),
        "train_vs_api_dev": len(train_queries & normalized_queries(api_dev)),
        "test_vs_formal_train": len(test_queries & normalized_queries(formal_train)),
        "test_vs_frozen_test": len(test_queries & normalized_queries(frozen_test)),
        "test_vs_api_dev": len(test_queries & normalized_queries(api_dev)),
    }
    train_ids = {str(row["id"]) for row in train}
    test_ids = {str(row["id"]) for row in test}
    trace_overlap = {
        "train_vs_test": len(train_ids & test_ids),
        "train_vs_formal_train": len(train_ids & {str(row["id"]) for row in formal_train}),
        "train_vs_frozen_test": len(train_ids & {str(row["id"]) for row in frozen_test}),
        "train_vs_api_dev": len(train_ids & {str(row["id"]) for row in api_dev}),
        "test_vs_formal_train": len(test_ids & {str(row["id"]) for row in formal_train}),
        "test_vs_frozen_test": len(test_ids & {str(row["id"]) for row in frozen_test}),
        "test_vs_api_dev": len(test_ids & {str(row["id"]) for row in api_dev}),
    }
    train_source_trace_ids = {
        str(row.get("construction", {}).get("source_trace_id"))
        for row in train_plan
    }
    test_source_trace_ids = {
        str(row.get("construction", {}).get("source_trace_id"))
        for row in test_plan
    }
    source_trace_overlap = {
        "train_unique_source_trace_ids": len(train_source_trace_ids),
        "test_unique_source_trace_ids": len(test_source_trace_ids),
        "shared_source_trace_ids": len(
            train_source_trace_ids & test_source_trace_ids
        ),
    }
    expression_train = [expression_proxy_signature(str(row["问题"])) for row in train]
    expression_test = [expression_proxy_signature(str(row["问题"])) for row in test]
    intent_terms = mine_intent_terms(train)
    context_train = [context_proxy_signature(row, intent_terms) for row in train]
    context_test = [context_proxy_signature(row, intent_terms) for row in test]
    text_structure = {
        "methodology": {
            "expression_cluster_proxy": (
                "replace quoted entities, numeric values, contact/order-like codes; "
                "compare the remaining request phrasing"
            ),
            "context_skeleton_proxy": (
                "mine train-only label-discriminative character 3/4-grams and "
                "mask gold-tool intent cues before comparing residual contexts"
            ),
            "near_duplicate_metric": "candidate-capped exact character-4gram Jaccard",
            "note": "These are deterministic leakage proxies, not semantic-equivalence labels.",
        },
        "expression_cluster_proxy": {
            **signature_overlap(train, test, expression_train, expression_test),
            "nearest_neighbor": nearest_jaccard(
                expression_train,
                expression_test,
                [str(row["id"]) for row in train],
                [str(row["id"]) for row in test],
            ),
        },
        "context_skeleton_proxy": {
            **signature_overlap(train, test, context_train, context_test),
            "nearest_neighbor": nearest_jaccard(
                context_train,
                context_test,
                [str(row["id"]) for row in train],
                [str(row["id"]) for row in test],
            ),
            "mined_intent_term_count": sum(len(values) for values in intent_terms.values()),
            "tools_with_mined_intent_terms": sum(bool(values) for values in intent_terms.values()),
        },
    }
    expression_risk_rate = (
        text_structure["expression_cluster_proxy"][
            "test_rows_same_tool_sequence_seen_under_signature"
        ]
        / max(1, len(test))
    )
    context_risk_rate = (
        text_structure["context_skeleton_proxy"][
            "test_rows_same_tool_sequence_seen_under_signature"
        ]
        / max(1, len(test))
    )
    text_structure["strict_thresholds"] = {
        "max_same_sequence_expression_signature_rate": 0.01,
        "max_same_sequence_context_signature_rate": 0.01,
        "rationale": (
            "Allow at most 1% naturally recurring short request frames; exact "
            "query overlap remains forbidden. Counts and nearest-neighbor "
            "distributions are always disclosed."
        ),
        "observed_expression_rate": expression_risk_rate,
        "observed_context_rate": context_risk_rate,
    }
    strict_ok = (
        train_result["strict_ok"]
        and test_result["strict_ok"]
        and not any(overlap.values())
        and not any(trace_overlap.values())
        and source_trace_overlap["shared_source_trace_ids"] == 0
        and expression_risk_rate <= 0.01
        and context_risk_rate <= 0.01
        and distribution(train)["tool_coverage"] == 100
        and distribution(test)["tool_coverage"] == 100
    )
    paths = [
        args.train,
        args.train_plan,
        args.train_accepted,
        args.train_generation_audit,
        args.test,
        args.test_plan,
        args.test_accepted,
        args.test_generation_audit,
        args.formal_train,
        args.frozen_test,
        args.api_dev,
    ]
    audit = {
        "audit_version": "assignment3-100tool-api-data-isolation-v2",
        "strict_ok": strict_ok,
        "train": train_result,
        "test": test_result,
        "normalized_query_overlap": overlap,
        "trace_id_overlap": trace_overlap,
        "source_trace_id_overlap": source_trace_overlap,
        "target_sequence_overlap": target_overlap(train, test),
        "text_structure_overlap": text_structure,
        "category_distribution": {
            "train": distribution(train),
            "test": distribution(test),
        },
        "frozen_test_used_for_training": False,
        "api_dev_used_for_training": False,
        "hashes": {str(path): sha256_file(path) for path in paths},
    }
    write_json(args.output, audit)
    print(
        f"API data audit train={len(train)} test={len(test)} "
        f"overlap={sum(overlap.values())} strict_ok={strict_ok}"
    )
    return 0 if strict_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
