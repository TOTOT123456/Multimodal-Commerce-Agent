#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Score whether static tool planners follow paired explicit order changes."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import load_jsonl, write_json  # noqa: E402
from finetune.tool_trace_common import sha256_file  # noqa: E402


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize(pairs: list[tuple[dict, dict]]) -> dict[str, float | int | None]:
    pair_count = len(pairs)
    row_count = 2 * pair_count
    rows = [row for pair in pairs for row in pair]
    both_set = sum(left["set_em"] and right["set_em"] for left, right in pairs)
    both_trace = sum(left["trace_em"] and right["trace_em"] for left, right in pairs)
    predicted_trace_changed = sum(left["pred_tools"] != right["pred_tools"] for left, right in pairs)
    expected_trace_changed = sum(left["gold_tools"] != right["gold_tools"] for left, right in pairs)
    return {
        "pairs": pair_count,
        "rows": row_count,
        "row_set_em": ratio(sum(row["set_em"] for row in rows), row_count),
        "row_trace_em": ratio(sum(row["trace_em"] for row in rows), row_count),
        "pair_both_set_em": ratio(both_set, pair_count),
        "pair_both_trace_em": ratio(both_trace, pair_count),
        "order_following_given_both_set_correct": ratio(both_trace, both_set),
        "predicted_trace_changed_rate": ratio(predicted_trace_changed, pair_count),
        "expected_trace_changed_rate": ratio(expected_trace_changed, pair_count),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--scored", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    gold_rows = load_jsonl(args.gold)
    scored_rows = load_jsonl(args.scored)
    gold_by_id = {str(row["id"]): row for row in gold_rows}
    scored_by_id = {str(row["id"]): row for row in scored_rows}
    if len(gold_by_id) != len(gold_rows) or len(scored_by_id) != len(scored_rows):
        raise SystemExit("duplicate ids")
    if set(gold_by_id) != set(scored_by_id):
        raise SystemExit("gold/scored id mismatch")
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for row_id, gold in gold_by_id.items():
        variant = str(gold.get("pair_variant", ""))
        pair_id = str(gold.get("pair_id", ""))
        if variant not in {"a", "b"} or not pair_id:
            raise SystemExit(f"invalid pair metadata: {row_id}")
        grouped[pair_id][variant] = scored_by_id[row_id]
    if any(set(values) != {"a", "b"} for values in grouped.values()):
        raise SystemExit("incomplete pairs")
    pairs = [(values["a"], values["b"]) for _, values in sorted(grouped.items())]
    by_count: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for pair in pairs:
        by_count[str(pair[0]["tool_count"])].append(pair)
    payload = {
        "analysis_version": "assignment3-order-counterfactual-pair-metrics-v1",
        "status": "complete",
        "overall": summarize(pairs),
        "tool_count": {key: summarize(value) for key, value in sorted(by_count.items())},
        "inputs": {str(args.gold): sha256_file(args.gold), str(args.scored): sha256_file(args.scored)},
    }
    if payload["overall"]["expected_trace_changed_rate"] != 1.0:
        raise SystemExit("counterfactual gold traces did not all change")
    write_json(args.out, payload)
    print(
        f"order-counterfactual pairs={payload['overall']['pairs']} "
        f"pair_both_trace={payload['overall']['pair_both_trace_em']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
