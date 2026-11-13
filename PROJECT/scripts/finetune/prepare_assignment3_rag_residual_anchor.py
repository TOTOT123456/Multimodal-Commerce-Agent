#!/usr/bin/env python3
"""Build a compact residual-error anchor without reading any frozen test label."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(seed: int, row_id: str) -> str:
    return hashlib.sha256(f"{seed}:{row_id}".encode()).hexdigest()


def unique_by_id(rows: list[dict[str, Any]], path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = str(row["id"])
        if row_id in result:
            raise ValueError(f"duplicate id {row_id}: {path}")
        result[row_id] = row
    return result


def source_id(row: dict[str, Any]) -> str:
    construction = row.get("construction") or {}
    return str(construction.get("source_trace_id") or row["id"])


def bucket(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["style"]), int(row["tool_count"])


def select_controls(
    correct_rows: list[dict[str, Any]], count: int, seed: int
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in correct_rows:
        groups[bucket(row)].append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: stable_key(seed, str(row["id"])))

    selected: list[dict[str, Any]] = []
    keys = sorted(groups)
    while len(selected) < count:
        progressed = False
        for key in keys:
            if groups[key] and len(selected) < count:
                selected.append(groups[key].pop(0))
                progressed = True
        if not progressed:
            raise ValueError(f"only {len(selected)} correct controls available; requested {count}")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--parent-scored", type=Path, required=True)
    parser.add_argument("--parent-predictions", type=Path, required=True)
    parser.add_argument("--preferences", type=Path, required=True)
    parser.add_argument("--frozen-test", type=Path, required=True)
    parser.add_argument("--control-count", type=int, default=190)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--gold-out", type=Path, required=True)
    parser.add_argument("--input-out", type=Path, required=True)
    parser.add_argument("--parent-predictions-out", type=Path, required=True)
    parser.add_argument("--audit-out", type=Path, required=True)
    args = parser.parse_args()

    gold = read_jsonl(args.gold)
    inputs = read_jsonl(args.input)
    scored = read_jsonl(args.parent_scored)
    parent_predictions = read_jsonl(args.parent_predictions)
    preferences = read_jsonl(args.preferences)
    frozen = read_jsonl(args.frozen_test)
    gold_by_id = unique_by_id(gold, args.gold)
    input_by_id = unique_by_id(inputs, args.input)
    scored_by_id = unique_by_id(scored, args.parent_scored)
    parent_predictions_by_id = unique_by_id(parent_predictions, args.parent_predictions)
    if (
        set(gold_by_id) != set(input_by_id)
        or set(gold_by_id) != set(scored_by_id)
        or set(gold_by_id) != set(parent_predictions_by_id)
    ):
        raise ValueError(
            "gold, label-free input, parent predictions, and parent-scored ID sets differ"
        )

    hard = [gold_by_id[row_id] for row_id, row in scored_by_id.items() if not row["trace_em"]]
    correct = [gold_by_id[row_id] for row_id, row in scored_by_id.items() if row["trace_em"]]
    controls = select_controls(correct, args.control_count, args.seed)
    selected_ids = {str(row["id"]) for row in hard + controls}
    selected_gold = [row for row in gold if str(row["id"]) in selected_ids]
    selected_input = [input_by_id[str(row["id"])] for row in selected_gold]
    selected_parent_predictions = [
        parent_predictions_by_id[str(row["id"])] for row in selected_gold
    ]

    pref_sources = {
        str((row.get("meta") or {}).get("source_trace_id") or "") for row in preferences
    }
    frozen_ids = {str(row["id"]) for row in frozen}
    frozen_queries = {str(row.get("问题") or row.get("query") or "") for row in frozen}
    selected_sources = {source_id(row) for row in selected_gold}
    selected_queries = {str(row.get("问题") or row.get("query") or "") for row in selected_gold}
    source_overlap = sorted(selected_sources & pref_sources)
    frozen_id_overlap = sorted(selected_ids & frozen_ids)
    frozen_query_overlap = sorted(selected_queries & frozen_queries)
    if source_overlap or frozen_id_overlap or frozen_query_overlap:
        raise ValueError("residual anchor violates preference or frozen-test isolation")

    hard_ids = {str(row["id"]) for row in hard}
    bucket_counts = Counter(
        ("hard" if str(row["id"]) in hard_ids else "control", *bucket(row))
        for row in selected_gold
    )
    audit = {
        "schema_version": 1,
        "status": "complete",
        "strict_ok": True,
        "purpose": "residual-error hyperparameter selection anchor; not a protected final set",
        "official_frozen_sets_used_for_selection": False,
        "selection": {
            "parent_errors": len(hard),
            "correct_controls": len(controls),
            "rows": len(selected_gold),
            "seed": args.seed,
            "rule": "include every parent TraceEM error, then add deterministic style/tool-count-balanced correct controls",
        },
        "isolation": {
            "preference_source_overlap": len(source_overlap),
            "frozen_id_overlap": len(frozen_id_overlap),
            "frozen_query_overlap": len(frozen_query_overlap),
        },
        "bucket_counts": {
            f"{kind}|{style}|{tool_count}": count
            for (kind, style, tool_count), count in sorted(bucket_counts.items())
        },
        "inputs": {
            str(args.gold): sha256(args.gold),
            str(args.input): sha256(args.input),
            str(args.parent_scored): sha256(args.parent_scored),
            str(args.parent_predictions): sha256(args.parent_predictions),
            str(args.preferences): sha256(args.preferences),
            str(args.frozen_test): sha256(args.frozen_test),
        },
    }
    write_jsonl(args.gold_out, selected_gold)
    write_jsonl(args.input_out, selected_input)
    write_jsonl(args.parent_predictions_out, selected_parent_predictions)
    args.audit_out.parent.mkdir(parents=True, exist_ok=True)
    args.audit_out.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"residual anchor rows={len(selected_gold)} hard={len(hard)} "
        f"controls={len(controls)} strict_ok=True"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
