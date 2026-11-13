#!/usr/bin/env python3
"""Select source-unique greedy-correct train rows for adaptive DPO sampling."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import CATALOG_PATH, load_catalog, write_json, write_jsonl  # noqa: E402
from finetune.tool_trace_common import (  # noqa: E402
    sha256_file,
    sha256_json,
    stable_seed,
    strict_parse_tools,
)


TOOL_COUNT_WEIGHTS = {1: 0.10, 2: 0.15, 3: 0.20, 4: 0.20, 5: 0.20, 6: 0.15}


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


def quotas(total: int) -> dict[int, int]:
    raw = {count: total * weight for count, weight in TOOL_COUNT_WEIGHTS.items()}
    result = {count: int(value) for count, value in raw.items()}
    for count in sorted(raw, key=lambda value: (-(raw[value] - result[value]), value))[
        : total - sum(result.values())
    ]:
        result[count] += 1
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument("--greedy-preds", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit-out", type=Path, required=True)
    parser.add_argument("--protected-source-plan", action="append", type=Path, default=[])
    parser.add_argument("--api-rows", type=int, default=8_000)
    parser.add_argument("--legacy-rows", type=int, default=2_000)
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    parser.add_argument("--seed", type=int, default=20_260_802)
    args = parser.parse_args()

    catalog_ids = {str(row["tool_id"]) for row in load_catalog(args.catalog)}
    compact_rows = read_jsonl(args.compact)
    compact = {str(row["trace_id"]): row for row in compact_rows}
    protected_sources = {
        str(row["construction"]["source_trace_id"])
        for path in args.protected_source_plan
        for row in read_jsonl(path)
    }
    predictions = read_jsonl(args.greedy_preds)
    if len(predictions) != len(compact) or {
        str(row["id"]) for row in predictions
    } != set(compact):
        raise ValueError("greedy predictions must exactly cover compact rows")

    correct_by_bucket: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    correct = 0
    semantic_errors = 0
    non_strict = 0
    for prediction in predictions:
        row = compact[str(prediction["id"])]
        if str(row["construction"]["source_trace_id"]) in protected_sources:
            continue
        candidates = list(row["candidate_tools"])
        predicted, strict, _ = strict_parse_tools(
            str(prediction.get("raw_output", "")), catalog_ids, set(candidates)
        )
        if not strict:
            non_strict += 1
            continue
        if predicted != list(row["gold_tools"]):
            semantic_errors += 1
            continue
        correct += 1
        correct_by_bucket[(str(row["source_split"]), int(row["tool_count"]))].append(
            row
        )

    for bucket, rows in correct_by_bucket.items():
        rows.sort(
            key=lambda row: stable_seed(
                args.seed, "adaptive-sample", bucket, row["trace_id"]
            )
        )

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    selected_sources: set[str] = set()
    targets = {
        "assignment3_api_native_train": args.api_rows,
        "legacy_train": args.legacy_rows,
    }
    for split, total in targets.items():
        before = len(selected)
        for tool_count, target in quotas(total).items():
            for row in correct_by_bucket[(split, tool_count)]:
                source_id = str(row["construction"]["source_trace_id"])
                if row["trace_id"] in selected_ids or source_id in selected_sources:
                    continue
                selected.append(row)
                selected_ids.add(str(row["trace_id"]))
                selected_sources.add(source_id)
                if sum(
                    item["source_split"] == split
                    and int(item["tool_count"]) == tool_count
                    for item in selected[before:]
                ) >= target:
                    break
        current = len(selected) - before
        if current < total:
            residual = sorted(
                (
                    row
                    for (row_split, _), rows in correct_by_bucket.items()
                    if row_split == split
                    for row in rows
                ),
                key=lambda row: stable_seed(
                    args.seed, "adaptive-sample-residual", row["trace_id"]
                ),
            )
            for row in residual:
                source_id = str(row["construction"]["source_trace_id"])
                if row["trace_id"] in selected_ids or source_id in selected_sources:
                    continue
                selected.append(row)
                selected_ids.add(str(row["trace_id"]))
                selected_sources.add(source_id)
                current += 1
                if current >= total:
                    break
        if current < total:
            raise ValueError(f"insufficient greedy-correct {split} rows: {current}/{total}")

    output_rows = [
        {
            "id": str(row["trace_id"]),
            "query": str(row["query"]),
            "candidate_tools": list(row["candidate_tools"]),
            "candidate_k": 10,
            "ranking_hash": sha256_json(row["candidate_tools"]),
        }
        for row in selected
    ]
    for row in output_rows:
        row["input_hash"] = sha256_json(row)
    write_jsonl(args.out, output_rows)
    audit = {
        "schema_version": 1,
        "status": "complete",
        "strict_ok": True,
        "selection": "greedy-correct rows only; source-unique; style quotas and tool-count targets before residual fill",
        "counts": {
            "compact": len(compact),
            "greedy_correct": correct,
            "greedy_strict_semantic_errors": semantic_errors,
            "greedy_non_strict": non_strict,
            "selected": len(selected),
            "selected_unique_queries": len(selected_ids),
            "selected_unique_sources": len(selected_sources),
            "protected_sources": len(protected_sources),
            "selected_by_style": dict(Counter(row["source_split"] for row in selected)),
            "selected_by_tool_count": dict(
                Counter(str(row["tool_count"]) for row in selected)
            ),
        },
        "inputs": {
            str(args.compact): sha256_file(args.compact),
            str(args.greedy_preds): sha256_file(args.greedy_preds),
            **{
                str(path): sha256_file(path) for path in args.protected_source_plan
            },
        },
        "output": {str(args.out): sha256_file(args.out)},
    }
    write_json(args.audit_out, audit)
    print(
        f"adaptive sample rows={len(selected)} greedy_errors={semantic_errors} "
        f"strict_ok=True"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
