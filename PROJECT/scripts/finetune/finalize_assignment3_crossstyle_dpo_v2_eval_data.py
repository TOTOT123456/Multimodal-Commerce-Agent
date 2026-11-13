#!/usr/bin/env python3
"""Merge and strictly audit sealed cross-style DPO-v2 dev/final datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


CELLS = (
    ("dev", "api_native"),
    ("dev", "legacy_compact"),
    ("final", "api_native"),
    ("final", "legacy_compact"),
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected object")
    return value


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_query(value: str) -> str:
    return re.sub(r"[\W_]+", "", value.casefold())


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("id", row.get("trace_id", "")))


def row_query(row: dict[str, Any]) -> str:
    value = row.get("问题", row.get("question", row.get("query", "")))
    return str(value).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument(
        "--blocked-query-file",
        action="append",
        type=Path,
        default=[],
    )
    parser.add_argument(
        "--blocked-source-plan",
        action="append",
        type=Path,
        default=[],
    )
    parser.add_argument("--expected-per-style", type=int, default=600)
    args = parser.parse_args()

    plan_audit_path = args.protocol_dir / "plan_audit.json"
    plan_audit = read_json(plan_audit_path)
    cell_rows: dict[str, list[dict[str, Any]]] = {}
    cell_plans: dict[str, list[dict[str, Any]]] = {}
    cell_audits: dict[str, dict[str, Any]] = {}
    alignment_errors: list[str] = []
    for role, style in CELLS:
        key = f"{role}_{style}"
        plan_path = args.protocol_dir / f"{key}_plan.jsonl"
        generation_dir = args.protocol_dir / key
        dataset_name = "dev.jsonl" if role == "dev" else "test.jsonl"
        dataset_path = generation_dir / dataset_name
        audit_path = generation_dir / "generation_audit.json"
        plan = read_jsonl(plan_path)
        rows = read_jsonl(dataset_path)
        audit = read_json(audit_path)
        plan_by_id = {str(row["id"]): row for row in plan}
        rows_by_id = {str(row["id"]): row for row in rows}
        if len(plan_by_id) != len(plan) or len(rows_by_id) != len(rows):
            alignment_errors.append(f"{key}:duplicate_ids")
        if set(plan_by_id) != set(rows_by_id):
            alignment_errors.append(f"{key}:id_set_mismatch")
        for trace_id in sorted(set(plan_by_id) & set(rows_by_id)):
            plan_row = plan_by_id[trace_id]
            data_row = rows_by_id[trace_id]
            if str(data_row.get("工具", "")).split(",") != plan_row["tools"]:
                alignment_errors.append(f"{key}:{trace_id}:tools")
            if int(data_row.get("tool_count", 0)) != int(plan_row["tool_count"]):
                alignment_errors.append(f"{key}:{trace_id}:tool_count")
            if data_row.get("style") != style:
                alignment_errors.append(f"{key}:{trace_id}:style")
            if data_row.get("construction") != plan_row.get("construction"):
                alignment_errors.append(f"{key}:{trace_id}:construction")
        if (
            not audit.get("strict_ok")
            or audit.get("training_use_allowed") is not False
            or audit.get("style_profile") != style
            or int(audit.get("accepted_count", -1)) != args.expected_per_style
        ):
            alignment_errors.append(f"{key}:generation_audit")
        cell_rows[key] = rows
        cell_plans[key] = plan
        cell_audits[key] = audit

    blocked_queries: set[str] = set()
    blocked_ids: set[str] = set()
    for path in args.blocked_query_file:
        for row in read_jsonl(path):
            trace_id = row_id(row)
            query = row_query(row)
            if trace_id:
                blocked_ids.add(trace_id)
            if query:
                blocked_queries.add(normalize_query(query))

    blocked_sources: set[str] = set()
    for path in args.blocked_source_plan:
        for row in read_jsonl(path):
            construction = row.get("construction")
            if isinstance(construction, dict) and construction.get("source_trace_id"):
                blocked_sources.add(str(construction["source_trace_id"]))

    all_rows = [row for key in cell_rows for row in cell_rows[key]]
    all_plans = [row for key in cell_plans for row in cell_plans[key]]
    ids = [str(row["id"]) for row in all_rows]
    queries = [normalize_query(str(row["问题"])) for row in all_rows]
    source_ids = [
        str(row["construction"]["source_trace_id"]) for row in all_plans
    ]
    overlap = {
        "evaluation_duplicate_ids": len(ids) - len(set(ids)),
        "evaluation_duplicate_normalized_queries": len(queries) - len(set(queries)),
        "evaluation_duplicate_source_trace_ids": len(source_ids)
        - len(set(source_ids)),
        "blocked_id_overlap": len(set(ids) & blocked_ids),
        "blocked_query_overlap": len(set(queries) & blocked_queries),
        "blocked_source_trace_overlap": len(set(source_ids) & blocked_sources),
    }

    dev_rows = cell_rows["dev_api_native"] + cell_rows["dev_legacy_compact"]
    final_rows = (
        cell_rows["final_api_native"] + cell_rows["final_legacy_compact"]
    )
    dev_path = args.protocol_dir / "dev.jsonl"
    final_path = args.protocol_dir / "final.jsonl"
    write_jsonl(dev_path, dev_rows)
    write_jsonl(final_path, final_rows)

    counts = {
        key: {
            "rows": len(rows),
            "tool_count": dict(
                sorted(Counter(int(row["tool_count"]) for row in rows).items())
            ),
            "unique_queries": len(
                {normalize_query(str(row["问题"])) for row in rows}
            ),
        }
        for key, rows in cell_rows.items()
    }
    strict_ok = (
        plan_audit.get("strict_ok") is True
        and not alignment_errors
        and not any(overlap.values())
        and len(dev_rows) == 2 * args.expected_per_style
        and len(final_rows) == 2 * args.expected_per_style
        and all(
            row["rows"] == args.expected_per_style
            and row["unique_queries"] == args.expected_per_style
            and row["tool_count"]
            == {value: args.expected_per_style // 6 for value in range(1, 7)}
            for row in counts.values()
        )
    )
    audit = {
        "audit_version": "assignment3-crossstyle-dpo-v2-eval-data-v1",
        "strict_ok": strict_ok,
        "training_use_allowed": False,
        "protocol_frozen_at": "2026-07-29",
        "counts": {
            "dev": len(dev_rows),
            "final": len(final_rows),
            "cells": counts,
        },
        "isolation": overlap,
        "alignment_error_count": len(alignment_errors),
        "alignment_errors": alignment_errors[:100],
        "expression_cluster_audit": {
            "generated_rows_with_expression_cluster_ids": sum(
                "expression_cluster_ids" in row for row in all_rows
            ),
            "note": (
                "Both styles are independently free-form API generated and do "
                "not use the legacy expression-cluster bank."
            ),
        },
        "context_skeleton_audit": {
            "query_templates_used": False,
            "target_plans_contain_queries": False,
            "exact_normalized_query_overlap": overlap["blocked_query_overlap"],
            "note": (
                "Template-family or context-skeleton identity is not applicable "
                "to these free-form generated queries; exact query and source "
                "trace isolation are enforced instead."
            ),
        },
        "sealing": {
            "dev_role": "checkpoint selection and guardrails only",
            "final_role": "one-shot evaluation only",
            "final_not_available_to_preference_builder": True,
            "post_final_tuning_forbidden": True,
        },
        "inputs": {
            str(plan_audit_path): sha256_file(plan_audit_path),
            **{
                str(path): sha256_file(path)
                for path in args.blocked_query_file + args.blocked_source_plan
            },
            **{
                f"{key}:generation_audit": sha256_file(
                    args.protocol_dir / key / "generation_audit.json"
                )
                for key in cell_audits
            },
        },
        "outputs": {
            "dev": {"path": str(dev_path), "sha256": sha256_file(dev_path)},
            "final": {"path": str(final_path), "sha256": sha256_file(final_path)},
        },
    }
    write_json(args.protocol_dir / "data_audit.json", audit)
    print(
        f"cross-style eval dev={len(dev_rows)} final={len(final_rows)} "
        f"blocked_query_overlap={overlap['blocked_query_overlap']} "
        f"strict_ok={strict_ok}"
    )
    return 0 if strict_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
