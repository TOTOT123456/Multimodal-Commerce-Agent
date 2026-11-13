#!/usr/bin/env python3
"""Prepare a train-only, eval-source-excluded pool for DPO-v2 mining."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SOURCE_SPLITS = ("assignment3_api_native_train", "legacy_train")
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


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def stable_hash(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).hexdigest()


def normalize_query(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def quotas(total: int) -> dict[int, int]:
    raw = {tool_count: total * weight for tool_count, weight in TOOL_COUNT_WEIGHTS.items()}
    result = {tool_count: int(value) for tool_count, value in raw.items()}
    remainder = total - sum(result.values())
    order = sorted(
        raw,
        key=lambda tool_count: (
            -(raw[tool_count] - result[tool_count]),
            tool_count,
        ),
    )
    for tool_count in order[:remainder]:
        result[tool_count] += 1
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument(
        "--protected-source-plan",
        action="append",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--historical-blocked-query-file",
        action="append",
        type=Path,
        default=[],
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument("--per-source", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_729)
    args = parser.parse_args()
    if args.per_source <= 0:
        raise SystemExit("--per-source must be positive")

    protected_sources: set[str] = set()
    for path in args.protected_source_plan:
        for row in read_jsonl(path):
            construction = row.get("construction")
            if not isinstance(construction, dict) or not construction.get(
                "source_trace_id"
            ):
                raise ValueError(f"{path}: missing construction.source_trace_id")
            protected_sources.add(str(construction["source_trace_id"]))

    historical_ids: set[str] = set()
    historical_queries: set[str] = set()
    for path in args.historical_blocked_query_file:
        for row in read_jsonl(path):
            historical_ids.add(str(row.get("id", row.get("trace_id", ""))))
            query = row.get("问题", row.get("question", row.get("query")))
            if isinstance(query, str) and query.strip():
                historical_queries.add(normalize_query(query))

    compact = read_jsonl(args.compact)
    buckets: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    excluded_protected = Counter()
    seen_ids: set[str] = set()
    seen_queries: set[str] = set()
    for row in compact:
        trace_id = str(row["trace_id"])
        query = normalize_query(str(row["query"]))
        source_split = str(row["source_split"])
        tool_count = int(row["tool_count"])
        source_trace_id = str(
            row.get("construction", {}).get("source_trace_id", trace_id)
        )
        if source_split not in SOURCE_SPLITS or tool_count not in TOOL_COUNT_WEIGHTS:
            raise ValueError(f"{trace_id}: unexpected source/tool count")
        if row.get("view") != "rag" or int(row.get("candidate_k", 0)) != 10:
            raise ValueError(f"{trace_id}: expected deployment RAG Top-10")
        candidates = row.get("candidate_tools")
        gold = row.get("gold_tools")
        if (
            not isinstance(candidates, list)
            or len(candidates) != 10
            or len(candidates) != len(set(candidates))
            or not isinstance(gold, list)
            or not set(gold).issubset(candidates)
        ):
            raise ValueError(f"{trace_id}: invalid candidates/gold")
        if trace_id in seen_ids or query in seen_queries:
            raise ValueError(f"{trace_id}: duplicate id or query")
        seen_ids.add(trace_id)
        seen_queries.add(query)
        if trace_id in protected_sources or source_trace_id in protected_sources:
            excluded_protected[source_split] += 1
            continue
        buckets[(source_split, tool_count)].append(row)

    targets = quotas(args.per_source)
    selected: list[dict[str, Any]] = []
    for source_split in SOURCE_SPLITS:
        for tool_count, target in sorted(targets.items()):
            pool = sorted(
                buckets[(source_split, tool_count)],
                key=lambda row: stable_hash(
                    args.seed,
                    source_split,
                    tool_count,
                    row["trace_id"],
                ),
            )
            if len(pool) < target:
                raise ValueError(
                    f"insufficient {source_split}/tool_count={tool_count}: "
                    f"need={target} have={len(pool)}"
                )
            selected.extend(pool[:target])
    selected.sort(key=lambda row: stable_hash(args.seed, "order", row["trace_id"]))

    selected_ids = {str(row["trace_id"]) for row in selected}
    selected_queries = {normalize_query(str(row["query"])) for row in selected}
    selected_sources = {
        str(row.get("construction", {}).get("source_trace_id", row["trace_id"]))
        for row in selected
    }
    leakage = {
        "protected_source_trace_overlap": len(selected_sources & protected_sources),
        "protected_trace_id_overlap": len(selected_ids & protected_sources),
        "historical_blocked_id_overlap": len(selected_ids & historical_ids),
        "historical_blocked_query_overlap": len(
            selected_queries & historical_queries
        ),
    }
    mining_rows = []
    for row in selected:
        payload = {
            "id": row["trace_id"],
            "query": row["query"],
            "candidate_tools": row["candidate_tools"],
            "candidate_k": 10,
            "ranking_hash": row["ranking_hash"],
        }
        payload["input_hash"] = sha256_json(payload)
        mining_rows.append(payload)
    write_jsonl(args.out, mining_rows)

    source_counts = Counter(str(row["source_split"]) for row in selected)
    tool_counts = Counter(
        f"{row['source_split']}:{row['tool_count']}" for row in selected
    )
    strict_ok = (
        len(selected) == 2 * args.per_source
        and len(selected_ids) == len(selected)
        and len(selected_queries) == len(selected)
        and not any(leakage.values())
        and all(source_counts[source] == args.per_source for source in SOURCE_SPLITS)
    )
    manifest = {
        "manifest_version": "assignment3-crossstyle-dpo-v2-mining-v1",
        "strict_ok": strict_ok,
        "purpose": "train_side_current_parent_hard_negative_mining_only",
        "seed": args.seed,
        "sampling": {
            "per_source": args.per_source,
            "tool_count_weights": TOOL_COUNT_WEIGHTS,
            "tool_count_quota_per_source": targets,
        },
        "counts": {
            "compact_rows": len(compact),
            "selected_rows": len(selected),
            "by_source": dict(sorted(source_counts.items())),
            "by_source_and_tool_count": dict(sorted(tool_counts.items())),
            "excluded_by_protected_source": dict(
                sorted(excluded_protected.items())
            ),
        },
        "candidate_protocol": {
            "view": "rag",
            "candidate_k": 10,
            "all_selected_rows_contain_all_gold": True,
            "candidate_order_variants_are_applied_only_during_inference": True,
        },
        "isolation": {
            **leakage,
            "protected_source_trace_ids": len(protected_sources),
            "new_dev_or_final_query_text_read_by_builder": False,
            "new_dev_or_final_gold_trajectory_used_for_training": False,
        },
        "inputs": {
            str(args.compact): sha256_file(args.compact),
            **{
                str(path): sha256_file(path)
                for path in args.protected_source_plan
                + args.historical_blocked_query_file
            },
        },
        "output": str(args.out),
        "output_sha256": sha256_file(args.out),
        "contains_gold_fields": False,
    }
    write_json(args.manifest_out, manifest)
    print(
        f"DPO-v2 mining rows={len(selected)} "
        f"api={source_counts['assignment3_api_native_train']} "
        f"legacy={source_counts['legacy_train']} strict_ok={strict_ok}"
    )
    return 0 if strict_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
