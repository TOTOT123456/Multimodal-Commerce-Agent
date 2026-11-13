#!/usr/bin/env python3
"""Prepare a balanced, label-free train-side pool for cross-style DPO mining."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXP = (
    ROOT
    / "PROJECT/data/tool_call_trace/_aux/assignment3"
    / "assignment3_crossstyle_rag10/dpo"
)
SOURCE_SPLITS = (
    "assignment3_api_native_train",
    "legacy_train",
)
TOOL_COUNT_WEIGHTS = {
    1: 0.10,
    2: 0.15,
    3: 0.20,
    4: 0.20,
    5: 0.20,
    6: 0.15,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compact",
        type=Path,
        default=ROOT
        / "PROJECT/data/tool_call_trace/_aux/assignment3"
        / "assignment3_crossstyle_rag10/sft_compact.jsonl",
    )
    parser.add_argument(
        "--legacy-frozen",
        type=Path,
        default=ROOT / "PROJECT/data/tool_call_trace/test.jsonl",
    )
    parser.add_argument(
        "--api-frozen",
        type=Path,
        default=ROOT
        / "PROJECT/data/tool_call_trace/_aux/assignment3"
        / "assignment3_100tool_api/test_api/test.jsonl",
    )
    parser.add_argument(
        "--selection-dev",
        type=Path,
        default=ROOT
        / "PROJECT/data/tool_call_trace/_aux/assignment3"
        / "api_native_dev_2400/dev.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_EXP / "mining_label_free.jsonl",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=DEFAULT_EXP / "mining_manifest.json",
    )
    parser.add_argument(
        "--per-source",
        type=int,
        default=18_000,
        help="Rows selected from each style. The final pool has twice this size.",
    )
    parser.add_argument("--seed", type=int, default=20_260_729)
    return parser.parse_args()


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
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def stable_hash(*parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def normalize_query(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def quotas(total: int) -> dict[int, int]:
    raw = {
        tool_count: total * weight
        for tool_count, weight in TOOL_COUNT_WEIGHTS.items()
    }
    result = {tool_count: int(value) for tool_count, value in raw.items()}
    remaining = total - sum(result.values())
    ranking = sorted(
        raw,
        key=lambda tool_count: (
            -(raw[tool_count] - result[tool_count]),
            tool_count,
        ),
    )
    for tool_count in ranking[:remaining]:
        result[tool_count] += 1
    return result


def blocked_index(paths: list[Path]) -> tuple[set[str], set[str]]:
    ids: set[str] = set()
    queries: set[str] = set()
    for path in paths:
        for row in read_jsonl(path):
            ids.add(str(row["id"]))
            queries.add(normalize_query(str(row["问题"])))
    return ids, queries


def main() -> int:
    args = parse_args()
    if args.per_source <= 0:
        raise ValueError("--per-source must be positive")

    compact = read_jsonl(args.compact)
    by_bucket: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    seen_ids: set[str] = set()
    seen_queries: set[str] = set()
    for row in compact:
        trace_id = str(row["trace_id"])
        query = normalize_query(str(row["query"]))
        source_split = str(row["source_split"])
        tool_count = int(row["tool_count"])
        if source_split not in SOURCE_SPLITS:
            raise ValueError(f"unexpected source split: {source_split}")
        if tool_count not in TOOL_COUNT_WEIGHTS:
            raise ValueError(f"unexpected tool count: {tool_count}")
        if row.get("view") != "rag" or int(row.get("candidate_k", 0)) != 10:
            raise ValueError(f"{trace_id}: expected RAG Top-10")
        candidates = row.get("candidate_tools")
        gold = row.get("gold_tools")
        if (
            not isinstance(candidates, list)
            or len(candidates) != 10
            or len(candidates) != len(set(candidates))
            or not isinstance(gold, list)
            or not set(gold).issubset(candidates)
        ):
            raise ValueError(f"{trace_id}: invalid candidates or gold")
        if trace_id in seen_ids:
            raise ValueError(f"duplicate trace id: {trace_id}")
        if query in seen_queries:
            raise ValueError(f"duplicate normalized query: {trace_id}")
        seen_ids.add(trace_id)
        seen_queries.add(query)
        by_bucket[(source_split, tool_count)].append(row)

    target_quotas = quotas(args.per_source)
    selected: list[dict[str, Any]] = []
    for source_split in SOURCE_SPLITS:
        for tool_count, target in sorted(target_quotas.items()):
            pool = sorted(
                by_bucket[(source_split, tool_count)],
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
                    f"need {target}, have {len(pool)}"
                )
            selected.extend(pool[:target])
    selected.sort(
        key=lambda row: stable_hash(args.seed, "final-order", row["trace_id"])
    )

    blocked_paths = [args.legacy_frozen, args.api_frozen, args.selection_dev]
    blocked_ids, blocked_queries = blocked_index(blocked_paths)
    selected_ids = {str(row["trace_id"]) for row in selected}
    selected_queries = {normalize_query(str(row["query"])) for row in selected}
    source_trace_ids = {
        str(row.get("construction", {}).get("source_trace_id"))
        for row in selected
        if row.get("construction", {}).get("source_trace_id")
    }
    leakage = {
        "blocked_id_overlap": len(selected_ids & blocked_ids),
        "blocked_query_overlap": len(selected_queries & blocked_queries),
        "api_source_trace_blocked_id_overlap": len(source_trace_ids & blocked_ids),
    }
    if any(leakage.values()):
        raise ValueError(f"blocked-evaluation leakage: {leakage}")

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

    source_counts = Counter(row["source_split"] for row in selected)
    tool_counts = Counter(
        f"{row['source_split']}:{row['tool_count']}" for row in selected
    )
    manifest = {
        "status": "complete",
        "strict_ok": True,
        "purpose": "train_side_dpo_negative_mining_only",
        "seed": args.seed,
        "sampling": {
            "per_source": args.per_source,
            "tool_count_weights": TOOL_COUNT_WEIGHTS,
            "target_quotas_per_source": target_quotas,
        },
        "counts": {
            "compact_rows": len(compact),
            "selected_rows": len(selected),
            "by_source": dict(sorted(source_counts.items())),
            "by_source_and_tool_count": dict(sorted(tool_counts.items())),
        },
        "candidate_protocol": {
            "view": "rag",
            "candidate_k": 10,
            "all_selected_rows_contain_all_gold": True,
            "label_free_field_whitelist": [
                "id",
                "query",
                "candidate_tools",
                "candidate_k",
                "ranking_hash",
                "input_hash",
            ],
        },
        "isolation": {
            **leakage,
            "blocked_paths": [str(path) for path in blocked_paths],
            "frozen_or_dev_rows_used_for_sampling": 0,
        },
        "inputs": {
            str(args.compact): sha256_file(args.compact),
            **{str(path): sha256_file(path) for path in blocked_paths},
        },
        "output": str(args.out),
        "output_sha256": sha256_file(args.out),
        "contains_gold_fields": False,
    }
    manifest["strict_ok"] = (
        len(selected) == 2 * args.per_source
        and len(selected_ids) == len(selected)
        and len(selected_queries) == len(selected)
        and not any(leakage.values())
        and all(
            count == args.per_source
            for count in source_counts.values()
        )
    )
    write_json(args.manifest_out, manifest)
    print(
        f"mining rows={len(selected)} "
        f"api={source_counts['assignment3_api_native_train']} "
        f"legacy={source_counts['legacy_train']} "
        f"strict_ok={manifest['strict_ok']}"
    )
    return 0 if manifest["strict_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
