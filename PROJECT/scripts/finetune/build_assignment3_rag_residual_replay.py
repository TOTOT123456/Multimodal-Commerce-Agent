#!/usr/bin/env python3
"""Mix prior residual-DPO errors with new residual errors for replay training."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(seed: int, tag: str, row: dict[str, Any]) -> str:
    trace_id = str(row["meta"]["trace_id"])
    return hashlib.sha256(f"{seed}:{tag}:{trace_id}".encode()).hexdigest()


def validate(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"empty preference data: {path}")
    for index, row in enumerate(rows):
        messages = row.get("messages")
        meta = row.get("meta")
        if not isinstance(messages, list) or len(messages) < 3:
            raise ValueError(f"{path}:{index}: invalid messages")
        if not isinstance(meta, dict) or not meta.get("trace_id"):
            raise ValueError(f"{path}:{index}: missing meta.trace_id")
        if not meta.get("source_trace_id") or not meta.get("semantic_pair"):
            raise ValueError(f"{path}:{index}: invalid semantic-pair metadata")
        chosen = messages[-1].get("content")
        rejected = row.get("rejected_response")
        if not chosen or not rejected or chosen == rejected:
            raise ValueError(f"{path}:{index}: invalid chosen/rejected pair")


def query_text(row: dict[str, Any]) -> str:
    return str(row["messages"][-2]["content"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--residual", type=Path, required=True)
    parser.add_argument("--replay-count", type=int, default=2500)
    parser.add_argument("--total", type=int, default=5000)
    parser.add_argument("--monitor-percent", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-out", type=Path, required=True)
    parser.add_argument("--monitor-out", type=Path, required=True)
    parser.add_argument("--audit-out", type=Path, required=True)
    args = parser.parse_args()

    if not 2000 <= args.total <= 5000 or args.total % 10:
        raise SystemExit("--total must be a multiple of 10 in 2000..5000")
    residual_count = args.total - args.replay_count
    if args.replay_count <= 0 or residual_count <= 0:
        raise SystemExit("both replay and residual counts must be positive")
    if not 0 < args.monitor_percent < 100:
        raise SystemExit("--monitor-percent must be between 1 and 99")
    for count in (args.replay_count, residual_count):
        if count * args.monitor_percent % 100:
            raise SystemExit("each source count must have an exact monitor split")

    sources = {
        "replay": (args.replay, args.replay_count),
        "residual": (args.residual, residual_count),
    }
    selected_by_source: dict[str, list[dict[str, Any]]] = {}
    inputs: dict[str, dict[str, Any]] = {}
    for tag, (path, count) in sources.items():
        rows = read_jsonl(path)
        validate(rows, path)
        if len(rows) < count:
            raise ValueError(f"{tag}: requested {count}, available {len(rows)}")
        chosen = sorted(rows, key=lambda row: stable_key(args.seed, tag, row))[:count]
        copied = []
        for row in chosen:
            value = json.loads(json.dumps(row, ensure_ascii=False))
            value["meta"]["replay_mix_source"] = tag
            copied.append(value)
        selected_by_source[tag] = copied
        inputs[tag] = {"path": str(path), "rows": len(rows), "sha256": sha256_file(path)}

    monitor: list[dict[str, Any]] = []
    train: list[dict[str, Any]] = []
    for tag, rows in selected_by_source.items():
        monitor_count = len(rows) * args.monitor_percent // 100
        ordered = sorted(rows, key=lambda row: stable_key(args.seed + 1, tag, row))
        monitor.extend(ordered[:monitor_count])
        train.extend(ordered[monitor_count:])
    selected = train + monitor
    selected.sort(key=lambda row: stable_key(args.seed + 2, "all", row))
    train.sort(key=lambda row: stable_key(args.seed + 3, "train", row))
    monitor.sort(key=lambda row: stable_key(args.seed + 4, "monitor", row))

    trace_ids = [str(row["meta"]["trace_id"]) for row in selected]
    source_ids = [str(row["meta"]["source_trace_id"]) for row in selected]
    queries = [query_text(row) for row in selected]
    train_ids = {str(row["meta"]["trace_id"]) for row in train}
    monitor_ids = {str(row["meta"]["trace_id"]) for row in monitor}
    represented_tools = {
        tool for row in selected for tool in row["meta"].get("candidate_tools", [])
    }
    checks = {
        "row_count": len(selected) == args.total,
        "train_count": len(train) == args.total * (100 - args.monitor_percent) // 100,
        "monitor_count": len(monitor) == args.total * args.monitor_percent // 100,
        "unique_trace_id": len(set(trace_ids)) == len(trace_ids),
        "unique_source_trace_id": len(set(source_ids)) == len(source_ids),
        "unique_query": len(set(queries)) == len(queries),
        "train_monitor_disjoint": not (train_ids & monitor_ids),
        "all_100_candidate_tools_represented": len(represented_tools) == 100,
        "all_pairs_semantic": all(row["meta"].get("semantic_pair") for row in selected),
        "all_rejected_strictly_wrong": all(
            row["meta"].get("strict_rejected") for row in selected
        ),
    }
    strict_ok = all(checks.values())
    if not strict_ok:
        raise ValueError(f"replay mix audit failed: {checks}")

    write_jsonl(args.out, selected)
    write_jsonl(args.train_out, train)
    write_jsonl(args.monitor_out, monitor)
    output = {
        "schema_version": 1,
        "status": "complete",
        "strict_ok": strict_ok,
        "protocol": {
            "purpose": "replay prior residual errors while learning fresh residual errors",
            "seed": args.seed,
            "monitor_percent": args.monitor_percent,
            "frozen_data_used": False,
        },
        "inputs": inputs,
        "counts": {
            "total": len(selected),
            "train": len(train),
            "monitor": len(monitor),
            "by_mix_source": dict(
                sorted(Counter(row["meta"]["replay_mix_source"] for row in selected).items())
            ),
            "train_by_mix_source": dict(
                sorted(Counter(row["meta"]["replay_mix_source"] for row in train).items())
            ),
            "monitor_by_mix_source": dict(
                sorted(Counter(row["meta"]["replay_mix_source"] for row in monitor).items())
            ),
            "by_tool_count": dict(
                sorted(Counter(str(row["meta"]["tool_count"]) for row in selected).items())
            ),
            "represented_candidate_tools": len(represented_tools),
        },
        "checks": checks,
        "outputs": {
            "preferences": {"path": str(args.out), "sha256": sha256_file(args.out)},
            "train": {"path": str(args.train_out), "sha256": sha256_file(args.train_out)},
            "monitor": {
                "path": str(args.monitor_out),
                "sha256": sha256_file(args.monitor_out),
            },
        },
    }
    write_json(args.audit_out, output)
    print(
        f"residual replay mix rows={len(selected)} train={len(train)} "
        f"monitor={len(monitor)} strict_ok={strict_ok}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
