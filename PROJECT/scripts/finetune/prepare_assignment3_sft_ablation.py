#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build deterministic nested Assignment3 SFT data-scale subsets."""
from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parent
ASSIGNMENT3 = (
    PROJECT_ROOT / "data" / "tool_call_trace" / "_aux" / "assignment3"
    / "assignment3_100tool_api"
)
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "assignment3_sft_ablation.json"
DEFAULT_COMPACT = ASSIGNMENT3 / "sft_compact.jsonl"
DEFAULT_RUNTIME = (
    PROJECT_ROOT / "finetune" / "outputs" / "tooltrace-data-cache" / "runs"
    / "assignment3_100tool_sft.runtime.jsonl"
)
DEFAULT_OUT_DIR = (
    PROJECT_ROOT / "finetune" / "outputs" / "tooltrace-data-cache"
    / "assignment3_sft_ablation"
)
DEFAULT_AUDIT = ASSIGNMENT3 / "sft_ablation" / "data_audit.json"
REFERENCE_DATASETS = {
    "api_native_dev_2400": (
        PROJECT_ROOT / "data" / "tool_call_trace" / "_aux" / "assignment3"
        / "api_native_dev_2400" / "dev.jsonl"
    ),
    "api_native_test_1000": ASSIGNMENT3 / "test_api" / "test.jsonl",
    "legacy_frozen_test_6000": PROJECT_ROOT / "data" / "tool_call_trace" / "test.jsonl",
}
EXPECTED_CONSTRUCTION_COUNTS = {
    9000: {
        "single_intent": 3000,
        "multi_candidate_order_shuffled": 4000,
        "multi_same_type_distractors": 2000,
    },
    18000: {
        "single_intent": 6000,
        "multi_candidate_order_shuffled": 8000,
        "multi_same_type_distractors": 4000,
    },
    36000: {
        "single_intent": 12000,
        "multi_candidate_order_shuffled": 16000,
        "multi_same_type_distractors": 8000,
    },
    54000: {
        "single_intent": 18000,
        "multi_candidate_order_shuffled": 24000,
        "multi_same_type_distractors": 12000,
    },
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT.resolve()))


def normalize_query(value: str) -> str:
    return "".join(unicodedata.normalize("NFKC", value).lower().split())


def query_of(row: dict[str, Any]) -> str:
    for key in ("query", "问题"):
        if isinstance(row.get(key), str):
            return row[key]
    raise ValueError(f"row has no query field: {sorted(row)}")


def id_of(row: dict[str, Any]) -> str:
    for key in ("trace_id", "id"):
        if isinstance(row.get(key), str):
            return row[key]
    raise ValueError(f"row has no id field: {sorted(row)}")


def gold_from_runtime(row: dict[str, Any]) -> list[str]:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("runtime row has no messages")
    payload = json.loads(messages[-1]["content"])
    tools = payload.get("tools")
    if not isinstance(tools, list) or not all(isinstance(tool, str) for tool in tools):
        raise ValueError("runtime assistant target is not a tools array")
    return tools


def stable_order(trace_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{trace_id}".encode("utf-8")).hexdigest()


def reference_snapshot(path: Path) -> tuple[set[str], set[str]]:
    queries: set[str] = set()
    ids: set[str] = set()
    for row in read_jsonl(path):
        queries.add(normalize_query(query_of(row)))
        try:
            ids.add(id_of(row))
        except ValueError:
            pass
    return queries, ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--compact", type=Path, default=DEFAULT_COMPACT)
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    args = parser.parse_args()

    config = read_json(args.config)
    sizes = [int(value) for value in config["data"]["sizes"]]
    seed = int(config["data"]["seed"])
    if sizes != sorted(sizes) or sizes[-1] != 54000:
        raise SystemExit("data sizes must be ascending and end at 54000")
    if set(sizes) != set(EXPECTED_CONSTRUCTION_COUNTS):
        raise SystemExit(f"unsupported data sizes: {sizes}")

    compact_rows = read_jsonl(args.compact)
    runtime_rows = read_jsonl(args.runtime)
    if len(compact_rows) != 54000 or len(runtime_rows) != 54000:
        raise SystemExit("expected exactly 54000 compact and runtime rows")

    compact_by_id = {row["trace_id"]: row for row in compact_rows}
    runtime_by_id = {row["meta"]["trace_id"]: row for row in runtime_rows}
    if len(compact_by_id) != 54000 or len(runtime_by_id) != 54000:
        raise SystemExit("duplicate trace IDs in source SFT data")
    if set(compact_by_id) != set(runtime_by_id):
        raise SystemExit("compact/runtime trace ID sets differ")
    for trace_id, compact in compact_by_id.items():
        runtime = runtime_by_id[trace_id]
        if compact["gold_tools"] != gold_from_runtime(runtime):
            raise SystemExit(f"compact/runtime target mismatch: {trace_id}")
        if compact["candidate_source"] != runtime["meta"]["construction_type"]:
            raise SystemExit(f"compact/runtime construction mismatch: {trace_id}")

    ordered_by_construction: dict[str, list[str]] = {}
    for construction, expected_count in EXPECTED_CONSTRUCTION_COUNTS[54000].items():
        ids = [
            row["trace_id"] for row in compact_rows
            if row["candidate_source"] == construction
        ]
        if len(ids) != expected_count:
            raise SystemExit(
                f"unexpected source construction count: {construction}={len(ids)}"
            )
        ordered_by_construction[construction] = sorted(
            ids, key=lambda value: (stable_order(value, seed), value)
        )

    references = {
        name: reference_snapshot(path)
        for name, path in REFERENCE_DATASETS.items()
    }
    source_order = [row["meta"]["trace_id"] for row in runtime_rows]
    selected_by_size: dict[int, set[str]] = {}
    scale_records: list[dict[str, Any]] = []
    strict_ok = True

    for size in sizes:
        quotas = EXPECTED_CONSTRUCTION_COUNTS[size]
        selected: set[str] = set()
        for construction, quota in quotas.items():
            selected.update(ordered_by_construction[construction][:quota])
        if len(selected) != size:
            raise SystemExit(f"selection size mismatch for {size}: {len(selected)}")
        selected_by_size[size] = selected

        rows = [runtime_by_id[trace_id] for trace_id in source_order if trace_id in selected]
        output = args.out_dir / f"sft_train_{size}.runtime.jsonl"
        write_jsonl(output, rows)

        construction_counts = Counter()
        tool_counts = Counter()
        tool_frequency = Counter()
        candidate_k = Counter()
        queries: set[str] = set()
        for trace_id in selected:
            compact = compact_by_id[trace_id]
            construction_counts[compact["candidate_source"]] += 1
            tool_counts[str(compact["tool_count"])] += 1
            tool_frequency.update(compact["gold_tools"])
            candidate_k[str(compact["candidate_k"])] += 1
            queries.add(normalize_query(compact["query"]))

        overlap = {}
        for name, (reference_queries, reference_ids) in references.items():
            overlap[name] = {
                "normalized_query_overlap": len(queries & reference_queries),
                "trace_id_overlap": len(selected & reference_ids),
            }
        scale_ok = (
            len(rows) == size
            and dict(construction_counts) == quotas
            and len(tool_frequency) == 100
            and len(queries) == size
            and all(
                values["normalized_query_overlap"] == 0
                and values["trace_id_overlap"] == 0
                for values in overlap.values()
            )
        )
        strict_ok = strict_ok and scale_ok
        scale_records.append(
            {
                "size": size,
                "path": relative(output),
                "sha256": sha256_file(output),
                "strict_ok": scale_ok,
                "construction_counts": dict(sorted(construction_counts.items())),
                "tool_count_distribution": dict(sorted(tool_counts.items())),
                "candidate_k_distribution": dict(sorted(candidate_k.items())),
                "unique_tools": len(tool_frequency),
                "minimum_tool_frequency": min(tool_frequency.values()),
                "maximum_tool_frequency": max(tool_frequency.values()),
                "unique_normalized_queries": len(queries),
                "reference_overlap": overlap,
            }
        )

    nesting = []
    for smaller, larger in zip(sizes, sizes[1:]):
        missing = selected_by_size[smaller] - selected_by_size[larger]
        nesting.append(
            {
                "smaller": smaller,
                "larger": larger,
                "is_subset": not missing,
                "missing_count": len(missing),
            }
        )
        strict_ok = strict_ok and not missing

    source_audit_path = ASSIGNMENT3 / "sft_data_audit.json"
    source_audit = read_json(source_audit_path)
    strict_ok = strict_ok and bool(source_audit.get("strict_ok"))
    audit = {
        "schema_version": 1,
        "task": "assignment3_sft_data_scale_ablation",
        "status": "complete" if strict_ok else "failed",
        "strict_ok": strict_ok,
        "selection_policy": {
            "seed": seed,
            "method": "每个construction内按sha256(seed:trace_id)稳定排序后取前缀",
            "nested_subsets": True,
            "source_order_preserved_in_runtime_files": True,
            "official_test_used_for_selection": False,
        },
        "source": {
            "compact": relative(args.compact),
            "compact_sha256": sha256_file(args.compact),
            "runtime": relative(args.runtime),
            "runtime_sha256": sha256_file(args.runtime),
            "source_sft_audit": relative(source_audit_path),
            "source_sft_audit_sha256": sha256_file(source_audit_path),
            "source_sft_strict_ok": bool(source_audit.get("strict_ok")),
        },
        "scales": scale_records,
        "nesting": nesting,
        "reference_snapshots": {
            name: {"path": relative(path), "sha256": sha256_file(path)}
            for name, path in REFERENCE_DATASETS.items()
        },
    }
    write_json(args.audit_out, audit)
    print(
        f"Assignment3 SFT ablation data scales={sizes} "
        f"strict_ok={strict_ok} audit={relative(args.audit_out)}"
    )
    return 0 if strict_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
