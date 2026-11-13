#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Select a checkpoint only from the protected API-native development split."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import read_json, write_json  # noqa: E402
from finetune.tool_trace_common import sha256_file  # noqa: E402


def parse_candidate(value: str) -> tuple[str, Path, Path]:
    parts = value.split("=", 2)
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError("--candidate must be NAME=ADAPTER=METRICS_JSON")
    return parts[0], Path(parts[1]), Path(parts[2])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", type=parse_candidate, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=2400)
    args = parser.parse_args()

    rows = []
    gold_hashes = set()
    for name, adapter, metrics_path in args.candidate:
        if not adapter.is_dir() or not metrics_path.is_file():
            raise SystemExit(f"missing candidate artifact: {name}")
        metrics = read_json(metrics_path)
        if metrics.get("status") != "complete" or metrics.get("count") != args.expected_count:
            raise SystemExit(f"incomplete dev metrics: {name}")
        if metrics.get("view") != "rag":
            raise SystemExit(f"checkpoint selection must use the fixed RAG dev view: {name}")
        gold_hashes.add(str(metrics.get("gold_snapshot_sha256", "")))
        overall = metrics["overall"]
        rows.append(
            {
                "name": name,
                "adapter": str(adapter),
                "adapter_sha256": sha256_file(adapter / "adapter_model.safetensors"),
                "metrics": str(metrics_path),
                "metrics_sha256": sha256_file(metrics_path),
                "count": metrics["count"],
                "gold_snapshot_sha256": metrics["gold_snapshot_sha256"],
                "trace_em": float(overall["trace_em"]),
                "set_em": float(overall["set_em"]),
                "strict_output_valid_rate": float(overall["strict_output_valid_rate"]),
                "keyword_micro_f1": float(overall["keyword_micro_f1"]),
            }
        )
    if len(gold_hashes) != 1 or "" in gold_hashes:
        raise SystemExit("candidate metrics do not share one immutable dev snapshot")
    ranked = sorted(
        rows,
        key=lambda row: (
            -row["trace_em"],
            -row["set_em"],
            -row["strict_output_valid_rate"],
            -row["keyword_micro_f1"],
            row["name"],
        ),
    )
    output = {
        "status": "complete",
        "selection_split": "assignment3_api_native_dev_2400",
        "official_test_used": False,
        "ranking": [
            "trace_em desc",
            "set_em desc",
            "strict_output_valid_rate desc",
            "keyword_micro_f1 desc",
            "name asc",
        ],
        "selected": ranked[0],
        "ranked_candidates": ranked,
    }
    write_json(args.out, output)
    print(f"selected={ranked[0]['name']} trace_em={ranked[0]['trace_em']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
