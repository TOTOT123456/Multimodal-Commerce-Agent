#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge deterministic inference shards in authoritative input order."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import load_jsonl, write_json, write_jsonl  # noqa: E402
from finetune.tool_trace_common import sha256_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--shards", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path)
    args = parser.parse_args()

    input_rows = load_jsonl(args.input)
    input_ids = [row["id"] for row in input_rows]
    rows = []
    shard_hashes = {}
    for shard in args.shards:
        shard_rows = load_jsonl(shard)
        if not shard_rows:
            raise SystemExit(f"empty shard: {shard}")
        rows.extend(shard_rows)
        shard_hashes[str(shard)] = sha256_file(shard)
    by_key = {}
    for row in rows:
        key = (row["id"], int(row.get("sample_index", 0)))
        if key in by_key:
            raise SystemExit(f"duplicate prediction key: {key}")
        by_key[key] = row
    sample_indices = sorted({sample_index for _, sample_index in by_key})
    expected = {(trace_id, sample_index) for trace_id in input_ids for sample_index in sample_indices}
    if set(by_key) != expected:
        missing = sorted(expected - set(by_key))[:10]
        extra = sorted(set(by_key) - expected)[:10]
        raise SystemExit(f"shard key mismatch missing={missing} extra={extra}")
    config_hashes = {row.get("run_config_hash") for row in rows}
    if len(config_hashes) != len(args.shards):
        # Shard index is part of run_config_hash, so one hash per shard is expected.
        raise SystemExit(f"unexpected shard run config hashes: {len(config_hashes)}")
    merged = [
        by_key[(trace_id, sample_index)]
        for trace_id in input_ids
        for sample_index in sample_indices
    ]
    write_jsonl(args.output, merged)
    manifest = {
        "status": "complete",
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "shards": shard_hashes,
        "shard_count": len(args.shards),
        "sample_indices": sample_indices,
        "expected_rows": len(expected),
        "merged_rows": len(merged),
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
    }
    manifest_out = args.manifest_out or args.output.with_suffix(args.output.suffix + ".merge.json")
    write_json(manifest_out, manifest)
    print(f"merged {len(args.shards)} shards, rows={len(merged)} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
