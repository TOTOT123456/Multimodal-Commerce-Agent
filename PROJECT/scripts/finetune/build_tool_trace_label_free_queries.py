#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create a label-free query input from an authoritative static-trace split."""
from __future__ import annotations

import argparse
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import load_jsonl, write_json, write_jsonl  # noqa: E402
from finetune.tool_trace_common import sha256_file, sha256_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-out", type=Path, required=True)
    args = parser.parse_args()

    authority = load_jsonl(args.input)
    rows = []
    for row in authority:
        payload = {"id": str(row["id"]), "query": str(row["问题"])}
        payload["input_hash"] = sha256_json(payload)
        rows.append(payload)
    field_errors = [row["id"] for row in rows if set(row) != {"id", "query", "input_hash"}]
    unique_ids = len({row["id"] for row in rows})
    unique_queries = len({row["query"] for row in rows})
    write_jsonl(args.output, rows)
    audit = {
        "audit_version": "tool-trace-label-free-query-v1",
        "strict_ok": (
            bool(rows)
            and not field_errors
            and unique_ids == len(rows)
            and unique_queries == len(rows)
        ),
        "authority_rows": len(authority),
        "output_rows": len(rows),
        "unique_ids": unique_ids,
        "unique_queries": unique_queries,
        "field_errors": field_errors,
        "training_use_allowed": False,
        "hashes": {
            "authority": sha256_file(args.input),
            "label_free_input": sha256_file(args.output),
        },
        "generator_allowed_inputs": [str(args.output), "catalog", "model checkpoint", "prompt config"],
        "generator_forbidden_inputs": [str(args.input)],
    }
    write_json(args.audit_out, audit)
    print(f"label-free queries rows={len(rows)} strict_ok={audit['strict_ok']} -> {args.output}")
    return 0 if audit["strict_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
