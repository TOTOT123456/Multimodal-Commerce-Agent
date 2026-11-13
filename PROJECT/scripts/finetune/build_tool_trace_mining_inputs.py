#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Derive label-free train-side prompts for DPO error mining."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import load_jsonl, write_json, write_jsonl  # noqa: E402
from finetune.tool_trace_common import ASSIGNMENT3_DIR, sha256_file, sha256_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact", type=Path, default=ASSIGNMENT3_DIR / "sft_train_compact.jsonl")
    parser.add_argument("--rag-out", type=Path, default=ASSIGNMENT3_DIR / "train_rag_mining_label_free.jsonl")
    parser.add_argument("--global-out", type=Path, default=ASSIGNMENT3_DIR / "train_global_mining_label_free.jsonl")
    parser.add_argument("--manifest-out", type=Path, default=ASSIGNMENT3_DIR / "train_mining_inputs_manifest.json")
    args = parser.parse_args()

    compact = load_jsonl(args.compact)
    rag_rows = []
    global_rows = []
    for row in compact:
        if row.get("source_split") not in {"train", "assignment3_api_native_train"}:
            raise SystemExit(f"non-train compact record: {row.get('trace_id')}")
        if row["view"] == "rag":
            payload = {
                "id": row["trace_id"],
                "query": row["query"],
                "candidate_tools": row["candidate_tools"],
                "candidate_k": row["candidate_k"],
                "ranking_hash": sha256_json(
                    {
                        "source": "sft_train_compact",
                        "trace_id": row["trace_id"],
                        "candidate_tools": row["candidate_tools"],
                    }
                ),
            }
            payload["input_hash"] = sha256_json(payload)
            rag_rows.append(payload)
        elif row["view"] == "global":
            # Global mining intentionally uses the fixed catalog order used at
            # deployment; the prediction itself records that exact order.
            payload = {"id": row["trace_id"], "query": row["query"]}
            payload["input_hash"] = sha256_json(payload)
            global_rows.append(payload)
        else:
            raise SystemExit(f"unknown compact view: {row.get('view')}")
    if len({row["id"] for row in rag_rows}) != len(rag_rows):
        raise SystemExit("duplicate RAG mining ids")
    if len({row["id"] for row in global_rows}) != len(global_rows):
        raise SystemExit("duplicate global mining ids")
    write_jsonl(args.rag_out, rag_rows)
    write_jsonl(args.global_out, global_rows)
    manifest = {
        "status": "complete",
        "source": str(args.compact),
        "source_sha256": sha256_file(args.compact),
        "rag_rows": len(rag_rows),
        "global_rows": len(global_rows),
        "rag_output": str(args.rag_out),
        "rag_sha256": sha256_file(args.rag_out),
        "global_output": str(args.global_out),
        "global_sha256": sha256_file(args.global_out),
        "field_whitelist": {
            "rag": ["id", "query", "candidate_tools", "candidate_k", "ranking_hash", "input_hash"],
            "global": ["id", "query", "input_hash"],
        },
        "contains_gold_fields": False,
    }
    write_json(args.manifest_out, manifest)
    print(f"mining inputs rag={len(rag_rows)} global={len(global_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
