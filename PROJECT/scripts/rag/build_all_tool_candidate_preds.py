#!/usr/bin/env python3
"""Build a deterministic all-100-tool candidate file for source-independent reranker scoring."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import load_jsonl, split_tools  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--docs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    dataset = load_jsonl(args.dataset)
    tools = sorted(str(row["tool_id"]) for row in load_jsonl(args.docs))
    if len(tools) != 100 or len(set(tools)) != 100:
        raise ValueError(f"expected 100 unique tools, found {len(set(tools))}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(dataset):
            payload = {
                "id": row.get("id", index),
                "query": row["问题"],
                "gold_tools": split_tools(row["工具"]),
                "pred_tools": tools,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    print(f"all-tool candidates rows={len(dataset)} tools={len(tools)} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
