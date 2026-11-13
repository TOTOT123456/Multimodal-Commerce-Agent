#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create ecommerce intent rows from the frozen tool-call test set."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import (
    INTENT_DIR,
    KEYWORD_MAP_PATH,
    TOOL_TRACE_DIR,
    load_jsonl,
    read_json,
    split_tools,
    tools_to_keywords,
    write_json,
    write_jsonl,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=TOOL_TRACE_DIR / "test.jsonl")
    parser.add_argument("--keyword-map", type=Path, default=KEYWORD_MAP_PATH)
    parser.add_argument("--out", type=Path, default=INTENT_DIR / "ecommerce.jsonl")
    parser.add_argument("--stats-out", type=Path, default=INTENT_DIR / "_aux" / "ecommerce_stats.json")
    args = parser.parse_args()

    mapping = read_json(args.keyword_map, {})
    rows = []
    empty = 0
    for idx, row in enumerate(load_jsonl(args.input)):
        tools = split_tools(row.get("工具"))
        keywords = tools_to_keywords(tools, mapping)
        if not keywords:
            empty += 1
        rows.append(
            {
                "id": f"ecommerce_{idx:06d}",
                "问题": row["问题"],
                "类型": "电商类",
                "关键词": keywords,
                "工具": ", ".join(tools),
                "source": "tool_trace_test",
                "trace_id": row.get("id", ""),
            }
        )
    write_jsonl(args.out, rows)
    write_json(args.stats_out, {"total": len(rows), "empty_keyword": empty, "source": str(args.input)})
    print(f"ecommerce intent rows={len(rows)} empty_keyword={empty} -> {args.out}")
    return 0 if empty == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
