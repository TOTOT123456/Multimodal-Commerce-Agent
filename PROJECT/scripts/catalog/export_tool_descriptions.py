#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export catalog-derived tool retrieval documents."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import CATALOG_PATH, TOOL_CORPUS_PATH, compact_tool_doc, load_catalog, write_jsonl


def build_rows(catalog: list[dict]) -> list[dict]:
    rows = []
    for tool in catalog:
        rows.append(
            {
                "tool_id": tool["tool_id"],
                "server": tool["server"],
                "domain": tool["domain"],
                "name_zh": tool["name_zh"],
                "keyword_label": tool["keyword_label"],
                "description": tool["description"],
                "enhanced_desc": tool["enhanced_desc"],
                "positive_phrases": tool["positive_phrases"],
                "negative_boundary": tool["negative_boundary"],
                "confusable_tools": tool["confusable_tools"],
                "depends_on": tool["depends_on"],
                "document": compact_tool_doc(tool),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    parser.add_argument("--out", type=Path, default=TOOL_CORPUS_PATH)
    args = parser.parse_args()

    rows = build_rows(load_catalog(args.catalog))
    write_jsonl(args.out, rows)
    print(f"exported {len(rows)} tool docs -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
