#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export tool_id -> keyword_label map from the unified catalog."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import CATALOG_PATH, KEYWORD_MAP_PATH, load_catalog, tool_keyword_map, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    parser.add_argument("--out", type=Path, default=KEYWORD_MAP_PATH)
    args = parser.parse_args()

    catalog = load_catalog(args.catalog)
    mapping = tool_keyword_map(catalog)
    write_json(args.out, mapping)
    print(f"exported {len(mapping)} keyword labels -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
