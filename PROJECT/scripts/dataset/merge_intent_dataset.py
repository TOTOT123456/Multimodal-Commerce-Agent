#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge ecommerce and chitchat intent rows into the frozen intent set."""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import INTENT_DIR, load_jsonl, normalize_question, write_json, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ecommerce", type=Path, default=INTENT_DIR / "ecommerce.jsonl")
    parser.add_argument("--chitchat", type=Path, default=INTENT_DIR / "chitchat.jsonl")
    parser.add_argument("--out", type=Path, default=INTENT_DIR / "intent_all.jsonl")
    parser.add_argument("--stats-out", type=Path, default=INTENT_DIR / "_aux" / "intent_stats.json")
    parser.add_argument("--seed", type=int, default=20260706)
    args = parser.parse_args()

    rows = load_jsonl(args.ecommerce) + load_jsonl(args.chitchat)
    seen: set[str] = set()
    deduped: list[dict] = []
    duplicates = 0
    for row in rows:
        key = normalize_question(row["问题"])
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        deduped.append(row)
    random.Random(args.seed).shuffle(deduped)
    write_jsonl(args.out, deduped)
    stats = {
        "total": len(deduped),
        "duplicates_removed": duplicates,
        "by_type": dict(Counter(row.get("类型", "") for row in deduped)),
        "empty_keyword_ecommerce": sum(1 for row in deduped if row.get("类型") == "电商类" and not row.get("关键词")),
    }
    write_json(args.stats_out, stats)
    print(f"intent rows={len(deduped)} -> {args.out}")
    return 0 if stats["empty_keyword_ecommerce"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
