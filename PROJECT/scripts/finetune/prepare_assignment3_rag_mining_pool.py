#!/usr/bin/env python3
"""Align gold rows with label-free RAG inputs and keep only AllHit@10 mining rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def split_tools(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(value) for value in raw]
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--rag-label-free", type=Path, required=True)
    parser.add_argument("--compact-out", type=Path, required=True)
    parser.add_argument("--rag-eligible-out", type=Path, required=True)
    parser.add_argument("--audit-out", type=Path, required=True)
    args = parser.parse_args()

    gold = read_jsonl(args.gold)
    rag = read_jsonl(args.rag_label_free)
    rag_by_id = {str(row["id"]): row for row in rag}
    if len(rag_by_id) != len(rag) or set(rag_by_id) != {str(row["id"]) for row in gold}:
        raise ValueError("gold and RAG label-free ID sets differ or contain duplicates")

    compact: list[dict[str, Any]] = []
    eligible_rag: list[dict[str, Any]] = []
    misses: Counter[str] = Counter()
    for source in gold:
        trace_id = str(source["id"])
        rag_row = rag_by_id[trace_id]
        query = str(source.get("问题") or source.get("query") or "")
        if query != rag_row["query"]:
            raise ValueError(f"query mismatch: {trace_id}")
        tools = split_tools(source["工具"])
        candidates = [str(tool) for tool in rag_row["candidate_tools"]]
        if not set(tools) <= set(candidates):
            misses[str(source["tool_count"])] += 1
            continue
        construction = dict(source.get("construction") or {})
        compact.append(
            {
                "trace_id": trace_id,
                "query": query,
                "gold_tools": tools,
                "candidate_tools": candidates,
                "tool_count": int(source["tool_count"]),
                "source_split": str(source["source_split"]),
                "construction": construction,
                "domain": source.get("domain"),
                "combo_seen_bucket": source.get("combo_seen_bucket"),
                "record_hash": sha256_json(source),
            }
        )
        eligible_rag.append(rag_row)

    represented_tools = {tool for row in compact for tool in row["gold_tools"]}
    checks = {
        "source_rows_complete": len(gold) == len(rag),
        "eligible_alignment": len(compact) == len(eligible_rag),
        "all_100_tools_represented": len(represented_tools) == 100,
        "all_compact_gold_in_candidates": all(
            set(row["gold_tools"]) <= set(row["candidate_tools"]) for row in compact
        ),
    }
    write_jsonl(args.compact_out, compact)
    write_jsonl(args.rag_eligible_out, eligible_rag)
    audit = {
        "schema_version": 1,
        "status": "complete" if all(checks.values()) else "failed",
        "strict_ok": all(checks.values()),
        "counts": {
            "source_rows": len(gold),
            "eligible_rows": len(compact),
            "retrieval_misses": len(gold) - len(compact),
            "all_hit_at_10": len(compact) / len(gold),
            "misses_by_tool_count": dict(sorted(misses.items())),
            "eligible_by_tool_count": dict(
                sorted(Counter(str(row["tool_count"]) for row in compact).items())
            ),
            "represented_tools": len(represented_tools),
        },
        "checks": checks,
        "outputs": {
            "compact": str(args.compact_out),
            "rag_eligible": str(args.rag_eligible_out),
        },
    }
    args.audit_out.parent.mkdir(parents=True, exist_ok=True)
    args.audit_out.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"RAG mining pool eligible={len(compact)}/{len(gold)} "
        f"AllHit@10={len(compact) / len(gold):.6f} strict_ok={audit['strict_ok']}"
    )
    return 0 if audit["strict_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
