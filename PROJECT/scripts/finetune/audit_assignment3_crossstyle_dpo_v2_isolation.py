#!/usr/bin/env python3
"""Post-hoc audit that DPO-v2 preferences do not overlap protected evaluations."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize(value: str) -> str:
    return re.sub(r"[\W_]+", "", value.casefold())


def preference_query(row: dict[str, Any]) -> str:
    marker = "\n用户问题：\n"
    for message in reversed(row.get("messages", [])[:-1]):
        if message.get("role") != "user":
            continue
        content = str(message.get("content", ""))
        return content.rsplit(marker, 1)[-1] if marker in content else content
    return ""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preferences", type=Path, required=True)
    parser.add_argument("--protected-eval", action="append", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    preferences = read_jsonl(args.preferences)
    train_ids = {str(row["meta"]["trace_id"]) for row in preferences}
    train_sources = {
        str(row["meta"]["source_trace_id"]) for row in preferences
    }
    train_queries = {
        normalize(preference_query(row))
        for row in preferences
    }
    eval_ids: set[str] = set()
    eval_sources: set[str] = set()
    eval_queries: set[str] = set()
    eval_rows = 0
    for path in args.protected_eval:
        for row in read_jsonl(path):
            eval_rows += 1
            eval_ids.add(str(row["id"]))
            eval_queries.add(normalize(str(row["问题"])))
            construction = row.get("construction")
            if isinstance(construction, dict) and construction.get("source_trace_id"):
                eval_sources.add(str(construction["source_trace_id"]))
    overlap = {
        "trace_id": len(train_ids & eval_ids),
        "source_trace_id": len(train_sources & eval_sources),
        "normalized_query": len(train_queries & eval_queries),
    }
    strict_ok = (
        len(preferences) == len(train_ids) == len(train_queries)
        and not any(overlap.values())
    )
    payload = {
        "audit_version": "assignment3-crossstyle-dpo-v2-posthoc-isolation-v1",
        "strict_ok": strict_ok,
        "counts": {
            "preferences": len(preferences),
            "unique_train_trace_ids": len(train_ids),
            "unique_train_source_trace_ids": len(train_sources),
            "unique_train_queries": len(train_queries),
            "protected_eval_rows": eval_rows,
            "protected_eval_source_trace_ids": len(eval_sources),
        },
        "overlap": overlap,
        "audit_timing": (
            "post-hoc only: protected query text was not read by the preference "
            "selection script; this audit ran after the preference file was frozen"
        ),
        "inputs": {
            str(args.preferences): sha256_file(args.preferences),
            **{str(path): sha256_file(path) for path in args.protected_eval},
        },
    }
    write_json(args.out, payload)
    print(f"DPO-v2 posthoc isolation overlap={overlap} strict_ok={strict_ok}")
    return 0 if strict_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
