#!/usr/bin/env python3
"""Create an audited deterministic prefix of a label-free inference file."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-out", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    args = parser.parse_args()
    rows = read_jsonl(args.input)
    if not 0 < args.count <= len(rows):
        raise SystemExit("--count must be within input row count")
    selected = rows[: args.count]
    ids = [str(row["id"]) for row in selected]
    write_jsonl(args.output, selected)
    audit = {
        "audit_version": "tool-trace-label-free-prefix-v1",
        "strict_ok": (
            len(ids) == len(set(ids))
            and all(
                set(row)
                == {
                    "id",
                    "query",
                    "candidate_tools",
                    "candidate_k",
                    "ranking_hash",
                    "input_hash",
                }
                for row in selected
            )
        ),
        "selection": "deterministic prefix of an already seeded mining pool",
        "input_rows": len(rows),
        "output_rows": len(selected),
        "unique_ids": len(set(ids)),
        "candidate_k": dict(
            sorted(Counter(str(row["candidate_k"]) for row in selected).items())
        ),
        "contains_gold_fields": False,
        "input": {"path": str(args.input), "sha256": sha256_file(args.input)},
        "output": {"path": str(args.output), "sha256": sha256_file(args.output)},
    }
    args.audit_out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.audit_out.with_suffix(args.audit_out.suffix + ".tmp")
    temporary.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.audit_out)
    print(
        f"label-free subset rows={len(selected)} strict_ok={audit['strict_ok']}"
    )
    return 0 if audit["strict_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
