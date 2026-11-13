#!/usr/bin/env python3
"""Paired comparison of two scored tool-trace prediction files."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_mcnemar(improved: int, regressed: int) -> float:
    discordant = improved + regressed
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value)
        for value in range(min(improved, regressed) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--trace-field", default="trace_em")
    parser.add_argument("--set-field", default="set_em")
    parser.add_argument("--strict-field", default="strict_parse_ok")
    parser.add_argument("--tool-count-field", default="tool_count")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    baseline_rows = read_jsonl(args.baseline)
    candidate_rows = read_jsonl(args.candidate)
    baseline = {str(row["id"]): row for row in baseline_rows}
    candidate = {str(row["id"]): row for row in candidate_rows}
    if (
        len(baseline) != len(baseline_rows)
        or len(candidate) != len(candidate_rows)
        or set(baseline) != set(candidate)
    ):
        raise ValueError("paired files have duplicate or mismatched IDs")

    fields = {
        "trace_em": args.trace_field,
        "set_em": args.set_field,
        "strict_parse_ok": args.strict_field,
    }
    paired: dict[str, Any] = {}
    for label, field in fields.items():
        if any(field not in row for row in (*baseline_rows, *candidate_rows)):
            raise ValueError(f"missing paired field: {field}")
        improved = sum(
            bool(candidate[row_id][field]) and not bool(baseline[row_id][field])
            for row_id in baseline
        )
        regressed = sum(
            bool(baseline[row_id][field]) and not bool(candidate[row_id][field])
            for row_id in baseline
        )
        paired[label] = {
            "improved": improved,
            "regressed": regressed,
            "net": improved - regressed,
            "discordant": improved + regressed,
            "exact_mcnemar_two_sided_p": exact_mcnemar(improved, regressed),
        }

    by_tool_count: dict[str, Counter[str]] = defaultdict(Counter)
    for row_id, left in baseline.items():
        right = candidate[row_id]
        bucket_value = left.get(args.tool_count_field)
        if bucket_value is None:
            bucket_value = (
                len(left.get("gold_tools", []))
                if left.get("gold_route") == "电商类"
                else "chitchat"
            )
        bucket = str(bucket_value)
        if bool(right[args.trace_field]) and not bool(left[args.trace_field]):
            by_tool_count[bucket]["improved"] += 1
        elif bool(left[args.trace_field]) and not bool(right[args.trace_field]):
            by_tool_count[bucket]["regressed"] += 1

    payload = {
        "schema_version": 1,
        "status": "complete",
        "count": len(baseline),
        "baseline_name": args.baseline_name,
        "candidate_name": args.candidate_name,
        "field_mapping": fields,
        "paired": paired,
        "trace_by_tool_count": {
            key: {
                "improved": counts["improved"],
                "regressed": counts["regressed"],
                "net": counts["improved"] - counts["regressed"],
            }
            for key, counts in sorted(by_tool_count.items())
        },
        "inputs": {
            str(args.baseline): sha256_file(args.baseline),
            str(args.candidate): sha256_file(args.candidate),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.out)
    trace = paired["trace_em"]
    print(
        f"paired TraceEM improved={trace['improved']} "
        f"regressed={trace['regressed']} p={trace['exact_mcnemar_two_sided_p']:.6g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
