#!/usr/bin/env python3
"""Rejudge accepted generated queries with one fixed independent judge model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import catalog_by_id, load_catalog, load_jsonl  # noqa: E402
from finetune.generate_assignment3_api_native_dev import (  # noqa: E402
    ApiClient,
    api_key_from_proxy_config,
    judge_pass,
    judge_system_prompt,
    parse_judgments,
    render_judge_user,
    target_specs,
    write_jsonl_atomic,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--target-plan", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--style-profile", required=True)
    parser.add_argument("--judge-model", default="gpt-5.4")
    parser.add_argument("--base-url", default="http://127.0.0.1:8317/v1")
    parser.add_argument(
        "--proxy-config",
        type=Path,
        default=Path("/root/.cli-proxy-api/config.yaml"),
    )
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--drop-rejected",
        action="store_true",
        help="Remove rows rejected by the independent judge so they can be regenerated.",
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")

    catalog = load_catalog(args.catalog)
    plans = load_jsonl(args.target_plan)
    targets = target_specs(plans, catalog_by_id(catalog), args.style_profile)
    target_by_id = {str(row["id"]): row for row in targets}
    rows = load_jsonl(args.accepted)
    selected = [row for row in rows if row.get("judge_model") != args.judge_model]
    if not selected:
        print("all accepted rows already use the requested judge")
        return 0

    if args.api_key_file:
        api_key = args.api_key_file.read_text(encoding="utf-8").strip()
    else:
        api_key = api_key_from_proxy_config(args.proxy_config)
    client = ApiClient(args.base_url, api_key, args.timeout, 5)
    row_by_id = {str(row["id"]): row for row in rows}
    rejected: list[str] = []
    for start in range(0, len(selected), args.batch_size):
        batch_rows = selected[start : start + args.batch_size]
        batch_targets = [target_by_id[str(row["id"])] for row in batch_rows]
        questions = {str(row["id"]): str(row["question"]) for row in batch_rows}
        raw, meta = client.chat(
            args.judge_model,
            judge_system_prompt(),
            render_judge_user(batch_targets, questions),
            0.0,
            4096,
        )
        judgments = parse_judgments(raw, batch_targets)
        for target in batch_targets:
            trace_id = str(target["id"])
            judgment = judgments[trace_id]
            if not judge_pass(judgment, int(target["tool_count"])):
                rejected.append(trace_id)
                continue
            row = row_by_id[trace_id]
            row.setdefault("prior_judgments", []).append(
                {
                    "judge_model": row.get("judge_model"),
                    "judgment": row.get("judgment"),
                    "judge_meta": row.get("judge_meta"),
                }
            )
            row["judge_model"] = args.judge_model
            row["judgment"] = judgment
            row["judge_meta"] = meta

    if rejected and not args.drop_rejected:
        raise ValueError(
            "independent rejudge rejected rows: "
            + json.dumps(rejected, ensure_ascii=False)
        )
    if rejected:
        rows = [row for row in rows if str(row["id"]) not in set(rejected)]
    write_jsonl_atomic(args.accepted, rows)
    print(
        f"rejudged {len(selected)} rows with {args.judge_model}; "
        f"dropped={len(rejected)}: {args.accepted}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
