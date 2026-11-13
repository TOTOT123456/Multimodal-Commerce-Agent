#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Decode MS-SWIFT's actual encoded labels for SFT or DPO smoke rows."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import load_jsonl, write_json  # noqa: E402
from finetune.tool_trace_common import ASSIGNMENT3_DIR, sha256_file  # noqa: E402


def supervised_decode(tokenizer: Any, input_ids: list[int], labels: list[int]) -> tuple[str, list[int]]:
    positions = [index for index, label in enumerate(labels) if label != -100]
    tokens = [input_ids[index] for index in positions]
    return tokenizer.decode(tokens, skip_special_tokens=False), positions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sft", "dpo"], default="sft")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"missing model: {args.model}")
    rows = load_jsonl(args.data)[: args.limit]
    if not rows:
        raise SystemExit(f"empty smoke data: {args.data}")

    from swift.arguments import RLHFArguments, SftArguments

    common = {
        "model": str(args.model),
        "dataset": [str(args.data)],
        "max_length": args.max_length,
        "truncation_strategy": "delete",
        "loss_scale": "last_round",
        "add_non_thinking_prefix": False,
        "remove_unused_columns": True,
    }
    if args.mode == "sft":
        swift_args = SftArguments(**common)
    else:
        swift_args = RLHFArguments(rlhf_type="dpo", **common)
    _, processor = swift_args.get_model_processor(load_model=False)
    template = swift_args.get_template(processor)
    template.set_mode("train" if args.mode == "sft" else "rlhf")
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    details = []
    failures = []
    for index, row in enumerate(rows):
        encoded = template.encode(row)
        if args.mode == "sft":
            variants = [("assistant", encoded["input_ids"], encoded["labels"], row["messages"][-1]["content"])]
        else:
            variants = []
            for prefix, target in [
                ("chosen", row["messages"][-1]["content"]),
                ("rejected", row["rejected_response"]),
            ]:
                input_key = f"{prefix}_input_ids"
                label_key = f"{prefix}_labels"
                if input_key not in encoded or label_key not in encoded:
                    failures.append({"row": index, "reason": f"missing_{prefix}_encoded_keys", "keys": sorted(encoded)})
                    continue
                variants.append((prefix, encoded[input_key], encoded[label_key], target))
        for name, input_ids, labels, target in variants:
            decoded, positions = supervised_decode(tokenizer, input_ids, labels)
            target_preserved = target in decoded
            prompt_masked = not positions or all(label == -100 for label in labels[: positions[0]])
            think_supervised = "<think>" in decoded or "</think>" in decoded
            detail = {
                "row": index,
                "variant": name,
                "input_tokens": len(input_ids),
                "supervised_tokens": len(positions),
                "first_supervised_position": positions[0] if positions else None,
                "prompt_tokens_masked": prompt_masked,
                "target_preserved": target_preserved,
                "think_supervised": think_supervised,
                "supervised_decode": decoded,
                "target": target,
            }
            details.append(detail)
            if not prompt_masked or not target_preserved or think_supervised or len(input_ids) > args.max_length:
                failures.append(detail)

    payload = {
        "status": "complete" if not failures else "failed",
        "strict_ok": not failures,
        "mode": args.mode,
        "model": str(args.model),
        "data": str(args.data),
        "data_sha256": sha256_file(args.data),
        "max_length": args.max_length,
        "rows": len(rows),
        "variants": len(details),
        "failures": failures,
        "details": details,
        "swift_template": swift_args.template,
        "loss_scale": swift_args.loss_scale,
        "add_non_thinking_prefix": swift_args.add_non_thinking_prefix,
    }
    args.out = args.out or (
        ASSIGNMENT3_DIR / f"{args.mode}_labels_smoke_{args.model.name.lower()}.json"
    )
    write_json(args.out, payload)
    print(
        f"MS-SWIFT {args.mode} labels model={args.model.name} "
        f"variants={len(details)} strict_ok={payload['strict_ok']}"
    )
    return 0 if payload["strict_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
