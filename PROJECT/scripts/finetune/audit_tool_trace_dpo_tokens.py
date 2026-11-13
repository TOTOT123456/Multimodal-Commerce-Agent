#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Full tokenizer length audit for chosen/rejected DPO sequences."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import PROJECT_ROOT, load_jsonl, write_json  # noqa: E402
from finetune.tool_trace_common import ASSIGNMENT3_DIR, sha256_file, summary_stats  # noqa: E402


def apply_template(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": False,
        "enable_thinking": False,
    }
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking")
        return tokenizer.apply_chat_template(messages, **kwargs)


def audit_model(model: Path, rows: list[dict[str, Any]], max_length: int) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    chosen_lengths = []
    rejected_lengths = []
    chosen_target_lengths = []
    rejected_target_lengths = []
    for row in rows:
        chosen_messages = row["messages"]
        rejected_messages = row["messages"][:-1] + [
            {"role": "assistant", "content": row["rejected_response"]}
        ]
        chosen_lengths.append(len(apply_template(tokenizer, chosen_messages)))
        rejected_lengths.append(len(apply_template(tokenizer, rejected_messages)))
        chosen_target_lengths.append(
            len(tokenizer.encode(row["messages"][-1]["content"], add_special_tokens=False))
        )
        rejected_target_lengths.append(
            len(tokenizer.encode(row["rejected_response"], add_special_tokens=False))
        )
    return {
        "model": str(model),
        "max_length": max_length,
        "chosen_sequence_tokens": summary_stats(chosen_lengths),
        "rejected_sequence_tokens": summary_stats(rejected_lengths),
        "chosen_target_tokens": summary_stats(chosen_target_lengths),
        "rejected_target_tokens": summary_stats(rejected_target_lengths),
        "chosen_over_max": sum(length > max_length for length in chosen_lengths),
        "rejected_over_max": sum(length > max_length for length in rejected_lengths),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--model-4b",
        type=Path,
        default=PROJECT_ROOT / "models" / "Qwen" / "Qwen3-4B",
    )
    parser.add_argument(
        "--model-8b",
        type=Path,
        default=PROJECT_ROOT / "models" / "Qwen" / "Qwen3-8B",
    )
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument(
        "--out",
        type=Path,
        default=ASSIGNMENT3_DIR / "dpo_token_length_audit.json",
    )
    args = parser.parse_args()
    rows = load_jsonl(args.data)
    if not rows:
        raise SystemExit("empty DPO data")
    models = {
        "qwen3_4b": audit_model(args.model_4b, rows, args.max_length),
        "qwen3_8b": audit_model(args.model_8b, rows, args.max_length),
    }
    strict_ok = all(
        payload["chosen_over_max"] == 0 and payload["rejected_over_max"] == 0
        for payload in models.values()
    )
    output = {
        "status": "complete" if strict_ok else "failed",
        "strict_ok": strict_ok,
        "data": str(args.data),
        "data_sha256": sha256_file(args.data),
        "rows": len(rows),
        "models": models,
    }
    write_json(args.out, output)
    print(f"DPO token audit rows={len(rows)} strict_ok={strict_ok} -> {args.out}")
    return 0 if strict_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
