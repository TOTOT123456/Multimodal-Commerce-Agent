#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit each sampled gold/evidence pair with a stricter semantic judge."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from review_tool_trace_gt import load_jsonl, sha256_file, sha256_json, write_jsonl


SYSTEM = """你是严格的数据标签审计员。只判断 evidence 是否明确、无歧义地要求指定工具的操作。
背景商品、预算、地点、语气词不能算工具证据。像“确认这件东西”“我想知道支付”“我想知道余量”这类过于宽泛的短语，如果没有表达工具描述中的核心动作，应判 fail；边界不清则判 uncertain。只输出：
{"verdict":"pass|fail|uncertain","reason":"简短理由"}"""


def parse(raw: str) -> tuple[str, str, bool]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return "uncertain", "judge_invalid_json", False
    if (
        not isinstance(payload, dict)
        or set(payload) != {"verdict", "reason"}
        or payload.get("verdict") not in {"pass", "fail", "uncertain"}
        or not isinstance(payload.get("reason"), str)
    ):
        return "uncertain", "judge_invalid_schema", False
    return payload["verdict"], payload["reason"], True


def pair_rows(
    review: list[dict[str, Any]],
    catalog_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    pairs = []
    for row in review:
        for tool_id in row["gold_tools"]:
            tool = catalog_by_id[tool_id]
            confusable = [
                {
                    "tool_id": other_id,
                    "description": catalog_by_id[other_id]["description"],
                }
                for other_id in tool.get("confusable_tools", [])
                if other_id in catalog_by_id
            ]
            pairs.append(
                {
                    "id": f"{row['id']}::{tool_id}",
                    "trace_id": row["id"],
                    "split": row["split"],
                    "query": row["query"],
                    "tool_id": tool_id,
                    "description": tool["description"],
                    "evidence": row["tool_evidence"][tool_id],
                    "confusable_tools": confusable,
                }
            )
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=80)
    parser.add_argument("--merge-shards", nargs="*", type=Path)
    args = parser.parse_args()

    catalog = load_jsonl(args.catalog)
    catalog_by_id = {row["tool_id"]: row for row in catalog}
    authoritative = pair_rows(load_jsonl(args.input), catalog_by_id)
    if args.merge_shards is not None:
        if not args.merge_shards:
            raise SystemExit("--merge-shards requires at least one path")
        merged = []
        for path in args.merge_shards:
            merged.extend(load_jsonl(path))
        by_id = {row["id"]: row for row in merged}
        expected_ids = [row["id"] for row in authoritative]
        if len(by_id) != len(merged) or set(by_id) != set(expected_ids):
            raise SystemExit("pair judgment shards do not match the review sample")
        write_jsonl(args.output, [by_id[pair_id] for pair_id in expected_ids])
        print(f"merged pair judgments rows={len(expected_ids)} -> {args.output}")
        return 0

    if args.model is None or not args.model.exists():
        raise SystemExit("--model must point to a local model")
    if not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("invalid shard index")
    rows = [
        row
        for index, row in enumerate(authoritative)
        if index % args.shard_count == args.shard_index
    ]
    existing = load_jsonl(args.output) if args.output.exists() else []
    completed = {row["id"] for row in existing}
    if len(completed) != len(existing) or not completed <= {row["id"] for row in rows}:
        raise SystemExit("existing pair output does not match this shard")
    pending = [row for row in rows if row["id"] not in completed]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    ).eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    output_rows = list(existing)
    started = time.perf_counter()
    for start in range(0, len(pending), args.batch_size):
        chunk = pending[start : start + args.batch_size]
        prompts = []
        conversations = []
        for row in chunk:
            user = (
                f"query：{row['query']}\n"
                f"指定工具：{row['tool_id']} | {row['description']}\n"
                f"evidence：{row['evidence']}\n"
                f"易混工具：{json.dumps(row['confusable_tools'], ensure_ascii=False)}"
            )
            conversation = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
            ]
            conversations.append(conversation)
            prompts.append(conversation)
        encoded = tokenizer.apply_chat_template(
            prompts,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            padding=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)
        prompt_width = encoded["input_ids"].shape[1]
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        for row, conversation, sequence in zip(chunk, conversations, generated):
            raw = tokenizer.decode(sequence[prompt_width:], skip_special_tokens=True).strip()
            verdict, reason, valid = parse(raw)
            output_rows.append(
                {
                    "id": row["id"],
                    "trace_id": row["trace_id"],
                    "split": row["split"],
                    "tool_id": row["tool_id"],
                    "evidence": row["evidence"],
                    "verdict": verdict,
                    "reason": reason,
                    "strict_parse_ok": valid,
                    "raw_output": raw,
                    "prompt_hash": sha256_json(conversation),
                    "review_input_hash": sha256_json(row),
                    "model": str(args.model),
                    "run_config_hash": sha256_json(
                        {
                            "model": str(args.model),
                            "model_revision_inputs": [
                                sha256_file(args.model / name)
                                for name in ("config.json", "tokenizer_config.json")
                            ],
                            "input_sha256": sha256_file(args.input),
                            "catalog_sha256": sha256_file(args.catalog),
                            "system": SYSTEM,
                            "shard_count": args.shard_count,
                            "shard_index": args.shard_index,
                        }
                    ),
                }
            )
        if len(output_rows) % args.save_every < len(chunk) or len(output_rows) == len(rows):
            write_jsonl(args.output, output_rows)
            print(
                f"pair review shard={args.shard_index}/{args.shard_count} "
                f"saved={len(output_rows)}/{len(rows)}",
                flush=True,
            )
    write_jsonl(args.output, output_rows)
    print(
        f"pair review complete shard={args.shard_index}/{args.shard_count} "
        f"rows={len(output_rows)} elapsed={time.perf_counter() - started:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
