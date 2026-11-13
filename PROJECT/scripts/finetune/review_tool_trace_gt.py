#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run a reproducible semantic judge over the stratified GT review sample."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM = """你是电商工具轨迹数据审计员。请逐个检查给定 query 的 gold_tools 是否：
1. 每个 gold 工具都被 query 明确要求，且 evidence 与工具描述语义一致；
2. query 中每个明确的、应由工具完成的子诉求都已被 gold 覆盖；
3. 工具数量和顺序没有明显错误。
自然语言不必写出工具名或全部参数：“我想知道品牌”足以支持商品基础信息工具，
“看看可用优惠券”足以支持优惠券查询。只有 evidence 没有可区分的核心语义或指向错误操作时才判错，
例如“确认这件东西”不能支持运费查询，“我想知道余量”不能支持拆单方案，
“我想知道支付”不能单独支持账户风险检查。不要因为背景商品、地点、预算或语气词添加工具。
务必检查多工具样本中的每一项，不要因整体格式完整而直接通过。只输出一个 JSON：
{"verdict":"pass|fail|uncertain","issues":["简短问题"]}
没有问题时 issues 必须为空数组。"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    blob = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def strict_judgment(raw: str) -> tuple[str, list[str], bool]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return "uncertain", ["judge_invalid_json"], False
    if (
        not isinstance(payload, dict)
        or set(payload) != {"verdict", "issues"}
        or payload.get("verdict") not in {"pass", "fail", "uncertain"}
        or not isinstance(payload.get("issues"), list)
        or not all(isinstance(item, str) for item in payload["issues"])
    ):
        return "uncertain", ["judge_invalid_schema"], False
    if payload["verdict"] == "pass" and payload["issues"]:
        return "uncertain", ["judge_pass_with_issues"], False
    return payload["verdict"], payload["issues"], True


def user_prompt(
    row: dict[str, Any],
    catalog_lines: str,
    catalog_by_id: dict[str, dict[str, Any]],
) -> str:
    gold = []
    for tool_id in row["gold_tools"]:
        tool = catalog_by_id[tool_id]
        gold.append(
            {
                "tool_id": tool_id,
                "description": tool["description"],
                "evidence": row["tool_evidence"][tool_id],
            }
        )
    return (
        f"query：{row['query']}\n"
        f"gold 与 evidence：{json.dumps(gold, ensure_ascii=False)}\n\n"
        "可用工具全集（用于检查明显遗漏）：\n"
        f"{catalog_lines}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=40)
    args = parser.parse_args()

    if not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("invalid shard index")
    catalog = load_jsonl(args.catalog)
    catalog_by_id = {row["tool_id"]: row for row in catalog}
    catalog_lines = "\n".join(
        f"{row['tool_id']} | {row['description']}" for row in catalog
    )
    authoritative = load_jsonl(args.input)
    rows = [
        row
        for index, row in enumerate(authoritative)
        if index % args.shard_count == args.shard_index
    ]
    existing = load_jsonl(args.output) if args.output.exists() else []
    completed = {row["id"] for row in existing}
    pending = [row for row in rows if row["id"] not in completed]
    if len(completed) != len(existing) or not completed <= {row["id"] for row in rows}:
        raise SystemExit("existing review output is not aligned with this shard")

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
        conversations = [
            [
                {"role": "system", "content": SYSTEM},
                {
                    "role": "user",
                    "content": user_prompt(row, catalog_lines, catalog_by_id),
                },
            ]
            for row in chunk
        ]
        encoded = tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            padding=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)
        input_lengths = encoded["attention_mask"].sum(dim=1).tolist()
        tick = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        elapsed_ms = (time.perf_counter() - tick) * 1000.0
        prompt_width = encoded["input_ids"].shape[1]
        for row, sequence, input_length in zip(chunk, generated, input_lengths):
            raw = tokenizer.decode(
                sequence[prompt_width:],
                skip_special_tokens=True,
            ).strip()
            verdict, issues, valid = strict_judgment(raw)
            prompt = [
                {"role": "system", "content": SYSTEM},
                {
                    "role": "user",
                    "content": user_prompt(row, catalog_lines, catalog_by_id),
                },
            ]
            output_rows.append(
                {
                    "id": row["id"],
                    "split": row["split"],
                    "verdict": verdict,
                    "issues": issues,
                    "strict_parse_ok": valid,
                    "raw_output": raw,
                    "model": str(args.model),
                    "input_tokens": int(input_length),
                    "output_tokens": int(sequence.shape[0] - prompt_width),
                    "batch_latency_ms": elapsed_ms,
                    "prompt_hash": sha256_json(prompt),
                    "review_input_hash": sha256_json(
                        {
                            "id": row["id"],
                            "query": row["query"],
                            "gold_tools": row["gold_tools"],
                            "tool_evidence": row["tool_evidence"],
                        }
                    ),
                    "sample_index": 0,
                    "run_config_hash": sha256_json(
                        {
                            "model": str(args.model),
                            "input_sha256": sha256_file(args.input),
                            "catalog_sha256": sha256_file(args.catalog),
                            "system": SYSTEM,
                            "shard_count": args.shard_count,
                            "shard_index": args.shard_index,
                            "max_new_tokens": args.max_new_tokens,
                        }
                    ),
                }
            )
        if len(output_rows) % args.save_every < len(chunk) or len(output_rows) == len(rows):
            write_jsonl(args.output, output_rows)
            print(
                f"semantic review shard={args.shard_index}/{args.shard_count} "
                f"saved={len(output_rows)}/{len(rows)}",
                flush=True,
            )
    write_jsonl(args.output, output_rows)
    print(
        f"semantic review complete shard={args.shard_index}/{args.shard_count} "
        f"rows={len(output_rows)} elapsed={time.perf_counter() - started:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
