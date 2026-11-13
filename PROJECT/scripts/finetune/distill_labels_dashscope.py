#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optional DashScope teacher-label distillation for intent rows."""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import FINETUNE_DIR, INTENT_DIR


SYSTEM_PROMPT = "判断用户问题是否为电商导购/购物服务意图，并输出 JSON：{\"类型\":\"电商类|闲聊类\",\"关键词\":\"...\"}。"


def call_dashscope(api_key: str, model: str, question: str, timeout: int) -> dict:
    payload = {
        "model": model,
        "input": {"messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]},
        "parameters": {"temperature": 0.0, "result_format": "message"},
    }
    req = urllib.request.Request(
        "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    text = body["output"]["choices"][0]["message"]["content"]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INTENT_DIR / "intent_all.jsonl")
    parser.add_argument("--out", type=Path, default=FINETUNE_DIR / "data" / "teacher_labels.jsonl")
    parser.add_argument("--model", default="qwen-plus")
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--sleep", type=float, default=0.2)
    args = parser.parse_args()

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise SystemExit("DASHSCOPE_API_KEY is not set; teacher distillation is pending.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])
    with args.input.open("r", encoding="utf-8") as f, args.out.open("a", encoding="utf-8") as out:
        for line in f:
            row = json.loads(line)
            if row["id"] in done:
                continue
            for attempt in range(4):
                try:
                    label = call_dashscope(api_key, args.model, row["问题"], args.timeout)
                    break
                except (urllib.error.URLError, TimeoutError) as exc:
                    if attempt == 3:
                        label = {"error": str(exc)}
                    else:
                        time.sleep(1.5 * (attempt + 1))
            out.write(json.dumps({"id": row["id"], "问题": row["问题"], "teacher": label}, ensure_ascii=False) + "\n")
            out.flush()
            time.sleep(args.sleep)
    print(f"teacher labels -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
