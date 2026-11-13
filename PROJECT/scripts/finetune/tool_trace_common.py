#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared, deterministic helpers for assignment-3 tool-trace experiments."""
from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable


PROMPT_VERSION = "tool-trace-v1"
QUERY_ONLY_PROMPT_VERSION = "tool-trace-query-only-v1"
SYSTEM_PROMPT = (
    "你是电商导购 Agent 的工具轨迹规划器。根据用户问题，从给定工具中选择用户明确需要的全部工具，"
    "并按用户叙述及合理业务执行顺序排列。\n\n"
    "规则：\n"
    "1. 只能使用给定列表中的工具 ID。\n"
    "2. 不要自行补充用户没有要求的工具或前置工具。\n"
    "3. 不要遗漏用户明确提出的任何子诉求。\n"
    "4. 只输出合法 JSON，不输出解释、Markdown 或思考过程。\n"
    '5. 输出格式必须为：{"tools":["tool_a","tool_b"]}。'
)
QUERY_ONLY_SYSTEM_PROMPT = (
    "你是电商导购 Agent 的工具轨迹规划器。仅根据用户问题，从本任务约定的工具 ID 空间中"
    "选择用户明确需要的全部工具，并按用户叙述及合理业务执行顺序排列。\n\n"
    "规则：\n"
    "1. 不提供工具目录；只能输出你能从任务知识中确定的工具 ID。\n"
    "2. 不要自行补充用户没有要求的工具或前置工具。\n"
    "3. 不要遗漏用户明确提出的任何子诉求。\n"
    "4. 只输出合法 JSON，不输出解释、Markdown 或思考过程。\n"
    '5. 输出格式必须为：{"tools":["tool_a","tool_b"]}。'
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
ASSIGNMENT3_DIR = PROJECT_ROOT / "data" / "tool_call_trace" / "_aux" / "assignment3"
TOOLTRACE_PREDS_DIR = PROJECT_ROOT / "finetune" / "outputs" / "preds" / "tool_trace"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def deterministic_shuffle(values: Iterable[str], *seed_parts: object) -> list[str]:
    result = list(values)
    random.Random(stable_seed(*seed_parts)).shuffle(result)
    return result


def tool_line(tool: dict[str, Any]) -> str:
    return " | ".join(
        [
            str(tool["tool_id"]),
            str(tool["domain"]),
            str(tool["operation_type"]),
            str(tool["description"]).strip(),
        ]
    )


def render_user_prompt(
    query: str,
    candidate_tools: list[str],
    catalog_by_id: dict[str, dict[str, Any]],
    view: str,
) -> str:
    if view not in {"query_only", "global", "rag"}:
        raise ValueError(f"unknown view: {view}")
    if view == "query_only":
        if candidate_tools:
            raise ValueError("query-only view must not receive candidate tools")
        return f"用户问题：\n{query}"
    unknown = [tool_id for tool_id in candidate_tools if tool_id not in catalog_by_id]
    if unknown:
        raise ValueError(f"unknown candidate tools: {unknown[:5]}")
    title = "可用工具" if view == "global" else "候选工具"
    lines = "\n".join(tool_line(catalog_by_id[tool_id]) for tool_id in candidate_tools)
    return f"{title}：\n{lines}\n\n用户问题：\n{query}"


def render_messages(
    query: str,
    candidate_tools: list[str],
    catalog_by_id: dict[str, dict[str, Any]],
    view: str,
    assistant: str | None = None,
) -> list[dict[str, str]]:
    system_prompt = QUERY_ONLY_SYSTEM_PROMPT if view == "query_only" else SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": render_user_prompt(query, candidate_tools, catalog_by_id, view),
        },
    ]
    if assistant is not None:
        messages.append({"role": "assistant", "content": assistant})
    return messages


def canonical_target(tools: Iterable[str]) -> str:
    return json.dumps({"tools": list(tools)}, ensure_ascii=False, separators=(",", ":"))


def prompt_version_for_view(view: str) -> str:
    return QUERY_ONLY_PROMPT_VERSION if view == "query_only" else PROMPT_VERSION


def prompt_hash(
    messages: list[dict[str, str]],
    prompt_version: str = PROMPT_VERSION,
) -> str:
    return sha256_json({"prompt_version": prompt_version, "messages": messages})


def strict_parse_tools(
    text: str,
    catalog_ids: set[str],
    candidate_ids: set[str] | None = None,
) -> tuple[list[str], bool, str]:
    """Parse the exact task schema without trimming explanatory text."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return [], False, "invalid_json"
    if not isinstance(payload, dict) or set(payload) != {"tools"}:
        return [], False, "schema_keys"
    tools = payload["tools"]
    if not isinstance(tools, list):
        return [], False, "tools_not_array"
    if any(not isinstance(tool_id, str) for tool_id in tools):
        return [], False, "tool_not_string"
    if len(tools) != len(set(tools)):
        return tools, False, "duplicate_tool"
    if any(tool_id not in catalog_ids for tool_id in tools):
        return tools, False, "unknown_tool"
    if candidate_ids is not None and any(tool_id not in candidate_ids for tool_id in tools):
        return tools, False, "candidate_escape"
    return tools, True, ""


def recover_parse_tools(text: str) -> tuple[list[str], bool]:
    candidates = [text]
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
    if fence:
        candidates.append(fence.group(1))
    candidates.extend(match.group(0) for match in re.finditer(r"\{.*?\}", text, flags=re.S))
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("tools"), list):
            tools = payload["tools"]
            if all(isinstance(tool_id, str) for tool_id in tools):
                return tools, True
    return [], False


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * pct / 100.0
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    weight = position - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def summary_stats(values: Iterable[int | float]) -> dict[str, float | int]:
    rows = [float(value) for value in values]
    if not rows:
        return {"count": 0, "min": 0, "p50": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0, "mean": 0}
    return {
        "count": len(rows),
        "min": min(rows),
        "p50": percentile(rows, 50),
        "p90": percentile(rows, 90),
        "p95": percentile(rows, 95),
        "p99": percentile(rows, 99),
        "max": max(rows),
        "mean": sum(rows) / len(rows),
    }
