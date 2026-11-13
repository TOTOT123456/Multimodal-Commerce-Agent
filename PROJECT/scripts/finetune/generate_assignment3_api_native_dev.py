#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate an API-native Assignment3 dev set without query templates."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import catalog_by_id, load_catalog, load_jsonl, split_tools  # noqa: E402
from finetune.tool_trace_common import sha256_file, sha256_json  # noqa: E402


GENERATOR_PROMPT_VERSION = "assignment3-crossstyle-generator-v3"
JUDGE_PROMPT_VERSION = "assignment3-crossstyle-judge-v3"
INTERNAL_RE = re.compile(
    r"tool[_ -]?id|工具\s*id|catalog|rag|api|mcp|候选工具|调用工具|内部字段|实现逻辑",
    re.I,
)
STYLE_HINTS = [
    "像真实用户临时想到后直接发消息，口语自然但诉求清楚",
    "带一点生活场景，避免客服话术和字段罗列",
    "表达简短，允许省略主语，但不能省略任何实际诉求",
    "先交代遇到的问题，再自然提出需要处理的事项",
    "像连续对话中的一条消息，可使用指代，但对象必须能理解",
    "语气有些着急，同时保留必要条件和处理顺序",
    "语气克制、直接，不使用编号式说明书语言",
    "允许有犹豫或担心，但最终请求必须明确",
    "把多个诉求串成自然的一段话，不逐条复述能力说明",
    "使用普通消费者会说的词，不使用平台内部术语",
    "包含真实决策背景，但背景不能新增待执行任务",
    "表达可以稍长，使用自然因果和转折，不套固定句式",
]
LEGACY_STYLE_HINTS = [
    "用一小段商品或订单背景开头，随后直接提出待处理事项",
    "表达偏紧凑、任务化，但仍是普通消费者口吻",
    "多项诉求用清楚的连接词串联，不写编号或字段清单",
    "允许略偏书面的“确认、核实、说明”，但不要复述能力定义",
    "先给一个简短限制条件，再明确说出需要完成的动作",
    "像合成任务语料中的完整单句，信息密度较高但语义自然",
    "可使用“这块、相关情况、结果”等概括词，但每项动作仍需明确",
    "加入商品、地区、时间或预算中的一两项背景，不引入额外任务",
    "多步请求保持原顺序，句式尽量紧凑，不展开客服式解释",
    "单步请求只表达一个能力，可带无关执行动作的生活背景",
    "用直接陈述和请求组合成一句，不使用平台内部术语",
    "允许轻微模板感，但不能照抄 capability 或 catalog 描述",
]
STYLE_HINTS_BY_PROFILE = {
    "api_native": STYLE_HINTS,
    "legacy_compact": LEGACY_STYLE_HINTS,
}


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def api_key_from_proxy_config(path: Path) -> str:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    entries = config.get("api-keys") or []
    if not entries:
        raise SystemExit(f"proxy config contains no api-keys: {path}")
    entry = entries[0]
    key = entry if isinstance(entry, str) else entry.get("key")
    if not isinstance(key, str) or not key:
        raise SystemExit(f"unable to resolve API key from: {path}")
    return key


def normalize(text: str) -> str:
    return re.sub(r"[\W_]+", "", text.casefold())


def target_hash(target: dict[str, Any]) -> str:
    payload = {
        "id": target["id"],
        "tools": target["tools"],
        "capabilities": target["capabilities"],
    }
    if target.get("construction") is not None:
        payload["construction"] = target["construction"]
    return sha256_json(payload)


def load_existing(
    path: Path,
    expected: dict[str, dict[str, Any]],
    regenerate_stale: bool,
) -> tuple[dict[str, dict[str, Any]], list[str], list[str], int]:
    if not path.exists():
        return {}, [], [], 0
    rows = load_jsonl(path)
    accepted: dict[str, dict[str, Any]] = {}
    metadata_retargeted = []
    semantic_stale = []
    for row in rows:
        trace_id = str(row.get("id"))
        if trace_id not in expected:
            raise SystemExit(f"existing output contains unknown id: {trace_id}")
        if row.get("target_hash") != target_hash(expected[trace_id]):
            if not regenerate_stale:
                raise SystemExit(
                    f"existing output target hash mismatch: {trace_id}; "
                    "pass --regenerate-stale-existing only after auditing the plan change"
                )
            target = expected[trace_id]
            if row.get("tools") == target["tools"]:
                previous_hash = row.get("target_hash")
                row = dict(row)
                row.update(
                    {
                        "tools": target["tools"],
                        "tool_count": target["tool_count"],
                        "domain": target["domain"],
                        "combo_seen_bucket": target["combo_seen_bucket"],
                        "construction": target.get("construction"),
                        "target_hash": target_hash(target),
                        "retargeting": {
                            "reason": "unchanged_exact_ordered_tools",
                            "previous_target_hash": previous_hash,
                        },
                    }
                )
                metadata_retargeted.append(trace_id)
            else:
                semantic_stale.append(trace_id)
                continue
        if trace_id in accepted:
            raise SystemExit(f"duplicate existing output id: {trace_id}")
        accepted[trace_id] = row
    return accepted, metadata_retargeted, semantic_stale, len(rows)


class ApiClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float,
        http_retries: int,
    ) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.timeout = timeout
        self.http_retries = http_retries

    def chat(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, dict[str, Any]]:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(self.http_retries):
            request = urllib.request.Request(
                self.url,
                data=body,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    result = json.load(response)
                choice = result["choices"][0]
                content = str(choice["message"]["content"]).strip()
                meta = {
                    "response_model": result.get("model"),
                    "finish_reason": choice.get("finish_reason"),
                    "usage": result.get("usage", {}),
                }
                return content, meta
            except (
                urllib.error.HTTPError,
                urllib.error.URLError,
                TimeoutError,
                KeyError,
                IndexError,
                json.JSONDecodeError,
            ) as exc:
                last_error = exc
                time.sleep(min(8.0, 0.75 * (2**attempt)))
        raise RuntimeError(f"API request failed after retries: {last_error}")


def target_specs(
    plan_rows: list[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
    style_profile: str,
) -> list[dict[str, Any]]:
    style_hints = STYLE_HINTS_BY_PROFILE[style_profile]
    targets = []
    for plan in plan_rows:
        tools_value = plan.get("tools", plan.get("工具"))
        tools = (
            [str(value) for value in tools_value]
            if isinstance(tools_value, list)
            else split_tools(str(tools_value))
        )
        trace_id = str(plan["id"])
        if not trace_id or not tools or any(tool_id not in by_id for tool_id in tools):
            raise SystemExit(f"invalid target plan row: {trace_id!r}")
        if int(plan.get("tool_count", len(tools))) != len(tools):
            raise SystemExit(f"target plan tool_count mismatch: {trace_id}")
        capabilities = []
        for tool_id in tools:
            tool = by_id[tool_id]
            confusable = [
                by_id[value]["description"]
                for value in tool.get("confusable_tools", [])
                if value in by_id and value not in tools
            ]
            capabilities.append(
                {
                    "capability": str(tool["description"]),
                    "operation_type": str(tool["operation_type"]),
                    "confusable_boundaries": confusable,
                }
            )
        targets.append(
            {
                "id": trace_id,
                "tools": tools,
                "capabilities": capabilities,
                "tool_count": len(tools),
                "domain": str(plan.get("domain", "")),
                "combo_seen_bucket": str(plan.get("combo_seen_bucket", "")),
                "style_hint": style_hints[
                    int.from_bytes(hashlib.sha256(trace_id.encode()).digest()[:4], "big")
                    % len(style_hints)
                ],
                "style_profile": style_profile,
                "construction": plan.get("construction"),
            }
        )
    return targets


def generation_system_prompt(style_profile: str) -> str:
    if style_profile == "legacy_compact":
        profile = """
本批次采用 legacy_compact 风格：接近仓库原有的紧凑合成任务表达。可以有简短的商品、地区、时间或预算背景，随后用偏任务化但仍可理解的消费者语言明确提出诉求；多步信息可集中在一个句子中。不得为了追求旧式风格而照抄能力说明、堆砌字段或制造含义不清的病句。"""
    else:
        profile = """
本批次采用 api_native 风格：像真实用户临时向线上助手发来的自由表达，叙事、长度和句式自然多样，避免模板感和能力说明式措辞。"""
    return """你是中文电商真实用户 query 数据作者。根据每个样本给出的“目标能力序列”，从零创作一条自然、可信的用户消息。
""" + profile + """

硬性规则：
1. query 必须恰好表达目标序列中的全部诉求，不增加、删除、合并或弱化任务。
2. 多步样本必须让先后顺序清楚，但不要机械套用同一种“先、再、最后”模板；单步样本只保留一个意图。
3. 不得照抄或轻微改写 capability 说明，尤其不能原样罗列说明中的全部字段。应转换成普通消费者会说的自然表达。
4. 不得出现英文工具 ID、工具名、API、RAG、catalog、MCP、候选工具、内部字段或实现过程。
5. 自然用户可能会说“订单、退款、发票”等普通业务词，这些可以使用；但不能写成工具目录或客服说明书。
6. 可自行创造商品、地区、时间、人物关系、问题背景和合理限制，但这些背景不能引入额外待执行动作。
7. 同一批样本之间要有明显的句式、长度和叙事差异。
8. 若样本给出 additional_generation_constraints，必须同时满足；这些约束不能改变目标能力序列。

只输出严格 JSON：
{"results":[{"id":"输入id","question":"自然用户query"}]}"""


def judge_system_prompt() -> str:
    return """你是独立、严格的中文工具意图数据审计员。输入包含自然用户 query 与有序的目标能力序列。判断 query 是否唯一、完整地表达了这些能力。

逐项检查：
1. 目标对象与每个操作意图是否明确；
2. 是否遗漏、弱化、增加或混入其他可执行任务；
3. 多步意图数量和顺序是否一致；单步是否仍是单一意图；
4. 写操作不能被弱化成“问问能否操作”，读操作不能被增强成执行修改；
5. 是否像真实用户消息，而不是能力说明、字段清单或工具文档；
6. 是否出现英文工具 ID、API、RAG、catalog、MCP、候选工具或内部实现术语。
7. 若给出 additional_generation_constraints，query 是否确实满足且没有因此增加额外动作。

verdict=pass 时所有布尔值必须为 true，observed_step_count 必须等于 expected_step_count，issues 必须为空。
只输出严格 JSON：
{"results":[{"id":"输入id","verdict":"pass|fail","observed_step_count":1,"all_intents_present":true,"no_extra_intent":true,"operation_strength_preserved":true,"order_preserved":true,"natural_user_query":true,"no_internal_leak":true,"issues":[]}]}"""


def render_generation_user(
    targets: list[dict[str, Any]],
    feedback: dict[str, list[str]],
) -> str:
    samples = []
    for target in targets:
        sample = {
                "id": target["id"],
                "expected_step_count": target["tool_count"],
                "ordered_capabilities": target["capabilities"],
                "expression_preference": target["style_hint"],
                "previous_rejection_reasons": feedback.get(target["id"], []),
            }
        constraints = str(
            (target.get("construction") or {}).get("generation_constraints", "")
        ).strip()
        if constraints:
            sample["additional_generation_constraints"] = constraints
        samples.append(sample)
    return json.dumps({"samples": samples}, ensure_ascii=False, separators=(",", ":"))


def render_judge_user(
    targets: list[dict[str, Any]],
    questions: dict[str, str],
) -> str:
    samples = []
    for target in targets:
        sample = {
                "id": target["id"],
                "question": questions[target["id"]],
                "expected_step_count": target["tool_count"],
                "ordered_capabilities": target["capabilities"],
            }
        constraints = str(
            (target.get("construction") or {}).get("generation_constraints", "")
        ).strip()
        if constraints:
            sample["additional_generation_constraints"] = constraints
        samples.append(sample)
    return json.dumps({"samples": samples}, ensure_ascii=False, separators=(",", ":"))


def parse_generation(raw: str, targets: list[dict[str, Any]]) -> dict[str, str]:
    payload = json.loads(raw)
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list):
        raise ValueError("generation results is not a list")
    expected = {target["id"] for target in targets}
    parsed: dict[str, str] = {}
    for item in results:
        if not isinstance(item, dict) or set(item) != {"id", "question"}:
            raise ValueError("generation item schema mismatch")
        trace_id = str(item["id"])
        question = str(item["question"]).strip()
        if trace_id not in expected or trace_id in parsed or not question:
            raise ValueError("generation id/question mismatch")
        parsed[trace_id] = question
    if set(parsed) != expected:
        raise ValueError("generation result ids mismatch")
    return parsed


def parse_judgments(
    raw: str,
    targets: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    payload = json.loads(raw)
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list):
        raise ValueError("judge results is not a list")
    expected = {target["id"]: target for target in targets}
    required = {
        "id",
        "verdict",
        "observed_step_count",
        "all_intents_present",
        "no_extra_intent",
        "operation_strength_preserved",
        "order_preserved",
        "natural_user_query",
        "no_internal_leak",
        "issues",
    }
    parsed: dict[str, dict[str, Any]] = {}
    for item in results:
        if not isinstance(item, dict) or set(item) != required:
            raise ValueError("judge item schema mismatch")
        trace_id = str(item["id"])
        if trace_id not in expected or trace_id in parsed:
            raise ValueError("judge id mismatch")
        if not isinstance(item["issues"], list) or not all(
            isinstance(value, str) for value in item["issues"]
        ):
            raise ValueError("judge issues mismatch")
        parsed[trace_id] = item
    if set(parsed) != set(expected):
        raise ValueError("judge result ids mismatch")
    return parsed


def forbidden_phrases(catalog: list[dict[str, Any]]) -> list[str]:
    # Synthetic positive phrases are deliberately not used here: short strings
    # such as “帮我确认收货” are also unavoidable, natural user language. Only
    # literal internal IDs are hard failures; catalog-description copying is
    # checked separately below.
    return sorted(
        {str(tool["tool_id"]) for tool in catalog if tool.get("tool_id")},
        key=len,
        reverse=True,
    )


def static_issues(
    question: str,
    target: dict[str, Any],
    forbidden: list[str],
    catalog: list[dict[str, Any]],
    blocked_queries: set[str],
) -> list[str]:
    issues = []
    if len(question) < 10:
        issues.append("too_short")
    if len(question) > 700:
        issues.append("too_long")
    if INTERNAL_RE.search(question):
        issues.append("internal_term")
    if normalize(question) in blocked_queries:
        issues.append("blocked_query_overlap")
    lowered = question.casefold()
    if any(value.casefold() in lowered for value in forbidden):
        issues.append("tool_id")
    normalized_question = normalize(question)
    for tool in catalog:
        description = normalize(str(tool["description"]))
        if description and description in normalized_question:
            issues.append("catalog_description_copied")
            break
    for capability in target["capabilities"]:
        similarity = SequenceMatcher(
            None,
            normalize(str(capability["capability"])),
            normalized_question,
        ).ratio()
        if similarity > 0.82:
            issues.append("capability_too_similar")
            break
    return sorted(set(issues))


def judge_pass(item: dict[str, Any], expected_steps: int) -> bool:
    return (
        item["verdict"] == "pass"
        and item["observed_step_count"] == expected_steps
        and all(
            item[key] is True
            for key in (
                "all_intents_present",
                "no_extra_intent",
                "operation_strength_preserved",
                "order_preserved",
                "natural_user_query",
                "no_internal_leak",
            )
        )
        and not item["issues"]
    )


def process_batch(
    batch: list[dict[str, Any]],
    client: ApiClient,
    generator_model: str,
    judge_model: str,
    temperature: float,
    max_tokens: int,
    max_attempts: int,
    forbidden: list[str],
    catalog: list[dict[str, Any]],
    blocked_queries: set[str],
    style_profile: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    pending = list(batch)
    accepted: list[dict[str, Any]] = []
    attempts_log: list[dict[str, Any]] = []
    feedback: dict[str, list[str]] = {}
    for attempt in range(1, max_attempts + 1):
        if not pending:
            break
        try:
            generation_raw, generation_meta = client.chat(
                generator_model,
                generation_system_prompt(style_profile),
                render_generation_user(pending, feedback),
                temperature,
                max_tokens,
            )
            questions = parse_generation(generation_raw, pending)
        except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
            for target in pending:
                feedback[target["id"]] = [f"generation_error:{type(exc).__name__}"]
                attempts_log.append(
                    {
                        "id": target["id"],
                        "attempt": attempt,
                        "stage": "generation",
                        "accepted": False,
                        "issues": feedback[target["id"]],
                    }
                )
            continue

        clean_targets = []
        for target in pending:
            issues = static_issues(
                questions[target["id"]],
                target,
                forbidden,
                catalog,
                blocked_queries,
            )
            if issues:
                feedback[target["id"]] = issues
                attempts_log.append(
                    {
                        "id": target["id"],
                        "attempt": attempt,
                        "stage": "static",
                        "accepted": False,
                        "issues": issues,
                        "question": questions[target["id"]],
                        "generation_meta": generation_meta,
                    }
                )
            else:
                clean_targets.append(target)

        judgments: dict[str, dict[str, Any]] = {}
        if clean_targets:
            try:
                judge_raw, judge_meta = client.chat(
                    judge_model,
                    judge_system_prompt(),
                    render_judge_user(clean_targets, questions),
                    0.0,
                    max_tokens,
                )
                judgments = parse_judgments(judge_raw, clean_targets)
            except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
                judge_meta = {}
                for target in clean_targets:
                    feedback[target["id"]] = [f"judge_error:{type(exc).__name__}"]
                    attempts_log.append(
                        {
                            "id": target["id"],
                            "attempt": attempt,
                            "stage": "judge",
                            "accepted": False,
                            "issues": feedback[target["id"]],
                            "question": questions[target["id"]],
                        }
                    )

        accepted_ids = set()
        for target in clean_targets:
            judgment = judgments.get(target["id"])
            if judgment is None:
                continue
            if judge_pass(judgment, target["tool_count"]):
                accepted_ids.add(target["id"])
                accepted.append(
                    {
                        "id": target["id"],
                        "question": questions[target["id"]],
                        "tools": target["tools"],
                        "tool_count": target["tool_count"],
                        "domain": target["domain"],
                        "combo_seen_bucket": target["combo_seen_bucket"],
                        "target_hash": target_hash(target),
                        "attempt": attempt,
                        "generator_model": generator_model,
                        "judge_model": judge_model,
                        "generator_prompt_version": GENERATOR_PROMPT_VERSION,
                        "judge_prompt_version": JUDGE_PROMPT_VERSION,
                        "style_profile": style_profile,
                        "construction": target.get("construction"),
                        "generation_meta": generation_meta,
                        "judgment": judgment,
                        "judge_meta": judge_meta,
                    }
                )
                attempts_log.append(
                    {
                        "id": target["id"],
                        "attempt": attempt,
                        "stage": "judge",
                        "accepted": True,
                        "issues": [],
                    }
                )
            else:
                issues = list(judgment["issues"]) or ["judge_inconsistent"]
                feedback[target["id"]] = issues
                attempts_log.append(
                    {
                        "id": target["id"],
                        "attempt": attempt,
                        "stage": "judge",
                        "accepted": False,
                        "issues": issues,
                        "question": questions[target["id"]],
                        "judgment": judgment,
                    }
                )
        pending = [target for target in pending if target["id"] not in accepted_ids]
    return accepted, attempts_log, pending


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target-plan",
        type=Path,
        required=True,
        help="Query-free JSONL plan containing id, tools, tool_count, domain, and combo bucket.",
    )
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8317/v1")
    parser.add_argument(
        "--proxy-config",
        type=Path,
        default=Path("/root/.cli-proxy-api/config.yaml"),
    )
    parser.add_argument(
        "--api-key-file",
        type=Path,
        help="Read the OpenAI-compatible bearer token from this external file.",
    )
    parser.add_argument("--generator-model", default="gpt-5.5")
    parser.add_argument("--judge-model", default="gpt-5.4")
    parser.add_argument(
        "--style-profile",
        choices=sorted(STYLE_HINTS_BY_PROFILE),
        default="api_native",
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--http-retries", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument(
        "--dataset-role",
        choices=["dev", "train", "test"],
        default="dev",
        help="Controls the final filename, provenance, and whether training use is allowed.",
    )
    parser.add_argument(
        "--checkpoint-batches",
        type=int,
        default=10,
        help="Atomically checkpoint accepted/attempt rows every N completed batches.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N target rows; intended for isolated pilot runs.",
    )
    parser.add_argument(
        "--blocked-query-file",
        action="append",
        type=Path,
        default=[],
        help=(
            "JSONL split whose question/query strings are forbidden in generated "
            "output. Repeat for every protected train/dev/test split."
        ),
    )
    parser.add_argument(
        "--regenerate-stale-existing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Permit an audited target-plan migration. Existing rows with the "
            "same exact ordered tools are metadata-retargeted; rows whose tool "
            "sequence changed are discarded and regenerated through the API."
        ),
    )
    args = parser.parse_args()
    if min(
        args.workers,
        args.batch_size,
        args.max_attempts,
        args.http_retries,
        args.checkpoint_batches,
    ) <= 0:
        raise SystemExit("workers, batch-size, attempts, retries, and checkpoint-batches must be positive")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("limit must be positive")

    catalog = load_catalog(args.catalog)
    by_id = catalog_by_id(catalog)
    plan_rows = load_jsonl(args.target_plan)
    if args.limit is not None:
        plan_rows = plan_rows[: args.limit]
    targets = target_specs(plan_rows, by_id, args.style_profile)
    target_by_id = {target["id"]: target for target in targets}
    if len(target_by_id) != len(targets):
        raise SystemExit("duplicate generated target ids")
    forbidden = forbidden_phrases(catalog)
    blocked_queries: set[str] = set()
    blocked_query_hashes = {}
    for path in args.blocked_query_file:
        if not path.exists():
            raise SystemExit(f"missing blocked query file: {path}")
        blocked_query_hashes[str(path)] = sha256_file(path)
        for row in load_jsonl(path):
            question = row.get("问题", row.get("question", row.get("query")))
            if isinstance(question, str) and question.strip():
                blocked_queries.add(normalize(question))
    if args.api_key_file is not None:
        api_key = args.api_key_file.read_text(encoding="utf-8").strip()
        if not api_key:
            raise SystemExit("api key file is empty")
    else:
        api_key = api_key_from_proxy_config(args.proxy_config)
    client = ApiClient(args.base_url, api_key, args.timeout, args.http_retries)

    accepted_path = args.output_dir / "accepted_generation.jsonl"
    attempts_path = args.output_dir / "generation_attempts.jsonl"
    (
        accepted,
        metadata_retargeted_ids,
        semantic_stale_ids,
        preexisting_file_rows,
    ) = load_existing(
        accepted_path,
        target_by_id,
        args.regenerate_stale_existing,
    )
    preexisting_accepted_before_blocking = len(accepted)
    attempts_log = load_jsonl(attempts_path) if attempts_path.exists() else []
    attempts_log.extend(
        {
            "id": trace_id,
            "attempt": 0,
            "stage": "metadata_target_retarget",
            "accepted": True,
            "issues": [],
        }
        for trace_id in metadata_retargeted_ids
    )
    attempts_log.extend(
        {
            "id": trace_id,
            "attempt": 0,
            "stage": "semantic_target_regeneration",
            "accepted": False,
            "issues": ["ordered_tools_changed"],
        }
        for trace_id in semantic_stale_ids
    )
    blocked_existing_ids = [
        trace_id
        for trace_id, row in accepted.items()
        if normalize(str(row["question"])) in blocked_queries
    ]
    for trace_id in blocked_existing_ids:
        row = accepted.pop(trace_id)
        attempts_log.append(
            {
                "id": trace_id,
                "attempt": int(row.get("attempt", 0)),
                "stage": "blocked_existing_query",
                "accepted": False,
                "issues": ["blocked_query_overlap"],
                "question": row["question"],
            }
        )
    preexisting_accepted_count = len(accepted)
    normalized_seen = {
        normalize(str(row["question"])): trace_id for trace_id, row in accepted.items()
    }
    if len(normalized_seen) != len(accepted):
        raise SystemExit("existing accepted output contains duplicate normalized queries")
    remaining = [target for target in targets if target["id"] not in accepted]
    random.Random(args.seed).shuffle(remaining)
    batches = [
        remaining[index : index + args.batch_size]
        for index in range(0, len(remaining), args.batch_size)
    ]
    lock = threading.Lock()
    failed: list[dict[str, Any]] = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                process_batch,
                batch,
                client,
                args.generator_model,
                args.judge_model,
                args.temperature,
                args.max_tokens,
                args.max_attempts,
                forbidden,
                catalog,
                blocked_queries,
                args.style_profile,
            )
            for batch in batches
        ]
        completed_batches = 0
        for future in as_completed(futures):
            batch_accepted, batch_attempts, batch_failed = future.result()
            with lock:
                for row in batch_accepted:
                    normalized_question = normalize(str(row["question"]))
                    previous_id = normalized_seen.get(normalized_question)
                    if previous_id is not None:
                        batch_attempts.append(
                            {
                                "id": row["id"],
                                "attempt": row["attempt"],
                                "stage": "global_duplicate",
                                "accepted": False,
                                "issues": [f"duplicate_of:{previous_id}"],
                                "question": row["question"],
                            }
                        )
                        batch_failed.append(target_by_id[row["id"]])
                        continue
                    accepted[row["id"]] = row
                    normalized_seen[normalized_question] = row["id"]
                attempts_log.extend(batch_attempts)
                failed.extend(batch_failed)
                completed_batches += 1
                if (
                    completed_batches % args.checkpoint_batches == 0
                    or completed_batches == len(batches)
                ):
                    ordered_accepted = [
                        accepted[target["id"]]
                        for target in targets
                        if target["id"] in accepted
                    ]
                    write_jsonl_atomic(accepted_path, ordered_accepted)
                    write_jsonl_atomic(attempts_path, attempts_log)
                if completed_batches % 10 == 0 or completed_batches == len(batches):
                    print(
                        f"api-native generation batches={completed_batches}/{len(batches)} "
                        f"accepted={len(accepted)}/{len(targets)}",
                        flush=True,
                    )

    ordered_accepted = [
        accepted[target["id"]]
        for target in targets
        if target["id"] in accepted
    ]
    questions = [normalize(row["question"]) for row in ordered_accepted]
    duplicate_count = len(questions) - len(set(questions))
    positive_phrases = sorted(
        {
            str(phrase)
            for tool in catalog
            for phrase in tool.get("positive_phrases", [])
            if phrase
        },
        key=len,
        reverse=True,
    )
    positive_phrase_hits = Counter(
        phrase
        for row in ordered_accepted
        for phrase in positive_phrases
        if phrase.casefold() in str(row["question"]).casefold()
    )
    static_failures = {
        row["id"]: static_issues(
            row["question"],
            target_by_id[row["id"]],
            forbidden,
            catalog,
            blocked_queries,
        )
        for row in ordered_accepted
    }
    static_failures = {key: value for key, value in static_failures.items() if value}
    all_pass = (
        len(ordered_accepted) == len(targets)
        and not failed
        and not static_failures
        and duplicate_count == 0
    )

    final_rows = []
    for row in ordered_accepted:
        target = target_by_id[row["id"]]
        final_row = {
            "id": row["id"],
            "问题": row["question"],
            "工具": ",".join(target["tools"]),
            "思考": "GT由无query能力计划确定；按用户明确叙述的业务顺序选择对应工具。",
            "source": (
                f"assignment3_dpo_v2_{args.style_profile}_gpt55_"
                f"{args.dataset_role}"
            ),
            "domain": target["domain"],
            "tool_count": target["tool_count"],
            "generation_split": (
                f"assignment3_dpo_v2_{args.style_profile}_{args.dataset_role}"
            ),
            "template_family": (
                "api_native_freeform_no_query_template"
                if args.style_profile == "api_native"
                else "legacy_compact_freeform_no_query_template"
            ),
            "style": args.style_profile,
            "combo_seen_bucket": target["combo_seen_bucket"],
        }
        if target.get("construction") is not None:
            final_row["construction"] = target["construction"]
        final_rows.append(final_row)
    dataset_path = args.output_dir / f"{args.dataset_role}.jsonl"
    if all_pass:
        write_jsonl_atomic(dataset_path, final_rows)
    cumulative_special_stages = {
        stage: sorted(
            {
                str(row.get("id"))
                for row in attempts_log
                if row.get("stage") == stage
            }
        )
        for stage in (
            "metadata_target_retarget",
            "semantic_target_regeneration",
            "blocked_existing_query",
            "global_duplicate",
        )
    }
    audit = {
        "audit_version": "assignment3-api-native-generation-audit-v3",
        "strict_ok": all_pass,
        "target_plan_count": len(plan_rows),
        "target_count": len(targets),
        "accepted_count": len(ordered_accepted),
        "failed_count": len(failed),
        "duplicate_normalized_queries": duplicate_count,
        "incidental_positive_phrase_match_rows": sum(
            any(
                phrase.casefold() in str(row["question"]).casefold()
                for phrase in positive_phrases
            )
            for row in ordered_accepted
        ),
        "incidental_positive_phrase_top_matches": dict(
            positive_phrase_hits.most_common(20)
        ),
        "static_failure_count": len(static_failures),
        "static_failures": static_failures,
        "failed_ids": [target["id"] for target in failed],
        "counts": {
            "tool_count": dict(
                sorted(Counter(target["tool_count"] for target in targets).items())
            ),
            "combo_seen_bucket": dict(
                sorted(Counter(target["combo_seen_bucket"] for target in targets).items())
            ),
            "accepted_attempt": dict(
                sorted(Counter(row["attempt"] for row in ordered_accepted).items())
            ),
            "accepted_generator_model": dict(
                sorted(Counter(str(row["generator_model"]) for row in ordered_accepted).items())
            ),
            "accepted_judge_model": dict(
                sorted(Counter(str(row["judge_model"]) for row in ordered_accepted).items())
            ),
        },
        "current_run_generator_model": args.generator_model,
        "current_run_judge_model": args.judge_model,
        "generator_prompt_version": GENERATOR_PROMPT_VERSION,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "style_profile": args.style_profile,
        "dataset_role": args.dataset_role,
        "query_templates_used": False,
        "target_plan_contains_queries": False,
        "training_use_allowed": args.dataset_role == "train",
        "blocked_query_protection": {
            "files": blocked_query_hashes,
            "unique_normalized_queries": len(blocked_queries),
            "preexisting_accepted_before_blocking": preexisting_accepted_before_blocking,
            "blocked_existing_count": len(blocked_existing_ids),
            "blocked_existing_ids": blocked_existing_ids,
            "final_overlap_count": sum(
                normalize(str(row["question"])) in blocked_queries
                for row in ordered_accepted
            ),
        },
        "cumulative_migration": {
            f"{stage}_count": len(trace_ids)
            for stage, trace_ids in cumulative_special_stages.items()
        }
        | {
            f"{stage}_ids": trace_ids
            for stage, trace_ids in cumulative_special_stages.items()
        },
        "resume": {
            "preexisting_file_rows": preexisting_file_rows,
            "preexisting_accepted": preexisting_accepted_count,
            "newly_accepted": len(ordered_accepted) - preexisting_accepted_count,
            "metadata_retargeted_count": len(metadata_retargeted_ids),
            "metadata_retargeted_ids": metadata_retargeted_ids,
            "semantic_stale_regenerated_count": len(semantic_stale_ids),
            "semantic_stale_ids": semantic_stale_ids,
            "last_run_elapsed_seconds": time.perf_counter() - started,
        },
        "inputs": {
            "target_plan": str(args.target_plan),
            "target_plan_sha256": sha256_file(args.target_plan),
            "catalog": str(args.catalog),
            "catalog_sha256": sha256_file(args.catalog),
        },
        "outputs": {
            "accepted_generation": str(accepted_path),
            "accepted_generation_sha256": sha256_file(accepted_path),
            "generation_attempts": str(attempts_path),
            "generation_attempts_sha256": sha256_file(attempts_path),
            "dataset": str(dataset_path) if all_pass else None,
            "dataset_sha256": sha256_file(dataset_path) if all_pass else None,
            "dev": str(dataset_path) if all_pass and args.dataset_role == "dev" else None,
            "dev_sha256": (
                sha256_file(dataset_path)
                if all_pass and args.dataset_role == "dev"
                else None
            ),
        },
    }
    write_json_atomic(args.output_dir / "generation_audit.json", audit)
    print(
        f"api-native generation complete accepted={len(ordered_accepted)}/{len(targets)} "
        f"failed={len(failed)} duplicates={duplicate_count} strict_ok={all_pass}"
    )
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
