#!/usr/bin/env python3
"""生成短会话中的自然承接与自然新动作，并由独立模型逐条审核。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from PROJECT.scripts.finetune.generate_assignment3_api_native_dev import (  # noqa: E402
    ApiClient,
)
from PROJECT.scripts.project_common import load_jsonl, write_json, write_jsonl  # noqa: E402


GENERATOR_PROMPT_VERSION = "natural-short-memory-target-generator"
JUDGE_PROMPT_VERSION = "natural-short-memory-target-judge"
INTERNAL_RE = re.compile(
    r"tool[_ -]?id|工具\s*id|catalog|rag|api|mcp|候选工具|评测|测试集|模型",
    re.I,
)
NUMBERED_TURN_RE = re.compile(r"第\s*(?:\d+|[一二三四五六七八九十]+)\s*轮")
EVALUATION_LANGUAGE_RE = re.compile(
    r"轮次|原样|整套需求|全部动作|所有操作|历史任务|旧任务|来源任务|"
    r"完整.{0,8}(?:重做|再做|再办|执行)|步骤.{0,8}顺序|"
    r"顺序.{0,8}(?:不变|别变)|调用.{0,8}(?:历史|之前|前面)"
)
ACTION_LIST_RE = re.compile(r"先.+(?:再|然后).+(?:再|然后|最后)|；.+；")
SOURCE_DETAIL_LIST_RE = re.compile(
    r"(?:[^，。]{2,18}[、，]){2,}|"
    r"[^，。]{2,18}、[^，。]{2,18}(?:和|以及|还有)[^，。]{2,18}(?:这些|那些|等)"
)
HISTORY_SIGNAL_RE = re.compile(
    r"刚才|刚刚|刚买|前面|之前|上次|那个|这个|那件|那单|那款|那双|那批|"
    r"这件|这单|这事|这款|这双|这批|照这个|按这个|就这么|继续|接着|还是|"
    r"那也|那顺便|顺便"
)
NATURAL_COLLECTIVE_SCOPE_RE = re.compile(
    r"(?:刚才|前面)(?:说|提|聊|谈)(?:过)?的?"
    r"(?:那些|这几件|那几件|几件|几个|这几项|那几项|几项)"
    r"(?:事|事情|需求|问题|安排)?|"
    r"(?:那些|这几件|那几件|这几项|那几项)(?:事|事情|需求|问题|安排)?|"
    r"(?:刚才|前面)(?:说|提|聊|谈)(?:过)?的?.{0,16}(?:都|一起)"
)
NATURAL_SINGLE_SCOPE_RE = re.compile(
    r"(?:刚才|刚刚|前面|之前).{0,28}(?:的事|这事|那事|这个问题|那个问题)"
)
CLEAR_IMPLICIT_RE = re.compile(
    r"^(?:嗯|好|好的|行|可以|没问题|成)[，,、\s]*[^，。；]{0,24}"
    r"(?:(?:按|照)(?:这个|这么|刚才说的|前面说的)|就这么)"
)


def normalize(value: str) -> str:
    return re.sub(r"[\W_]+", "", value.casefold())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    body = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def api_key(path: Path) -> str:
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise SystemExit(f"empty API key file: {path}")
    return value


def system_prompt(target_kind: str) -> str:
    shared = """你是中文电商产品的真实对话数据作者。给你一段很短的用户请求历史，写下一条普通消费者会自然发送的消息。

硬性要求：
1. 不得说“第N轮”、轮次、历史任务、来源任务、调用、步骤顺序、完整重做、原样再来等评测或实现语言。
2. 像手机聊天中的一条消息，口语化，不写客服话术，不列编号；不要为了变短而漏掉用户的新动作。
3. 不出现工具名、英文 ID、API、RAG、模型或测试术语。
4. 只输出严格 JSON：{"query":"..."}。
"""
    if target_kind == "natural_replay":
        return shared + """
当前消息表示用户确认、恢复或继续办理 source_query 中已经说清楚的整件事，但不能重新列出其中动作。
- implicit_confirmation：只用于紧接上一条的场景，并保留“按这个、按刚才说的、就这么”之一，例如“行，就这么办吧”“可以，按这个来”；不要改写成“按那个尺码/订单”。
- recent_resume：用“刚才说的那些事”“刚才聊的那几项”“前面说的都照办”一类自然集合线索，明确承接整件事。
- object_anchor：用商品、订单或问题的简短对象线索唯一指回 source_query，同时自然带出集合范围，例如“耳机那单，前面说的都接着办吧”。
如果 source_query 确实只有一个动作，可以自然说“刚才那单价格的事继续看下”，不必硬写成“几件事”。
非紧邻确认不能只写“那个耳机继续看下”，这种说法可能只指其中一个动作。不要写“全部、整套、原样、顺序不变、再执行一遍”。
除一个用于定位的商品、订单或问题对象外，不得写出 source_query 的具体动作、检查项或名词清单。
"""
    if target_kind == "contextual_new_action":
        return shared + """
当前消息必须自然承接 source_query 中的对象或上下文（商品、订单、购物车、地址或账号等），但明确提出 target_base_query 的新处理动作。
- 保留 target_base_query 的动作；可以换成自然说法，也应把其中偶然出现的商品、订单或账号改成 source_query 正在谈的对象，不能凭空带入另一个对象。
- 不能继续、恢复或夹带 source_query 的旧动作。
- 例如历史查过某订单物流，而新目标是取消，可写“刚才那个订单我不要了，直接取消吧”。
"""
    raise ValueError(f"unsupported target kind: {target_kind}")


def user_prompt(row: dict[str, Any], feedback: str = "") -> str:
    payload = {
        "case_id": row["case_id"],
        "target_kind": row["target_kind"],
        "reference_family": row["reference_family"],
        "history_gap": row["history_gap"],
        "source_turn": row["source_turn"],
        "source_query": row["source_query"],
        "target_base_query": row["target_base_query"],
        "history_turns": [
            {"turn_id": item["turn_id"], "query": item["query"]}
            for item in row["history_turns"]
        ],
    }
    if feedback:
        payload["previous_attempt_feedback"] = feedback
        payload["instruction"] = "修正问题并换一种自然说法。"
    return json.dumps(payload, ensure_ascii=False)


def judge_system_prompt(target_kind: str) -> str:
    shared = """你是电商多轮对话数据审计员。根据完整历史判断 candidate 是否像真实用户，并严格输出 JSON。不得因为措辞流畅就放宽语义要求。
"""
    if target_kind == "natural_replay":
        return shared + """
输出格式：
{"natural":true,"unique_reference":true,"continues_whole_source_intent":true,"history_required_for_actions":true,"no_action_restatement":true,"no_source_action_details":true,"recency_valid":true,"no_numbered_turn":true,"no_evaluation_language":true,"reason":"简短原因"}

判定：
- unique_reference：结合 history_turns 只能指向 source_turn；紧邻确认可以靠位置唯一。
- continues_whole_source_intent：自然含义是继续办理 source_query 已谈妥的整件事，没有删改或新增动作。
- 对 recent_resume/object_anchor，多动作来源必须有自然的集合范围线索，例如“那些事、几件事、那几项、前面说的都照办”；只有“那个订单继续看看”不够明确。单动作来源可用自然的“那单价格的事”。
- history_required_for_actions：脱离历史无法恢复需要调用的完整有序动作。
- no_action_restatement：没有逐项复述 source_query；对象锚点不算动作复述。
- no_source_action_details：除一个定位对象外，没有写出 source_query 中的具体动作、检查项或名词清单。
- recency_valid：“刚才”只用于很近的内容；距离较远时必须有唯一对象线索。
- no_numbered_turn/no_evaluation_language：没有“第N轮、原样完整重做、步骤顺序”等造题语言。
"""
    if target_kind == "contextual_new_action":
        return shared + """
输出格式：
{"natural":true,"unique_object_reference":true,"target_actions_complete":true,"target_order_preserved":true,"does_not_inherit_source_actions":true,"no_numbered_turn":true,"no_evaluation_language":true,"reason":"简短原因"}

判定：
- unique_object_reference：对象指代结合 history_turns 只能落到 source_turn。
- target_actions_complete/target_order_preserved：candidate 完整保留 target_base_query 的动作和顺序。
- does_not_inherit_source_actions：没有继续或混入 source_query 的旧动作。
- no_numbered_turn/no_evaluation_language：没有“第N轮、历史任务、完整重做、步骤顺序”等造题语言。
"""
    raise ValueError(f"unsupported target kind: {target_kind}")


def judge_user_prompt(row: dict[str, Any], query: str) -> str:
    return json.dumps(
        {
            "target_kind": row["target_kind"],
            "reference_family": row["reference_family"],
            "history_gap": row["history_gap"],
            "source_turn": row["source_turn"],
            "source_query": row["source_query"],
            "target_base_query": row["target_base_query"],
            "history_turns": [
                {"turn_id": item["turn_id"], "query": item["query"]}
                for item in row["history_turns"]
            ],
            "candidate": query,
        },
        ensure_ascii=False,
    )


def prompt_sha256(target_kind: str, role: str) -> str:
    if role not in {"generator", "judge"}:
        raise ValueError(f"unsupported prompt role: {role}")
    if target_kind == "standalone_switch":
        prompt = f"{role}:source-generation-audit-carried-v1"
    elif role == "generator":
        prompt = system_prompt(target_kind)
    else:
        prompt = judge_system_prompt(target_kind)
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def required_judgments(target_kind: str) -> tuple[str, ...]:
    if target_kind == "natural_replay":
        return (
            "natural",
            "unique_reference",
            "continues_whole_source_intent",
            "history_required_for_actions",
            "no_action_restatement",
            "no_source_action_details",
            "recency_valid",
            "no_numbered_turn",
            "no_evaluation_language",
        )
    return (
        "natural",
        "unique_object_reference",
        "target_actions_complete",
        "target_order_preserved",
        "does_not_inherit_source_actions",
        "no_numbered_turn",
        "no_evaluation_language",
    )


def static_issues(query: str, row: dict[str, Any]) -> list[str]:
    issues = []
    max_length = 44 if row["target_kind"] == "natural_replay" else 190
    if len(query) < 2 or len(query) > max_length:
        issues.append("length")
    if INTERNAL_RE.search(query):
        issues.append("internal_term")
    if NUMBERED_TURN_RE.search(query):
        issues.append("numbered_turn")
    if EVALUATION_LANGUAGE_RE.search(query):
        issues.append("evaluation_language")
    if ACTION_LIST_RE.search(query) and row["target_kind"] == "natural_replay":
        issues.append("action_list_restatement")
    if SOURCE_DETAIL_LIST_RE.search(query) and row["target_kind"] == "natural_replay":
        issues.append("source_detail_list")
    if not HISTORY_SIGNAL_RE.search(query):
        issues.append("missing_natural_history_signal")
    family = str(row["reference_family"])
    gap = int(row["history_gap"])
    if family == "implicit_confirmation":
        if gap != 1 or not CLEAR_IMPLICIT_RE.search(query):
            issues.append("invalid_implicit_confirmation")
    else:
        source_tool_count = len(row.get("source_tools", []))
        has_scope = bool(NATURAL_COLLECTIVE_SCOPE_RE.search(query))
        if source_tool_count == 1:
            has_scope = has_scope or bool(NATURAL_SINGLE_SCOPE_RE.search(query))
        if not has_scope:
            issues.append("ambiguous_whole_source_scope")
    if gap > 2 and re.search(r"刚才|就这么|按这个|照这个", query):
        issues.append("invalid_recency")
    return issues


def generate_one(
    row: dict[str, Any],
    client: ApiClient,
    generator_model: str,
    judge_model: str,
    blocked: set[str],
    max_attempts: int,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    target_kind = str(row["target_kind"])
    generator_prompt_sha256 = prompt_sha256(target_kind, "generator")
    judge_prompt_sha256 = prompt_sha256(target_kind, "judge")
    if target_kind == "standalone_switch":
        query = str(row["target_base_query"]).strip()
        if not normalize(query):
            return None, [{"case_id": row["case_id"], "issues": ["empty_standalone"]}]
        return (
            {
                **row,
                "query": query,
                "plan_hash": sha256_json(row),
                "generation_mode": "audited_standalone_base_query",
                "generator_model": "source-generator-output",
                "judge_model": "source-generator-judge",
                "generator_prompt_version": GENERATOR_PROMPT_VERSION,
                "judge_prompt_version": JUDGE_PROMPT_VERSION,
                "generator_prompt_sha256": generator_prompt_sha256,
                "judge_prompt_sha256": judge_prompt_sha256,
                "judgment": {"source_generation_audit_carried": True},
                "accepted_attempt": 0,
            },
            [],
        )

    attempts = []
    feedback = ""
    previous_queries: list[str] = []
    for attempt in range(1, max_attempts + 1):
        try:
            raw, meta = client.chat(
                generator_model,
                system_prompt(target_kind),
                user_prompt(row, feedback),
                temperature=0.9,
                max_tokens=384,
            )
            query = str(json.loads(raw).get("query", "")).strip()
            issues = static_issues(query, row)
            judgment: dict[str, Any] = {}
            judge_raw = ""
            judge_meta: dict[str, Any] = {}
            if not issues:
                judge_raw, judge_meta = client.chat(
                    judge_model,
                    judge_system_prompt(target_kind),
                    judge_user_prompt(row, query),
                    temperature=0.0,
                    max_tokens=220,
                )
                judgment = json.loads(judge_raw)
                issues.extend(
                    f"judge_{field}"
                    for field in required_judgments(target_kind)
                    if judgment.get(field) is not True
                )
            if not issues:
                key = normalize(query)
                if not key or key in blocked:
                    issues.append("blocked_query_overlap")
        except Exception as exc:
            raw = ""
            meta = {}
            query = ""
            issues = [f"generation_error:{type(exc).__name__}"]
            judgment = {}
            judge_raw = ""
            judge_meta = {}
        attempts.append(
            {
                "case_id": row["case_id"],
                "attempt": attempt,
                "query": query,
                "issues": issues,
                "raw": raw,
                "meta": meta,
                "judge_raw": judge_raw,
                "judge_meta": judge_meta,
                "judgment": judgment,
            }
        )
        if query:
            previous_queries.append(query)
        if not issues:
            return (
                {
                    **row,
                    "query": query,
                    "plan_hash": sha256_json(row),
                    "generation_mode": "api_generated_and_independently_judged",
                    "generator_model": generator_model,
                    "judge_model": judge_model,
                    "generator_prompt_version": GENERATOR_PROMPT_VERSION,
                    "judge_prompt_version": JUDGE_PROMPT_VERSION,
                    "generator_prompt_sha256": generator_prompt_sha256,
                    "judge_prompt_sha256": judge_prompt_sha256,
                    "judgment": judgment,
                    "accepted_attempt": attempt,
                },
                attempts,
            )
        feedback = json.dumps(
            {
                "issues": issues,
                "judge_reason": str(judgment.get("reason", "")),
                "previous_candidates_to_avoid": previous_queries[-6:],
                "instruction": "换一种没有出现过的自然说法；不得只改标点或语气词。",
            },
            ensure_ascii=False,
        )
    return None, attempts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--api-key-file", type=Path, default=Path("/root/.cli-proxy-api/api_key.txt")
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8317/v1")
    parser.add_argument("--generator-model", default="gpt-5.4-mini")
    parser.add_argument("--judge-model", default="gpt-5.5")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--http-retries", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--min-acceptance-rate", type=float, default=0.98)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--blocked-queries", type=Path, action="append", default=[])
    args = parser.parse_args()
    if min(args.workers, args.max_attempts, args.http_retries, args.checkpoint_every) <= 0:
        raise SystemExit("worker, retry, attempt and checkpoint values must be positive")
    if not 0.90 <= args.min_acceptance_rate <= 1.0:
        raise SystemExit("--min-acceptance-rate must be in [0.90, 1.0]")
    plan = load_jsonl(args.plan)
    if args.limit:
        plan = plan[: args.limit]
    if not plan or any(str(row.get("target_kind")) not in {
        "natural_replay", "contextual_new_action", "standalone_switch"
    } for row in plan):
        raise SystemExit("invalid or empty target plan")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    accepted_path = args.output_dir / "accepted.jsonl"
    attempts_path = args.output_dir / "attempts.jsonl"
    plan_by_id = {str(row["case_id"]): row for row in plan}
    existing_rows = load_jsonl(accepted_path)
    accepted = {}
    for row in existing_rows:
        case_id = str(row.get("case_id"))
        if case_id not in plan_by_id or row.get("plan_hash") != sha256_json(plan_by_id[case_id]):
            raise SystemExit(f"stale accepted target: {case_id}")
        target_kind = str(plan_by_id[case_id]["target_kind"])
        expected_generator_hash = prompt_sha256(target_kind, "generator")
        expected_judge_hash = prompt_sha256(target_kind, "judge")
        if (
            row.get("generator_prompt_version") != GENERATOR_PROMPT_VERSION
            or row.get("judge_prompt_version") != JUDGE_PROMPT_VERSION
            or row.get("generator_prompt_sha256") != expected_generator_hash
            or row.get("judge_prompt_sha256") != expected_judge_hash
        ):
            raise SystemExit(f"stale accepted prompt provenance: {case_id}")
        accepted[case_id] = row
    blocked_keys: set[str] = set()
    blocked_artifacts = []
    for path in args.blocked_queries:
        rows = load_jsonl(path)
        values = {
            normalize(str(row.get("query", row.get("问题", ""))))
            for row in rows
            if str(row.get("query", row.get("问题", ""))).strip()
        }
        blocked_keys.update(values)
        blocked_artifacts.append(
            {"path": str(path.resolve()), "rows": len(rows), "sha256": sha256_file(path)}
        )
    client = ApiClient(
        args.base_url,
        api_key(args.api_key_file),
        timeout=args.timeout,
        http_retries=args.http_retries,
    )
    pending = [row for row in plan if str(row["case_id"]) not in accepted]
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                generate_one,
                row,
                client,
                args.generator_model,
                args.judge_model,
                blocked_keys,
                args.max_attempts,
            ): row
            for row in pending
        }
        with attempts_path.open("a", encoding="utf-8") as attempts_handle:
            for future in as_completed(futures):
                row = futures[future]
                result, attempts = future.result()
                for attempt in attempts:
                    attempts_handle.write(
                        json.dumps(attempt, ensure_ascii=False, separators=(",", ":")) + "\n"
                    )
                attempts_handle.flush()
                if result is not None:
                    accepted[str(row["case_id"])] = result
                completed += 1
                if completed % args.checkpoint_every == 0:
                    write_jsonl(
                        accepted_path,
                        [accepted[key] for key in plan_by_id if key in accepted],
                    )
                    print(
                        f"completed={completed}/{len(pending)} accepted={len(accepted)}/{len(plan)}",
                        flush=True,
                    )
    write_jsonl(accepted_path, [accepted[key] for key in plan_by_id if key in accepted])
    missing = [case_id for case_id in plan_by_id if case_id not in accepted]
    accepted_rows = [accepted[key] for key in plan_by_id if key in accepted]
    duplicate_count = len(accepted_rows) - len(
        {normalize(str(row["query"])) for row in accepted_rows}
    )
    static_failures = sum(
        bool(static_issues(str(row["query"]), row))
        for row in accepted_rows
        if row["target_kind"] != "standalone_switch"
    )
    judge_failures = sum(
        any(
            row.get("judgment", {}).get(field) is not True
            for field in required_judgments(str(row["target_kind"]))
        )
        for row in accepted_rows
        if row["target_kind"] != "standalone_switch"
    )
    blocked_overlap = sum(
        row["target_kind"] != "standalone_switch"
        and normalize(str(row["query"])) in blocked_keys
        for row in accepted_rows
    )
    attempts_by_case: dict[str, list[dict[str, Any]]] = {}
    if attempts_path.exists():
        for attempt_row in load_jsonl(attempts_path):
            attempts_by_case.setdefault(str(attempt_row.get("case_id")), []).append(
                attempt_row
            )
    exclusions = []
    for case_id in missing:
        case_attempts = attempts_by_case.get(case_id, [])
        issue_counts = Counter(
            str(issue)
            for attempt_row in case_attempts
            for issue in attempt_row.get("issues", [])
        )
        exclusions.append(
            {
                "case_id": case_id,
                "target_kind": str(plan_by_id[case_id]["target_kind"]),
                "reference_family": str(plan_by_id[case_id]["reference_family"]),
                "attempts": len(case_attempts),
                "issue_counts": dict(sorted(issue_counts.items())),
                "policy_reason": "failed pre-evaluation naturalness or semantic-clarity audit",
            }
        )
    audited_exclusions_ok = all(
        exclusion["target_kind"] != "standalone_switch"
        and exclusion["attempts"] >= args.max_attempts
        and any(
            not issue.startswith("generation_error:")
            for issue in exclusion["issue_counts"]
        )
        for exclusion in exclusions
    )
    acceptance_rate = len(accepted_rows) / len(plan)
    strict_ok = (
        acceptance_rate >= args.min_acceptance_rate
        and audited_exclusions_ok
        and static_failures == 0
        and judge_failures == 0
        and blocked_overlap == 0
    )
    audit = {
        "status": "complete" if not missing else "complete_with_audited_exclusions",
        "strict_ok": strict_ok,
        "plan": str(args.plan.resolve()),
        "plan_sha256": sha256_file(args.plan),
        "rows": len(plan),
        "accepted": len(accepted_rows),
        "acceptance_rate": acceptance_rate,
        "min_acceptance_rate": args.min_acceptance_rate,
        "missing": missing,
        "exclusions": exclusions,
        "audited_exclusions_ok": audited_exclusions_ok,
        "target_kind": dict(
            sorted(
                {
                    kind: sum(row["target_kind"] == kind for row in accepted_rows)
                    for kind in ("natural_replay", "contextual_new_action", "standalone_switch")
                }.items()
            )
        ),
        "generator_model": args.generator_model,
        "judge_model": args.judge_model,
        "generator_prompt_version": GENERATOR_PROMPT_VERSION,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "generator_prompt_sha256_by_kind": {
            kind: prompt_sha256(kind, "generator")
            for kind in ("natural_replay", "contextual_new_action", "standalone_switch")
        },
        "judge_prompt_sha256_by_kind": {
            kind: prompt_sha256(kind, "judge")
            for kind in ("natural_replay", "contextual_new_action", "standalone_switch")
        },
        "semantic_judge_passed_rows": sum(
            row["target_kind"] != "standalone_switch" for row in accepted_rows
        )
        - judge_failures,
        "duplicate_count": duplicate_count,
        "static_failure_count": static_failures,
        "judge_failure_count": judge_failures,
        "generated_blocked_query_overlap": blocked_overlap,
        "blocked_artifacts": blocked_artifacts,
        "accepted_sha256": sha256_file(accepted_path),
        "attempts_sha256": sha256_file(attempts_path) if attempts_path.exists() else None,
    }
    write_json(args.output_dir / "audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False))
    return 0 if audit["strict_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
