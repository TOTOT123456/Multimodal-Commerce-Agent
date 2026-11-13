#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate audited API-native query pairs that differ only in requested order."""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import catalog_by_id, load_catalog, load_jsonl  # noqa: E402
from finetune.generate_assignment3_api_native_dev import (  # noqa: E402
    ApiClient,
    api_key_from_proxy_config,
    forbidden_phrases,
    normalize,
    static_issues,
    target_specs,
    write_json_atomic,
    write_jsonl_atomic,
)
from finetune.tool_trace_common import sha256_file, sha256_json  # noqa: E402


GENERATOR_PROMPT_VERSION = "assignment3-order-counterfactual-generator-v1"
JUDGE_PROMPT_VERSION = "assignment3-order-counterfactual-judge-v1"


def pair_hash(pair: dict[str, Any]) -> str:
    return sha256_json(
        {
            "pair_id": pair["pair_id"],
            "tools_a": pair["tools_a"],
            "tools_b": pair["tools_b"],
        }
    )


def generation_system_prompt() -> str:
    return """你是中文电商真实用户 query 数据作者。现在要为每个样本创作一对“顺序反事实”用户消息。

每对 query 的硬性规则：
1. 两条消息必须具有完全相同的用户、商品/订单/地区、时间条件、限制、任务集合和结果要求；不得增加、删除、合并或弱化任务。
2. 唯一的任务语义差异是用户明确要求的处理顺序，必须分别严格对应给出的 A、B 两个能力序列。
3. 两条消息都要自然、可信、口语化；可以使用不同连接词，但不许写成工具目录、编号清单或能力说明书。
4. 不能出现英文工具 ID、工具名、API、RAG、catalog、MCP、候选工具、内部字段或实现过程。
5. 不得照抄能力说明，也不能为了改变顺序而篡改对象、数量、时效或操作强度。

只输出严格 JSON：
{"results":[{"pair_id":"输入pair_id","question_a":"A顺序的自然query","question_b":"B顺序的自然query"}]}"""


def judge_system_prompt() -> str:
    return """你是独立、严格的中文工具意图数据审计员。输入是一对用户 query 以及各自有序目标能力序列。

逐项检查：
1. A 和 B 是否表达完全相同的对象、条件、限制、任务集合、操作强度和结果要求；
2. A 是否严格表达 A 的有序能力序列，B 是否严格表达 B 的有序能力序列；
3. 两条 query 的唯一任务语义差异是否为显式请求顺序；
4. 是否自然、没有遗漏/新增任务，并且不含工具 ID、API、RAG、catalog、MCP、候选工具或内部实现术语。

verdict=pass 时所有布尔值必须为 true，issues 必须为空。
只输出严格 JSON：
{"results":[{"pair_id":"输入pair_id","verdict":"pass|fail","same_context_and_constraints":true,"same_task_set":true,"operation_strength_preserved":true,"order_a_preserved":true,"order_b_preserved":true,"natural_user_query":true,"no_internal_leak":true,"issues":[]}]}"""


def parse_generation(raw: str, expected_pair_ids: set[str]) -> dict[str, tuple[str, str]]:
    payload = json.loads(raw)
    values = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(values, list):
        raise ValueError("generation results is not a list")
    parsed: dict[str, tuple[str, str]] = {}
    for value in values:
        if not isinstance(value, dict) or set(value) != {"pair_id", "question_a", "question_b"}:
            raise ValueError("generation item schema mismatch")
        pair_id = str(value["pair_id"])
        question_a = str(value["question_a"]).strip()
        question_b = str(value["question_b"]).strip()
        if pair_id not in expected_pair_ids or pair_id in parsed or not question_a or not question_b:
            raise ValueError("generation pair mismatch")
        parsed[pair_id] = (question_a, question_b)
    if set(parsed) != expected_pair_ids:
        raise ValueError("generation result pair ids mismatch")
    return parsed


JUDGE_FIELDS = {
    "pair_id",
    "verdict",
    "same_context_and_constraints",
    "same_task_set",
    "operation_strength_preserved",
    "order_a_preserved",
    "order_b_preserved",
    "natural_user_query",
    "no_internal_leak",
    "issues",
}


def parse_judgments(raw: str, expected_pair_ids: set[str]) -> dict[str, dict[str, Any]]:
    payload = json.loads(raw)
    values = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(values, list):
        raise ValueError("judge results is not a list")
    parsed: dict[str, dict[str, Any]] = {}
    for value in values:
        if not isinstance(value, dict) or set(value) != JUDGE_FIELDS:
            raise ValueError("judge item schema mismatch")
        pair_id = str(value["pair_id"])
        if pair_id not in expected_pair_ids or pair_id in parsed:
            raise ValueError("judge pair mismatch")
        if not isinstance(value["issues"], list) or not all(isinstance(item, str) for item in value["issues"]):
            raise ValueError("judge issues mismatch")
        parsed[pair_id] = value
    if set(parsed) != expected_pair_ids:
        raise ValueError("judge result pair ids mismatch")
    return parsed


def judge_pass(value: dict[str, Any]) -> bool:
    return value["verdict"] == "pass" and all(
        value[key] is True
        for key in (
            "same_context_and_constraints",
            "same_task_set",
            "operation_strength_preserved",
            "order_a_preserved",
            "order_b_preserved",
            "natural_user_query",
            "no_internal_leak",
        )
    ) and not value["issues"]


def pair_targets(pair: dict[str, Any], catalog: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = [
        {
            "id": pair["id_a"],
            "tools": pair["tools_a"],
            "tool_count": pair["tool_count"],
            "domain": pair["domain"],
            "combo_seen_bucket": pair["combo_seen_bucket"],
        },
        {
            "id": pair["id_b"],
            "tools": pair["tools_b"],
            "tool_count": pair["tool_count"],
            "domain": pair["domain"],
            "combo_seen_bucket": pair["combo_seen_bucket"],
        },
    ]
    left, right = target_specs(raw, catalog)
    return left, right


def render_generation_user(batch: list[dict[str, Any]]) -> str:
    samples = []
    for pair in batch:
        left, right = pair["targets"]
        samples.append(
            {
                "pair_id": pair["pair_id"],
                "shared_expression_preference": left["style_hint"],
                "ordered_capabilities_a": left["capabilities"],
                "ordered_capabilities_b": right["capabilities"],
            }
        )
    return json.dumps({"samples": samples}, ensure_ascii=False, separators=(",", ":"))


def render_judge_user(batch: list[dict[str, Any]], questions: dict[str, tuple[str, str]]) -> str:
    samples = []
    for pair in batch:
        left, right = pair["targets"]
        question_a, question_b = questions[pair["pair_id"]]
        samples.append(
            {
                "pair_id": pair["pair_id"],
                "question_a": question_a,
                "question_b": question_b,
                "ordered_capabilities_a": left["capabilities"],
                "ordered_capabilities_b": right["capabilities"],
            }
        )
    return json.dumps({"samples": samples}, ensure_ascii=False, separators=(",", ":"))


def process_batch(
    batch: list[dict[str, Any]],
    client: ApiClient,
    args: argparse.Namespace,
    forbidden: list[str],
    catalog_rows: list[dict[str, Any]],
    blocked_queries: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    pending = list(batch)
    accepted: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    for attempt in range(1, args.max_attempts + 1):
        if not pending:
            break
        pair_ids = {pair["pair_id"] for pair in pending}
        try:
            raw, generation_meta = client.chat(
                args.generator_model,
                generation_system_prompt(),
                render_generation_user(pending),
                args.temperature,
                args.max_tokens,
            )
            questions = parse_generation(raw, pair_ids)
        except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
            attempts.extend(
                {
                    "pair_id": pair["pair_id"],
                    "attempt": attempt,
                    "stage": "generation",
                    "accepted": False,
                    "issues": [f"generation_error:{type(exc).__name__}"],
                }
                for pair in pending
            )
            continue

        clean = []
        for pair in pending:
            question_a, question_b = questions[pair["pair_id"]]
            left, right = pair["targets"]
            issues = static_issues(
                question_a, left, forbidden, catalog_rows, blocked_queries
            ) + static_issues(
                question_b, right, forbidden, catalog_rows, blocked_queries
            )
            if question_a == question_b:
                issues.append("pair_questions_identical")
            if issues:
                attempts.append(
                    {
                        "pair_id": pair["pair_id"],
                        "attempt": attempt,
                        "stage": "static",
                        "accepted": False,
                        "issues": sorted(set(issues)),
                    }
                )
            else:
                clean.append(pair)

        judgments: dict[str, dict[str, Any]] = {}
        judge_meta: dict[str, Any] = {}
        if clean:
            try:
                raw, judge_meta = client.chat(
                    args.judge_model,
                    judge_system_prompt(),
                    render_judge_user(clean, questions),
                    0.0,
                    args.max_tokens,
                )
                judgments = parse_judgments(raw, {pair["pair_id"] for pair in clean})
            except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
                attempts.extend(
                    {
                        "pair_id": pair["pair_id"],
                        "attempt": attempt,
                        "stage": "judge",
                        "accepted": False,
                        "issues": [f"judge_error:{type(exc).__name__}"],
                    }
                    for pair in clean
                )

        accepted_ids = set()
        for pair in clean:
            judgment = judgments.get(pair["pair_id"])
            if judgment is not None and judge_pass(judgment):
                accepted_ids.add(pair["pair_id"])
                question_a, question_b = questions[pair["pair_id"]]
                accepted.append(
                    {
                        "pair_id": pair["pair_id"],
                        "id_a": pair["id_a"],
                        "id_b": pair["id_b"],
                        "question_a": question_a,
                        "question_b": question_b,
                        "tools_a": pair["tools_a"],
                        "tools_b": pair["tools_b"],
                        "tool_count": pair["tool_count"],
                        "domain": pair["domain"],
                        "combo_seen_bucket": pair["combo_seen_bucket"],
                        "target_hash": pair_hash(pair),
                        "attempt": attempt,
                        "generator_model": args.generator_model,
                        "judge_model": args.judge_model,
                        "generator_prompt_version": GENERATOR_PROMPT_VERSION,
                        "judge_prompt_version": JUDGE_PROMPT_VERSION,
                        "generation_meta": generation_meta,
                        "judgment": judgment,
                        "judge_meta": judge_meta,
                    }
                )
                attempts.append(
                    {"pair_id": pair["pair_id"], "attempt": attempt, "stage": "judge", "accepted": True, "issues": []}
                )
            else:
                issues = list(judgment["issues"]) if judgment else ["judge_inconsistent"]
                attempts.append(
                    {
                        "pair_id": pair["pair_id"],
                        "attempt": attempt,
                        "stage": "judge",
                        "accepted": False,
                        "issues": issues,
                    }
                )
        pending = [pair for pair in pending if pair["pair_id"] not in accepted_ids]
    return accepted, attempts, pending


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-plan", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8317/v1")
    parser.add_argument("--proxy-config", type=Path, default=Path("/root/.cli-proxy-api/config.yaml"))
    parser.add_argument("--generator-model", default="gpt-5.5")
    parser.add_argument("--judge-model", default="gpt-5.4")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--http-retries", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument(
        "--blocked-query-file",
        action="append",
        type=Path,
        default=[],
        help="JSONL split whose queries are forbidden; repeat for every protected split.",
    )
    args = parser.parse_args()
    if min(args.workers, args.batch_size, args.max_attempts, args.http_retries) <= 0:
        raise SystemExit("workers, batch-size, attempts, and retries must be positive")

    catalog_rows = load_catalog(args.catalog)
    catalog = catalog_by_id(catalog_rows)
    blocked_queries: set[str] = set()
    blocked_query_hashes: dict[str, str] = {}
    for path in args.blocked_query_file:
        if not path.exists():
            raise SystemExit(f"missing blocked query file: {path}")
        blocked_query_hashes[str(path)] = sha256_file(path)
        for row in load_jsonl(path):
            question = row.get("问题", row.get("question", row.get("query")))
            if isinstance(question, str) and question.strip():
                blocked_queries.add(normalize(question))
    raw_pairs = load_jsonl(args.pair_plan)
    pairs = []
    for pair in raw_pairs:
        required = {"pair_id", "id_a", "id_b", "tools_a", "tools_b", "tool_count", "domain", "combo_seen_bucket"}
        if set(pair) != required or set(pair["tools_a"]) != set(pair["tools_b"]) or pair["tools_a"] == pair["tools_b"]:
            raise SystemExit(f"invalid pair plan row: {pair.get('pair_id')}")
        left, right = pair_targets(pair, catalog)
        pairs.append({**pair, "targets": (left, right)})
    if len({pair["pair_id"] for pair in pairs}) != len(pairs):
        raise SystemExit("duplicate pair ids")

    accepted_path = args.output_dir / "accepted_generation_pairs.jsonl"
    attempts_path = args.output_dir / "generation_attempts_pairs.jsonl"
    accepted: dict[str, dict[str, Any]] = {}
    if accepted_path.exists():
        for row in load_jsonl(accepted_path):
            pair_id = str(row["pair_id"])
            matching = next((pair for pair in pairs if pair["pair_id"] == pair_id), None)
            if matching is None or row.get("target_hash") != pair_hash(matching) or pair_id in accepted:
                raise SystemExit(f"invalid existing accepted pair: {pair_id}")
            if normalize(str(row["question_a"])) in blocked_queries or normalize(
                str(row["question_b"])
            ) in blocked_queries:
                raise SystemExit(f"existing accepted pair overlaps a protected split: {pair_id}")
            accepted[pair_id] = row
    attempts = load_jsonl(attempts_path) if attempts_path.exists() else []
    seen_questions = {
        normalize(question)
        for row in accepted.values()
        for question in (row["question_a"], row["question_b"])
    }
    if len(seen_questions) != 2 * len(accepted):
        raise SystemExit("existing accepted pairs contain duplicate queries")
    pending = [pair for pair in pairs if pair["pair_id"] not in accepted]
    random.Random(args.seed).shuffle(pending)
    batches = [pending[index : index + args.batch_size] for index in range(0, len(pending), args.batch_size)]
    client = ApiClient(
        args.base_url,
        api_key_from_proxy_config(args.proxy_config),
        args.timeout,
        args.http_retries,
    )
    forbidden = forbidden_phrases(catalog_rows)
    failed: list[dict[str, Any]] = []
    lock = threading.Lock()
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                process_batch,
                batch,
                client,
                args,
                forbidden,
                catalog_rows,
                blocked_queries,
            )
            for batch in batches
        ]
        for completed, future in enumerate(as_completed(futures), start=1):
            batch_accepted, batch_attempts, batch_failed = future.result()
            with lock:
                for row in batch_accepted:
                    pair_questions = {normalize(row["question_a"]), normalize(row["question_b"])}
                    if len(pair_questions) != 2 or seen_questions.intersection(pair_questions):
                        batch_attempts.append(
                            {"pair_id": row["pair_id"], "attempt": row["attempt"], "stage": "global_duplicate", "accepted": False, "issues": ["duplicate_query"]}
                        )
                        batch_failed.append(next(pair for pair in pairs if pair["pair_id"] == row["pair_id"]))
                        continue
                    accepted[row["pair_id"]] = row
                    seen_questions.update(pair_questions)
                attempts.extend(batch_attempts)
                failed.extend(batch_failed)
                ordered = [accepted[pair["pair_id"]] for pair in pairs if pair["pair_id"] in accepted]
                write_jsonl_atomic(accepted_path, ordered)
                write_jsonl_atomic(attempts_path, attempts)
                if completed % 10 == 0 or completed == len(batches):
                    print(f"order-counterfactual generation batches={completed}/{len(batches)} accepted={len(accepted)}/{len(pairs)}", flush=True)

    ordered = [accepted[pair["pair_id"]] for pair in pairs if pair["pair_id"] in accepted]
    all_pass = len(ordered) == len(pairs) and not failed
    dev_rows = []
    for pair in ordered:
        for suffix, question_key, tools_key in (("a", "question_a", "tools_a"), ("b", "question_b", "tools_b")):
            dev_rows.append(
                {
                    "id": pair[f"id_{suffix}"],
                    "问题": pair[question_key],
                    "工具": ",".join(pair[tools_key]),
                    "思考": "按用户明确叙述的业务顺序选择对应工具。",
                    "source": "assignment3_api_native_order_counterfactual",
                    "domain": pair["domain"],
                    "tool_count": pair["tool_count"],
                    "generation_split": "assignment3_api_native_order_counterfactual",
                    "template_family": "api_native_freeform_paired_counterfactual",
                    "combo_seen_bucket": pair["combo_seen_bucket"],
                    "pair_id": pair["pair_id"],
                    "pair_variant": suffix,
                }
            )
    dev_path = args.output_dir / "counterfactual_dev.jsonl"
    if all_pass:
        write_jsonl_atomic(dev_path, dev_rows)
    audit = {
        "audit_version": "assignment3-order-counterfactual-generation-audit-v1",
        "strict_ok": all_pass,
        "pair_count": len(pairs),
        "accepted_pair_count": len(ordered),
        "query_count": 2 * len(ordered),
        "failed_pair_count": len(failed),
        "failed_pair_ids": [pair["pair_id"] for pair in failed],
        "generator_model": args.generator_model,
        "judge_model": args.judge_model,
        "generator_prompt_version": GENERATOR_PROMPT_VERSION,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "query_templates_used": False,
        "pair_context_jointly_generated": True,
        "training_use_allowed": False,
        "blocked_query_protection": {
            "files": blocked_query_hashes,
            "unique_normalized_queries": len(blocked_queries),
            "final_overlap_count": sum(
                normalize(question) in blocked_queries
                for row in ordered
                for question in (row["question_a"], row["question_b"])
            ),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "inputs": {"pair_plan": str(args.pair_plan), "pair_plan_sha256": sha256_file(args.pair_plan), "catalog": str(args.catalog), "catalog_sha256": sha256_file(args.catalog)},
        "outputs": {
            "accepted_pairs": str(accepted_path),
            "accepted_pairs_sha256": sha256_file(accepted_path),
            "attempts": str(attempts_path),
            "attempts_sha256": sha256_file(attempts_path),
            "dev": str(dev_path) if all_pass else None,
            "dev_sha256": sha256_file(dev_path) if all_pass else None,
        },
    }
    write_json_atomic(args.output_dir / "generation_audit.json", audit)
    print(f"order-counterfactual generation accepted={len(ordered)}/{len(pairs)} strict_ok={all_pass}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
