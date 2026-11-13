#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit residual TopK failures for the final retrieval and reranking pipeline."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def is_allhit(row: dict[str, Any], k: int) -> bool:
    gold = set(row.get("gold_tools", []))
    return bool(gold) and gold.issubset(set(row.get("pred_tools", [])[:k]))


def sorted_counter(counter: Counter[Any], limit: int | None = None) -> list[list[Any]]:
    rows = [[key, value] for key, value in counter.items()]
    rows.sort(key=lambda row: (-row[1], str(row[0])))
    return rows if limit is None else rows[:limit]


def rank_band(rank: int | None, k: int) -> str:
    if rank is None:
        return "absent"
    if rank <= 10:
        return f"{k + 1}-10"
    if rank <= 30:
        return "11-30"
    if rank <= 50:
        return "31-50"
    return ">50"


def query_rank_band(ranks: list[int | None], k: int) -> str:
    numeric = [rank for rank in ranks if rank is not None]
    if len(numeric) != len(ranks):
        return "has_gold_>50_or_absent"
    farthest = max(numeric)
    if farthest <= 10:
        return f"all_gold_{k + 1}-10"
    if farthest <= 30:
        return "has_gold_11-30"
    if farthest <= 50:
        return "has_gold_31-50"
    return "has_gold_>50_or_absent"


def compact_case(
    row: dict[str, Any], missing: list[str], ranks: list[int | None], all_in_candidate: bool, k: int
) -> dict[str, Any]:
    return {
        "id": row.get("id", ""),
        "query": row.get("query", ""),
        "gold_tools": row.get("gold_tools", []),
        "topk_tools": row.get("pred_tools", [])[:k],
        "missing_tools": missing,
        "missing_ranks": ranks,
        "all_gold_in_reranker_candidate": all_in_candidate,
    }


def analyze(
    single_rows: list[dict[str, Any]],
    reranker_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    k: int,
    candidate_k: int,
    curve_ks: list[int],
) -> dict[str, Any]:
    single_by_id = {str(row.get("id", "")): row for row in single_rows}
    reranker_by_id = {str(row.get("id", "")): row for row in reranker_rows}
    if len(single_by_id) != len(single_rows) or len(reranker_by_id) != len(reranker_rows):
        raise ValueError("prediction ids must be unique")

    transitions: Counter[str] = Counter()
    stage_transitions: dict[str, Counter[str]] = {
        "embedding_to_final": Counter(),
        "first_stage_to_final": Counter(),
        "reranker_to_final": Counter(),
    }
    stage_successes: dict[str, Counter[int]] = {
        "embedding": Counter(),
        "first_stage": Counter(),
        "reranker": Counter(),
        "final": Counter(),
    }
    total_by_gold: Counter[int] = Counter()
    single_fail_by_gold: Counter[int] = Counter()
    final_fail_by_gold: Counter[int] = Counter()
    missing_count: Counter[int] = Counter()
    missing_rank_bands: Counter[str] = Counter()
    query_rank_bands: Counter[str] = Counter()
    candidate_coverage: Counter[str] = Counter()
    candidate_coverage_missing_tools: Counter[str] = Counter()
    missing_tools: Counter[str] = Counter()
    competing_tools: Counter[str] = Counter()
    competing_pairs: Counter[str] = Counter()
    examples: dict[str, dict[str, Any] | None] = {
        "candidate_coverage_miss": None,
        "candidate_present_but_final_miss": None,
        "prefix_ranking_miss": None,
        "six_tool_capacity_case": None,
    }

    for final_row in final_rows:
        row_id = str(final_row.get("id", ""))
        single_row = single_by_id.get(row_id)
        reranker_row = reranker_by_id.get(row_id)
        if single_row is None or reranker_row is None:
            raise ValueError(f"missing aligned prediction for id={row_id}")
        if (
            single_row.get("query") != final_row.get("query")
            or reranker_row.get("query") != final_row.get("query")
            or single_row.get("gold_tools") != final_row.get("gold_tools")
            or reranker_row.get("gold_tools") != final_row.get("gold_tools")
        ):
            raise ValueError(f"prediction alignment mismatch for id={row_id}")

        gold = set(final_row.get("gold_tools", []))
        if not gold:
            continue
        candidate_tools = (
            reranker_row.get("candidate_tools") or reranker_row.get("pred_tools", [])
        )[:candidate_k]
        stage_rankings = {
            "embedding": single_row.get("pred_tools", []),
            "first_stage": candidate_tools,
            "reranker": reranker_row.get("pred_tools", []),
            "final": final_row.get("pred_tools", []),
        }
        for stage, ranking in stage_rankings.items():
            for curve_k in curve_ks:
                stage_successes[stage][curve_k] += int(gold.issubset(set(ranking[:curve_k])))
        gold_count = len(gold)
        total_by_gold[gold_count] += 1
        single_ok = is_allhit(single_row, k)
        first_stage_ok = gold.issubset(set(candidate_tools[:k]))
        reranker_ok = is_allhit(reranker_row, k)
        final_ok = is_allhit(final_row, k)
        if not single_ok:
            single_fail_by_gold[gold_count] += 1
        if not final_ok:
            final_fail_by_gold[gold_count] += 1
        transitions[f"single_{'success' if single_ok else 'fail'}_final_{'success' if final_ok else 'fail'}"] += 1
        for stage_name, stage_ok in (
            ("embedding_to_final", single_ok),
            ("first_stage_to_final", first_stage_ok),
            ("reranker_to_final", reranker_ok),
        ):
            stage_transitions[stage_name][
                f"{'success' if stage_ok else 'fail'}_to_{'success' if final_ok else 'fail'}"
            ] += 1
        if final_ok:
            continue

        topk = final_row.get("pred_tools", [])[:k]
        topk_set = set(topk)
        missing = sorted(gold - topk_set)
        missing_count[len(missing)] += 1
        rank_by_tool = {tool: index + 1 for index, tool in enumerate(final_row.get("pred_tools", []))}
        ranks = [rank_by_tool.get(tool) for tool in missing]

        candidate_missing = gold - set(candidate_tools)
        all_in_candidate = not candidate_missing
        candidate_coverage["all_gold_in_reranker_N"] += int(all_in_candidate)
        candidate_coverage["some_gold_outside_reranker_N"] += int(not all_in_candidate)
        candidate_coverage_missing_tools.update(candidate_missing)

        for tool, rank in zip(missing, ranks):
            missing_tools[tool] += 1
            missing_rank_bands[rank_band(rank, k)] += 1
            for competitor in topk:
                if competitor not in gold:
                    competing_tools[competitor] += 1
                    competing_pairs[f"{tool} <- {competitor}"] += 1
        query_rank_bands[query_rank_band(ranks, k)] += 1

        case = compact_case(final_row, missing, ranks, all_in_candidate, k)
        if not all_in_candidate and examples["candidate_coverage_miss"] is None:
            examples["candidate_coverage_miss"] = case
        if all_in_candidate and examples["candidate_present_but_final_miss"] is None:
            examples["candidate_present_but_final_miss"] = case
        if all(rank is not None and k < rank <= 10 for rank in ranks) and examples["prefix_ranking_miss"] is None:
            examples["prefix_ranking_miss"] = case
        if gold_count == 6 and examples["six_tool_capacity_case"] is None:
            examples["six_tool_capacity_case"] = case

    final_failures = sum(final_fail_by_gold.values())
    total = sum(total_by_gold.values())
    per_gold_count = []
    for gold_count in sorted(total_by_gold):
        bucket_total = total_by_gold[gold_count]
        single_failures = single_fail_by_gold[gold_count]
        final_failures_for_count = final_fail_by_gold[gold_count]
        per_gold_count.append(
            {
                "gold_tool_count": gold_count,
                "total": bucket_total,
                "single_failures": single_failures,
                "single_failure_rate": single_failures / bucket_total,
                "final_failures": final_failures_for_count,
                "final_failure_rate": final_failures_for_count / bucket_total,
            }
        )

    return {
        "k": k,
        "candidate_k": candidate_k,
        "curve_ks": curve_ks,
        "total": total,
        "max_gold_tool_count": max(total_by_gold, default=0),
        "allhit_by_stage": {
            stage: {
                str(curve_k): {
                    "successes": successes[curve_k],
                    "failures": total - successes[curve_k],
                    "rate": successes[curve_k] / total if total else 0.0,
                }
                for curve_k in curve_ks
            }
            for stage, successes in stage_successes.items()
        },
        "single_failures": sum(single_fail_by_gold.values()),
        "final_failures": final_failures,
        "transitions": dict(transitions),
        "stage_transitions": {
            stage: dict(counter) for stage, counter in stage_transitions.items()
        },
        "per_gold_count": per_gold_count,
        "missing_tool_count_per_failure": sorted_counter(missing_count),
        "missing_tool_rank_bands": sorted_counter(missing_rank_bands),
        "failure_query_rank_bands": sorted_counter(query_rank_bands),
        "reranker_candidate_coverage": dict(candidate_coverage),
        "failure_origin": {
            f"outside_first_stage_top_{candidate_k}": candidate_coverage.get(
                "some_gold_outside_reranker_N", 0
            ),
            f"inside_first_stage_top_{candidate_k}_but_final_miss": candidate_coverage.get(
                "all_gold_in_reranker_N", 0
            ),
        },
        "candidate_coverage_missing_tools": sorted_counter(candidate_coverage_missing_tools, 10),
        "top_missing_tools": sorted_counter(missing_tools, 10),
        "top_competing_tools": sorted_counter(competing_tools, 10),
        "top_missing_competitor_pairs": sorted_counter(competing_pairs, 12),
        "examples": examples,
    }


def fmt_rate(value: float) -> str:
    return f"{value:.2%}"


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    transitions = payload["transitions"]
    k = payload["k"]
    total = payload["total"]
    final_failures = payload["final_failures"]
    max_gold_tool_count = payload["max_gold_tool_count"]
    all_in_candidate = payload["reranker_candidate_coverage"].get("all_gold_in_reranker_N", 0)
    outside_candidate = payload["reranker_candidate_coverage"].get(
        "some_gold_outside_reranker_N", 0
    )
    candidate_coverage_missing_tools = payload["candidate_coverage_missing_tools"]
    all_prefix = (
        dict(payload["failure_query_rank_bands"]).get(f"all_gold_{k + 1}-10", 0)
        if k < 10
        else 0
    )
    query_band_labels = {
        f"all_gold_{k + 1}-10": f"所有漏召 gold 位于 rank {k + 1}-10",
        "has_gold_11-30": "至少一个漏召 gold 位于 rank 11-30",
        "has_gold_31-50": "至少一个漏召 gold 位于 rank 31-50",
        "has_gold_>50_or_absent": "至少一个漏召 gold 位于 rank >50 或不在最终排序中",
    }
    query_band_order = [
        f"all_gold_{k + 1}-10",
        "has_gold_11-30",
        "has_gold_31-50",
        "has_gold_>50_or_absent",
    ]
    query_band_counts = dict(payload["failure_query_rank_bands"])
    lines = [
        f"# 最终链路 Top{k} Badcase 审计",
        "",
        "分析对象是最终 source/reranker RRF；单路 embedding 仅作前后对照，不单独把它的失败当作当前上线链路的 badcase。BM25、候选池和裸 reranker 不是最终输出，只用候选覆盖和前后转移定位根因。严格 AllHit 只要求 gold 完整进入 TopK，非 gold 工具本身不是错误；只有它把必需工具挤出 TopK 时才是竞争信号。",
        f"输入预测：single `{payload['single_preds']}`；reranker `{payload['reranker_preds']}`；final `{payload['final_preds']}`。",
        "",
        "## 结果变化",
        "",
        "| 单路 embedding -> 最终链路 | query 数 | 含义 |",
        "|---|---:|---|",
        f"| fail -> success | {transitions.get('single_fail_final_success', 0)} | 最终链路实际恢复的 Top{k} 失败 |",
        f"| fail -> fail | {transitions.get('single_fail_final_fail', 0)} | 当前需要分析的残余失败主体 |",
        f"| success -> fail | {transitions.get('single_success_final_fail', 0)} | 融合引入的回归，后续必须单列监控 |",
        f"| success -> success | {transitions.get('single_success_final_success', 0)} | 两阶段均已正确 |",
        "",
        f"最终链路共有 {final_failures} / {total} 条 Top{k} 未完整覆盖，以下所有归因都只针对这 {final_failures} 条。",
        "",
        "## Gold 数量边界",
        "",
        "| gold 工具数 | query 数 | 单路失败率 | 最终失败率 | 最终失败条数 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in payload["per_gold_count"]:
        lines.append(
            f"| {row['gold_tool_count']} | {row['total']} | {fmt_rate(row['single_failure_rate'])} | {fmt_rate(row['final_failure_rate'])} | {row['final_failures']} |"
        )
    lines.extend(
        [
            "",
            (
                f"当 gold 工具数为 {k} 时，Top{k} 没有任何干扰工具容忍位；该 bucket 是最严格的排序条件。"
                if max_gold_tool_count == k
                else f"本数据最多包含 {max_gold_tool_count} 个 gold；Top{k} 至少保留 {k - max_gold_tool_count} 个干扰工具容忍位。"
            ),
            "",
            "## 残余失败位置",
            "",
            f"| 检查项 | 条数 | 解释 |",
            "|---|---:|---|",
            f"| 所有 gold 已在 reranker Top{payload['candidate_k']} 候选池 | {all_in_candidate} / {final_failures} | 该部分属于候选池内的后续排序失败 |",
            f"| 至少一个 gold 在 reranker Top{payload['candidate_k']} 候选池外 | {outside_candidate} / {final_failures} | 该部分属于一级覆盖或候选截断失败 |",
        ]
    )
    if k < 10:
        lines.append(
            f"| 所有漏召 gold 位于 rank {k + 1}-10 | {all_prefix} / {final_failures} | 仅差前缀排序位置；Top10 仅作诊断，不改变 Top{k} 主指标 |"
        )
    for label in query_band_order:
        if label == f"all_gold_{k + 1}-10" or label not in query_band_counts:
            continue
        lines.append(f"| {query_band_labels[label]} | {query_band_counts[label]} / {final_failures} | 至少一个 required tool 排在更靠后的位置 |")
    lines.extend(
        [
            "",
            "候选池外的 required tool：`"
            + ", ".join(f"{tool}={count}" for tool, count in candidate_coverage_missing_tools)
            + "`。该小类应先审计工具描述和标注边界，不能直接归因于候选深度。",
            "",
            "漏召工具位置按工具次数统计：`" + ", ".join(f"{label}={count}" for label, count in payload["missing_tool_rank_bands"]) + "`。",
            "",
            "## 竞争信号",
            "",
            "下表只统计“required tool 漏召，同时 non-gold tool 落入最终 TopK”的共现次数。它用于挑选 hard negative，不把共现直接写成唯一根因。",
            "",
            "| 漏召工具 <- TopK 竞争工具 | 共现次数 |",
            "|---|---:|",
        ]
    )
    for label, count in payload["top_missing_competitor_pairs"][:8]:
        lines.append(f"| `{label}` | {count} |")
    lines.extend(
        [
            "",
            "## 后续动作与结论边界",
            "",
            f"- 已完成的改进是任务内 reranker 加 source/reranker RRF：它恢复 {transitions.get('single_fail_final_success', 0)} 条单路失败，但也新增 {transitions.get('single_success_final_fail', 0)} 条回归；不能写成所有边界已经解决。",
            f"- 候选池外失败为 {outside_candidate} / {final_failures}；候选池内排序失败为 {all_in_candidate} / {final_failures}。下一轮应按占比决定优先检查一级覆盖/候选深度，还是 coverage-aware 排序目标。",
            f"- 把 {transitions.get('single_success_final_fail', 0)} 条相对单路 Embedding 的回归单列为保护集。",
            "- 训练样本可从上表竞争对构造，但验证必须使用新的表达簇或独立 holdout；冻结 test query 与 corpus 不直接改写。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-preds", type=Path, required=True)
    parser.add_argument("--reranker-preds", type=Path, required=True)
    parser.add_argument("--final-preds", type=Path, required=True)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--curve-ks", type=int, nargs="+", default=[6, 7, 8, 9, 10])
    parser.add_argument("--route-json", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()
    if args.k < 6:
        raise SystemExit("--k must be at least 6 for this project")
    if args.candidate_k < args.k:
        raise SystemExit("--candidate-k must be at least --k")
    curve_ks = sorted(set(args.curve_ks))
    if not curve_ks or curve_ks[0] < 1 or curve_ks[-1] > args.candidate_k:
        raise SystemExit("--curve-ks must be within 1..candidate-k")

    payload = analyze(
        load_jsonl(args.single_preds),
        load_jsonl(args.reranker_preds),
        load_jsonl(args.final_preds),
        args.k,
        args.candidate_k,
        curve_ks,
    )
    payload["single_preds"] = display_path(args.single_preds)
    payload["reranker_preds"] = display_path(args.reranker_preds)
    payload["final_preds"] = display_path(args.final_preds)
    payload["input_sha256"] = {
        "single_preds": sha256_file(args.single_preds),
        "reranker_preds": sha256_file(args.reranker_preds),
        "final_preds": sha256_file(args.final_preds),
    }
    if args.route_json:
        route = json.loads(args.route_json.read_text(encoding="utf-8"))
        payload["canonical_route"] = {
            "path": display_path(args.route_json),
            "sha256": sha256_file(args.route_json),
            "status": route.get("status"),
            "embedding_recipe": route.get("embedding", {}).get("recipe"),
            "reranker_base_model": route.get("reranker", {}).get("base_model"),
            "reranker_negative_recipe": route.get("reranker", {}).get("negative_recipe"),
        }
    write_json(args.output_json, payload)
    print(f"wrote {display_path(args.output_json)}")
    if args.output_md:
        write_markdown(args.output_md, payload)
        print(f"wrote {display_path(args.output_md)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
