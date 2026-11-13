#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize the standalone API-native Assignment3 stress evaluation."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import load_jsonl, split_tools, write_json  # noqa: E402
from finetune.tool_trace_common import sha256_file  # noqa: E402


BASE_VIEWS = ("query-only", "global", "rag-k5", "rag-k10", "rag-k20", "rag-k30")
SFT_VIEWS = ("query-only", "global", "rag-k5", "rag-k10", "rag-k20", "rag-k30")
BASE_STEMS = {view: f"api-native-qwen3-4b-base-{view}" for view in BASE_VIEWS}
SFT_STEMS = {view: f"api-native-sft4_s1-{view}" for view in SFT_VIEWS}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_summary(payload: dict[str, Any]) -> dict[str, Any]:
    overall = payload["overall"]
    retrieval = payload["rag_decomposition"]
    return {
        "set_em": overall["set_em"],
        "trace_em": overall["trace_em"],
        "json_schema_valid_rate": overall["json_schema_valid_rate"],
        "strict_output_valid_rate": overall["strict_output_valid_rate"],
        "json_valid_rate": overall["json_valid_rate"],
        "identifiable_order_accuracy": overall["identifiable_order_accuracy"],
        "identifiable_set_correct_count": overall["identifiable_set_correct_count"],
        "retriever_upper_bound": retrieval["retriever_upper_bound"],
        "conditional_set_accuracy": retrieval["conditional_set_accuracy"],
        "error_counts": payload["error_analysis"]["counts"],
        "format_types": payload["error_analysis"]["format_types"],
        "mean_input_tokens": payload["efficiency"]["input_tokens"]["mean"],
        "tool_count": payload["buckets"]["tool_count"],
        "combo": payload["buckets"]["combo"],
    }


def retrieval_metrics(
    gold_rows: list[dict[str, Any]],
    ranking_rows: list[dict[str, Any]],
    ks: tuple[int, ...] = (5, 10, 20, 30),
) -> dict[str, float]:
    gold_by_id = {
        str(row["id"]): set(split_tools(row["工具"])) for row in gold_rows
    }
    ranking_by_id = {str(row["id"]): row["pred_tools"] for row in ranking_rows}
    if set(gold_by_id) != set(ranking_by_id):
        raise ValueError("ranking ids do not match API-native dev ids")
    metrics: dict[str, float] = {}
    for k in ks:
        all_hits = 0
        recall_sum = 0.0
        for row_id, gold in gold_by_id.items():
            prediction = set(ranking_by_id[row_id][:k])
            all_hits += gold <= prediction
            recall_sum += len(gold & prediction) / len(gold)
        metrics[f"allhit_at_{k}"] = all_hits / len(gold_by_id)
        metrics[f"macro_recall_at_{k}"] = recall_sum / len(gold_by_id)
    return metrics


def retrieval_metrics_by_tool_count(
    gold_rows: list[dict[str, Any]],
    ranking_rows: list[dict[str, Any]],
    ks: tuple[int, ...] = (5, 10, 20, 30),
) -> dict[str, dict[str, dict[str, float | int]]]:
    ranking_by_id = {str(row["id"]): row["pred_tools"] for row in ranking_rows}
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in gold_rows:
        grouped[int(row["tool_count"])].append(row)
    output: dict[str, dict[str, dict[str, float | int]]] = {}
    for k in ks:
        k_rows: dict[str, dict[str, float | int]] = {}
        for tool_count, rows in sorted(grouped.items()):
            all_hits = 0
            recall_sum = 0.0
            for row in rows:
                gold = set(split_tools(row["工具"]))
                prediction = set(ranking_by_id[str(row["id"])][:k])
                all_hits += gold <= prediction
                recall_sum += len(gold & prediction) / len(gold)
            k_rows[str(tool_count)] = {
                "rows": len(rows),
                "allhit": all_hits / len(rows),
                "macro_recall": recall_sum / len(rows),
            }
        output[str(k)] = k_rows
    return output


def ranking_order_stats(
    gold_rows: list[dict[str, Any]],
    ranking_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    ranking_by_id = {str(row["id"]): row for row in ranking_rows}
    counts: dict[int, Counter[str]] = defaultdict(Counter)
    for row in gold_rows:
        gold = split_tools(row["工具"])
        ranking = ranking_by_id[str(row["id"])]["pred_tools"]
        relative = [tool_id for tool_id in ranking if tool_id in set(gold)]
        bucket = counts[len(gold)]
        bucket["rows"] += 1
        bucket["relative_order_exact"] += relative == gold
        bucket["top_n_exact"] += ranking[: len(gold)] == gold
    return {
        str(tool_count): {
            "rows": bucket["rows"],
            "ranking_gold_relative_order_rate": (
                bucket["relative_order_exact"] / bucket["rows"]
            ),
            "ranking_top_n_exact_rate": bucket["top_n_exact"] / bucket["rows"],
            "random_permutation_baseline": 1.0 / math.factorial(tool_count),
        }
        for tool_count, bucket in sorted(counts.items())
    }


def confusion_summary(scored_rows: list[dict[str, Any]]) -> dict[str, Any]:
    omissions: Counter[str] = Counter()
    extras: Counter[str] = Counter()
    substitutions: Counter[str] = Counter()
    for row in scored_rows:
        gold = set(row["gold_tools"])
        prediction = set(row["pred_tools"])
        missing = gold - prediction
        added = prediction - gold
        omissions.update(missing)
        extras.update(added)
        substitutions.update(
            f"{left} -> {right}" for left in missing for right in added
        )
    return {
        "top_omissions": dict(omissions.most_common(10)),
        "top_extras": dict(extras.most_common(10)),
        "top_substitutions": dict(substitutions.most_common(10)),
    }


def prediction_protocol(path: Path) -> dict[str, Any]:
    rows = load_jsonl(path)
    return {
        "rows": len(rows),
        "unique_ids": len({str(row["id"]) for row in rows}),
        "views": sorted({str(row.get("view", "")) for row in rows}),
        "prompt_versions": sorted(
            {str(row.get("prompt_version", "")) for row in rows}
        ),
        "candidate_ks": sorted({int(row.get("candidate_k", 0)) for row in rows}),
        "nonempty_candidate_rows": sum(
            bool(row.get("candidate_tools")) for row in rows
        ),
    }


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-dev", type=Path, required=True)
    parser.add_argument("--api-rankings", type=Path, required=True)
    parser.add_argument("--api-untuned-rankings", type=Path, required=True)
    parser.add_argument("--api-pred-dir", type=Path, required=True)
    parser.add_argument("--isolation-audit", type=Path, required=True)
    parser.add_argument("--generation-audit", type=Path, required=True)
    parser.add_argument("--retrieval-manifest", type=Path, required=True)
    parser.add_argument("--sft-data-audit", type=Path, required=True)
    parser.add_argument("--prompt-length-stats", type=Path, required=True)
    parser.add_argument("--sft-compact", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()

    api_dev = load_jsonl(args.api_dev)
    isolation = load_json(args.isolation_audit)
    generation = load_json(args.generation_audit)
    retrieval = load_json(args.retrieval_manifest)
    sft_data_audit = load_json(args.sft_data_audit)
    prompt_length_stats = load_json(args.prompt_length_stats)
    sft_compact = load_jsonl(args.sft_compact)

    base_metrics = {
        view: metric_summary(
            load_json(args.api_pred_dir / f"{BASE_STEMS[view]}.metrics.json")
        )
        for view in BASE_VIEWS
    }
    sft_metrics = {
        view: metric_summary(
            load_json(args.api_pred_dir / f"{SFT_STEMS[view]}.metrics.json")
        )
        for view in SFT_VIEWS
    }
    tuned_ranking_rows = load_jsonl(args.api_rankings)
    untuned_ranking_rows = load_jsonl(args.api_untuned_rankings)
    tuned_retrieval = retrieval_metrics(api_dev, tuned_ranking_rows)
    untuned_retrieval = retrieval_metrics(api_dev, untuned_ranking_rows)
    tuned_retrieval_by_tool_count = retrieval_metrics_by_tool_count(
        api_dev, tuned_ranking_rows
    )
    for prefix, derived, recorded in (
        ("tuned", tuned_retrieval, retrieval["metrics"]),
        ("untuned", untuned_retrieval, retrieval["untuned_metrics"]),
    ):
        for k in (10, 20, 30):
            key = f"allhit_at_{k}"
            if not math.isclose(derived[key], recorded[key], abs_tol=1e-12):
                raise ValueError(
                    f"{prefix} retrieval manifest mismatch at K={k}: "
                    f"{derived[key]} != {recorded[key]}"
                )
    ranking_order = ranking_order_stats(api_dev, tuned_ranking_rows)
    api_global_scored = load_jsonl(
        args.api_pred_dir / "api-native-sft4_s1-global.scored.jsonl"
    )
    api_global_by_id = {str(row["id"]): row for row in api_global_scored}
    global_train_rows = [row for row in sft_compact if row["view"] == "global"]
    global_train_sequences = {tuple(row["gold_tools"]) for row in global_train_rows}
    all_train_sequences = {tuple(row["gold_tools"]) for row in sft_compact}
    global_tool_occurrences = Counter(
        tool_id for row in global_train_rows for tool_id in row["gold_tools"]
    )
    api_multi = [row for row in api_dev if int(row["tool_count"]) > 1]
    sequence_seen_global = [
        row
        for row in api_multi
        if tuple(split_tools(row["工具"])) in global_train_sequences
    ]
    sequence_unseen_global = [
        row
        for row in api_multi
        if tuple(split_tools(row["工具"])) not in global_train_sequences
    ]

    def set_accuracy(selected: list[dict[str, Any]]) -> float:
        return sum(
            bool(api_global_by_id[str(row["id"])]["set_em"]) for row in selected
        ) / max(1, len(selected))

    tokenizer_views = next(
        value["views"]
        for key, value in prompt_length_stats["tokenizers"].items()
        if key.endswith("/Qwen3-4B")
    )
    training_exposure = {
        "rag_records": sft_data_audit["rag_records"],
        "global_records": sft_data_audit["global_records"],
        "compact_records": sft_data_audit["compact_records"],
        "actual_global_input_token_share": prompt_length_stats[
            "actual_global_input_token_share"
        ],
        "global_mean_input_tokens": tokenizer_views["global"]["input_tokens"]["mean"],
        "global_tools_covered": len(global_tool_occurrences),
        "global_per_tool_occurrence": {
            "min": min(global_tool_occurrences.values()),
            "median": statistics.median(global_tool_occurrences.values()),
            "max": max(global_tool_occurrences.values()),
        },
        "api_multi_exact_sequence_seen_any_sft": sum(
            tuple(split_tools(row["工具"])) in all_train_sequences for row in api_multi
        ),
        "api_multi_exact_sequence_unseen_any_sft": sum(
            tuple(split_tools(row["工具"])) not in all_train_sequences for row in api_multi
        ),
        "api_multi_exact_sequence_seen_global": {
            "rows": len(sequence_seen_global),
            "set_em": set_accuracy(sequence_seen_global),
        },
        "api_multi_exact_sequence_unseen_global": {
            "rows": len(sequence_unseen_global),
            "set_em": set_accuracy(sequence_unseen_global),
        },
    }
    summary = {
        "analysis_version": "assignment3-api-native-stress-analysis-v3",
        "status": "complete",
        "dataset": {
            "generator_model": generation["generator_model"],
            "judge_model": generation["judge_model"],
            "rows": generation["target_count"],
            "strict_generation_ok": generation["strict_ok"],
            "strict_isolation_ok": isolation["strict_ok"],
            "query_templates_used": generation["query_templates_used"],
            "target_plan_contains_queries": generation["target_plan_contains_queries"],
            "training_use_allowed": generation["training_use_allowed"],
        },
        "metrics": {
            "qwen3_4b_base": base_metrics,
            "qwen3_4b_sft_checkpoint_878": sft_metrics,
        },
        "prediction_protocols": {
            "qwen3_4b_base": {
                view: prediction_protocol(
                    args.api_pred_dir / f"{BASE_STEMS[view]}.jsonl"
                )
                for view in BASE_VIEWS
            },
            "qwen3_4b_sft_checkpoint_878": {
                view: prediction_protocol(
                    args.api_pred_dir / f"{SFT_STEMS[view]}.jsonl"
                )
                for view in SFT_VIEWS
            },
        },
        "retrieval": {
            "tuned": tuned_retrieval,
            "untuned": untuned_retrieval,
            "tuned_by_tool_count": tuned_retrieval_by_tool_count,
        },
        "ranking_order_diagnostic": ranking_order,
        "training_exposure": training_exposure,
        "global_confusions": confusion_summary(api_global_scored),
        "isolation": {
            "query_overlap": isolation["query_overlap"],
            "trace_id_overlap": isolation["trace_id_overlap"],
            "expression_phrase_substring_overlap": isolation[
                "expression_phrase_substring_overlap"
            ],
            "literal_rule_diagnostics": isolation["literal_rule_diagnostics"],
            "internal_leakage": isolation["internal_leakage"],
        },
        "hashes": {
            str(path): sha256_file(path)
            for path in (
                args.api_dev,
                args.api_rankings,
                args.api_untuned_rankings,
                args.isolation_audit,
                args.generation_audit,
                args.retrieval_manifest,
                args.sft_data_audit,
                args.prompt_length_stats,
                args.sft_compact,
            )
        },
    }
    for stems in (BASE_STEMS, SFT_STEMS):
        for stem in stems.values():
            for suffix in (".jsonl", ".metrics.json", ".scored.jsonl"):
                path = args.api_pred_dir / f"{stem}{suffix}"
                summary["hashes"][str(path)] = sha256_file(path)
    write_json(args.summary_out, summary)

    table_rows = []
    for model_label, views, metrics in (
        ("Qwen3-4B base", BASE_VIEWS, base_metrics),
        ("Qwen3-4B SFT checkpoint-878", SFT_VIEWS, sft_metrics),
    ):
        for view in views:
            metric = metrics[view]
            label = "full-catalog" if view == "global" else view
            is_rag = view.startswith("rag-")
            upper = (
                percent(metric["retriever_upper_bound"]) if is_rag else "—"
            )
            conditional = (
                percent(metric["conditional_set_accuracy"]) if is_rag else "—"
            )
            table_rows.append(
                f"| {model_label} | {label} | "
                f"{percent(metric['set_em'])} | {percent(metric['trace_em'])} | "
                f"{percent(metric['json_schema_valid_rate'])} | "
                f"{percent(metric['strict_output_valid_rate'])} | "
                f"{metric['mean_input_tokens']:.1f} | {upper} | {conditional} |"
            )

    order_rows = []
    for tool_count in range(2, 7):
        row = ranking_order[str(tool_count)]
        order_rows.append(
            f"| {tool_count} | {percent(row['ranking_gold_relative_order_rate'])} | "
            f"{percent(row['random_permutation_baseline'])} |"
        )
    retrieval_count_rows = []
    for tool_count in range(1, 7):
        retrieval_count_rows.append(
            f"| {tool_count} | "
            f"{percent(tuned_retrieval_by_tool_count['5'][str(tool_count)]['allhit'])} | "
            f"{percent(tuned_retrieval_by_tool_count['10'][str(tool_count)]['allhit'])} | "
            f"{percent(tuned_retrieval_by_tool_count['20'][str(tool_count)]['allhit'])} |"
        )

    literal = isolation["literal_rule_diagnostics"]
    base_query_only = base_metrics["query-only"]
    base_global = base_metrics["global"]
    base_k20 = base_metrics["rag-k20"]
    base_k30 = base_metrics["rag-k30"]
    sft_query_only = sft_metrics["query-only"]
    full_catalog = sft_metrics["global"]
    k5 = sft_metrics["rag-k5"]
    k10 = sft_metrics["rag-k10"]
    k20 = sft_metrics["rag-k20"]
    k30 = sft_metrics["rag-k30"]
    phrase_overlap = isolation["expression_phrase_substring_overlap"]
    exposure = summary["training_exposure"]
    k5_total_misses = round(
        len(api_dev) * (1.0 - tuned_retrieval["allhit_at_5"])
    )
    k5_forced_misses = sum(int(row["tool_count"]) > 5 for row in api_dev)
    report = f"""# Assignment3 API 原生压力测试（含真正 Wo-RAG 消融）

> 日期：2026-07-26  
> 唯一开发评测集：API 原生 2400 条。结论仅适用于固定 100 工具的合成闭集。

## 结论

API 原生集与训练集、冻结测试集的完整 query 和 trace id 重叠均为 0，可作为独立的开发压力集使用；它只用于评测，未混入 SFT 或 DPO 数据。

真正不提供 catalog 的 query-only（闭卷 Wo-RAG）结果并不高：Qwen3-4B base / SFT 的 SetEM 只有 {percent(base_query_only['set_em'])} / {percent(sft_query_only['set_em'])}。此前 {percent(full_catalog['set_em'])} 的 SFT 结果来自 full-catalog：虽然没有检索模块，但 prompt 中仍提供全部 100 个工具的 ID 和描述。它应叫“无检索裁剪的全 catalog 开卷选择”，不能叫闭卷 Wo-RAG。

SFT 在 RAG@5/10/20/30 的 SetEM 为 {percent(k5['set_em'])} / {percent(k10['set_em'])} / {percent(k20['set_em'])} / {percent(k30['set_em'])}。RAG@5 和 RAG@10 的主要瓶颈是检索覆盖；在 gold 全部进入候选集后，条件规划准确率仍有 {percent(k5['conditional_set_accuracy'])} / {percent(k10['conditional_set_accuracy'])}。

基座模型在 RAG@20/30 达到 {percent(base_k20['set_em'])} / {percent(base_k30['set_em'])}，高于 full-catalog 的 {percent(base_global['set_em'])}；相关性排序与候选裁剪能减少 100 工具全量 prompt 的干扰。SFT 在覆盖率足够后对候选规模不再敏感，full-catalog、RAG@20 和 RAG@30 聚集在 {percent(full_catalog['set_em'])}～{percent(max(k20['set_em'], k30['set_em']))}。

## 数据构造与隔离

- 2400 条均由 GPT-5.5 从无 query 的目标能力计划自由生成，不使用 query 模板；GPT-5.4 独立核验意图完整性、操作强度、步数和顺序，2400/2400 通过。
- 单工具 1200 条；工具数 2～6 各 240 条。全部 100 个工具均被覆盖。
- 与 train、冻结 test 的归一化完整 query 重叠均为 0；trace id 重叠均为 0。
- 命中 train/test 完整表达短语的 API query 分别为 {phrase_overlap['train']['api_rows_containing_full_reference_phrase']} / {phrase_overlap['test']['api_rows_containing_full_reference_phrase']}。
- 未出现工具 ID、内部实现术语或完整 catalog 能力描述。
- 完整能力描述字面规则的 SetEM 为 {percent(literal['cleaned_full_description']['set_em'])}；catalog 表面词规则的 SetEM 为 {percent(literal['catalog_surface_terms']['set_em'])}。

## 三种协议必须分开命名

- `query-only`：只给用户问题和通用 JSON 输出约束；不提供候选工具、工具 ID 或能力描述。这才是本报告中的 Wo-RAG/闭卷诊断。
- `full-catalog`：给出全部 100 个工具，省略检索裁剪，但仍有完整 ontology grounding。
- `RAG@K`：只给检索得到的 K 个候选工具；端到端分数同时受检索覆盖和候选内规划影响。

query-only 使用独立的 `tool-trace-query-only-v1` prompt。2400 条预测的 `candidate_k` 均为 0，候选数组均为空；其他视图沿用 `tool-trace-v1`。

## 评测结果

所有行固定使用 Qwen3-4B 和贪心解码；SFT 行使用 `tooltrace-sft4_s1/checkpoint-878`。

| 模型 | 视图 | SetEM | TraceEM | JSON 结构合法 | 严格输出合法 | 平均输入 token | 检索 AllHit 上限 | 条件规划准确率 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table_rows)}

微调 embedding + reranker 融合检索的 AllHit@5/10/20/30 为 {percent(tuned_retrieval['allhit_at_5'])} / {percent(tuned_retrieval['allhit_at_10'])} / {percent(tuned_retrieval['allhit_at_20'])} / {percent(tuned_retrieval['allhit_at_30'])}；未微调检索为 {percent(untuned_retrieval['allhit_at_5'])} / {percent(untuned_retrieval['allhit_at_10'])} / {percent(untuned_retrieval['allhit_at_20'])} / {percent(untuned_retrieval['allhit_at_30'])}。

RAG 的严格端到端 SetEM 可分解为：

$$
\\text{{SetEM}} = \\text{{AllHit@K}} \\times P(\\text{{SetEM}} \\mid \\text{{AllHit@K}})
$$

例如 SFT RAG@5 为 {percent(k5['retriever_upper_bound'])} × {percent(k5['conditional_set_accuracy'])} = {percent(k5['set_em'])}；RAG@10 为 {percent(k10['retriever_upper_bound'])} × {percent(k10['conditional_set_accuracy'])} = {percent(k10['set_em'])}。这说明 K=5/10 的下降主要不是工具链规划突然失效，而是正确工具没有全部进入候选集。

K=5 还存在一个协议硬上限：数据中有 {k5_forced_misses} 条六工具样本，而 5 个候选不可能覆盖 6 个 gold 工具。这 {k5_forced_misses} 条占 K=5 共 {k5_total_misses} 条检索失败的 {percent(k5_forced_misses / k5_total_misses)}。按工具数拆开后，链越长，AllHit 越低：

| gold 工具数 | AllHit@5 | AllHit@10 | AllHit@20 |
|---:|---:|---:|---:|
{chr(10).join(retrieval_count_rows)}

## 为什么此前的“SFT Wo-RAG”会虚高

1. **协议命名混淆。** full-catalog 是 retrieval-free，但不是 catalog-free。它平均约 {full_catalog['mean_input_tokens']:.0f} 个输入 token，直接展示固定 100 工具的语义信息；query-only 只有约 {sft_query_only['mean_input_tokens']:.0f} 个输入 token。
2. SFT 包含 {exposure['rag_records']} 条 RAG 记录和 {exposure['global_records']} 条 full-catalog 记录；后者占实际输入 token 的 {percent(exposure['actual_global_input_token_share'])}。
3. full-catalog 训练覆盖全部 {exposure['global_tools_covered']} 个工具；每个工具作为 gold 至少出现 {exposure['global_per_tool_occurrence']['min']} 次，中位数 {exposure['global_per_tool_occurrence']['median']:.0f} 次。
4. 工具 ID 自带语义，且 API query 仍由同一 catalog 的能力边界生成。这测试了表达变化，但没有测试新 ontology、工具版本变化或开放世界泛化。
5. API 多工具样本中，{exposure['api_multi_exact_sequence_seen_any_sft']} / {len(api_multi)} 条完整有序链曾在某个 SFT 视图出现。仅看 full-catalog 训练，已见链 {exposure['api_multi_exact_sequence_seen_global']['rows']} 条的 SetEM 为 {percent(exposure['api_multi_exact_sequence_seen_global']['set_em'])}，未见链 {exposure['api_multi_exact_sequence_unseen_global']['rows']} 条仍为 {percent(exposure['api_multi_exact_sequence_unseen_global']['set_em'])}。
6. 同一 full-catalog prompt 下，基座模型 SetEM 仅 {percent(base_global['set_em'])}，说明高分主要来自强任务内监督，而不是该任务天然简单到任意模型都能完成。

query-only 中，base / SFT 的 JSON 结构合法率其实有 {percent(base_query_only['json_schema_valid_rate'])} / {percent(sft_query_only['json_schema_valid_rate'])}，但严格输出合法率只有 {percent(base_query_only['strict_output_valid_rate'])} / {percent(sft_query_only['strict_output_valid_rate'])}；其中分别有 {base_query_only['format_types'].get('unknown_tool', 0)} / {sft_query_only['format_types'].get('unknown_tool', 0)} 条输出因不存在的工具 ID 被严格判错。因此失败根因不是模型不会输出 JSON，而是没有稳定闭卷恢复 100 个工具 ID。当前 SFT 主要学会了“在给定 ontology 中选择与排序”。也要注意，训练数据没有 query-only 视图，因此这个消融是有意的分布外压力诊断，不能单独解释为模型不理解用户语义。

## 顺序轨迹口径

TraceEM 检查有序工具 ID 列表，但本集要求 query 按目标顺序自然表达，因此集合正确时几乎总能同时满足顺序正确。RAG 候选中 gold 工具的相对顺序如下，整体没有直接提供可靠执行顺序：

| 工具数 | 检索相对顺序命中 | 随机排列基线 |
|---:|---:|---:|
{chr(10).join(order_rows)}

当前“轨迹”仍不含参数、工具返回、状态变化、失败重试或 observation-conditioned 分支，不能外推为真实 Agent 的端到端执行准确率。

## 下一步

在保持 API-only 开发集不变的前提下，下一阶段应把“静态有序工具 ID”升级为 observation-conditioned 系统评测：实际执行工具、注入返回结果与失败，再检查后续动作是否按状态自适应。该阶段才能回答“模型是否真正会走工具轨迹”，而不只是“能否从候选表中复原目标链”。

## 可复现产物

- API 数据：`{args.api_dev}`
- 生成审计：`{args.generation_audit}`
- 隔离审计：`{args.isolation_audit}`
- 检索清单：`{args.retrieval_manifest}`
- tuned/untuned 完整排序：`{args.api_rankings}` / `{args.api_untuned_rankings}`
- 指标目录：`{args.api_pred_dir}`
- 结构化汇总：`{args.summary_out}`
"""
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report_out.with_suffix(args.report_out.suffix + ".tmp")
    temporary.write_text(report, encoding="utf-8")
    temporary.replace(args.report_out)
    print(f"api-native analysis complete -> {args.summary_out}, {args.report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
