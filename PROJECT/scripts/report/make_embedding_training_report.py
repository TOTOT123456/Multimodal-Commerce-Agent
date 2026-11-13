#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render the archived pre-reselection embedding training diagnostic."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import CATALOG_PATH, PROJECT_ROOT, RAG_DIR, REPORT_DIR, load_jsonl, read_json, split_tools
from make_embedding_negative_report import build_report as build_explicit_negative_section


BASELINE_METRICS = RAG_DIR / "_aux" / "rag_baseline" / "metrics.json"
BASELINE_PREDS = RAG_DIR / "_aux" / "rag_baseline" / "preds"
EMBEDDING_STATS = RAG_DIR / "_aux" / "embedding_dataset_stats.json"
EXPERIMENT_CONFIG = PROJECT_ROOT / "configs" / "embedding_finetune_experiments.json"
FINETUNE_DIR = RAG_DIR / "_aux" / "embedding_finetune"
GENERATION_AUDIT = PROJECT_ROOT / "data" / "tool_call_trace" / "_aux" / "generation_leakage_audit.json"
CONTEXT_PROBE_AUDIT = PROJECT_ROOT / "data" / "tool_call_trace" / "_aux" / "context_isolated_probe" / "generation_leakage_audit.json"
OUT = RAG_DIR / "_aux" / "embedding_finetune" / "pre_reselection_training_diagnostic.md"
K_VALUES = [6, 7, 8, 9, 10]


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "待跑"
    if isinstance(value, float):
        if value != 0 and abs(value) < 10 ** (-digits):
            return f"{value:.2e}"
        return f"{value:.{digits}f}"
    return str(value)


def metric_at(payload: dict[str, Any], k: int, name: str) -> float | None:
    return payload.get("metrics", {}).get(f"@{k}", {}).get(name)


def mrr(payload: dict[str, Any]) -> float | None:
    return payload.get("metrics", {}).get("overall", {}).get("MRR")


def timing_value(payload: dict[str, Any], key: str) -> float | None:
    value = payload.get("timing", {}).get(key)
    return float(value) if isinstance(value, (int, float)) else None


def safe_method_file(method: str) -> str:
    return method.replace(":", "_").replace("/", "_") + ".jsonl"


def baseline_model_rows(metrics: dict[str, Any]) -> list[str]:
    rows = []
    for method in ["vector:bge-base", "vector:bge-large", "vector:qwen3-embedding-0.6b"]:
        payload = metrics.get(method, {})
        model = method.replace("vector:", "")
        allhit10 = metric_at(payload, 10, "AllHit@K")
        allhit30 = metric_at(payload, 30, "AllHit@K")
        encode_ms = timing_value(payload, "encode_avg_ms")
        score = (allhit10 / encode_ms) if allhit10 is not None and encode_ms else None
        rows.append(
            f"| {model} | {fmt(allhit10)} | {fmt(allhit30)} | {fmt(mrr(payload))} | "
            f"{fmt(encode_ms)} | {fmt(score)} |"
        )
    return rows


def delta_pp(value: float | None, baseline: float | None) -> str:
    if value is None or baseline is None:
        return "待跑"
    return f"{(value - baseline) * 100:+.2f}"


def embedding_dataset_lines(stats: dict[str, Any]) -> list[str]:
    if not stats:
        return ["- 数据集统计待生成。"]

    policy = stats.get("policy", {})
    train = stats.get("train", {})
    test = stats.get("test", {})
    outputs = stats.get("outputs", {})
    return [
        "100 工具版保留 24k/6k 工具轨迹切分中的全部正样本，并按 query 级别构造类型化负样本。",
        "",
        f"- 采样策略：单意图每条 query {policy.get('single_neg_per_query')} 个负样本，`同域:跨域 = 1:1`；多意图每条 query {policy.get('multi_neg_per_query')} 个负样本，`部分替代:新增无关 = 1:1`。",
        f"- 规模说明：{policy.get('scale_note', '')}",
        "",
        "| Split | Query数 | Pair总数 | 正样本pair | 负样本pair | 负/正比例 | Grouped文件 | Pair文件 |",
        "|---|---:|---:|---:|---:|---:|---|---|",
        f"| train | {train.get('queries')} | {train.get('pairs')} | {train.get('positive_pairs')} | {train.get('negative_pairs')} | {train.get('negative_to_positive_ratio')} | `{display_path(outputs.get('train_grouped'))}` | `{display_path(outputs.get('train_pairs'))}` |",
        f"| test | {test.get('queries')} | {test.get('pairs')} | {test.get('positive_pairs')} | {test.get('negative_pairs')} | {test.get('negative_to_positive_ratio')} | `{display_path(outputs.get('test_grouped'))}` | `{display_path(outputs.get('test_pairs'))}` |",
        "",
        "| Split | single_same_domain | single_cross_domain | multi_partial_replacement | multi_add_irrelevant |",
        "|---|---:|---:|---:|---:|",
        f"| train | {train.get('negative_types', {}).get('single_same_domain', 0)} | {train.get('negative_types', {}).get('single_cross_domain', 0)} | {train.get('negative_types', {}).get('multi_partial_replacement', 0)} | {train.get('negative_types', {}).get('multi_add_irrelevant', 0)} |",
        f"| test | {test.get('negative_types', {}).get('single_same_domain', 0)} | {test.get('negative_types', {}).get('single_cross_domain', 0)} | {test.get('negative_types', {}).get('multi_partial_replacement', 0)} | {test.get('negative_types', {}).get('multi_add_irrelevant', 0)} |",
        "",
        "输出同时保留 pair-level JSONL 和 grouped 多负样本 JSONL：前者兼容分类/重排训练，后者适合 FlagEmbedding/Qwen embedding 的 `query + pos + neg` 训练格式。",
        "原 2026-07-09 MNRL 主实验只展开 `query + pos` 正样本对。2026-07-30 起训练入口支持分别选择同领域/部分替代、跨领域/新增无关或全部显式负例；消融结果统一记录在 `PROJECT/reports/03_向量训练.md`。",
        "",
        "输出文件：",
        "",
        "```text",
        "PROJECT/data/rag/embedding_train.jsonl",
        "PROJECT/data/rag/embedding_test.jsonl",
        "PROJECT/data/rag/embedding_grouped_train.jsonl",
        "PROJECT/data/rag/embedding_grouped_test.jsonl",
        "PROJECT/data/rag/_aux/embedding_dataset_stats.json",
        "```",
    ]


def load_runs(current_stats: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    if not FINETUNE_DIR.exists():
        return runs
    current_train = (current_stats or {}).get("train", {})
    current_source_rows = current_train.get("queries")
    current_positive_pairs = current_train.get("positive_pairs")
    for run_dir in sorted(path for path in FINETUNE_DIR.iterdir() if path.is_dir()):
        metrics_path = run_dir / "retrieval" / "metrics.json"
        manifest_path = run_dir / "model" / "training_manifest.json"
        if not metrics_path.exists():
            continue
        metrics = read_json(metrics_path, {})
        manifest = read_json(manifest_path, {})
        train_stats = manifest.get("train_stats", {})
        if current_source_rows is not None and train_stats.get("source_rows") != current_source_rows:
            continue
        if current_positive_pairs is not None and train_stats.get("positive_pairs") != current_positive_pairs:
            continue
        vector_methods = [name for name in metrics if name.startswith("vector:")]
        if not vector_methods:
            continue
        method = vector_methods[0]
        runs.append(
            {
                "run_name": run_dir.name,
                "run_dir": run_dir,
                "method": method,
                "metrics": metrics,
                "payload": metrics[method],
                "manifest": manifest,
                "preds": run_dir / "retrieval" / "preds" / safe_method_file(method),
            }
        )
    return runs


def run_summary_rows(runs: list[dict[str, Any]], baseline_allhit10: float | None) -> list[str]:
    if not runs:
        return ["| 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 |"]
    rows = []
    for run in runs:
        manifest = run["manifest"]
        payload = run["payload"]
        actual_epoch = manifest.get("train_metrics", {}).get("epoch")
        allhit10 = metric_at(payload, 10, "AllHit@K")
        rows.append(
            f"| {run['run_name']} | {fmt(manifest.get('epochs'))} | {fmt(actual_epoch)} | {fmt(manifest.get('learning_rate'))} | "
            f"{fmt(manifest.get('effective_batch_size'))} | {fmt(allhit10)} | {delta_pp(allhit10, baseline_allhit10)} | "
            f"{fmt(metric_at(payload, 30, 'AllHit@K'))} | {fmt(mrr(payload))} |"
        )
    return rows


def topk_rows(metrics: dict[str, Any], runs: list[dict[str, Any]]) -> list[str]:
    rows = []
    baseline_payload = metrics.get("vector:bge-base")
    if baseline_payload:
        for k in K_VALUES:
            rows.append(
                f"| baseline:bge-base | {k} | {fmt(metric_at(baseline_payload, k, 'Hit@K'))} | "
                f"{fmt(metric_at(baseline_payload, k, 'AllHit@K'))} | {fmt(metric_at(baseline_payload, k, 'Recall@K'))} | "
                f"{fmt(metric_at(baseline_payload, k, 'Precision@K'))} | {fmt(mrr(baseline_payload))} |"
            )
    for run in runs:
        payload = run["payload"]
        for k in K_VALUES:
            rows.append(
                f"| {run['run_name']} | {k} | {fmt(metric_at(payload, k, 'Hit@K'))} | "
                f"{fmt(metric_at(payload, k, 'AllHit@K'))} | {fmt(metric_at(payload, k, 'Recall@K'))} | "
                f"{fmt(metric_at(payload, k, 'Precision@K'))} | {fmt(mrr(payload))} |"
            )
    return rows or ["| 待跑 |  |  |  |  |  |  |"]


def generation_metric_at(generation_audit: dict[str, Any], retriever: str, k: int, metric: str) -> float | None:
    return (
        generation_audit.get("rule_retrievers", {})
        .get(retriever, {})
        .get(f"@{k}", {})
        .get(metric)
    )


def context_bucket_metric_at(generation_audit: dict[str, Any], bucket: str, k: int, metric: str) -> float | None:
    return (
        generation_audit.get("rule_retrievers", {})
        .get("context_skeleton_matcher_by_bucket", {})
        .get(bucket, {})
        .get(f"@{k}", {})
        .get(metric)
    )


def improvement_audit_lines(generation_audit: dict[str, Any], probe_audit: dict[str, Any]) -> list[str]:
    overlap = generation_audit.get("context_skeleton_overlap", {})
    by_bucket = overlap.get("by_bucket", {})
    lines = [
        "本次 embedding 微调提升很大，主要来自两个因素：一是任务是 100 工具闭集检索，训练阶段每个工具都有大量 `query -> 工具文档` 正样本；二是 zero-shot BGE 不熟悉本合成数据里的改写说法，微调后专门学到了这些说法到工具描述的映射。",
        "",
        "但 2026-07-08 复核发现，当前 split 还存在非意图上下文骨架复用风险：train/test 的表达短语本身隔离了，但商品、城市、预算、模板位置等上下文由同一 deterministic index 生成。把表达短语替换成 `{PHRASE}` 后，仅靠 context skeleton 的规则匹配器就能在 overall 上取得较高分，尤其单意图子集明显偏高。",
        "",
        "| 检查项 | 结果 | 解释 |",
        "|---|---:|---|",
        f"| strict_ok | `{generation_audit.get('strict_ok', '待审计')}` | `ok` 仍表示旧短语/精确 query 泄漏未复现；`strict_ok=False` 表示新增 context skeleton 风险未通过。 |",
        f"| context signatures seen in train | {overlap.get('context_signatures_seen_in_train', '待审计')} | test context skeleton 与 train skeleton 的唯一签名交集。 |",
        f"| test rows with context seen | {fmt(overlap.get('test_rows_with_context_seen_in_train_rate'))} | 测试样本中，去掉表达短语后上下文骨架曾在 train 出现的比例。 |",
        f"| exact tool sequence seen | {fmt(overlap.get('test_rows_with_exact_tool_sequence_seen_rate'))} | 在同一 context skeleton 下，测试工具序列已在 train 出现的比例。 |",
        f"| context matcher AllHit@10 | {fmt(generation_metric_at(generation_audit, 'context_skeleton_matcher', 10, 'AllHit@K'))} | 不看真实表达短语，只用上下文骨架投票得到的 overall Top10 完整覆盖率。 |",
        f"| context matcher single AllHit@10 | {fmt(context_bucket_metric_at(generation_audit, 'single', 10, 'AllHit@K'))} | 单意图受影响最重，说明 single split 当前不够干净。 |",
        f"| context matcher multi_seen AllHit@10 | {fmt(context_bucket_metric_at(generation_audit, 'multi_seen_combo', 10, 'AllHit@K'))} | seen combo 多意图中，context skeleton 仍有少量预测力。 |",
        f"| context matcher multi_unseen AllHit@10 | {fmt(context_bucket_metric_at(generation_audit, 'multi_unseen_combo', 10, 'AllHit@K'))} | unseen combo 多意图受该问题影响较小。 |",
        "",
        f"按 bucket 看，context skeleton 已见条数为：single={fmt(by_bucket.get('single', {}).get('test_rows_with_context_seen_in_train'))}，multi_seen={fmt(by_bucket.get('multi_seen_combo', {}).get('test_rows_with_context_seen_in_train'))}，multi_unseen={fmt(by_bucket.get('multi_unseen_combo', {}).get('test_rows_with_context_seen_in_train'))}。",
        "",
        "因此，当前 `bge-base-mnrl-e1-lr2e-5-bs64` 的 `AllHit@10=0.9412` 可以作为“同生成器、闭集、表达短语隔离”条件下的上限实验结果，但不能写成真实线上泛化结论。后续更可信的推进路线是：重生成数据时给 train/test 使用不同 context salt，新增 context skeleton 审计为主审计项，然后重跑 baseline、embedding 微调和报告。",
    ]
    if probe_audit:
        lines.extend(
            [
                "",
                "已做一个不覆盖冻结集的修复探针：使用 `generate_diverse_tool_trace.py --test-context-salt 100003` 临时生成 probe split，再跑同一审计；保留的审计证据为 `PROJECT/data/tool_call_trace/_aux/context_isolated_probe/generation_leakage_audit.json`。该探针结果为：",
                "",
                "| Probe检查项 | 结果 |",
                "|---|---:|",
                f"| strict_ok | `{probe_audit.get('strict_ok', '待审计')}` |",
                f"| context matcher AllHit@10 | {fmt(generation_metric_at(probe_audit, 'context_skeleton_matcher', 10, 'AllHit@K'))} |",
                f"| context matcher single AllHit@10 | {fmt(context_bucket_metric_at(probe_audit, 'single', 10, 'AllHit@K'))} |",
                f"| test rows with context seen | {fmt(probe_audit.get('context_skeleton_overlap', {}).get('test_rows_with_context_seen_in_train_rate'))} |",
                "",
                "探针说明：这只验证生成器修复方向，没有提升为新的冻结集，也没有重跑 GPU 指标；当前正式指标仍对应原冻结集。",
            ]
        )
    return lines


def load_tool_names() -> dict[str, str]:
    return {row["tool_id"]: row.get("name_zh", row["tool_id"]) for row in load_jsonl(CATALOG_PATH)}


def analyze_badcases(preds: Path, k: int = 10, sample_limit: int = 8) -> dict[str, Any]:
    if not preds.exists():
        return {"available": False, "preds": str(preds)}

    tool_names = load_tool_names()
    rows = load_jsonl(preds)
    failures = []
    missing = Counter()
    competitors = Counter()
    by_tool_count = Counter()
    for row in rows:
        gold = set(split_tools(row.get("gold_tools")))
        pred = split_tools(row.get("pred_tools"))[:k]
        pred_set = set(pred)
        if gold and gold <= pred_set:
            continue
        miss = sorted(gold - pred_set)
        extra = [tool for tool in pred if tool not in gold]
        missing.update(miss)
        competitors.update(extra[:3])
        by_tool_count[len(gold)] += 1
        if len(failures) < sample_limit:
            failures.append(
                {
                    "query": row.get("query", ""),
                    "gold": [f"{tool}({tool_names.get(tool, tool)})" for tool in sorted(gold)],
                    "topk": [f"{tool}({tool_names.get(tool, tool)})" for tool in pred[:k]],
                    "missing": [f"{tool}({tool_names.get(tool, tool)})" for tool in miss],
                }
            )
    return {
        "available": True,
        "preds": str(preds),
        "total": len(rows),
        "failures": len(rows) - sum(1 for row in rows if set(split_tools(row.get("gold_tools"))) <= set(split_tools(row.get("pred_tools"))[:k])),
        "by_tool_count": dict(sorted(by_tool_count.items())),
        "top_missing": missing.most_common(10),
        "top_competitor": competitors.most_common(10),
        "samples": failures,
    }


def best_badcase_target(metrics: dict[str, Any], runs: list[dict[str, Any]]) -> tuple[str, Path]:
    if runs:
        best = max(runs, key=lambda run: metric_at(run["payload"], 10, "AllHit@K") or -1.0)
        return best["run_name"], best["preds"]
    return "baseline:bge-base", BASELINE_PREDS / "vector_bge-base.jsonl"


def badcase_lines(name: str, analysis: dict[str, Any], k: int) -> list[str]:
    if not analysis.get("available"):
        return [f"- `{name}` 的预测文件不存在：`{analysis.get('preds')}`，badcase 待跑后生成。"]
    lines = [
        f"- 分析对象：`{name}`，预测文件 `{display_path(analysis['preds'])}`。",
        f"- Top{k} 未完整覆盖：{analysis['failures']} / {analysis['total']}。",
        f"- 失败样本按 gold 工具数分布：{analysis['by_tool_count']}。",
        f"- 高频漏召工具：{analysis['top_missing'][:5]}。",
        f"- 高频 Top{k} 竞争工具（非 gold，不单独视为错误）：{analysis['top_competitor'][:5]}。",
        "",
        f"| Query | Gold | Top{k} | Missing |",
        "|---|---|---|---|",
    ]
    for item in analysis["samples"]:
        query = str(item["query"]).replace("|", "\\|")
        lines.append(
            f"| {query} | {'<br>'.join(item['gold'])} | {'<br>'.join(item['topk'])} | {'<br>'.join(item['missing'])} |"
        )
    return lines


def display_path(path: Any) -> str:
    text = str(path)
    repo_prefix = str(PROJECT_ROOT.parent) + "/"
    return text.replace(repo_prefix, "", 1) if text.startswith(repo_prefix) else text


def build_report() -> list[str]:
    metrics = read_json(BASELINE_METRICS, {})
    config = read_json(EXPERIMENT_CONFIG, {})
    stats = read_json(EMBEDDING_STATS, {})
    generation_audit = read_json(GENERATION_AUDIT, {})
    probe_audit = read_json(CONTEXT_PROBE_AUDIT, {})
    runs = load_runs(stats)
    selected = config.get("selected_model", {})
    fixed = config.get("fixed_hyperparameters", {})
    sweep = config.get("sweep", [])
    explicit_ablation = config.get("explicit_negative_ablation", {})
    train = stats.get("train", {})
    test = stats.get("test", {})
    badcase_name, badcase_path = best_badcase_target(metrics, runs)
    badcase_k = 6
    badcases = analyze_badcases(badcase_path, k=badcase_k)
    generation_phrase_matcher10 = (
        generation_audit.get("rule_retrievers", {})
        .get("train_expression_phrase_matcher", {})
        .get("@10", {})
        .get("AllHit@K")
    )
    baseline_bge_payload = metrics.get("vector:bge-base", {})
    baseline_bge_allhit10 = metric_at(baseline_bge_payload, 10, "AllHit@K") if baseline_bge_payload else None

    lines = [
        "# 向量训练",
        "",
        "> 日期：2026-07-07；2026-07-08 补充 embedding 提升原因与 context skeleton 风险复核；2026-07-30 补充显式负例消融",
        "> 任务：100 工具表达簇隔离数据集上的向量模型训练。",
        "> 主指标：多工具检索使用严格 `AllHit@K` 作为 TopK 准确率，即 TopK 必须完整覆盖该 query 的全部 gold tools。",
        "",
        "## 1. 数据集构建与 100 工具版口径",
        "",
    ]
    lines.extend(embedding_dataset_lines(stats))
    lines += [
        "",
        "评测口径：",
        "",
        "- 100 工具版 gold 工具最多 6 个，且测试集单意图/多意图各 3000 条；报告从 K=6 开始保留 K=6/7/8/9/10，同时建议继续看 K=15/20/30 观察召回上限。",
        "- 本阶段不修改冻结测试集 `PROJECT/data/tool_call_trace/test.jsonl`，也不把测试集混入训练。",
        "",
        "## 2. 合理性复核",
        "",
        "本轮已重构数据并重新复核泄漏风险。旧 `HARD_PHRASES` 短语级泄漏未在新数据上复现；本报告只载入 train/test 统计与当前重构数据一致的 embedding 微调 run，旧 run 因数据统计不匹配会被过滤。",
        "",
        f"- 完整 query 归一化重叠：{generation_audit.get('exact_normalized_query_overlap', '待审计')}；trace id 重叠：{generation_audit.get('trace_id_overlap', '待审计')}。",
        f"- 测试工具组合已在训练集出现：{generation_audit.get('tool_combinations', {}).get('test_rows_with_combination_seen_in_train', '待审计')} / {test.get('queries', '待审计')}；unique combo {generation_audit.get('tool_combinations', {}).get('test_unique_combinations_seen_in_train', '待审计')} / {generation_audit.get('tool_combinations', {}).get('test_unique_combinations', '待审计')}。",
        f"- 当前表达簇审计：测试 gold 实例中表达短语已在训练集出现 {generation_audit.get('expression_cluster_overlap', {}).get('test_gold_instances_phrase_seen_in_train', '待审计')} / {generation_audit.get('expression_cluster_overlap', {}).get('test_gold_instances', '待审计')}。",
        f"- context skeleton matcher `AllHit@10={fmt(generation_audit.get('rule_retrievers', {}).get('context_skeleton_matcher', {}).get('@10', {}).get('AllHit@K'))}`；生成泄漏审计 `strict_ok={generation_audit.get('strict_ok', '待审计')}`。",
        f"- 新表达簇审计：`ok={generation_audit.get('ok', '待审计')}`；train expression phrase matcher `AllHit@10={fmt(generation_phrase_matcher10)}`；test gold instance phrase seen in train rate={fmt(generation_audit.get('expression_cluster_overlap', {}).get('test_gold_instances_phrase_seen_in_train_rate'))}。",
        "",
        "因此，本报告只保留新数据上的 baseline 与已通过统计校验的微调结果；未落盘或统计不匹配的 run 不进入结论。",
        "",
        "## 3. 模型选择",
        "",
        f"首轮微调选择 `{selected.get('model_name_or_path', 'BAAI/bge-base-zh-v1.5')}`。理由：{selected.get('reason', '')}",
        "",
        "| Baseline向量模型 | AllHit@10 | AllHit@30 | MRR | Query编码ms | AllHit@10/ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lines.extend(baseline_model_rows(metrics))
    lines += [
        "",
        "## 4. 超参数与对比设计",
        "",
        "| 超参数 | 当前取值 | 说明 |",
        "|---|---|---|",
        f"| loss | `{fixed.get('loss', 'mnrl')}` | 原主实验使用 query-positive 与 batch 内负例；显式 `negative_n` 消融见向量训练报告。 |",
        f"| learning_rate | `{fixed.get('learning_rate', 2e-5)}` | 首轮固定学习率，只扫 epoch，避免同时改变多个因素。 |",
        f"| batch_size | `{fixed.get('per_device_train_batch_size', 64)}` | H200 上 bge-base 的主实验批量；有效 batch 还受梯度累积影响。 |",
        f"| gradient_accumulation | `{fixed.get('gradient_accumulation_steps', 1)}` | 显存不足时优先增大该值并降低 batch。 |",
        f"| warmup_ratio | `{fixed.get('warmup_ratio', 0.05)}` | 减少训练初期震荡。 |",
        f"| max_seq_length | `{fixed.get('max_seq_length', 256)}` | 工具文档较短，先控制到 256 提升吞吐。 |",
        f"| batch_sampler | `{fixed.get('batch_sampler', 'no_duplicates')}` | 多意图 query 会展开为多个正样本，去重 batch 可降低 false negative 风险。 |",
        f"| query_instruction | `{fixed.get('query_instruction', '')}` | 与 baseline 评测时 query 编码前缀保持一致。 |",
        "",
        "| Run | Epoch | 其他设置 | 状态 |",
        "|---|---:|---|---|",
    ]
    existing = {run["run_name"] for run in runs}
    for item in sweep:
        run_name = item.get("run_name", "")
        status = "已评测" if run_name in existing else "待跑"
        lines.append(f"| {run_name} | {item.get('epochs')} | lr=2e-5, bs=64, MNRL | {status} |")
    for item in explicit_ablation.get("runs", []):
        run_name = item.get("run_name", "")
        status = "已评测" if run_name in existing else "待跑"
        lines.append(
            f"| {run_name} | {explicit_ablation.get('shared_hyperparameters', {}).get('epochs', 1)} | "
            f"lr=2e-5, micro/grad=8/8, negative_mode={item.get('negative_mode')} | {status} |"
        )

    lines += [
        "",
        "## 5. 微调评测",
        "",
        f"`AllHit@10 提升(pp)` 以零样本 `baseline:bge-base` 的 `AllHit@10={fmt(baseline_bge_allhit10)}` 为基准。",
        "",
        "| Run | Requested epoch | Actual epoch | LR | Effective batch | AllHit@10 | AllHit@10提升(pp) | AllHit@30 | MRR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(run_summary_rows(runs, baseline_bge_allhit10))
    lines += [
        "",
        "说明：`Actual epoch` 取自 trainer 输出；使用 no-duplicates batch sampler 时，实际训练步数可能小于 requested epoch。当前表包含原主实验 9 个 run 与显式负例补充实验 4 个 run；显式负例的同设置比较与解释边界统一保留在 `PROJECT/reports/03_向量训练.md`。",
        "",
    ]
    lines.extend(build_explicit_negative_section().strip().splitlines())
    lines += [
        "",
        "### K=6 到 10",
        "",
        "| Run | K | Hit@K | AllHit@K | Recall@K | Precision@K | MRR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(topk_rows(metrics, runs))

    lines += [
        "",
        "## 6. 提升原因与风险复核",
        "",
    ]
    lines.extend(improvement_audit_lines(generation_audit, probe_audit))

    lines += [
        "",
        "## 7. Embedding 阶段 Badcase（非最终链路）",
        "",
        "本节只定位微调 embedding 的检索边界；最终 source/reranker RRF 应单独基于其预测文件做残余失败分析。",
    ]
    lines.extend(badcase_lines(badcase_name, badcases, badcase_k))

    lines += [
        "",
        "## 8. 复现命令",
        "",
        "```bash",
        "cd /renyuanliu/Agent",
        "python3 PROJECT/scripts/dataset/audit_generation_leakage.py --tokenizer fallback",
        "bash PROJECT/scripts/rag/run_embedding_finetune_experiments.sh",
        "bash PROJECT/scripts/rag/run_embedding_explicit_negative_experiments.sh",
        "python3 PROJECT/scripts/report/make_embedding_training_report.py",
        "```",
        "",
        "单个实验可以拆开跑：",
        "",
        "```bash",
        "conda run -n llm_bge python PROJECT/scripts/rag/train_embedding.py --run --model BAAI/bge-base-zh-v1.5 --epochs 1 --bf16 --device cuda --out-dir PROJECT/data/rag/_aux/embedding_finetune/bge-base-mnrl-e1-lr2e-5-bs64/model",
        "conda run -n llm_bge python PROJECT/scripts/rag/build_vector_dbs.py --models bge-base-mnrl-e1-lr2e-5-bs64 --model-path bge-base-mnrl-e1-lr2e-5-bs64=PROJECT/data/rag/_aux/embedding_finetune/bge-base-mnrl-e1-lr2e-5-bs64/model/final --output-dir PROJECT/data/rag/vector_dbs_finetuned --device cuda --force",
        "conda run -n llm_bge python PROJECT/scripts/rag/run_retrieval_baselines.py --mode all --vector-db-dir PROJECT/data/rag/vector_dbs_finetuned --models bge-base-mnrl-e1-lr2e-5-bs64 --output-dir PROJECT/data/rag/_aux/embedding_finetune/bge-base-mnrl-e1-lr2e-5-bs64/retrieval --ks 6 7 8 9 10 15 20 30 50 --device cuda",
        "python3 PROJECT/scripts/report/make_embedding_training_report.py",
        "```",
    ]
    return lines


def main() -> int:
    OUT.write_text("\n".join(build_report()) + "\n", encoding="utf-8")
    print(f"embedding training report -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
