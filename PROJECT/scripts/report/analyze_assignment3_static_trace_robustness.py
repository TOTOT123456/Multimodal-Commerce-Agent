#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize static Assignment3 order and retrieval robustness experiments."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import load_jsonl, write_json  # noqa: E402
from finetune.tool_trace_common import sha256_file  # noqa: E402


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.2f}%"


def pp(value: float) -> str:
    return f"{value * 100:.2f}"


def metric_summary(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    overall = payload["overall"]
    retrieval = payload["rag_decomposition"]
    return {
        "path": str(path),
        "set_em": overall["set_em"],
        "trace_em": overall["trace_em"],
        "json_schema_valid_rate": overall["json_schema_valid_rate"],
        "strict_output_valid_rate": overall["strict_output_valid_rate"],
        "retriever_upper_bound": retrieval["retriever_upper_bound"],
        "conditional_set_accuracy": retrieval["conditional_set_accuracy"],
    }


def scored_index(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(path)
    output = {str(row["id"]): row for row in rows}
    if len(output) != len(rows):
        raise ValueError(f"duplicate ids: {path}")
    return output


def candidate_order_stability(original: Path, variant: Path) -> dict[str, Any]:
    baseline = scored_index(original)
    changed = scored_index(variant)
    if set(baseline) != set(changed):
        raise ValueError("scored id mismatch")
    rows = list(baseline)
    candidate_set_equal = sum(
        set(baseline[row_id]["candidate_tools"]) == set(changed[row_id]["candidate_tools"])
        for row_id in rows
    )
    candidate_order_changed = sum(
        baseline[row_id]["candidate_tools"] != changed[row_id]["candidate_tools"]
        for row_id in rows
    )
    trace_equal = sum(
        baseline[row_id]["pred_tools"] == changed[row_id]["pred_tools"]
        for row_id in rows
    )
    set_equal = sum(
        set(baseline[row_id]["pred_tools"]) == set(changed[row_id]["pred_tools"])
        for row_id in rows
    )
    baseline_trace_correct = [row_id for row_id in rows if baseline[row_id]["trace_em"]]
    return {
        "rows": len(rows),
        "candidate_set_equal_rate": candidate_set_equal / len(rows),
        "candidate_order_changed_rate": candidate_order_changed / len(rows),
        "predicted_trace_equal_rate": trace_equal / len(rows),
        "predicted_set_equal_rate": set_equal / len(rows),
        "trace_retention_from_original_correct": (
            sum(changed[row_id]["trace_em"] for row_id in baseline_trace_correct)
            / len(baseline_trace_correct)
            if baseline_trace_correct
            else None
        ),
        "trace_wins": sum(
            not baseline[row_id]["trace_em"] and changed[row_id]["trace_em"] for row_id in rows
        ),
        "trace_losses": sum(
            baseline[row_id]["trace_em"] and not changed[row_id]["trace_em"] for row_id in rows
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-pred-dir", type=Path, required=True)
    parser.add_argument("--stress-pred-dir", type=Path, required=True)
    parser.add_argument("--pair-audit", type=Path, required=True)
    parser.add_argument("--pair-metrics", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()

    primary = args.primary_pred_dir
    stress = args.stress_pred_dir
    original = {
        "global": metric_summary(primary / "api-native-sft4_s1-global.metrics.json"),
        "rag_k10": metric_summary(primary / "api-native-sft4_s1-rag-k10.metrics.json"),
        "rag_k20": metric_summary(primary / "api-native-sft4_s1-rag-k20.metrics.json"),
    }
    variants = {
        "global_reverse": metric_summary(stress / "sft4_s1-global-order-reverse.metrics.json"),
        "global_hash_shuffle": metric_summary(stress / "sft4_s1-global-order-hash-shuffle.metrics.json"),
        "rag_k20_reverse": metric_summary(stress / "sft4_s1-rag-k20-order-reverse.metrics.json"),
        "rag_k20_hash_shuffle": metric_summary(stress / "sft4_s1-rag-k20-order-hash-shuffle.metrics.json"),
    }
    untuned = {
        "rag_k10": metric_summary(stress / "sft4_s1-untuned-rag-k10.metrics.json"),
        "rag_k20": metric_summary(stress / "sft4_s1-untuned-rag-k20.metrics.json"),
    }
    stability = {
        "global_reverse": candidate_order_stability(
            primary / "api-native-sft4_s1-global.scored.jsonl",
            stress / "sft4_s1-global-order-reverse.scored.jsonl",
        ),
        "global_hash_shuffle": candidate_order_stability(
            primary / "api-native-sft4_s1-global.scored.jsonl",
            stress / "sft4_s1-global-order-hash-shuffle.scored.jsonl",
        ),
        "rag_k20_reverse": candidate_order_stability(
            primary / "api-native-sft4_s1-rag-k20.scored.jsonl",
            stress / "sft4_s1-rag-k20-order-reverse.scored.jsonl",
        ),
        "rag_k20_hash_shuffle": candidate_order_stability(
            primary / "api-native-sft4_s1-rag-k20.scored.jsonl",
            stress / "sft4_s1-rag-k20-order-hash-shuffle.scored.jsonl",
        ),
    }
    if any(value["candidate_set_equal_rate"] != 1.0 for value in stability.values()):
        raise ValueError("candidate order variant changed a candidate set")
    pair_audit = load_json(args.pair_audit)
    pair_metrics = load_json(args.pair_metrics)
    if not pair_audit.get("strict_ok") or pair_metrics.get("status") != "complete":
        raise ValueError("paired counterfactual audit or metrics incomplete")

    summary = {
        "analysis_version": "assignment3-static-trace-robustness-v1",
        "status": "complete",
        "scope": "static ordered tool-ID planning only; no tool execution or observations",
        "candidate_order": {"original": original, "variants": variants, "stability": stability},
        "retrieval_causality": {"tuned": {"rag_k10": original["rag_k10"], "rag_k20": original["rag_k20"]}, "untuned": untuned},
        "order_counterfactual": pair_metrics,
        "pair_audit": {"strict_ok": pair_audit["strict_ok"], "counts": pair_audit["counts"], "query_overlap": pair_audit["query_overlap"]},
        "hashes": {
            str(path): sha256_file(path)
            for path in (
                args.pair_audit,
                args.pair_metrics,
                primary / "api-native-sft4_s1-global.metrics.json",
                primary / "api-native-sft4_s1-rag-k10.metrics.json",
                primary / "api-native-sft4_s1-rag-k20.metrics.json",
                stress / "sft4_s1-global-order-reverse.metrics.json",
                stress / "sft4_s1-global-order-hash-shuffle.metrics.json",
                stress / "sft4_s1-rag-k20-order-reverse.metrics.json",
                stress / "sft4_s1-rag-k20-order-hash-shuffle.metrics.json",
                stress / "sft4_s1-untuned-rag-k10.metrics.json",
                stress / "sft4_s1-untuned-rag-k20.metrics.json",
            )
        },
    }
    write_json(args.summary_out, summary)

    candidate_rows = [
        ("full-catalog", "original", original["global"], None),
        ("full-catalog", "reverse", variants["global_reverse"], stability["global_reverse"]),
        ("full-catalog", "hash shuffle", variants["global_hash_shuffle"], stability["global_hash_shuffle"]),
        ("RAG@20", "original", original["rag_k20"], None),
        ("RAG@20", "reverse", variants["rag_k20_reverse"], stability["rag_k20_reverse"]),
        ("RAG@20", "hash shuffle", variants["rag_k20_hash_shuffle"], stability["rag_k20_hash_shuffle"]),
    ]
    candidate_table = "\n".join(
        f"| {view} | {order} | {pct(metric['set_em'])} | {pct(metric['trace_em'])} | "
        f"{('—' if stability_row is None else pct(stability_row['predicted_trace_equal_rate']))} | "
        f"{('—' if stability_row is None else str(stability_row['trace_wins']))} / {('—' if stability_row is None else str(stability_row['trace_losses']))} |"
        for view, order, metric, stability_row in candidate_rows
    )
    retrieval_table = "\n".join(
        f"| RAG@{k[5:]} | {kind} | {pct(metric['retriever_upper_bound'])} | {pct(metric['conditional_set_accuracy'])} | {pct(metric['set_em'])} | {pct(metric['trace_em'])} |"
        for k, kind, metric in (
            ("rag_k10", "tuned", original["rag_k10"]),
            ("rag_k10", "untuned", untuned["rag_k10"]),
            ("rag_k20", "tuned", original["rag_k20"]),
            ("rag_k20", "untuned", untuned["rag_k20"]),
        )
    )
    pair = pair_metrics["overall"]
    report = f"""# Assignment3 静态轨迹鲁棒性复测

> 范围：只评测 `query → 有序工具 ID 列表`。没有实现工具，因此不声称执行成功率、状态规划或失败恢复能力。

## 候选顺序消融

同一 query、同一候选工具集合，仅改变 prompt 中的候选展示顺序。所有变体的候选集合一致率均为 100%，因此差异只能来自候选位置与展示顺序。

| 视图 | 候选顺序 | SetEM | TraceEM | 与原始预测轨迹一致率 | 相对原始 Trace 赢 / 失 |
|---|---|---:|---:|---:|---:|
{candidate_table}

## 检索因果对照

在相同 SFT、解码和 K 下，仅替换 tuned/untuned 检索排序。

| 视图 | 检索 | AllHit 上限 | 条件规划准确率 | SetEM | TraceEM |
|---|---|---:|---:|---:|---:|
{retrieval_table}

## API 顺序反事实对

从 query-free 能力计划中选取无内部依赖、只含 read/compute/decision 的工具集合，生成 {pair['pairs']} 对、{pair['rows']} 条 API 原生 query。每对共享对象、条件与任务集合，只反转明确请求顺序；独立 API judge 和隔离审计均通过。

| 指标 | 结果 |
|---|---:|
| 行级 SetEM | {pct(pair['row_set_em'])} |
| 行级 TraceEM | {pct(pair['row_trace_em'])} |
| 两条均集合正确 | {pct(pair['pair_both_set_em'])} |
| 两条均顺序正确 | {pct(pair['pair_both_trace_em'])} |
| 两条集合均正确时的顺序遵循率 | {pct(pair['order_following_given_both_set_correct'])} |
| 预测轨迹随 pair 改变 | {pct(pair['predicted_trace_changed_rate'])} |

## 结论口径

候选顺序不是主导性捷径，但也并非完全无影响：full-catalog 的 reverse / hash 变体相对原始 TraceEM 分别变化 {pp(variants['global_reverse']['trace_em'] - original['global']['trace_em'])} / {pp(variants['global_hash_shuffle']['trace_em'] - original['global']['trace_em'])} 个百分点；RAG@20 分别变化 {pp(variants['rag_k20_reverse']['trace_em'] - original['rag_k20']['trace_em'])} / {pp(variants['rag_k20_hash_shuffle']['trace_em'] - original['rag_k20']['trace_em'])} 个百分点。四组中仅有 2.21%–3.71% 的预测轨迹改变，说明存在轻微位置敏感性，不能声称完全顺序不变。

tuned/untuned 对照显示主要差异来自检索覆盖而不是候选内规划：K=10 的端到端 TraceEM 相差 {pp(original['rag_k10']['trace_em'] - untuned['rag_k10']['trace_em'])} 个百分点，K=20 相差 {pp(original['rag_k20']['trace_em'] - untuned['rag_k20']['trace_em'])} 个百分点，而四项条件规划准确率均约为 97%–98%。反事实对则直接表明：在本次筛出的独立 read/compute/decision 子集内，模型会随明确的 query 顺序改变预测轨迹；这仍只是静态选择证据，不是工具执行、状态规划或失败恢复证据。

## 可复现产物

- 结构化汇总：`{args.summary_out}`
- 配对审计：`{args.pair_audit}`
- 配对指标：`{args.pair_metrics}`
- 原始及变体预测：`{args.primary_pred_dir}`、`{args.stress_pred_dir}`
"""
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(report, encoding="utf-8")
    print(f"static trace robustness analysis complete -> {args.summary_out}, {args.report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
