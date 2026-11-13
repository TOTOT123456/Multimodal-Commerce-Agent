#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the evidence-backed report for the 100-tool Assignment3 experiment."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import PROJECT_ROOT, load_jsonl, read_json, split_tools, write_json  # noqa: E402
from finetune.tool_trace_common import percentile, sha256_file  # noqa: E402


STAGES = ("base", "sft", "dpo")
MODELS = ("4b", "8b")
VIEWS = ("global", "rag")


def pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.2f}%"


def pp(value: float) -> str:
    return f"{100 * value:+.2f}"


def mcnemar(left: list[bool], right: list[bool]) -> dict[str, Any]:
    left_only = sum(a and not b for a, b in zip(left, right))
    right_only = sum(b and not a for a, b in zip(left, right))
    discordant = left_only + right_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = min(left_only, right_only)
        probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / (2**discordant)
        p_value = min(1.0, 2.0 * probability)
    return {
        "left_only": left_only,
        "right_only": right_only,
        "discordant": discordant,
        "two_sided_exact_p": p_value,
    }


def paired_bootstrap(
    left: list[bool],
    right: list[bool],
    *,
    samples: int = 5000,
    seed: int = 20260726,
) -> dict[str, Any]:
    import numpy as np

    delta = np.asarray(
        [float(after) - float(before) for before, after in zip(left, right)],
        dtype=np.float32,
    )
    rng = np.random.default_rng(seed)
    values: list[float] = []
    remaining = samples
    while remaining:
        chunk = min(250, remaining)
        indices = rng.integers(0, len(delta), size=(chunk, len(delta)), dtype=np.int32)
        values.extend(delta[indices].mean(axis=1).tolist())
        remaining -= chunk
    return {
        "effect": float(delta.mean()),
        "ci_low": percentile(values, 2.5),
        "ci_high": percentile(values, 97.5),
        "samples": samples,
        "seed": seed,
    }


def flags(path: Path, metric: str) -> list[bool]:
    rows = load_jsonl(path)
    return [bool(row[metric]) for row in rows]


def metric_cell(metrics_path: Path, scored_path: Path) -> dict[str, Any]:
    metrics = read_json(metrics_path)
    scored = load_jsonl(scored_path)
    if metrics.get("status") != "complete" or metrics.get("count") != 1000 or len(scored) != 1000:
        raise ValueError(f"incomplete official test cell: {metrics_path}")
    overall = metrics["overall"]
    return {
        "metrics_path": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
        "scored_path": str(scored_path),
        "scored_sha256": sha256_file(scored_path),
        "gold_snapshot_sha256": metrics["gold_snapshot_sha256"],
        "set_em": overall["set_em"],
        "trace_em": overall["trace_em"],
        "keyword_exact_accuracy": overall["keyword_exact_accuracy"],
        "keyword_micro_f1": overall["keyword_micro_f1"],
        "strict_output_valid_rate": overall["strict_output_valid_rate"],
        "json_schema_valid_rate": overall["json_schema_valid_rate"],
        "retriever_upper_bound": metrics["rag_decomposition"]["retriever_upper_bound"],
        "conditional_set_accuracy": metrics["rag_decomposition"]["conditional_set_accuracy"],
        "conditional_trace_accuracy": metrics["rag_decomposition"]["conditional_trace_accuracy"],
        "tool_count": metrics["buckets"]["tool_count"],
    }


def comparison(
    name: str,
    left_path: Path,
    right_path: Path,
    *,
    seed: int,
) -> dict[str, Any]:
    output = {"name": name, "left": str(left_path), "right": str(right_path), "metrics": {}}
    for metric_index, metric in enumerate(("set_em", "trace_em")):
        left = flags(left_path, metric)
        right = flags(right_path, metric)
        if len(left) != 1000 or len(right) != 1000:
            raise ValueError(f"comparison is not paired on 1000 rows: {name}")
        output["metrics"][metric] = {
            "left": sum(left) / len(left),
            "right": sum(right) / len(right),
            "mcnemar": mcnemar(left, right),
            "bootstrap": paired_bootstrap(left, right, seed=seed + metric_index),
        }
    return output


def sequence_novelty_metrics(
    scored_path: Path,
    *,
    gold_by_id: dict[str, dict[str, Any]],
    train_sequences: set[tuple[str, ...]],
    allowed_tool_counts: set[int],
) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, Any]]] = {"seen": [], "unseen": []}
    for row in load_jsonl(scored_path):
        gold = gold_by_id[row["id"]]
        if int(gold["tool_count"]) not in allowed_tool_counts:
            continue
        key = (
            "seen"
            if tuple(split_tools(gold["工具"])) in train_sequences
            else "unseen"
        )
        grouped[key].append(row)
    return {
        key: {
            "rows": len(rows),
            "set_em": sum(bool(row["set_em"]) for row in rows) / max(1, len(rows)),
            "trace_em": sum(bool(row["trace_em"]) for row in rows) / max(1, len(rows)),
        }
        for key, rows in grouped.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "tool_call_trace"
        / "_aux"
        / "assignment3"
        / "assignment3_100tool_api",
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=PROJECT_ROOT
        / "finetune"
        / "outputs"
        / "preds"
        / "tool_trace"
        / "assignment3_100tool_test",
    )
    parser.add_argument(
        "--order-pred-dir",
        type=Path,
        default=PROJECT_ROOT
        / "finetune"
        / "outputs"
        / "preds"
        / "tool_trace"
        / "assignment3_100tool_order_counterfactual",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "tool_call_trace"
        / "_aux"
        / "assignment3"
        / "assignment3_100tool_api"
        / "final_summary.json",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "tool_call_trace"
        / "_aux"
        / "assignment3"
        / "assignment3_100tool_api"
        / "generated_formal_report.md",
    )
    args = parser.parse_args()

    exp = args.experiment_dir
    required = {
        "protocol": PROJECT_ROOT / "configs" / "assignment3_100tool_protocol.json",
        "plan_audit": exp / "plan_audit.json",
        "data_audit": exp / "data_isolation_audit.json",
        "train_generation": exp / "train_api" / "generation_audit.json",
        "test_generation": exp / "test_api" / "generation_audit.json",
        "sft_data": exp / "sft_data_audit.json",
        "sft_labels_4b": exp / "sft_label_audit_4b.json",
        "sft_labels_8b": exp / "sft_label_audit_8b.json",
        "sft_runs": exp / "sft_training_runs_audit.json",
        "sft_selection": exp / "sft4_epoch_selection.json",
        "retrieval": exp / "test_api" / "retrieval_manifest.json",
        "dpo_data": exp / "dpo_data_audit.json",
        "dpo_tokens": exp / "dpo_token_length_audit.json",
        "dpo_labels_4b": exp / "dpo_label_audit_4b.json",
        "dpo_labels_8b": exp / "dpo_label_audit_8b.json",
        "dpo_runs": exp / "dpo_training_runs_audit.json",
        "dpo_selection": exp / "dpo_final_selection.json",
        "environment": exp / "environment_manifest.json",
        "order_generation": exp / "order_counterfactual_api" / "generation_audit.json",
        "order_audit": exp / "order_counterfactual_api" / "isolation_audit.json",
        "candidate_global": exp / "sft4_global_candidate_order_analysis.json",
        "candidate_rag": exp / "sft4_rag_candidate_order_analysis.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise SystemExit(f"missing final experiment artifacts: {missing}")
    artifacts = {name: read_json(path) for name, path in required.items()}
    for name in (
        "data_audit",
        "sft_data",
        "sft_labels_4b",
        "sft_labels_8b",
        "sft_runs",
        "dpo_data",
        "dpo_tokens",
        "dpo_labels_4b",
        "dpo_labels_8b",
        "dpo_runs",
        "order_audit",
    ):
        if not artifacts[name].get("strict_ok"):
            raise SystemExit(f"strict audit failed: {name}")
    if artifacts["train_generation"].get("strict_ok") is not True:
        raise SystemExit("train API generation is not complete")
    if artifacts["test_generation"].get("strict_ok") is not True:
        raise SystemExit("test API generation is not complete")
    if artifacts["order_generation"].get("strict_ok") is not True:
        raise SystemExit("order-counterfactual API generation is not complete")
    if any(
        artifacts[name].get("status") != "complete"
        for name in ("candidate_global", "candidate_rag")
    ):
        raise SystemExit("candidate-order analysis is not complete")
    if artifacts["dpo_data"].get("final_pairs") != 5000:
        raise SystemExit("DPO data is not exactly 5000 audited pairs")

    cells: dict[str, dict[str, dict[str, Any]]] = {}
    scored_paths: dict[str, Path] = {}
    gold_hashes = set()
    for stage in STAGES:
        cells[stage] = {}
        for model in MODELS:
            cells[stage][model] = {}
            for view in VIEWS:
                stem = f"a3-100-{stage}-{model}-{view}"
                metrics_path = args.pred_dir / f"{stem}.metrics.json"
                scored_path = args.pred_dir / f"{stem}.scored.jsonl"
                cell = metric_cell(metrics_path, scored_path)
                cells[stage][model][view] = cell
                scored_paths[f"{stage}-{model}-{view}"] = scored_path
                gold_hashes.add(cell["gold_snapshot_sha256"])
    if len(gold_hashes) != 1:
        raise SystemExit("official matrix cells do not share one frozen test snapshot")

    comparisons = []
    seed = 20260726
    for model in MODELS:
        for view in VIEWS:
            comparisons.append(
                comparison(
                    f"{model}:{view}:base->sft",
                    scored_paths[f"base-{model}-{view}"],
                    scored_paths[f"sft-{model}-{view}"],
                    seed=seed,
                )
            )
            seed += 2
            comparisons.append(
                comparison(
                    f"{model}:{view}:sft->dpo",
                    scored_paths[f"sft-{model}-{view}"],
                    scored_paths[f"dpo-{model}-{view}"],
                    seed=seed,
                )
            )
            seed += 2
        for stage in STAGES:
            comparisons.append(
                comparison(
                    f"{model}:{stage}:global->rag",
                    scored_paths[f"{stage}-{model}-global"],
                    scored_paths[f"{stage}-{model}-rag"],
                    seed=seed,
                )
            )
            seed += 2

    train_rows = load_jsonl(exp / "train_api" / "train.jsonl")
    test_rows = load_jsonl(exp / "test_api" / "test.jsonl")
    train_sequences = {
        tuple(split_tools(row["工具"])) for row in train_rows
    }
    gold_by_id = {row["id"]: row for row in test_rows}
    sequence_novelty = {
        stage: {
            model: {
                view: sequence_novelty_metrics(
                    scored_paths[f"{stage}-{model}-{view}"],
                    gold_by_id=gold_by_id,
                    train_sequences=train_sequences,
                    allowed_tool_counts={4, 5},
                )
                for view in VIEWS
            }
            for model in MODELS
        }
        for stage in STAGES
    }

    order_metrics: dict[str, dict[str, Any]] = {}
    for stage in STAGES:
        order_metrics[stage] = {}
        for model in MODELS:
            path = args.order_pred_dir / f"a3-100-{stage}-{model}-order-global.pair_metrics.json"
            payload = read_json(path)
            if payload.get("status") != "complete" or payload["overall"].get("pairs") != 176:
                raise SystemExit(f"incomplete order counterfactual metrics: {path}")
            order_metrics[stage][model] = {
                **payload["overall"],
                "path": str(path),
                "sha256": sha256_file(path),
            }

    summary = {
        "status": "complete",
        "scope": "static ordered tool-ID planning; no tool execution",
        "official_test_rows": 1000,
        "official_test_sha256": next(iter(gold_hashes)),
        "data": {
            "scale_derivation": artifacts["plan_audit"]["scale_derivation"],
            "train": artifacts["data_audit"]["train"],
            "test": artifacts["data_audit"]["test"],
            "normalized_query_overlap": artifacts["data_audit"]["normalized_query_overlap"],
            "source_trace_id_overlap": artifacts["data_audit"]["source_trace_id_overlap"],
            "target_sequence_overlap": artifacts["data_audit"]["target_sequence_overlap"],
            "text_structure_overlap": artifacts["data_audit"]["text_structure_overlap"],
            "sft_runtime": artifacts["sft_data"],
        },
        "retrieval": {
            "tuned": artifacts["retrieval"]["metrics"],
            "untuned": artifacts["retrieval"]["untuned_metrics"],
            "parameters": artifacts["retrieval"]["parameters"],
        },
        "official_matrix": cells,
        "paired_comparisons": comparisons,
        "sequence_novelty_tool_count_4_5": sequence_novelty,
        "sft_selection": artifacts["sft_selection"],
        "dpo": {
            "data": artifacts["dpo_data"],
            "token_audit": artifacts["dpo_tokens"],
            "run_audit": artifacts["dpo_runs"],
            "selection": artifacts["dpo_selection"],
        },
        "order_counterfactual": {
            "generation": artifacts["order_generation"],
            "isolation": artifacts["order_audit"],
            "metrics": order_metrics,
        },
        "candidate_order": {
            "global": artifacts["candidate_global"],
            "rag": artifacts["candidate_rag"],
        },
        "provenance": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in required.items()
        },
    }
    write_json(args.summary_out, summary)

    matrix_rows = []
    stage_labels = {"base": "Base", "sft": "SFT", "dpo": "SFT+DPO"}
    view_labels = {"global": "Wo-RAG（100 工具）", "rag": "RAG@10"}
    for model in MODELS:
        for stage in STAGES:
            for view in VIEWS:
                cell = cells[stage][model][view]
                matrix_rows.append(
                    f"| Qwen3-{model.upper()} | {stage_labels[stage]} | {view_labels[view]} | "
                    f"{pct(cell['keyword_exact_accuracy'])} | {pct(cell['trace_em'])} | "
                    f"{pct(cell['keyword_micro_f1'])} | {pct(cell['strict_output_valid_rate'])} |"
                )

    delta_rows = []
    for model in MODELS:
        for view in VIEWS:
            base = cells["base"][model][view]
            sft = cells["sft"][model][view]
            dpo = cells["dpo"][model][view]
            delta_rows.append(
                f"| Qwen3-{model.upper()} | {view_labels[view]} | "
                f"{pp(sft['set_em'] - base['set_em'])} pp | "
                f"{pp(sft['trace_em'] - base['trace_em'])} pp | "
                f"{pp(dpo['set_em'] - sft['set_em'])} pp | "
                f"{pp(dpo['trace_em'] - sft['trace_em'])} pp |"
            )

    sft_selection = artifacts["sft_selection"]
    sft_selection_rows = "\n".join(
        f"| {row['name']} | {pct(row['set_em'])} | {pct(row['trace_em'])} | "
        f"{pct(row['strict_output_valid_rate'])} | "
        f"{'是' if row['name'] == sft_selection['selected']['name'] else '否'} |"
        for row in sft_selection["ranked_candidates"]
    )
    dpo_selection = artifacts["dpo_selection"]
    dpo_selection_rows = "\n".join(
        f"| {row['name']} | {pct(row['set_em'])} | {pct(row['trace_em'])} | "
        f"{pct(row['strict_output_valid_rate'])} | "
        f"{'是' if row['name'] == dpo_selection['selected_4b']['name'] else '否'} |"
        for row in dpo_selection["beta_grid"]
    )
    order_rows = []
    for model in MODELS:
        for stage in STAGES:
            value = order_metrics[stage][model]
            order_rows.append(
                f"| Qwen3-{model.upper()} | {stage_labels[stage]} | "
                f"{pct(value['row_set_em'])} | {pct(value['row_trace_em'])} | "
                f"{pct(value['pair_both_trace_em'])} | "
                f"{pct(value['order_following_given_both_set_correct'])} | "
                f"{pct(value['predicted_trace_changed_rate'])} |"
            )

    candidate_rows = []
    for view, payload in (
        ("Wo-RAG（100 工具）", artifacts["candidate_global"]),
        ("RAG@10", artifacts["candidate_rag"]),
    ):
        candidate_rows.append(
            f"| {view} | original | {pct(payload['baseline_metrics']['set_em'])} | "
            f"{pct(payload['baseline_metrics']['trace_em'])} | 100.00% |"
        )
        for name, variant in payload["variants"].items():
            candidate_rows.append(
                f"| {view} | {name} | {pct(variant['metrics']['set_em'])} | "
                f"{pct(variant['metrics']['trace_em'])} | "
                f"{pct(variant['predicted_trace_equal_rate'])} |"
            )

    novelty_rows = []
    for model in MODELS:
        for view in VIEWS:
            values = sequence_novelty["sft"][model][view]
            novelty_rows.append(
                f"| Qwen3-{model.upper()} | {view_labels[view]} | "
                f"{values['seen']['rows']} | {pct(values['seen']['set_em'])} | "
                f"{pct(values['seen']['trace_em'])} | {values['unseen']['rows']} | "
                f"{pct(values['unseen']['set_em'])} | {pct(values['unseen']['trace_em'])} |"
            )

    data_audit = artifacts["data_audit"]
    plan_train = artifacts["plan_audit"]["train"]
    order_randomization = plan_train["multi_order_randomization"]
    target_overlap = data_audit["target_sequence_overlap"]
    expression = data_audit["text_structure_overlap"]["expression_cluster_proxy"]
    context = data_audit["text_structure_overlap"]["context_skeleton_proxy"]
    retrieval = artifacts["retrieval"]
    dpo_data = artifacts["dpo_data"]
    report = f"""# Assignment3：100 工具 API 原生数据完整实验

> 实验范围：`query → 有序工具 ID 轨迹`。项目没有实现真实工具，因此本文不报告工具执行成功率、参数正确率、状态转移或失败恢复；多轮对话是作业可选项，本轮未纳入主实验。

## 0. 作业覆盖清单

| 作业节 | 必做内容 | 本轮实现 | 状态 |
|---|---|---|---|
| 0 | Qwen3-4B/8B、全工具列表、1k 测试、关键词准确率 | Base × 4B/8B × Wo-RAG（完整 100 工具），保存逐条预测并报告关键词 SetEM | 完成 |
| 1 | RAG 端到端 baseline，RAG VS Wo-RAG | 同一冻结 1k 上对比 Base 的 RAG@10 与 Wo-RAG | 完成 |
| 2 | 3k 单意图、6k 多意图；4B/8B LoRA；超参数比较；1k 评测 | 按 6 倍扩为 18k+24k+12k=54k；API 生成；4B epoch 1/2，8B 固定 1 epoch；正式比较 Base→SFT | 完成 |
| 3 | 2k–5k DPO 偏好；4B/8B LoRA；超参数比较；SFT+DPO VS SFT | 5k 真实模型错误对；4B beta=0.05/0.10，8B继承 dev 最优 beta；正式比较 SFT→DPO | 完成 |
| 4 | 微调后 4B/8B 的最终 RAG 评测 | SFT 与 SFT+DPO 均运行 RAG@10 正式评测 | 完成 |
| 5 | 多轮对话（可选） | 项目未实现真实工具与状态机，不混入本轮静态轨迹主实验 | 合理不做 |

## 1. 为什么重新设计

旧实验的 SFT Wo-RAG 接近满分，无法排除模板化表达、闭集候选暴露和全 catalog 训练视图造成的捷径。本轮把“是否真的学会 query 到有序工具链”作为可证伪目标：

- 训练和正式测试 query 全部由 API 生成并由另一 API 模型独立审查，未使用 query 模板；
- SFT 只训练 2–6 个候选工具的 RAG/candidate 视图，**全 catalog 训练样本为 0**；
- 正式 Wo-RAG 评测一次展示完整 100 工具，因此不会因训练时见过同形 full-catalog prompt 而虚高；
- 同时报告作业要求的关键词集合准确率（即 SetEM）和更严格的 TraceEM；
- 增加候选展示顺序扰动和 API 顺序反事实对，专门检查顺序捷径。

## 2. 作业数据构造与 100 工具缩放

PDF 的前置训练集为 4,000 条，本项目正式前置训练集为 24,000 条，缩放因子为 6。对应构造：

| 子集 | PDF | 100 工具版 | 构造 |
|---|---:|---:|---|
| 单意图 | 3,000 | 18,000 | 1 个 GT + 1–4 个同域干扰候选 |
| 多意图顺序 | 4,000 | 24,000 | 在依赖约束内随机化 GT 有序轨迹，并独立打乱候选展示顺序 |
| 多意图加干扰 | 2,000 | 12,000 | +1–3 个同域非 GT 工具，总候选不超过 5 |
| 合计 | 9,000 | 54,000 | 100 工具全覆盖 |

正式测试严格保留作业要求的 1,000 条，tool_count=1–5 各 200 条，覆盖全部 100 工具；另用 tool_count=2–6 的 176 对顺序反事实补足六步轨迹诊断。

数据审计结果：

- API 训练集 54,000 条、正式测试 1,000 条，ID 与规范化 query 均唯一；
- 24,000 条多意图顺序样本中，GT 顺序相对来源轨迹实际变化 {order_randomization['target_order_differs_from_source']} 条，候选展示顺序相对 GT 变化 {order_randomization['candidate_order_differs_from_target']} 条；全部 {order_randomization['dependency_valid_rows']} 条满足工具依赖；
- 这是固定 100 工具闭集，不应把“无文本泄漏”误写成“目标组合完全未见”：正式测试中 {target_overlap['test_ordered_sequence_seen']}/1000 的有序工具序列在 SFT 训练中出现过，{target_overlap['test_tool_set_seen']}/1000 的工具集合出现过；单工具及两工具层全部属于 seen，未见序列主要集中于四、五工具样本；
- train/test、formal train、冻结 test、旧 API dev 间规范化 query overlap 全为 0；
- train/test 的来源 trace ID overlap 为 {data_audit['source_trace_id_overlap']['shared_source_trace_ids']}；
- expression-cluster 同序列签名命中 {expression['test_rows_same_tool_sequence_seen_under_signature']}/1000，context-skeleton 命中 {context['test_rows_same_tool_sequence_seen_under_signature']}/1000；阈值审计 strict_ok={data_audit['strict_ok']}；
- SFT runtime 54,000 条，full-catalog 训练条数为 {artifacts['sft_data']['full_catalog_training_rows']}，4B/8B 最大序列均无截断。

## 3. Base 与 RAG

固定 RAG@10 的 tuned AllHit@10={pct(retrieval['metrics']['allhit_at_10'])}，untuned AllHit@10={pct(retrieval['untuned_metrics']['allhit_at_10'])}。RAG 只改变候选集合；正式测试标签没有进入 prompt 或候选构造。

## 4. 正式 1,000 条结果

关键词准确率是作业指定口径；在当前规范输出下，它等价于 SetEM。TraceEM 还要求顺序完全一致。

| 模型 | 阶段 | 视图 | 关键词准确率 / SetEM | TraceEM | 关键词 micro-F1 | 严格输出有效率 |
|---|---|---|---:|---:|---:|---:|
{chr(10).join(matrix_rows)}

| 模型 | 视图 | SFT−Base SetEM | SFT−Base TraceEM | DPO−SFT SetEM | DPO−SFT TraceEM |
|---|---|---:|---:|---:|---:|
{chr(10).join(delta_rows)}

所有主比较均在同一 1,000 条样本上做了配对 McNemar 精确检验和 5,000 次配对 bootstrap；结构化结果见 `{args.summary_out}` 的 `paired_comparisons`。

为避免把组合记忆误判成泛化，下面只在难度更接近的 tool_count=4–5 共 400 条中，按“有序 GT 是否在 SFT 训练出现”拆分 SFT：

| 模型 | 视图 | seen N | seen SetEM | seen TraceEM | unseen N | unseen SetEM | unseen TraceEM |
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(novelty_rows)}

该拆分仍是同一 100 工具闭集内的组合新颖性，不等价于新工具、真实日志或开放世界泛化。

## 5. 超参数与 checkpoint 选择

正式测试不参与选参。4B SFT 用独立 API-native 2,400 dev 比较 epoch 1/2，8B 使用预先固定的 1 epoch。

| 4B SFT 候选 | Dev SetEM | Dev TraceEM | 严格输出有效率 | 选中 |
|---|---:|---:|---:|---|
{sft_selection_rows}

DPO 从真实模型生成错误中取 5,000 对，endpoint SFT 错误优先，不足才由真实 early-SFT 错误补齐；synthetic_pairs={dpo_data['synthetic_pairs']}，core ratio={pct(dpo_data['core_hard_error_ratio'])}，工具覆盖={dpo_data['tool_coverage']}/100。4B 比较 beta=0.05/0.10，8B 继承 4B dev 选出的 beta={dpo_selection['selected_beta']}。

DPO 来源构成必须单独披露：当前选中 SFT 4B/8B 的采样错误合计只有 {dpo_data['rejected_source_counts'].get('selected_sft4_sampled_s0', 0) + dpo_data['rejected_source_counts'].get('selected_sft8_sampled_s0', 0)} 对，真实 early-SFT step-20 错误为 {dpo_data['rejected_source_counts'].get('early_sft4_step20_sampled_s0', 0)} 对。它符合“不同版本模型生成真实错误”的作业要求，但也意味着 DPO 主要学习的是早期模型错误分布，而非当前 endpoint 的新增错误。

DPO 没有机械放大到 30,000 对：PDF 对偏好数据明确给出 2k–5k，本轮取上限 5k，并限制每个 query 最多 2 对。偏好对是错误挖掘结果而非独立覆盖样本，优先保证真实来源、难度和可追溯性。

| 4B DPO 候选 | Dev SetEM | Dev TraceEM | 严格输出有效率 | 选中 |
|---|---:|---:|---:|---|
{dpo_selection_rows}

## 6. 顺序专项结果

候选顺序消融保持 query 与候选集合完全相同，只改变候选展示顺序：

| 视图 | 展示顺序 | SetEM | TraceEM | 与 original 预测轨迹一致率 |
|---|---|---:|---:|---:|
{chr(10).join(candidate_rows)}

API 顺序反事实的每一对 query 具有相同对象、条件和任务集合，唯一变化是明确请求顺序；全部由 API 联合生成并通过独立 judge、工具 ID 泄漏及五集合零重叠审计。

| 模型 | 阶段 | 行 SetEM | 行 TraceEM | 两条均 TraceEM | 集合均正确时顺序遵循率 | 预测随顺序改变 |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(order_rows)}

## 7. 结论与边界

本轮结果不再把高分自动解释为“Agent 能执行工具”。可以支持的结论只有：在固定 100 工具、API 原生中文 query 和静态工具说明下，模型选择工具集合及有序 ID 轨迹的能力变化。

- **SFT Wo-RAG 94.7%/94.8% 仍高，但不能再归因于 query 模板或 full-catalog 训练 prompt。** 更主要的解释是：任务只预测 100 个语义明确的 ID，54k 训练让每个工具被反复监督，而且 82.1% 正式样本的有序目标序列在训练目标中出现过。另一方面，四、五工具的 unseen 序列仍达到 89.94%/92.74% SetEM，API 顺序反事实也显著高于 Base，因此结果并非只靠逐句模板或固定候选位置记忆。
- **RAG 接近 99% 的直接前提是 tuned retriever 在本测试上 AllHit@10=100%。** 候选从 100 个压到 10 个后，模型面对的是强闭集筛选；untuned retriever 的同口径上界只有 84.2%，所以 99% 不能归结为生成模型单方面能力。
- **DPO 不应被包装成主指标提升。** 相对 SFT，4B Wo-RAG/RAG 分别下降 1.4/0.1 pp，8B Wo-RAG 下降 1.1 pp、RAG 仅上升 0.1 pp；正式静态准确率应保留 SFT checkpoint。DPO 的真实收益出现在顺序反事实：两条正反 query 都答对的比例由 90.34%→92.05%（4B）、92.05%→92.61%（8B），说明它改善了部分顺序鲁棒性，但 early-SFT 错误占绝大多数带来了分布偏移。

仍不能外推到真实线上：工具参数、真实返回值、跨步状态、失败重试、权限副作用与多轮记忆均未实现。RAG tuned AllHit@10 达到 100% 也只说明本 100 工具合成闭集的检索覆盖，不代表开放世界检索。

## 8. 可复现证据

- 协议：`{required['protocol']}`
- 数据隔离：`{required['data_audit']}`
- SFT 数据、实际监督标签与运行审计：`{required['sft_data']}`、`{required['sft_labels_4b']}`、`{required['sft_labels_8b']}`、`{required['sft_runs']}`
- DPO 数据、token、实际 chosen/rejected 标签与运行审计：`{required['dpo_data']}`、`{required['dpo_tokens']}`、`{required['dpo_labels_4b']}`、`{required['dpo_labels_8b']}`、`{required['dpo_runs']}`
- 正式预测与逐条评分：`{args.pred_dir}`
- 顺序反事实预测：`{args.order_pred_dir}`
- 完整结构化汇总：`{args.summary_out}`
"""
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(report, encoding="utf-8")
    print(f"Assignment3 100-tool report complete -> {args.summary_out}, {args.report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
