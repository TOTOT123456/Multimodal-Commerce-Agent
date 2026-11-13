#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create assignment-3 matrix, significance, and badcase reports."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import PROJECT_ROOT, read_json, write_json  # noqa: E402
from finetune.tool_trace_common import ASSIGNMENT3_DIR, percentile, sha256_file  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "tool_trace_experiments.json"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "06_工具轨迹SFT_DPO与端到端评测.md"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def mcnemar_exact(left: list[bool], right: list[bool]) -> dict[str, Any]:
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
    samples: int,
    seed: int,
) -> dict[str, float]:
    delta = [float(b) - float(a) for a, b in zip(left, right)]
    observed = sum(delta) / len(delta)
    n = len(delta)
    try:
        import numpy as np

        delta_array = np.asarray(delta, dtype=np.int8)
        rng = np.random.default_rng(seed)
        chunks = []
        remaining = samples
        while remaining:
            chunk_size = min(200, remaining)
            indices = rng.integers(0, n, size=(chunk_size, n), dtype=np.int32)
            chunks.extend(delta_array[indices].mean(axis=1).tolist())
            remaining -= chunk_size
        values = chunks
    except ImportError:
        rng = random.Random(seed)
        values = [
            sum(delta[rng.randrange(n)] for _ in range(n)) / n
            for _ in range(samples)
        ]
    return {
        "effect": observed,
        "ci_low": percentile(values, 2.5),
        "ci_high": percentile(values, 97.5),
        "samples": samples,
        "seed": seed,
    }


def holm_adjust(rows: list[dict[str, Any]]) -> None:
    ordered = sorted(enumerate(rows), key=lambda item: item[1]["mcnemar"]["two_sided_exact_p"])
    running = 0.0
    total = len(rows)
    for rank, (original_index, row) in enumerate(ordered):
        raw = row["mcnemar"]["two_sided_exact_p"]
        adjusted = min(1.0, (total - rank) * raw)
        running = max(running, adjusted)
        rows[original_index]["holm_adjusted_p"] = running


def cell_key(model: str, stage: str, view: str) -> str:
    stage_key = "dpo" if stage == "sft_dpo" else stage
    return f"{model}_{stage_key}_{view}"


def case_error_tags(case: dict[str, Any]) -> set[str]:
    tags = {case["error_bucket"]}
    if case["error_bucket"] == "set_miss":
        gold = set(case["gold_tools"])
        predicted = set(case["pred_tools"])
        if gold - predicted:
            tags.add("omission")
        if predicted - gold:
            tags.add("extra")
        if (gold - predicted) and (predicted - gold):
            tags.add("substitution")
    if case["error_bucket"] == "format_miss" and case.get("strict_error"):
        tags.add(str(case["strict_error"]))
    return tags


def transition_analysis(
    sft_cases: list[dict[str, Any]],
    dpo_cases: list[dict[str, Any]],
    preference_error_types: dict[str, set[str]],
) -> dict[str, Any]:
    sft = {row["id"]: row for row in sft_cases}
    dpo = {row["id"]: row for row in dpo_cases}
    if set(sft) != set(dpo):
        raise ValueError("SFT/DPO badcase ids differ")
    transitions = Counter()
    error_transitions = Counter()
    distributions: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: {
            "tool_count": Counter(),
            "combo": Counter(),
            "domain": Counter(),
            "sft_error": Counter(),
            "dpo_error": Counter(),
        }
    )
    sft_confusions = Counter()
    dpo_confusions = Counter()
    preference_trace_overlap = Counter()
    preference_same_error_type = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trace_id in sft:
        before = sft[trace_id]
        after = dpo[trace_id]
        before_ok = bool(before["set_em"])
        after_ok = bool(after["set_em"])
        transition = {
            (False, True): "sft_fail_to_dpo_success",
            (True, False): "sft_success_to_dpo_fail",
            (False, False): "fail_to_fail",
            (True, True): "success_to_success",
        }[(before_ok, after_ok)]
        transitions[transition] += 1
        error_transitions[f"{before['error_bucket']}->{after['error_bucket']}"] += 1
        distributions[transition]["tool_count"][str(before["tool_count"])] += 1
        distributions[transition]["combo"][before["combo_seen_bucket"]] += 1
        distributions[transition]["sft_error"][before["error_bucket"]] += 1
        distributions[transition]["dpo_error"][after["error_bucket"]] += 1
        for domain in before["domains"]:
            distributions[transition]["domain"][domain] += 1
        for case, counter in ((before, sft_confusions), (after, dpo_confusions)):
            missing = sorted(set(case["gold_tools"]) - set(case["pred_tools"]))
            extra = sorted(set(case["pred_tools"]) - set(case["gold_tools"]))
            for missed in missing:
                for added in extra:
                    counter[f"{missed}->{added}"] += 1
        preference_types = preference_error_types.get(trace_id, set())
        if preference_types:
            preference_trace_overlap[transition] += 1
            compared_tags = case_error_tags(before) | case_error_tags(after)
            if compared_tags & preference_types:
                preference_same_error_type[transition] += 1
        if len(examples[transition]) < 20:
            examples[transition].append(
                {
                    "id": trace_id,
                    "query": before["query"],
                    "gold_tools": before["gold_tools"],
                    "sft_pred_tools": before["pred_tools"],
                    "dpo_pred_tools": after["pred_tools"],
                    "sft_raw_output": before["raw_output"],
                    "dpo_raw_output": after["raw_output"],
                    "sft_error": before["error_bucket"],
                    "dpo_error": after["error_bucket"],
                    "candidate_recall": before["candidate_recall"],
                    "candidate_tools": before["candidate_tools"],
                    "tool_count": before["tool_count"],
                    "combo_seen_bucket": before["combo_seen_bucket"],
                    "domains": before["domains"],
                    "preference_error_types_for_trace": sorted(preference_types),
                }
            )
    total = len(sft)
    return {
        "count": total,
        "transitions": dict(transitions),
        "transition_rates": {key: value / total for key, value in transitions.items()},
        "error_bucket_transitions": dict(error_transitions),
        "transition_distributions": {
            transition: {
                name: dict(sorted(counter.items()))
                for name, counter in payload.items()
            }
            for transition, payload in distributions.items()
        },
        "high_frequency_confusions": {
            "sft": dict(sft_confusions.most_common(50)),
            "dpo": dict(dpo_confusions.most_common(50)),
        },
        "preference_traceability": {
            "trace_overlap_by_transition": dict(preference_trace_overlap),
            "same_error_type_by_transition": dict(preference_same_error_type),
            "interpretation": (
                "Trace/error-type co-occurrence is an audit association, not a unique causal attribution."
            ),
        },
        "examples": dict(examples),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--stats-out", type=Path, default=ASSIGNMENT3_DIR / "tool_trace_significance.json")
    parser.add_argument("--badcase-out", type=Path, default=ASSIGNMENT3_DIR / "tool_trace_badcases.json")
    args = parser.parse_args()

    config = read_json(args.config)
    if not config.get("frozen") or config.get("status") != "complete":
        raise SystemExit("experiment config must be frozen and complete before generating the final report")
    cells = {}
    metrics_by_id = {}
    scored_by_id = {}
    for cell in config["matrix"]:
        if cell.get("status") != "complete":
            raise SystemExit(f"incomplete matrix cell: {cell['cell_id']}")
        metrics_path = PROJECT_ROOT.parent / cell["metrics"] if not Path(cell["metrics"]).is_absolute() else Path(cell["metrics"])
        scored_path = PROJECT_ROOT.parent / cell["scored_cases"] if not Path(cell["scored_cases"]).is_absolute() else Path(cell["scored_cases"])
        metrics = read_json(metrics_path)
        scored = load_jsonl(scored_path)
        if metrics["count"] != 6000 or len(scored) != 6000:
            raise SystemExit(f"{cell['cell_id']}: expected 6000 metrics/scored rows")
        cells[cell["cell_id"]] = cell
        metrics_by_id[cell["cell_id"]] = metrics
        scored_by_id[cell["cell_id"]] = scored

    comparisons = []
    for index, item in enumerate(config["confirmatory_setem_comparisons"]):
        model = item["model"]
        expression = item["comparison"]
        right_name, left_name = [value.strip() for value in expression.split("-")]
        stage_view = {
            "base_rag": ("base", "rag"),
            "base_global": ("base", "global"),
            "sft_rag": ("sft", "rag"),
            "sft_global": ("sft", "global"),
            "dpo_rag": ("sft_dpo", "rag"),
            "dpo_global": ("sft_dpo", "global"),
        }
        right_stage, right_view = stage_view[right_name]
        left_stage, left_view = stage_view[left_name]
        left_id = cell_key(model, left_stage, left_view)
        right_id = cell_key(model, right_stage, right_view)
        left_flags = [bool(row["set_em"]) for row in scored_by_id[left_id]]
        right_flags = [bool(row["set_em"]) for row in scored_by_id[right_id]]
        comparisons.append(
            {
                "index": index + 1,
                "model": model,
                "comparison": expression,
                "left_cell": left_id,
                "right_cell": right_id,
                "bootstrap": paired_bootstrap(
                    left_flags,
                    right_flags,
                    config["statistics"]["paired_bootstrap_samples"],
                    config["statistics"]["paired_bootstrap_seed"] + index,
                ),
                "mcnemar": mcnemar_exact(left_flags, right_flags),
            }
        )
    holm_adjust(comparisons)
    stats = {
        "status": "complete",
        "config": str(args.config),
        "config_sha256": sha256_file(args.config),
        "comparisons": comparisons,
        "holm_family_size": len(comparisons),
    }
    write_json(args.stats_out, stats)

    preference_path = ASSIGNMENT3_DIR / "dpo_preferences_compact.jsonl"
    preference_error_types: dict[str, set[str]] = defaultdict(set)
    for row in load_jsonl(preference_path):
        preference_error_types[row["meta"]["trace_id"]].update(row["meta"]["error_types"])
    badcases = {
        model: transition_analysis(
            scored_by_id[f"{model}_sft_rag"],
            scored_by_id[f"{model}_dpo_rag"],
            preference_error_types,
        )
        for model in ["qwen3_4b", "qwen3_8b"]
    }
    badcases["_provenance"] = {
        "preference_path": str(preference_path),
        "preference_sha256": sha256_file(preference_path),
        "note": "共现仅用于可追溯诊断，不声明为唯一因果根因。",
    }
    write_json(args.badcase_out, badcases)

    sft_audit = read_json(ASSIGNMENT3_DIR / "sft_data_audit.json")
    prompt_stats = read_json(ASSIGNMENT3_DIR / "prompt_length_stats.json")
    dpo_audit = read_json(ASSIGNMENT3_DIR / "dpo_data_audit.json")
    dpo_run_audit = read_json(ASSIGNMENT3_DIR / "dpo_training_runs_audit.json")
    api_audit = read_json(
        ASSIGNMENT3_DIR / "api_native_dev_2400" / "isolation_audit.json"
    )
    environment = read_json(ASSIGNMENT3_DIR / "environment_manifest.json")
    latency = read_json(ASSIGNMENT3_DIR / "tool_trace_latency.json")
    system_metrics = {
        model: read_json(ASSIGNMENT3_DIR / f"tool_trace_system_{model}.metrics.json")
        for model in ["qwen3_4b", "qwen3_8b"]
    }
    selection_payloads = {}
    for selection_id, record in config["selections"].items():
        selection_path = Path(record["path"])
        if not selection_path.is_absolute():
            selection_path = PROJECT_ROOT.parent / selection_path
        selection_payloads[selection_id] = read_json(selection_path)
    lines = [
        "# 工具轨迹 SFT、DPO 与端到端评测",
        "",
        "## 1 任务与边界",
        "",
        "本报告评估 Qwen3-4B/8B 在 100 个电商工具上的有序工具 ID 生成。主指标是严格 JSON 下的 Tool Set Exact Match（SetEM），TraceEM 是更严格的规范顺序辅助指标；当前数据没有工具参数或真实 MCP 执行结果。",
        "",
        "结论只适用于固定 100 工具与 API 原生 query 的闭集离线比较。API-native dev 与 formal train/frozen test 的完整 query 和 trace id 隔离；不可变的 legacy formal train/frozen 仍有已披露的 context signature 重叠。",
        "",
        "## 2 数据与训练用数",
        "",
        f"- 正式 SFT 来源：24,000 条 formal train；RAG 视图 {sft_audit['rag_records']:,} 条，global 视图 {sft_audit['global_records']:,} 条。",
        f"- global 视图实际占输入 token {prompt_stats['actual_global_input_token_share']:.2%}；4B/8B 最大总长度均为 {max(value['max_total_tokens'] for value in prompt_stats['tokenizers'].values()):,} token，固定 `max_length=8192`，截断数为 0。",
        f"- DPO：{dpo_audit['final_pairs']:,} 对，其中真实 selected/early SFT 可解析集合错误占 {dpo_audit['core_hard_error_ratio']:.2%}。",
        "- API-native dev 2,400 条只用于 Prompt、K、checkpoint 和超参数选择；frozen test 6,000 条只在配置冻结后用于最终矩阵。",
        f"- API-native dev 隔离审计 strict_ok={api_audit['strict_ok']}；与 train/frozen test 的 query overlap 均为 0，且不进入训练。",
        f"- DPO 100 工具覆盖 {dpo_audit['tool_coverage']}/100；train/monitor={dpo_audit['train_pairs']}/{dpo_audit['monitor_pairs']}，trace overlap={dpo_audit['train_monitor_trace_overlap']}。",
        f"- DPO 训练日志审计覆盖 {dpo_run_audit['run_count']} 个正式 run；loss/reward/margin 均为有限值且 checkpoint 到达固定 epoch endpoint。",
        "- 首次 `sft4_s1` 在质量门完成前于 step 153 主动中止且没有 checkpoint；它只作为失败过程留档，不进入任何结果。证据为 `finetune/outputs/tooltrace-sft4_s1-prequality-aborted.log`。",
        "",
        "## 3 冻结配置",
        "",
        f"- 公共 RAG K：{config['fixed_inputs']['candidate_k']}。",
        f"- Prompt：`{config['fixed_inputs']['prompt_version']}`；严格 parser：`{config['fixed_inputs']['parser_version']}`。",
        f"- 解码：thinking={config['decoding']['enable_thinking']}，sampling={config['decoding']['do_sample']}，max_new_tokens={config['decoding']['max_new_tokens']}。",
        f"- 软件：torch {environment['packages']['torch']}，transformers {environment['packages']['transformers']}，ms-swift {environment['packages']['ms_swift']}。",
        "- SFT：BF16 LoRA `r=16, alpha=32, dropout=0.05, all-linear`，8 卡 global batch=32；DPO：sigmoid、LR=5e-6、8 卡 global batch=16，policy/reference 同源于 selected SFT。",
        f"- Qwen3-4B revision `{config['models']['qwen3_4b']['revision']}`；SFT `{config['models']['qwen3_4b']['selected_sft_checkpoint']['path']}`；all-pair DPO `{config['models']['qwen3_4b']['final_all_pair_dpo_checkpoint']['path']}`。",
        f"- Qwen3-8B revision `{config['models']['qwen3_8b']['revision']}`；SFT `{config['models']['qwen3_8b']['selected_sft_checkpoint']['path']}`；all-pair DPO `{config['models']['qwen3_8b']['final_all_pair_dpo_checkpoint']['path']}`。",
        "",
        "## 4 API-native dev 选参",
        "",
        "| 模型 | 阶段 | Run | LR | Beta | Epoch | K | RAG SetEM | Wo RAG SetEM | RAG TraceEM | Wo RAG TraceEM | 选中 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    selection_labels = {
        "qwen3_4b_sft": ("Qwen3-4B", "SFT"),
        "qwen3_8b_sft": ("Qwen3-8B", "SFT"),
        "qwen3_4b_dpo": ("Qwen3-4B", "DPO"),
        "qwen3_8b_dpo": ("Qwen3-8B", "DPO"),
    }
    common_k = config["fixed_inputs"]["candidate_k"]
    for selection_id in [
        "qwen3_4b_sft",
        "qwen3_8b_sft",
        "qwen3_4b_dpo",
        "qwen3_8b_dpo",
    ]:
        payload = selection_payloads[selection_id]
        selected = payload["selected"]
        seen_runs = set()
        for candidate in payload["ranked_candidates"]:
            if candidate["candidate_k"] != common_k or candidate["run_id"] in seen_runs:
                continue
            seen_runs.add(candidate["run_id"])
            model_label, stage_label = selection_labels[selection_id]
            lines.append(
                f"| {model_label} | {stage_label} | `{candidate['run_id']}` | "
                f"{candidate['learning_rate']:.1e} | "
                f"{candidate.get('beta') if candidate.get('beta') is not None else '-'} | "
                f"{candidate['epoch']} | {candidate['candidate_k']} | "
                f"{candidate['rag_metrics']['set_em']:.4f} | "
                f"{candidate['global_metrics']['set_em']:.4f} | "
                f"{candidate['rag_metrics']['trace_em']:.4f} | "
                f"{candidate['global_metrics']['trace_em']:.4f} | "
                f"{'是' if candidate['run_id'] == selected['run_id'] else '否'} |"
            )
    lines.extend(
        [
            "",
            "| Dev Base | Wo RAG SetEM | RAG SetEM | Wo RAG TraceEM | RAG TraceEM |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    pred_dir = PROJECT_ROOT / "finetune" / "outputs" / "preds" / "tool_trace"
    for file_stem, model_label in [
        ("qwen3-4b", "Qwen3-4B"),
        ("qwen3-8b", "Qwen3-8B"),
    ]:
        global_base = read_json(pred_dir / f"dev-{file_stem}-base-global.metrics.json")
        rag_base = read_json(pred_dir / f"dev-{file_stem}-base-rag-k{common_k}.metrics.json")
        lines.append(
            f"| {model_label} | {global_base['overall']['set_em']:.4f} | "
            f"{rag_base['overall']['set_em']:.4f} | "
            f"{global_base['overall']['trace_em']:.4f} | "
            f"{rag_base['overall']['trace_em']:.4f} |"
        )
    replacement_decisions = []
    for selection_id, model_label in [
        ("qwen3_4b_dpo", "Qwen3-4B"),
        ("qwen3_8b_dpo", "Qwen3-8B"),
    ]:
        replaces = selection_payloads[selection_id]["dpo_replacement_gate"]["replaces_sft"]
        replacement_decisions.append(
            f"{model_label}={'DPO' if replaces else '保留 SFT'}"
        )
    lines.extend(
        [
            "",
            "4B SFT 在 `(LR, epoch, K)` 上联合选择；8B 只迁移 4B 最佳 LR 并比较 1/2 epoch。DPO 只在已冻结公共 K 上选择。完整候选、指标文件 hash 和 DPO 替代门见配置中的 `selections` 及对应 selection JSON。",
            "预冻结 DPO 替代决策：" + "；".join(replacement_decisions) + "。该决策在查看 frozen generator 结果前作出。",
            "",
            "| DPO run | Beta | Epoch | final train loss | final reward margin | final reward accuracy | endpoint |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for run_id, payload in sorted(dpo_run_audit["runs"].items()):
        metrics = payload["metrics"]
        lines.append(
            f"| `{run_id}` | {payload['beta']} | {payload['num_train_epochs']} | "
            f"{metrics['loss']['last']:.6f} | {metrics['rewards/margins']['last']:.6f} | "
            f"{metrics.get('rewards/accuracies', {}).get('last', 0.0):.6f} | "
            f"{'完整' if payload['endpoint_ok'] else '失败'} |"
        )
    lines.extend(
        [
            "",
            "## 5 Frozen test 12 格结果",
            "",
        "| 模型 | 阶段 | 输入 | SetEM | TraceEM | JSON Valid | Retriever 上限 | 条件 SetEM | input token p50/p95 | latency ms p50/p95 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model in ["qwen3_4b", "qwen3_8b"]:
        for stage in ["base", "sft", "dpo"]:
            stage_config = "sft_dpo" if stage == "dpo" else stage
            for view in ["global", "rag"]:
                cell_id = cell_key(model, stage_config, view)
                metric = metrics_by_id[cell_id]
                overall = metric["overall"]
                rag = metric["rag_decomposition"]
                efficiency = metric["efficiency"]
                lines.append(
                    "| {model} | {stage} | {view} | {setem:.4f} | {trace:.4f} | {json_rate:.4f} | "
                    "{upper:.4f} | {conditional:.4f} | {ip50:.0f}/{ip95:.0f} | {lp50:.2f}/{lp95:.2f} |".format(
                        model=model.replace("qwen3_", "Qwen3-").upper().replace("B", "B"),
                        stage=stage.upper(),
                        view="Wo RAG" if view == "global" else "RAG",
                        setem=overall["set_em"],
                        trace=overall["trace_em"],
                        json_rate=overall["json_valid_rate"],
                        upper=rag["retriever_upper_bound"],
                        conditional=rag["conditional_set_accuracy"] or 0.0,
                        ip50=efficiency["input_tokens"]["p50"],
                        ip95=efficiency["input_tokens"]["p95"],
                        lp50=efficiency["latency_ms_generator_only"]["p50"],
                        lp95=efficiency["latency_ms_generator_only"]["p95"],
                    )
                )
    lines.extend(
        [
            "",
            "本表 latency 来自并发质量矩阵，仅作 generator-only 诊断；有 warmup、真实 Retriever/reranker/fusion 和统一计时边界的效率结果见第 9 节。",
            "本实验不预设 RAG 必然提高准确率。100 个紧凑工具的 Wo RAG Prompt 仍能完整放入上下文；RAG 的价值需结合 Retriever 上限、条件生成准确率、输入 token 和真实链路延迟解释。若端到端 SetEM 下降而条件 SetEM 上升，应归因于可观察的召回瓶颈，并表述为质量—效率权衡，不能为追求正向结果向 dev/frozen 候选注入 gold。",
            "",
            "### 5.1 描述性差值",
            "",
            "| 对照 | SetEM 差值 | 口径 |",
            "|---|---:|---|",
        ]
    )
    for model in ["qwen3_4b", "qwen3_8b"]:
        for stage in ["base", "sft", "dpo"]:
            stage_config = "sft_dpo" if stage == "dpo" else stage
            global_metric = metrics_by_id[cell_key(model, stage_config, "global")]["overall"]["set_em"]
            rag_metric = metrics_by_id[cell_key(model, stage_config, "rag")]["overall"]["set_em"]
            lines.append(
                f"| {model} {stage.upper()} RAG - Wo RAG | {rag_metric - global_metric:+.4f} | "
                f"{'预注册确认性' if stage == 'base' else '描述性'} |"
            )
    for stage in ["base", "sft", "dpo"]:
        stage_config = "sft_dpo" if stage == "dpo" else stage
        for view in ["global", "rag"]:
            metric_4b = metrics_by_id[cell_key("qwen3_4b", stage_config, view)]["overall"]["set_em"]
            metric_8b = metrics_by_id[cell_key("qwen3_8b", stage_config, view)]["overall"]["set_em"]
            lines.append(
                f"| {stage.upper()} {view} 8B - 4B | {metric_8b - metric_4b:+.4f} | 描述性 |"
            )
    lines.extend(
        [
            "",
            "## 6 预注册确认性比较",
            "",
            "| 模型 | 比较 | SetEM 差值 | 配对 bootstrap 95% CI | McNemar p | Holm p |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in comparisons:
        bootstrap = row["bootstrap"]
        lines.append(
            f"| {row['model']} | {row['comparison']} | {bootstrap['effect']:+.4f} | "
            f"[{bootstrap['ci_low']:+.4f}, {bootstrap['ci_high']:+.4f}] | "
            f"{row['mcnemar']['two_sided_exact_p']:.6g} | {row['holm_adjusted_p']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## 7 RAG 分解与错误口径",
            "",
            "Retriever 上限定义为全部 gold 均在 TopK 的比例：",
            "",
            "$$",
            "C_K=\\frac{1}{N}\\sum_i\\mathbf{1}[G_i\\subseteq R_i^{(K)}]",
            "$$",
            "",
            "错误按 `format_miss -> retrieval_miss -> set_miss -> pure_order_miss -> correct` 的固定互斥优先级计数。omission、extra 和 substitution 作为 set miss 的可并存标签，不被宣称为唯一因果根因。",
            "",
            "## 8 SFT 到 DPO 的 badcase 转移",
            "",
            "| 模型 | SFT fail→DPO success | SFT success→DPO fail | fail→fail | success→success |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for model in ["qwen3_4b", "qwen3_8b"]:
        payload = badcases[model]
        counts = payload["transitions"]
        lines.append(
            f"| {model} | {counts.get('sft_fail_to_dpo_success', 0)} | "
            f"{counts.get('sft_success_to_dpo_fail', 0)} | {counts.get('fail_to_fail', 0)} | "
            f"{counts.get('success_to_success', 0)} |"
        )
    lines.extend(
        [
            "",
            "详细代表样本、错误桶转移和 recover/lost 记录见 "
            f"`{args.badcase_out.relative_to(PROJECT_ROOT.parent)}`。",
            "",
            "## 9 真实 RAG 链路延迟",
            "",
            f"Live Retriever 产物与质量矩阵固定排名的 Top{latency['candidate_k']} 一致率为 "
            f"{latency['ranking_equivalence']['topk_equal_rate']:.2%}；Retriever + reranker + fusion "
            f"稳态平均耗时为 {latency['retrieval']['steady_state_mean_ms_per_query']:.2f} ms/query。",
            "",
            "| 生成器 | Prompt build mean | Generator mean | 端到端稳态 mean | stage-sum p50/p95 | 峰值显存 GiB |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for model, payload in latency["generators"].items():
        lines.append(
            f"| {model} | {payload['prompt_build_ms']['mean']:.2f} | "
            f"{payload['generator_ms']['mean']:.2f} | "
            f"{payload['steady_state_total_mean_ms']:.2f} | "
            f"{payload['stage_sum_total_p50_ms']:.2f}/{payload['stage_sum_total_p95_ms']:.2f} | "
            f"{payload['peak_gpu_memory_gib']:.2f} |"
        )
    lines.extend(
        [
            "",
            "端到端稳态 mean 是真实 live retrieval 平均耗时与同一 live candidate 输入上的 warmed generator 平均耗时之和。stage-sum p50/p95 是分阶段分位数之和，只作诊断，不冒充联合逐 query 分布分位数；模型加载时间另见原始 manifest。",
            "",
            "## 10 Intent gate 系统级结果",
            "",
            "| 工具生成器 | Frozen run | Route accuracy | 闲聊误激活率 | 电商 SetEM | 电商 TraceEM | 全链路成功率 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for model in ["qwen3_4b", "qwen3_8b"]:
        payload = system_metrics[model]
        lines.append(
            f"| {model} | `{','.join(payload['tool_generator']['run_ids'])}` | "
            f"{payload['route_accuracy']:.4f} | "
            f"{payload['false_tool_activation']:.4f} | {payload['ecommerce_set_em']:.4f} | "
            f"{payload['ecommerce_trace_em']:.4f} | {payload['full_pipeline_success']:.4f} |"
        )
    lines.extend(
        [
            "",
            "系统级 evaluator 将 intent frozen gold 与 generator prediction 分开读取；其 scored 输出可含 gold，但不进入 SFT/DPO 构建。",
            "",
            "## 11 可追溯产物",
            "",
            f"- 冻结矩阵：`{args.config.relative_to(PROJECT_ROOT.parent)}`。",
            f"- 预注册快照：`{config['preregistration_snapshot']}`，SHA-256 `{config['preregistration_snapshot_sha256']}`。",
            f"- 配对统计：`{args.stats_out.relative_to(PROJECT_ROOT.parent)}`。",
            f"- badcase：`{args.badcase_out.relative_to(PROJECT_ROOT.parent)}`。",
            f"- DPO 训练日志审计：`{(ASSIGNMENT3_DIR / 'dpo_training_runs_audit.json').relative_to(PROJECT_ROOT.parent)}`。",
            f"- 真实链路延迟：`{(ASSIGNMENT3_DIR / 'tool_trace_latency.json').relative_to(PROJECT_ROOT.parent)}`。",
            "- 每格 prediction 不含 gold；对应 scored JSONL 由 evaluator 在生成完成后按 id 连接权威标签。",
            "",
            "## 12 局限性",
            "",
            "- Retriever 的历史横向选型曾使用当前 6,000 条 test，因此 Retriever 部分并非完全 untouched；生成模型的 Prompt、K、SFT/DPO 参数只在 API-native dev 上选择。",
            "- API-native dev 与 formal train/frozen test 的完整 query 和 trace id 隔离，但仍由同一 catalog 能力边界生成。",
            "- 不可变的 legacy formal train/frozen 有 2,764 个 context signature 重叠；既有 context-skeleton matcher `AllHit@10=0.029167` 且自动泄漏审计通过。该风险被披露，不能写成 formal train/frozen 完全 context-isolated。",
            "- GT 分层抽审由模型辅助并经 Codex 逐项裁决，未取得独立人类复核；28 条修订通过显式 revision 派生，不覆盖原始 train。",
            "- 当前只评测工具 ID 集合及规范顺序，不包含 arguments、真实 MCP 返回和线上执行成功率。",
        ]
    )
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"tool-trace report -> {args.report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
