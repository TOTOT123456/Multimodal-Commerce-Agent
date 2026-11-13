#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit the archived first-pass fixed-Planner replay."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[2]
PROTOCOL = (
    PROJECT / "data/rag/_aux/embedding_downstream_selection/protocol.snapshot.json"
)
BASE_AUX = (
    PROJECT
    / "data"
    / "tool_call_trace"
    / "_aux"
    / "assignment3"
    / "assignment3_crossstyle_rag10"
    / "eval"
)
REPLAY_ROOT = (
    PROJECT
    / "data"
    / "tool_call_trace"
    / "_aux"
    / "assignment3"
    / "embedding_hard2_replay_20260731"
)
REPLAY_EVAL = REPLAY_ROOT / "eval"
BASE_INPUTS = {
    "legacy6000": (
        PROJECT
        / "data"
        / "tool_call_trace"
        / "_aux"
        / "assignment3"
        / "frozen_rag_candidates_k10_label_free.jsonl"
    ),
    "api1000": (
        PROJECT
        / "data"
        / "tool_call_trace"
        / "_aux"
        / "assignment3"
        / "assignment3_100tool_api"
        / "test_api"
        / "rag_k10_label_free.jsonl"
    ),
}
REPLAY_INPUTS = {
    "legacy6000": REPLAY_ROOT / "legacy_rag_k10_label_free.jsonl",
    "api1000": REPLAY_ROOT / "api1000_rag_k10_label_free.jsonl",
}
TRAINING_SUMMARY = (
    PROJECT
    / "data"
    / "tool_call_trace"
    / "_aux"
    / "assignment3"
    / "assignment3_crossstyle_rag10"
    / "final_summary.json"
)
RETRIEVAL_SUMMARY = (
    PROJECT / "data" / "rag" / "_aux" / "embedding_downstream_selection" / "summary.json"
)
SUMMARY_OUT = REPLAY_ROOT / "planner_replay_summary.json"
REPORT_OUT = REPLAY_ROOT / "first_pass_planner_replay_diagnostic.md"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected object: {path}")
    return payload


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_mcnemar_p(improved: int, regressed: int) -> float:
    discordant = improved + regressed
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(improved, regressed) + 1)
    )
    return min(1.0, 2.0 * tail / (1 << discordant))


def paired_boolean(
    baseline: list[dict[str, Any]],
    challenger: list[dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    base = {str(row["id"]): row for row in baseline}
    new = {str(row["id"]): row for row in challenger}
    if set(base) != set(new):
        raise ValueError(f"paired id mismatch for {field}")
    differences = []
    improved = 0
    regressed = 0
    for row_id in sorted(base):
        old_value = bool(base[row_id][field])
        new_value = bool(new[row_id][field])
        differences.append(int(new_value) - int(old_value))
        improved += int(new_value and not old_value)
        regressed += int(old_value and not new_value)
    count = len(differences)
    delta = sum(differences) / count
    variance = (
        sum((value - delta) ** 2 for value in differences) / (count - 1)
        if count > 1
        else 0.0
    )
    half_width = 1.96 * math.sqrt(variance / count)
    return {
        "field": field,
        "count": count,
        "delta": delta,
        "paired_wald_95ci": [delta - half_width, delta + half_width],
        "improved": improved,
        "regressed": regressed,
        "unchanged": count - improved - regressed,
        "exact_mcnemar_p": exact_mcnemar_p(improved, regressed),
    }


def candidate_change_summary(old_path: Path, new_path: Path) -> dict[str, Any]:
    old = {str(row["id"]): row for row in load_jsonl(old_path)}
    new = {str(row["id"]): row for row in load_jsonl(new_path)}
    if set(old) != set(new):
        raise ValueError(f"candidate input id mismatch: {old_path} vs {new_path}")
    same_order = 0
    same_set = 0
    overlaps = []
    for row_id in old:
        old_candidates = old[row_id]["candidate_tools"]
        new_candidates = new[row_id]["candidate_tools"]
        same_order += int(old_candidates == new_candidates)
        same_set += int(set(old_candidates) == set(new_candidates))
        overlaps.append(len(set(old_candidates) & set(new_candidates)))
    count = len(old)
    return {
        "count": count,
        "same_order_count": same_order,
        "same_order_rate": same_order / count,
        "same_set_count": same_set,
        "same_set_rate": same_set / count,
        "mean_top10_set_overlap": sum(overlaps) / count,
        "min_top10_set_overlap": min(overlaps),
    }


def retrieval_trace_transition(
    baseline: list[dict[str, Any]],
    challenger: list[dict[str, Any]],
) -> dict[str, Any]:
    old = {str(row["id"]): row for row in baseline}
    new = {str(row["id"]): row for row in challenger}
    buckets: dict[str, dict[str, int]] = {}
    for row_id in sorted(old):
        old_recall = bool(old[row_id]["candidate_recall"])
        new_recall = bool(new[row_id]["candidate_recall"])
        if old_recall and new_recall:
            bucket = "both_retrieved"
        elif not old_recall and new_recall:
            bucket = "retrieval_rescued"
        elif old_recall and not new_recall:
            bucket = "retrieval_lost"
        else:
            bucket = "both_missed"
        row = buckets.setdefault(
            bucket,
            {"count": 0, "trace_improved": 0, "trace_regressed": 0, "trace_unchanged": 0},
        )
        row["count"] += 1
        old_trace = bool(old[row_id]["trace_em"])
        new_trace = bool(new[row_id]["trace_em"])
        row["trace_improved"] += int(new_trace and not old_trace)
        row["trace_regressed"] += int(old_trace and not new_trace)
        row["trace_unchanged"] += int(old_trace == new_trace)
    return buckets


def metric_paths(model: int, split: str) -> dict[str, Path]:
    base_stem = f"a3-crossstyle-{model}b-{split}"
    new_stem = f"hard2-crossstyle-{model}b-{split}"
    paths = {
        "baseline_metrics": BASE_AUX / f"{base_stem}.metrics.json",
        "baseline_scored": BASE_AUX / f"{base_stem}.scored.jsonl",
        "challenger_metrics": REPLAY_EVAL / f"{new_stem}.metrics.json",
        "challenger_scored": REPLAY_EVAL / f"{new_stem}.scored.jsonl",
    }
    if split == "legacy6000":
        paths.update(
            {
                "baseline_system_metrics": BASE_AUX / f"{base_stem}.system.metrics.json",
                "baseline_system": BASE_AUX / f"{base_stem}.system.jsonl",
                "challenger_system_metrics": REPLAY_EVAL / f"{new_stem}.system.metrics.json",
                "challenger_system": REPLAY_EVAL / f"{new_stem}.system.jsonl",
            }
        )
    return paths


def compact_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    overall = payload["overall"]
    rag = payload["rag_decomposition"]
    return {
        "count": payload["count"],
        "set_em": overall["set_em"],
        "trace_em": overall["trace_em"],
        "strict_output_valid_rate": overall["strict_output_valid_rate"],
        "retrieval_allhit_at_10": rag["retriever_upper_bound"],
        "conditional_set_accuracy": rag["conditional_set_accuracy"],
        "conditional_trace_accuracy": rag["conditional_trace_accuracy"],
    }


def build_summary() -> dict[str, Any]:
    protocol = load_json(PROTOCOL)
    artifacts: list[Path] = [
        PROTOCOL,
        TRAINING_SUMMARY,
        RETRIEVAL_SUMMARY,
        *BASE_INPUTS.values(),
        *REPLAY_INPUTS.values(),
        REPLAY_ROOT / "legacy_label_free_k10_audit.json",
        REPLAY_ROOT / "api1000_label_free_k10_audit.json",
        REPLAY_ROOT / "api1000_retrieval_manifest.json",
    ]
    results: dict[str, dict[str, Any]] = {}
    for model in (4, 8):
        model_result: dict[str, Any] = {}
        for split in ("legacy6000", "api1000"):
            paths = metric_paths(model, split)
            artifacts.extend(paths.values())
            baseline_metrics = load_json(paths["baseline_metrics"])
            challenger_metrics = load_json(paths["challenger_metrics"])
            baseline_scored = load_jsonl(paths["baseline_scored"])
            challenger_scored = load_jsonl(paths["challenger_scored"])
            split_result: dict[str, Any] = {
                "baseline": compact_metrics(baseline_metrics),
                "challenger": compact_metrics(challenger_metrics),
                "paired_set": paired_boolean(baseline_scored, challenger_scored, "set_em"),
                "paired_trace": paired_boolean(baseline_scored, challenger_scored, "trace_em"),
                "paired_strict": paired_boolean(
                    baseline_scored,
                    challenger_scored,
                    "strict_parse_ok",
                ),
                "retrieval_trace_transitions": retrieval_trace_transition(
                    baseline_scored,
                    challenger_scored,
                ),
                "paths": {name: str(path) for name, path in paths.items()},
            }
            if split == "legacy6000":
                baseline_system_metrics = load_json(paths["baseline_system_metrics"])
                challenger_system_metrics = load_json(paths["challenger_system_metrics"])
                baseline_system = load_jsonl(paths["baseline_system"])
                challenger_system = load_jsonl(paths["challenger_system"])
                split_result["system"] = {
                    "baseline_full_set": baseline_system_metrics["full_pipeline_success"],
                    "challenger_full_set": challenger_system_metrics["full_pipeline_success"],
                    "baseline_full_trace": baseline_system_metrics["full_pipeline_trace_success"],
                    "challenger_full_trace": challenger_system_metrics["full_pipeline_trace_success"],
                    "paired_full_set": paired_boolean(
                        baseline_system,
                        challenger_system,
                        "full_pipeline_success",
                    ),
                    "paired_full_trace": paired_boolean(
                        baseline_system,
                        challenger_system,
                        "full_pipeline_trace_success",
                    ),
                }
                split_result["tool_count"] = {
                    bucket: {
                        "count": row["count"],
                        "baseline_trace_em": row["trace_em"],
                        "challenger_trace_em": challenger_metrics["buckets"]["tool_count"][bucket]["trace_em"],
                        "delta": (
                            challenger_metrics["buckets"]["tool_count"][bucket]["trace_em"]
                            - row["trace_em"]
                        ),
                    }
                    for bucket, row in baseline_metrics["buckets"]["tool_count"].items()
                }
            model_result[split] = split_result
        results[str(model)] = model_result

    candidate_changes = {
        split: candidate_change_summary(BASE_INPUTS[split], REPLAY_INPUTS[split])
        for split in ("legacy6000", "api1000")
    }
    gate_rows = []

    def add_gate(name: str, observed: float, threshold: float, relation: str, detail: str) -> None:
        if relation == "ge":
            passed = observed >= threshold
        else:
            raise ValueError(f"unsupported relation: {relation}")
        gate_rows.append(
            {
                "name": name,
                "observed": observed,
                "threshold": threshold,
                "relation": relation,
                "passed": passed,
                "detail": detail,
            }
        )

    deployment = results["8"]
    add_gate(
        "8B Legacy TraceEM does not decrease",
        deployment["legacy6000"]["challenger"]["trace_em"] - deployment["legacy6000"]["baseline"]["trace_em"],
        0.0,
        "ge",
        "primary tool-split metric",
    )
    add_gate(
        "8B Full Trace does not decrease",
        (
            deployment["legacy6000"]["system"]["challenger_full_trace"]
            - deployment["legacy6000"]["system"]["baseline_full_trace"]
        ),
        0.0,
        "ge",
        "9000-row static route plus Planner metric",
    )
    add_gate(
        "8B API TraceEM delta >= -0.2pp",
        deployment["api1000"]["challenger"]["trace_em"] - deployment["api1000"]["baseline"]["trace_em"],
        -0.002,
        "ge",
        "cross-protocol regression guardrail",
    )
    for split in ("legacy6000", "api1000"):
        add_gate(
            f"8B {split} strict-output delta >= -0.2pp",
            (
                deployment[split]["challenger"]["strict_output_valid_rate"]
                - deployment[split]["baseline"]["strict_output_valid_rate"]
            ),
            -0.002,
            "ge",
            "format and candidate-escape guardrail",
        )
    for bucket, row in deployment["legacy6000"]["tool_count"].items():
        if row["count"] >= 100:
            add_gate(
                f"8B Legacy tool_count={bucket} TraceEM delta >= -0.5pp",
                row["delta"],
                -0.005,
                "ge",
                f"n={row['count']}",
            )
    gate_passed = all(row["passed"] for row in gate_rows)
    training = load_json(TRAINING_SUMMARY)["training"]
    return {
        "schema_version": 1,
        "status": "complete",
        "date": "2026-07-31",
        "scope": "fixed 4B/8B Planner replay under hard2 retrieval Top10",
        "is_real_tool_execution_success": False,
        "protocol": str(PROTOCOL),
        "retrieval_winner": load_json(RETRIEVAL_SUMMARY)["winner"],
        "candidate_changes": candidate_changes,
        "results": results,
        "promotion_gate": {
            "passed": gate_passed,
            "decision": (
                "promote_hard2_to_canonical"
                if gate_passed
                else "retain_original_canonical_and_keep_hard2_as_challenger"
            ),
            "rows": gate_rows,
        },
        "clean_retrain_workload_reference": {
            "training_rows": training["common"]["train_rows"],
            "gpu_count_per_run": training["common"]["gpu_count"],
            "qwen3_4b_train_runtime_seconds": training["qwen3_4b"]["train_runtime_seconds"],
            "qwen3_4b_train_runtime_human": training["qwen3_4b"]["train_runtime_human"],
            "qwen3_8b_train_runtime_seconds": training["qwen3_8b"]["train_runtime_seconds"],
            "qwen3_8b_train_runtime_human": training["qwen3_8b"]["train_runtime_human"],
            "serial_historical_runtime_seconds": (
                training["qwen3_4b"]["train_runtime_seconds"]
                + training["qwen3_8b"]["train_runtime_seconds"]
            ),
            "note": "Historical two-GPU wall times only; current 3090 throughput is not assumed equal.",
        },
        "input_hashes": {
            str(path): sha256_file(path)
            for path in sorted(set(artifacts))
        },
    }


def pct(value: float, digits: int = 4) -> str:
    return f"{value * 100:.{digits}f}%"


def pp(value: float) -> str:
    return f"{value * 100:+.4f}pp"


def p_value(value: float) -> str:
    return f"{value:.3e}" if value < 0.001 else f"{value:.4f}"


def format_seconds(value: float) -> str:
    hours = int(value // 3600)
    minutes = int((value % 3600) // 60)
    seconds = int(round(value % 60))
    return f"{hours}h{minutes:02d}m{seconds:02d}s"


def build_report(summary: dict[str, Any]) -> str:
    results = summary["results"]
    gate = summary["promotion_gate"]
    deploy = results["8"]
    legacy = deploy["legacy6000"]
    api = deploy["api1000"]
    system = legacy["system"]
    failed = [row for row in gate["rows"] if not row["passed"]]
    lines = [
        "# hard2 检索链：固定 Planner 与 Full Trace 回放",
        "",
        "> 日期：2026-07-31  ",
        "> 固定项：4B/8B 跨风格 SFT `checkpoint-2429`、RAG Top10、original 顺序、greedy 解码。  ",
        "> 变化项：只把原 embedding 检索完整排序替换为 hard2 完整链排序。",
        "",
        "## 结论",
        "",
        f"- 8B Legacy TraceEM 从 {pct(legacy['baseline']['trace_em'])} 到 {pct(legacy['challenger']['trace_em'])}（{pp(legacy['challenger']['trace_em'] - legacy['baseline']['trace_em'])}）；9k Full Trace 从 {pct(system['baseline_full_trace'])} 到 {pct(system['challenger_full_trace'])}（{pp(system['challenger_full_trace'] - system['baseline_full_trace'])}）。",
        f"- API-native 1k 的 8B TraceEM 保持 {pct(api['challenger']['trace_em'])}；Legacy 检索 AllHit@10 从 {pct(legacy['baseline']['retrieval_allhit_at_10'])} 到 {pct(legacy['challenger']['retrieval_allhit_at_10'])}。",
        f"- **预注册 promotion gate：{'通过' if gate['passed'] else '未通过'}。当前决策是 `{gate['decision']}`。**",
    ]
    if failed:
        lines.append(
            "- 失败项："
            + "；".join(
                f"{row['name']}，实测 {pp(row['observed'])}，门槛 {pp(row['threshold'])}"
                for row in failed
            )
            + "。"
        )
    lines += [
        "- 因而 hard2 是当前完整检索链的 point-estimate 赢家，但**不替换 canonical Planner 链**；总体微增掩盖了长链局部回退。",
        "",
        "## 固定 Planner 回放结果",
        "",
        "| 模型 | Split | Retrieval A10：原→新 | SetEM：原→新 | TraceEM：原→新 | Strict：原→新 | Trace 改善/回退 | McNemar p |",
        "|---|---|---|---|---|---|---:|---:|",
    ]
    for model in ("4", "8"):
        for split in ("api1000", "legacy6000"):
            row = results[model][split]
            old = row["baseline"]
            new = row["challenger"]
            paired = row["paired_trace"]
            lines.append(
                f"| {model}B | {split} | "
                f"{pct(old['retrieval_allhit_at_10'], 2)}→{pct(new['retrieval_allhit_at_10'], 2)} | "
                f"{pct(old['set_em'])}→{pct(new['set_em'])} | "
                f"{pct(old['trace_em'])}→{pct(new['trace_em'])} | "
                f"{pct(old['strict_output_valid_rate'])}→{pct(new['strict_output_valid_rate'])} | "
                f"{paired['improved']}/{paired['regressed']} | {p_value(paired['exact_mcnemar_p'])} |"
            )
    lines += [
        "",
        "9k 静态系统指标由逐样本联合判断得到，不是 Intent 与 Planner 聚合准确率相乘：",
        "",
        "| 模型 | Full Set：原→新 | Full Trace：原→新 | Trace 改善/回退 |",
        "|---|---|---|---:|",
    ]
    for model in ("4", "8"):
        row = results[model]["legacy6000"]["system"]
        paired = row["paired_full_trace"]
        lines.append(
            f"| {model}B | {pct(row['baseline_full_set'])}→{pct(row['challenger_full_set'])} | "
            f"{pct(row['baseline_full_trace'])}→{pct(row['challenger_full_trace'])} | "
            f"{paired['improved']}/{paired['regressed']} |"
        )
    lines += [
        "",
        "## 8B Legacy 工具数分层 guardrail",
        "",
        "| Gold 工具数 | n | TraceEM：原→新 | Δ | Gate |",
        "|---:|---:|---|---:|---|",
    ]
    for bucket, row in sorted(legacy["tool_count"].items(), key=lambda item: int(item[0])):
        passed = row["delta"] >= -0.005
        lines.append(
            f"| {bucket} | {row['count']} | {pct(row['baseline_trace_em'])}→"
            f"{pct(row['challenger_trace_em'])} | {pp(row['delta'])} | "
            f"{'PASS' if passed else '**FAIL**'} |"
        )
    lines += [
        "",
        "5 工具层从 "
        f"{pct(legacy['tool_count']['5']['baseline_trace_em'])} 降到 "
        f"{pct(legacy['tool_count']['5']['challenger_trace_em'])}，"
        f"回退 {pp(legacy['tool_count']['5']['delta'])}。这不是检索 Top10 总覆盖率能揭示的问题，"
        "而是新候选集合/顺序与既有 Planner 训练分布的交互。",
        "",
        "## 候选变化与检索—规划耦合",
        "",
        "| Split | Top10 顺序完全相同 | Top10 集合相同 | 平均集合交集 / 10 | 最小交集 |",
        "|---|---:|---:|---:|---:|",
    ]
    for split, row in summary["candidate_changes"].items():
        lines.append(
            f"| {split} | {pct(row['same_order_rate'])} | {pct(row['same_set_rate'])} | "
            f"{row['mean_top10_set_overlap']:.4f} | {row['min_top10_set_overlap']} |"
        )
    transitions = legacy["retrieval_trace_transitions"]
    lines += [
        "",
        "8B Legacy 的逐样本拆分：",
        "",
        "| 检索状态 | n | Planner Trace 改善 | Planner Trace 回退 |",
        "|---|---:|---:|---:|",
    ]
    for name in ("retrieval_rescued", "retrieval_lost", "both_retrieved", "both_missed"):
        row = transitions.get(
            name,
            {"count": 0, "trace_improved": 0, "trace_regressed": 0},
        )
        lines.append(
            f"| {name} | {row['count']} | {row['trace_improved']} | {row['trace_regressed']} |"
        )
    lines += [
        "",
        "## 是否需要重做后续训练",
        "",
        "本轮已经完成无需重新训练即可公平完成的部分：五个 embedding 候选统一 first-stage、Top50 task-reranker 重排、最终融合；并对 hard2 重建 API/Legacy label-free Top10，回放固定 4B/8B Planner 和 9k Full Trace。",
        "",
        "- **任务内 reranker 不必因 embedding 变化而重训。** 它的训练集是 `embedding_grouped_train.jsonl` 的动态 1 正 + 7 负 groupwise CE，不是从原 embedding Top50 采出来的监督；但本轮已对每个候选自己的 Top50 全量重新打分。",
        "- **Planner 若要让 hard2 正式替换 canonical，建议重训。** 现有 Planner 的 77,724 条 SFT 输入使用原检索链 Top10，5 工具层回退说明只做 inference replay 不够。",
        "- 不能在已经观察过的 Legacy 6k/API 1k 上重训后继续把它们当作新终评。应先冻结新的 selection/final split，再重建 hard2 训练候选、重训 4B/8B，并做一次性终评。",
        "",
        "历史同规模训练工作量（两个 GPU/每个 run，硬件吞吐只作参考）：",
        "",
        f"- 4B：{summary['clean_retrain_workload_reference']['qwen3_4b_train_runtime_human']}。",
        f"- 8B：{summary['clean_retrain_workload_reference']['qwen3_8b_train_runtime_human']}。",
        f"- 串行合计：{format_seconds(summary['clean_retrain_workload_reference']['serial_historical_runtime_seconds'])}；若资源足够并行，训练墙钟下界约由 8B 的近 5 小时决定，另加候选重建、审计和双冻结评测。",
        "",
        "因此，当前最稳妥的工程决策是：保留原 canonical 链；把 hard2、参数 `first-stage N=50/w=0.97/k=30`、`final w=0.6/k=1` 固化为下一轮 Planner 重训 challenger。",
        "",
        "## 可追溯产物",
        "",
        f"- 结构化汇总：`{SUMMARY_OUT}`",
        f"- Legacy label-free 审计：`{REPLAY_ROOT / 'legacy_label_free_k10_audit.json'}`",
        f"- API label-free 审计：`{REPLAY_ROOT / 'api1000_label_free_k10_audit.json'}`",
        f"- Planner 预测：`{PROJECT / 'finetune' / 'outputs' / 'preds' / 'tool_trace' / 'embedding_hard2_replay_20260731'}`",
        f"- 逐条评分与系统指标：`{REPLAY_EVAL}`",
        "- 所有关键输入 SHA-256 已写入结构化汇总。",
        "",
        "复现汇总：",
        "",
        "```bash",
        "cd /renyuanliu/Agent",
        "python3 PROJECT/scripts/report/analyze_embedding_planner_replay.py",
        "```",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    summary = build_summary()
    SUMMARY_OUT.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    REPORT_OUT.write_text(build_report(summary), encoding="utf-8")
    print(f"promotion_gate={summary['promotion_gate']['passed']}")
    print(f"decision={summary['promotion_gate']['decision']}")
    print(f"summary -> {SUMMARY_OUT}")
    print(f"report -> {REPORT_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
