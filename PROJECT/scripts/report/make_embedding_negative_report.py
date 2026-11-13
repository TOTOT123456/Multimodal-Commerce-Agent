#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the archived single-seed explicit-negative diagnostic."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAG_DIR = PROJECT_ROOT / "data" / "rag"
CONFIG_PATH = PROJECT_ROOT / "configs" / "embedding_finetune_experiments.json"
BASELINE_METRICS = RAG_DIR / "_aux" / "rag_baseline" / "metrics.json"
FINETUNE_DIR = RAG_DIR / "_aux" / "embedding_finetune"
ORIGINAL_RUN = "bge-large-mnrl-e1-lr2e-5-bs64"
OUT = RAG_DIR / "_aux" / "embedding_finetune" / "single_seed_negative_diagnostic.md"
SECTION_START = "<!-- BEGIN GENERATED: embedding-explicit-negative -->"
SECTION_END = "<!-- END GENERATED: embedding-explicit-negative -->"
MODE_LABELS = {
    "none": "无显式负例",
    "hard": "同领域、部分替代负例",
    "easy": "跨领域、新增无关负例",
    "all": "同时加入全部显式负例",
}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    return value if isinstance(value, dict) else {}


def metric_at(payload: dict[str, Any], k: int, name: str = "AllHit@K") -> float | None:
    value = payload.get("metrics", {}).get(f"@{k}", {}).get(name)
    return float(value) if isinstance(value, (int, float)) else None


def mrr(payload: dict[str, Any]) -> float | None:
    value = payload.get("metrics", {}).get("overall", {}).get("MRR")
    return float(value) if isinstance(value, (int, float)) else None


def vector_payload(metrics: dict[str, Any], run_name: str) -> dict[str, Any]:
    preferred = metrics.get(f"vector:{run_name}")
    if isinstance(preferred, dict):
        return preferred
    for name, payload in metrics.items():
        if name.startswith("vector:") and isinstance(payload, dict):
            return payload
    return {}


def run_label(run: dict[str, Any]) -> str:
    stats = run["manifest"].get("train_stats", {})
    mode = stats.get("explicit_negative_mode", run["negative_mode"])
    if run["run_name"] == ORIGINAL_RUN:
        return "原 bs64 MNRL，无显式负例"
    if mode == "none":
        return "bs8 无显式负例对照"
    return MODE_LABELS.get(mode, str(mode))


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "待跑"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def delta(value: float | None, baseline: float | None) -> str:
    if value is None or baseline is None:
        return "待跑"
    return f"{value - baseline:+.4f}"


def load_run(run_name: str, negative_mode: str, purpose: str) -> dict[str, Any]:
    run_dir = FINETUNE_DIR / run_name
    manifest = read_json(run_dir / "model" / "training_manifest.json")
    metrics = read_json(run_dir / "retrieval" / "metrics.json")
    payload = vector_payload(metrics, run_name)
    return {
        "run_name": run_name,
        "negative_mode": negative_mode,
        "purpose": purpose,
        "manifest": manifest,
        "payload": payload,
        "available": bool(payload),
    }


def result_line(run: dict[str, Any], original_a10: float | None, control_a10: float | None) -> str:
    manifest = run["manifest"]
    stats = manifest.get("train_stats", {})
    payload = run["payload"]
    a10 = metric_at(payload, 10)
    mode = stats.get("explicit_negative_mode", run["negative_mode"])
    mode_label = run_label(run)
    negatives_per_example = stats.get(
        "explicit_negatives_per_example",
        manifest.get("explicit_negatives_per_example"),
    )
    if negatives_per_example is None and mode == "none":
        negatives_per_example = 0
    micro_batch_size = manifest.get(
        "contrastive_micro_batch_size",
        manifest.get("per_device_train_batch_size"),
    )
    return (
        f"| {mode_label} | {fmt(negatives_per_example)} | "
        f"{fmt(micro_batch_size)} / {fmt(manifest.get('gradient_accumulation_steps'))} / "
        f"{fmt(manifest.get('effective_batch_size'))} | {fmt(manifest.get('train_metrics', {}).get('epoch'))} | "
        f"{fmt(a10)} | {fmt(metric_at(payload, 30))} | {fmt(mrr(payload))} | "
        f"{delta(a10, original_a10)} | {delta(a10, control_a10)} |"
    )


def efficiency_line(run: dict[str, Any]) -> str:
    timing = run["payload"].get("timing", {})
    return (
        f"| {run_label(run).removeprefix('bs8 ')} | "
        f"{fmt(timing.get('encode_avg_ms'))}ms / 条 | "
        f"{fmt(timing.get('search_p50_ms'))} / "
        f"{fmt(timing.get('search_p95_ms'))}ms |"
    )


def comparison_lines(runs: list[dict[str, Any]], original_a10: float | None) -> list[str]:
    completed = [run for run in runs if run["available"]]
    by_mode = {run["negative_mode"]: run for run in completed}
    if not completed:
        return ["- 显式负例实验尚未全部落盘，结论待跑。"]

    lines: list[str] = []
    explicit = [run for run in completed if run["negative_mode"] != "none"]
    if explicit:
        best = max(explicit, key=lambda run: metric_at(run["payload"], 10) or -1.0)
        best_a10 = metric_at(best["payload"], 10)
        lines.append(
            f"- 最强显式负例配置是“{MODE_LABELS.get(best['negative_mode'], best['negative_mode'])}”，"
            f"AllHit@10={fmt(best_a10)}，"
            f"相对原 bs64 MNRL 为 {delta(best_a10, original_a10)}。"
        )
    if "all" in by_mode and "none" in by_mode:
        all_a10 = metric_at(by_mode["all"]["payload"], 10)
        control_a10 = metric_at(by_mode["none"]["payload"], 10)
        lines.append(
            f"- 在相同 bs8/g8 条件下，同时加入全部显式负例相对无显式负例对照的 "
            f"AllHit@10 差值为 "
            f"{delta(all_a10, control_a10)}。"
        )
    if "hard" in by_mode and "easy" in by_mode:
        hard_a10 = metric_at(by_mode["hard"]["payload"], 10)
        easy_a10 = metric_at(by_mode["easy"]["payload"], 10)
        lines.append(
            f"- “同领域、部分替代负例”和“跨领域、新增无关负例”的负例数量相同；"
            f"AllHit@10 分别为 {fmt(hard_a10)} 和 {fmt(easy_a10)}，前者减后者为 "
            f"{delta(hard_a10, easy_a10)}。"
        )
    if len(completed) != len(runs):
        lines.append("- 部分配置仍为待跑，现有比较不是完整四组消融。")
    return lines


def build_report() -> str:
    config = read_json(CONFIG_PATH).get("explicit_negative_ablation", {})
    specs = config.get("runs", [])
    runs = [
        load_run(
            str(spec.get("run_name", "")),
            str(spec.get("negative_mode", "none")),
            str(spec.get("purpose", "")),
        )
        for spec in specs
    ]

    baseline_payload = vector_payload(read_json(BASELINE_METRICS), "bge-large")
    original = load_run(ORIGINAL_RUN, "none", "原 bs64 MNRL 主实验")
    original_a10 = metric_at(original["payload"], 10)
    control = next((run for run in runs if run["negative_mode"] == "none"), None)
    control_a10 = metric_at(control["payload"], 10) if control else None
    shared = config.get("shared_hyperparameters", {})

    lines = [
        SECTION_START,
        "### 显式负例 MNRL 消融",
        "",
        "> 执行日期：2026-07-30",
        "> 训练数据只使用 `PROJECT/data/rag/embedding_grouped_train.jsonl`；冻结测试集仅用于最终检索评测。",
        "> 主指标：严格 `AllHit@K`。本报告只读取真实 manifest/metrics；缺失结果统一显示“待跑”。",
        "",
        "#### 实验设计",
        "",
        "原主实验只把 `query + positive` 输入 MNRL，负例来自 batch 内其他正样本文档。本轮把 grouped 数据中的显式 `neg[]` 直接展开成 `negative_1 ... negative_n`，由 SentenceTransformers MNRL 放入同一 softmax 分母。",
        "",
        "| 配置 | 显式负例 | 含义 |",
        "|---|---:|---|",
        "| 无显式负例对照 | 0 | 同 bs8/g8 的 query-positive 对照，用于隔离 micro-batch 变化 |",
        "| 同领域、部分替代负例 | 2 | 单意图使用同领域负例；多意图使用部分替代负例 |",
        "| 跨领域、新增无关负例 | 2 | 单意图使用跨领域负例；多意图使用新增无关负例 |",
        "| 同时加入全部显式负例 | 4 | 同时使用上述四个类型化负例 |",
        "",
        "同领域和部分替代负例来自数据构造阶段，不是由当前模型在线检索得到的高排名负例。",
        "",
        "共同设置：",
        "",
        f"- 模型：`{config.get('model_name_or_path', '待配置')}`。",
        f"- Epoch/LR：`{shared.get('epochs', '待配置')}` / `{shared.get('learning_rate', '待配置')}`。",
        f"- Micro-batch/梯度累积/优化器有效 batch：`{shared.get('per_device_train_batch_size', '待配置')}` / `{shared.get('gradient_accumulation_steps', '待配置')}` / `{shared.get('optimizer_effective_batch_size', '待配置')}`。",
        f"- Batch sampler：`{shared.get('batch_sampler', '待配置')}`；精度：`{shared.get('precision', '待配置')}`。",
        "",
        "MNRL 的 batch 内负例池只由 micro-batch 决定，梯度累积不会把 8 个 micro-batch 合并成 64 个 in-batch negatives。选择 bs8 是为了让 no-duplicates sampler 在 100 工具闭集、同时加入全部显式负例时每行 5 个候选文档的约束下稳定成批；因此同时保留 bs8 无显式负例对照，并继续报告原 bs64 结果。",
        "",
        "#### 检索结果",
        "",
        "| 负例方案 | 显式负例/pair | micro/grad/effective batch | Actual epoch | AllHit@10 | AllHit@30 | MRR | ΔA10 vs 原bs64 | ΔA10 vs bs8无显式负例 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| zero-shot bge-large | - | - | - | {fmt(metric_at(baseline_payload, 10))} | "
            f"{fmt(metric_at(baseline_payload, 30))} | {fmt(mrr(baseline_payload))} | - | - |"
        ),
        result_line(original, original_a10, control_a10),
    ]
    lines.extend(result_line(run, original_a10, control_a10) for run in runs)
    encode_values = [
        float(run["payload"]["timing"]["encode_avg_ms"])
        for run in runs
        if isinstance(run["payload"].get("timing", {}).get("encode_avg_ms"), (int, float))
    ]
    encode_range = max(encode_values) - min(encode_values) if encode_values else None
    lines += [
        "",
        "#### 推理效率",
        "",
        "四个版本均在 6000 条固定集合上以批量 64 编码，延迟为离线批内均摊；运行使用单张 Tesla V100 32GB。",
        "",
        "| 负例方案 | Query Encoding 均值 | FAISS p50 / p95 |",
        "|---|---:|---:|",
    ]
    lines.extend(efficiency_line(run) for run in runs)
    lines += [
        "",
        (
            f"四种方案的编码均值只相差 {fmt(encode_range)}ms；"
            "当前证据没有显示加入显式负例会造成实质性的推理速度回退。"
            if encode_range is not None
            else "推理效率待完整结果落盘后比较。"
        ),
        "",
        "#### 结论与边界",
        "",
    ]
    lines.extend(comparison_lines(runs, original_a10))
    lines += [
        "",
        "解释边界：",
        "",
        "- 三种显式负例方案与 bs8 无显式负例对照是同设置消融；它们与原 bs64 主实验的差异同时包含显式负例和 contrastive micro-batch 变化，不能只归因于负例。",
        "- 负例均来自训练集构造产物，未使用冻结测试 query、测试标签或测试预测挖掘负例。",
        "- 四组结果仍在既有 6000 条冻结测试集上横向比较，因此适合项目内消融，不等价于从未参与模型选择的新 holdout。",
        "",
        "#### 可追溯产物与复现",
        "",
        "```text",
        "PROJECT/configs/embedding_finetune_experiments.json",
        "PROJECT/scripts/rag/train_embedding.py",
        "PROJECT/scripts/rag/run_embedding_explicit_negative_experiments.sh",
        f"PROJECT/data/rag/_aux/embedding_finetune/{ORIGINAL_RUN}/model/training_manifest.json",
        f"PROJECT/data/rag/_aux/embedding_finetune/{ORIGINAL_RUN}/retrieval/metrics.json",
    ]
    for run in runs:
        run_name = run["run_name"]
        lines.extend(
            [
                f"PROJECT/data/rag/_aux/embedding_finetune/{run_name}/model/training_manifest.json",
                f"PROJECT/data/rag/_aux/embedding_finetune/{run_name}/retrieval/metrics.json",
            ]
        )
    lines += [
        "```",
        "",
        "复现命令：",
        "",
        "```bash",
        "cd /renyuanliu/Agent",
        "CUDA_VISIBLE_DEVICES=0 bash PROJECT/scripts/rag/run_embedding_explicit_negative_experiments.sh",
        "```",
        SECTION_END,
    ]
    return "\n".join(lines) + "\n"


def replace_generated_section(document: str, section: str) -> str:
    start = document.find(SECTION_START)
    end = document.find(SECTION_END)
    if start < 0 or end < 0 or end < start:
        raise RuntimeError(f"generated section markers not found in {OUT}")
    end += len(SECTION_END)
    return document[:start] + section.rstrip() + document[end:]


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build_report(), encoding="utf-8")
    print(f"embedding explicit-negative diagnostic -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
