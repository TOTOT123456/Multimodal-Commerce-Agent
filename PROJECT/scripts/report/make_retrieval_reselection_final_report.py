#!/usr/bin/env python3
"""Generate the final Chinese report for the frozen retrieval re-selection."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / "PROJECT"
RUN_ROOT = PROJECT / "data/rag/_aux/retrieval_pipeline_reselection"
DEFAULT_REPORT = RUN_ROOT / "canonical_retrieval_report.md"
DEFAULT_SUMMARY = RUN_ROOT / "final_summary.json"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float) -> str:
    return f"{100.0 * float(value):.4f}%"


def pp(value: float) -> str:
    return f"{100.0 * float(value):+.4f}pp"


def mean_std(metric: dict[str, Any]) -> str:
    return f"{pct(metric['mean'])} ± {100.0 * float(metric['population_std']):.4f}pp"


def route_text(route: dict[str, Any]) -> str:
    return (
        f"一级 N={route['first_candidate_n']}、vector_weight={route['first_vector_weight']}, "
        f"RRF-k={route['first_rrf_k']}；Reranker Top-{route['candidate_k']}；"
        f"二级 source_weight={route['second_source_weight']}、RRF-k={route['second_rrf_k']}"
    )


def metric_row(label: str, metrics: dict[str, Any]) -> str:
    return (
        f"| {label} | {pct(metrics['allhit_at_6'])} | {pct(metrics['allhit_at_10'])} | "
        f"{pct(metrics['allhit_at_20'])} | {pct(metrics['allhit_at_30'])} | "
        f"{pct(metrics['allhit_at_50'])} | {float(metrics['mrr']):.6f} |"
    )


def evaluation_payload(label: str) -> dict[str, Any]:
    directory = RUN_ROOT / "evaluation" / label
    manifest_path = directory / "evaluation_manifest.json"
    for _ in range(120):
        if manifest_path.exists():
            manifest = load_json(manifest_path)
            if manifest.get("status") == "complete_no_post_evaluation_tuning":
                break
        time.sleep(1)
    else:
        raise RuntimeError(f"evaluation manifest did not complete: {manifest_path}")
    return {
        "manifest": manifest,
        "challenger": load_json(directory / "challenger/metrics.json"),
        "baseline": load_json(directory / "baseline/metrics.json"),
        "comparison": load_json(directory / "comparison.json"),
    }


def canonical_report(
    *,
    frozen: dict[str, Any],
    embedding: dict[str, Any],
    first_stage: dict[str, Any],
    architecture: dict[str, Any],
    negative: dict[str, Any],
    round_final: dict[str, Any],
    legacy: dict[str, Any],
    api: dict[str, Any],
    benchmark: dict[str, Any],
) -> str:
    """Render only the canonical route and its absolute metrics."""
    route = frozen["selected"]["route"]
    reranker = frozen["reranker"]
    lines = [
        "# 检索全链路 canonical",
        "",
        "> Embedding、一级融合、候选深度、Reranker 和二级融合按顺序选择并冻结；本报告定义项目唯一检索路线。",
        "",
        "## Canonical 定义",
        "",
        f"- Embedding：`{frozen['embedding']['recipe']}`，seed `{frozen['embedding']['seed']}`。",
        f"- 一级检索：N={route['first_candidate_n']}，vector_weight={route['first_vector_weight']}，RRF-k={route['first_rrf_k']}。",
        f"- Reranker：`{reranker['base_model']}`，负例 `{reranker['negative_recipe']}`，lr `{reranker['learning_rate']}`，Top-{route['candidate_k']}。",
        f"- 二级融合：source_weight={route['second_source_weight']}，RRF-k={route['second_rrf_k']}。",
        "- 机器可读配置：`PROJECT/configs/retrieval_canonical_route.json`。",
        "",
        "## Embedding 三随机种子",
        "",
        "| Recipe | AllHit@10 | AllHit@20 | AllHit@50 | MRR |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in embedding["recipes"]:
        aggregate = row["aggregate"]
        lines.append(
            f"| {row['recipe']} | {mean_std(aggregate['allhit_at_10'])} | "
            f"{mean_std(aggregate['allhit_at_20'])} | {mean_std(aggregate['allhit_at_50'])} | "
            f"{aggregate['mrr']['mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 下游选择",
            "",
            f"一级网格 {first_stage['grid_size']} 组，得到 {first_stage['unique_ranking_count']} 个唯一排名；保留 {first_stage['retained_size']} 个候选进入联合搜索。",
            "",
            "| Reranker 架构 | Mean AllHit@6 | Mean AllHit@10 |",
            "|---|---:|---:|",
        ]
    )
    for row in architecture["selected_by_group"]:
        lines.append(
            f"| {row['group']} | {pct(row['aggregate']['allhit_at_6']['mean'])} | "
            f"{pct(row['aggregate']['allhit_at_10']['mean'])} |"
        )
    lines.extend(
        [
            "",
            "| Reranker 负例 | Mean AllHit@6 | Mean AllHit@10 |",
            "|---|---:|---:|",
        ]
    )
    for row in negative["selected_by_group"]:
        lines.append(
            f"| {row['group']} | {pct(row['aggregate']['allhit_at_6']['mean'])} | "
            f"{pct(row['aggregate']['allhit_at_10']['mean'])} |"
        )
    lines.extend(
        [
            "",
            "## Canonical 绝对指标",
            "",
            "| 数据集 | n | AllHit@6 | AllHit@10 | AllHit@20 | MRR |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for label, payload in (
        ("Round-final", round_final),
        ("Legacy frozen", legacy),
        ("API frozen", api),
    ):
        metrics = payload["challenger"]["overall"]
        lines.append(
            f"| {label} | {payload['manifest']['rows']} | {pct(metrics['allhit_at_6'])} | "
            f"{pct(metrics['allhit_at_10'])} | {pct(metrics['allhit_at_20'])} | "
            f"{metrics['mrr']:.6f} |"
        )
    latency = benchmark["latency_ms_per_query"]["complete_path"]
    lines.extend(
        [
            "",
            "## 端到端效率",
            "",
            f"单张 RTX 3090 warmed benchmark：{benchmark['throughput_queries_per_second']:.2f} query/s，p50 {latency['p50']:.4f}ms，p95 {latency['p95']:.4f}ms，GPU 峰值 allocated {benchmark['hardware']['gpu']['peak_allocated_gib']:.3f}GiB；Top-10 一致率 {pct(benchmark['ranking_equivalence']['top10_equal_rate'])}。",
            "",
            "结果只适用于 100 工具合成闭集。完整 Planner 回放见 `PROJECT/reports/06_Assignment3_100工具实验总报告.md`。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    protocol = load_json(RUN_ROOT / "protocol.snapshot.json")
    integrity_audit = load_json(RUN_ROOT / "integrity_audit.json")
    embedding = load_json(RUN_ROOT / "embedding_selection.json")
    first_stage = load_json(RUN_ROOT / "downstream/first_stage_search/first_stage_search.json")
    lr_screens = {
        key: load_json(RUN_ROOT / f"downstream/lr_screen/{key}/pipeline_search.json")
        for key in ("bge_base", "bge_large", "bge_v2_m3", "qwen3_0p6b")
    }
    architecture = load_json(RUN_ROOT / "downstream/architecture_screen/pipeline_search.json")
    negative = load_json(RUN_ROOT / "downstream/negative_screen/pipeline_search.json")
    interaction_plan = load_json(RUN_ROOT / "downstream/negative_interaction_plan.json")
    interaction = (
        load_json(
            RUN_ROOT / "downstream/negative_interaction_screen/pipeline_search.json"
        )
        if interaction_plan["required"]
        else None
    )
    frozen = load_json(RUN_ROOT / "frozen_pipeline.json")
    round_final = evaluation_payload("round_final")
    legacy = evaluation_payload("legacy_frozen_6k")
    api = evaluation_payload("api_frozen_1k")
    benchmark_challenger = load_json(
        RUN_ROOT / "evaluation/round_final/benchmark/challenger.json"
    )
    benchmark_baseline = load_json(
        RUN_ROOT / "evaluation/round_final/benchmark/baseline.json"
    )

    primary = round_final["comparison"]["overall"]["metrics"]["allhit_at_10"]
    gate = round_final["comparison"]["promotion_gate"]
    promoted = bool(gate["passed"])
    decision = "canonical_explicit_negative_route"
    selected_route = frozen["selected"]["route"]
    summary = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": protocol["scope"],
        "integrity_audit": integrity_audit,
        "decision": decision,
        "promotion_gate_passed": promoted,
        "development_selection": {
            "embedding": {
                "selected": embedding["selected"],
                "recipes": [
                    {"recipe": row["recipe"], "aggregate": row["aggregate"]}
                    for row in embedding["recipes"]
                ],
            },
            "first_stage": {
                "grid_size": first_stage["grid_size"],
                "unique_ranking_count": first_stage["unique_ranking_count"],
                "retained_size": first_stage["retained_size"],
                "mining_source_selected": first_stage["mining_source_selected"],
            },
            "learning_rates": {
                key: payload["selected"] for key, payload in lr_screens.items()
            },
            "architecture": {
                "selected": architecture["selected"],
                "candidates": architecture["selected_by_group"],
            },
            "negative_recipe": {
                "selected": negative["selected"],
                "candidates": negative["selected_by_group"],
            },
            "interaction_plan": interaction_plan,
            "interaction": (
                {
                    "selected": interaction["selected"],
                    "candidates": interaction["selected_by_group"],
                }
                if interaction is not None
                else None
            ),
        },
        "embedding": frozen["embedding"],
        "reranker": frozen["reranker"],
        "route": selected_route,
        "round_final": {
            "challenger": round_final["challenger"]["overall"],
            "baseline": round_final["baseline"]["overall"],
            "comparison": primary,
            "promotion_gate": gate,
        },
        "regression": {
            "legacy_frozen_6k": legacy["manifest"]["results"],
            "api_frozen_1k": api["manifest"]["results"],
        },
        "benchmark": {
            "challenger": benchmark_challenger,
            "baseline": benchmark_baseline,
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    reranker = frozen["reranker"]
    decision_cn = "晋升新链为检索 canonical" if promoted else "保留原 bs64 检索 canonical"
    lines = [
        "# 检索全链路重新选型",
        "",
        "> 实验：2026-07-31 至 2026-08-01；范围：100 工具合成闭集。Embedding → 一级融合 → 候选深度 → Reranker → 二级融合按顺序冻结；本轮不包含 Planner 重训或 Planner 选型。",
        "",
        "## 结论",
        "",
        f"- Embedding 获胜 recipe：`{frozen['embedding']['recipe']}`，部署 seed `{frozen['embedding']['seed']}`。",
        f"- Reranker：`{reranker['architecture']}`（基座 `{reranker['base_model']}`），负例 `{reranker['negative_recipe']}`，seed `{reranker['seed']}`，lr `{reranker['learning_rate']}`。",
        f"- 冻结路线：{route_text(selected_route)}。",
        f"- round-final AllHit@10：原链 {pct(primary['baseline'])}，新链 {pct(primary['challenger'])}，差值 {pp(primary['point_difference'])}，95% CI [{pp(primary['ci95_low'])}, {pp(primary['ci95_high'])}]。",
        f"- 预注册 promotion gate：`{str(promoted).lower()}`；决策：**{decision_cn}**。",
        "- 结论只适用于当前 100 工具合成闭集；round-final 曾用于旧 Planner/DPO 项目，但本轮新检索候选在整链冻结前未读取该 split。",
        "",
        "## 1. Embedding：五种 recipe、三随机种子",
        "",
        "选择先看下游允许的最大接口 AllHit@50，再依次看 AllHit@20、AllHit@10、MRR 与 worst seed；同一 BGE-large 架构不使用并发测得的毫秒噪声破坏平局。",
        "",
        "| Recipe | AllHit@10 | AllHit@20 | AllHit@50 | MRR |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in embedding["recipes"]:
        aggregate = row["aggregate"]
        lines.append(
            f"| {row['recipe']} | {mean_std(aggregate['allhit_at_10'])} | "
            f"{mean_std(aggregate['allhit_at_20'])} | {mean_std(aggregate['allhit_at_50'])} | "
            f"{aggregate['mrr']['mean']:.6f} ± {aggregate['mrr']['population_std']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"冻结 checkpoint：`{frozen['embedding']['checkpoint']}`；SHA-256 `{frozen['embedding']['model_sha256']}`。",
            "",
            "## 2. 一级检索与候选深度",
            "",
            f"一级网格共 {first_stage['grid_size']} 个参数组合，去重后 {first_stage['unique_ranking_count']} 个排名，保留 {first_stage['retained_size']} 个 coverage 前沿/各深度强候选进入联合搜索。",
            "",
            f"最终一级参数为 N={selected_route['first_candidate_n']}、vector_weight={selected_route['first_vector_weight']}、RRF-k={selected_route['first_rrf_k']}；Reranker 实际候选深度为 Top-{selected_route['candidate_k']}。这两个值由冻结 Embedding 上的新搜索产生，不沿用旧链 Top-50。",
            "",
            "## 3. Reranker 选择",
            "",
            "实现烟测在任何 Reranker 候选完成前修正了混合精度：训练使用 FP32 主参数 + FP16 autocast/GradScaler，并在保持 effective group batch=16 的同时降低 micro-batch；候选评测明确以 FP16 加载。失败烟测不进入任何选择指标，修订记录保存在协议 JSON。",
            "",
            "mined-negative 提前烟测在任何该类候选完成前发现 fallback 会在负例已满后继续追加；修正后对 24,000 组完整重建，source_mined7 与 mixed_typed3_mined4 均严格保持每组 1 正 + 7 负。失败启动没有模型或指标，不进入选择。",
            "",
            "Qwen 推理批量 128/256 虽复现相同 headline 指标，却分别改变 53/57 条开发请求的 Top-10 顺序，因此所有正式 Qwen 候选与最终回放继续固定 pair batch=32；两次提速烟测均排除。",
            "",
            "### 3.1 每种架构的学习率",
            "",
            "| 架构 | 开发集选中的 LR group | Mean AllHit@10 | Mean MRR |",
            "|---|---|---:|---:|",
        ]
    )
    for key, payload in lr_screens.items():
        selected = payload["selected"]
        lines.append(
            f"| {key} | `{selected['group']}` | "
            f"{pct(selected['aggregate']['allhit_at_10']['mean'])} | "
            f"{selected['aggregate']['mrr']['mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "### 3.2 架构三随机种子比较",
            "",
            "| 候选 | seeds | Mean AllHit@6 | Mean AllHit@10 | Mean MRR | Top-K | 参数量 | 估算 p50 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in architecture["selected_by_group"]:
        lines.append(
            f"| {row['group']} | {','.join(map(str, row['seeds']))} | "
            f"{pct(row['aggregate']['allhit_at_6']['mean'])} | "
            f"{pct(row['aggregate']['allhit_at_10']['mean'])} | "
            f"{row['aggregate']['mrr']['mean']:.6f} | {row['route']['candidate_k']} | "
            f"{row['parameter_count']:,} | "
            f"{row['estimated_rerank_p50_ms']:.4f}ms |"
        )
    lines.extend(
        [
            "",
            "### 3.3 负例 recipe 与交互检查",
            "",
            "| 负例候选 | seeds | Mean AllHit@10 | Worst AllHit@10 | Mean MRR |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in negative["selected_by_group"]:
        lines.append(
            f"| {row['group']} | {','.join(map(str, row['seeds']))} | "
            f"{pct(row['aggregate']['allhit_at_10']['mean'])} | "
            f"{pct(row['aggregate']['allhit_at_10']['min'])} | "
            f"{row['aggregate']['mrr']['mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"负例×架构交互补跑：required=`{str(interaction_plan['required']).lower()}`。最终冻结的 selection group 为 `{reranker['selection_group']}`。",
        ]
    )
    if interaction is not None:
        lines.extend(
            [
                "",
                "| 交互候选 | seeds | Mean AllHit@6 | Mean AllHit@10 | Mean MRR | Top-K |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in interaction["selected_by_group"]:
            lines.append(
                f"| {row['group']} | {','.join(map(str, row['seeds']))} | "
                f"{pct(row['aggregate']['allhit_at_6']['mean'])} | "
                f"{pct(row['aggregate']['allhit_at_10']['mean'])} | "
                f"{row['aggregate']['mrr']['mean']:.6f} | "
                f"{row['route']['candidate_k']} |"
            )
    lines.extend(
        [
            "",
            "## 4. 一次性 round-final",
            "",
            "| 路线 | AllHit@6 | AllHit@10 | AllHit@20 | AllHit@30 | AllHit@50 | MRR |",
            "|---|---:|---:|---:|---:|---:|---:|",
            metric_row("原 bs64 canonical", round_final["baseline"]["overall"]),
            metric_row("冻结 challenger", round_final["challenger"]["overall"]),
            "",
            f"AllHit@10 配对不一致样本：原链独有 {primary['mcnemar']['baseline_only']}，新链独有 {primary['mcnemar']['challenger_only']}；exact McNemar p={primary['mcnemar']['exact_two_sided_p']:.6g}。",
            "",
            "promotion gate 要求 point estimate 不低于原链，且回退 CI 下界不低于 -0.2pp。round-final 之后未重新调参。",
            "",
            "## 5. 冻结回归集",
            "",
            "| 数据集 | n | 原链 A@10 | 新链 A@10 | ΔA@10 | 95% CI | gate |",
            "|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for label, payload in (("Legacy frozen", legacy), ("API frozen", api)):
        item = payload["comparison"]["overall"]["metrics"]["allhit_at_10"]
        lines.append(
            f"| {label} | {payload['manifest']['rows']} | {pct(item['baseline'])} | "
            f"{pct(item['challenger'])} | {pp(item['point_difference'])} | "
            f"[{pp(item['ci95_low'])}, {pp(item['ci95_high'])}] | "
            f"{payload['comparison']['promotion_gate']['passed']} |"
        )
    challenger_latency = benchmark_challenger["latency_ms_per_query"]["complete_path"]
    baseline_latency = benchmark_baseline["latency_ms_per_query"]["complete_path"]
    challenger_deployment_a10 = benchmark_challenger["quality_metrics"][
        "deployment_candidate_only_scoring"
    ]["@10"]["AllHit@K"]
    challenger_reference_a10 = benchmark_challenger["quality_metrics"][
        "provided_reference_rankings"
    ]["@10"]["AllHit@K"]
    baseline_deployment_a10 = benchmark_baseline["quality_metrics"][
        "deployment_candidate_only_scoring"
    ]["@10"]["AllHit@K"]
    baseline_reference_a10 = benchmark_baseline["quality_metrics"][
        "provided_reference_rankings"
    ]["@10"]["AllHit@K"]
    lines.extend(
        [
            "",
            "## 6. 端到端效率",
            "",
            "| 路线 | queries/s | 完整路径 p50 | 完整路径 p95 | GPU 峰值 allocated | 与离线排名一致率 |",
            "|---|---:|---:|---:|---:|---:|",
            f"| 原链 | {benchmark_baseline['throughput_queries_per_second']:.2f} | {baseline_latency['p50']:.4f}ms | {baseline_latency['p95']:.4f}ms | {benchmark_baseline['hardware']['gpu']['peak_allocated_gib']:.3f}GiB | {pct(benchmark_baseline['ranking_equivalence']['top10_equal_rate'])} |",
            f"| challenger | {benchmark_challenger['throughput_queries_per_second']:.2f} | {challenger_latency['p50']:.4f}ms | {challenger_latency['p95']:.4f}ms | {benchmark_challenger['hardware']['gpu']['peak_allocated_gib']:.3f}GiB | {pct(benchmark_challenger['ranking_equivalence']['top10_equal_rate'])} |",
            "",
            f"独立端到端 candidate-only 回放的 round-final AllHit@10：原链 {pct(baseline_deployment_a10)}（冻结评测参考 {pct(baseline_reference_a10)}），challenger {pct(challenger_deployment_a10)}（冻结评测参考 {pct(challenger_reference_a10)}）。该项诊断实现与批次效应，不反向调参。",
            "",
            "## 7. 逻辑闭环与边界",
            "",
            "1. Reranker 训练与 Embedding 训练解耦；但候选分布、candidate_k 与两级 RRF 不解耦，所以 Embedding 冻结后重新做了完整下游搜索。",
            "2. 开发集联合搜索以全 100 工具统一打分后截取各条路线的 Top-10/20/30/40/50，因此候选深度比较没有被旧 Top-50 路线绑死；整链冻结后，最终集与回归集改按实际 candidate_k 重新打分，避免 FP16 批次组成造成部署偏差。",
            "3. round-final 只在 `frozen_pipeline.json` 写入模型 SHA-256、路线和部署 seed 后打开；每个最终/回归评测都有 exposure manifest 和输入哈希。",
            "4. 静态检索结果不把 Planner 回放混入 Embedding/Reranker 选型；检索冻结与最终门禁完成后，另以固定 Planner 独立回放，结果写入报告 11。",
            "5. 这些结果不能外推到动态 catalog、真实线上 query、工具参数生成、工具执行异常或重试。",
            "",
            "## 可追溯产物",
            "",
            "- 协议快照：`PROJECT/data/rag/_aux/retrieval_pipeline_reselection/protocol.snapshot.json`",
            "- 数据/泄漏审计：`PROJECT/data/rag/_aux/retrieval_pipeline_reselection/integrity_audit.json`",
            "- Embedding 冻结：`PROJECT/data/rag/_aux/retrieval_pipeline_reselection/embedding_selection.json`",
            "- 整链冻结：`PROJECT/data/rag/_aux/retrieval_pipeline_reselection/frozen_pipeline.json`",
            "- round-final：`PROJECT/data/rag/_aux/retrieval_pipeline_reselection/evaluation/round_final/`",
            "- 结构化总表：`PROJECT/data/rag/_aux/retrieval_pipeline_reselection/final_summary.json`",
            "",
        ]
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        canonical_report(
            frozen=frozen,
            embedding=embedding,
            first_stage=first_stage,
            architecture=architecture,
            negative=negative,
            round_final=round_final,
            legacy=legacy,
            api=api,
            benchmark=benchmark_challenger,
        ),
        encoding="utf-8",
    )
    print(f"retrieval re-selection report -> {args.report}")
    print(f"retrieval re-selection summary -> {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
