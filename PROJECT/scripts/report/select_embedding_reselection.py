#!/usr/bin/env python3
"""Aggregate the frozen three-seed Embedding development selection."""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / "PROJECT"
AUX = PROJECT / "data" / "rag" / "_aux" / "retrieval_pipeline_reselection"
OUT_JSON = AUX / "embedding_selection.json"
OUT_MD = AUX / "embedding_selection.md"
SEEDS = (20260707, 20260731, 20260801)
RECIPES = ("bs64_none", "bs8_control", "easy2", "hard2", "all4")
NEAR_TOLERANCE = 0.001


SEED_20260707_MODELS = {
    "bs64_none": PROJECT
    / "data/rag/_aux/embedding_finetune/bge-large-mnrl-e1-lr2e-5-bs64/model/final",
    "bs8_control": PROJECT
    / "data/rag/_aux/embedding_finetune/bge-large-mnrl-control-e1-lr2e-5-bs8-g8/model/final",
    "easy2": PROJECT
    / "data/rag/_aux/embedding_finetune/bge-large-mnrl-explicit-easy2-e1-lr2e-5-bs8-g8/model/final",
    "hard2": PROJECT
    / "data/rag/_aux/embedding_finetune/bge-large-mnrl-explicit-hard2-e1-lr2e-5-bs8-g8/model/final",
    "all4": PROJECT
    / "data/rag/_aux/embedding_finetune/bge-large-mnrl-explicit-all4-e1-lr2e-5-bs8-g8/model/final",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_dir(recipe: str, seed: int) -> Path:
    if seed == 20260707:
        return SEED_20260707_MODELS[recipe]
    return (
        AUX
        / "embedding_train"
        / f"bge-large-reselection-{recipe}-s{seed}"
        / "model/final"
    )


def metric_path(recipe: str, seed: int) -> Path:
    return AUX / "embedding_dev" / f"seed_{seed}" / recipe / "metrics.json"


def vector_payload(metrics: dict) -> dict:
    keys = [key for key in metrics if key.startswith("vector:")]
    if len(keys) != 1:
        raise ValueError(f"expected one vector result, found {keys}")
    return metrics[keys[0]]


def seed_row(recipe: str, seed: int) -> dict:
    path = metric_path(recipe, seed)
    if not path.exists():
        raise FileNotFoundError(path)
    model = model_dir(recipe, seed)
    weights = model / "model.safetensors"
    if not weights.exists():
        raise FileNotFoundError(weights)
    payload = vector_payload(load_json(path))
    metrics = payload["metrics"]
    return {
        "seed": seed,
        "checkpoint": str(model.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(weights),
        "metrics_path": str(path.relative_to(ROOT)),
        "metrics_sha256": sha256_file(path),
        "allhit_at_6": metrics["@6"]["AllHit@K"],
        "allhit_at_10": metrics["@10"]["AllHit@K"],
        "allhit_at_20": metrics["@20"]["AllHit@K"],
        "allhit_at_30": metrics["@30"]["AllHit@K"],
        "allhit_at_50": metrics["@50"]["AllHit@K"],
        "mrr": metrics["overall"]["MRR"],
        "encode_avg_ms": payload["timing"]["encode_avg_ms"],
    }


def aggregate(recipe: str) -> dict:
    rows = [seed_row(recipe, seed) for seed in SEEDS]
    metric_names = (
        "allhit_at_6",
        "allhit_at_10",
        "allhit_at_20",
        "allhit_at_30",
        "allhit_at_50",
        "mrr",
        "encode_avg_ms",
    )
    summary = {}
    for name in metric_names:
        values = [float(row[name]) for row in rows]
        summary[name] = {
            "mean": statistics.fmean(values),
            "population_std": statistics.pstdev(values),
            "min": min(values),
            "max": max(values),
        }
    return {"recipe": recipe, "seeds": rows, "aggregate": summary}


def retain_near_best(rows: list[dict], accessor, higher: bool = True) -> list[dict]:
    values = [accessor(row) for row in rows]
    target = max(values) if higher else min(values)
    if higher:
        return [row for row in rows if accessor(row) >= target - NEAR_TOLERANCE]
    return [row for row in rows if accessor(row) <= target + NEAR_TOLERANCE]


def select_recipe(rows: list[dict]) -> tuple[dict, list[dict]]:
    trace = []
    active = retain_near_best(
        rows, lambda row: row["aggregate"]["allhit_at_50"]["mean"]
    )
    trace.append({"criterion": "mean_allhit_at_50", "remaining": [r["recipe"] for r in active]})
    criteria = (
        ("mean_allhit_at_20", lambda row: row["aggregate"]["allhit_at_20"]["mean"], True),
        ("mean_allhit_at_10", lambda row: row["aggregate"]["allhit_at_10"]["mean"], True),
        ("mean_mrr", lambda row: row["aggregate"]["mrr"]["mean"], True),
        ("worst_seed_allhit_at_50", lambda row: row["aggregate"]["allhit_at_50"]["min"], True),
    )
    for name, accessor, higher in criteria:
        if len(active) <= 1:
            break
        active = retain_near_best(active, accessor, higher=higher)
        trace.append({"criterion": name, "remaining": [r["recipe"] for r in active]})
    if len(active) > 1:
        # All candidates have exactly the same BGE-large architecture and
        # therefore the same inference FLOPs/parameter count.  Millisecond
        # differences collected while six independent jobs share the host are
        # measurement noise, not a deployment-cost distinction.
        trace.append(
            {
                "criterion": "same_architecture_latency_treated_as_tie",
                "remaining": [r["recipe"] for r in active],
            }
        )
        simplicity = {recipe: index for index, recipe in enumerate(RECIPES)}
        active.sort(key=lambda row: simplicity[row["recipe"]])
        trace.append({"criterion": "simpler_recipe", "remaining": [active[0]["recipe"]]})
    return active[0], trace


def select_checkpoint(recipe_row: dict) -> dict:
    return max(
        recipe_row["seeds"],
        key=lambda row: (
            row["allhit_at_50"],
            row["allhit_at_20"],
            row["allhit_at_10"],
            row["mrr"],
            -row["seed"],
        ),
    )


def render_markdown(payload: dict) -> str:
    lines = [
        "# 检索全链路重新选型",
        "",
        "> 本页由 `select_embedding_reselection.py` 生成。当前只完成 Embedding 冻结；一级融合、Reranker、二级融合和 round-final 仍不得提前写结论。",
        "",
        "## 1. Embedding 三随机种子开发集结果",
        "",
        "开发集为跨风格 1200 条；选择接口是下游最大候选深度 50。数值为三随机种子均值 ± 总体标准差。",
        "",
        "| Recipe | AllHit@10 | AllHit@20 | AllHit@50 | MRR | encode ms/query |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["recipes"]:
        agg = row["aggregate"]
        cell = lambda name: f"{agg[name]['mean']:.6f} ± {agg[name]['population_std']:.6f}"
        lines.append(
            f"| {row['recipe']} | {cell('allhit_at_10')} | {cell('allhit_at_20')} | "
            f"{cell('allhit_at_50')} | {cell('mrr')} | {cell('encode_avg_ms')} |"
        )
    chosen = payload["selected"]
    lines.extend(
        [
            "",
            "## 2. 已冻结结论",
            "",
            f"- 获胜 recipe：`{chosen['recipe']}`。",
            f"- 部署 checkpoint seed：`{chosen['checkpoint']['seed']}`。",
            f"- checkpoint：`{chosen['checkpoint']['checkpoint']}`。",
            f"- SHA256：`{chosen['checkpoint']['checkpoint_sha256']}`。",
            "- 该结论只冻结 Embedding；不沿用旧一级/二级 RRF 或旧 candidate_k。",
            "",
            "选择轨迹：",
            "",
        ]
    )
    for step in payload["selection_trace"]:
        lines.append(f"- `{step['criterion']}` → {', '.join(step['remaining'])}")
    lines.extend(
        [
            "",
            "## 3. 下一阶段",
            "",
            "只在上述冻结 Embedding 上生成一级 BM25 + vector 网格，保留 coverage Pareto 前沿；随后联合选择 candidate_k、Reranker 和二级 RRF。round-final 在整链冻结前不运行。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    recipes = [aggregate(recipe) for recipe in RECIPES]
    selected_recipe, trace = select_recipe(recipes)
    selected_checkpoint = select_checkpoint(selected_recipe)
    payload = {
        "protocol": "PROJECT/data/rag/_aux/retrieval_pipeline_reselection/protocol.snapshot.json",
        "development_split": "PROJECT/data/tool_call_trace/_aux/assignment3/assignment3_crossstyle_rag10/dpo_v2/eval_protocol/dev.jsonl",
        "seeds": list(SEEDS),
        "near_tolerance": NEAR_TOLERANCE,
        "recipes": recipes,
        "selection_trace": trace,
        "selected": {
            "recipe": selected_recipe["recipe"],
            "checkpoint": selected_checkpoint,
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUT_MD.write_text(render_markdown(payload), encoding="utf-8")
    print(
        f"selected recipe={selected_recipe['recipe']} "
        f"seed={selected_checkpoint['seed']} -> {OUT_JSON}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
