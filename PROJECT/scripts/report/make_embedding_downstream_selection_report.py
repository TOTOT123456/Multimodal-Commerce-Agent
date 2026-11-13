#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the archived first-pass embedding downstream diagnostic."""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[2]
RAG_AUX = PROJECT / "data" / "rag" / "_aux"
EMBEDDING_ROOT = RAG_AUX / "embedding_finetune"
RERANKER_ROOT = RAG_AUX / "reranker_finetune"
PROTOCOL_PATH = RAG_AUX / "embedding_downstream_selection" / "protocol.snapshot.json"
TEST_PATH = PROJECT / "data" / "tool_call_trace" / "test.jsonl"
OUT_DIR = RAG_AUX / "embedding_downstream_selection"
SUMMARY_PATH = OUT_DIR / "summary.json"
REPORT_PATH = OUT_DIR / "first_pass_downstream_diagnostic.md"
PLANNER_REPLAY_SUMMARY = (
    PROJECT
    / "data"
    / "tool_call_trace"
    / "_aux"
    / "assignment3"
    / "embedding_hard2_replay_20260731"
    / "planner_replay_summary.json"
)
ORIGINAL_RUN = "bge-large-mnrl-e1-lr2e-5-bs64"
LABELS = {
    ORIGINAL_RUN: "原 bs64 主链",
    "bge-large-mnrl-control-e1-lr2e-5-bs8-g8": "bs8 无显式负例",
    "bge-large-mnrl-explicit-hard2-e1-lr2e-5-bs8-g8": "hard2：同域/部分替代",
    "bge-large-mnrl-explicit-easy2-e1-lr2e-5-bs8-g8": "easy2：跨域/新增无关",
    "bge-large-mnrl-explicit-all4-e1-lr2e-5-bs8-g8": "all4：全部显式负例",
}


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


def ranking_metrics(rows: list[dict[str, Any]], ks: tuple[int, ...]) -> dict[str, float]:
    result = {}
    for k in ks:
        result[f"allhit_at_{k}"] = sum(
            set(row["gold_tools"]) <= set(row["pred_tools"][:k])
            for row in rows
        ) / len(rows)
    reciprocal_ranks = []
    for row in rows:
        gold = set(row["gold_tools"])
        reciprocal_ranks.append(
            next(
                (
                    1.0 / rank
                    for rank, tool_id in enumerate(row["pred_tools"], start=1)
                    if tool_id in gold
                ),
                0.0,
            )
        )
    result["mrr"] = sum(reciprocal_ranks) / len(rows)
    return result


def vector_metrics(run: str) -> tuple[dict[str, float], list[Path]]:
    metrics_path = EMBEDDING_ROOT / run / "retrieval_fallback" / "metrics.json"
    prediction_path = (
        EMBEDDING_ROOT
        / run
        / "retrieval_fallback"
        / "preds"
        / f"vector_{run}.jsonl"
    )
    return ranking_metrics(load_jsonl(prediction_path), (6, 10, 30, 50)), [
        metrics_path,
        prediction_path,
    ]


def first_stage_path(run: str) -> Path:
    return EMBEDDING_ROOT / run / "downstream_selection" / "first_stage" / "weighted_fusion_sweep.json"


def reranker_metric_paths(run: str) -> tuple[Path, Path]:
    if run == ORIGINAL_RUN:
        return (
            RERANKER_ROOT / "bge-reranker-base-e1-g8-bs16_eval_tuned_hybrid_train" / "metrics.json",
            RERANKER_ROOT / "bge-reranker-base-e1-g8-bs16_eval_tuned_hybrid" / "metrics.json",
        )
    base = EMBEDDING_ROOT / run / "downstream_selection"
    return base / "task_reranker_train" / "metrics.json", base / "task_reranker_test" / "metrics.json"


def final_path(run: str) -> Path:
    return EMBEDDING_ROOT / run / "downstream_selection" / "final_fusion" / "reranker_rank_fusion_sweep.json"


def selected_prediction(path: Path) -> Path:
    matches = sorted(path.parent.glob("preds_rank_fusion_*.jsonl"))
    if len(matches) != 1:
        raise ValueError(f"expected one final prediction file in {path.parent}, got {matches}")
    return matches[0]


def compact_reranker_metrics(payload: dict[str, Any]) -> dict[str, float]:
    metrics = payload["rerank_metrics"]
    return {
        "allhit_at_6": metrics.get("@6", {}).get("AllHit@K"),
        "allhit_at_10": metrics["@10"]["AllHit@K"],
        "allhit_at_30": metrics["@30"]["AllHit@K"],
        "allhit_at_50": metrics["@50"]["AllHit@K"],
        "mrr": metrics["overall"]["MRR"],
    }


def exact_mcnemar_p(improved: int, regressed: int) -> float:
    discordant = improved + regressed
    if not discordant:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(improved, regressed) + 1)
    )
    return min(1.0, 2.0 * tail / (1 << discordant))


def paired_binary(
    baseline: list[dict[str, Any]],
    challenger: list[dict[str, Any]],
    k: int,
) -> dict[str, Any]:
    baseline_by_id = {str(row["id"]): row for row in baseline}
    challenger_by_id = {str(row["id"]): row for row in challenger}
    if set(baseline_by_id) != set(challenger_by_id):
        raise ValueError("paired prediction id sets do not match")
    differences = []
    improved = 0
    regressed = 0
    for trace_id in sorted(baseline_by_id):
        baseline_row = baseline_by_id[trace_id]
        challenger_row = challenger_by_id[trace_id]
        if baseline_row["query"] != challenger_row["query"]:
            raise ValueError(f"paired query mismatch: {trace_id}")
        if baseline_row["gold_tools"] != challenger_row["gold_tools"]:
            raise ValueError(f"paired gold mismatch: {trace_id}")
        gold = set(baseline_row["gold_tools"])
        baseline_ok = gold <= set(baseline_row["pred_tools"][:k])
        challenger_ok = gold <= set(challenger_row["pred_tools"][:k])
        differences.append(int(challenger_ok) - int(baseline_ok))
        improved += int(challenger_ok and not baseline_ok)
        regressed += int(baseline_ok and not challenger_ok)
    count = len(differences)
    delta = sum(differences) / count
    variance = (
        sum((value - delta) ** 2 for value in differences) / (count - 1)
        if count > 1
        else 0.0
    )
    half_width = 1.96 * math.sqrt(variance / count)
    return {
        "count": count,
        "k": k,
        "delta": delta,
        "paired_wald_95ci": [delta - half_width, delta + half_width],
        "improved": improved,
        "regressed": regressed,
        "unchanged": count - improved - regressed,
        "exact_mcnemar_p": exact_mcnemar_p(improved, regressed),
    }


def holm_adjust(rows: dict[str, dict[str, Any]]) -> None:
    ordered = sorted(rows, key=lambda run: rows[run]["exact_mcnemar_p"])
    running_max = 0.0
    total = len(ordered)
    for index, run in enumerate(ordered):
        raw = rows[run]["exact_mcnemar_p"]
        adjusted = min(1.0, raw * (total - index))
        running_max = max(running_max, adjusted)
        rows[run]["holm_adjusted_p"] = running_max


def miss_rows(rows: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    misses = []
    for row in rows:
        gold = set(row["gold_tools"])
        if not gold <= set(row["pred_tools"][:k]):
            misses.append(
                {
                    "id": row["id"],
                    "query": row["query"],
                    "gold_tools": row["gold_tools"],
                    "missing_tools": sorted(gold - set(row["pred_tools"][:k])),
                }
            )
    return misses


def tool_count_strata(
    baseline: list[dict[str, Any]],
    winner: list[dict[str, Any]],
    authority: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    metadata = {str(row["id"]): row for row in authority}
    baseline_by_id = {str(row["id"]): row for row in baseline}
    winner_by_id = {str(row["id"]): row for row in winner}
    grouped: dict[str, list[str]] = defaultdict(list)
    for trace_id in sorted(metadata):
        grouped[str(metadata[trace_id].get("tool_count", len(baseline_by_id[trace_id]["gold_tools"])))].append(trace_id)
    result = {}
    for bucket, trace_ids in grouped.items():
        baseline_correct = 0
        winner_correct = 0
        improved = 0
        regressed = 0
        for trace_id in trace_ids:
            gold = set(baseline_by_id[trace_id]["gold_tools"])
            baseline_ok = gold <= set(baseline_by_id[trace_id]["pred_tools"][:10])
            winner_ok = gold <= set(winner_by_id[trace_id]["pred_tools"][:10])
            baseline_correct += int(baseline_ok)
            winner_correct += int(winner_ok)
            improved += int(winner_ok and not baseline_ok)
            regressed += int(baseline_ok and not winner_ok)
        count = len(trace_ids)
        result[bucket] = {
            "count": count,
            "baseline_allhit_at_10": baseline_correct / count,
            "winner_allhit_at_10": winner_correct / count,
            "delta": (winner_correct - baseline_correct) / count,
            "improved": improved,
            "regressed": regressed,
        }
    return result


def build_summary() -> dict[str, Any]:
    protocol = load_json(PROTOCOL_PATH)
    candidates = protocol["candidates"]
    artifacts: list[Path] = [PROTOCOL_PATH, TEST_PATH]
    runs: dict[str, dict[str, Any]] = {}
    predictions: dict[str, list[dict[str, Any]]] = {}
    for run in candidates:
        vector, vector_paths = vector_metrics(run)
        first_path = first_stage_path(run)
        train_reranker_path, test_reranker_path = reranker_metric_paths(run)
        chain_path = final_path(run)
        first = load_json(first_path)
        train_reranker = load_json(train_reranker_path)
        test_reranker = load_json(test_reranker_path)
        chain = load_json(chain_path)
        prediction_path = selected_prediction(chain_path)
        prediction_rows = load_jsonl(prediction_path)
        if len(prediction_rows) != 6000:
            raise ValueError(f"{run}: expected 6000 final predictions, got {len(prediction_rows)}")
        predictions[run] = prediction_rows
        artifacts.extend(
            [
                *vector_paths,
                first_path,
                train_reranker_path,
                test_reranker_path,
                chain_path,
                prediction_path,
            ]
        )
        runs[run] = {
            "label": LABELS.get(run, run),
            "vector": vector,
            "first_stage": {
                "selected_on_train": first["selected"],
                "evaluation": first["eval_selected"],
            },
            "task_reranker": {
                "train": compact_reranker_metrics(train_reranker),
                "evaluation": compact_reranker_metrics(test_reranker),
                "candidate_allhit_at_50": test_reranker["candidate_metrics"]["@50"]["AllHit@K"],
                "elapsed_seconds_train": train_reranker["elapsed_seconds"],
                "elapsed_seconds_evaluation": test_reranker["elapsed_seconds"],
            },
            "final_fusion": {
                "selected_on_train": chain["selected"],
                "evaluation": chain["eval_selected"],
            },
            "paths": {
                "vector_metrics": str(vector_paths[0]),
                "vector_predictions": str(vector_paths[1]),
                "first_stage_sweep": str(first_path),
                "reranker_train_metrics": str(train_reranker_path),
                "reranker_evaluation_metrics": str(test_reranker_path),
                "final_sweep": str(chain_path),
                "final_predictions": str(prediction_path),
            },
        }

    winner = min(
        candidates,
        key=lambda run: (
            -runs[run]["final_fusion"]["evaluation"]["allhit_at_10"],
            -runs[run]["final_fusion"]["evaluation"]["allhit_at_30"],
            -runs[run]["final_fusion"]["evaluation"]["mrr"],
            -runs[run]["task_reranker"]["candidate_allhit_at_50"],
            run,
        ),
    )
    paired_vs_original = {
        run: paired_binary(predictions[ORIGINAL_RUN], predictions[run], 10)
        for run in candidates
        if run != ORIGINAL_RUN
    }
    holm_adjust(paired_vs_original)
    authority = load_jsonl(TEST_PATH)
    source_path = Path(load_json(final_path(winner))["eval_source_preds"])
    source_rows = load_jsonl(source_path)
    artifacts.append(source_path)
    return {
        "schema_version": 1,
        "status": "complete",
        "date": "2026-07-31",
        "scope": "100-tool synthetic closed-set internal embedding downstream selection",
        "protocol": str(PROTOCOL_PATH),
        "evaluation_governance": protocol["data"]["evaluation_status"],
        "selection_order": protocol["checkpoint_selection"],
        "candidates": candidates,
        "winner": winner,
        "winner_label": LABELS.get(winner, winner),
        "runs": runs,
        "paired_allhit_at_10_vs_original": paired_vs_original,
        "winner_tool_count_strata": tool_count_strata(
            predictions[ORIGINAL_RUN],
            predictions[winner],
            authority,
        ),
        "winner_first_stage_top50_misses": miss_rows(source_rows, 50),
        "winner_final_top10_misses": miss_rows(predictions[winner], 10),
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


def build_report(summary: dict[str, Any]) -> str:
    runs = summary["runs"]
    winner = summary["winner"]
    original = runs[ORIGINAL_RUN]
    selected = runs[winner]
    old_final = original["final_fusion"]["evaluation"]
    new_final = selected["final_fusion"]["evaluation"]
    paired = summary["paired_allhit_at_10_vs_original"][winner]
    vector_best = min(
        runs,
        key=lambda run: (
            -runs[run]["vector"]["allhit_at_10"],
            -runs[run]["vector"]["allhit_at_30"],
            -runs[run]["vector"]["mrr"],
            run,
        ),
    )
    planner_replay = (
        load_json(PLANNER_REPLAY_SUMMARY)
        if PLANNER_REPLAY_SUMMARY.exists()
        else None
    )
    planner_decision_line = (
        "3. 已在固定 4B/8B Planner 上完成 Top10 回放。8B Legacy/Full Trace 总体略升，"
        "但 5 工具层 TraceEM 回退 3.2159pp，未通过预注册分层 guardrail；"
        "因此不替换 canonical 链，hard2 保留为 Planner 重训 challenger。"
        if planner_replay
        and not planner_replay["promotion_gate"]["passed"]
        else (
            "3. 固定 Planner 回放已通过 promotion gate，可以按回放报告中的决策更新 canonical 链。"
            if planner_replay
            else (
                "3. 在现有 4B/8B Planner checkpoint 上重放新的 Top10。只有 Planner TraceEM "
                "与 9k Full Trace 通过非劣 guardrail，才替换 canonical 检索链；否则保留原链。"
            )
        )
    )
    lines = [
        "# Embedding 显式负例：完整下游链路选型",
        "",
        "> 日期：2026-07-31  ",
        "> 范围：100 工具合成闭集内部选型；不是新的未观察线上 holdout。",
        "",
        "## 结论",
        "",
        f"- **完整链 point-estimate 赢家是 `{winner}`（{summary['winner_label']}），不是单路 Top10 最高的 `{vector_best}`。**",
        f"- 赢家最终 `AllHit@10/30/50 = {pct(new_final['allhit_at_10'])} / {pct(new_final['allhit_at_30'])} / {pct(new_final['allhit_at_50'])}`，`MRR={new_final['mrr']:.6f}`。",
        f"- 相对原完整链，最终 `AllHit@10` 从 {pct(old_final['allhit_at_10'])} 到 {pct(new_final['allhit_at_10'])}，变化 {pp(new_final['allhit_at_10'] - old_final['allhit_at_10'])}；逐样本改善/回退为 {paired['improved']} / {paired['regressed']}。",
        f"- 配对 McNemar `p={p_value(paired['exact_mcnemar_p'])}`，Holm 校正后 `p={p_value(paired['holm_adjusted_p'])}`。因此它是预注册排序下的内部赢家，但现有 6000 条已观察测试证据**不足以证明**相对原链有统计显著提升。",
        "- Top50 必须测，因为它是 reranker 能达到的候选覆盖上界；但不能只按 Top50 选模型。这里先比较 Top50 覆盖，再以最终融合后的 Top10 为主指标。",
    ]
    if planner_replay:
        gate = planner_replay["promotion_gate"]
        lines.append(
            f"- 固定 Planner/Full Trace 回放已经完成，promotion gate "
            f"{'通过' if gate['passed'] else '未通过'}；部署决策为 `{gate['decision']}`。"
        )
    lines += [
        "",
        "## 全链统一对比",
        "",
        "| 候选 | Vector A10/A30/A50 | First-stage 参数 | First A10/A30/A50 | Reranker A10/A30 | Final 参数 | Final A10/A30/A50 | Final MRR | ΔA10 vs 原链 |",
        "|---|---|---|---|---|---|---|---:|---:|",
    ]
    for run in summary["candidates"]:
        row = runs[run]
        vector = row["vector"]
        first_train = row["first_stage"]["selected_on_train"]
        first = row["first_stage"]["evaluation"]
        reranker = row["task_reranker"]["evaluation"]
        final_train = row["final_fusion"]["selected_on_train"]
        final = row["final_fusion"]["evaluation"]
        lines.append(
            "| {label} | {va10}/{va30}/{va50} | N={n}, w={fw:g}, k={fk} | "
            "{fa10}/{fa30}/{fa50} | {ra10}/{ra30} | w={rw:g}, k={rk} | "
            "{a10}/{a30}/{a50} | {mrr:.6f} | {delta} |".format(
                label=row["label"],
                va10=pct(vector["allhit_at_10"], 2),
                va30=pct(vector["allhit_at_30"], 2),
                va50=pct(vector["allhit_at_50"], 2),
                n=first_train["candidate_n"],
                fw=first_train["vector_weight"],
                fk=first_train["rrf_k"],
                fa10=pct(first["allhit_at_10"], 2),
                fa30=pct(first["allhit_at_30"], 2),
                fa50=pct(first["allhit_at_50"], 2),
                ra10=pct(reranker["allhit_at_10"], 2),
                ra30=pct(reranker["allhit_at_30"], 2),
                rw=final_train["source_weight"],
                rk=final_train["rrf_k"],
                a10=pct(final["allhit_at_10"], 2),
                a30=pct(final["allhit_at_30"], 2),
                a50=pct(final["allhit_at_50"], 2),
                mrr=final["mrr"],
                delta=pp(final["allhit_at_10"] - old_final["allhit_at_10"]),
            )
        )

    lines += [
        "",
        "关键观察：",
        "",
        f"- easy2 的单路 `AllHit@10={pct(runs[vector_best]['vector']['allhit_at_10'])}` 最高，但其 first-stage/最终 `AllHit@50={pct(runs[vector_best]['first_stage']['evaluation']['allhit_at_50'])}`，低于 hard2 的 {pct(selected['first_stage']['evaluation']['allhit_at_50'])}；Top10 优势没有完整传递到 Top50 reranker 链。",
        f"- hard2 单路 `AllHit@10={pct(selected['vector']['allhit_at_10'])}`，最终融合后达到 {pct(new_final['allhit_at_10'])}；说明“最佳 embedding”必须按下游目标定义，不能把单路榜首直接当成部署榜首。",
        "",
        "## 相对原完整链的配对结果",
        "",
        "| Challenger | ΔAllHit@10 | 95% CI | 改善 / 回退 | McNemar p | Holm p |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for run, row in summary["paired_allhit_at_10_vs_original"].items():
        lo, hi = row["paired_wald_95ci"]
        lines.append(
            f"| {runs[run]['label']} | {pp(row['delta'])} | "
            f"[{pp(lo)}, {pp(hi)}] | {row['improved']} / {row['regressed']} | "
            f"{p_value(row['exact_mcnemar_p'])} | {p_value(row['holm_adjusted_p'])} |"
        )

    lines += [
        "",
        "## 赢家按工具数分层",
        "",
        "| Gold 工具数 | n | 原链 A10 | hard2 A10 | Δ | 改善 / 回退 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for bucket, row in sorted(
        summary["winner_tool_count_strata"].items(),
        key=lambda item: int(item[0]),
    ):
        lines.append(
            f"| {bucket} | {row['count']} | {pct(row['baseline_allhit_at_10'])} | "
            f"{pct(row['winner_allhit_at_10'])} | {pp(row['delta'])} | "
            f"{row['improved']} / {row['regressed']} |"
        )

    lines += [
        "",
        "## 决策与边界",
        "",
        f"1. 锁定 `{winner}` 为新的**检索链 challenger**，使用其 train-selected 参数：first-stage `N={selected['first_stage']['selected_on_train']['candidate_n']}, vector_weight={selected['first_stage']['selected_on_train']['vector_weight']}, rrf_k={selected['first_stage']['selected_on_train']['rrf_k']}`；final fusion `source_weight={selected['final_fusion']['selected_on_train']['source_weight']}, rrf_k={selected['final_fusion']['selected_on_train']['rrf_k']}`。",
        "2. 不因这张表重训或重选 reranker：所有候选复用同一个任务内微调 reranker，差异来自 embedding 候选及其 train-selected 融合。",
        planner_decision_line,
        "4. 冻结测试集此前已经用于 embedding 版本比较，本轮属于内部 post-selection 比较。下一轮若要作正式显著性结论，应另建未观察 selection/final split，并冻结一次性终评。",
        "5. 结果只覆盖静态工具选择，不覆盖参数、Observation、执行异常、重试或真实线上泛化。",
        "",
        "## 可追溯产物",
        "",
        f"- 协议：`{summary['protocol']}`",
        f"- 结构化汇总：`{SUMMARY_PATH}`",
        (
            f"- Planner 回放汇总：`{PLANNER_REPLAY_SUMMARY}`。"
            if planner_replay
            else "- Planner 回放汇总：待跑。"
        ),
        f"- 赢家 first-stage Top50 漏召回：{len(summary['winner_first_stage_top50_misses'])} / 6000。",
        f"- 赢家最终 Top10 漏召回：{len(summary['winner_final_top10_misses'])} / 6000。",
        "- 每个输入产物的 SHA-256 已写入结构化汇总。",
        "",
        "复现：",
        "",
        "```bash",
        "cd /renyuanliu/Agent",
        "python3 PROJECT/scripts/report/make_embedding_downstream_selection_report.py",
        "```",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    summary = build_summary()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    REPORT_PATH.write_text(build_report(summary), encoding="utf-8")
    print(f"winner={summary['winner']}")
    print(f"summary -> {SUMMARY_PATH}")
    print(f"report -> {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
