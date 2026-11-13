#!/usr/bin/env python3
"""Record and summarize fixed-Planner replay for the frozen retrieval challenger."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from analyze_embedding_planner_replay import (
    candidate_change_summary,
    compact_metrics,
    paired_boolean,
    p_value,
    pct,
    pp,
    retrieval_trace_transition,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / "PROJECT"
RUN_ROOT = PROJECT / "data/rag/_aux/retrieval_pipeline_reselection"
REPLAY_ROOT = (
    PROJECT
    / "data/tool_call_trace/_aux/assignment3"
    / "retrieval_reselection_planner_replay_20260731"
)
REPLAY_EVAL = REPLAY_ROOT / "eval"
BASE_EVAL = (
    PROJECT
    / "data/tool_call_trace/_aux/assignment3"
    / "assignment3_crossstyle_rag10/eval"
)
BASE_INPUTS = {
    "legacy6000": (
        PROJECT
        / "data/tool_call_trace/_aux/assignment3"
        / "frozen_rag_candidates_k10_label_free.jsonl"
    ),
    "api1000": (
        PROJECT
        / "data/tool_call_trace/_aux/assignment3"
        / "assignment3_100tool_api/test_api/rag_k10_label_free.jsonl"
    ),
}
NEW_INPUTS = {
    "legacy6000": REPLAY_ROOT / "legacy_rag_k10_label_free.jsonl",
    "api1000": REPLAY_ROOT / "api1000_rag_k10_label_free.jsonl",
}
MANIFEST = REPLAY_ROOT / "replay_manifest.json"
SUMMARY = REPLAY_ROOT / "planner_replay_summary.json"
REPORT = REPLAY_ROOT / "canonical_planner_report.md"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def metric_paths(model: int, split: str) -> dict[str, Path]:
    base = BASE_EVAL / f"a3-crossstyle-{model}b-{split}"
    new = REPLAY_EVAL / f"reselection-crossstyle-{model}b-{split}"
    paths = {
        "baseline_metrics": Path(f"{base}.metrics.json"),
        "baseline_scored": Path(f"{base}.scored.jsonl"),
        "challenger_metrics": Path(f"{new}.metrics.json"),
        "challenger_scored": Path(f"{new}.scored.jsonl"),
    }
    if split == "legacy6000":
        paths.update(
            {
                "baseline_system_metrics": Path(f"{base}.system.metrics.json"),
                "baseline_system": Path(f"{base}.system.jsonl"),
                "challenger_system_metrics": Path(f"{new}.system.metrics.json"),
                "challenger_system": Path(f"{new}.system.jsonl"),
            }
        )
    return paths


def begin() -> None:
    frozen = RUN_ROOT / "frozen_pipeline.json"
    final_summary = RUN_ROOT / "final_summary.json"
    required = [
        frozen,
        final_summary,
        RUN_ROOT / "evaluation/legacy_frozen_6k/comparison.json",
        RUN_ROOT / "evaluation/api_frozen_1k/comparison.json",
        *BASE_INPUTS.values(),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"planner replay cannot begin; missing={missing}")
    frozen_payload = load_json(frozen)
    if frozen_payload.get("status") != "frozen_before_round_final":
        raise SystemExit("retrieval pipeline is not frozen")
    hashes = {str(path): sha256_file(path) for path in required}
    if MANIFEST.exists():
        existing = load_json(MANIFEST)
        if existing.get("frozen_input_hashes") != hashes:
            raise SystemExit("existing Planner replay manifest has different frozen inputs")
        print(f"resume authorized Planner replay: {MANIFEST}")
        return
    write_json(
        MANIFEST,
        {
            "schema_version": 1,
            "status": "authorized_after_retrieval_freeze",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "post_replay_tuning_forbidden": True,
            "frozen_input_hashes": hashes,
        },
    )
    print(f"authorized Planner replay: {MANIFEST}")


def build_summary() -> dict[str, Any]:
    manifest = load_json(MANIFEST)
    if manifest.get("status") not in {
        "authorized_after_retrieval_freeze",
        "complete_no_post_replay_tuning",
    }:
        raise SystemExit("Planner replay lacks a valid pre-exposure manifest")
    artifacts = [*BASE_INPUTS.values(), *NEW_INPUTS.values()]
    results: dict[str, Any] = {}
    for model in (4, 8):
        model_rows: dict[str, Any] = {}
        for split in ("legacy6000", "api1000"):
            paths = metric_paths(model, split)
            missing = [str(path) for path in paths.values() if not path.exists()]
            if missing:
                raise SystemExit(f"incomplete Planner replay: {missing}")
            artifacts.extend(paths.values())
            base_metrics = load_json(paths["baseline_metrics"])
            new_metrics = load_json(paths["challenger_metrics"])
            base_scored = load_jsonl(paths["baseline_scored"])
            new_scored = load_jsonl(paths["challenger_scored"])
            row: dict[str, Any] = {
                "baseline": compact_metrics(base_metrics),
                "challenger": compact_metrics(new_metrics),
                "paired_set": paired_boolean(base_scored, new_scored, "set_em"),
                "paired_trace": paired_boolean(base_scored, new_scored, "trace_em"),
                "paired_strict": paired_boolean(
                    base_scored, new_scored, "strict_parse_ok"
                ),
                "retrieval_trace_transitions": retrieval_trace_transition(
                    base_scored, new_scored
                ),
                "paths": {name: str(path) for name, path in paths.items()},
            }
            if split == "legacy6000":
                base_system_metrics = load_json(paths["baseline_system_metrics"])
                new_system_metrics = load_json(paths["challenger_system_metrics"])
                base_system = load_jsonl(paths["baseline_system"])
                new_system = load_jsonl(paths["challenger_system"])
                row["system"] = {
                    "baseline_full_set": base_system_metrics["full_pipeline_success"],
                    "challenger_full_set": new_system_metrics["full_pipeline_success"],
                    "baseline_full_trace": base_system_metrics[
                        "full_pipeline_trace_success"
                    ],
                    "challenger_full_trace": new_system_metrics[
                        "full_pipeline_trace_success"
                    ],
                    "paired_full_set": paired_boolean(
                        base_system, new_system, "full_pipeline_success"
                    ),
                    "paired_full_trace": paired_boolean(
                        base_system, new_system, "full_pipeline_trace_success"
                    ),
                }
                row["tool_count"] = {
                    bucket: {
                        "count": values["count"],
                        "baseline_trace_em": values["trace_em"],
                        "challenger_trace_em": new_metrics["buckets"]["tool_count"][
                            bucket
                        ]["trace_em"],
                        "delta": new_metrics["buckets"]["tool_count"][bucket][
                            "trace_em"
                        ]
                        - values["trace_em"],
                    }
                    for bucket, values in base_metrics["buckets"]["tool_count"].items()
                }
            model_rows[split] = row
        results[str(model)] = model_rows

    deployment = results["8"]
    legacy = deployment["legacy6000"]
    api = deployment["api1000"]
    gate_rows: list[dict[str, Any]] = []

    def gate(name: str, observed: float, threshold: float) -> None:
        gate_rows.append(
            {
                "name": name,
                "observed": observed,
                "threshold": threshold,
                "passed": observed >= threshold,
            }
        )

    gate(
        "8B Legacy TraceEM does not decrease",
        legacy["challenger"]["trace_em"] - legacy["baseline"]["trace_em"],
        0.0,
    )
    gate(
        "8B 9k Full Trace does not decrease",
        legacy["system"]["challenger_full_trace"]
        - legacy["system"]["baseline_full_trace"],
        0.0,
    )
    gate(
        "8B API TraceEM delta >= -0.2pp",
        api["challenger"]["trace_em"] - api["baseline"]["trace_em"],
        -0.002,
    )
    for split in ("legacy6000", "api1000"):
        values = deployment[split]
        gate(
            f"8B {split} strict-output delta >= -0.2pp",
            values["challenger"]["strict_output_valid_rate"]
            - values["baseline"]["strict_output_valid_rate"],
            -0.002,
        )
    for bucket, values in legacy["tool_count"].items():
        if values["count"] >= 100:
            gate(
                f"8B Legacy tool_count={bucket} TraceEM delta >= -0.5pp",
                values["delta"],
                -0.005,
            )

    planner_passed = all(row["passed"] for row in gate_rows)
    retrieval_summary = load_json(RUN_ROOT / "final_summary.json")
    retrieval_passed = bool(retrieval_summary["promotion_gate_passed"])
    return {
        "schema_version": 1,
        "status": "complete_no_post_replay_tuning",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "retrieval_promotion_gate_passed": retrieval_passed,
        "planner_replay_gate": {"passed": planner_passed, "rows": gate_rows},
        "overall_promotion_passed": retrieval_passed and planner_passed,
        "decision": "canonical_explicit_negative_planner_route",
        "candidate_changes": {
            split: candidate_change_summary(BASE_INPUTS[split], NEW_INPUTS[split])
            for split in ("legacy6000", "api1000")
        },
        "results": results,
        "input_hashes": {str(path): sha256_file(path) for path in sorted(set(artifacts))},
    }


def make_report(summary: dict[str, Any]) -> str:
    lines = [
        "# 冻结检索新链：固定 Planner 回放",
        "",
        "> 固定 4B/8B 跨风格 SFT checkpoint-2429、RAG Top-10、original 顺序和 greedy 解码；只替换冻结检索排序，不重训 Planner，也不允许回放结果反向调检索参数。",
        "",
        "## 结论",
        "",
        f"- 检索 promotion gate：`{str(summary['retrieval_promotion_gate_passed']).lower()}`。",
        f"- Planner replay gate：`{str(summary['planner_replay_gate']['passed']).lower()}`。",
        f"- 联合决策：`{summary['decision']}`。",
        "",
        "## 固定 Planner 结果",
        "",
        "| 模型 | Split | Retrieval A10 原→新 | SetEM 原→新 | TraceEM 原→新 | Strict 原→新 | Trace 改善/回退 | McNemar p |",
        "|---|---|---|---|---|---|---:|---:|",
    ]
    for model in ("4", "8"):
        for split in ("api1000", "legacy6000"):
            row = summary["results"][model][split]
            old, new = row["baseline"], row["challenger"]
            paired = row["paired_trace"]
            lines.append(
                f"| {model}B | {split} | {pct(old['retrieval_allhit_at_10'], 2)}→{pct(new['retrieval_allhit_at_10'], 2)} | "
                f"{pct(old['set_em'])}→{pct(new['set_em'])} | {pct(old['trace_em'])}→{pct(new['trace_em'])} | "
                f"{pct(old['strict_output_valid_rate'])}→{pct(new['strict_output_valid_rate'])} | "
                f"{paired['improved']}/{paired['regressed']} | {p_value(paired['exact_mcnemar_p'])} |"
            )
    lines.extend(
        [
            "",
            "## 9k 静态系统结果",
            "",
            "| 模型 | Full Set 原→新 | Full Trace 原→新 | Trace 改善/回退 |",
            "|---|---|---|---:|",
        ]
    )
    for model in ("4", "8"):
        system = summary["results"][model]["legacy6000"]["system"]
        paired = system["paired_full_trace"]
        lines.append(
            f"| {model}B | {pct(system['baseline_full_set'])}→{pct(system['challenger_full_set'])} | "
            f"{pct(system['baseline_full_trace'])}→{pct(system['challenger_full_trace'])} | "
            f"{paired['improved']}/{paired['regressed']} |"
        )
    lines.extend(
        [
            "",
            "## 8B Legacy 工具数分层",
            "",
            "| 工具数 | n | TraceEM 原→新 | Δ | gate |",
            "|---:|---:|---|---:|---|",
        ]
    )
    for bucket, row in sorted(
        summary["results"]["8"]["legacy6000"]["tool_count"].items(),
        key=lambda item: int(item[0]),
    ):
        passed = row["count"] < 100 or row["delta"] >= -0.005
        lines.append(
            f"| {bucket} | {row['count']} | {pct(row['baseline_trace_em'])}→{pct(row['challenger_trace_em'])} | "
            f"{pp(row['delta'])} | {'PASS' if passed else '**FAIL**'} |"
        )
    failed = [row for row in summary["planner_replay_gate"]["rows"] if not row["passed"]]
    lines.extend(["", "## Gate 明细", ""])
    if failed:
        lines.extend(
            f"- FAIL：{row['name']}，实测 {pp(row['observed'])}，阈值 {pp(row['threshold'])}。"
            for row in failed
        )
    else:
        lines.append("- 所有预注册 Planner guardrail 均通过。")
    lines.extend(
        [
            "",
            "结果只代表静态工具候选与固定 Planner 的离线回放，不代表工具参数、真实执行、异常恢复或线上泛化。",
            "",
            "## 可追溯产物",
            "",
            "- `PROJECT/data/tool_call_trace/_aux/assignment3/retrieval_reselection_planner_replay_20260731/planner_replay_summary.json`",
            "- `PROJECT/finetune/outputs/preds/tool_trace/retrieval_reselection_planner_replay_20260731/`",
            "",
        ]
    )
    return "\n".join(lines)


def make_canonical_report(summary: dict[str, Any]) -> str:
    """Render absolute Planner results for the canonical retrieval route."""
    lines = [
        "# Canonical 检索链：固定 Planner 回放",
        "",
        "> 固定 4B/8B 跨风格 SFT checkpoint-2429、RAG Top-10、检索顺序和 greedy 解码，只接入 canonical 检索候选，不重训 Planner。",
        "",
        "## Canonical 端到端结果",
        "",
        "| Planner | Split | Retrieval AllHit@10 | SetEM | TraceEM | Strict |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for model in ("4", "8"):
        for split in ("api1000", "legacy6000"):
            row = summary["results"][model][split]["challenger"]
            lines.append(
                f"| {model}B | {split} | {pct(row['retrieval_allhit_at_10'])} | "
                f"{pct(row['set_em'])} | {pct(row['trace_em'])} | "
                f"{pct(row['strict_output_valid_rate'])} |"
            )
    lines.extend(
        [
            "",
            "## 9k 静态系统结果",
            "",
            "| Planner | Full Set | Full Trace |",
            "|---|---:|---:|",
        ]
    )
    for model in ("4", "8"):
        system = summary["results"][model]["legacy6000"]["system"]
        lines.append(
            f"| {model}B | {pct(system['challenger_full_set'])} | "
            f"{pct(system['challenger_full_trace'])} |"
        )
    lines.extend(
        [
            "",
            "## 8B Legacy 工具数分层",
            "",
            "| 工具数 | n | TraceEM |",
            "|---:|---:|---:|",
        ]
    )
    for bucket, row in sorted(
        summary["results"]["8"]["legacy6000"]["tool_count"].items(),
        key=lambda item: int(item[0]),
    ):
        lines.append(
            f"| {bucket} | {row['count']} | {pct(row['challenger_trace_em'])} |"
        )
    lines.extend(
        [
            "",
            "该路线是项目唯一 canonical，不以回放 guardrail 决定是否接入。结果只代表静态工具候选与固定 Planner 的离线回放。",
            "",
            "- Canonical 配置：`PROJECT/configs/retrieval_canonical_route.json`",
            "- 回放指标：`PROJECT/data/tool_call_trace/_aux/assignment3/retrieval_reselection_planner_replay_20260731/eval/`",
            "- 回放预测：`PROJECT/finetune/outputs/preds/tool_trace/retrieval_reselection_planner_replay_20260731/`",
            "",
        ]
    )
    return "\n".join(lines)


def complete() -> None:
    summary = build_summary()
    write_json(SUMMARY, summary)
    REPORT.write_text(make_canonical_report(summary), encoding="utf-8")
    manifest = load_json(MANIFEST)
    manifest.update(
        {
            "status": "complete_no_post_replay_tuning",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "summary": str(SUMMARY),
            "summary_sha256": sha256_file(SUMMARY),
        }
    )
    write_json(MANIFEST, manifest)
    print(f"Planner replay gate={summary['planner_replay_gate']['passed']}")
    print(f"decision={summary['decision']}")
    print(f"summary -> {SUMMARY}")
    print(f"report -> {REPORT}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("begin", "complete"), required=True)
    args = parser.parse_args()
    if args.stage == "begin":
        begin()
    else:
        complete()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
