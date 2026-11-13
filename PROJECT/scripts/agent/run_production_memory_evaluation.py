#!/usr/bin/env python3
"""运行短会话生产近似协议，并按预注册双指标锁定与汇总候选。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from PROJECT.scripts.project_common import load_jsonl, write_json  # noqa: E402


ROOT = REPO_ROOT / "PROJECT/data/agent/_aux/natural_short_memory_evaluation"
EVALUATOR = REPO_ROOT / "PROJECT/scripts/agent/evaluate_multiturn_agent.py"


@dataclass(frozen=True, slots=True)
class Variant:
    label: str
    name: str
    architecture: str
    catalog_alias_guard: bool = False
    single_tool_goal_normalization: bool = False
    recent_over_generic_long: bool = False
    natural_request_boundary_guard: bool = False
    whole_turn_reference_replay: bool = False
    standalone_multigoal_guard: bool = False
    whole_turn_plan_cache: bool = False
    semantic_reference_strategy: str = "off"


def variants() -> list[Variant]:
    return [
        Variant("a_stateless", "不读取历史", "A"),
        Variant("b_direct_concat", "直接拼接最近两轮", "B"),
        Variant(
            "c_structured_recent",
            "结构化近期状态",
            "C",
            semantic_reference_strategy="anchor_bm25",
        ),
        Variant(
            "d_rule_summary",
            "加入规则摘要",
            "D",
            semantic_reference_strategy="anchor_bm25",
        ),
        Variant(
            "e_long_memory",
            "加入按需记忆检索",
            "E",
            semantic_reference_strategy="anchor_bm25",
        ),
        Variant(
            "f_route_precedence",
            "加入近期路由优先级",
            "E",
            recent_over_generic_long=True,
            semantic_reference_strategy="anchor_bm25",
        ),
        Variant(
            "g_request_boundary",
            "加入请求边界",
            "E",
            recent_over_generic_long=True,
            natural_request_boundary_guard=True,
            semantic_reference_strategy="anchor_bm25",
        ),
        Variant(
            "h_single_tool_normalization",
            "加入单工具目标归并",
            "E",
            single_tool_goal_normalization=True,
            recent_over_generic_long=True,
            natural_request_boundary_guard=True,
            semantic_reference_strategy="anchor_bm25",
        ),
        Variant(
            "i_catalog_alias",
            "加入严格 Catalog 能力名直达",
            "E",
            catalog_alias_guard=True,
            single_tool_goal_normalization=True,
            recent_over_generic_long=True,
            natural_request_boundary_guard=True,
            semantic_reference_strategy="anchor_bm25",
        ),
        Variant(
            "j_whole_turn_recovery",
            "加入完整意图恢复",
            "E",
            catalog_alias_guard=True,
            single_tool_goal_normalization=True,
            recent_over_generic_long=True,
            natural_request_boundary_guard=True,
            whole_turn_reference_replay=True,
            semantic_reference_strategy="anchor_bm25",
        ),
        Variant(
            "k_standalone_guard",
            "加入自包含保护",
            "E",
            catalog_alias_guard=True,
            single_tool_goal_normalization=True,
            recent_over_generic_long=True,
            natural_request_boundary_guard=True,
            whole_turn_reference_replay=True,
            standalone_multigoal_guard=True,
            semantic_reference_strategy="anchor_bm25",
        ),
        Variant(
            "l_final_cache",
            "加入同版本计划缓存",
            "E",
            catalog_alias_guard=True,
            single_tool_goal_normalization=True,
            recent_over_generic_long=True,
            natural_request_boundary_guard=True,
            whole_turn_reference_replay=True,
            standalone_multigoal_guard=True,
            whole_turn_plan_cache=True,
            semantic_reference_strategy="anchor_bm25",
        ),
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def line_count(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def verify_preregistered_implementation(root: Path) -> None:
    path = root / "preregistration.json"
    if not path.exists():
        raise SystemExit("missing preregistration.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    for relative, expected in payload.get("implementation_sha256", {}).items():
        implementation = REPO_ROOT / str(relative)
        if not implementation.is_file() or sha256_file(implementation) != str(expected):
            raise SystemExit(f"implementation changed after preregistration: {relative}")


def bool_flag(name: str, enabled: bool) -> str:
    return f"--{name}" if enabled else f"--no-{name}"


def evaluator_command(
    variant: Variant,
    split: str,
    gpu: int,
    root: Path,
    python_bin: Path,
    backend_cache: Path | None,
) -> list[str]:
    output = root / f"runs/{split}/{variant.label}"
    direct = variant.architecture == "B"
    command = [
        str(python_bin),
        str(EVALUATOR),
        "--input",
        str(root / f"{split}.jsonl"),
        "--split",
        split,
        "--output-dir",
        str(output),
        "--run-id",
        f"natural-short-memory-{split}-{variant.label}",
        "--allow-tool-combination-overlap",
        "--allow-duplicate-unscored-queries",
        "--allow-duplicate-scored-queries",
        "--allow-reproduced-adapter",
        "--fixed-ecommerce-intent",
        "--intent-device",
        "cuda:0",
        "--embedding-device",
        "cuda:0",
        "--reranker-device",
        "cuda:0",
        "--planner-device",
        "cuda:0",
        "--architecture",
        variant.architecture,
        "--recent-event-limit",
        "2",
        "--long-memory-top-k",
        "2",
        "--target-input-tokens",
        "4096" if direct else "640",
        "--hard-input-tokens",
        "4096" if direct else "1024",
        "--direct-history-max-extra-tokens",
        "4096" if direct else "0",
        bool_flag("catalog-alias-guard", variant.catalog_alias_guard),
        bool_flag(
            "single-tool-goal-normalization",
            variant.single_tool_goal_normalization,
        ),
        bool_flag("recent-over-generic-long", variant.recent_over_generic_long),
        bool_flag(
            "natural-request-boundary-guard",
            variant.natural_request_boundary_guard,
        ),
        bool_flag(
            "whole-turn-reference-replay", variant.whole_turn_reference_replay
        ),
        bool_flag("standalone-multigoal-guard", variant.standalone_multigoal_guard),
        bool_flag("whole-turn-plan-cache", variant.whole_turn_plan_cache),
        "--semantic-reference-strategy",
        variant.semantic_reference_strategy,
    ]
    if backend_cache is not None:
        command.extend(["--backend-cache", str(backend_cache)])
    return command


def complete_run(output: Path, expected_rows: int) -> bool:
    manifest = output / "manifest.json"
    metrics = output / "metrics.json"
    if not manifest.exists() or not metrics.exists():
        return False
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    metric_payload = json.loads(metrics.read_text(encoding="utf-8"))
    backend_errors = int(
        metric_payload.get("overall", {}).get("status_counts", {}).get(
            "backend_error", 0
        )
    )
    return (
        payload.get("status") == "complete"
        and payload.get("prediction_rows") == expected_rows
        and backend_errors == 0
    )


def run_bucket(
    gpu: int,
    bucket: list[Variant],
    split: str,
    root: Path,
    expected_rows: int,
    python_bin: Path,
    backend_cache: Path | None,
) -> list[dict[str, Any]]:
    records = []
    for variant in bucket:
        output = root / f"runs/{split}/{variant.label}"
        if complete_run(output, expected_rows):
            records.append({"label": variant.label, "gpu": gpu, "status": "reused"})
            continue
        output.mkdir(parents=True, exist_ok=True)
        command = evaluator_command(
            variant,
            split,
            gpu,
            root,
            python_bin,
            backend_cache,
        )
        with (output / "run.log").open("a", encoding="utf-8") as handle:
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if result.returncode != 0:
            raise RuntimeError(f"{variant.label} failed; see {output / 'run.log'}")
        if not complete_run(output, expected_rows):
            raise RuntimeError(
                f"{variant.label} produced an incomplete run or backend errors; "
                f"see {output / 'run.log'}"
            )
        records.append({"label": variant.label, "gpu": gpu, "status": "complete"})
    return records


def run_stage(args: argparse.Namespace) -> int:
    verify_preregistered_implementation(args.root)
    selected = variants()
    if args.labels:
        requested = {value for value in args.labels.split(",") if value}
        known = {variant.label for variant in selected}
        if requested - known:
            raise SystemExit(f"unknown labels: {sorted(requested - known)}")
        selected = [variant for variant in selected if variant.label in requested]
    if args.split == "test":
        lock_path = args.root / "selection.lock.json"
        if not lock_path.exists():
            raise SystemExit("test evaluation requires selection.lock.json")
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        allowed = set(lock["test_labels"])
        if not args.labels:
            selected = [variant for variant in selected if variant.label in allowed]
        elif {variant.label for variant in selected} != allowed:
            raise SystemExit(f"test labels must be exactly the locked labels: {sorted(allowed)}")
    dataset = load_jsonl(args.root / f"{args.split}.jsonl")
    if not args.python_bin.is_file():
        raise SystemExit(f"evaluation Python does not exist: {args.python_bin}")
    gpu_ids = [int(value) for value in args.gpus.split(",") if value.strip()]
    if not gpu_ids or len(gpu_ids) != len(set(gpu_ids)):
        raise SystemExit("--gpus must contain unique GPU ids")
    buckets = [selected[index :: len(gpu_ids)] for index in range(len(gpu_ids))]
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = [
            executor.submit(
                run_bucket,
                gpu,
                bucket,
                args.split,
                args.root,
                len(dataset),
                args.python_bin,
                args.backend_cache,
            )
            for gpu, bucket in zip(gpu_ids, buckets)
            if bucket
        ]
        records = [record for future in futures for record in future.result()]
    write_json(
        args.root / f"runs/{args.split}/matrix.json",
        {
            "status": "complete",
            "split": args.split,
            "input_sha256": sha256_file(args.root / f"{args.split}.jsonl"),
            "rows": len(dataset),
            "python_bin": str(args.python_bin.resolve()),
            "backend_cache": (
                None
                if args.backend_cache is None
                else {
                    "path": str(args.backend_cache.resolve()),
                    "sha256": sha256_file(args.backend_cache),
                }
            ),
            "variants": [asdict(variant) for variant in selected],
            "records": sorted(records, key=lambda row: row["label"]),
        },
    )
    print(f"{args.split}: completed {len(records)} candidates")
    return 0


def wilson(correct: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def p95(values: list[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = 0.95 * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def metric(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(bool(row["trace_exact"]) for row in rows)
    planner_model_calls = sum(
        str(row["event"].get("prompt_version", "")) == "tool-trace-v1"
        for row in rows
    )
    plan_cache_hits = sum(
        "whole_turn_plan_cache_applied" in row["event"].get("warnings", [])
        for row in rows
    )
    return {
        "rows": len(rows),
        "correct": correct,
        "trace_em": correct / len(rows) if rows else 0.0,
        "wilson95": wilson(correct, len(rows)),
        "strict_output": sum(bool(row["strict_output"]) for row in rows) / len(rows) if rows else 0.0,
        "planner_model_calls": planner_model_calls,
        "planner_model_call_rate": planner_model_calls / len(rows) if rows else 0.0,
        "validated_plan_cache_hits": plan_cache_hits,
        "input_tokens": {
            "mean": statistics.fmean(int(row["input_tokens"]) for row in rows) if rows else 0.0,
            "p95": p95([int(row["input_tokens"]) for row in rows]),
            "max": max((int(row["input_tokens"]) for row in rows), default=0),
        },
    }


def reference_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    exact = 0
    contains = 0
    context_modes: dict[str, int] = {}
    warnings: dict[str, int] = {}
    for row in rows:
        predicted = [int(value) for value in row["event"]["reference_turn_ids"]]
        gold = [int(value) for value in row["gold_row"]["gold_reference_turn_ids"]]
        exact += predicted == gold
        contains += set(gold) <= set(predicted)
        mode = str(row["event"]["context_mode"])
        context_modes[mode] = context_modes.get(mode, 0) + 1
        for warning in row["event"].get("warnings", []):
            value = str(warning)
            warnings[value] = warnings.get(value, 0) + 1
    total = len(rows)
    return {
        "rows": total,
        "reference_turn_exact": exact,
        "reference_turn_exact_rate": exact / total if total else 0.0,
        "reference_turn_contains_gold": contains,
        "reference_turn_contains_gold_rate": contains / total if total else 0.0,
        "context_mode_counts": dict(sorted(context_modes.items())),
        "warning_counts": dict(sorted(warnings.items())),
    }


def summarize_candidate(root: Path, split: str, variant: Variant) -> dict[str, Any]:
    dataset = {str(row["id"]): row for row in load_jsonl(root / f"{split}.jsonl")}
    metadata = {
        str(row["id"]): row
        for row in load_jsonl(root / f"target_metadata_{split}.jsonl")
    }
    if set(metadata) != set(dataset):
        raise SystemExit(f"{split}: target metadata ids differ from dataset")
    prediction_path = root / f"runs/{split}/{variant.label}/predictions.jsonl"
    predictions = load_jsonl(prediction_path)
    if set(dataset) != {str(row["id"]) for row in predictions}:
        raise SystemExit(f"{split}/{variant.label}: prediction ids differ from dataset")
    joined = [
        {
            **row,
            "gold_row": dataset[str(row["id"])],
            "target_metadata": metadata[str(row["id"])],
        }
        for row in predictions
    ]
    targets = [row for row in joined if row["target_metadata"]["is_scored_target"]]
    history = [row for row in joined if not row["target_metadata"]["is_scored_target"]]

    def target_kind(kind: str) -> list[dict[str, Any]]:
        return [
            row for row in targets if row["target_metadata"]["target_kind"] == kind
        ]

    replay = target_kind("natural_replay")
    standalone = target_kind("standalone_switch")
    contextual = target_kind("contextual_new_action")
    no_inherit = standalone
    replay_metric = metric(replay)
    no_inherit_metric = metric(no_inherit)
    return {
        "label": variant.label,
        "name": variant.name,
        "policy": asdict(variant),
        "targets": metric(targets),
        "replay": replay_metric,
        "replay_reference_diagnostics": reference_diagnostics(replay),
        "contextual_new_action": metric(contextual),
        "standalone_switch": metric(standalone),
        "no_inherit": no_inherit_metric,
        "balanced_macro_trace_em": (
            replay_metric["trace_em"] + no_inherit_metric["trace_em"]
        )
        / 2.0,
        "ordinary_history": metric(history),
        "replay_by_distance": {
            str(distance): metric(
                [
                    row
                    for row in replay
                    if int(row["target_metadata"]["history_gap"]) == distance
                ]
            )
            for distance in range(1, 6)
        },
        "replay_by_family": {
            family: metric(
                [
                    row
                    for row in replay
                    if row["target_metadata"]["reference_family"] == family
                ]
            )
            for family in ("implicit_confirmation", "recent_resume", "object_anchor")
        },
        "replay_by_tool_count": {
            str(count): metric(
                [row for row in replay if len(row["gold_tools"]) == count]
            )
            for count in range(1, 7)
        },
        "predictions": str(prediction_path.relative_to(REPO_ROOT)),
        "predictions_sha256": sha256_file(prediction_path),
    }


def paired(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    left_by_id = {str(row["id"]): bool(row["trace_exact"]) for row in left}
    right_by_id = {str(row["id"]): bool(row["trace_exact"]) for row in right}
    if set(left_by_id) != set(right_by_id):
        raise ValueError("paired prediction ids differ")
    improved = sum(not left_by_id[key] and right_by_id[key] for key in left_by_id)
    regressed = sum(left_by_id[key] and not right_by_id[key] for key in left_by_id)
    discordant = improved + regressed
    if discordant:
        tail = sum(math.comb(discordant, value) for value in range(min(improved, regressed) + 1))
        p_value = min(1.0, 2.0 * tail / (2**discordant))
    else:
        p_value = 1.0
    return {
        "improved": improved,
        "regressed": regressed,
        "net_correct": improved - regressed,
        "delta_pp": 100.0 * (improved - regressed) / len(left_by_id),
        "mcnemar_exact_two_sided_p": p_value,
    }


def selection_stage(args: argparse.Namespace) -> int:
    verify_preregistered_implementation(args.root)
    if any(args.root.glob("runs/test/*/predictions.jsonl")):
        raise SystemExit("test candidate outputs already exist before selection lock")
    expected_rows = len(load_jsonl(args.root / "dev.jsonl"))
    incomplete = [
        variant.label
        for variant in variants()
        if not complete_run(
            args.root / f"runs/dev/{variant.label}", expected_rows
        )
    ]
    if incomplete:
        raise SystemExit(
            f"dev candidates are incomplete or contain backend errors: {incomplete}"
        )
    summaries = [summarize_candidate(args.root, "dev", variant) for variant in variants()]
    baseline = next(row for row in summaries if row["label"] == "a_stateless")
    floor = baseline["no_inherit"]["trace_em"] - 0.01
    for row in summaries:
        row["eligible"] = row["no_inherit"]["trace_em"] + 1e-12 >= floor
        row["no_inherit_floor"] = floor
    eligible = [row for row in summaries if row["eligible"]]
    if not eligible:
        raise SystemExit("no candidate satisfies the pre-registered non-inferiority gate")
    order = {variant.label: index for index, variant in enumerate(variants())}
    selected = max(
        eligible,
        key=lambda row: (
            row["replay"]["trace_em"],
            row["standalone_switch"]["trace_em"],
            -row["targets"]["input_tokens"]["p95"],
            -order[row["label"]],
        ),
    )
    target_ids = {
        str(row["id"])
        for row in load_jsonl(args.root / "target_metadata_dev.jsonl")
        if row["is_scored_target"]
    }
    prediction_rows = {
        variant.label: [
            row
            for row in load_jsonl(
                args.root / f"runs/dev/{variant.label}/predictions.jsonl"
            )
            if str(row["id"]) in target_ids
        ]
        for variant in variants()
    }
    adjacent_comparisons = {
        f"{left.label}_to_{right.label}": paired(
            prediction_rows[left.label], prediction_rows[right.label]
        )
        for left, right in zip(variants(), variants()[1:])
    }
    # 开发集先锁定 selected；独立测试一次性报告整条预注册链，测试结果不再回流选型。
    test_labels = [variant.label for variant in variants()]
    lock = {
        "status": "locked_on_dev_before_test",
        "protocol_sha256": sha256_file(args.root / "protocol.json"),
        "preregistration_sha256": sha256_file(args.root / "preregistration.json"),
        "selection_rule": json.loads(
            (args.root / "preregistration.json").read_text(encoding="utf-8")
        )["selection_rule"],
        "stateless_no_inherit_trace_em": baseline["no_inherit"]["trace_em"],
        "eligibility_floor": floor,
        "candidate_summaries": summaries,
        "adjacent_comparisons": adjacent_comparisons,
        "selected_label": selected["label"],
        "test_labels": test_labels,
        "test_prediction_files_before_lock": 0,
    }
    write_json(args.root / "selection.lock.json", lock)
    print(json.dumps({"selected": selected["label"], "test_labels": test_labels}, ensure_ascii=False))
    return 0


def results_stage(args: argparse.Namespace) -> int:
    verify_preregistered_implementation(args.root)
    lock = json.loads((args.root / "selection.lock.json").read_text(encoding="utf-8"))
    by_label = {variant.label: variant for variant in variants()}
    labels = [str(value) for value in lock["test_labels"]]
    expected_rows = len(load_jsonl(args.root / "test.jsonl"))
    incomplete = [
        label
        for label in labels
        if not complete_run(args.root / f"runs/test/{label}", expected_rows)
    ]
    if incomplete:
        raise SystemExit(
            f"test candidates are incomplete or contain backend errors: {incomplete}"
        )
    confirmation = [summarize_candidate(args.root, "test", by_label[label]) for label in labels]
    selected_label = str(lock["selected_label"])
    selected = next(row for row in confirmation if row["label"] == selected_label)
    target_ids = {
        str(row["id"])
        for row in load_jsonl(args.root / "target_metadata_test.jsonl")
        if row["is_scored_target"]
    }
    prediction_rows = {
        label: [
            row
            for row in load_jsonl(args.root / f"runs/test/{label}/predictions.jsonl")
            if str(row["id"]) in target_ids
        ]
        for label in labels
    }
    comparisons = {
        f"{baseline}_to_selected": paired(
            prediction_rows[baseline], prediction_rows[selected_label]
        )
        for baseline in labels
        if baseline != selected_label
    }
    adjacent_comparisons = {
        f"{left}_to_{right}": paired(
            prediction_rows[left], prediction_rows[right]
        )
        for left, right in zip(labels, labels[1:])
    }
    metadata = {
        str(row["id"]): row
        for row in load_jsonl(args.root / "target_metadata_test.jsonl")
    }
    pairwise_by_slice = {}
    prevalence_break_even = None
    if {"a_stateless", "b_direct_concat"} <= set(labels):
        pairwise_by_slice["a_stateless_to_b_direct_concat"] = {
            "all_scored_targets": paired(
                prediction_rows["a_stateless"],
                prediction_rows["b_direct_concat"],
            ),
            "natural_replay": paired(
                [
                    row
                    for row in prediction_rows["a_stateless"]
                    if metadata[str(row["id"])]["target_kind"] == "natural_replay"
                ],
                [
                    row
                    for row in prediction_rows["b_direct_concat"]
                    if metadata[str(row["id"])]["target_kind"] == "natural_replay"
                ],
            ),
            "standalone_switch": paired(
                [
                    row
                    for row in prediction_rows["a_stateless"]
                    if metadata[str(row["id"])]["target_kind"] == "standalone_switch"
                ],
                [
                    row
                    for row in prediction_rows["b_direct_concat"]
                    if metadata[str(row["id"])]["target_kind"] == "standalone_switch"
                ],
            ),
        }
        by_summary = {row["label"]: row for row in confirmation}
        stateless = by_summary["a_stateless"]
        concat = by_summary["b_direct_concat"]
        standalone_advantage = (
            stateless["standalone_switch"]["trace_em"]
            - concat["standalone_switch"]["trace_em"]
        )
        replay_advantage = (
            concat["replay"]["trace_em"] - stateless["replay"]["trace_em"]
        )
        denominator = standalone_advantage + replay_advantage
        if denominator > 0:
            prevalence_break_even = standalone_advantage / denominator
    sensitivity = {}
    for prevalence in (0.05, 0.10, 0.25, 0.50):
        sensitivity[str(prevalence)] = {
            row["label"]: prevalence * row["replay"]["trace_em"]
            + (1 - prevalence) * row["no_inherit"]["trace_em"]
            for row in confirmation
        }
    all_run_labels = {
        "dev": [variant.label for variant in variants()],
        "test": labels,
    }
    run_audit = []
    aggregate_status: dict[str, int] = {}
    for split, run_labels in all_run_labels.items():
        for label in run_labels:
            output = args.root / f"runs/{split}/{label}"
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            status_counts = {
                str(key): int(value)
                for key, value in metrics["overall"]["status_counts"].items()
            }
            for status, count in status_counts.items():
                aggregate_status[status] = aggregate_status.get(status, 0) + count
            predictions = output / "predictions.jsonl"
            run_audit.append(
                {
                    "split": split,
                    "label": label,
                    "prediction_rows": line_count(predictions),
                    "predictions_sha256": sha256_file(predictions),
                    "status_counts": status_counts,
                }
            )
    cache_manifest_path = args.root / "backend_cache_final.manifest.json"
    cache_audit = None
    if cache_manifest_path.exists():
        cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
        cache_audit = {
            "path": str(cache_manifest_path.relative_to(REPO_ROOT)),
            "sha256": sha256_file(cache_manifest_path),
            "counts": cache_manifest["counts"],
            "output_sha256": cache_manifest["output_sha256"],
        }
    results = {
        "schema_version": 1,
        "status": "complete",
        "metric": "ordered tool-list exact match",
        "protocol": json.loads((args.root / "protocol.json").read_text(encoding="utf-8")),
        "selection": lock,
        "execution": {
            "formal_runs": len(run_audit),
            "prediction_rows": sum(row["prediction_rows"] for row in run_audit),
            "aggregate_status_counts": dict(sorted(aggregate_status.items())),
            "runs": run_audit,
            "backend_cache_audit": cache_audit,
        },
        "confirmation": {
            "candidate_summaries": confirmation,
            "selected_label": selected_label,
            "paired_comparisons": comparisons,
            "adjacent_comparisons": adjacent_comparisons,
            "paired_comparisons_by_slice": pairwise_by_slice,
            "reference_prevalence_sensitivity": sensitivity,
            "reference_prevalence_break_even_a_vs_b": prevalence_break_even,
            "interpretation": (
                "The prevalence weights are sensitivity analysis, not measured production traffic shares."
            ),
        },
    }
    write_json(args.root / "results_final.json", results)
    print(
        json.dumps(
            {
                "selected": selected_label,
                "replay": selected["replay"],
                "no_inherit": selected["no_inherit"],
                "comparisons": comparisons,
            },
            ensure_ascii=False,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("run", "select", "results")
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--gpus", default="3,4,5")
    parser.add_argument("--labels", default="")
    parser.add_argument(
        "--python-bin",
        type=Path,
        default=Path("/root/miniconda/envs/qwen-intent-ft/bin/python"),
    )
    parser.add_argument(
        "--backend-cache",
        type=Path,
        default=None,
        help="Reuse exact backend outputs by input hash; missing entries still run live.",
    )
    args = parser.parse_args()
    args.root = args.root.resolve()
    if args.backend_cache is not None:
        args.backend_cache = args.backend_cache.resolve()
        if not args.backend_cache.is_file():
            raise SystemExit(f"backend cache does not exist: {args.backend_cache}")
    if args.action == "run":
        return run_stage(args)
    if args.action == "select":
        return selection_stage(args)
    return results_stage(args)


if __name__ == "__main__":
    raise SystemExit(main())
