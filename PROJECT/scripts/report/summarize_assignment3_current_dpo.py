#!/usr/bin/env python3
"""Summarize current-parent DPO development and protected comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_DIR.parent
EXP_ID = "assignment3_current_api9k_dpo_20260802"
EXP_DIR = PROJECT_DIR / "data/tool_call_trace/_aux/assignment3" / EXP_ID
CONTROLLED_PARENT_ENDPOINT = "parent-controlled"


def parse_endpoint(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("endpoint must be 4b=NAME or 8b=NAME")
    model, name = value.split("=", 1)
    if model not in {"4b", "8b"} or not name:
        raise argparse.ArgumentTypeError("endpoint must be 4b=NAME or 8b=NAME")
    return model, name


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_mcnemar(improved: int, regressed: int) -> float:
    discordant = improved + regressed
    if not discordant:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(improved, regressed) + 1)
    ) / (2**discordant)
    return min(1.0, 2 * tail)


def paired(left_path: Path, right_path: Path, field: str) -> dict[str, Any]:
    left = {str(row["id"]): row for row in read_jsonl(left_path)}
    right = {str(row["id"]): row for row in read_jsonl(right_path)}
    if set(left) != set(right) or len(left) == 0:
        raise ValueError(f"unaligned paired rows: {left_path} vs {right_path}")
    improved = regressed = both_true = both_false = 0
    for trace_id in sorted(left):
        before = left[trace_id]
        after = right[trace_id]
        if before.get("query") != after.get("query"):
            raise ValueError(f"paired query mismatch: {trace_id}")
        left_value = bool(before[field])
        right_value = bool(after[field])
        improved += int(not left_value and right_value)
        regressed += int(left_value and not right_value)
        both_true += int(left_value and right_value)
        both_false += int(not left_value and not right_value)
    count = len(left)
    return {
        "count": count,
        "before": (both_true + regressed) / count,
        "after": (both_true + improved) / count,
        "delta": (improved - regressed) / count,
        "improved": improved,
        "regressed": regressed,
        "both_true": both_true,
        "both_false": both_false,
        "exact_mcnemar_p": exact_mcnemar(improved, regressed),
    }


def metric_snapshot(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if value.get("status") != "complete":
        raise ValueError(f"incomplete metrics: {path}")
    overall = value["overall"]
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "sha256": sha256_file(path),
        "count": int(value["count"]),
        "set_em": float(overall["set_em"]),
        "trace_em": float(overall["trace_em"]),
        "strict_output_valid_rate": float(overall["strict_output_valid_rate"]),
        "micro_precision": float(overall["micro_precision"]),
        "micro_recall": float(overall["micro_recall"]),
        "micro_f1": float(overall["micro_f1"]),
        "omission_rate": float(overall["omission_rate"]),
        "over_call_rate": float(overall["over_call_rate"]),
        "keyword_micro_f1": float(overall["keyword_micro_f1"]),
    }


def protected_panel(model: str, endpoint: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "endpoint": endpoint,
        "controlled_parent_endpoint": CONTROLLED_PARENT_ENDPOINT,
        "datasets": {},
    }
    for split, expected in (("legacy6000", 6000), ("api1000", 1000)):
        result["datasets"][split] = {}
        for view in ("global", "rag"):
            baseline_dir = (
                EXP_DIR
                / "evaluation/protected"
                / model
                / CONTROLLED_PARENT_ENDPOINT
                / split
            )
            dpo_dir = EXP_DIR / "evaluation/protected" / model / endpoint / split
            baseline_metrics = baseline_dir / f"{view}.metrics.json"
            dpo_metrics = dpo_dir / f"{view}.metrics.json"
            baseline_scored = baseline_dir / f"{view}.scored.jsonl"
            dpo_scored = dpo_dir / f"{view}.scored.jsonl"
            before = metric_snapshot(baseline_metrics)
            after = metric_snapshot(dpo_metrics)
            if before["count"] != expected or after["count"] != expected:
                raise ValueError(f"unexpected protected count: {model}/{split}/{view}")
            result["datasets"][split][view] = {
                "sft": before,
                "sft_dpo": after,
                "delta": {
                    key: after[key] - before[key]
                    for key in (
                        "set_em",
                        "trace_em",
                        "strict_output_valid_rate",
                        "micro_precision",
                        "micro_recall",
                        "micro_f1",
                        "omission_rate",
                        "over_call_rate",
                        "keyword_micro_f1",
                    )
                },
                "paired_set_em": paired(baseline_scored, dpo_scored, "set_em"),
                "paired_trace_em": paired(
                    baseline_scored, dpo_scored, "trace_em"
                ),
            }
    baseline_system = (
        EXP_DIR
        / "evaluation/protected"
        / model
        / CONTROLLED_PARENT_ENDPOINT
        / "legacy6000/rag.system.metrics.json"
    )
    dpo_system = (
        EXP_DIR
        / "evaluation/protected"
        / model
        / endpoint
        / "legacy6000/rag.system.metrics.json"
    )
    baseline_system_rows = baseline_system.with_name("rag.system.jsonl")
    dpo_system_rows = dpo_system.with_name("rag.system.jsonl")
    before_system = read_json(baseline_system)
    after_system = read_json(dpo_system)
    if before_system.get("status") != "complete" or after_system.get("status") != "complete":
        raise ValueError(f"incomplete system metrics: {model}")
    result["static_system_9000"] = {
        "sft": {
            "path": str(baseline_system.relative_to(REPO_ROOT)),
            "sha256": sha256_file(baseline_system),
            "full_set": float(before_system["full_pipeline_success"]),
            "full_trace": float(before_system["full_pipeline_trace_success"]),
        },
        "sft_dpo": {
            "path": str(dpo_system.relative_to(REPO_ROOT)),
            "sha256": sha256_file(dpo_system),
            "full_set": float(after_system["full_pipeline_success"]),
            "full_trace": float(after_system["full_pipeline_trace_success"]),
        },
        "delta": {
            "full_set": float(after_system["full_pipeline_success"])
            - float(before_system["full_pipeline_success"]),
            "full_trace": float(after_system["full_pipeline_trace_success"])
            - float(before_system["full_pipeline_trace_success"]),
        },
        "paired_full_set": paired(
            baseline_system_rows, dpo_system_rows, "full_pipeline_success"
        ),
        "paired_full_trace": paired(
            baseline_system_rows, dpo_system_rows, "full_pipeline_trace_success"
        ),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", action="append", type=parse_endpoint, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=EXP_DIR / "evaluation/final_summary.json",
    )
    args = parser.parse_args()
    endpoints = dict(args.endpoint)
    if set(endpoints) != {"4b", "8b"} or len(args.endpoint) != 2:
        parser.error("provide exactly one 4b endpoint and one 8b endpoint")
    models = {}
    for model, endpoint in sorted(endpoints.items()):
        grid_path = EXP_DIR / "selection" / f"{model}_grid.json"
        confirmation_path = EXP_DIR / "selection" / f"{model}_confirmation.json"
        models[model] = {
            "grid_selection": {
                "path": str(grid_path.relative_to(REPO_ROOT)),
                "sha256": sha256_file(grid_path),
                "result": read_json(grid_path),
            },
            "three_seed_confirmation": {
                "path": str(confirmation_path.relative_to(REPO_ROOT)),
                "sha256": sha256_file(confirmation_path),
                "result": read_json(confirmation_path),
            },
            "protected": protected_panel(model, endpoint),
        }
    output = {
        "schema_version": 1,
        "status": "complete",
        "experiment_id": EXP_ID,
        "official_frozen_sets_used_for_grid_or_confirmation": False,
        "models": models,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.out)
    print(f"current DPO summary -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
