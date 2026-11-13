#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize live retrieval plus warmed generator latency measurements."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import load_jsonl, read_json, write_json  # noqa: E402
from finetune.tool_trace_common import sha256_file, summary_stats  # noqa: E402


def parse_generator(value: str) -> tuple[str, Path, Path]:
    fields = value.split("=", 1)
    if len(fields) != 2:
        raise argparse.ArgumentTypeError("--generator must be NAME=PREDICTIONS,MANIFEST")
    name, paths = fields
    path_fields = paths.split(",", 1)
    if not name or len(path_fields) != 2:
        raise argparse.ArgumentTypeError("--generator must be NAME=PREDICTIONS,MANIFEST")
    return name, Path(path_fields[0]), Path(path_fields[1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-ranking", type=Path, required=True)
    parser.add_argument("--fixed-ranking", type=Path, required=True)
    parser.add_argument("--retrieval-manifest", type=Path, required=True)
    parser.add_argument("--candidate-k", type=int, required=True)
    parser.add_argument("--generator", action="append", type=parse_generator, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    live = load_jsonl(args.live_ranking)
    fixed = load_jsonl(args.fixed_ranking)
    if len(live) != 6000 or len(fixed) != 6000:
        raise SystemExit("latency ranking comparison requires 6000 rows")
    fixed_by_id = {row["id"]: row for row in fixed}
    if set(fixed_by_id) != {row["id"] for row in live}:
        raise SystemExit("live/fixed ranking ids differ")
    full_equal = 0
    topk_equal = 0
    for row in live:
        reference = fixed_by_id[row["id"]]
        if row["query"] != reference["query"]:
            raise SystemExit(f"query mismatch: {row['id']}")
        full_equal += int(row["pred_tools"] == reference["pred_tools"])
        topk_equal += int(
            row["pred_tools"][: args.candidate_k]
            == reference["pred_tools"][: args.candidate_k]
        )

    retrieval = read_json(args.retrieval_manifest)
    if retrieval.get("status") != "complete" or retrieval.get("count") != 6000:
        raise SystemExit("incomplete live retrieval manifest")
    timing = retrieval["timing"]
    retrieval_mean = timing["production_steady_state_avg_ms_per_query"]
    retrieval_p50_stage_sum = (
        timing["bm25"]["p50_ms"]
        + timing["fine_embedding"]["encode_avg_ms"]
        + timing["fine_embedding"]["search_p50_ms"]
        + timing["source_fusion_ms_per_query"]
        + timing["task_reranker"]["p50_ms_per_query"]
        + timing["final_fusion_ms_per_query"]
    )
    retrieval_p95_stage_sum = (
        timing["bm25"]["p95_ms"]
        + timing["fine_embedding"]["encode_avg_ms"]
        + timing["fine_embedding"]["search_p95_ms"]
        + timing["source_fusion_ms_per_query"]
        + timing["task_reranker"]["p95_ms_per_query"]
        + timing["final_fusion_ms_per_query"]
    )

    generators: dict[str, Any] = {}
    for name, prediction_path, manifest_path in args.generator:
        predictions = load_jsonl(prediction_path)
        manifest = read_json(manifest_path)
        if len(predictions) != 6000 or manifest.get("status") != "complete":
            raise SystemExit(f"{name}: incomplete warmed generator run")
        if manifest.get("warmup_batches", 0) < 2:
            raise SystemExit(f"{name}: expected at least two warmup batches")
        if manifest.get("prediction_sha256") != sha256_file(prediction_path):
            raise SystemExit(f"{name}: prediction manifest hash mismatch")
        generator_latencies = [float(row["latency_ms"]) for row in predictions]
        prompt_latencies = [float(row["prompt_build_latency_ms"]) for row in predictions]
        total_generator_side = [
            prompt + generator
            for prompt, generator in zip(prompt_latencies, generator_latencies)
        ]
        generator_stats = summary_stats(generator_latencies)
        prompt_stats = summary_stats(prompt_latencies)
        generator_side_stats = summary_stats(total_generator_side)
        generators[name] = {
            "predictions": str(prediction_path),
            "predictions_sha256": sha256_file(prediction_path),
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "batch_size": manifest["run_config"].get("batch_size"),
            "warmup_batches": manifest["warmup_batches"],
            "prompt_build_ms": prompt_stats,
            "generator_ms": generator_stats,
            "prompt_plus_generator_ms": generator_side_stats,
            "steady_state_total_mean_ms": retrieval_mean + generator_side_stats["mean"],
            "stage_sum_total_p50_ms": retrieval_p50_stage_sum + generator_side_stats["p50"],
            "stage_sum_total_p95_ms": retrieval_p95_stage_sum + generator_side_stats["p95"],
            "peak_gpu_memory_gib": manifest.get("peak_gpu_memory_gib"),
            "peak_gpu_reserved_gib": manifest.get("peak_gpu_reserved_gib"),
            "strict_parse_rate": manifest["strict_parse_rate"],
        }

    output = {
        "status": "complete",
        "candidate_k": args.candidate_k,
        "ranking_equivalence": {
            "count": len(live),
            "full_ranking_equal": full_equal,
            "full_ranking_equal_rate": full_equal / len(live),
            "topk_equal": topk_equal,
            "topk_equal_rate": topk_equal / len(live),
            "live_ranking": str(args.live_ranking),
            "live_ranking_sha256": sha256_file(args.live_ranking),
            "fixed_ranking": str(args.fixed_ranking),
            "fixed_ranking_sha256": sha256_file(args.fixed_ranking),
        },
        "retrieval": {
            "manifest": str(args.retrieval_manifest),
            "manifest_sha256": sha256_file(args.retrieval_manifest),
            "timing": timing,
            "steady_state_mean_ms_per_query": retrieval_mean,
            "stage_sum_p50_ms": retrieval_p50_stage_sum,
            "stage_sum_p95_ms": retrieval_p95_stage_sum,
        },
        "generators": generators,
        "measurement_scope": {
            "steady_state_total_mean": (
                "Measured live retrieval steady-state average plus measured warmed "
                "prompt-build/generation average on the resulting live candidates."
            ),
            "stage_sum_quantiles": (
                "Stage-wise p50/p95 sums are diagnostics, not quantiles sampled from "
                "one joint per-query latency distribution."
            ),
            "model_load": "Excluded from steady-state totals and reported in retrieval/generator manifests.",
        },
    }
    write_json(args.out, output)
    print(f"tool-trace latency summary -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
