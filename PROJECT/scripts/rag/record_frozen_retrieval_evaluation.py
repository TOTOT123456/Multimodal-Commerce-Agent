#!/usr/bin/env python3
"""Record that an evaluation split was exposed only after the pipeline freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / "PROJECT"
ORIGINAL_ROUTE = (
    PROJECT
    / "data/rag/_aux/retrieval_pipeline_reselection/bs64_control_route.snapshot.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def jsonl_rows(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("begin", "complete"), required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    frozen = load_json(args.frozen)
    if frozen.get("status") != "frozen_before_round_final":
        raise SystemExit("pipeline is not in frozen_before_round_final state")
    if frozen.get("post_final_tuning_forbidden") is not True:
        raise SystemExit("frozen pipeline does not prohibit post-final tuning")
    embedding_weights = Path(frozen["embedding"]["checkpoint"]) / "model.safetensors"
    reranker_weights = Path(frozen["reranker"]["model"]) / "model.safetensors"
    for label, path, expected in (
        ("embedding", embedding_weights, frozen["embedding"]["model_sha256"]),
        ("reranker", reranker_weights, frozen["reranker"]["model_sha256"]),
    ):
        if not path.exists() or sha256_file(path) != expected:
            raise SystemExit(f"frozen {label} weights are missing or changed: {path}")
    dataset_rows = jsonl_rows(args.dataset)
    if dataset_rows != args.expected_rows:
        raise SystemExit(
            f"dataset row count mismatch: expected={args.expected_rows} actual={dataset_rows}"
        )

    frozen_hash = sha256_file(args.frozen)
    dataset_hash = sha256_file(args.dataset)
    original = load_json(ORIGINAL_ROUTE)
    original_embedding = ROOT / original["embedding_checkpoint"] / "model.safetensors"
    original_reranker = ROOT / original["reranker_checkpoint"] / "model.safetensors"
    if not original_embedding.exists() or not original_reranker.exists():
        raise SystemExit("original canonical checkpoints are missing")
    manifest_path = args.out_dir / "evaluation_manifest.json"
    now = datetime.now(timezone.utc).isoformat()

    if args.stage == "begin":
        if manifest_path.exists():
            existing = load_json(manifest_path)
            if (
                existing.get("frozen_pipeline_sha256") != frozen_hash
                or existing.get("dataset_sha256") != dataset_hash
                or existing.get("rows") != dataset_rows
                or existing.get("original_route_sha256") != sha256_file(ORIGINAL_ROUTE)
                or existing.get("original_embedding_model_sha256")
                != sha256_file(original_embedding)
                or existing.get("original_reranker_model_sha256")
                != sha256_file(original_reranker)
            ):
                raise SystemExit("existing evaluation manifest does not match frozen inputs")
            print(f"resume authorized evaluation: {manifest_path}")
            return 0
        forbidden_existing = [
            args.out_dir / "challenger/metrics.json",
            args.out_dir / "baseline/metrics.json",
            args.out_dir / "comparison.json",
        ]
        if any(path.exists() for path in forbidden_existing):
            raise SystemExit("evaluation output exists without a prior exposure manifest")
        write_json(
            manifest_path,
            {
                "schema_version": 1,
                "status": "authorized_after_pipeline_freeze",
                "label": args.label,
                "started_at_utc": now,
                "rows": dataset_rows,
                "dataset": str(args.dataset),
                "dataset_sha256": dataset_hash,
                "frozen_pipeline": str(args.frozen),
                "frozen_pipeline_sha256": frozen_hash,
                "embedding_model_sha256": frozen["embedding"]["model_sha256"],
                "reranker_model_sha256": frozen["reranker"]["model_sha256"],
                "original_route": str(ORIGINAL_ROUTE),
                "original_route_sha256": sha256_file(ORIGINAL_ROUTE),
                "original_embedding_model_sha256": sha256_file(original_embedding),
                "original_reranker_model_sha256": sha256_file(original_reranker),
                "post_evaluation_tuning_forbidden": True,
            },
        )
        print(f"authorized evaluation after freeze: {manifest_path}")
        return 0

    if not manifest_path.exists():
        raise SystemExit("cannot complete evaluation without begin manifest")
    manifest = load_json(manifest_path)
    if (
        manifest.get("frozen_pipeline_sha256") != frozen_hash
        or manifest.get("dataset_sha256") != dataset_hash
        or manifest.get("original_route_sha256") != sha256_file(ORIGINAL_ROUTE)
        or manifest.get("original_embedding_model_sha256")
        != sha256_file(original_embedding)
        or manifest.get("original_reranker_model_sha256")
        != sha256_file(original_reranker)
    ):
        raise SystemExit("evaluation inputs changed after begin manifest")
    required = {
        "challenger_metrics": args.out_dir / "challenger/metrics.json",
        "baseline_metrics": args.out_dir / "baseline/metrics.json",
        "comparison": args.out_dir / "comparison.json",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise SystemExit(f"cannot complete evaluation; missing={missing}")
    challenger = load_json(required["challenger_metrics"])
    baseline = load_json(required["baseline_metrics"])
    comparison = load_json(required["comparison"])
    manifest.update(
        {
            "status": "complete_no_post_evaluation_tuning",
            "completed_at_utc": now,
            "results": {
                "challenger": challenger["overall"],
                "baseline": baseline["overall"],
                "comparison_primary": comparison["overall"]["metrics"]["allhit_at_10"],
                "promotion_gate": comparison["promotion_gate"],
            },
            "artifact_sha256": {
                name: sha256_file(path) for name, path in required.items()
            },
        }
    )
    write_json(manifest_path, manifest)
    print(f"completed frozen evaluation: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
