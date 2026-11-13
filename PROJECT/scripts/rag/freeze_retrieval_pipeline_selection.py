#!/usr/bin/env python3
"""Freeze the selected development route before any round-final evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / "PROJECT"
DEFAULT_AUX = PROJECT / "data/rag/_aux/retrieval_pipeline_reselection"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def reranker_metadata(model_dir: Path, selection_group: str) -> dict[str, Any]:
    manifest_path = model_dir / "training_manifest.json"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        base_model = str(manifest.get("model", ""))
        architecture_by_model = {
            "BAAI/bge-reranker-base": "bge_base",
            "BAAI/bge-reranker-large": "bge_large",
            "BAAI/bge-reranker-v2-m3": "bge_v2_m3",
            "Qwen/Qwen3-Reranker-0.6B": "qwen3_0p6b",
        }
        architecture = architecture_by_model.get(base_model)
        if architecture is None:
            raise ValueError(f"unknown reranker base model in manifest: {base_model}")
        return {
            "architecture": architecture,
            "base_model": base_model,
            "model_kind": manifest.get("model_kind"),
            "negative_recipe": manifest.get("negative_recipe"),
            "learning_rate": manifest.get("learning_rate"),
            "parameter_counts": manifest.get("parameter_counts"),
            "training_manifest": str(manifest_path),
            "training_manifest_sha256": sha256_file(manifest_path),
        }
    if "bge-reranker-base-e1-g8-bs16" in str(model_dir):
        return {
            "architecture": "existing_task_bge_base",
            "base_model": "BAAI/bge-reranker-base",
            "model_kind": "encoder",
            "negative_recipe": "legacy_task_finetune",
            "learning_rate": 6e-5,
            "parameter_counts": None,
            "training_manifest": None,
            "training_manifest_sha256": None,
        }
    raise ValueError(
        f"cannot infer reranker architecture for selection_group={selection_group}: {model_dir}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection-json",
        type=Path,
        default=DEFAULT_AUX / "downstream/negative_screen/pipeline_search.json",
    )
    parser.add_argument(
        "--embedding-json",
        type=Path,
        default=DEFAULT_AUX / "embedding_selection.json",
    )
    parser.add_argument(
        "--protocol-json",
        type=Path,
        default=DEFAULT_AUX / "protocol.snapshot.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_AUX / "frozen_pipeline.json",
    )
    args = parser.parse_args()

    selection = load_json(args.selection_json)
    embedding = load_json(args.embedding_json)
    selected = selection["selected"]
    group = str(selected["group"])
    seed = int(selected["deployment_seed"])
    matching_specs = [
        spec
        for spec in selection["reranker_specs"]
        if str(spec["group"]) == group and int(spec["seed"]) == seed
    ]
    if len(matching_specs) != 1:
        raise ValueError(
            f"expected one selected reranker spec for group={group} seed={seed}, found {len(matching_specs)}"
        )
    selected_preds = resolve_path(str(matching_specs[0]["path"]))
    reranker_metrics_path = selected_preds.parent / "metrics.json"
    reranker_metrics = load_json(reranker_metrics_path)
    reranker_model = resolve_path(str(reranker_metrics["model"]))
    reranker_weights = reranker_model / "model.safetensors"
    if not reranker_weights.exists():
        raise FileNotFoundError(reranker_weights)
    reranker_info = reranker_metadata(reranker_model, group)

    embedding_checkpoint = resolve_path(
        str(embedding["selected"]["checkpoint"]["checkpoint"])
    )
    embedding_weights = embedding_checkpoint / "model.safetensors"
    if not embedding_weights.exists():
        raise FileNotFoundError(embedding_weights)

    payload = {
        "schema_version": 1,
        "status": "frozen_before_round_final",
        "selection_split": "development_crossstyle_1200",
        "post_final_tuning_forbidden": True,
        "selected": {
            "group": group,
            "deployment_seed": seed,
            "route": selected["route"],
            "development_aggregate": selected["aggregate"],
            "development_seed_metrics": selected["deployment_seed_metrics"],
        },
        "embedding": {
            "recipe": embedding["selected"]["recipe"],
            "seed": embedding["selected"]["checkpoint"]["seed"],
            "checkpoint": str(embedding_checkpoint),
            "model_sha256": sha256_file(embedding_weights),
        },
        "reranker": {
            "group": reranker_info["architecture"],
            "selection_group": group,
            "seed": seed,
            "model": str(reranker_model),
            "model_sha256": sha256_file(reranker_weights),
            "development_predictions": str(selected_preds),
            "development_predictions_sha256": sha256_file(selected_preds),
            **reranker_info,
        },
        "inputs": {
            str(path): sha256_file(path)
            for path in (
                args.selection_json,
                args.embedding_json,
                args.protocol_json,
                reranker_metrics_path,
            )
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"frozen group={group} seed={seed} AllHit@10="
        f"{selected['deployment_seed_metrics']['allhit_at_10']:.6f} -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
