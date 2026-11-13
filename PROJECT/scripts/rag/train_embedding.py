#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Embedding training launcher for the 100-tool retrieval task.

By default this script only writes a reproducible manifest. Pass ``--run`` in
the ``llm_bge`` GPU environment to fine-tune a SentenceTransformer model with
MultipleNegativesRankingLoss.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import RAG_DIR, write_json


DEFAULT_QUERY_INSTRUCTION = "为电商导购助手检索最相关的工具描述："
NEGATIVE_TYPES_BY_MODE = {
    "hard": {"single_same_domain", "multi_partial_replacement"},
    "easy": {"single_cross_domain", "multi_add_irrelevant"},
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{lineno}: expected JSON object")
            rows.append(obj)
    return rows


def with_instruction(query: str, instruction: str) -> str:
    return f"{instruction}{query}" if instruction else query


def select_explicit_negatives(row: dict[str, Any], negative_mode: str) -> tuple[list[str], list[str]]:
    if negative_mode == "none":
        return [], []

    negatives = row.get("neg")
    negative_types = row.get("negative_types")
    if not isinstance(negatives, list) or not isinstance(negative_types, list):
        raise ValueError(
            f"negative_mode={negative_mode!r} requires grouped rows with aligned neg[] and negative_types[]"
        )
    if len(negatives) != len(negative_types):
        raise ValueError(f"neg/negative_types length mismatch: {len(negatives)} != {len(negative_types)}")

    allowed_types = NEGATIVE_TYPES_BY_MODE.get(negative_mode)
    selected: list[str] = []
    selected_types: list[str] = []
    seen: set[str] = set()
    for document, negative_type in zip(negatives, negative_types):
        document = str(document).strip()
        negative_type = str(negative_type).strip()
        if not document or document in seen:
            continue
        if allowed_types is not None and negative_type not in allowed_types:
            continue
        seen.add(document)
        selected.append(document)
        selected_types.append(negative_type)
    if not selected:
        raise ValueError(f"negative_mode={negative_mode!r} selected no explicit negatives")
    return selected, selected_types


def expand_training_examples(
    path: Path,
    limit: int,
    query_instruction: str,
    negative_mode: str,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Expand grouped or pair-level JSONL into MNRL pairs or n-tuples."""
    rows = load_jsonl(path)
    if limit > 0:
        rows = rows[:limit]

    examples: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    grouped_rows = 0
    pair_rows = 0
    duplicate_pairs = 0
    expected_negatives_per_example: int | None = None
    source_negative_types: Counter[str] = Counter()
    assigned_negative_types: Counter[str] = Counter()
    candidate_documents: set[str] = set()
    for row in rows:
        query = str(row.get("query", "")).strip()
        if not query:
            continue
        anchor = with_instruction(query, query_instruction)
        if isinstance(row.get("pos"), list):
            grouped_rows += 1
            positives = [str(doc).strip() for doc in row["pos"] if str(doc).strip()]
            negatives, negative_types = select_explicit_negatives(row, negative_mode)
            source_negative_types.update(negative_types)
        elif int(row.get("label", 0)) == 1 and str(row.get("document", "")).strip():
            if negative_mode != "none":
                raise ValueError(
                    f"negative_mode={negative_mode!r} requires grouped input; pair-level positive rows do not carry neg[]"
                )
            pair_rows += 1
            positives = [str(row["document"]).strip()]
            negatives, negative_types = [], []
        else:
            continue

        if expected_negatives_per_example is None:
            expected_negatives_per_example = len(negatives)
        elif len(negatives) != expected_negatives_per_example:
            raise ValueError(
                "MNRL n-tuples require a fixed number of negatives per row: "
                f"expected {expected_negatives_per_example}, got {len(negatives)}"
            )

        for positive in positives:
            key = (anchor, positive)
            if key in seen:
                duplicate_pairs += 1
                continue
            if positive in negatives:
                raise ValueError("an explicit negative duplicates the paired positive document")
            seen.add(key)
            example = {"anchor": anchor, "positive": positive}
            for index, negative in enumerate(negatives, 1):
                example[f"negative_{index}"] = negative
            examples.append(example)
            candidate_documents.add(positive)
            candidate_documents.update(negatives)
            assigned_negative_types.update(negative_types)

    negatives_per_example = expected_negatives_per_example or 0
    documents_per_example = 1 + negatives_per_example
    stats = {
        "source_rows": len(rows),
        "grouped_rows": grouped_rows,
        "pair_rows": pair_rows,
        "positive_pairs": len(examples),
        "duplicate_pairs_skipped": duplicate_pairs,
        "explicit_negative_mode": negative_mode,
        "explicit_negatives_per_example": negatives_per_example,
        "explicit_negative_assignments": sum(assigned_negative_types.values()),
        "source_negative_types": dict(sorted(source_negative_types.items())),
        "assigned_negative_types": dict(sorted(assigned_negative_types.items())),
        "documents_per_example": documents_per_example,
        "unique_candidate_documents": len(candidate_documents),
        "max_no_duplicates_batch_size": len(candidate_documents) // documents_per_example,
        "query_instruction": query_instruction,
        "limit": limit,
    }
    return examples, stats


def build_manifest(args: argparse.Namespace, status: str, train_stats: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "status": status,
        "model": args.model,
        "train": str(args.train),
        "out_dir": str(args.out_dir),
        "final_model_dir": str(args.out_dir / "final"),
        "loss": args.loss,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps,
        "warmup_ratio": args.warmup_ratio,
        "max_seq_length": args.max_seq_length,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "seed": args.seed,
        "batch_sampler": "no_duplicates" if args.no_duplicates else "batch_sampler",
        "precompute_no_duplicate_hashes": args.precompute_no_duplicate_hashes,
        "query_instruction": args.query_instruction,
        "explicit_negative_mode": args.negative_mode,
        "explicit_negatives_per_example": (train_stats or {}).get("explicit_negatives_per_example", 0),
        "contrastive_micro_batch_size": args.per_device_train_batch_size,
        "contrastive_candidates_per_anchor": (
            args.per_device_train_batch_size * (train_stats or {}).get("documents_per_example", 1) - 1
        ),
        "train_stats": train_stats or {},
        "note": (
            "Gradient accumulation changes the optimizer effective batch, but not the MNRL in-batch candidate pool. "
            "Explicit negatives are passed as negative_1..negative_n columns and enter the MNRL denominator."
        ),
    }


def run_training(args: argparse.Namespace, train_examples: list[dict[str, str]]) -> dict[str, Any]:
    try:
        from datasets import Dataset
        from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
        from sentence_transformers.sentence_transformer.losses import (
            CachedMultipleNegativesRankingLoss,
            MultipleNegativesRankingLoss,
        )
        from sentence_transformers.sentence_transformer.training_args import BatchSamplers
    except ImportError as exc:
        raise SystemExit(
            "Missing training dependency. Use `conda run -n llm_bge python ...` or install sentence-transformers/datasets/torch.\n"
            f"Original error: {exc}"
        )

    model_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if args.device != "auto":
        model_kwargs["device"] = args.device
    model = SentenceTransformer(args.model, **model_kwargs)
    if args.max_seq_length > 0:
        model.max_seq_length = args.max_seq_length

    if args.loss == "cached_mnrl":
        loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=args.cached_mini_batch_size)
    else:
        loss = MultipleNegativesRankingLoss(model)

    dataset = Dataset.from_list(train_examples)
    checkpoints_dir = args.out_dir / "checkpoints"
    save_strategy = "steps" if args.save_steps > 0 else "no"
    if args.no_duplicates:
        batch_sampler = (
            BatchSamplers.NO_DUPLICATES_HASHED
            if args.precompute_no_duplicate_hashes
            else BatchSamplers.NO_DUPLICATES
        )
    else:
        batch_sampler = BatchSamplers.BATCH_SAMPLER

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(checkpoints_dir),
        overwrite_output_dir=args.overwrite_output_dir,
        do_train=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_strategy=save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        seed=args.seed,
        data_seed=args.seed,
        dataloader_drop_last=True,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        report_to="none",
        gradient_checkpointing=args.gradient_checkpointing,
        batch_sampler=batch_sampler,
    )
    trainer = SentenceTransformerTrainer(model=model, args=training_args, train_dataset=dataset, loss=loss)
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    trainer.save_model(str(args.out_dir / "final"))
    return {key: float(value) for key, value in train_result.metrics.items() if isinstance(value, (int, float))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=RAG_DIR / "embedding_grouped_train.jsonl")
    parser.add_argument("--model", default="BAAI/bge-base-zh-v1.5")
    parser.add_argument("--out-dir", type=Path, default=RAG_DIR / "_aux" / "embedding_model_pending")
    parser.add_argument("--loss", choices=["mnrl", "cached_mnrl"], default="mnrl")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--per-device-train-batch-size", type=int, default=64)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--cached-mini-batch-size", type=int, default=32)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Trade compute for activation memory without changing the contrastive batch size.",
    )
    parser.add_argument("--no-duplicates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--precompute-no-duplicate-hashes",
        action="store_true",
        help="Precompute equivalent duplicate-check hashes to avoid slow sampler tail scans.",
    )
    parser.add_argument("--query-instruction", default=DEFAULT_QUERY_INSTRUCTION)
    parser.add_argument(
        "--negative-mode",
        choices=["none", "hard", "easy", "all"],
        default="none",
        help=(
            "Explicit negatives from grouped neg[]: hard=same-domain/partial-replacement, "
            "easy=cross-domain/add-irrelevant, all=all typed negatives."
        ),
    )
    parser.add_argument("--limit", type=int, default=0, help="Debug with the first N source rows.")
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--overwrite-output-dir", action="store_true")
    parser.add_argument("--resume-from-checkpoint", default="")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    if not args.train.exists():
        raise SystemExit(f"missing train file: {args.train}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_examples, train_stats = expand_training_examples(
        args.train,
        args.limit,
        args.query_instruction,
        args.negative_mode,
    )
    if not train_examples:
        raise SystemExit(f"no positive training pairs found in {args.train}")
    if args.no_duplicates:
        max_batch_size = train_stats["max_no_duplicates_batch_size"]
        if args.per_device_train_batch_size > max_batch_size:
            raise SystemExit(
                "no-duplicates cannot form a full batch with this n-tuple layout: "
                f"batch_size={args.per_device_train_batch_size}, upper_bound={max_batch_size}, "
                f"unique_candidate_documents={train_stats['unique_candidate_documents']}, "
                f"documents_per_example={train_stats['documents_per_example']}. "
                "Lower --per-device-train-batch-size or disable --no-duplicates."
            )

    status = "pending" if not args.run else "requested"
    manifest = build_manifest(args, status, train_stats)
    write_json(args.out_dir / "training_manifest.json", manifest)
    if not args.run:
        print(f"embedding training manifest -> {args.out_dir / 'training_manifest.json'}")
        return 0

    started = time.perf_counter()
    train_metrics = run_training(args, train_examples)
    manifest["status"] = "complete"
    manifest["elapsed_sec"] = round(time.perf_counter() - started, 4)
    manifest["train_metrics"] = train_metrics
    write_json(args.out_dir / "training_manifest.json", manifest)
    print(f"embedding model -> {args.out_dir / 'final'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
