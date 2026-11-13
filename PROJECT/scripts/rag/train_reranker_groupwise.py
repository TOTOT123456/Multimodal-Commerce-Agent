#!/usr/bin/env python3
"""Auditable groupwise cross-entropy training for task-specific rerankers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_order(values: list[str], *parts: object) -> list[str]:
    prefix = "\x1f".join(str(part) for part in parts)
    return sorted(
        values,
        key=lambda value: hashlib.sha256(f"{prefix}\x1f{value}".encode()).digest(),
    )


def load_docs(path: Path) -> tuple[dict[str, str], list[str]]:
    docs = {}
    for row in load_jsonl(path):
        tool_id = str(row["tool_id"])
        docs[tool_id] = str(
            row.get("document")
            or row.get("text")
            or row.get("enhanced_desc")
            or row.get("description")
            or ""
        )
    if not docs:
        raise ValueError(f"empty tool corpus: {path}")
    return docs, sorted(docs)


def load_mined(path: Path | None) -> dict[str, list[str]]:
    if path is None:
        return {}
    result = {}
    for row in load_jsonl(path):
        trace_id = str(row.get("id", row.get("trace_id", "")))
        if not trace_id or trace_id in result:
            raise ValueError(f"invalid or duplicate mined prediction id: {trace_id!r}")
        result[trace_id] = [str(value) for value in row.get("pred_tools", [])]
    return result


def fill_unique(
    selected: list[str],
    candidates: list[str],
    excluded: set[str],
    target: int,
) -> None:
    if len(selected) >= target:
        return
    seen = set(selected) | excluded
    for value in candidates:
        if value in seen:
            continue
        selected.append(value)
        seen.add(value)
        if len(selected) >= target:
            return


def build_groups(
    rows: list[dict[str, Any]],
    all_tools: list[str],
    mined: dict[str, list[str]],
    recipe: str,
    seed: int,
    group_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups = []
    negative_sources: Counter[str] = Counter()
    missing_mined_ids = []
    for index, row in enumerate(rows):
        trace_id = str(row.get("trace_id", row.get("id", index)))
        positives = [str(value) for value in row.get("pos_tool_ids", [])]
        typed = [str(value) for value in row.get("neg_tool_ids", [])]
        if not positives:
            raise ValueError(f"row has no positive tools: {trace_id}")
        positive = stable_order(positives, seed, trace_id, "positive")[0]
        excluded = set(positives)
        random_pool = stable_order(all_tools, seed, trace_id, "random_negative")
        mined_pool = mined.get(trace_id, [])
        if recipe != "typed4_random3" and not mined_pool:
            missing_mined_ids.append(trace_id)

        negatives: list[str] = []
        if recipe == "typed4_random3":
            ordered_typed = stable_order(typed, seed, trace_id, "typed")
            before = len(negatives)
            fill_unique(negatives, ordered_typed, excluded, group_size - 1)
            negative_sources["typed"] += len(negatives) - before
            before = len(negatives)
            fill_unique(negatives, random_pool, excluded, group_size - 1)
            negative_sources["random_fill"] += len(negatives) - before
        elif recipe == "source_mined7":
            before = len(negatives)
            fill_unique(negatives, mined_pool, excluded, group_size - 1)
            negative_sources["source_mined"] += len(negatives) - before
            before = len(negatives)
            fill_unique(negatives, random_pool, excluded, group_size - 1)
            negative_sources["random_fill"] += len(negatives) - before
        elif recipe == "mixed_typed3_mined4":
            ordered_typed = stable_order(typed, seed, trace_id, "typed")
            fill_unique(negatives, ordered_typed, excluded, min(3, group_size - 1))
            negative_sources["typed"] += len(negatives)
            before = len(negatives)
            fill_unique(negatives, mined_pool, excluded, group_size - 1)
            negative_sources["source_mined"] += len(negatives) - before
            before = len(negatives)
            fill_unique(negatives, random_pool, excluded, group_size - 1)
            negative_sources["random_fill"] += len(negatives) - before
        else:
            raise ValueError(f"unknown negative recipe: {recipe}")
        if len(negatives) != group_size - 1:
            raise ValueError(f"cannot build full group for {trace_id}: {len(negatives)}")
        groups.append(
            {
                "id": trace_id,
                "query": str(row["query"]),
                "tool_ids": [positive, *negatives],
            }
        )
    if missing_mined_ids:
        raise ValueError(
            f"missing mined candidates for {len(missing_mined_ids)} rows, examples={missing_mined_ids[:5]}"
        )
    stats = {
        "groups": len(groups),
        "group_size": group_size,
        "positive_per_group": 1,
        "negative_per_group": group_size - 1,
        "negative_recipe": recipe,
        "negative_assignments_by_source": dict(sorted(negative_sources.items())),
        "unique_trace_ids": len({row["id"] for row in groups}),
    }
    return groups, stats


def copy_qwen_crossencoder_metadata(model_name: str, output_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(model_name, local_files_only=True))
    for filename in (
        "config_sentence_transformers.json",
        "modules.json",
        "sentence_bert_config.json",
    ):
        source = snapshot / filename
        if source.exists():
            shutil.copy2(source, output_dir / filename)
    source_module = snapshot / "1_LogitScore"
    target_module = output_dir / "1_LogitScore"
    if source_module.exists():
        shutil.copytree(source_module, target_module, dirs_exist_ok=True)


def train(args: argparse.Namespace, groups: list[dict[str, Any]], docs: dict[str, str]) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    class GroupDataset(Dataset):
        def __len__(self) -> int:
            return len(groups)

        def __getitem__(self, index: int) -> dict[str, Any]:
            return groups[index]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    qwen_prefix_tokens: list[int] = []
    qwen_suffix_tokens: list[int] = []
    token_true_id = -1
    token_false_id = -1
    if args.model_kind == "qwen_causal":
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        prefix = (
            '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query '
            'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
            '<|im_start|>user\n'
        )
        suffix = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
        qwen_prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
        qwen_suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
        token_true_id = tokenizer.convert_tokens_to_ids("yes")
        token_false_id = tokenizer.convert_tokens_to_ids("no")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            trust_remote_code=True,
        )
        model.config.use_cache = False
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model,
            trust_remote_code=True,
        )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    parameter_counts = {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }
    model.to(args.device)
    model.train()
    torch.cuda.reset_peak_memory_stats(args.device)

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        queries = []
        passages = []
        ids = []
        for row in batch:
            ids.append(row["id"])
            for tool_id in row["tool_ids"]:
                queries.append(row["query"])
                passages.append(docs[tool_id])
        if args.model_kind == "qwen_causal":
            pair_texts = [
                f"<Instruct>: {args.instruction}\n<Query>: {query}\n<Document>: {passage}"
                for query, passage in zip(queries, passages)
            ]
            encoded = tokenizer(
                pair_texts,
                padding=False,
                truncation="longest_first",
                add_special_tokens=False,
                max_length=args.max_length - len(qwen_prefix_tokens) - len(qwen_suffix_tokens),
            )
            encoded["input_ids"] = [
                qwen_prefix_tokens + values + qwen_suffix_tokens
                for values in encoded["input_ids"]
            ]
            features = tokenizer.pad(encoded, padding=True, return_tensors="pt")
        else:
            features = tokenizer(
                queries,
                passages,
                padding=True,
                truncation="longest_first",
                max_length=args.max_length,
                return_tensors="pt",
            )
        return {"features": features, "ids": ids, "groups": len(batch)}

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        GroupDataset(),
        batch_size=args.per_device_group_batch_size,
        shuffle=True,
        generator=generator,
        drop_last=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        collate_fn=collate,
    )
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight")
    parameter_groups = [
        {
            "params": [
                parameter
                for name, parameter in model.named_parameters()
                if parameter.requires_grad and not any(value in name for value in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                parameter
                for name, parameter in model.named_parameters()
                if parameter.requires_grad and any(value in name for value in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(parameter_groups, lr=args.learning_rate)
    optimizer_steps = math.floor(len(loader) / args.gradient_accumulation_steps) * args.epochs
    warmup_steps = round(optimizer_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, optimizer_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16)

    logs = []
    rolling_loss = 0.0
    rolling_count = 0
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    for epoch in range(args.epochs):
        for batch_index, batch in enumerate(loader, 1):
            features = {key: value.to(args.device, non_blocking=True) for key, value in batch["features"].items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.fp16):
                if args.model_kind == "qwen_causal":
                    outputs = model(**features, logits_to_keep=1)
                    last_logits = outputs.logits[:, -1, :]
                    scores = last_logits[:, token_true_id] - last_logits[:, token_false_id]
                else:
                    outputs = model(**features)
                    scores = outputs.logits.reshape(-1)
                scores = scores.reshape(batch["groups"], args.group_size)
                labels = torch.zeros(batch["groups"], dtype=torch.long, device=args.device)
                raw_loss = functional.cross_entropy(scores, labels)
                loss = raw_loss / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
            rolling_loss += float(raw_loss.detach())
            rolling_count += 1
            if batch_index % args.gradient_accumulation_steps:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if global_step % args.logging_steps == 0 or global_step == optimizer_steps:
                row = {
                    "step": global_step,
                    "epoch": epoch + batch_index / len(loader),
                    "loss": rolling_loss / max(1, rolling_count),
                    "learning_rate": scheduler.get_last_lr()[0],
                }
                logs.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
                rolling_loss = 0.0
                rolling_count = 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.out_dir)
    if args.model_kind == "qwen_causal":
        copy_qwen_crossencoder_metadata(args.model, args.out_dir)
    return {
        "elapsed_sec": time.perf_counter() - started,
        "optimizer_steps": global_step,
        "warmup_steps": warmup_steps,
        "log_history": logs,
        "environment": {
            "gpu": torch.cuda.get_device_name(args.device),
            "torch": torch.__version__,
        },
        "parameter_counts": parameter_counts,
        "peak_memory": {
            "allocated_gib": torch.cuda.max_memory_allocated(args.device) / (1024**3),
            "reserved_gib": torch.cuda.max_memory_reserved(args.device) / (1024**3),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--docs", type=Path, required=True)
    parser.add_argument("--mined-preds", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-kind", choices=["encoder", "qwen_causal"], default="encoder")
    parser.add_argument("--negative-recipe", choices=["typed4_random3", "source_mined7", "mixed_typed3_mined4"], default="typed4_random3")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--per-device-group-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=6e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--instruction", default="Given a web search query, retrieve relevant passages that answer the query")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    if args.group_size < 2:
        parser.error("--group-size must be at least 2")
    if args.per_device_group_batch_size <= 0 or args.gradient_accumulation_steps <= 0:
        parser.error("batch sizes must be positive")
    if args.negative_recipe != "typed4_random3" and args.mined_preds is None:
        parser.error(f"--negative-recipe {args.negative_recipe} requires --mined-preds")
    train_rows = load_jsonl(args.train)
    if args.limit:
        train_rows = train_rows[: args.limit]
    docs, all_tools = load_docs(args.docs)
    mined = load_mined(args.mined_preds)
    groups, group_stats = build_groups(
        train_rows,
        all_tools,
        mined,
        args.negative_recipe,
        args.seed,
        args.group_size,
    )
    manifest = {
        "status": "pending" if not args.run else "requested",
        "model": args.model,
        "model_kind": args.model_kind,
        "train": str(args.train),
        "train_sha256": sha256_file(args.train),
        "docs": str(args.docs),
        "docs_sha256": sha256_file(args.docs),
        "mined_preds": str(args.mined_preds) if args.mined_preds else None,
        "mined_preds_sha256": sha256_file(args.mined_preds) if args.mined_preds else None,
        "negative_recipe": args.negative_recipe,
        "group_stats": group_stats,
        "group_size": args.group_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "per_device_group_batch_size": args.per_device_group_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_group_batch_size": args.per_device_group_batch_size * args.gradient_accumulation_steps,
        "max_length": args.max_length,
        "seed": args.seed,
        "fp16": args.fp16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "out_dir": str(args.out_dir),
    }
    write_json(args.out_dir / "training_manifest.json", manifest)
    if not args.run:
        print(f"reranker training manifest -> {args.out_dir / 'training_manifest.json'}")
        return 0
    result = train(args, groups, docs)
    manifest.update(result)
    manifest["status"] = "complete"
    write_json(args.out_dir / "training_manifest.json", manifest)
    print(f"reranker model -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
