#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run label-free Qwen3 tool-trace inference with resumable shard output."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import CATALOG_PATH, catalog_by_id, load_catalog, load_jsonl, write_json, write_jsonl  # noqa: E402
from finetune.tool_trace_common import (  # noqa: E402
    prompt_hash,
    prompt_version_for_view,
    recover_parse_tools,
    render_messages,
    deterministic_shuffle,
    sha256_file,
    sha256_json,
    strict_parse_tools,
)


def validate_input_hash(row: dict[str, Any]) -> bool:
    payload = {key: value for key, value in row.items() if key != "input_hash"}
    return row.get("input_hash") == sha256_json(payload)


def apply_chat_template(tokenizer: Any, messages: list[dict[str, str]]) -> tuple[str, bool]:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    try:
        return tokenizer.apply_chat_template(messages, **kwargs), True
    except TypeError:
        kwargs.pop("enable_thinking")
        return tokenizer.apply_chat_template(messages, **kwargs), False


def model_identity(model_path: Path, adapter: Path | None) -> dict[str, Any]:
    files = {}
    for name in ["config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json"]:
        path = model_path / name
        if path.exists():
            files[f"model/{name}"] = sha256_file(path)
    if adapter:
        for name in ["adapter_config.json", "adapter_model.safetensors", "args.json"]:
            path = adapter / name
            if path.exists():
                files[f"adapter/{name}"] = sha256_file(path)
    return {
        "model_path": str(model_path),
        "adapter": str(adapter) if adapter else "",
        "files": files,
        "revision_hash": sha256_json(files),
    }


def output_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row.get("id")), int(row.get("sample_index", 0))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Audited label-free JSONL.")
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    parser.add_argument("--view", choices=["query_only", "global", "rag"], required=True)
    parser.add_argument(
        "--candidate-order",
        choices=["original", "reverse", "hash_shuffle"],
        default="original",
        help="Presentation order for an unchanged candidate set.",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-name", default="")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--max-memory-per-gpu-gib",
        type=int,
        default=0,
        help="Required with --device auto; caps each visible GPU for model sharding.",
    )
    parser.add_argument("--torch-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=0,
        help="Untimed generation warmup batches; intended for separate latency runs.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--samples-per-prompt", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260725)
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"missing model: {args.model}")
    if args.adapter and not args.adapter.exists():
        raise SystemExit(f"missing adapter: {args.adapter}")
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("invalid shard count/index")
    if not args.do_sample and args.samples_per_prompt != 1:
        raise SystemExit("deterministic inference requires --samples-per-prompt 1")
    if args.samples_per_prompt <= 0:
        raise SystemExit("--samples-per-prompt must be positive")
    if args.device == "auto" and args.max_memory_per_gpu_gib <= 0:
        raise SystemExit("--device auto requires --max-memory-per-gpu-gib")
    if args.device != "auto" and args.max_memory_per_gpu_gib:
        raise SystemExit("--max-memory-per-gpu-gib is only valid with --device auto")
    if args.view == "query_only" and args.candidate_order != "original":
        raise SystemExit("query_only does not have candidates to reorder")

    input_rows = load_jsonl(args.input)
    if not input_rows:
        raise SystemExit(f"empty input: {args.input}")
    expected_fields = {"id", "query", "input_hash"}
    if args.view == "rag":
        expected_fields |= {"candidate_tools", "candidate_k", "ranking_hash"}
    for index, row in enumerate(input_rows):
        if set(row) != expected_fields:
            raise SystemExit(f"input row {index}: field whitelist failed: {sorted(row)}")
        if not validate_input_hash(row):
            raise SystemExit(f"input row {index}: input_hash mismatch")
    shard_rows = [
        row for index, row in enumerate(input_rows) if index % args.shard_count == args.shard_index
    ]
    if args.limit:
        shard_rows = shard_rows[: args.limit]

    catalog = load_catalog(args.catalog)
    by_id = catalog_by_id(catalog)
    catalog_ids = set(by_id)
    catalog_order = [row["tool_id"] for row in catalog]
    for row in shard_rows:
        if args.view == "rag":
            candidates = row["candidate_tools"]
            if (
                len(candidates) != row["candidate_k"]
                or len(candidates) != len(set(candidates))
                or not set(candidates) <= catalog_ids
            ):
                raise SystemExit(f"{row['id']}: invalid RAG candidates")

    identity = model_identity(args.model, args.adapter)
    input_sha = sha256_file(args.input)
    catalog_sha = sha256_file(args.catalog)
    prompt_version = prompt_version_for_view(args.view)
    run_config = {
        "run_id": args.run_id,
        "input_sha256": input_sha,
        "catalog_sha256": catalog_sha,
        "view": args.view,
        "candidate_order": args.candidate_order,
        "model_identity": identity,
        "prompt_version": prompt_version,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "device": args.device,
        "max_memory_per_gpu_gib": args.max_memory_per_gpu_gib,
        "torch_dtype": args.torch_dtype,
        "attn_implementation": args.attn_implementation,
        "do_sample": args.do_sample,
        "temperature": args.temperature if args.do_sample else None,
        "top_p": args.top_p if args.do_sample else None,
        "top_k": args.top_k if args.do_sample else None,
        "samples_per_prompt": args.samples_per_prompt,
        "seed": args.seed,
        "warmup_batches": args.warmup_batches,
        "limit": args.limit,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
    }
    run_config_hash = sha256_json(run_config)
    existing = load_jsonl(args.output) if args.resume and args.output.exists() else []
    if existing and any(row.get("run_config_hash") != run_config_hash for row in existing):
        raise SystemExit(f"{args.output}: existing rows belong to a different run configuration")
    existing_by_key = {output_key(row): row for row in existing}

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.torch_dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    load_start = time.perf_counter()
    model_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
        "attn_implementation": args.attn_implementation,
    }
    if args.device == "auto":
        if not torch.cuda.is_available():
            raise SystemExit("--device auto requires CUDA")
        model_kwargs.update(
            {
                "device_map": "auto",
                "max_memory": {
                    **{
                        index: f"{args.max_memory_per_gpu_gib}GiB"
                        for index in range(torch.cuda.device_count())
                    },
                    "cpu": "128GiB",
                },
            }
        )
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs).to(
            args.device
        )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    input_device = model.get_input_embeddings().weight.device
    load_seconds = time.perf_counter() - load_start

    pending = [
        row
        for row in shard_rows
        if any((row["id"], sample_index) not in existing_by_key for sample_index in range(args.samples_per_prompt))
    ]
    processed_since_save = 0
    thinking_control_supported = True
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "num_return_sequences": args.samples_per_prompt,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.do_sample:
        generation_kwargs.update(
            {"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k}
        )
    if args.device == "auto":
        synchronize = lambda: [
            torch.cuda.synchronize(index) for index in range(torch.cuda.device_count())
        ]
    else:
        synchronize = torch.cuda.synchronize if args.device.startswith("cuda") else lambda: None

    def ordered_candidates(row: dict[str, Any]) -> list[str]:
        candidates = (
            catalog_order
            if args.view == "global"
            else row["candidate_tools"]
            if args.view == "rag"
            else []
        )
        if args.candidate_order == "original":
            return list(candidates)
        if args.candidate_order == "reverse":
            return list(reversed(candidates))
        return deterministic_shuffle(
            candidates,
            "tool-trace-candidate-order-v1",
            args.view,
            row["id"],
        )

    def prepare_batch(
        batch: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], list[list[str]], list[str], bool, float]:
        prepare_start = time.perf_counter()
        prompts = []
        batch_candidates = []
        batch_prompt_hashes = []
        supported_for_batch = True
        for row in batch:
            candidates = ordered_candidates(row)
            messages = render_messages(row["query"], candidates, by_id, args.view)
            prompt, supported = apply_chat_template(tokenizer, messages)
            supported_for_batch = supported_for_batch and supported
            prompts.append(prompt)
            batch_candidates.append(candidates)
            batch_prompt_hashes.append(prompt_hash(messages, prompt_version))
        encoded = tokenizer(prompts, return_tensors="pt", padding=True)
        encoded = {key: value.to(input_device) for key, value in encoded.items()}
        synchronize()
        prepare_ms = (time.perf_counter() - prepare_start) * 1000.0
        return encoded, batch_candidates, batch_prompt_hashes, supported_for_batch, prepare_ms

    warmup_count = min(args.warmup_batches, len(pending))
    for warmup_index in range(warmup_count):
        warmup_start = (warmup_index * args.batch_size) % len(pending)
        warmup_batch = pending[warmup_start : warmup_start + args.batch_size]
        if not warmup_batch:
            warmup_batch = pending[: args.batch_size]
        warmup_encoded, _, _, supported, _ = prepare_batch(warmup_batch)
        thinking_control_supported = thinking_control_supported and supported
        with torch.inference_mode():
            model.generate(**warmup_encoded, **generation_kwargs)
        synchronize()

    if args.device == "auto":
        for index in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(index)
    elif args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for batch_start in range(0, len(pending), args.batch_size):
        batch = pending[batch_start : batch_start + args.batch_size]
        encoded, batch_candidates, batch_prompt_hashes, supported, prepare_ms = prepare_batch(batch)
        thinking_control_supported = thinking_control_supported and supported
        synchronize()
        infer_start = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**encoded, **generation_kwargs)
        synchronize()
        elapsed_ms = (time.perf_counter() - infer_start) * 1000.0
        input_width = encoded["input_ids"].shape[1]
        generated_only = generated[:, input_width:]
        for row_index, row in enumerate(batch):
            input_tokens = int(encoded["attention_mask"][row_index].sum().item())
            candidates = batch_candidates[row_index]
            candidate_set = set(candidates) if args.view == "rag" else None
            for sample_index in range(args.samples_per_prompt):
                output_index = row_index * args.samples_per_prompt + sample_index
                token_ids = generated_only[output_index]
                raw_output = tokenizer.decode(token_ids, skip_special_tokens=True)
                pred_tools, strict_ok, strict_error = strict_parse_tools(
                    raw_output,
                    catalog_ids,
                    candidate_set,
                )
                recover_tools, recover_ok = recover_parse_tools(raw_output)
                prediction = {
                    "id": row["id"],
                    "query": row["query"],
                    "raw_output": raw_output,
                    "pred_tools": pred_tools,
                    "strict_parse_ok": strict_ok,
                    "strict_error": strict_error,
                    "recover_tools": recover_tools,
                    "recover_parse_ok": recover_ok,
                    "view": args.view,
                    "candidate_order_variant": args.candidate_order,
                    "candidate_k": len(candidates),
                    "candidate_tools": candidates,
                    "ranking_hash": row.get("ranking_hash", ""),
                    "model": args.model_name or args.model.name,
                    "model_path": str(args.model),
                    "model_revision": identity["revision_hash"],
                    "adapter": str(args.adapter) if args.adapter else "",
                    "prompt_version": prompt_version,
                    "prompt_hash": batch_prompt_hashes[row_index],
                    "input_hash": row["input_hash"],
                    "input_tokens": input_tokens,
                    "output_tokens": int(
                        (token_ids != tokenizer.pad_token_id).sum().item()
                    ),
                    "latency_ms": elapsed_ms / max(1, len(batch) * args.samples_per_prompt),
                    "prompt_build_latency_ms": prepare_ms / max(1, len(batch)),
                    "sample_index": sample_index,
                    "run_id": args.run_id,
                    "run_config_hash": run_config_hash,
                    "contains_think_tag": "<think>" in raw_output or "</think>" in raw_output,
                }
                existing_by_key[(row["id"], sample_index)] = prediction
        processed_since_save += len(batch) * args.samples_per_prompt
        if processed_since_save >= args.save_every or batch_start + len(batch) == len(pending):
            ordered_output = [
                existing_by_key[(row["id"], sample_index)]
                for row in shard_rows
                for sample_index in range(args.samples_per_prompt)
                if (row["id"], sample_index) in existing_by_key
            ]
            write_jsonl(args.output, ordered_output)
            processed_since_save = 0
            print(
                f"{args.run_id} shard={args.shard_index}/{args.shard_count} "
                f"saved={len(ordered_output)}/{len(shard_rows) * args.samples_per_prompt}",
                flush=True,
            )

    final_rows = load_jsonl(args.output)
    expected_count = len(shard_rows) * args.samples_per_prompt
    complete = len(final_rows) == expected_count
    manifest_path = args.manifest_out or args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest = {
        "status": "complete" if complete else "partial",
        "run_config": run_config,
        "run_config_hash": run_config_hash,
        "input": str(args.input),
        "input_sha256": input_sha,
        "catalog": str(args.catalog),
        "catalog_sha256": catalog_sha,
        "prediction_path": str(args.output),
        "prediction_sha256": sha256_file(args.output),
        "expected_rows": expected_count,
        "prediction_rows": len(final_rows),
        "model_load_seconds": load_seconds,
        "inference_seconds_this_invocation": time.perf_counter() - started,
        "thinking_control_supported_by_template": thinking_control_supported,
        "contains_think_rows": sum(row["contains_think_tag"] for row in final_rows),
        "warmup_batches": args.warmup_batches,
        "strict_parse_rate": sum(row["strict_parse_ok"] for row in final_rows) / max(1, len(final_rows)),
        "peak_gpu_memory_gib": (
            torch.cuda.max_memory_allocated() / (1024**3)
            if args.device.startswith("cuda")
            else None
        ),
        "peak_gpu_reserved_gib": (
            torch.cuda.max_memory_reserved() / (1024**3)
            if args.device.startswith("cuda")
            else None
        ),
        "peak_gpu_memory_gib_by_device": (
            {
                str(index): torch.cuda.max_memory_allocated(index) / (1024**3)
                for index in range(torch.cuda.device_count())
            }
            if args.device == "auto"
            else None
        ),
        "environment": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "peft": __import__("peft").__version__ if args.adapter else "",
            "device": args.device,
        },
    }
    write_json(manifest_path, manifest)
    print(f"inference status={manifest['status']} rows={len(final_rows)} -> {args.output}")
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
