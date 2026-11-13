#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run intent inference and metric calculation.

Without --checkpoint, this script runs a catalog-keyword lexical baseline.
With --checkpoint, it expects transformers/peft-compatible local weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import CATALOG_PATH, FINETUNE_DIR, INTENT_DIR, PROJECT_ROOT, load_catalog, load_jsonl, write_json, write_jsonl


SYSTEM_PROMPT = (
    "你是电商导购意图识别模型。判断用户问题的意图类型并提取关键词。"
    "类型只能是“电商类”或“闲聊类”。"
    "电商类包括商品、订单、支付、物流、优惠、会员、购物车、售后、用户偏好、风控和人工客服等购物服务诉求。"
    "闲聊类是与购物决策或电商服务无关的日常、百科、写作、编程、旅行、健康、做菜、清洁等问题。"
    "只输出 JSON：{\"类型\":\"电商类|闲聊类\",\"关键词\":\"...\"}。"
)
KEYWORD_SEPARATOR = re.compile(r"[,，、;；|/\n]+")
KEYWORD_EDGE_PUNCTUATION = " \t\r。.!！?？\"'“”‘’"


def safe_div(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else 0.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classification_metrics(gold: list[str], pred: list[str]) -> dict[str, Any]:
    labels = [label for label in ("电商类", "闲聊类") if label in set(gold) | set(pred)]
    labels.extend(sorted((set(gold) | set(pred)) - set(labels)))
    per_class: dict[str, dict[str, int | float]] = {}
    for label in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, pred) if g == label and p != label)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        per_class[label] = {
            "support": tp + fn,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    total_tp = sum(int(row["tp"]) for row in per_class.values())
    total_fp = sum(int(row["fp"]) for row in per_class.values())
    total_fn = sum(int(row["fn"]) for row in per_class.values())
    micro_precision = safe_div(total_tp, total_tp + total_fp)
    micro_recall = safe_div(total_tp, total_tp + total_fn)
    return {
        "labels": labels,
        "per_class": per_class,
        "macro_precision": safe_div(sum(float(row["precision"]) for row in per_class.values()), len(labels)),
        "macro_recall": safe_div(sum(float(row["recall"]) for row in per_class.values()), len(labels)),
        "macro_f1": safe_div(sum(float(row["f1"]) for row in per_class.values()), len(labels)),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": safe_div(2 * micro_precision * micro_recall, micro_precision + micro_recall),
    }


def split_keyword_labels(value: Any) -> list[str]:
    labels = []
    for part in KEYWORD_SEPARATOR.split(str(value or "")):
        normalized = part.strip(KEYWORD_EDGE_PUNCTUATION)
        if normalized:
            labels.append(normalized)
    return labels


def keyword_quality(
    rows: list[dict[str, Any]],
    preds: list[tuple[str, str]],
    catalog: list[dict[str, Any]],
) -> dict[str, Any]:
    catalog_labels = sorted({str(tool["keyword_label"]) for tool in catalog})
    catalog_label_set = set(catalog_labels)
    cases = []
    missing_keywords: Counter[str] = Counter()
    extra_keywords: Counter[str] = Counter()
    out_of_catalog_keywords: Counter[str] = Counter()
    raw_prediction_count = 0
    duplicate_prediction_count = 0
    for row, (_, predicted_text) in zip(rows, preds):
        if row.get("类型") != "电商类":
            continue
        gold = set(split_keyword_labels(row.get("关键词", "")))
        predicted_list = split_keyword_labels(predicted_text)
        predicted = set(predicted_list)
        raw_prediction_count += len(predicted_list)
        duplicate_prediction_count += len(predicted_list) - len(predicted)
        missing_keywords.update(gold - predicted)
        extra_keywords.update(predicted - gold)
        out_of_catalog_keywords.update(predicted - catalog_label_set)
        cases.append({"gold": gold, "pred": predicted})

    def aggregate(selected: list[dict[str, set[str]]]) -> dict[str, int | float]:
        tp = sum(len(case["gold"] & case["pred"]) for case in selected)
        fp = sum(len(case["pred"] - case["gold"]) for case in selected)
        fn = sum(len(case["gold"] - case["pred"]) for case in selected)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        sample_precision = [safe_div(len(case["gold"] & case["pred"]), len(case["pred"])) for case in selected]
        sample_recall = [safe_div(len(case["gold"] & case["pred"]), len(case["gold"])) for case in selected]
        sample_f1 = [safe_div(2 * p * r, p + r) for p, r in zip(sample_precision, sample_recall)]
        return {
            "count": len(selected),
            "exact_match_accuracy": safe_div(sum(case["gold"] == case["pred"] for case in selected), len(selected)),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "micro_precision": precision,
            "micro_recall": recall,
            "micro_f1": safe_div(2 * precision * recall, precision + recall),
            "sample_macro_precision": safe_div(sum(sample_precision), len(sample_precision)),
            "sample_macro_recall": safe_div(sum(sample_recall), len(sample_recall)),
            "sample_macro_f1": safe_div(sum(sample_f1), len(sample_f1)),
            "empty_prediction_rate": safe_div(sum(not case["pred"] for case in selected), len(selected)),
        }

    per_label = {}
    for label in catalog_labels:
        tp = sum(label in case["gold"] and label in case["pred"] for case in cases)
        fp = sum(label not in case["gold"] and label in case["pred"] for case in cases)
        fn = sum(label in case["gold"] and label not in case["pred"] for case in cases)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        per_label[label] = {
            "support": tp + fn,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": safe_div(2 * precision * recall, precision + recall),
        }

    predicted_instance_count = sum(len(case["pred"]) for case in cases)
    chitchat = [
        split_keyword_labels(predicted_text)
        for row, (_, predicted_text) in zip(rows, preds)
        if row.get("类型") == "闲聊类"
    ]
    result = aggregate(cases)
    result.update(
        {
            "scope": "gold_ecommerce_rows_only",
            "matching": "literal_normalized_label_set",
            "gold_keyword_instances": sum(len(case["gold"]) for case in cases),
            "predicted_keyword_instances_after_dedup": predicted_instance_count,
            "raw_predicted_keyword_instances": raw_prediction_count,
            "duplicate_predicted_keyword_instances": duplicate_prediction_count,
            "catalog_valid_prediction_rate": (
                safe_div(
                    sum(len(case["pred"] & catalog_label_set) for case in cases),
                    predicted_instance_count,
                )
                if predicted_instance_count
                else None
            ),
            "catalog_label_macro_precision": safe_div(
                sum(float(row["precision"]) for row in per_label.values()), len(per_label)
            ),
            "catalog_label_macro_recall": safe_div(
                sum(float(row["recall"]) for row in per_label.values()), len(per_label)
            ),
            "catalog_label_macro_f1": safe_div(
                sum(float(row["f1"]) for row in per_label.values()), len(per_label)
            ),
            "chitchat_empty_keyword_accuracy": safe_div(sum(not labels for labels in chitchat), len(chitchat)),
            "by_gold_keyword_count": {
                str(count): aggregate([case for case in cases if len(case["gold"]) == count])
                for count in sorted({len(case["gold"]) for case in cases})
            },
            "top_missing_keywords": missing_keywords.most_common(15),
            "top_extra_keywords": extra_keywords.most_common(15),
            "top_out_of_catalog_keywords": out_of_catalog_keywords.most_common(15),
            "per_catalog_label": per_label,
        }
    )
    return result


def summary_stats(values: list[float]) -> dict:
    ordered = sorted(values)
    if not ordered:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}

    def percentile(q: float) -> float:
        position = (len(ordered) - 1) * q
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {
        "mean": sum(ordered) / len(ordered),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": ordered[-1],
    }


def cpu_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "not_reported"


def lexical_predict(question: str, catalog: list[dict]) -> tuple[str, str]:
    hits: list[str] = []
    for tool in catalog:
        terms = [tool["keyword_label"], tool["name_zh"]] + list(tool.get("positive_phrases", []))
        if any(term and term in question for term in terms):
            if tool["keyword_label"] not in hits:
                hits.append(tool["keyword_label"])
    ecommerce_words = ["订单", "商品", "购物车", "退款", "优惠", "券", "物流", "发票", "会员", "售后", "库存", "下单", "支付", "配送", "价格"]
    if hits or any(word in question for word in ecommerce_words):
        return "电商类", "，".join(hits[:6])
    return "闲聊类", ""


def parse_json_response(text: str) -> tuple[str, str]:
    obj = {}
    try:
        obj = json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            try:
                obj = json.loads(match.group(0))
            except Exception:
                obj = {}
    intent_type = obj.get("类型", "") if isinstance(obj, dict) else ""
    keywords = obj.get("关键词", "") if isinstance(obj, dict) else ""
    if not intent_type:
        type_match = re.search(r"(电商类|闲聊类)", text)
        intent_type = type_match.group(1) if type_match else ""
    if not keywords:
        kw_match = re.search(r"关键词[\"'”]?\s*[:：]\s*[\"“]?([^\"”'}\n]+)", text)
        keywords = kw_match.group(1).strip(" ，。") if kw_match else ""
    if intent_type not in {"电商类", "闲聊类"}:
        intent_type = "电商类" if keywords else "闲聊类"
    return intent_type, str(keywords)


def resolve_base_model(checkpoint: str, base_model: str = "") -> str:
    if base_model:
        return base_model
    adapter_config = Path(checkpoint) / "adapter_config.json"
    if adapter_config.exists():
        obj = json.loads(adapter_config.read_text(encoding="utf-8"))
        candidate = obj.get("base_model_name_or_path")
        if candidate:
            return str(candidate)
    lower = checkpoint.lower()
    if "1.7" in lower:
        return str(PROJECT_ROOT / "models" / "Qwen" / "Qwen3-1.7B")
    return str(PROJECT_ROOT / "models" / "Qwen" / "Qwen3-0.6B")


def model_predict(
    rows: list[dict],
    checkpoint: str,
    base_model: str,
    batch_size: int,
    max_new_tokens: int,
    partial_out: Path | None = None,
    run_name: str = "",
    save_every: int = 500,
    warmup_batches: int = 0,
) -> tuple[list[tuple[str, str]], dict]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        import torch
    except ImportError as exc:
        raise SystemExit("transformers and peft are required for checkpoint inference") from exc

    load_started = time.perf_counter()
    base_path = resolve_base_model(checkpoint, base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(base_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
    if (Path(checkpoint) / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(model, checkpoint)
    elif Path(checkpoint).resolve() != Path(base_path).resolve():
        model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - load_started

    def prepare_batch(batch: list[dict]):
        prompts = []
        for row in batch:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": row["问题"]}]
            try:
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)
        return tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

    for warmup_index in range(min(warmup_batches, math.ceil(len(rows) / batch_size))):
        warmup = rows[warmup_index * batch_size : (warmup_index + 1) * batch_size]
        warmup_inputs = prepare_batch(warmup)
        with torch.no_grad():
            model.generate(
                **warmup_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    outputs = []
    next_save = save_every
    per_query_generation_ms: list[float] = []
    per_query_total_ms: list[float] = []
    total_generation_seconds = 0.0
    total_batch_seconds = 0.0
    total_output_tokens = 0
    total_input_tokens = 0
    measured_started = time.perf_counter()
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        batch_started = time.perf_counter()
        inputs = prepare_batch(batch)
        total_input_tokens += int(inputs["attention_mask"].sum().item())
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        generation_started = time.perf_counter()
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        generation_seconds = time.perf_counter() - generation_started
        total_generation_seconds += generation_seconds
        input_len = inputs["input_ids"].shape[1]
        for item in generated:
            new_token_ids = item[input_len:]
            text = tokenizer.decode(new_token_ids, skip_special_tokens=True)
            outputs.append(parse_json_response(text))
            output_token_count = 0
            for token_id in new_token_ids.tolist():
                output_token_count += 1
                if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
                    break
            total_output_tokens += output_token_count
        batch_total_seconds = time.perf_counter() - batch_started
        total_batch_seconds += batch_total_seconds
        per_query_generation_ms.extend([generation_seconds * 1000.0 / len(batch)] * len(batch))
        per_query_total_ms.extend([batch_total_seconds * 1000.0 / len(batch)] * len(batch))
        if partial_out and (len(outputs) >= next_save or len(outputs) == len(rows)):
            partial_rows = []
            for row, (intent_type, keywords) in zip(rows[: len(outputs)], outputs):
                partial_rows.append({**row, "预测类型": intent_type, "预测关键词": keywords, "run_name": run_name})
            write_jsonl(partial_out, partial_rows)
            while next_save <= len(outputs):
                next_save += save_every
        if len(outputs) % max(batch_size * 10, 100) == 0 or len(outputs) == len(rows):
            print(f"inferred {len(outputs)}/{len(rows)}")
    loop_wall_seconds = time.perf_counter() - measured_started
    gpu = {}
    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        gpu = {
            "name": torch.cuda.get_device_name(device_index),
            "logical_device_index": device_index,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device_index) / (1024**3),
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device_index) / (1024**3),
        }
    efficiency = {
        "protocol": "batched_greedy_generation",
        "samples": len(rows),
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "warmup_batches": warmup_batches,
        "model_load_seconds": model_load_seconds,
        "measured_seconds": total_batch_seconds,
        "loop_wall_seconds_including_checkpoints": loop_wall_seconds,
        "generation_seconds": total_generation_seconds,
        "generation_latency_ms_per_query": summary_stats(per_query_generation_ms),
        "total_latency_ms_per_query": summary_stats(per_query_total_ms),
        "generation_throughput_samples_per_second": (
            len(rows) / total_generation_seconds if total_generation_seconds else 0.0
        ),
        "end_to_end_throughput_samples_per_second": (
            len(rows) / total_batch_seconds if total_batch_seconds else 0.0
        ),
        "output_throughput_tokens_per_second": (
            total_output_tokens / total_generation_seconds if total_generation_seconds else 0.0
        ),
        "mean_input_tokens": total_input_tokens / len(rows) if rows else 0.0,
        "mean_output_tokens": total_output_tokens / len(rows) if rows else 0.0,
        "precision": "bfloat16",
        "gpu": gpu,
    }
    return outputs, efficiency


def lexical_predict_timed(rows: list[dict], catalog: list[dict]) -> tuple[list[tuple[str, str]], dict]:
    per_query_ms: list[float] = []
    predictions: list[tuple[str, str]] = []
    started = time.perf_counter()
    for row in rows:
        query_started = time.perf_counter()
        predictions.append(lexical_predict(row["问题"], catalog))
        per_query_ms.append((time.perf_counter() - query_started) * 1000.0)
    elapsed_seconds = time.perf_counter() - started
    return predictions, {
        "protocol": "single_thread_cpu_lexical",
        "samples": len(rows),
        "measured_seconds": elapsed_seconds,
        "total_latency_ms_per_query": summary_stats(per_query_ms),
        "end_to_end_throughput_samples_per_second": len(rows) / elapsed_seconds if elapsed_seconds else 0.0,
        "cpu": cpu_name(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INTENT_DIR / "intent_all.jsonl")
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--base-model", default="")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--warmup-batches", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--run-name", default="keyword_baseline")
    parser.add_argument("--out", type=Path, default=FINETUNE_DIR / "outputs" / "preds" / "keyword_baseline.jsonl")
    parser.add_argument("--metrics-out", type=Path, default=FINETUNE_DIR / "outputs" / "preds" / "keyword_baseline_metrics.json")
    parser.add_argument("--predictions-in", type=Path, help="只离线重算已有预测，不重新推理")
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    if args.limit > 0:
        rows = rows[: args.limit]
    catalog = load_catalog(args.catalog)
    prediction_path = args.out
    inference_efficiency = None
    if args.predictions_in:
        prediction_rows = load_jsonl(args.predictions_in)
        if args.limit > 0:
            prediction_rows = prediction_rows[: args.limit]
        if len(prediction_rows) != len(rows):
            raise SystemExit(
                f"prediction/input count mismatch: {len(prediction_rows)} != {len(rows)}"
            )
        for index, (row, prediction) in enumerate(zip(rows, prediction_rows)):
            if (
                prediction.get("id") != row.get("id")
                or prediction.get("问题") != row.get("问题")
                or prediction.get("类型") != row.get("类型")
                or prediction.get("关键词") != row.get("关键词")
            ):
                raise SystemExit(f"prediction/input alignment mismatch at row {index}")
        preds = [
            (str(row.get("预测类型", "")), str(row.get("预测关键词", "")))
            for row in prediction_rows
        ]
        prediction_path = args.predictions_in
    elif args.checkpoint:
        preds, inference_efficiency = model_predict(
            rows,
            args.checkpoint,
            args.base_model,
            args.batch_size,
            args.max_new_tokens,
            args.out,
            args.run_name,
            args.save_every,
            args.warmup_batches,
        )
    else:
        preds, inference_efficiency = lexical_predict_timed(rows, catalog)
    if not args.predictions_in:
        out_rows = []
        for row, (intent_type, keywords) in zip(rows, preds):
            out_rows.append({**row, "预测类型": intent_type, "预测关键词": keywords, "run_name": args.run_name})
        write_jsonl(args.out, out_rows)
    gold = [row.get("类型", "") for row in rows]
    pred = [p[0] for p in preds]
    acc = sum(1 for g, p in zip(gold, pred) if g == p) / len(gold) if gold else 0.0
    classification = classification_metrics(gold, pred)
    keyword = keyword_quality(rows, preds, catalog)
    existing_metrics = {}
    if args.predictions_in and args.metrics_out.exists():
        existing_metrics = json.loads(args.metrics_out.read_text(encoding="utf-8"))
    metrics = {
        **existing_metrics,
        "metric_schema_version": 2,
        "run_name": args.run_name,
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "catalog": str(args.catalog),
        "catalog_sha256": sha256_file(args.catalog),
        "predictions": str(prediction_path),
        "predictions_sha256": sha256_file(prediction_path),
        "total": len(rows),
        "accuracy": acc,
        "macro_precision": classification["macro_precision"],
        "macro_recall": classification["macro_recall"],
        "macro_f1": classification["macro_f1"],
        "classification": classification,
        "by_gold": dict(Counter(gold)),
        "by_pred": dict(Counter(pred)),
        "ecommerce_empty_keyword_rate": keyword["empty_prediction_rate"],
        "keyword_quality": keyword,
        "checkpoint": args.checkpoint or existing_metrics.get("checkpoint", ""),
        "base_model": args.base_model or existing_metrics.get("base_model", ""),
    }
    if inference_efficiency is not None:
        metrics["inference_efficiency"] = inference_efficiency
    write_json(args.metrics_out, metrics)
    print(
        f"intent metrics accuracy={acc:.4f} macro_f1={metrics['macro_f1']:.4f} "
        f"keyword_micro_f1={keyword['micro_f1']:.4f} -> {args.metrics_out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
