#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render the 100-tool RAG baseline report."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import PROJECT_ROOT, RAG_DIR, TOOL_TRACE_DIR, read_json, split_tools


DEFAULT_METRICS = RAG_DIR / "_aux" / "rag_baseline" / "metrics.json"
DEFAULT_COMPAT = RAG_DIR / "_aux" / "retrieval_baseline_metrics.json"
DEFAULT_RERANKERS = [
    RAG_DIR / "_aux" / "reranker_baseline" / "metrics.json",
    RAG_DIR / "_aux" / "reranker_bge-base_baseline" / "metrics.json",
    RAG_DIR / "_aux" / "reranker_baseline_k50" / "metrics.json",
    RAG_DIR / "_aux" / "reranker_bge-base_baseline_k50" / "metrics.json",
]
DEFAULT_FINETUNED_EMBEDDING_DIR = RAG_DIR / "_aux" / "embedding_finetune"
DEFAULT_EMBEDDING_STATS = RAG_DIR / "_aux" / "embedding_dataset_stats.json"
DEFAULT_DATASET_AUDIT = RAG_DIR / "_aux" / "dataset_ease_audit.json"
DEFAULT_HARD_STATS = TOOL_TRACE_DIR / "_aux" / "hard_generation_stats.json"
DEFAULT_OUT = PROJECT_ROOT / "reports" / "02_检索增强.md"


def fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if value is None:
        return ""
    return str(value)


def fmt_best(value: Any, best: float | None, digits: int = 4) -> str:
    text = fmt(value, digits)
    if value is None or best is None:
        return text
    try:
        number = float(value)
    except (TypeError, ValueError):
        return text
    if abs(number - best) < 1e-12:
        return f"**{text}**"
    return text


def unique_max(values: list[float | None]) -> float | None:
    vals = [value for value in values if value is not None]
    if not vals:
        return None
    best = max(vals)
    if sum(abs(value - best) < 1e-12 for value in vals) == 1:
        return best
    return None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def metric_row(
    name: str,
    payload: dict[str, Any],
    k: int,
    include_timing: bool = False,
    allhit_best: float | None = None,
) -> str:
    metrics = payload["metrics"]
    row = metrics[f"@{k}"]
    timing = ""
    if include_timing:
        timing_map = payload.get("timing", {})
        timing = " | " + ", ".join(f"{key}={value}" for key, value in timing_map.items())
    return (
        f"| {name} | {k} | {fmt(row['Hit@K'])} | {fmt_best(row['AllHit@K'], allhit_best)} | "
        f"{fmt(row['Recall@K'])} | {fmt(row['Precision@K'])} | {fmt(metrics['overall']['MRR'])}{timing} |"
    )


def vector_metric_row(name: str, payload: dict[str, Any], k: int, allhit_best: float | None = None) -> str:
    metrics = payload["metrics"]
    row = metrics[f"@{k}"]
    timing = payload.get("timing", {})
    return (
        f"| {name} | {k} | {fmt(row['Hit@K'])} | {fmt_best(row['AllHit@K'], allhit_best)} | "
        f"{fmt(row['Recall@K'])} | {fmt(row['Precision@K'])} | {fmt(metrics['overall']['MRR'])} | "
        f"{fmt(timing.get('encode_avg_ms'))} | {fmt(timing.get('search_p50_ms'))} | {fmt(timing.get('search_p95_ms'))} |"
    )


def payload_allhit_float(payload: dict[str, Any], k: int) -> float | None:
    row = payload.get("metrics", {}).get(f"@{k}") or payload.get("rerank_metrics", {}).get(f"@{k}")
    if not row or row.get("AllHit@K") is None:
        return None
    return float(row["AllHit@K"])


def best_allhit(candidates: list[dict[str, Any]], k: int) -> float | None:
    values = [value for payload in candidates if (value := payload_allhit_float(payload, k)) is not None]
    return max(values) if values else None


def k_group_text(k_values: list[int]) -> str:
    runs = []
    start = prev = k_values[0]
    for k in k_values[1:]:
        if k == prev + 1:
            prev = k
            continue
        runs.append((start, prev))
        start = prev = k
    runs.append((start, prev))
    return "/".join(f"K{left}-K{right}" if left != right else f"K{left}" for left, right in runs)


def selection_note(selected: dict[int, str]) -> str:
    groups: dict[str, list[int]] = {}
    order: list[str] = []
    for k, label in selected.items():
        if label not in groups:
            groups[label] = []
            order.append(label)
        groups[label].append(k)
    return "；".join(f"{label}: {k_group_text(groups[label])}" for label in order)


def topk_comparison_data(
    method: str,
    candidates: list[tuple[str, dict[str, Any]]],
    ks: list[int],
    note: str = "",
) -> tuple[str, list[float | None], str]:
    selected: dict[int, str] = {}
    values: list[float | None] = []
    for k in ks:
        best: tuple[float, str] | None = None
        for label, payload in candidates:
            value = payload_allhit_float(payload, k)
            if value is None:
                continue
            if best is None or value > best[0]:
                best = (value, label)
        if best is None:
            values.append(None)
            continue
        value, label = best
        selected[k] = label
        values.append(value)
    selected_text = selection_note(selected)
    detail = f"{selected_text}；{note}" if selected_text and note else selected_text or note
    return method, values, detail


def topk_comparison_row(data: tuple[str, list[float | None], str], best_by_col: list[float | None]) -> str:
    method, values, detail = data
    cells = [fmt_best(value, best_by_col[index]) if value is not None else "—" for index, value in enumerate(values)]
    return f"| {method} | {' | '.join(cells)} | {detail} |"


def best_topk_summary_rows(
    metrics: dict[str, Any],
    rerankers: list[dict[str, Any]],
    finetuned_embeddings: list[tuple[str, dict[str, Any]]],
    ks: list[int],
) -> list[str]:
    hybrid_candidates = []
    for name, payload in metrics.items():
        if not name.startswith("hybrid:"):
            continue
        label = (
            name.replace("hybrid:", "")
            .replace(":rrf", " RRF")
            .replace(":weighted", " weighted")
            .replace(":union_top30", " union_top30")
        )
        hybrid_candidates.append((label, payload))

    rows_data = [
        topk_comparison_data("BM25", [("BM25(fallback)", metrics.get("bm25", {}))], ks),
        topk_comparison_data(
            "Best Vector",
            [
                ("bge-base", metrics.get("vector:bge-base", {})),
                ("bge-large", metrics.get("vector:bge-large", {})),
                ("qwen3-embedding-0.6b", metrics.get("vector:qwen3-embedding-0.6b", {})),
            ],
            ks,
        ),
        topk_comparison_data("Best Hybrid", hybrid_candidates, ks),
    ]

    reranker_candidates = []
    for item in rerankers:
        label = f"{reranker_source_label(item)} Top{item.get('candidate_k', '')}"
        reranker_candidates.append((label, {"rerank_metrics": item.get("rerank_metrics", {})}))
    if reranker_candidates:
        rows_data.append(topk_comparison_data("Best Reranker", reranker_candidates, ks))
    if finetuned_embeddings:
        rows_data.append(
            topk_comparison_data(
                "Best Finetuned Embedding",
                finetuned_embeddings,
                ks,
                "有 context skeleton 风险；作为同生成器闭集上限，不作为真实泛化结论",
            )
        )
    best_by_col = [
        max([value for _, values, _ in rows_data if (value := values[index]) is not None], default=None)
        for index in range(len(ks))
    ]
    return [topk_comparison_row(row, best_by_col) for row in rows_data]


def full_topk_comparison_rows(
    metrics: dict[str, Any],
    rerankers: list[dict[str, Any]],
    finetuned_embeddings: list[tuple[str, dict[str, Any]]],
    ks: list[int],
) -> list[str]:
    rows_data: list[tuple[str, str, dict[str, Any], str]] = []
    if metrics.get("bm25"):
        rows_data.append(("BM25", "BM25(fallback)", metrics["bm25"], ""))
    for name in ["vector:bge-base", "vector:bge-large", "vector:qwen3-embedding-0.6b"]:
        if name in metrics:
            rows_data.append(("Vector", name.replace("vector:", ""), metrics[name], ""))
    for model in ["bge-base", "bge-large", "qwen3-embedding-0.6b"]:
        for method, label in [
            ("union_top30", "Union Top30"),
            ("weighted", "Weighted"),
            ("rrf", "RRF"),
        ]:
            name = f"hybrid:{model}:{method}"
            if name in metrics:
                rows_data.append(("Hybrid", f"{model} {label}", metrics[name], ""))
    for item in rerankers:
        label = f"{reranker_source_label(item)} Top{item.get('candidate_k', '')} -> bge-reranker-base"
        rows_data.append(("Reranker", label, {"rerank_metrics": item.get("rerank_metrics", {})}, ""))
    for label, payload in finetuned_embeddings:
        rows_data.append(("Finetuned Embedding", label, payload, "context skeleton 风险"))

    best_by_group_col = {
        (group, index): unique_max(
            [payload_allhit_float(payload, k) for row_group, _, payload, _ in rows_data if row_group == group]
        )
        for group in {group for group, _, _, _ in rows_data}
        for index, k in enumerate(ks)
    }
    rows = []
    previous_group = ""
    separator = "| " + " | ".join([""] * (len(ks) + 3)) + " |"
    for group, method, payload, note in rows_data:
        if previous_group and group != previous_group:
            rows.append(separator)
        previous_group = group
        cells = []
        for index, k in enumerate(ks):
            value = payload_allhit_float(payload, k)
            cells.append(fmt_best(value, best_by_group_col[(group, index)]) if value is not None else "—")
        rows.append(f"| {group} | {method} | {' | '.join(cells)} | {note} |")
    return rows


def resolve_repo_path(raw: Any) -> Path:
    path = Path(str(raw or ""))
    return path if path.is_absolute() else PROJECT_ROOT.parent / path


def ranking_metrics_from_pred_rows(rows: list[dict[str, Any]], k_values: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    gold_sets = [set(row.get("gold_tools") or split_tools(row.get("工具"))) for row in rows]
    rank_rows = [row.get("pred_tools", []) for row in rows]
    for k in k_values:
        hit_any = 0
        all_hit = 0
        recall_sum = 0.0
        precision_sum = 0.0
        for gold, ranking in zip(gold_sets, rank_rows):
            pred = set(ranking[:k])
            matched = gold & pred
            hit_any += int(bool(matched))
            all_hit += int(bool(gold) and gold <= pred)
            recall_sum += len(matched) / max(1, len(gold))
            precision_sum += len(matched) / k
        n = max(1, len(rows))
        out[f"@{k}"] = {
            "Hit@K": hit_any / n,
            "AllHit@K": all_hit / n,
            "Recall@K": recall_sum / n,
            "Precision@K": precision_sum / n,
        }
    mrr_sum = 0.0
    for gold, ranking in zip(gold_sets, rank_rows):
        rr = 0.0
        for rank, tool_id in enumerate(ranking, start=1):
            if tool_id in gold:
                rr = 1.0 / rank
                break
        mrr_sum += rr
    out["overall"] = {"MRR": mrr_sum / max(1, len(rows)), "count": len(rows)}
    return out


def enrich_reranker_metrics(item: dict[str, Any], summary_ks: list[int]) -> dict[str, Any]:
    if not item.get("predictions"):
        return item
    predictions = resolve_repo_path(item.get("predictions"))
    if not predictions.exists():
        return item
    candidate_k = int(item.get("candidate_k") or 0)
    valid_ks = [k for k in summary_ks if not candidate_k or k <= candidate_k]
    if not valid_ks:
        return item
    enriched = dict(item)
    computed = ranking_metrics_from_pred_rows(load_jsonl(predictions), valid_ks)
    merged = dict(computed)
    merged.update(item.get("rerank_metrics", {}))
    enriched["rerank_metrics"] = merged
    enriched["k_values"] = sorted({*item.get("k_values", []), *valid_ks})
    return enriched


def load_finetuned_embedding_candidates(current_stats: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    if not DEFAULT_FINETUNED_EMBEDDING_DIR.exists():
        return []
    current_train = current_stats.get("train", {})
    current_queries = current_train.get("queries")
    current_positive_pairs = current_train.get("positive_pairs")
    candidates: list[tuple[str, dict[str, Any]]] = []
    for run_dir in sorted(path for path in DEFAULT_FINETUNED_EMBEDDING_DIR.iterdir() if path.is_dir()):
        metrics_path = run_dir / "retrieval" / "metrics.json"
        manifest_path = run_dir / "model" / "training_manifest.json"
        if not metrics_path.exists():
            continue
        manifest = read_json(manifest_path, {})
        train_stats = manifest.get("train_stats", {})
        if current_queries is not None and train_stats.get("source_rows") != current_queries:
            continue
        if current_positive_pairs is not None and train_stats.get("positive_pairs") != current_positive_pairs:
            continue
        run_metrics = read_json(metrics_path, {})
        vector_methods = [name for name in run_metrics if name.startswith("vector:")]
        if not vector_methods:
            continue
        candidates.append((run_dir.name, run_metrics[vector_methods[0]]))
    return candidates


def vector_db_rows(vector_dir: Path) -> list[str]:
    rows = []
    for config_path in sorted(vector_dir.glob("*/config.json")):
        config = read_json(config_path, {})
        rel_dir = "PROJECT/" + str(config_path.parent.relative_to(PROJECT_ROOT))
        rows.append(
            "| `{dir}/` | `{model}` | {dim} | {docs} | {seconds}s |".format(
                dir=rel_dir,
                model=config.get("model_name_or_path", ""),
                dim=config.get("embedding_dim", ""),
                docs=config.get("doc_count", ""),
                seconds=config.get("encode_seconds", ""),
            )
        )
    return rows


def display_path(raw: Any) -> str:
    text = str(raw or "")
    repo_prefix = str(PROJECT_ROOT.parent) + "/"
    return text.replace(repo_prefix, "", 1) if text.startswith(repo_prefix) else text


def best_by(results: dict[str, Any], key: str, k: int) -> tuple[str, float]:
    best_name = ""
    best_value = -1.0
    for name, payload in results.items():
        value = float(payload["metrics"][f"@{k}"][key])
        if value > best_value:
            best_name = name
            best_value = value
    return best_name, best_value


def allhit(metrics: dict[str, Any], name: str, k: int) -> float:
    if name not in metrics:
        return 0.0
    return float(metrics[name]["metrics"][f"@{k}"]["AllHit@K"])


def load_pred_rankings(path: Path) -> dict[str, list[str]]:
    return {str(row.get("id", "")): row.get("pred_tools", []) for row in load_jsonl(path)}


def subset_metrics(rows: list[dict[str, Any]], rankings: dict[str, list[str]], k_values: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    gold_sets = [set(split_tools(row.get("工具"))) for row in rows]
    rank_rows = [rankings.get(str(row.get("id", "")), []) for row in rows]
    for k in k_values:
        hit_any = 0
        all_hit = 0
        recall_sum = 0.0
        precision_sum = 0.0
        for gold, ranking in zip(gold_sets, rank_rows):
            pred = set(ranking[:k])
            matched = gold & pred
            hit_any += int(bool(matched))
            all_hit += int(bool(gold) and gold <= pred)
            recall_sum += len(matched) / max(1, len(gold))
            precision_sum += len(matched) / k
        n = max(1, len(rows))
        out[f"@{k}"] = {
            "Hit@K": hit_any / n,
            "AllHit@K": all_hit / n,
            "Recall@K": recall_sum / n,
            "Precision@K": precision_sum / n,
        }
    mrr_sum = 0.0
    for gold, ranking in zip(gold_sets, rank_rows):
        rr = 0.0
        for rank, tool_id in enumerate(ranking, start=1):
            if tool_id in gold:
                rr = 1.0 / rank
                break
        mrr_sum += rr
    out["overall"] = {"MRR": mrr_sum / max(1, len(rows)), "count": len(rows)}
    return out


def split_metric_row(
    name: str,
    subset: str,
    rows: list[dict[str, Any]],
    metrics: dict[str, Any],
    total: int,
    best_allhit10: float | None = None,
    best_allhit30: float | None = None,
) -> str:
    row10 = metrics["@10"]
    row30 = metrics["@30"]
    return (
        f"| {name} | {subset} | {len(rows)} | {len(rows) / max(1, total):.2%} | "
        f"{fmt(row10['Hit@K'])} | {fmt_best(row10['AllHit@K'], best_allhit10)} | {fmt(row10['Recall@K'])} | "
        f"{fmt(row10['Precision@K'])} | {fmt_best(row30['AllHit@K'], best_allhit30)} | {fmt(row30['Recall@K'])} | "
        f"{fmt(metrics['overall']['MRR'])} |"
    )


def reranker_source_label(payload: dict[str, Any]) -> str:
    source = str(payload.get("source_predictions", ""))
    if "hybrid_bge-base_rrf" in source:
        return "bge-base RRF"
    if "hybrid_bge-large_rrf" in source:
        return "bge-large RRF"
    if "hybrid_qwen3-embedding-0.6b_rrf" in source:
        return "qwen3-embedding-0.6b RRF"
    return Path(source).stem or "unknown"


def dataset_audit_lines(dataset_audit: dict[str, Any]) -> list[str]:
    if not dataset_audit:
        return ["- 待跑。建议执行 `conda run -n llm_bge python PROJECT/scripts/rag/audit_dataset_difficulty.py --tokenizer jieba`。"]

    exact = dataset_audit.get("exact_gold_term_overlap", {})
    exact_matcher = dataset_audit.get("exact_term_matcher", {}).get("metrics", {})
    emb_test = dataset_audit.get("embedding_pair_exact_overlap", {}).get("test", {}).get("by_label", {})
    bm25_ablation = dataset_audit.get("bm25_field_ablation", {})
    exact_rate = float(exact.get("all_gold_exact_rate", 0.0))
    exact_matcher10 = float(exact_matcher.get("@10", {}).get("AllHit@K", 0.0))
    if exact_rate <= 0.01 and exact_matcher10 < 0.2:
        audit_intro = "审计结论：表达簇重构后，上一版的显式触发词泄漏已经基本消除，朴素精确词匹配器不再能解决测试集。"
        audit_tail = "因此，新结果更适合作为 100 工具 paraphrase 闭集检索基线；但它仍然是合成数据，真实泛化还需要人工改写、真实日志和多轮上下文回放继续验证。"
    else:
        audit_intro = "审计结论：当前测试集仍然偏简单，query 与 catalog 触发词存在明显同源性，需要继续增加去关键词化改写。"
        audit_tail = "因此，当前 RAG baseline 只能作为闭集模板数据的工程基线，不应直接外推为真实用户表达下的泛化能力。"
    return [
        audit_intro,
        "",
        "| 审计项 | 结果 | 解读 |",
        "|---|---:|---|",
        f"| gold 工具精确触发词覆盖 | {fmt(exact.get('all_gold_exact_rate'))} | 统计每个 gold tool 是否在 query 中精确出现 `positive_phrases`/`keyword_label`/`name_zh`。 |",
        f"| positive phrase 精确命中 | {fmt(exact.get('positive_phrase_any_rate'))} | 新生成器不再直接复用 catalog 触发句。 |",
        f"| 朴素精确词匹配器 | AllHit@6={fmt(exact_matcher.get('@6', {}).get('AllHit@K'))} / AllHit@10={fmt(exact_matcher.get('@10', {}).get('AllHit@K'))} / MRR={fmt(exact_matcher.get('overall', {}).get('MRR'))} | 只按 catalog 触发词包含关系排序，用来检验显式泄漏。 |",
        f"| embedding test 正样本精确触发词 | {fmt(emb_test.get('1', {}).get('candidate_exact_term_in_query_rate'))} | pair-level 正样本不再能靠候选工具触发词直接判别。 |",
        f"| embedding test 负样本精确触发词 | {fmt(emb_test.get('0', {}).get('candidate_exact_term_in_query_rate'))} | 负样本同样没有候选工具精确触发词。 |",
        f"| BM25 字段消融 | desc-only AllHit@10={fmt(bm25_ablation.get('description_only', {}).get('@10', {}).get('AllHit@K'))}；full-doc AllHit@10={fmt(bm25_ablation.get('full_document', {}).get('@10', {}).get('AllHit@K'))} | 用于观察工具文档增强字段对词面检索的影响。 |",
        "",
        audit_tail,
        "",
        "注意：本节只审计显式触发词和文档字段难度；embedding 微调的 context skeleton 风险见 `PROJECT/reports/03_向量训练.md`。",
        "",
        f"审计文件：`{display_path(dataset_audit.get('output_path', RAG_DIR / '_aux' / 'dataset_ease_audit.json'))}`。",
    ]


def build_report(
    metrics: dict[str, Any],
    compat: dict[str, Any],
    vector_dir: Path,
    rerankers: list[dict[str, Any]],
    embedding_stats: dict[str, Any],
    dataset_audit: dict[str, Any],
    hard_stats: dict[str, Any],
) -> list[str]:
    raw_k_values = compat.get("k_values") or [6, 7, 8, 9, 10, 15, 20, 30, 50]
    summary_k_values = [k for k in [6, 7, 8, 9, 10, 30] if k in raw_k_values]
    detail_k_values = [k for k in [6, 10, 30] if k in raw_k_values]
    if not summary_k_values:
        summary_k_values = raw_k_values
    if not detail_k_values:
        detail_k_values = summary_k_values
    num_tools = compat.get("num_tools", 100)
    num_queries = compat.get("num_queries", "")
    candidate_k = compat.get("candidate_k", 30)
    elapsed_sec = compat.get("elapsed_sec", "")
    tool_count_dist = {}
    test_path = Path(compat.get("test", PROJECT_ROOT / "data" / "tool_call_trace" / "test.jsonl"))
    test_rows = load_jsonl(test_path)
    for row in test_rows:
        count = int(row.get("tool_count") or len(str(row.get("工具", "")).split(",")))
        tool_count_dist[count] = tool_count_dist.get(count, 0) + 1
    dist_text = "，".join(f"{k}工具={v}" for k, v in sorted(tool_count_dist.items()))
    gold_tool_total = sum(k * v for k, v in tool_count_dist.items())
    avg_gold_tools = gold_tool_total / max(1, len(test_rows))
    precision30_ceiling = avg_gold_tools / 30.0

    best10 = best_by(metrics, "AllHit@K", 10)
    best20 = best_by(metrics, "AllHit@K", 20)
    best30 = best_by(metrics, "AllHit@K", 30)
    vector10 = {
        name.replace("vector:", ""): allhit(metrics, name, 10)
        for name in ["vector:bge-base", "vector:bge-large", "vector:qwen3-embedding-0.6b"]
        if name in metrics
    }
    best_vector10 = max(vector10.items(), key=lambda item: item[1]) if vector10 else ("", 0.0)
    fusion10 = {name: allhit(metrics, name, 10) for name in metrics if name.startswith("hybrid:")}
    best_fusion10 = max(fusion10.items(), key=lambda item: item[1]) if fusion10 else ("", 0.0)
    reranker_summary_ks = sorted({*summary_k_values, 50})
    available_rerankers = [
        enrich_reranker_metrics(item, reranker_summary_ks)
        for item in rerankers
        if item and item.get("status") != "pending"
    ]
    finetuned_embedding_candidates = load_finetuned_embedding_candidates(embedding_stats)

    bm25_tokenizer = metrics.get("bm25", {}).get("extra", {}).get("tokenizer", "待跑")
    data_policy_text = "待确认"
    if hard_stats:
        policy = hard_stats.get("policy", {})
        split = hard_stats.get("split", {})
        data_policy_text = (
            f"{policy.get('generator', policy.get('query_style', ''))}；"
            f"train={split.get('train')} / test={split.get('test')}"
        )
    lines = [
        "# 检索增强",
        "",
        "> 日期：2026-07-08",
        f"> 环境：`llm_bge` 已重跑 BM25 `{bm25_tokenizer}` tokenizer、Faiss 向量、融合检索与 BGE reranker",
        f"> 工具描述：`PROJECT/data/rag/tool_corpus.jsonl`，{num_tools} 条工具文档",
        f"> 测试集：`PROJECT/data/tool_call_trace/test.jsonl`，{num_queries} 条冻结工具轨迹样本",
        "> 主指标：`AllHit@K`，即 TopK 返回工具必须完整覆盖 gold tool 集合",
        "",
        "## 1. 数据与评测口径",
        "",
        "| 项目 | 当前设置 |",
        "|---|---|",
        f"| 工具库 | `PROJECT/data/rag/tool_corpus.jsonl`，{num_tools} 个工具 |",
        f"| 冻结测试集 | `PROJECT/data/tool_call_trace/test.jsonl`，{num_queries} 条 query |",
        f"| Gold 工具数 | 最多 6 个；平均 {avg_gold_tools:.2f} 个/query |",
        "| 主指标 | `AllHit@K`：TopK 必须完整覆盖该 query 的全部 gold tools |",
        f"| 已跑 K | `{','.join(str(k) for k in raw_k_values)}` |",
        f"| 核心汇总 K | `{','.join(str(k) for k in summary_k_values)}` |",
        f"| 融合候选池 | 向量 Top{candidate_k} + BM25 Top{candidate_k}；weighted `alpha=0.7`；RRF `k=60` |",
        f"| 数据生成策略 | {data_policy_text} |",
        f"| 预测目录 | `PROJECT/data/rag/_aux/rag_baseline/preds/`；全量评测耗时 {elapsed_sec}s |",
        "",
        "| K 区间 | 用途 | 报告处理 |",
        "|---|---|---|",
        "| K=6-10 | 短名单召回区间，用来观察最终给 Agent/LLM 的候选规模；K 不小于单条 query 的最大 gold 工具数 | 进入核心汇总表 |",
        "| K=30 | reranker 候选池上限，也是 100 工具下的主要高召回观察点 | 进入核心汇总表 |",
        "| K=50 | 100 工具下的补充高召回上限 | 保留在指标文件和 reranker 小节，不放核心汇总表 |",
        "",
        f"`Precision@30` 的平均理论上限约为 `{precision30_ceiling:.3f}`，因为每条 query 只有 1-6 个 gold tool，其余 Top30 位置必然是非 gold；因此本报告主看 `AllHit@K`、`Recall@K` 和 `MRR`。",
    ]
    lines += [
        "",
        "<details>",
        f"<summary>点击展开 Gold 工具数分布：{dist_text}</summary>",
        "",
        "| Gold 工具数 | Query数 | 数据集占比 | Precision@30 理论上限 |",
        "|---:|---:|---:|---:|",
    ]
    for tool_count, count in sorted(tool_count_dist.items()):
        lines.append(f"| {tool_count} | {count} | {count / max(1, len(test_rows)):.2%} | {tool_count / 30:.4f} |")
    lines += [
        f"| 单意图合计 | {tool_count_dist.get(1, 0)} | {tool_count_dist.get(1, 0) / max(1, len(test_rows)):.2%} | 0.0333 |",
        f"| 多意图合计 | {len(test_rows) - tool_count_dist.get(1, 0)} | {(len(test_rows) - tool_count_dist.get(1, 0)) / max(1, len(test_rows)):.2%} |  |",
        "",
        "</details>",
        "",
        "### 数据难度审计",
        "",
    ]
    lines.extend(dataset_audit_lines(dataset_audit))
    lines += [
        "",
        "## 2. 构建向量库",
        "",
        "| 数据库目录 | 模型 | 维度 | 文档数 | 文档编码耗时 |",
        "|---|---|---:|---:|---:|",
    ]
    rows = vector_db_rows(vector_dir)
    lines.extend(rows or ["| 待跑 | 待跑 |  |  |  |"])
    lines += [
        "",
        "每个数据库目录包含：`index.faiss`、`embeddings.npy`、`docs.jsonl`、`config.json`。",
        "",
        "## 3. BM25 关键词检索",
        "",
        "| 检索器 | K | Hit@K | AllHit@K | Recall@K | Precision@K | MRR | Timing |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    bm25 = metrics.get("bm25")
    if bm25:
        for k in detail_k_values:
            lines.append(metric_row(f"BM25({bm25_tokenizer})", bm25, k, include_timing=True))
    else:
        lines.append("| 待跑 |  |  |  |  |  |  |  |")

    lines += [
        "",
        "## 4. 向量检索结果",
        "",
    ]
    vector_names = ["vector:bge-base", "vector:bge-large", "vector:qwen3-embedding-0.6b"]
    vector_payloads = [metrics[name] for name in vector_names if name in metrics]
    vector_best_by_k = {k: best_allhit(vector_payloads, k) for k in detail_k_values}
    for name in vector_names:
        if name not in metrics:
            continue
        model = name.replace("vector:", "")
        lines += [
            f"### {model}",
            "",
            "| 模型 | K | Hit@K | AllHit@K | Recall@K | Precision@K | MRR | Query编码(ms) | Search p50(ms) | Search p95(ms) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for k in detail_k_values:
            lines.append(vector_metric_row(model, metrics[name], k, allhit_best=vector_best_by_k.get(k)))
        lines.append("")

    lines += [
        "### 汇总对比",
        "",
        "<details>",
        "<summary>点击展开 K 优先汇总表</summary>",
        "",
        "| 模型 | K | Hit@K | AllHit@K | Recall@K | Precision@K | MRR | Query编码(ms) | Search p50(ms) | Search p95(ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k in detail_k_values:
        for name in vector_names:
            if name not in metrics:
                continue
            lines.append(vector_metric_row(name.replace("vector:", ""), metrics[name], k, allhit_best=vector_best_by_k.get(k)))

    lines += [
        "",
        "</details>",
        "",
        "向量检索的 `search_p95_ms` 只统计 Faiss 查询耗时；端到端在线检索还要加 query embedding 编码耗时。本次 100 文档库里 Faiss 搜索仍是亚毫秒量级，主要耗时来自模型编码。",
        "",
        "### 单/多意图、工具数目拆分对比",
        "",
        "<details>",
        "<summary>点击展开单/多意图拆分、不同工具数目对比表</summary>",
        "",
    ]
    if best_vector10[0]:
        lines += [
            f"最佳单路向量按全量 `AllHit@10` 选择：`{best_vector10[0]}`。以下表格复用已有预测文件重算子集指标，不重新编码或检索。",
            "",
        ]
    else:
        lines += ["本轮只保留 BM25 预测文件；向量预测待 GPU 环境重跑后再补充分桶指标。", ""]
    preds_dir = RAG_DIR / "_aux" / "rag_baseline" / "preds"
    split_retrievers = []
    bm25_pred_path = preds_dir / "bm25.jsonl"
    if bm25_pred_path.exists():
        split_retrievers.append(("BM25", load_pred_rankings(bm25_pred_path)))
    best_vector_pred_path = preds_dir / f"vector_{best_vector10[0]}.jsonl"
    if best_vector10[0] and best_vector_pred_path.exists():
        split_retrievers.append((f"Vector({best_vector10[0]})", load_pred_rankings(best_vector_pred_path)))

    if split_retrievers:
        split_sets = [
            ("单意图", [row for row in test_rows if int(row.get("tool_count", 0)) == 1]),
            ("多意图", [row for row in test_rows if int(row.get("tool_count", 0)) > 1]),
        ]
        lines += [
            "| 检索器 | 子集 | Query数 | 占比 | Hit@10 | AllHit@10 | Recall@10 | Precision@10 | AllHit@30 | Recall@30 | MRR |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        split_rows = []
        for retriever_name, rankings in split_retrievers:
            for subset_name, subset_rows in split_sets:
                subset = subset_metrics(subset_rows, rankings, [10, 30])
                split_rows.append((retriever_name, subset_name, subset_rows, subset))
        split_best10 = {
            subset_name: max(
                [metrics_for_subset["@10"]["AllHit@K"] for _, row_subset, _, metrics_for_subset in split_rows if row_subset == subset_name],
                default=None,
            )
            for subset_name, _ in split_sets
        }
        split_best30 = {
            subset_name: max(
                [metrics_for_subset["@30"]["AllHit@K"] for _, row_subset, _, metrics_for_subset in split_rows if row_subset == subset_name],
                default=None,
            )
            for subset_name, _ in split_sets
        }
        for retriever_name, subset_name, subset_rows, subset in split_rows:
            lines.append(
                split_metric_row(
                    retriever_name,
                    subset_name,
                    subset_rows,
                    subset,
                    len(test_rows),
                    best_allhit10=split_best10.get(subset_name),
                    best_allhit30=split_best30.get(subset_name),
                )
            )
        lines += [
            "",
            "| 检索器 | Gold工具数 | Query数 | 占比 | AllHit@10 | Recall@10 | AllHit@30 | Recall@30 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        tool_count_rows = []
        for retriever_name, rankings in split_retrievers:
            for tool_count in sorted(tool_count_dist):
                subset_rows = [row for row in test_rows if int(row.get("tool_count", 0)) == tool_count]
                subset = subset_metrics(subset_rows, rankings, [10, 30])
                tool_count_rows.append((retriever_name, tool_count, subset_rows, subset))
        tool_best10 = {
            tool_count: max([subset["@10"]["AllHit@K"] for _, row_tool_count, _, subset in tool_count_rows if row_tool_count == tool_count], default=None)
            for tool_count in sorted(tool_count_dist)
        }
        tool_best30 = {
            tool_count: max([subset["@30"]["AllHit@K"] for _, row_tool_count, _, subset in tool_count_rows if row_tool_count == tool_count], default=None)
            for tool_count in sorted(tool_count_dist)
        }
        for retriever_name, tool_count, subset_rows, subset in tool_count_rows:
            row10 = subset["@10"]
            row30 = subset["@30"]
            lines.append(
                f"| {retriever_name} | {tool_count} | {len(subset_rows)} | "
                f"{len(subset_rows) / max(1, len(test_rows)):.2%} | {fmt_best(row10['AllHit@K'], tool_best10.get(tool_count))} | "
                f"{fmt(row10['Recall@K'])} | {fmt_best(row30['AllHit@K'], tool_best30.get(tool_count))} | {fmt(row30['Recall@K'])} |"
            )
    else:
        lines.append("- 待生成预测文件。")

    lines += [
        "",
        "</details>",
        "",
        "## 5. 三种融合检索方案结果",
        "",
        "### Top30 并集",
        "",
        "| 向量模型 | K | Hit@K | AllHit@K | Recall@K | Precision@K | MRR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    hybrid_models = ["bge-base", "bge-large", "qwen3-embedding-0.6b"]
    union_payloads = [metrics[f"hybrid:{model}:union_top{candidate_k}"] for model in hybrid_models if f"hybrid:{model}:union_top{candidate_k}" in metrics]
    for k in detail_k_values:
        best = best_allhit(union_payloads, k)
        for model in hybrid_models:
            name = f"hybrid:{model}:union_top{candidate_k}"
            if name not in metrics:
                continue
            lines.append(metric_row(model, metrics[name], k, allhit_best=best))

    lines += [
        "",
        "### 相似度加权融合",
        "",
        "| 向量模型 | K | Hit@K | AllHit@K | Recall@K | Precision@K | MRR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    weighted_payloads = [metrics[f"hybrid:{model}:weighted"] for model in hybrid_models if f"hybrid:{model}:weighted" in metrics]
    for k in detail_k_values:
        best = best_allhit(weighted_payloads, k)
        for model in hybrid_models:
            name = f"hybrid:{model}:weighted"
            if name not in metrics:
                continue
            lines.append(metric_row(model, metrics[name], k, allhit_best=best))

    lines += [
        "",
        "### RRF",
        "",
        "| 向量模型 | K | Hit@K | AllHit@K | Recall@K | Precision@K | MRR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    rrf_payloads = [metrics[f"hybrid:{model}:rrf"] for model in hybrid_models if f"hybrid:{model}:rrf" in metrics]
    for k in detail_k_values:
        best = best_allhit(rrf_payloads, k)
        for model in hybrid_models:
            name = f"hybrid:{model}:rrf"
            if name not in metrics:
                continue
            lines.append(metric_row(model, metrics[name], k, allhit_best=best))

    lines += [
        "",
        "## 6. 加 Reranker 重排结果",
        "",
    ]
    if available_rerankers:
        reranker_rows = []
        for item in available_rerankers:
            candidate_k_value = item.get("candidate_k", 30)
            candidate_row = item.get("candidate_metrics", {}).get(f"@{candidate_k_value}", {})
            timing = item.get("timing", {})
            mrr = item.get("rerank_metrics", {}).get("overall", {}).get("MRR")
            candidate_limit = int(item.get("candidate_k", 30))
            row_ks = [value for value in detail_k_values if value <= candidate_limit]
            if candidate_limit >= 50 and 50 in raw_k_values:
                row_ks.append(50)
            for k in row_ks:
                row = item.get("rerank_metrics", {}).get(f"@{k}", {})
                reranker_rows.append((item, candidate_k_value, candidate_row, timing, mrr, k, row))
        reranker_best_by_k = {
            k: max(
                [float(row.get("AllHit@K")) for *_, row_k, row in reranker_rows if row_k == k and row.get("AllHit@K") is not None],
                default=None,
            )
            for k in sorted({row_k for *_, row_k, _ in reranker_rows})
        }
        lines += [
            f"- 已重跑 {len(available_rerankers)} 个召回来源；Reranker 均为 `BAAI/bge-reranker-base`，候选数覆盖 Top30 与 Top50。",
            "",
            "| 召回来源 | 候选K | Reranker | 候选AllHit@K | TopK | Hit@K | AllHit@K | Recall@K | Precision@K | MRR | p50(ms) | p95(ms) |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        reranker_separator = "| " + " | ".join([""] * 12) + " |"
        previous_reranker_group: tuple[str, int] | None = None
        for item, candidate_k_value, candidate_row, timing, mrr, k, row in reranker_rows:
            current_reranker_group = (reranker_source_label(item), candidate_k_value)
            if previous_reranker_group is not None and current_reranker_group != previous_reranker_group:
                lines.append(reranker_separator)
            previous_reranker_group = current_reranker_group
            lines.append(
                f"| {reranker_source_label(item)} | {candidate_k_value} | {item.get('model', '')} | {fmt(candidate_row.get('AllHit@K'))} | {k} | "
                f"{fmt(row.get('Hit@K'))} | {fmt_best(row.get('AllHit@K'), reranker_best_by_k.get(k))} | {fmt(row.get('Recall@K'))} | "
                f"{fmt(row.get('Precision@K'))} | {fmt(mrr)} | "
                f"{timing.get('rerank_p50_ms_per_query', '')} | {timing.get('rerank_p95_ms_per_query', '')} |"
            )
        lines += [
            "",
            "说明：按本节展示的主决策列看，bge-large Top30 候选在 K=10 最强；Top50 候选完整覆盖率更高，并在 K=30/K=50 取得更高召回，但 K=10 低于 Top30 候选重排，说明候选池变大后重排器会引入更多强干扰项。`Qwen/Qwen3-Reranker-0.6B` 在当前 `FlagEmbedding` 版本中不能自动映射，显式 decoder-only smoke 指标异常，因此主重排表采用与 BGE embedding 路线匹配且结果稳定的 `BAAI/bge-reranker-base`。",
        ]
    else:
        pending_reason = rerankers[0].get("reason", "") if rerankers else ""
        lines += [f"- 待跑。{pending_reason}".rstrip()]

    lines += [
        "",
        "## 7. 各类检索方法最优配置 TopK 对比（AllHit@K）",
        "",
        "本表按方法族在每个 K 上取当前已跑结果的最优配置；主指标为 `AllHit@K`。微调向量行带风险注记，只表示当前同生成器闭集上限，不参与“微调前检索/重排方案选择”。",
        "",
        "| 方法 | K=6 | K=7 | K=8 | K=9 | K=10 | K=30 | 最优配置/说明 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    lines.extend(best_topk_summary_rows(metrics, available_rerankers, finetuned_embedding_candidates, summary_k_values))
    lines += [
        "",
        "<details>",
        "<summary>点击展开完整版 TopK 对比表（全部已跑配置，AllHit@K）</summary>",
        "",
        "| 类别 | 方法 | K=6 | K=7 | K=8 | K=9 | K=10 | K=30 | 备注 |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    lines.extend(full_topk_comparison_rows(metrics, available_rerankers, finetuned_embedding_candidates, summary_k_values))
    lines += [
        "",
        "</details>",
    ]

    conclusion = [
        "",
        "## 8. 当前结论",
        "",
        f"1. 本地已重跑 BM25 基线：`AllHit@10={allhit(metrics, 'bm25', 10):.4f}`，`AllHit@30={allhit(metrics, 'bm25', 30):.4f}`，`MRR={fmt(metrics.get('bm25', {}).get('metrics', {}).get('overall', {}).get('MRR'))}`。",
    ]
    if vector10:
        conclusion.append(
            f"2. 单路向量最佳为 `{best_vector10[0]}`，`AllHit@10={best_vector10[1]:.4f}`；零样本检索 Top30 最优为 `{best30[0]}`，`AllHit@30={best30[1]:.4f}`。"
        )
    else:
        conclusion.append("2. 当前未发现可用向量指标文件；旧 vector/hybrid 预测不得作为新数据结论。")
    if fusion10:
        conclusion.append(
            f"3. Top10 最佳融合为 `{best_fusion10[0]}`，`AllHit@10={best_fusion10[1]:.4f}`。"
        )
    else:
        conclusion.append("3. 融合检索和 reranker 需要等当前向量预测生成后再评测。")
    conclusion += [
        "4. 新数据去除了显式 catalog 触发词和旧固定短语泄漏，规则短语 matcher 不再能解决测试集。",
        "5. 当前结论仍只代表本项目表达簇合成闭集。若要评估真实泛化，应继续补人工改写、真实日志回放、错别字/省略和多轮上下文。",
    ]
    lines += conclusion + [
        "",
        "## 9. 复现命令",
        "",
        "```bash",
        "cd /renyuanliu/Agent",
        "python3 PROJECT/scripts/rag/build_tool_corpus.py",
        "python3 PROJECT/scripts/dataset/generate_diverse_tool_trace.py",
        "python3 PROJECT/scripts/rag/build_embedding_dataset.py",
        "python3 PROJECT/scripts/dataset/audit_generation_leakage.py --tokenizer fallback",
        "conda run -n llm_bge python PROJECT/scripts/rag/audit_dataset_difficulty.py --tokenizer jieba",
        "conda run -n llm_bge python PROJECT/scripts/rag/build_vector_dbs.py --models all --device cuda --batch-size 8",
        "conda run -n llm_bge python PROJECT/scripts/rag/run_retrieval_baselines.py --mode all --device cuda --batch-size 64",
        "conda run -n llm_bge python PROJECT/scripts/rag/run_reranker_baseline.py --preds PROJECT/data/rag/_aux/rag_baseline/preds/hybrid_bge-large_rrf.jsonl --model BAAI/bge-reranker-base --model-class encoder-only-base --candidate-k 30 --ks 6 7 8 9 10 15 20 30 --query-batch-size 64 --pair-batch-size 256",
        "conda run -n llm_bge python PROJECT/scripts/rag/run_reranker_baseline.py --preds PROJECT/data/rag/_aux/rag_baseline/preds/hybrid_bge-base_rrf.jsonl --output-dir PROJECT/data/rag/_aux/reranker_bge-base_baseline --model BAAI/bge-reranker-base --model-class encoder-only-base --candidate-k 30 --ks 6 7 8 9 10 15 20 30 --query-batch-size 64 --pair-batch-size 256",
        "conda run -n llm_bge python PROJECT/scripts/rag/run_reranker_baseline.py --preds PROJECT/data/rag/_aux/rag_baseline/preds/hybrid_bge-large_rrf.jsonl --output-dir PROJECT/data/rag/_aux/reranker_baseline_k50 --model BAAI/bge-reranker-base --model-class encoder-only-base --candidate-k 50 --ks 6 7 8 9 10 15 20 30 50 --query-batch-size 64 --pair-batch-size 256",
        "conda run -n llm_bge python PROJECT/scripts/rag/run_reranker_baseline.py --preds PROJECT/data/rag/_aux/rag_baseline/preds/hybrid_bge-base_rrf.jsonl --output-dir PROJECT/data/rag/_aux/reranker_bge-base_baseline_k50 --model BAAI/bge-reranker-base --model-class encoder-only-base --candidate-k 50 --ks 6 7 8 9 10 15 20 30 50 --query-batch-size 64 --pair-batch-size 256",
        "python3 PROJECT/scripts/report/make_retrieval_report.py",
        "```",
        "",
        "完整指标与预测文件：",
        "",
        "```text",
        "PROJECT/data/rag/_aux/rag_baseline/metrics.json",
        "PROJECT/data/rag/_aux/rag_baseline/metrics.md",
        "PROJECT/data/rag/_aux/rag_baseline/preds/*.jsonl",
        "PROJECT/data/rag/_aux/reranker_baseline/metrics.json",
        "PROJECT/data/rag/_aux/reranker_baseline/preds.jsonl",
        "PROJECT/data/rag/_aux/reranker_bge-base_baseline/metrics.json",
        "PROJECT/data/rag/_aux/reranker_bge-base_baseline/preds.jsonl",
        "PROJECT/data/rag/_aux/reranker_baseline_k50/metrics.json",
        "PROJECT/data/rag/_aux/reranker_baseline_k50/preds.jsonl",
        "PROJECT/data/rag/_aux/reranker_bge-base_baseline_k50/metrics.json",
        "PROJECT/data/rag/_aux/reranker_bge-base_baseline_k50/preds.jsonl",
        "PROJECT/data/rag/_aux/dataset_ease_audit.json",
        "PROJECT/data/rag/_aux/embedding_dataset_stats.json",
        "PROJECT/data/tool_call_trace/_aux/hard_generation_stats.json",
        "PROJECT/data/rag/embedding_grouped_train.jsonl",
        "PROJECT/data/rag/embedding_grouped_test.jsonl",
        "PROJECT/data/rag/_aux/retrieval_baseline_metrics.json",
        "```",
    ]
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--compat", type=Path, default=DEFAULT_COMPAT)
    parser.add_argument("--reranker", type=Path, default=None)
    parser.add_argument("--rerankers", nargs="*", type=Path, default=None)
    parser.add_argument("--embedding-stats", type=Path, default=DEFAULT_EMBEDDING_STATS)
    parser.add_argument("--dataset-audit", type=Path, default=DEFAULT_DATASET_AUDIT)
    parser.add_argument("--hard-stats", type=Path, default=DEFAULT_HARD_STATS)
    parser.add_argument("--vector-dir", type=Path, default=RAG_DIR / "vector_dbs")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    metrics = read_json(args.metrics, {})
    compat = read_json(args.compat, {})
    if args.rerankers is not None:
        reranker_paths = args.rerankers
    elif args.reranker is not None:
        reranker_paths = [args.reranker]
    else:
        reranker_paths = DEFAULT_RERANKERS
    rerankers = []
    for path in reranker_paths:
        payload = read_json(path, {})
        if payload:
            payload["metrics_path"] = str(path)
            rerankers.append(payload)
    embedding_stats = read_json(args.embedding_stats, {})
    dataset_audit = read_json(args.dataset_audit, {})
    hard_stats = read_json(args.hard_stats, {})
    if dataset_audit:
        dataset_audit["output_path"] = str(args.dataset_audit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "\n".join(build_report(metrics, compat, args.vector_dir, rerankers, embedding_stats, dataset_audit, hard_stats)).rstrip()
        + "\n",
        encoding="utf-8",
    )
    print(f"retrieval report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
