#!/usr/bin/env python3
"""Build leakage-safe cross-style SFT data with deployment-equivalent RAG Top-10."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_AUX = (
    ROOT
    / "PROJECT/data/tool_call_trace/_aux/assignment3"
    / "assignment3_crossstyle_rag10"
)
SYSTEM_PROMPT = """你是电商导购 Agent 的工具轨迹规划器。根据用户问题，从给定工具中选择用户明确需要的全部工具，并按用户叙述及合理业务执行顺序排列。

规则：
1. 只能使用给定列表中的工具 ID。
2. 不要自行补充用户没有要求的工具或前置工具。
3. 不要遗漏用户明确提出的任何子诉求。
4. 只输出合法 JSON，不输出解释、Markdown 或思考过程。
5. 输出格式必须为：{"tools":["tool_a","tool_b"]}。"""
PROMPT_VERSION = "tool-trace-crossstyle-rag10-v1"
CANDIDATE_SOURCE = "tuned_embedding_bm25_task_reranker_rrf_top10"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--api-train",
        type=Path,
        default=ROOT
        / "PROJECT/data/tool_call_trace/_aux/assignment3"
        / "assignment3_100tool_api/train_api/train.jsonl",
    )
    parser.add_argument(
        "--api-rankings",
        type=Path,
        default=ROOT
        / "PROJECT/data/tool_call_trace/_aux/assignment3"
        / "assignment3_100tool_api/train_api/rag_rankings_deployed.jsonl",
    )
    parser.add_argument(
        "--legacy-train",
        type=Path,
        default=ROOT / "PROJECT/data/tool_call_trace/train.jsonl",
    )
    parser.add_argument(
        "--legacy-rankings",
        type=Path,
        default=ROOT
        / "PROJECT/data/tool_call_trace/_aux/assignment3/rag_train_rankings.jsonl",
    )
    parser.add_argument(
        "--legacy-frozen",
        type=Path,
        default=ROOT / "PROJECT/data/tool_call_trace/test.jsonl",
    )
    parser.add_argument(
        "--api-frozen",
        type=Path,
        default=ROOT
        / "PROJECT/data/tool_call_trace/_aux/assignment3"
        / "assignment3_100tool_api/test_api/test.jsonl",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=ROOT / "PROJECT/configs/tool_catalog.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_AUX)
    parser.add_argument(
        "--runtime-out",
        type=Path,
        default=ROOT
        / "PROJECT/finetune/outputs/tooltrace-data-cache/runs"
        / "assignment3_crossstyle_rag10_sft.runtime.jsonl",
    )
    parser.add_argument(
        "--tokenizer-models",
        type=Path,
        nargs="+",
        default=[
            ROOT / "PROJECT/models/Qwen/Qwen3-4B",
            ROOT / "PROJECT/models/Qwen/Qwen3-8B",
        ],
    )
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--prompt-version", default=PROMPT_VERSION)
    parser.add_argument("--candidate-source", default=CANDIDATE_SOURCE)
    parser.add_argument("--expected-api", type=int, default=54_000)
    parser.add_argument("--expected-legacy", type=int, default=24_000)
    parser.add_argument("--seed", type=int, default=20_260_728)
    parser.add_argument(
        "--length-candidates",
        type=int,
        nargs="+",
        default=[512, 768, 1024, 1536, 2048],
    )
    parser.add_argument("--safety-margin", type=int, default=64)
    return parser.parse_args()


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    temporary.replace(path)


def normalize_query(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"invalid query: {value!r}")
    return re.sub(r"\s+", " ", value).strip()


def parse_tools(value: Any) -> list[str]:
    if isinstance(value, list):
        tools = value
    elif isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            tools = json.loads(stripped)
        elif re.search(r"[,，|]", stripped):
            tools = re.split(r"[,，|]\s*", stripped)
        else:
            tools = [stripped]
    else:
        raise ValueError(f"invalid tools: {value!r}")
    if not tools or not all(isinstance(tool, str) and tool for tool in tools):
        raise ValueError(f"invalid tools: {value!r}")
    if len(tools) != len(set(tools)):
        raise ValueError(f"duplicate target tool: {tools}")
    return tools


def load_catalog(path: Path) -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        tool_id = row["tool_id"]
        if tool_id in catalog:
            raise ValueError(f"duplicate catalog tool: {tool_id}")
        catalog[tool_id] = row
    if len(catalog) != 100:
        raise ValueError(f"expected 100 catalog tools, got {len(catalog)}")
    return catalog


def load_rankings(path: Path) -> dict[str, dict[str, Any]]:
    rankings: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        trace_id = row["id"]
        if trace_id in rankings:
            raise ValueError(f"duplicate ranking id in {path}: {trace_id}")
        rankings[trace_id] = row
    return rankings


def build_user_prompt(
    query: str,
    candidate_tools: list[str],
    catalog: dict[str, dict[str, Any]],
) -> str:
    candidate_lines = []
    for tool_id in candidate_tools:
        tool = catalog[tool_id]
        candidate_lines.append(
            " | ".join(
                [
                    tool_id,
                    tool["domain"],
                    tool["operation_type"],
                    tool["description"],
                ]
            )
        )
    return f"候选工具：\n{chr(10).join(candidate_lines)}\n\n用户问题：\n{query}"


def source_rows(
    *,
    source_name: str,
    rows: list[dict[str, Any]],
    rankings: dict[str, dict[str, Any]],
    catalog: dict[str, dict[str, Any]],
    candidate_k: int,
    prompt_version: str,
    candidate_source: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    compact_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    missing_gold: list[dict[str, Any]] = []
    tool_count_input: Counter[int] = Counter()
    tool_count_kept: Counter[int] = Counter()
    gold_rank: Counter[int] = Counter()
    seen_ids: set[str] = set()

    for row in rows:
        trace_id = row["id"]
        if trace_id in seen_ids:
            raise ValueError(f"duplicate source id in {source_name}: {trace_id}")
        seen_ids.add(trace_id)
        if trace_id not in rankings:
            raise ValueError(f"missing ranking for {source_name}:{trace_id}")
        ranking = rankings[trace_id]
        query = normalize_query(row["问题"])
        ranked_query = normalize_query(ranking["query"])
        if query != ranked_query:
            raise ValueError(f"query mismatch for {source_name}:{trace_id}")
        gold_tools = parse_tools(row["工具"])
        ranked_gold = parse_tools(ranking["gold_tools"])
        if gold_tools != ranked_gold:
            raise ValueError(
                f"gold mismatch for {source_name}:{trace_id}: "
                f"{gold_tools!r} != {ranked_gold!r}"
            )
        if any(tool not in catalog for tool in gold_tools):
            raise ValueError(f"unknown gold tool for {source_name}:{trace_id}")
        ranked_tools = parse_tools(ranking["pred_tools"])
        if len(ranked_tools) != len(set(ranked_tools)):
            raise ValueError(f"duplicate ranked tool for {source_name}:{trace_id}")
        if any(tool not in catalog for tool in ranked_tools):
            raise ValueError(f"unknown ranked tool for {source_name}:{trace_id}")
        candidate_tools = ranked_tools[:candidate_k]
        tool_count_input[len(gold_tools)] += 1
        if not set(gold_tools).issubset(candidate_tools):
            missing_gold.append(
                {
                    "trace_id": trace_id,
                    "gold_tools": gold_tools,
                    "candidate_tools": candidate_tools,
                    "missing_tools": [
                        tool for tool in gold_tools if tool not in candidate_tools
                    ],
                }
            )
            continue

        for tool in gold_tools:
            gold_rank[candidate_tools.index(tool) + 1] += 1
        target = json.dumps(
            {"tools": gold_tools},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        user_prompt = build_user_prompt(query, candidate_tools, catalog)
        prompt_hash = sha256_text(
            json_dumps({"system": SYSTEM_PROMPT, "user": user_prompt})
        )
        source_split = (
            "assignment3_api_native_train"
            if source_name == "api_native"
            else "legacy_train"
        )
        compact: dict[str, Any] = {
            "trace_id": trace_id,
            "query": query,
            "gold_tools": gold_tools,
            "target": target,
            "source_split": source_split,
            "view": "rag",
            "candidate_tools": candidate_tools,
            "candidate_source": candidate_source,
            "candidate_k": candidate_k,
            "prompt_version": prompt_version,
            "prompt_hash": prompt_hash,
            "ranking_hash": sha256_text(json_dumps(candidate_tools)),
            "tool_count": len(gold_tools),
            "domain": row.get("domain"),
            "combo_seen_bucket": row.get("combo_seen_bucket"),
        }
        if source_name == "api_native":
            compact["construction"] = row.get("construction", {})
        else:
            compact["template_family"] = row.get("template_family")
            compact["expression_cluster_ids"] = row.get(
                "expression_cluster_ids", {}
            )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": target},
        ]
        meta: dict[str, Any] = {
            "trace_id": trace_id,
            "source_split": source_split,
            "view": "rag",
            "candidate_source": candidate_source,
            "candidate_k": candidate_k,
            "prompt_hash": prompt_hash,
            "prompt_version": prompt_version,
        }
        record_hash = sha256_text(json_dumps({"messages": messages, "meta": meta}))
        meta["record_hash"] = record_hash
        compact["record_hash"] = record_hash
        compact_rows.append(compact)
        runtime_rows.append({"messages": messages, "meta": meta})
        tool_count_kept[len(gold_tools)] += 1

    extra_rankings = sorted(set(rankings) - seen_ids)
    if extra_rankings:
        raise ValueError(
            f"{source_name} rankings contain {len(extra_rankings)} extra ids; "
            f"first={extra_rankings[0]}"
        )
    stats = {
        "input_rows": len(rows),
        "kept_rows": len(compact_rows),
        "dropped_missing_gold": len(missing_gold),
        "all_hit_at_k": len(compact_rows) / len(rows),
        "tool_count_input": dict(sorted(tool_count_input.items())),
        "tool_count_kept": dict(sorted(tool_count_kept.items())),
        "gold_tool_rank_histogram": dict(sorted(gold_rank.items())),
        "missing_gold_examples": missing_gold[:20],
        "missing_gold_ids_sha256": sha256_text(
            json_dumps(sorted(item["trace_id"] for item in missing_gold))
        ),
    }
    return compact_rows, runtime_rows, stats


def frozen_index(path: Path) -> tuple[set[str], set[str]]:
    ids: set[str] = set()
    queries: set[str] = set()
    for row in read_jsonl(path):
        ids.add(row["id"])
        queries.add(normalize_query(row["问题"]))
    return ids, queries


def percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot summarize empty values")
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def describe(values: list[int]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "min": min(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def tokenizer_identity(model_path: Path) -> dict[str, Any]:
    files: dict[str, str] = {}
    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "config.json",
        "generation_config.json",
    ]:
        path = model_path / name
        if path.exists():
            files[name] = sha256_file(path)
    return {
        "model_path": str(model_path),
        "files": files,
        "identity_hash": sha256_text(json_dumps(files)),
    }


def token_stats(
    runtime_rows: list[dict[str, Any]],
    model_paths: list[Path],
    length_candidates: list[int],
    safety_margin: int,
    prompt_version: str,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    conversations = [row["messages"] for row in runtime_rows]
    cache: dict[str, tuple[list[int], list[int], list[int]]] = {}
    result: dict[str, Any] = {
        "status": "complete",
        "prompt_version": prompt_version,
        "records": len(runtime_rows),
        "tokenizers": {},
    }
    for model_path in model_paths:
        identity = tokenizer_identity(model_path)
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=True,
        )
        tokenizer_key = identity["files"].get(
            "tokenizer.json", identity["identity_hash"]
        )
        if tokenizer_key not in cache:
            input_lengths: list[int] = []
            total_lengths: list[int] = []
            target_lengths: list[int] = []
            for start in range(0, len(conversations), 256):
                batch = conversations[start : start + 256]
                total_ids = tokenizer.apply_chat_template(
                    batch,
                    tokenize=True,
                    add_generation_prompt=False,
                    enable_thinking=False,
                )
                input_ids = tokenizer.apply_chat_template(
                    [messages[:2] for messages in batch],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for messages, total, prompt in zip(batch, total_ids, input_ids):
                    total_lengths.append(len(total))
                    input_lengths.append(len(prompt))
                    target_lengths.append(
                        len(
                            tokenizer.encode(
                                messages[2]["content"],
                                add_special_tokens=False,
                            )
                        )
                    )
            cache[tokenizer_key] = (
                input_lengths,
                total_lengths,
                target_lengths,
            )
        input_lengths, total_lengths, target_lengths = cache[tokenizer_key]
        required = max(total_lengths) + safety_margin
        selected = next(
            (length for length in sorted(length_candidates) if length >= required),
            None,
        )
        if selected is None:
            raise ValueError(
                f"{model_path}: max tokens {max(total_lengths)} plus margin "
                f"{safety_margin} exceeds candidates {length_candidates}"
            )
        identity["tokenizer_class"] = tokenizer.__class__.__name__
        identity["model_max_length"] = tokenizer.model_max_length
        result["tokenizers"][str(model_path)] = {
            "input_tokens": describe(input_lengths),
            "total_tokens": describe(total_lengths),
            "target_tokens": describe(target_lengths),
            "max_total_tokens": max(total_lengths),
            "target_preserved_rate": 1.0,
            "identity": identity,
            "selected_max_length": selected,
            "safety_margin": safety_margin,
            "truncated_records": 0,
        }
    return result


def main() -> int:
    args = parse_args()
    if args.candidate_k != 10:
        raise ValueError("this protocol is fixed to candidate_k=10")

    catalog = load_catalog(args.catalog)
    api_rows = read_jsonl(args.api_train)
    legacy_rows = read_jsonl(args.legacy_train)
    if len(api_rows) != args.expected_api:
        raise ValueError(
            f"expected {args.expected_api} API rows, got {len(api_rows)}"
        )
    if len(legacy_rows) != args.expected_legacy:
        raise ValueError(
            f"expected {args.expected_legacy} legacy rows, got {len(legacy_rows)}"
        )

    api_compact, api_runtime, api_stats = source_rows(
        source_name="api_native",
        rows=api_rows,
        rankings=load_rankings(args.api_rankings),
        catalog=catalog,
        candidate_k=args.candidate_k,
        prompt_version=args.prompt_version,
        candidate_source=args.candidate_source,
    )
    legacy_compact, legacy_runtime, legacy_stats = source_rows(
        source_name="legacy",
        rows=legacy_rows,
        rankings=load_rankings(args.legacy_rankings),
        catalog=catalog,
        candidate_k=args.candidate_k,
        prompt_version=args.prompt_version,
        candidate_source=args.candidate_source,
    )

    compact_rows = api_compact + legacy_compact
    runtime_rows = api_runtime + legacy_runtime
    paired = list(zip(compact_rows, runtime_rows))
    random.Random(args.seed).shuffle(paired)
    compact_rows = [item[0] for item in paired]
    runtime_rows = [item[1] for item in paired]

    train_ids = [row["trace_id"] for row in compact_rows]
    train_queries = [normalize_query(row["query"]) for row in compact_rows]
    legacy_frozen_ids, legacy_frozen_queries = frozen_index(args.legacy_frozen)
    api_frozen_ids, api_frozen_queries = frozen_index(args.api_frozen)
    source_trace_ids = {
        row.get("construction", {}).get("source_trace_id")
        for row in api_compact
        if row.get("construction", {}).get("source_trace_id")
    }
    leakage = {
        "duplicate_train_ids": len(train_ids) - len(set(train_ids)),
        "duplicate_train_queries": len(train_queries) - len(set(train_queries)),
        "legacy_frozen_id_overlap": len(set(train_ids) & legacy_frozen_ids),
        "api_frozen_id_overlap": len(set(train_ids) & api_frozen_ids),
        "legacy_frozen_query_overlap": len(
            set(train_queries) & legacy_frozen_queries
        ),
        "api_frozen_query_overlap": len(set(train_queries) & api_frozen_queries),
        "api_source_trace_legacy_frozen_overlap": len(
            source_trace_ids & legacy_frozen_ids
        ),
        "api_legacy_train_query_overlap": len(
            {normalize_query(row["query"]) for row in api_compact}
            & {normalize_query(row["query"]) for row in legacy_compact}
        ),
    }
    if any(leakage.values()):
        raise ValueError(f"strict isolation audit failed: {leakage}")

    compact_out = args.output_dir / "sft_compact.jsonl"
    audit_out = args.output_dir / "data_audit.json"
    stats_out = args.output_dir / "sft_prompt_stats.json"
    smoke_out = args.output_dir / "sft_smoke.jsonl"
    write_jsonl(compact_out, compact_rows)
    write_jsonl(args.runtime_out, runtime_rows)

    smoke_keys: set[tuple[str, int]] = set()
    smoke: list[dict[str, Any]] = []
    for compact, runtime in zip(compact_rows, runtime_rows):
        key = (compact["source_split"], compact["tool_count"])
        if key not in smoke_keys:
            smoke_keys.add(key)
            smoke.append(runtime)
    write_jsonl(smoke_out, smoke)

    stats = token_stats(
        runtime_rows,
        args.tokenizer_models,
        args.length_candidates,
        args.safety_margin,
        args.prompt_version,
    )
    write_json(stats_out, stats)

    source_counts = Counter(row["source_split"] for row in compact_rows)
    tool_counts = Counter(row["tool_count"] for row in compact_rows)
    audit = {
        "status": "complete",
        "strict_ok": True,
        "protocol": {
            "prompt_version": args.prompt_version,
            "candidate_source": args.candidate_source,
            "candidate_k": args.candidate_k,
            "missing_gold_policy": "drop_before_training",
            "shuffle_seed": args.seed,
            "frozen_sets_are_evaluation_only": True,
        },
        "counts": {
            "input_total": len(api_rows) + len(legacy_rows),
            "kept_total": len(compact_rows),
            "dropped_missing_gold_total": (
                api_stats["dropped_missing_gold"]
                + legacy_stats["dropped_missing_gold"]
            ),
            "by_source": dict(sorted(source_counts.items())),
            "by_tool_count": dict(sorted(tool_counts.items())),
        },
        "sources": {
            "api_native": api_stats,
            "legacy": legacy_stats,
        },
        "leakage": leakage,
        "token_stats": {
            model: {
                "max_total_tokens": value["max_total_tokens"],
                "selected_max_length": value["selected_max_length"],
                "truncated_records": value["truncated_records"],
            }
            for model, value in stats["tokenizers"].items()
        },
        "inputs": {
            str(path): sha256_file(path)
            for path in [
                args.api_train,
                args.api_rankings,
                args.legacy_train,
                args.legacy_rankings,
                args.legacy_frozen,
                args.api_frozen,
                args.catalog,
            ]
        },
        "outputs": {
            str(compact_out): sha256_file(compact_out),
            str(args.runtime_out): sha256_file(args.runtime_out),
            str(stats_out): sha256_file(stats_out),
            str(smoke_out): sha256_file(smoke_out),
        },
    }
    write_json(audit_out, audit)
    print(
        f"cross-style SFT rows={len(compact_rows)} "
        f"api={source_counts['assignment3_api_native_train']} "
        f"legacy={source_counts['legacy_train']} "
        f"dropped={audit['counts']['dropped_missing_gold_total']} "
        f"strict_ok={audit['strict_ok']}"
    )
    print(f"runtime={args.runtime_out}")
    print(f"audit={audit_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
