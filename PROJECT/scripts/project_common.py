#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared helpers for the 100-tool e-commerce Agent project."""
from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
REPORT_DIR = PROJECT_ROOT / "reports"
TOOL_TRACE_DIR = DATA_DIR / "tool_call_trace"
INTENT_DIR = DATA_DIR / "intent_recognition"
RAG_DIR = DATA_DIR / "rag"
FINETUNE_DIR = PROJECT_ROOT / "finetune"

CATALOG_PATH = CONFIG_DIR / "tool_catalog.jsonl"
KEYWORD_MAP_PATH = CONFIG_DIR / "tool_keyword_map.json"
TOOL_CORPUS_PATH = RAG_DIR / "tool_corpus.jsonl"

REQUIRED_CATALOG_FIELDS = {
    "tool_id",
    "server",
    "domain",
    "name_zh",
    "description",
    "input_schema_text",
    "output_example",
    "keyword_label",
    "enhanced_desc",
    "positive_phrases",
    "negative_boundary",
    "confusable_tools",
    "operation_type",
    "depends_on",
}
OPTIONAL_CATALOG_FIELDS = {"tags", "version"}
OPERATION_TYPES = {"read", "write", "compute", "decision"}
TOOL_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
SERVER_RE = re.compile(r"^mcp-[a-z0-9-]+$")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{lineno}: expected JSON object")
            rows.append(obj)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_question(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", "", text)
    return re.sub(r"[，。？！、,.?!~…:：；;\"'“”‘’（）()\[\]【】]", "", text)


def split_tools(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        items = [str(x).strip() for x in raw]
    else:
        items = [x.strip() for x in re.split(r"[,\s，、]+", str(raw)) if x.strip()]
    out: list[str] = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out


def join_tools(tools: Iterable[str]) -> str:
    return ", ".join(tools)


def load_catalog(path: Path = CATALOG_PATH) -> list[dict[str, Any]]:
    return load_jsonl(path)


def catalog_by_id(catalog: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["tool_id"]: row for row in catalog}


def validate_catalog_rows(catalog: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    warnings: list[str] = []
    ids: list[str] = []
    allowed = REQUIRED_CATALOG_FIELDS | OPTIONAL_CATALOG_FIELDS

    for idx, row in enumerate(catalog, 1):
        missing = sorted(REQUIRED_CATALOG_FIELDS - row.keys())
        extra = sorted(set(row) - allowed)
        tool_id = str(row.get("tool_id", f"<row-{idx}>"))
        ids.append(tool_id)
        if missing:
            errors.append(f"row {idx} {tool_id}: missing fields {missing}")
        if extra:
            errors.append(f"row {idx} {tool_id}: unexpected fields {extra}")
        if "tool_id" in row and not TOOL_ID_RE.match(str(row["tool_id"])):
            errors.append(f"row {idx} {tool_id}: invalid tool_id")
        if "server" in row and not SERVER_RE.match(str(row["server"])):
            errors.append(f"row {idx} {tool_id}: invalid server")
        if row.get("operation_type") not in OPERATION_TYPES:
            errors.append(f"row {idx} {tool_id}: invalid operation_type")
        phrases = row.get("positive_phrases", [])
        if not isinstance(phrases, list) or len(phrases) < 3:
            errors.append(f"row {idx} {tool_id}: positive_phrases must contain at least 3 items")
        for field in ["confusable_tools", "depends_on"]:
            if field in row and not isinstance(row[field], list):
                errors.append(f"row {idx} {tool_id}: {field} must be an array")
        for field in ["domain", "name_zh", "description", "input_schema_text", "keyword_label", "enhanced_desc", "negative_boundary"]:
            if field in row and not str(row[field]).strip():
                errors.append(f"row {idx} {tool_id}: empty {field}")
        if tool_id and any(tool_id in str(p) for p in phrases):
            warnings.append(f"row {idx} {tool_id}: positive phrase contains tool_id")

    duplicate_ids = [tool_id for tool_id, n in Counter(ids).items() if n > 1]
    if duplicate_ids:
        errors.append(f"duplicate tool_id: {duplicate_ids}")

    id_set = set(ids)
    for row in catalog:
        tool_id = row.get("tool_id", "<unknown>")
        for field in ["confusable_tools", "depends_on"]:
            for ref in row.get(field, []):
                if ref not in id_set:
                    errors.append(f"{tool_id}: {field} references missing tool {ref}")

    stats = {
        "total": len(catalog),
        "by_domain": dict(Counter(row.get("domain", "") for row in catalog)),
        "by_server": dict(Counter(row.get("server", "") for row in catalog)),
        "by_operation_type": dict(Counter(row.get("operation_type", "") for row in catalog)),
        "warnings": warnings,
    }
    return errors, stats


def tool_keyword_map(catalog: list[dict[str, Any]]) -> dict[str, str]:
    return {row["tool_id"]: row["keyword_label"] for row in catalog}


def tools_to_keywords(tools: Iterable[str], mapping: dict[str, str]) -> str:
    labels: list[str] = []
    for tool in tools:
        label = mapping.get(tool, "")
        if label and label not in labels:
            labels.append(label)
    return "，".join(labels)


def operation_rank(tool: dict[str, Any]) -> int:
    order = {"read": 0, "decision": 1, "compute": 2, "write": 3}
    return order.get(tool.get("operation_type"), 9)


def order_tool_chain(tools: Iterable[str], by_id: dict[str, dict[str, Any]]) -> list[str]:
    deduped = []
    for tool in tools:
        if tool in by_id and tool not in deduped:
            deduped.append(tool)
    return sorted(deduped, key=lambda t: (operation_rank(by_id[t]), by_id[t]["domain"], t))


def has_tool_id_leak(question: str, tool_ids: Iterable[str]) -> bool:
    lowered = question.lower()
    return any(tool_id.lower() in lowered for tool_id in tool_ids)


def tool_counts(rows: Iterable[dict[str, Any]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter.update(split_tools(row.get("工具") or row.get("tools")))
    return counter


def domain_counts(rows: Iterable[dict[str, Any]], by_id: dict[str, dict[str, Any]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for row in rows:
        domains = {by_id[t]["domain"] for t in split_tools(row.get("工具") or row.get("tools")) if t in by_id}
        counter.update(domains)
    return counter


def stable_shuffle(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    return rows


def sample_without(values: list[str], blocked: set[str], n: int, rng: random.Random) -> list[str]:
    pool = [x for x in values if x not in blocked]
    rng.shuffle(pool)
    return pool[:n]


def add_script_path() -> None:
    scripts_dir = str(PROJECT_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)


def compact_tool_doc(tool: dict[str, Any]) -> str:
    parts = [
        tool["name_zh"],
        tool["description"],
        tool["enhanced_desc"],
        "触发说法：" + "；".join(tool.get("positive_phrases", [])),
        "边界：" + tool.get("negative_boundary", ""),
    ]
    return "\n".join(str(x) for x in parts if x)


def trace_row_id(prefix: str, idx: int) -> str:
    return f"{prefix}_{idx:06d}"


def group_by_domain(catalog: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in catalog:
        grouped[row["domain"]].append(row)
    return dict(grouped)
