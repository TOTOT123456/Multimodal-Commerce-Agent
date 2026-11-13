#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate PROJECT/configs/tool_catalog.jsonl and write audit reports."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import CATALOG_PATH, PROJECT_ROOT, load_catalog, validate_catalog_rows, write_json


def render_markdown(errors: list[str], stats: dict) -> str:
    lines = [
        "# Catalog Audit",
        "",
        f"- total_tools: {stats.get('total', 0)}",
        f"- status: {'PASS' if not errors else 'FAIL'}",
        "",
        "## Domain Distribution",
        "",
    ]
    for key, value in sorted(stats.get("by_domain", {}).items()):
        lines.append(f"- {key}: {value}")
    lines += ["", "## Server Distribution", ""]
    for key, value in sorted(stats.get("by_server", {}).items()):
        lines.append(f"- {key}: {value}")
    lines += ["", "## Operation Types", ""]
    for key, value in sorted(stats.get("by_operation_type", {}).items()):
        lines.append(f"- {key}: {value}")
    if stats.get("warnings"):
        lines += ["", "## Warnings", ""]
        lines.extend(f"- {item}" for item in stats["warnings"])
    if errors:
        lines += ["", "## Errors", ""]
        lines.extend(f"- {item}" for item in errors)
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    parser.add_argument("--json-out", type=Path, default=PROJECT_ROOT / "reports" / "catalog_audit.json")
    parser.add_argument("--md-out", type=Path, default=None, help="Optional Markdown summary output path.")
    parser.add_argument("--no-fail", action="store_true", help="Write reports but return 0 even when validation fails.")
    args = parser.parse_args()

    catalog = load_catalog(args.catalog)
    errors, stats = validate_catalog_rows(catalog)
    stats["expected_total"] = 100
    if len(catalog) != 100:
        errors.append(f"catalog must contain exactly 100 tools, got {len(catalog)}")
    payload = {"ok": not errors, "errors": errors, **stats}
    write_json(args.json_out, payload)
    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(render_markdown(errors, stats), encoding="utf-8")
    print(f"catalog tools={len(catalog)} errors={len(errors)} -> {args.json_out}")
    return 0 if (not errors or args.no_fail) else 1


if __name__ == "__main__":
    raise SystemExit(main())
