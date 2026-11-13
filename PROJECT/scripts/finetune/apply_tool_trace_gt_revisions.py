#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Apply explicit reviewed query/evidence revisions to derived train artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import TOOL_TRACE_DIR, load_jsonl, normalize_question, write_json, write_jsonl  # noqa: E402
from finetune.tool_trace_common import ASSIGNMENT3_DIR, sha256_file  # noqa: E402


def replace_once(text: str, original: str, revised: str, trace_id: str) -> str:
    if text.count(original) != 1:
        raise SystemExit(
            f"{trace_id}: expected one occurrence of evidence {original!r}, "
            f"found {text.count(original)}"
        )
    return text.replace(original, revised, 1)


def revision_map(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(path)
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise SystemExit("duplicate revision id")
    if any(row.get("split") != "train" for row in rows):
        raise SystemExit("this derivation only accepts train revisions")
    return by_id


def apply_data(args: argparse.Namespace) -> None:
    revisions = revision_map(args.revisions)
    train = load_jsonl(args.train)
    train_by_id = {row["id"]: row for row in train}
    if not set(revisions) <= set(train_by_id):
        raise SystemExit("revision ids are not all in formal train")
    meta = load_jsonl(args.formal_meta)
    meta_by_id = {row["trace_id"]: row for row in meta}
    if not set(revisions) <= set(meta_by_id):
        raise SystemExit("revision ids are not all in formal generation meta")

    revised_train = []
    revised_subset = []
    effective_queries: dict[str, str] = {}
    for source in train:
        row = dict(source)
        revision = revisions.get(row["id"])
        if revision:
            meta_row = dict(meta_by_id[row["id"]])
            phrases = dict(meta_row["expression_phrases"])
            evidence = dict(meta_row["tool_evidence"])
            query = row["问题"]
            for tool_id, replacement in revision["evidence_replacements"].items():
                if phrases.get(tool_id) != replacement["original"]:
                    raise SystemExit(f"{row['id']}:{tool_id}: expression phrase mismatch")
                if evidence.get(tool_id) != replacement["original"]:
                    raise SystemExit(f"{row['id']}:{tool_id}: evidence mismatch")
                query = replace_once(
                    query,
                    replacement["original"],
                    replacement["revised"],
                    row["id"],
                )
                phrases[tool_id] = replacement["revised"]
                evidence[tool_id] = replacement["revised"]
            row["问题"] = query
            meta_row["query"] = query
            meta_row["expression_phrases"] = phrases
            meta_row["tool_evidence"] = evidence
            meta_by_id[row["id"]] = meta_row
            revised_subset.append(row)
        effective_queries[row["id"]] = row["问题"]
        revised_train.append(row)

    revised_meta = [meta_by_id[row["trace_id"]] for row in meta]
    write_jsonl(args.effective_train, revised_train)
    write_jsonl(args.effective_meta, revised_meta)
    write_jsonl(args.revised_subset, revised_subset)

    dev_queries = {
        normalize_question(row["问题"]) for row in load_jsonl(args.dev)
    }
    frozen_queries = {
        normalize_question(row["问题"]) for row in load_jsonl(args.frozen)
    }
    effective_norms = {normalize_question(row["问题"]) for row in revised_train}
    payload = {
        "status": "complete",
        "revision_count": len(revisions),
        "revised_trace_ids": sorted(revisions),
        "source_hashes": {
            str(args.train): sha256_file(args.train),
            str(args.formal_meta): sha256_file(args.formal_meta),
            str(args.revisions): sha256_file(args.revisions),
        },
        "output_hashes": {
            str(args.effective_train): sha256_file(args.effective_train),
            str(args.effective_meta): sha256_file(args.effective_meta),
            str(args.revised_subset): sha256_file(args.revised_subset),
        },
        "effective_train_rows": len(revised_train),
        "revised_subset_rows": len(revised_subset),
        "effective_query_unique": len(effective_norms),
        "effective_dev_query_overlap": len(effective_norms & dev_queries),
        "effective_frozen_query_overlap": len(effective_norms & frozen_queries),
        "source_files_unchanged": True,
        "policy": (
            "Only derived assignment-3 artifacts are revised. Formal train/test "
            "and frozen labels are not overwritten."
        ),
    }
    if (
        len(revised_train) != 24000
        or len(revised_subset) != len(revisions)
        or len(effective_norms) != len(revised_train)
        or payload["effective_dev_query_overlap"]
        or payload["effective_frozen_query_overlap"]
    ):
        payload["status"] = "failed"
    write_json(args.manifest, payload)
    print(
        f"effective train rows={len(revised_train)} revisions={len(revisions)} "
        f"dev_overlap={payload['effective_dev_query_overlap']} "
        f"frozen_overlap={payload['effective_frozen_query_overlap']}"
    )
    if payload["status"] != "complete":
        raise SystemExit(1)


def patch_rankings(args: argparse.Namespace) -> None:
    revisions = revision_map(args.revisions)
    effective = load_jsonl(args.effective_train)
    effective_by_id = {row["id"]: row for row in effective}
    patch_final = {row["id"]: row for row in load_jsonl(args.revised_final)}
    patch_untuned = {row["id"]: row for row in load_jsonl(args.revised_untuned)}
    if set(patch_final) != set(revisions) or set(patch_untuned) != set(revisions):
        raise SystemExit("live revised rankings do not cover exactly the revision ids")

    def build(source_path: Path, patch: dict[str, dict[str, Any]], output: Path) -> None:
        source = load_jsonl(source_path)
        if {row["id"] for row in source} != set(effective_by_id):
            raise SystemExit(f"{source_path}: ranking ids do not match effective train")
        rows = []
        for row in source:
            trace_id = row["id"]
            if trace_id in patch:
                replacement = patch[trace_id]
                if replacement["query"] != effective_by_id[trace_id]["问题"]:
                    raise SystemExit(f"{trace_id}: live ranking query mismatch")
                rows.append(replacement)
            else:
                updated = dict(row)
                updated["query"] = effective_by_id[trace_id]["问题"]
                rows.append(updated)
        write_jsonl(output, rows)

    build(args.final_rankings, patch_final, args.effective_final_rankings)
    build(args.untuned_rankings, patch_untuned, args.effective_untuned_rankings)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest["ranking_patch"] = {
        "revised_final": str(args.revised_final),
        "revised_final_sha256": sha256_file(args.revised_final),
        "revised_untuned": str(args.revised_untuned),
        "revised_untuned_sha256": sha256_file(args.revised_untuned),
        "effective_final": str(args.effective_final_rankings),
        "effective_final_sha256": sha256_file(args.effective_final_rankings),
        "effective_untuned": str(args.effective_untuned_rankings),
        "effective_untuned_sha256": sha256_file(args.effective_untuned_rankings),
    }
    write_json(args.manifest, manifest)
    print(f"patched full train rankings revisions={len(revisions)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["data", "rankings"], required=True)
    parser.add_argument("--train", type=Path, default=TOOL_TRACE_DIR / "train.jsonl")
    parser.add_argument("--formal-meta", type=Path, default=TOOL_TRACE_DIR / "_aux" / "generation_meta.jsonl")
    parser.add_argument(
        "--dev",
        type=Path,
        default=ASSIGNMENT3_DIR / "api_native_dev_2400" / "dev.jsonl",
    )
    parser.add_argument("--frozen", type=Path, default=TOOL_TRACE_DIR / "test.jsonl")
    parser.add_argument("--revisions", type=Path, default=ASSIGNMENT3_DIR / "gt_revisions.jsonl")
    parser.add_argument("--effective-train", type=Path, default=ASSIGNMENT3_DIR / "train.reviewed.jsonl")
    parser.add_argument("--effective-meta", type=Path, default=ASSIGNMENT3_DIR / "generation_meta.reviewed.jsonl")
    parser.add_argument("--revised-subset", type=Path, default=ASSIGNMENT3_DIR / "train_revised_subset.jsonl")
    parser.add_argument("--manifest", type=Path, default=ASSIGNMENT3_DIR / "gt_revisions_manifest.json")
    parser.add_argument("--final-rankings", type=Path, default=ASSIGNMENT3_DIR / "rag_train_rankings.jsonl")
    parser.add_argument("--untuned-rankings", type=Path, default=ASSIGNMENT3_DIR / "rag_train_untuned_rankings.jsonl")
    parser.add_argument("--revised-final", type=Path, default=ASSIGNMENT3_DIR / "rag_train_revised_rows.jsonl")
    parser.add_argument("--revised-untuned", type=Path, default=ASSIGNMENT3_DIR / "rag_train_revised_untuned_rows.jsonl")
    parser.add_argument("--effective-final-rankings", type=Path, default=ASSIGNMENT3_DIR / "rag_train_rankings.reviewed.jsonl")
    parser.add_argument("--effective-untuned-rankings", type=Path, default=ASSIGNMENT3_DIR / "rag_train_untuned_rankings.reviewed.jsonl")
    args = parser.parse_args()
    if args.phase == "data":
        apply_data(args)
    else:
        patch_rankings(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
