#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Join semantic judgments into the GT review sheet without claiming human review."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import load_jsonl, write_json, write_jsonl  # noqa: E402
from finetune.tool_trace_common import ASSIGNMENT3_DIR, sha256_file, sha256_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path, default=ASSIGNMENT3_DIR / "gt_review_sample.jsonl")
    parser.add_argument("--judgments", type=Path, default=ASSIGNMENT3_DIR / "gt_review_judgments.jsonl")
    parser.add_argument("--overrides", type=Path, default=ASSIGNMENT3_DIR / "gt_review_overrides.jsonl")
    parser.add_argument("--revisions", type=Path, default=ASSIGNMENT3_DIR / "gt_revisions.jsonl")
    parser.add_argument(
        "--pair-judgments",
        type=Path,
        default=ASSIGNMENT3_DIR / "gt_review_pair_judgments.jsonl",
    )
    parser.add_argument("--audit", type=Path, default=ASSIGNMENT3_DIR / "gt_quality_audit.json")
    parser.add_argument("--out", type=Path, default=ASSIGNMENT3_DIR / "gt_review_sample.reviewed.jsonl")
    args = parser.parse_args()

    review = load_jsonl(args.review)
    judgments = load_jsonl(args.judgments)
    overrides = load_jsonl(args.overrides) if args.overrides.exists() else []
    revisions = load_jsonl(args.revisions) if args.revisions.exists() else []
    revision_by_id = {row["id"]: row for row in revisions}
    by_id = {row["id"]: row for row in judgments}
    override_by_id = {row["id"]: row for row in overrides}
    if len(by_id) != len(judgments) or len(override_by_id) != len(overrides):
        raise SystemExit("duplicate judgment or override id")
    review_ids = {row["id"] for row in review}
    if set(by_id) != review_ids or not set(override_by_id) <= review_ids:
        raise SystemExit("review/judgment/override ids are not aligned")

    final_rows = []
    unresolved = []
    final_counts: Counter[str] = Counter()
    for row in review:
        judgment = by_id[row["id"]]
        expected_hash = sha256_json(
            {
                "id": row["id"],
                "query": row["query"],
                "gold_tools": row["gold_tools"],
                "tool_evidence": row["tool_evidence"],
            }
        )
        if judgment["review_input_hash"] != expected_hash:
            raise SystemExit(f"review input hash mismatch: {row['id']}")
        revision = revision_by_id.get(row["id"])
        override = override_by_id.get(row["id"])
        if revision:
            verdict = "corrected"
            notes = revision["reason"]
        else:
            verdict = override["verdict"] if override else judgment["verdict"]
            notes = override.get("notes", "") if override else "; ".join(judgment["issues"])
        if verdict not in {"pass", "corrected", "fail", "uncertain"}:
            raise SystemExit(f"invalid final verdict: {row['id']}")
        if verdict not in {"pass", "corrected"}:
            unresolved.append({"id": row["id"], "verdict": verdict, "notes": notes})
        final_counts[verdict] += 1
        updated = dict(row)
        updated["semantic_review"] = {
            "status": (
                "approved_after_explicit_revision"
                if verdict == "corrected"
                else "approved_agent_review"
                if verdict == "pass"
                else "requires_revision"
            ),
            "verdict": verdict,
            "reviewer_type": (
                "Qwen3-8B deterministic semantic judge; Codex adjudication"
                if override or revision
                else "Qwen3-8B deterministic semantic judge"
            ),
            "notes": notes,
            "judgment_prompt_hash": judgment["prompt_hash"],
            "judgment_raw_output": judgment["raw_output"],
            "independent_human_review": False,
        }
        final_rows.append(updated)
    write_jsonl(args.out, final_rows)

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    audit["review_sample"].update(
        {
            "semantic_review_status": (
                "complete_agent_review_no_independent_human"
                if not unresolved
                else "agent_review_has_unresolved_cases"
            ),
            "semantic_verdict_counts": dict(final_counts),
            "unresolved": unresolved,
            "reviewed_path": str(args.out),
            "reviewed_sha256": sha256_file(args.out),
            "judgments_path": str(args.judgments),
            "judgments_sha256": sha256_file(args.judgments),
            "overrides_path": str(args.overrides),
            "overrides_sha256": sha256_file(args.overrides) if args.overrides.exists() else None,
            "revisions_path": str(args.revisions),
            "revisions_sha256": sha256_file(args.revisions),
            "pair_judgments_path": str(args.pair_judgments),
            "pair_judgments_sha256": sha256_file(args.pair_judgments),
            "pair_judgment_verdict_counts": dict(
                Counter(row["verdict"] for row in load_jsonl(args.pair_judgments))
            ),
            "pair_judgment_interpretation": (
                "The deliberately over-strict pair diagnostic is retained for "
                "sensitivity analysis; final decisions use row-level judgment "
                "plus Codex adjudication and explicit revisions."
            ),
            "independent_human_review": False,
        }
    )
    if unresolved:
        audit["status"] = "agent_semantic_review_failed"
    elif audit.get("strict_automatic_ok"):
        audit["status"] = "complete_with_agent_semantic_review_no_independent_human"
    write_json(args.audit, audit)
    print(
        f"GT semantic review rows={len(final_rows)} verdicts={dict(final_counts)} "
        f"unresolved={len(unresolved)}"
    )
    return 0 if not unresolved else 1


if __name__ == "__main__":
    raise SystemExit(main())
