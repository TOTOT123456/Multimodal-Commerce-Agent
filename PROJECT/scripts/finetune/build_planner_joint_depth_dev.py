#!/usr/bin/env python3
"""Build a representative, label-free plan for joint Reranker/Planner depth selection."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import (  # noqa: E402
    CATALOG_PATH,
    catalog_by_id,
    load_catalog,
    load_jsonl,
    split_tools,
    write_json,
    write_jsonl,
)
from finetune.build_planner_candidate_depth_isolated_dev import (  # noqa: E402
    build_candidate_pools,
    dataset_combo,
    dataset_query,
    dependency_errors,
    dependency_order,
    normalize_query,
    runtime_query,
    runtime_target,
)
from finetune.tool_trace_common import sha256_file, stable_seed  # noqa: E402


SPLIT_COUNTS = {
    "selection": {
        "legacy_like": {1: 600, 2: 96, 3: 136, 4: 118, 5: 131, 6: 119},
        "api_like": {1: 240, 2: 240, 3: 240, 4: 240, 5: 240},
    },
    "confirmation": {
        "legacy_like": {1: 300, 2: 48, 3: 68, 4: 59, 5: 66, 6: 59},
        "api_like": {1: 120, 2: 120, 3: 120, 4: 120, 5: 120},
    },
}
LEGACY_SEEN_RATIOS = {
    2: 253 / 481,
    3: 346 / 681,
    4: 300 / 592,
    5: 300 / 653,
    6: 301 / 593,
}
STYLE_BY_PANEL = {"legacy_like": "legacy_compact", "api_like": "api_native"}


def selected_splits(args: argparse.Namespace) -> tuple[str, ...]:
    return (
        tuple(SPLIT_COUNTS)
        if args.dataset_role == "both"
        else (args.dataset_role,)
    )


def choose_combo(
    pool: list[frozenset[str]],
    usage: Counter[frozenset[str]],
    exposure: Counter[str],
    seed: int,
    *parts: object,
) -> frozenset[str]:
    if not pool:
        raise ValueError(f"empty combination pool for {parts}")
    selected = min(
        pool,
        key=lambda combo: (
            usage[combo],
            max(exposure[tool] + 1 for tool in combo),
            sum((exposure[tool] + 1) ** 2 for tool in combo),
            stable_seed(seed, *parts, *sorted(combo)),
        ),
    )
    usage[selected] += 1
    exposure.update(selected)
    return selected


def seen_rows(panel: str, tool_count: int, total: int) -> int:
    if panel == "api_like" or tool_count == 1:
        return total
    return round(total * LEGACY_SEEN_RATIOS[tool_count])


def plan_stage(args: argparse.Namespace) -> int:
    catalog_rows = load_catalog(args.catalog)
    catalog = catalog_by_id(catalog_rows)
    sft_rows = load_jsonl(args.sft_runtime)
    sft_combos = {frozenset(runtime_target(row)) for row in sft_rows}
    sft_queries = {normalize_query(runtime_query(row)) for row in sft_rows}
    sft_source_ids = {str(row["meta"]["trace_id"]) for row in sft_rows}

    blocked_combos: set[frozenset[str]] = set()
    blocked_queries = set(sft_queries)
    for path in args.blocked_dataset:
        for row in load_jsonl(path):
            combo = dataset_combo(row)
            if combo:
                blocked_combos.add(combo)
            query = dataset_query(row)
            if query:
                blocked_queries.add(normalize_query(query))

    unseen_pools, _ = build_candidate_pools(sft_combos, blocked_combos, catalog)
    seen_pools = {
        tool_count: sorted(
            (combo for combo in sft_combos if len(combo) == tool_count),
            key=lambda combo: tuple(sorted(combo)),
        )
        for tool_count in range(1, 7)
    }
    novel_pools = {
        tool_count: sorted(values, key=lambda combo: tuple(sorted(combo)))
        for tool_count, values in unseen_pools.items()
    }

    global_usage: Counter[frozenset[str]] = Counter()
    active_splits = selected_splits(args)
    split_novel: dict[str, set[frozenset[str]]] = {
        split: set() for split in active_splits
    }
    all_rows: list[dict[str, Any]] = []
    plan_paths: dict[str, str] = {}
    for split in active_splits:
        panels = SPLIT_COUNTS[split]
        for panel, counts in panels.items():
            exposure: Counter[str] = Counter()
            rows: list[dict[str, Any]] = []
            local_index = 0
            for tool_count, total in counts.items():
                familiar = seen_rows(panel, tool_count, total)
                buckets = ["sft_seen"] * familiar + ["sft_unseen"] * (total - familiar)
                buckets.sort(
                    key=lambda bucket: stable_seed(
                        args.seed, split, panel, tool_count, bucket, len(buckets)
                    )
                )
                for bucket_index, bucket in enumerate(buckets):
                    pool = (
                        seen_pools[tool_count]
                        if bucket == "sft_seen"
                        else novel_pools[tool_count]
                    )
                    if bucket == "sft_unseen":
                        other_split = "confirmation" if split == "selection" else "selection"
                        if other_split in split_novel:
                            pool = [
                                value
                                for value in pool
                                if value not in split_novel[other_split]
                            ]
                    combo = choose_combo(
                        pool,
                        global_usage,
                        exposure,
                        args.seed,
                        split,
                        panel,
                        tool_count,
                        bucket,
                        bucket_index,
                    )
                    if bucket == "sft_unseen":
                        split_novel[split].add(combo)
                    trace_id = (
                        f"{args.id_prefix}_{split}_{panel}_{local_index:04d}"
                    )
                    ordered = dependency_order(
                        combo, catalog, args.seed, split, panel, local_index
                    )
                    rows.append(
                        {
                            "id": trace_id,
                            "tools": ordered,
                            "tool_count": tool_count,
                            "domain": ",".join(
                                sorted({str(catalog[tool]["domain"]) for tool in ordered})
                            ),
                            "combo_seen_bucket": bucket,
                            "construction": {
                                "type": "planner_joint_depth_representative_dev",
                                "dataset_role": split,
                                "panel": panel,
                                "style_profile": STYLE_BY_PANEL[panel],
                                "source_trace_id": trace_id,
                                "sft_combo_seen": bucket == "sft_seen",
                                "seed": args.seed,
                            },
                        }
                    )
                    local_index += 1
            path = args.output_dir / "plan" / f"{split}_{panel}.jsonl"
            write_jsonl(path, rows)
            plan_paths[f"{split}_{panel}"] = str(path)
            all_rows.extend(rows)

    ids = [str(row["id"]) for row in all_rows]
    source_overlap = set(ids) & sft_source_ids
    dependency_error_rows = [
        {"id": row["id"], "errors": dependency_errors(row["tools"], catalog)}
        for row in all_rows
        if dependency_errors(row["tools"], catalog)
    ]
    expected = sum(
        sum(sum(counts.values()) for counts in panels.values())
        for split, panels in SPLIT_COUNTS.items()
        if split in active_splits
    )
    selection_novel = split_novel.get("selection", set())
    confirmation_novel = split_novel.get("confirmation", set())
    strict_ok = (
        len(all_rows) == expected
        and len(set(ids)) == expected
        and not source_overlap
        and not dependency_error_rows
        and not (selection_novel & confirmation_novel)
        and all(combo not in sft_combos for combo in selection_novel | confirmation_novel)
        and all(combo not in blocked_combos for combo in selection_novel | confirmation_novel)
    )
    audit = {
        "audit_version": "planner-joint-depth-representative-plan-v1",
        "status": "complete" if strict_ok else "failed",
        "strict_ok": strict_ok,
        "training_use_allowed": False,
        "design": {
            "selection_rows": 2400,
            "sealed_confirmation_rows": 1200,
            "generated_roles": list(active_splits),
            "id_prefix": args.id_prefix,
            "panels": {
                "legacy_like": (
                    "50% singleton; multi-tool count and SFT-seen/unseen mixture "
                    "match the declared Legacy construction"
                ),
                "api_like": (
                    "tool counts 1-5 balanced; all combinations sampled from SFT-seen "
                    "classes, matching the API test construction"
                ),
            },
            "singleton_combo_overlap_is_structural": (
                "The closed set has only 100 singleton labels and all occur in SFT; "
                "query and source identity, not label identity, are isolated."
            ),
            "counts": SPLIT_COUNTS,
            "legacy_seen_ratios": LEGACY_SEEN_RATIOS,
        },
        "counts": {
            "rows": len(all_rows),
            "tool_count": dict(sorted(Counter(row["tool_count"] for row in all_rows).items())),
            "panel": dict(
                sorted(Counter(row["construction"]["panel"] for row in all_rows).items())
            ),
            "split": dict(
                sorted(
                    Counter(row["construction"]["dataset_role"] for row in all_rows).items()
                )
            ),
            "combo_seen_bucket": dict(
                sorted(Counter(row["combo_seen_bucket"] for row in all_rows).items())
            ),
            "sft_unique_combinations": len(sft_combos),
            "blocked_normalized_queries": len(blocked_queries),
        },
        "isolation": {
            "source_trace_id_overlap_with_sft": len(source_overlap),
            "selection_confirmation_unseen_combo_overlap": len(
                selection_novel & confirmation_novel
            ),
            "unseen_combo_overlap_with_sft": len(
                (selection_novel | confirmation_novel) & sft_combos
            ),
            "unseen_combo_overlap_with_blocked_eval": len(
                (selection_novel | confirmation_novel) & blocked_combos
            ),
            "dependency_error_count": len(dependency_error_rows),
            "dependency_errors": dependency_error_rows[:50],
        },
        "inputs": {
            str(args.catalog): sha256_file(args.catalog),
            str(args.sft_runtime): sha256_file(args.sft_runtime),
            **{str(path): sha256_file(path) for path in args.blocked_dataset},
        },
        "outputs": {
            key: {"path": path, "sha256": sha256_file(Path(path))}
            for key, path in plan_paths.items()
        },
    }
    write_json(args.output_dir / "plan_audit.json", audit)
    print(f"representative plan rows={len(all_rows)} strict_ok={strict_ok}")
    return 0 if strict_ok else 1


def finalize_stage(args: argparse.Namespace) -> int:
    sft_rows = load_jsonl(args.sft_runtime)
    blocked_queries = {normalize_query(runtime_query(row)) for row in sft_rows}
    for path in args.blocked_dataset:
        for row in load_jsonl(path):
            query = dataset_query(row)
            if query:
                blocked_queries.add(normalize_query(query))

    active_splits = selected_splits(args)
    split_rows: dict[str, list[dict[str, Any]]] = {
        split: [] for split in active_splits
    }
    generation: dict[str, Any] = {}
    alignment_errors: list[str] = []
    for split in active_splits:
        panels = SPLIT_COUNTS[split]
        for panel in panels:
            key = f"{split}_{panel}"
            plan_path = args.output_dir / "plan" / f"{key}.jsonl"
            generation_dir = args.output_dir / "generation" / key
            generated_path = generation_dir / "dev.jsonl"
            audit_path = generation_dir / "generation_audit.json"
            plan = {str(row["id"]): row for row in load_jsonl(plan_path)}
            generated = load_jsonl(generated_path)
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            if not audit.get("strict_ok") or len(generated) != len(plan):
                alignment_errors.append(f"{key}:generation_incomplete")
            for row in generated:
                target = plan.get(str(row["id"]))
                if target is None or split_tools(row["工具"]) != target["tools"]:
                    alignment_errors.append(f"{row.get('id')}:target_mismatch")
                if row.get("style") != STYLE_BY_PANEL[panel]:
                    alignment_errors.append(f"{row.get('id')}:style_mismatch")
            split_rows[split].extend(generated)
            generation[key] = {
                "strict_ok": bool(audit.get("strict_ok")),
                "rows": len(generated),
                "generator_models": audit.get("counts", {}).get(
                    "accepted_generator_model", {}
                ),
                "judge_models": audit.get("counts", {}).get(
                    "accepted_judge_model", {}
                ),
                "audit": str(audit_path),
                "audit_sha256": sha256_file(audit_path),
            }

    all_rows = [row for split in active_splits for row in split_rows[split]]
    normalized = [normalize_query(str(row["问题"])) for row in all_rows]
    source_ids = [str(row["construction"]["source_trace_id"]) for row in all_rows]
    split_query_sets = {
        split: {normalize_query(str(row["问题"])) for row in rows}
        for split, rows in split_rows.items()
    }
    expected = {
        split: count
        for split, count in {"selection": 2400, "confirmation": 1200}.items()
        if split in active_splits
    }
    cross_split_query_overlap = (
        len(split_query_sets["selection"] & split_query_sets["confirmation"])
        if set(("selection", "confirmation")) <= set(active_splits)
        else 0
    )
    strict_ok = (
        all(len(split_rows[split]) == count for split, count in expected.items())
        and len(normalized) == len(set(normalized))
        and not (set(normalized) & blocked_queries)
        and len(source_ids) == len(set(source_ids))
        and cross_split_query_overlap == 0
        and not alignment_errors
        and all(value["strict_ok"] for value in generation.values())
    )
    outputs = {}
    for split, rows in split_rows.items():
        rows.sort(key=lambda row: str(row["id"]))
        path = args.output_dir / f"{split}.jsonl"
        write_jsonl(path, rows)
        outputs[split] = {"path": str(path), "sha256": sha256_file(path)}
    audit = {
        "audit_version": "planner-joint-depth-representative-data-v1",
        "status": "complete" if strict_ok else "failed",
        "strict_ok": strict_ok,
        "training_use_allowed": False,
        "counts": {
            split: {
                "rows": len(rows),
                "panel": dict(
                    sorted(Counter(row["construction"]["panel"] for row in rows).items())
                ),
                "style": dict(sorted(Counter(row["style"] for row in rows).items())),
                "tool_count": dict(
                    sorted(Counter(int(row["tool_count"]) for row in rows).items())
                ),
                "combo_seen_bucket": dict(
                    sorted(Counter(row["combo_seen_bucket"] for row in rows).items())
                ),
            }
            for split, rows in split_rows.items()
        },
        "isolation": {
            "exact_query_overlap_with_sft_or_blocked": len(
                set(normalized) & blocked_queries
            ),
            "selection_confirmation_exact_query_overlap": cross_split_query_overlap,
            "duplicate_normalized_queries": len(normalized) - len(set(normalized)),
            "duplicate_source_trace_ids": len(source_ids) - len(set(source_ids)),
            "alignment_error_count": len(alignment_errors),
            "alignment_errors": alignment_errors[:50],
        },
        "generation_evidence": generation,
        "outputs": outputs,
    }
    write_json(args.output_dir / "data_audit.json", audit)
    print(
        "representative data "
        + " ".join(f"{split}={len(split_rows[split])}" for split in active_splits)
        + f" strict_ok={strict_ok}"
    )
    return 0 if strict_ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("plan", "finalize"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    parser.add_argument("--sft-runtime", type=Path, required=True)
    parser.add_argument("--blocked-dataset", action="append", type=Path, default=[])
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument(
        "--dataset-role",
        choices=("selection", "confirmation", "both"),
        default="both",
        help="Generate both historical roles by default, or only one isolated role.",
    )
    parser.add_argument(
        "--id-prefix",
        default="planner_joint",
        help="Stable source/id namespace for independently generated snapshots.",
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_]+", args.id_prefix):
        raise SystemExit("--id-prefix must contain only letters, digits, and underscores")
    return plan_stage(args) if args.stage == "plan" else finalize_stage(args)


if __name__ == "__main__":
    raise SystemExit(main())
