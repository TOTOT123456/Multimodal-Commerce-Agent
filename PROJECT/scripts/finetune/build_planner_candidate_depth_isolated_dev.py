#!/usr/bin/env python3
"""Build and finalize a Planner-K dev set whose tool sets are unseen by SFT."""

from __future__ import annotations

import argparse
import itertools
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
from finetune.tool_trace_common import sha256_file, stable_seed  # noqa: E402


STYLES = ("api_native", "legacy_compact")
TOOL_COUNTS = (2, 3, 4, 5, 6)


def normalize_query(value: str) -> str:
    return re.sub(r"[\W_]+", "", value.casefold())


def runtime_target(row: dict[str, Any]) -> tuple[str, ...]:
    assistant = next(
        message for message in row["messages"] if message.get("role") == "assistant"
    )
    payload = json.loads(assistant["content"])
    tools = payload.get("tools")
    if not isinstance(tools, list) or not tools:
        raise ValueError("invalid SFT assistant target")
    return tuple(str(value) for value in tools)


def runtime_query(row: dict[str, Any]) -> str:
    user = next(message for message in row["messages"] if message.get("role") == "user")
    content = str(user["content"])
    marker = "用户问题：\n"
    if marker not in content:
        raise ValueError("SFT user prompt lacks query marker")
    return content.rsplit(marker, 1)[1].strip()


def dataset_query(row: dict[str, Any]) -> str | None:
    for field in ("问题", "question", "query"):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def dataset_combo(row: dict[str, Any]) -> frozenset[str] | None:
    value = row.get("工具", row.get("tools", row.get("gold_tools")))
    if value is None:
        return None
    tools = split_tools(value)
    return frozenset(tools) if tools else None


def dependency_order(
    tools: frozenset[str],
    catalog: dict[str, dict[str, Any]],
    *seed_parts: object,
) -> list[str]:
    remaining = set(tools)
    ordered: list[str] = []
    step = 0
    while remaining:
        ready = [
            tool_id
            for tool_id in remaining
            if not (set(catalog[tool_id].get("depends_on", [])) & remaining)
        ]
        if not ready:
            raise ValueError(f"dependency cycle in {sorted(remaining)}")
        ready.sort(key=lambda tool_id: stable_seed(*seed_parts, step, tool_id))
        selected = ready[0]
        ordered.append(selected)
        remaining.remove(selected)
        step += 1
    return ordered


def dependency_errors(
    tools: list[str], catalog: dict[str, dict[str, Any]]
) -> list[str]:
    positions = {tool_id: index for index, tool_id in enumerate(tools)}
    return [
        f"{dependency}->{tool_id}"
        for tool_id in tools
        for dependency in catalog[tool_id].get("depends_on", [])
        if dependency in positions and positions[dependency] > positions[tool_id]
    ]


def confusable_pair(
    left: str, right: str, catalog: dict[str, dict[str, Any]]
) -> bool:
    return right in catalog[left].get("confusable_tools", []) or left in catalog[
        right
    ].get("confusable_tools", [])


def build_candidate_pools(
    sft_combos: set[frozenset[str]],
    forbidden_combos: set[frozenset[str]],
    catalog: dict[str, dict[str, Any]],
) -> tuple[dict[int, set[frozenset[str]]], Counter[frozenset[str]]]:
    pair_counts: Counter[frozenset[str]] = Counter()
    by_count: dict[int, set[frozenset[str]]] = {value: set() for value in TOOL_COUNTS}
    for combo in sft_combos:
        for pair in itertools.combinations(combo, 2):
            pair_counts[frozenset(pair)] += 1
        if len(combo) in by_count:
            by_count[len(combo)].add(combo)

    all_tools = sorted(catalog)
    pools: dict[int, set[frozenset[str]]] = {value: set() for value in TOOL_COUNTS}
    blocked = sft_combos | forbidden_combos
    for tool_count in TOOL_COUNTS:
        for base in by_count[tool_count]:
            for removed in base:
                retained = base - {removed}
                for added in all_tools:
                    candidate = retained | {added}
                    if len(candidate) != tool_count or candidate in blocked:
                        continue
                    if any(
                        confusable_pair(left, right, catalog)
                        for left, right in itertools.combinations(candidate, 2)
                    ):
                        continue
                    if tool_count >= 3 and any(
                        pair_counts[frozenset((added, other))] == 0
                        for other in retained
                    ):
                        continue
                    pools[tool_count].add(frozenset(candidate))
    return pools, pair_counts


def coherence_score(
    combo: frozenset[str],
    pair_counts: Counter[frozenset[str]],
    catalog: dict[str, dict[str, Any]],
) -> tuple[int, int, int]:
    pairs = [frozenset(pair) for pair in itertools.combinations(combo, 2)]
    observed = [pair_counts[pair] for pair in pairs]
    same_domain = sum(
        catalog[left]["domain"] == catalog[right]["domain"]
        for left, right in itertools.combinations(combo, 2)
    )
    return min(observed, default=0), sum(observed), same_domain


def select_combos(
    pools: dict[int, set[frozenset[str]]],
    pair_counts: Counter[frozenset[str]],
    catalog: dict[str, dict[str, Any]],
    per_count: int,
    seed: int,
) -> dict[int, list[frozenset[str]]]:
    exposure: Counter[str] = Counter()
    selected: dict[int, list[frozenset[str]]] = {}
    used: set[frozenset[str]] = set()
    # Select the most constrained large combinations first; the abundant
    # two-tool pool can then repair global tool-exposure balance.
    for tool_count in reversed(TOOL_COUNTS):
        remaining = set(pools[tool_count]) - used
        chosen: list[frozenset[str]] = []
        for local_index in range(per_count):
            if not remaining:
                raise ValueError(f"insufficient unseen combinations for k={tool_count}")
            combo = min(
                remaining,
                key=lambda value: (
                    max(exposure[tool_id] + 1 for tool_id in value),
                    sum((exposure[tool_id] + 1) ** 2 for tool_id in value),
                    tuple(-part for part in coherence_score(value, pair_counts, catalog)),
                    stable_seed(seed, "planner-k-isolated", tool_count, local_index, *sorted(value)),
                ),
            )
            remaining.remove(combo)
            chosen.append(combo)
            used.add(combo)
            exposure.update(combo)
        selected[tool_count] = chosen
    return selected


def plan_stage(args: argparse.Namespace) -> int:
    catalog_rows = load_catalog(args.catalog)
    catalog = catalog_by_id(catalog_rows)
    sft_rows = load_jsonl(args.sft_runtime)
    sft_targets = [runtime_target(row) for row in sft_rows]
    sft_combos = {frozenset(target) for target in sft_targets}
    sft_source_ids = {str(row["meta"]["trace_id"]) for row in sft_rows}
    sft_queries = {normalize_query(runtime_query(row)) for row in sft_rows}

    forbidden_combos: set[frozenset[str]] = set()
    blocked_dataset_audits: dict[str, dict[str, int]] = {}
    for path in args.blocked_dataset:
        rows = load_jsonl(path)
        source_ids: set[str] = set()
        queries: set[str] = set()
        combos: set[frozenset[str]] = set()
        for row in rows:
            combo = dataset_combo(row)
            if combo:
                forbidden_combos.add(combo)
                combos.add(combo)
            query = dataset_query(row)
            if query:
                queries.add(normalize_query(query))
            construction = row.get("construction", {})
            source_id = (
                construction.get("source_trace_id")
                if isinstance(construction, dict)
                else None
            )
            if source_id is not None:
                source_ids.add(str(source_id))
        blocked_dataset_audits[str(path)] = {
            "rows": len(rows),
            "unique_source_trace_ids_when_declared": len(source_ids),
            "source_trace_id_overlap_with_sft": len(source_ids & sft_source_ids),
            "exact_query_overlap_with_sft": len(queries & sft_queries),
            "unique_tool_combinations": len(combos),
            "tool_combination_overlap_with_sft": len(combos & sft_combos),
        }

    pools, pair_counts = build_candidate_pools(sft_combos, forbidden_combos, catalog)
    selected = select_combos(
        pools, pair_counts, catalog, args.per_count, args.seed
    )

    base_rows: list[dict[str, Any]] = []
    style_rows: dict[str, list[dict[str, Any]]] = {style: [] for style in STYLES}
    for tool_count in TOOL_COUNTS:
        for local_index, combo in enumerate(selected[tool_count]):
            combo_id = f"planner_k_iso_combo_{tool_count}_{local_index:03d}"
            tools = dependency_order(combo, catalog, args.seed, combo_id)
            domains = sorted({str(catalog[tool_id]["domain"]) for tool_id in tools})
            base_rows.append(
                {
                    "id": combo_id,
                    "tools": tools,
                    "tool_count": tool_count,
                    "domain": ",".join(domains),
                    "combo_seen_bucket": "sft_combo_unseen",
                }
            )
            for style in STYLES:
                trace_id = f"{combo_id}_{style}"
                style_rows[style].append(
                    {
                        "id": trace_id,
                        "tools": tools,
                        "tool_count": tool_count,
                        "domain": ",".join(domains),
                        "combo_seen_bucket": "sft_combo_unseen",
                        "style": style,
                        "construction": {
                            "type": "planner_k_sft_combo_disjoint_dev",
                            "dataset_role": "dev",
                            "style_profile": style,
                            "source_trace_id": trace_id,
                            "novel_combo_id": combo_id,
                            "source": "catalog_combinatorial_holdout",
                            "sft_combo_overlap": False,
                            "blocked_eval_combo_overlap": False,
                            "seed": args.seed,
                        },
                    }
                )

    output_dir = args.output_dir
    plan_dir = output_dir / "plan"
    plan_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(plan_dir / "base_combos.jsonl", base_rows)
    for style in STYLES:
        write_jsonl(plan_dir / f"{style}.jsonl", style_rows[style])

    all_style_rows = [row for style in STYLES for row in style_rows[style]]
    selected_combos = {frozenset(row["tools"]) for row in base_rows}
    selected_source_ids = {
        str(row["construction"]["source_trace_id"]) for row in all_style_rows
    }
    dependency_error_rows = [
        {"id": row["id"], "errors": dependency_errors(row["tools"], catalog)}
        for row in base_rows
        if dependency_errors(row["tools"], catalog)
    ]
    exposure = Counter(tool_id for row in base_rows for tool_id in row["tools"])
    pairwise_unseen = sum(
        any(
            pair_counts[frozenset(pair)] == 0
            for pair in itertools.combinations(row["tools"], 2)
        )
        for row in base_rows
        if row["tool_count"] >= 3
    )
    expected_base = args.per_count * len(TOOL_COUNTS)
    expected_total = expected_base * len(STYLES)
    strict_ok = (
        len(base_rows) == expected_base
        and len(selected_combos) == expected_base
        and len(all_style_rows) == expected_total
        and len(selected_source_ids) == expected_total
        and not (selected_source_ids & sft_source_ids)
        and not (selected_combos & sft_combos)
        and not (selected_combos & forbidden_combos)
        and not dependency_error_rows
        and not pairwise_unseen
        and len(exposure) == len(catalog)
    )
    audit = {
        "audit_version": "planner-k-sft-combo-disjoint-plan-v1",
        "status": "complete" if strict_ok else "failed",
        "strict_ok": strict_ok,
        "training_use_allowed": False,
        "design": {
            "styles": list(STYLES),
            "tool_counts": list(TOOL_COUNTS),
            "per_tool_count_per_style": args.per_count,
            "base_combinations": expected_base,
            "total_queries": expected_total,
            "singleton_exclusion": (
                "All 100 singleton tool sets occur in SFT, so strict combination "
                "disjointness is impossible for tool_count=1."
            ),
            "same_novel_combo_rendered_once_per_style": True,
            "candidate_construction": (
                "one-tool mutation of an SFT combination; exact set unseen; for "
                "tool_count>=3 every constituent pair was observed in SFT"
            ),
        },
        "counts": {
            "sft_rows": len(sft_rows),
            "sft_unique_source_trace_ids": len(sft_source_ids),
            "sft_unique_tool_combinations": len(sft_combos),
            "blocked_unique_tool_combinations": len(forbidden_combos),
            "base_rows": len(base_rows),
            "style_rows": {style: len(rows) for style, rows in style_rows.items()},
            "tool_count": dict(sorted(Counter(row["tool_count"] for row in base_rows).items())),
            "tool_coverage": len(exposure),
            "tool_exposure_min": min(exposure.values()),
            "tool_exposure_max": max(exposure.values()),
            "pool_sizes": {str(key): len(value) for key, value in pools.items()},
        },
        "isolation": {
            "source_trace_id_overlap_with_sft": len(selected_source_ids & sft_source_ids),
            "tool_combination_overlap_with_sft": len(selected_combos & sft_combos),
            "tool_combination_overlap_with_blocked_eval": len(
                selected_combos & forbidden_combos
            ),
            "pairwise_unseen_rows_for_tool_count_ge_3": pairwise_unseen,
            "dependency_error_count": len(dependency_error_rows),
            "dependency_errors": dependency_error_rows[:50],
        },
        "blocked_dataset_diagnostics": blocked_dataset_audits,
        "inputs": {
            str(args.catalog): sha256_file(args.catalog),
            str(args.sft_runtime): sha256_file(args.sft_runtime),
            **{str(path): sha256_file(path) for path in args.blocked_dataset},
        },
        "outputs": {
            str(plan_dir / "base_combos.jsonl"): sha256_file(
                plan_dir / "base_combos.jsonl"
            ),
            **{
                str(plan_dir / f"{style}.jsonl"): sha256_file(
                    plan_dir / f"{style}.jsonl"
                )
                for style in STYLES
            },
        },
    }
    write_json(output_dir / "plan_audit.json", audit)
    print(
        f"isolated plan base={len(base_rows)} queries={len(all_style_rows)} "
        f"strict_ok={strict_ok}"
    )
    return 0 if strict_ok else 1


def finalize_stage(args: argparse.Namespace) -> int:
    catalog = catalog_by_id(load_catalog(args.catalog))
    sft_rows = load_jsonl(args.sft_runtime)
    sft_source_ids = {str(row["meta"]["trace_id"]) for row in sft_rows}
    sft_combos = {frozenset(runtime_target(row)) for row in sft_rows}
    sft_queries = {normalize_query(runtime_query(row)) for row in sft_rows}

    blocked_queries = set(sft_queries)
    blocked_combos: set[frozenset[str]] = set()
    for path in args.blocked_dataset:
        for row in load_jsonl(path):
            query = dataset_query(row)
            if query:
                blocked_queries.add(normalize_query(query))
            combo = dataset_combo(row)
            if combo:
                blocked_combos.add(combo)

    plan_dir = args.output_dir / "plan"
    generated_rows: list[dict[str, Any]] = []
    alignment_errors: list[str] = []
    generation_evidence: dict[str, dict[str, Any]] = {}
    for style in STYLES:
        plan_rows = load_jsonl(plan_dir / f"{style}.jsonl")
        plan_by_id = {str(row["id"]): row for row in plan_rows}
        generated_path = args.output_dir / "generation" / style / "dev.jsonl"
        rows = load_jsonl(generated_path)
        accepted_path = (
            args.output_dir / "generation" / style / "accepted_generation.jsonl"
        )
        generation_audit_path = (
            args.output_dir / "generation" / style / "generation_audit.json"
        )
        accepted_rows = load_jsonl(accepted_path)
        generation_audit = json.loads(
            generation_audit_path.read_text(encoding="utf-8")
        )
        generation_evidence[style] = {
            "strict_ok": bool(generation_audit.get("strict_ok")),
            "accepted_rows": len(accepted_rows),
            "generator_model": dict(
                sorted(
                    Counter(
                        str(value.get("generator_model"))
                        for value in accepted_rows
                    ).items()
                )
            ),
            "judge_model": dict(
                sorted(
                    Counter(
                        str(value.get("judge_model")) for value in accepted_rows
                    ).items()
                )
            ),
        }
        if (
            not generation_audit.get("strict_ok")
            or {str(row["id"]) for row in accepted_rows} != set(plan_by_id)
        ):
            alignment_errors.append(f"{style}:generation_evidence")
        if {str(row["id"]) for row in rows} != set(plan_by_id):
            alignment_errors.append(f"{style}:id_set")
        for row in rows:
            trace_id = str(row["id"])
            plan = plan_by_id.get(trace_id)
            if plan is None or split_tools(row["工具"]) != list(plan["tools"]):
                alignment_errors.append(f"{trace_id}:tools")
            if row.get("style") != style:
                alignment_errors.append(f"{trace_id}:style")
        generated_rows.extend(rows)

    generated_rows.sort(
        key=lambda row: (
            str(row["construction"]["novel_combo_id"]),
            STYLES.index(str(row["style"])),
        )
    )
    queries = [normalize_query(str(row["问题"])) for row in generated_rows]
    source_ids = {
        str(row["construction"]["source_trace_id"]) for row in generated_rows
    }
    combos = [frozenset(split_tools(row["工具"])) for row in generated_rows]
    unique_combos = set(combos)
    combo_counts = Counter(combos)
    dependency_error_rows = [
        {"id": row["id"], "errors": dependency_errors(split_tools(row["工具"]), catalog)}
        for row in generated_rows
        if dependency_errors(split_tools(row["工具"]), catalog)
    ]
    expected_base = args.per_count * len(TOOL_COUNTS)
    expected_total = expected_base * len(STYLES)
    strict_ok = (
        len(generated_rows) == expected_total
        and len(set(queries)) == expected_total
        and not (set(queries) & blocked_queries)
        and len(source_ids) == expected_total
        and not (source_ids & sft_source_ids)
        and len(unique_combos) == expected_base
        and all(count == len(STYLES) for count in combo_counts.values())
        and not (unique_combos & sft_combos)
        and not (unique_combos & blocked_combos)
        and not alignment_errors
        and not dependency_error_rows
    )
    dev_path = args.output_dir / "dev.jsonl"
    write_jsonl(dev_path, generated_rows)
    audit = {
        "audit_version": "planner-k-sft-combo-disjoint-data-v1",
        "status": "complete" if strict_ok else "failed",
        "strict_ok": strict_ok,
        "training_use_allowed": False,
        "counts": {
            "rows": len(generated_rows),
            "unique_queries": len(set(queries)),
            "unique_source_trace_ids": len(source_ids),
            "unique_tool_combinations": len(unique_combos),
            "style": dict(sorted(Counter(str(row["style"]) for row in generated_rows).items())),
            "tool_count": dict(
                sorted(Counter(int(row["tool_count"]) for row in generated_rows).items())
            ),
        },
        "isolation": {
            "exact_query_overlap_with_sft_or_blocked": len(
                set(queries) & blocked_queries
            ),
            "source_trace_id_overlap_with_sft": len(source_ids & sft_source_ids),
            "tool_combination_overlap_with_sft": len(unique_combos & sft_combos),
            "tool_combination_overlap_with_blocked_eval": len(
                unique_combos & blocked_combos
            ),
            "each_combo_has_two_style_renderings": all(
                count == len(STYLES) for count in combo_counts.values()
            ),
            "alignment_error_count": len(alignment_errors),
            "alignment_errors": alignment_errors[:50],
            "dependency_error_count": len(dependency_error_rows),
            "dependency_errors": dependency_error_rows[:50],
        },
        "generation_evidence": generation_evidence,
        "inputs": {
            str(args.catalog): sha256_file(args.catalog),
            str(args.sft_runtime): sha256_file(args.sft_runtime),
            **{str(path): sha256_file(path) for path in args.blocked_dataset},
            **{
                str(args.output_dir / "generation" / style / "dev.jsonl"): sha256_file(
                    args.output_dir / "generation" / style / "dev.jsonl"
                )
                for style in STYLES
            },
            **{
                str(
                    args.output_dir
                    / "generation"
                    / style
                    / "accepted_generation.jsonl"
                ): sha256_file(
                    args.output_dir
                    / "generation"
                    / style
                    / "accepted_generation.jsonl"
                )
                for style in STYLES
            },
            **{
                str(
                    args.output_dir
                    / "generation"
                    / style
                    / "generation_audit.json"
                ): sha256_file(
                    args.output_dir
                    / "generation"
                    / style
                    / "generation_audit.json"
                )
                for style in STYLES
            },
        },
        "outputs": {str(dev_path): sha256_file(dev_path)},
    }
    write_json(args.output_dir / "data_audit.json", audit)
    print(
        f"isolated dev rows={len(generated_rows)} combos={len(unique_combos)} "
        f"strict_ok={strict_ok}"
    )
    return 0 if strict_ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("plan", "finalize"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    parser.add_argument("--sft-runtime", type=Path, required=True)
    parser.add_argument("--blocked-dataset", action="append", type=Path, default=[])
    parser.add_argument("--per-count", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260801)
    args = parser.parse_args()
    if args.per_count <= 0:
        raise SystemExit("--per-count must be positive")
    return plan_stage(args) if args.stage == "plan" else finalize_stage(args)


if __name__ == "__main__":
    raise SystemExit(main())
