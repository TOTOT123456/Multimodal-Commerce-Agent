#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the query-free 100-tool Assignment3 SFT/test target plans."""
from __future__ import annotations

import argparse
import copy
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import (  # noqa: E402
    CATALOG_PATH,
    TOOL_TRACE_DIR,
    catalog_by_id,
    load_catalog,
    load_jsonl,
    split_tools,
    write_json,
    write_jsonl,
)
from finetune.tool_trace_common import sha256_file, stable_seed  # noqa: E402


def randomized_topological_order(
    tools: list[str],
    by_id: dict[str, dict[str, Any]],
    *seed_parts: object,
) -> list[str]:
    """Randomize independent tools while preserving catalog dependencies."""
    values = list(dict.fromkeys(tools))
    value_set = set(values)
    incoming = {
        tool_id: {
            dependency
            for dependency in by_id[tool_id].get("depends_on", [])
            if dependency in value_set
        }
        for tool_id in values
    }
    result: list[str] = []
    remaining = set(values)
    step = 0
    while remaining:
        ready = [tool_id for tool_id in remaining if not (incoming[tool_id] & remaining)]
        if not ready:
            raise ValueError(f"dependency cycle inside target tools: {sorted(remaining)}")
        ready.sort(key=lambda tool_id: stable_seed(*seed_parts, step, tool_id))
        chosen = ready[0]
        result.append(chosen)
        remaining.remove(chosen)
        step += 1
    return result


def dependency_violations(
    tools: list[str],
    by_id: dict[str, dict[str, Any]],
) -> list[str]:
    position = {tool_id: index for index, tool_id in enumerate(tools)}
    return [
        f"{dependency}->{tool_id}"
        for tool_id in tools
        for dependency in by_id[tool_id].get("depends_on", [])
        if dependency in position and position[dependency] > position[tool_id]
    ]


def make_plan_row(
    trace_id: str,
    tools: list[str],
    by_id: dict[str, dict[str, Any]],
    construction: dict[str, Any],
) -> dict[str, Any]:
    domains = sorted({by_id[tool_id]["domain"] for tool_id in tools})
    return {
        "id": trace_id,
        "tools": tools,
        "tool_count": len(tools),
        "domain": ",".join(domains),
        "combo_seen_bucket": construction["type"],
        "construction": construction,
    }


def build_train_plan(
    train_rows: list[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
    single_count: int,
    order_count: int,
    augmented_count: int,
) -> list[dict[str, Any]]:
    singles_by_tool: dict[str, list[dict[str, Any]]] = {tool_id: [] for tool_id in by_id}
    multi_rows = []
    augment_bases = []
    for row in train_rows:
        tools = split_tools(row["工具"])
        if len(tools) == 1:
            singles_by_tool[tools[0]].append(row)
        else:
            multi_rows.append(row)
        if 2 <= len(tools) <= 4:
            augment_bases.append(row)
    if any(not values for values in singles_by_tool.values()):
        raise SystemExit("formal train does not cover every tool with a single-intent row")
    if single_count % len(by_id):
        raise SystemExit("single count must be divisible by catalog size")

    output: list[dict[str, Any]] = []
    per_tool_single = single_count // len(by_id)
    for tool_index, tool_id in enumerate(sorted(by_id)):
        sources = sorted(
            singles_by_tool[tool_id],
            key=lambda row: stable_seed("a3-100-single-source", row["id"]),
        )
        for item_index in range(per_tool_single):
            source = sources[item_index % len(sources)]
            variant = item_index // len(sources)
            trace_id = f"a3_100_sft_single_{tool_index:03d}_{item_index:03d}"
            distractor_count = 1 + stable_seed(trace_id, "distractors") % 4
            same_domain = [
                candidate_id
                for candidate_id, candidate in by_id.items()
                if candidate["domain"] == by_id[tool_id]["domain"]
                and candidate_id != tool_id
            ]
            same_domain.sort(
                key=lambda candidate_id: stable_seed(
                    trace_id, "single-distractor", candidate_id
                )
            )
            distractors = same_domain[:distractor_count]
            candidates = [tool_id] + distractors
            candidates.sort(
                key=lambda candidate_id: stable_seed(
                    trace_id, "single-candidate-order", candidate_id
                )
            )
            output.append(
                make_plan_row(
                    trace_id,
                    [tool_id],
                    by_id,
                    {
                        "type": "single_intent",
                        "source_trace_id": source["id"],
                        "source_tools": [tool_id],
                        "intent_label": by_id[tool_id]["keyword_label"],
                        "api_variant": variant,
                        "candidate_tools": candidates,
                        "distractor_tools": distractors,
                        "same_type_distractor_count": distractor_count,
                    },
                )
            )

    ordered_sources = sorted(
        multi_rows,
        key=lambda row: stable_seed("a3-100-order-source", row["id"]),
    )
    order_rows = []
    round_index = 0
    while len(order_rows) < order_count:
        accepted_this_round = 0
        for source in ordered_sources:
            source_tools = split_tools(source["工具"])
            gold_tools = randomized_topological_order(
                source_tools,
                by_id,
                "a3-100-gold-order",
                round_index,
                source["id"],
            )
            candidate_tools = sorted(
                gold_tools,
                key=lambda tool_id: stable_seed(
                    "a3-100-candidate-order",
                    round_index,
                    source["id"],
                    tool_id,
                ),
            )
            if candidate_tools == gold_tools:
                candidate_tools = list(reversed(candidate_tools))
            item_index = len(order_rows)
            trace_id = f"a3_100_sft_order_{item_index:05d}"
            order_rows.append(
                make_plan_row(
                    trace_id,
                    gold_tools,
                    by_id,
                    {
                        "type": "multi_candidate_order_shuffled",
                        "source_trace_id": source["id"],
                        "source_tools": source_tools,
                        "candidate_tools": candidate_tools,
                        "order_changed": True,
                        "permutation_round": round_index,
                    },
                )
            )
            accepted_this_round += 1
            if len(order_rows) == order_count:
                break
        if not accepted_this_round:
            raise SystemExit("unable to construct additional valid order permutations")
        round_index += 1
    output.extend(order_rows)

    ordered_bases = sorted(
        augment_bases,
        key=lambda row: stable_seed("a3-100-augment-source", row["id"]),
    )
    distractor_counts: Counter[str] = Counter()
    seen_augmented: set[tuple[str, tuple[str, ...]]] = set()
    augmented_rows = []
    round_index = 0
    while len(augmented_rows) < augmented_count:
        accepted_this_round = 0
        for source in ordered_bases:
            base_tools = split_tools(source["工具"])
            capacity = min(3, 5 - len(base_tools))
            if capacity <= 0:
                continue
            trace_index = len(augmented_rows)
            trace_id = f"a3_100_sft_augmented_{trace_index:05d}"
            add_count = 1 + stable_seed(trace_id, "add-count") % capacity
            anchor = base_tools[
                stable_seed(trace_id, "anchor") % len(base_tools)
            ]
            anchor_domain = by_id[anchor]["domain"]
            preferred = [
                tool_id
                for tool_id in by_id[anchor].get("confusable_tools", [])
                if tool_id not in base_tools and by_id[tool_id]["domain"] == anchor_domain
            ]
            same_domain = [
                tool_id
                for tool_id, tool in by_id.items()
                if tool["domain"] == anchor_domain and tool_id not in base_tools
            ]
            pool = list(dict.fromkeys(preferred + same_domain))
            pool.sort(
                key=lambda tool_id: (
                    distractor_counts[tool_id],
                    stable_seed(trace_id, "added-tool", tool_id),
                )
            )
            added = pool[:add_count]
            if len(added) != add_count:
                continue
            gold_tools = randomized_topological_order(
                base_tools,
                by_id,
                "a3-100-augmented-gold-order",
                round_index,
                source["id"],
                trace_index,
            )
            candidate_tools = gold_tools + added
            candidate_tools.sort(
                key=lambda tool_id: stable_seed(
                    trace_id, "augmented-candidate-order", tool_id
                )
            )
            signature = (source["id"], tuple(candidate_tools))
            if signature in seen_augmented:
                continue
            seen_augmented.add(signature)
            for tool_id in added:
                distractor_counts[tool_id] += 1
            augmented_rows.append(
                make_plan_row(
                    trace_id,
                    gold_tools,
                    by_id,
                    {
                        "type": "multi_same_type_distractors",
                        "source_trace_id": source["id"],
                        "base_tools": base_tools,
                        "candidate_tools": candidate_tools,
                        "distractor_tools": added,
                        "distractor_count": len(added),
                        "anchor_tool": anchor,
                        "anchor_domain": anchor_domain,
                        "max_candidate_tools": 5,
                        "augmentation_round": round_index,
                    },
                )
            )
            accepted_this_round += 1
            if len(augmented_rows) == augmented_count:
                break
        if not accepted_this_round:
            raise SystemExit("unable to construct additional same-type augmentations")
        round_index += 1
    output.extend(augmented_rows)
    return output


def build_test_plan(
    train_rows: list[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
    test_count: int,
) -> list[dict[str, Any]]:
    if test_count % 5:
        raise SystemExit("test count must be divisible by five")
    by_count: dict[int, list[dict[str, Any]]] = {value: [] for value in range(1, 6)}
    for row in train_rows:
        tools = split_tools(row["工具"])
        if len(tools) in by_count:
            by_count[len(tools)].append(row)
    output = []
    exposure: Counter[str] = Counter()
    each = test_count // 5
    for tool_count in range(1, 6):
        sources = sorted(
            by_count[tool_count],
            key=lambda row: stable_seed("a3-100-test-source", tool_count, row["id"]),
        )
        if len(sources) < each:
            raise SystemExit(f"no source targets for test tool_count={tool_count}")
        remaining = list(sources)
        selected = []
        for local_index in range(each):
            source = min(
                remaining,
                key=lambda row: (
                    max(exposure[tool_id] + 1 for tool_id in split_tools(row["工具"])),
                    sum(
                        (exposure[tool_id] + 1) ** 2
                        for tool_id in split_tools(row["工具"])
                    ),
                    stable_seed(
                        "a3-100-test-balance",
                        tool_count,
                        local_index,
                        row["id"],
                    ),
                ),
            )
            remaining.remove(source)
            selected.append(source)
            exposure.update(split_tools(source["工具"]))
        for local_index, source in enumerate(selected):
            source_tools = split_tools(source["工具"])
            if tool_count > 1:
                tools = randomized_topological_order(
                    source_tools,
                    by_id,
                    "a3-100-test-order",
                    tool_count,
                    local_index,
                    source["id"],
                )
            else:
                tools = source_tools
            trace_id = f"a3_100_test_{tool_count}_{local_index:04d}"
            output.append(
                make_plan_row(
                    trace_id,
                    tools,
                    by_id,
                    {
                        "type": "heldout_api_test",
                        "source_trace_id": source["id"],
                        "source_tools": source_tools,
                        "tool_count_stratum": tool_count,
                    },
                )
            )
    return output


def isolate_source_traces(
    train_plan: list[dict[str, Any]],
    test_plan: list[dict[str, Any]],
    formal_train: list[dict[str, Any]],
) -> dict[str, Any]:
    """Reserve every test source trace and retarget train provenance.

    If an unreserved formal row has the same ordered tools, only provenance
    metadata changes. Source queries are never exposed to API generation, so
    the generated query remains valid. Rare source-unique rows borrow a
    complete valid target from the same train construction and are regenerated
    by the downstream target-hash gate.
    """
    formal_by_id = {row["id"]: row for row in formal_train}
    reserved = {
        row["construction"]["source_trace_id"]
        for row in test_plan
    }
    exact_alternatives: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in formal_train:
        if row["id"] in reserved:
            continue
        exact_alternatives.setdefault(tuple(split_tools(row["工具"])), []).append(row)
    donors_by_type: dict[str, list[dict[str, Any]]] = {}
    for row in train_plan:
        if row["construction"]["source_trace_id"] in reserved:
            continue
        donors_by_type.setdefault(row["construction"]["type"], []).append(row)

    metadata_retargets = []
    semantic_retargets = []
    for row in train_plan:
        old_source_id = row["construction"]["source_trace_id"]
        if old_source_id not in reserved:
            continue
        source_tools = tuple(split_tools(formal_by_id[old_source_id]["工具"]))
        alternatives = exact_alternatives.get(source_tools, [])
        if alternatives:
            replacement = min(
                alternatives,
                key=lambda item: stable_seed(
                    "a3-100-source-isolation",
                    row["id"],
                    item["id"],
                ),
            )
            row["construction"]["source_trace_id"] = replacement["id"]
            metadata_retargets.append(
                {
                    "id": row["id"],
                    "old_source_trace_id": old_source_id,
                    "new_source_trace_id": replacement["id"],
                }
            )
            continue

        construction_type = row["construction"]["type"]
        donors = donors_by_type.get(construction_type, [])
        if not donors:
            raise SystemExit(f"no unreserved donor for {construction_type}")
        donor = min(
            donors,
            key=lambda item: stable_seed(
                "a3-100-source-isolation-donor",
                row["id"],
                item["id"],
            ),
        )
        old_tools = list(row["tools"])
        row["tools"] = list(donor["tools"])
        row["tool_count"] = donor["tool_count"]
        row["domain"] = donor["domain"]
        row["combo_seen_bucket"] = donor["combo_seen_bucket"]
        row["construction"] = copy.deepcopy(donor["construction"])
        semantic_retargets.append(
            {
                "id": row["id"],
                "old_source_trace_id": old_source_id,
                "new_source_trace_id": row["construction"]["source_trace_id"],
                "old_tools": old_tools,
                "new_tools": row["tools"],
                "donor_plan_id": donor["id"],
            }
        )

    train_sources = {
        row["construction"]["source_trace_id"]
        for row in train_plan
    }
    test_sources = {
        row["construction"]["source_trace_id"]
        for row in test_plan
    }
    return {
        "strict_ok": not (train_sources & test_sources),
        "reserved_test_source_trace_ids": len(reserved),
        "train_unique_source_trace_ids": len(train_sources),
        "test_unique_source_trace_ids": len(test_sources),
        "source_trace_id_overlap": len(train_sources & test_sources),
        "metadata_only_retargets": len(metadata_retargets),
        "semantic_retargets_requiring_api_regeneration": len(semantic_retargets),
        "metadata_retarget_examples": metadata_retargets[:20],
        "semantic_retarget_examples": semantic_retargets[:20],
    }


def audit_plan(
    rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
    expected: dict[str, int],
) -> dict[str, Any]:
    source_ids = {row["id"] for row in train_rows}
    ids = [row["id"] for row in rows]
    type_counts = Counter(row["construction"]["type"] for row in rows)
    tool_counts = Counter()
    dependency_error_rows = []
    construction_errors = []
    order_rows = [
        row
        for row in rows
        if row["construction"]["type"] == "multi_candidate_order_shuffled"
    ]
    for row in rows:
        tools = row["tools"]
        tool_counts.update(tools)
        errors = dependency_violations(tools, by_id)
        if errors:
            dependency_error_rows.append({"id": row["id"], "errors": errors})
        construction = row["construction"]
        kind = construction["type"]
        if construction["source_trace_id"] not in source_ids:
            construction_errors.append(f"{row['id']}:unknown_source")
        if kind == "single_intent":
            candidates = construction["candidate_tools"]
            distractors = construction["distractor_tools"]
            if (
                len(tools) != 1
                or not 1 <= len(distractors) <= 4
                or len(candidates) != len(distractors) + 1
                or not set(tools) <= set(candidates)
            ):
                construction_errors.append(f"{row['id']}:single_candidate_size")
            if any(
                by_id[tool_id]["domain"] != by_id[tools[0]]["domain"]
                for tool_id in distractors
            ):
                construction_errors.append(f"{row['id']}:single_distractor_domain")
        if kind == "multi_candidate_order_shuffled":
            candidates = construction["candidate_tools"]
            if (
                set(candidates) != set(tools)
                or candidates == tools
                or not construction["order_changed"]
            ):
                construction_errors.append(f"{row['id']}:candidate_order_not_changed")
        if kind == "multi_same_type_distractors":
            distractors = construction["distractor_tools"]
            candidates = construction["candidate_tools"]
            if (
                not 1 <= len(distractors) <= 3
                or len(candidates) > 5
                or not set(tools) <= set(candidates)
                or set(tools) & set(distractors)
            ):
                construction_errors.append(f"{row['id']}:distractor_size")
            if any(
                by_id[tool_id]["domain"] != construction["anchor_domain"]
                for tool_id in distractors
            ):
                construction_errors.append(f"{row['id']}:distractor_domain")
            if set(tools + distractors) != set(candidates):
                construction_errors.append(f"{row['id']}:candidate_alignment")
    query_fields = {"query", "question", "问题"} & set().union(*(row.keys() for row in rows))
    strict_ok = (
        len(ids) == len(set(ids))
        and dict(type_counts) == expected
        and not dependency_error_rows
        and not construction_errors
        and not query_fields
        and len(tool_counts) == len(by_id)
    )
    return {
        "audit_version": "assignment3-100tool-query-free-plan-v1",
        "strict_ok": strict_ok,
        "rows": len(rows),
        "unique_ids": len(set(ids)),
        "construction_counts": dict(sorted(type_counts.items())),
        "expected_construction_counts": expected,
        "tool_count_distribution": dict(sorted(Counter(len(row["tools"]) for row in rows).items())),
        "tool_coverage": len(tool_counts),
        "tool_occurrence_min": min(tool_counts.values()),
        "tool_occurrence_max": max(tool_counts.values()),
        "multi_order_randomization": {
            "rows": len(order_rows),
            "target_order_differs_from_source": sum(
                row["tools"] != row["construction"]["source_tools"]
                for row in order_rows
            ),
            "candidate_order_differs_from_target": sum(
                row["construction"]["candidate_tools"] != row["tools"]
                for row in order_rows
            ),
            "dependency_valid_rows": len(order_rows)
            - sum(
                bool(dependency_violations(row["tools"], by_id))
                for row in order_rows
            ),
        },
        "query_fields_present": sorted(query_fields),
        "dependency_error_rows": dependency_error_rows[:100],
        "dependency_error_count": len(dependency_error_rows),
        "construction_errors": construction_errors[:100],
        "construction_error_count": len(construction_errors),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    parser.add_argument("--formal-train", type=Path, default=TOOL_TRACE_DIR / "train.jsonl")
    parser.add_argument("--train-plan-out", type=Path, required=True)
    parser.add_argument("--test-plan-out", type=Path, required=True)
    parser.add_argument("--audit-out", type=Path, required=True)
    parser.add_argument("--single-count", type=int, default=18000)
    parser.add_argument("--order-count", type=int, default=24000)
    parser.add_argument("--augmented-count", type=int, default=12000)
    parser.add_argument("--test-count", type=int, default=1000)
    args = parser.parse_args()

    catalog = load_catalog(args.catalog)
    by_id = catalog_by_id(catalog)
    train_rows = load_jsonl(args.formal_train)
    if len(catalog) != 100 or len(train_rows) != 24000:
        raise SystemExit("expected the established 100-tool / 24k formal-train project scale")
    train_plan = build_train_plan(
        train_rows,
        by_id,
        args.single_count,
        args.order_count,
        args.augmented_count,
    )
    test_plan = build_test_plan(train_rows, by_id, args.test_count)
    source_isolation = isolate_source_traces(train_plan, test_plan, train_rows)
    train_expected = {
        "single_intent": args.single_count,
        "multi_candidate_order_shuffled": args.order_count,
        "multi_same_type_distractors": args.augmented_count,
    }
    test_expected = {"heldout_api_test": args.test_count}
    train_audit = audit_plan(train_plan, train_rows, by_id, train_expected)
    test_audit = audit_plan(test_plan, train_rows, by_id, test_expected)
    previous_train_plan_sha256 = (
        sha256_file(args.train_plan_out) if args.train_plan_out.exists() else None
    )
    previous_test_plan_sha256 = (
        sha256_file(args.test_plan_out) if args.test_plan_out.exists() else None
    )
    write_jsonl(args.train_plan_out, train_plan)
    write_jsonl(args.test_plan_out, test_plan)
    audit = {
        "status": (
            "complete"
            if train_audit["strict_ok"]
            and test_audit["strict_ok"]
            and source_isolation["strict_ok"]
            else "failed"
        ),
        "scale_derivation": {
            "pdf_reference_train_rows": 4000,
            "project_formal_train_rows": 24000,
            "scale_factor": 6,
            "pdf_assignment3_single": 3000,
            "pdf_assignment3_multi_order": 4000,
            "pdf_assignment3_multi_augmented": 2000,
            "project_single": args.single_count,
            "project_multi_order": args.order_count,
            "project_multi_augmented": args.augmented_count,
            "project_total": len(train_plan),
        },
        "query_generation_policy": "target plans contain no query text; all queries must be generated and judged by API",
        "train": train_audit,
        "test": test_audit,
        "source_trace_isolation": source_isolation,
        "plan_migration": {
            "previous_train_plan_sha256": previous_train_plan_sha256,
            "previous_test_plan_sha256": previous_test_plan_sha256,
            "test_plan_unchanged": (
                previous_test_plan_sha256 == sha256_file(args.test_plan_out)
                if previous_test_plan_sha256
                else None
            ),
        },
        "inputs": {
            str(args.catalog): sha256_file(args.catalog),
            str(args.formal_train): sha256_file(args.formal_train),
        },
        "outputs": {
            str(args.train_plan_out): sha256_file(args.train_plan_out),
            str(args.test_plan_out): sha256_file(args.test_plan_out),
        },
    }
    audit["strict_ok"] = audit["status"] == "complete"
    write_json(args.audit_out, audit)
    print(
        f"train_plan={len(train_plan)} test_plan={len(test_plan)} "
        f"strict_ok={audit['strict_ok']}"
    )
    return 0 if audit["strict_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
