#!/usr/bin/env python3
"""Plan the pre-registered runner-up interaction check after negative selection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from search_retrieval_pipeline import select_aggregate_route  # noqa: E402


ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / "PROJECT"
AUX = PROJECT / "data/rag/_aux/retrieval_pipeline_reselection"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def family(group: str) -> str:
    if group in {"bge_base", "existing_task_bge_base"}:
        return "bge_base"
    if group in {"bge_large", "bge_v2_m3", "qwen3_0p6b"}:
        return group
    raise ValueError(f"unknown architecture group: {group}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architecture-json",
        type=Path,
        default=AUX / "downstream/architecture_screen/pipeline_search.json",
    )
    parser.add_argument(
        "--negative-json",
        type=Path,
        default=AUX / "downstream/negative_screen/pipeline_search.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=AUX / "downstream/negative_interaction_plan.json",
    )
    args = parser.parse_args()

    architecture = load_json(args.architecture_json)
    negative = load_json(args.negative_json)
    winner_group = str(architecture["selected"]["group"])
    winner_model = family(winner_group)
    negative_recipe = str(negative["selected"]["group"])
    required = negative_recipe in {"source_mined7", "mixed_typed3_mined4"}
    payload = {
        "required": required,
        "winner_architecture_group": winner_group,
        "winner_model_key": winner_model,
        "selected_negative_recipe": negative_recipe,
    }
    if required:
        remaining = [
            row
            for row in architecture["selected_by_group"]
            if family(str(row["group"])) != winner_model
        ]
        if not remaining:
            raise ValueError("no distinct runner-up architecture remains")
        runner, trace = select_aggregate_route(
            remaining,
            float(architecture["selection_tolerance"]),
            compare_latency=True,
        )
        runner_group = str(runner["group"])
        payload.update(
            {
                "runner_up_architecture_group": runner_group,
                "runner_up_model_key": family(runner_group),
                "runner_up_typed_metrics": runner["aggregate"],
                "runner_up_selection_trace": trace,
            }
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"negative interaction required={required} recipe={negative_recipe} "
        f"runner={payload.get('runner_up_model_key', 'none')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
