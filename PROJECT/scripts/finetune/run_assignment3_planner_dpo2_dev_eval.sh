#!/usr/bin/env bash
set -euo pipefail

[[ "$#" -gt 0 ]] || {
  echo "usage: $0 NAME=ADAPTER [...]" >&2
  exit 2
}

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_ID="assignment3_planner_dpo2_20260808"
EXP_DIR="$PROJECT_DIR/data/tool_call_trace/_aux/assignment3/$EXP_ID"
INPUT="$PROJECT_DIR/data/rag/_aux/current_canonical_dataset_retrieval/planner_dpo2_dev_20260808/rag10_label_free.jsonl"
GOLD="$EXP_DIR/data/dev_gold.jsonl"
PRED_DIR="$PROJECT_DIR/finetune/outputs/preds/tool_trace/$EXP_ID/dev"
RUNNER="$PROJECT_DIR/scripts/finetune/run_assignment3_eval.sh"
EVALUATOR="$PROJECT_DIR/scripts/finetune/evaluate_tool_trace.py"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda/envs/qwen-intent-ft/bin/python}"
MODEL="$PROJECT_DIR/models/Qwen/Qwen3-8B"

[[ -f "$INPUT" && "$(wc -l < "$INPUT")" -eq 1200 ]] || {
  echo "missing complete development input: $INPUT" >&2
  exit 2
}
[[ -f "$GOLD" && "$(wc -l < "$GOLD")" -eq 1200 ]] || {
  echo "missing complete development gold: $GOLD" >&2
  exit 2
}

for spec in "$@"; do
  name="${spec%%=*}"
  adapter="${spec#*=}"
  [[ -n "$name" && "$adapter" != "$spec" && -d "$adapter" ]] || {
    echo "invalid endpoint: $spec" >&2
    exit 2
  }
  output="$PRED_DIR/$name.jsonl"
  metrics="$EXP_DIR/evaluation/dev/$name/rag.metrics.json"
  scored="$EXP_DIR/evaluation/dev/$name/rag.scored.jsonl"
  mkdir -p "$(dirname "$output")" "$(dirname "$metrics")"
  PYTHON_BIN="$PYTHON_BIN" \
  GPU_LIST="${GPU_LIST:-0,1,2,3}" \
  BATCH_SIZE="${BATCH_SIZE:-12}" \
  MAX_NEW_TOKENS=96 \
  SAVE_EVERY=300 \
  SEED=20260818 \
  CANDIDATE_ORDER=original \
    bash "$RUNNER" "$MODEL" "$adapter" "$INPUT" rag "$output" \
      "planner-dpo2-dev-$name"
  "$PYTHON_BIN" "$EVALUATOR" \
    --predictions "$output" \
    --gold "$GOLD" \
    --metrics-out "$metrics" \
    --scored-out "$scored" \
    --run-id "planner-dpo2-dev-$name"
done

echo "planner DPO2 development evaluation complete endpoints=$#"
