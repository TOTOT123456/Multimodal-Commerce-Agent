#!/usr/bin/env bash
set -euo pipefail

ADAPTER_4B_E1="${1:?usage: run_assignment3_100tool_dev_selection.sh <4b-e1> <4b-e2> <8b-e1>}"
ADAPTER_4B_E2="${2:?missing 4B epoch-2 adapter}"
ADAPTER_8B_E1="${3:?missing 8B epoch-1 adapter}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda/envs/qwen-intent-ft/bin/python}"
DEV_DIR="$PROJECT_DIR/data/tool_call_trace/_aux/assignment3/api_native_dev_2400"
PRED_DIR="$PROJECT_DIR/finetune/outputs/preds/tool_trace/assignment3_100tool_dev_selection"
INPUT="$DEV_DIR/rag_k10_label_free.jsonl"
GOLD="$DEV_DIR/dev.jsonl"
RUNNER="$PROJECT_DIR/scripts/finetune/run_assignment3_eval.sh"
EVALUATOR="$PROJECT_DIR/scripts/finetune/evaluate_tool_trace.py"
mkdir -p "$PRED_DIR"

run_cell() {
  local model_size="$1"
  local adapter="$2"
  local tag="$3"
  local gpu_list="$4"
  local batch_size="$5"
  local model="$PROJECT_DIR/models/Qwen/Qwen3-${model_size^^}"
  local run_id="a3-100-${tag}-dev-rag-k10"
  PYTHON_BIN="$PYTHON_BIN" GPU_LIST="$gpu_list" BATCH_SIZE="$batch_size" \
    bash "$RUNNER" "$model" "$adapter" "$INPUT" rag \
      "$PRED_DIR/$run_id.jsonl" "$run_id" \
      >"$PRED_DIR/$run_id.infer.log" 2>&1
}

run_cell 4b "$ADAPTER_4B_E1" sft4-e1 "0,1" 16 &
p0="$!"
run_cell 4b "$ADAPTER_4B_E2" sft4-e2 "2,3" 16 &
p1="$!"
run_cell 8b "$ADAPTER_8B_E1" sft8-e1 "4,5" 8 &
p2="$!"

status=0
for pid in "$p0" "$p1" "$p2"; do
  wait "$pid" || status=1
done
[[ "$status" -eq 0 ]] || {
  echo "one or more dev-selection inference cells failed" >&2
  exit "$status"
}

for tag in sft4-e1 sft4-e2 sft8-e1; do
  run_id="a3-100-${tag}-dev-rag-k10"
  "$PYTHON_BIN" "$EVALUATOR" \
    --predictions "$PRED_DIR/$run_id.jsonl" \
    --gold "$GOLD" \
    --metrics-out "$PRED_DIR/$run_id.metrics.json" \
    --scored-out "$PRED_DIR/$run_id.scored.jsonl" \
    --run-id "$run_id"
done

echo "Assignment3 100-tool dev selection complete"
