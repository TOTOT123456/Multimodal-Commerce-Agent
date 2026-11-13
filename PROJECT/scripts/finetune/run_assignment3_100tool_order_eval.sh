#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:?usage: run_assignment3_100tool_order_eval.sh <stage> <4b-adapter> <8b-adapter>}"
ADAPTER_4B="${2:?missing 4B adapter}"
ADAPTER_8B="${3:?missing 8B adapter}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda/envs/qwen-intent-ft/bin/python}"
DATA_DIR="$PROJECT_DIR/data/tool_call_trace/_aux/assignment3/assignment3_100tool_api/order_counterfactual_api"
PRED_DIR="$PROJECT_DIR/finetune/outputs/preds/tool_trace/assignment3_100tool_order_counterfactual"
INPUT="$DATA_DIR/queries_label_free.jsonl"
GOLD="$DATA_DIR/counterfactual_dev.jsonl"
RUNNER="$PROJECT_DIR/scripts/finetune/run_assignment3_eval.sh"
EVALUATOR="$PROJECT_DIR/scripts/finetune/evaluate_tool_trace.py"
PAIR_EVALUATOR="$PROJECT_DIR/scripts/finetune/evaluate_order_counterfactual_pairs.py"
mkdir -p "$PRED_DIR"

run_cell() {
  local model_size="$1"
  local adapter="$2"
  local gpu_list="$3"
  local batch_size="$4"
  local run_id="a3-100-${STAGE}-${model_size}-order-global"
  PYTHON_BIN="$PYTHON_BIN" GPU_LIST="$gpu_list" BATCH_SIZE="$batch_size" \
    bash "$RUNNER" \
      "$PROJECT_DIR/models/Qwen/Qwen3-${model_size^^}" \
      "$adapter" "$INPUT" global \
      "$PRED_DIR/$run_id.jsonl" "$run_id" \
      >"$PRED_DIR/$run_id.infer.log" 2>&1
}

run_cell 4b "$ADAPTER_4B" 0,1,2,3 8 &
p0="$!"
run_cell 8b "$ADAPTER_8B" 4,5,6,7 4 &
p1="$!"
status=0
wait "$p0" || status=1
wait "$p1" || status=1
[[ "$status" -eq 0 ]] || {
  echo "order-counterfactual inference failed" >&2
  exit "$status"
}

for model_size in 4b 8b; do
  run_id="a3-100-${STAGE}-${model_size}-order-global"
  "$PYTHON_BIN" "$EVALUATOR" \
    --predictions "$PRED_DIR/$run_id.jsonl" \
    --gold "$GOLD" \
    --metrics-out "$PRED_DIR/$run_id.metrics.json" \
    --scored-out "$PRED_DIR/$run_id.scored.jsonl" \
    --run-id "$run_id"
  "$PYTHON_BIN" "$PAIR_EVALUATOR" \
    --gold "$GOLD" \
    --scored "$PRED_DIR/$run_id.scored.jsonl" \
    --out "$PRED_DIR/$run_id.pair_metrics.json"
done

echo "Assignment3 100-tool order evaluation complete: $STAGE"
