#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:?usage: run_assignment3_100tool_candidate_order_eval.sh <stage> <4b-adapter>}"
ADAPTER_4B="${2:?missing selected 4B adapter}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda/envs/qwen-intent-ft/bin/python}"
TEST_DIR="$PROJECT_DIR/data/tool_call_trace/_aux/assignment3/assignment3_100tool_api/test_api"
PRED_DIR="$PROJECT_DIR/finetune/outputs/preds/tool_trace/assignment3_100tool_candidate_order"
RUNNER="$PROJECT_DIR/scripts/finetune/run_assignment3_eval.sh"
EVALUATOR="$PROJECT_DIR/scripts/finetune/evaluate_tool_trace.py"
MODEL="$PROJECT_DIR/models/Qwen/Qwen3-4B"
GOLD="$TEST_DIR/test.jsonl"
mkdir -p "$PRED_DIR"

run_cell() {
  local view="$1"
  local order="$2"
  local input="$3"
  local gpu_list="$4"
  local batch_size="$5"
  local run_id="a3-100-${STAGE}-4b-${view}-candidate-${order}"
  PYTHON_BIN="$PYTHON_BIN" \
  GPU_LIST="$gpu_list" \
  BATCH_SIZE="$batch_size" \
  CANDIDATE_ORDER="$order" \
    bash "$RUNNER" "$MODEL" "$ADAPTER_4B" "$input" "$view" \
      "$PRED_DIR/$run_id.jsonl" "$run_id" \
      >"$PRED_DIR/$run_id.infer.log" 2>&1
}

run_cell global reverse "$TEST_DIR/queries_label_free_k10.jsonl" 0,1 8 &
p0="$!"
run_cell global hash_shuffle "$TEST_DIR/queries_label_free_k10.jsonl" 2,3 8 &
p1="$!"
run_cell rag reverse "$TEST_DIR/rag_k10_label_free.jsonl" 4,5 16 &
p2="$!"
run_cell rag hash_shuffle "$TEST_DIR/rag_k10_label_free.jsonl" 6,7 16 &
p3="$!"
status=0
for pid in "$p0" "$p1" "$p2" "$p3"; do
  wait "$pid" || status=1
done
[[ "$status" -eq 0 ]] || {
  echo "candidate-order inference failed" >&2
  exit "$status"
}

for view in global rag; do
  for order in reverse hash_shuffle; do
    run_id="a3-100-${STAGE}-4b-${view}-candidate-${order}"
    "$PYTHON_BIN" "$EVALUATOR" \
      --predictions "$PRED_DIR/$run_id.jsonl" \
      --gold "$GOLD" \
      --metrics-out "$PRED_DIR/$run_id.metrics.json" \
      --scored-out "$PRED_DIR/$run_id.scored.jsonl" \
      --run-id "$run_id"
  done
done

echo "Assignment3 100-tool candidate-order evaluation complete: $STAGE"
