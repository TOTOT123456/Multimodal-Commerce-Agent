#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:?usage: run_assignment3_100tool_eval_stage.sh <stage> <4b-adapter-or-none> <8b-adapter-or-none>}"
ADAPTER_4B="${2:?missing 4B adapter or none}"
ADAPTER_8B="${3:?missing 8B adapter or none}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda/envs/qwen-intent-ft/bin/python}"
TEST_DIR="${TEST_DIR:-$PROJECT_DIR/data/tool_call_trace/_aux/assignment3/assignment3_100tool_api/test_api}"
PRED_DIR="${PRED_DIR:-$PROJECT_DIR/finetune/outputs/preds/tool_trace/assignment3_100tool_test}"
GOLD="$TEST_DIR/test.jsonl"
QUERY_INPUT="$TEST_DIR/queries_label_free_k10.jsonl"
RAG_INPUT="$TEST_DIR/rag_k10_label_free.jsonl"
RUNNER="$PROJECT_DIR/scripts/finetune/run_assignment3_eval.sh"
EVALUATOR="$PROJECT_DIR/scripts/finetune/evaluate_tool_trace.py"
mkdir -p "$PRED_DIR"

run_cell() {
  local model_size="$1"
  local model_path="$2"
  local adapter="$3"
  local view="$4"
  local input="$5"
  local gpu="$6"
  local batch="$7"
  local run_id="a3-100-${stage}-${model_size}-${view}"
  PYTHON_BIN="$PYTHON_BIN" GPU_LIST="$gpu" BATCH_SIZE="$batch" \
    bash "$RUNNER" "$model_path" "$adapter" "$input" "$view" \
      "$PRED_DIR/$run_id.jsonl" "$run_id" \
      >"$PRED_DIR/$run_id.infer.log" 2>&1
}

stage="$STAGE"
# Full-catalog prompts are much longer than RAG prompts.  Give each global
# cell two independent inference shards while keeping one GPU per RAG cell.
run_cell 4b "$PROJECT_DIR/models/Qwen/Qwen3-4B" "$ADAPTER_4B" global "$QUERY_INPUT" "0,1" 8 &
p0="$!"
run_cell 4b "$PROJECT_DIR/models/Qwen/Qwen3-4B" "$ADAPTER_4B" rag "$RAG_INPUT" 2 16 &
p1="$!"
run_cell 8b "$PROJECT_DIR/models/Qwen/Qwen3-8B" "$ADAPTER_8B" global "$QUERY_INPUT" "4,5" 4 &
p2="$!"
run_cell 8b "$PROJECT_DIR/models/Qwen/Qwen3-8B" "$ADAPTER_8B" rag "$RAG_INPUT" 6 8 &
p3="$!"

status=0
for pid in "$p0" "$p1" "$p2" "$p3"; do
  wait "$pid" || status=1
done
[[ "$status" -eq 0 ]] || {
  echo "one or more Assignment3 evaluation cells failed" >&2
  exit "$status"
}

for model_size in 4b 8b; do
  for view in global rag; do
    run_id="a3-100-${stage}-${model_size}-${view}"
    "$PYTHON_BIN" "$EVALUATOR" \
      --predictions "$PRED_DIR/$run_id.jsonl" \
      --gold "$GOLD" \
      --metrics-out "$PRED_DIR/$run_id.metrics.json" \
      --scored-out "$PRED_DIR/$run_id.scored.jsonl" \
      --run-id "$run_id"
  done
done

echo "Assignment3 stage evaluation complete: $STAGE"
