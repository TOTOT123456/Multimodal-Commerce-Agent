#!/usr/bin/env bash
set -euo pipefail

SELECTED_ADAPTER="${1:?usage: run_tool_trace_preference_mining.sh <selected-4b-adapter> <early-4b-adapter>}"
EARLY_ADAPTER="${2:?missing early 4B SFT adapter}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="$PROJECT_DIR/models/Qwen/Qwen3-4B"
ASSIGNMENT_DIR="$PROJECT_DIR/data/tool_call_trace/_aux/assignment3"
PRED_DIR="$PROJECT_DIR/finetune/outputs/preds/tool_trace"
mkdir -p "$PRED_DIR"

run_sampled() {
  name="$1"
  adapter="$2"
  input="$3"
  view="$4"
  gpu_list="$5"
  seed="$6"
  batch_size="$7"
  PYTHON_BIN="$PYTHON_BIN" \
  GPU_LIST="$gpu_list" \
  BATCH_SIZE="$batch_size" \
  MAX_NEW_TOKENS=128 \
  DO_SAMPLE=true \
  TEMPERATURE=0.7 \
  TOP_P=0.8 \
  TOP_K=20 \
  SAMPLES_PER_PROMPT=2 \
  SEED="$seed" \
    bash "$PROJECT_DIR/scripts/finetune/run_assignment3_eval.sh" \
      "$MODEL" "$adapter" "$input" "$view" \
      "$PRED_DIR/$name.jsonl" "$name" \
      > "$PRED_DIR/$name.infer.log" 2>&1
}

run_sampled \
  train-selected_sft4-rag-sampled \
  "$SELECTED_ADAPTER" \
  "$ASSIGNMENT_DIR/train_rag_mining_label_free.jsonl" \
  rag 0,1,2,3 20260725 16 &
selected_rag_pid="$!"
run_sampled \
  train-early_sft4-step50-rag-sampled \
  "$EARLY_ADAPTER" \
  "$ASSIGNMENT_DIR/train_rag_mining_label_free.jsonl" \
  rag 4,5,6,7 20260726 16 &
early_rag_pid="$!"
rag_status=0
wait "$selected_rag_pid" || rag_status=1
wait "$early_rag_pid" || rag_status=1
[[ "$rag_status" -eq 0 ]] || {
  echo "RAG preference mining inference failed; inspect per-source logs" >&2
  exit "$rag_status"
}

run_sampled \
  train-selected_sft4-global-sampled \
  "$SELECTED_ADAPTER" \
  "$ASSIGNMENT_DIR/train_global_mining_label_free.jsonl" \
  global 0,1,2,3 20260727 8 &
selected_global_pid="$!"
run_sampled \
  train-early_sft4-step50-global-sampled \
  "$EARLY_ADAPTER" \
  "$ASSIGNMENT_DIR/train_global_mining_label_free.jsonl" \
  global 4,5,6,7 20260728 8 &
early_global_pid="$!"
global_status=0
wait "$selected_global_pid" || global_status=1
wait "$early_global_pid" || global_status=1
[[ "$global_status" -eq 0 ]] || {
  echo "global preference mining inference failed; inspect per-source logs" >&2
  exit "$global_status"
}

"$PYTHON_BIN" "$PROJECT_DIR/scripts/finetune/build_dpo_preferences.py" \
  --train "$ASSIGNMENT_DIR/train.reviewed.jsonl" \
  --predictions "selected_sft4_rag=$PRED_DIR/train-selected_sft4-rag-sampled.jsonl" \
  --predictions "early_sft4_step50_rag=$PRED_DIR/train-early_sft4-step50-rag-sampled.jsonl" \
  --predictions "selected_sft4_global=$PRED_DIR/train-selected_sft4-global-sampled.jsonl" \
  --predictions "early_sft4_step50_global=$PRED_DIR/train-early_sft4-step50-global-sampled.jsonl" \
  --min-pairs 4000 \
  --max-pairs 4000 \
  --max-per-query 2 \
  --min-core-ratio 0.70 \
  --no-allow-synthetic

echo "preference mining complete"
