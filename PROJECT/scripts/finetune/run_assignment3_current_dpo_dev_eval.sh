#!/usr/bin/env bash
set -euo pipefail

MODEL_SIZE="${1:?usage: run_assignment3_current_dpo_dev_eval.sh <4b|8b> NAME=ADAPTER [...] }"
shift
[[ "$#" -gt 0 ]] || { echo "at least one NAME=ADAPTER is required" >&2; exit 2; }

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_ID="assignment3_current_api9k_dpo_20260802"
EXP_DIR="$PROJECT_DIR/data/tool_call_trace/_aux/assignment3/$EXP_ID"
PRED_DIR="$PROJECT_DIR/finetune/outputs/preds/tool_trace/$EXP_ID/dev"
RUNNER="$PROJECT_DIR/scripts/finetune/run_assignment3_eval.sh"
EVALUATOR="$PROJECT_DIR/scripts/finetune/evaluate_tool_trace.py"
PYTHON_BIN="/root/miniconda/envs/qwen-intent-ft/bin/python"
EVAL_GPU_LIST="${GPU_LIST:-0,1,2,3,4,5}"

case "$MODEL_SIZE" in
  4b)
    MODEL="$PROJECT_DIR/models/Qwen/Qwen3-4B"
    RAG_BATCH="${RAG_BATCH_SIZE:-16}"
    GLOBAL_BATCH="${GLOBAL_BATCH_SIZE:-6}"
    ;;
  8b)
    MODEL="$PROJECT_DIR/models/Qwen/Qwen3-8B"
    RAG_BATCH="${RAG_BATCH_SIZE:-8}"
    GLOBAL_BATCH="${GLOBAL_BATCH_SIZE:-2}"
    ;;
  *)
    echo "MODEL_SIZE must be 4b or 8b" >&2
    exit 2
    ;;
esac

for spec in "$@"; do
  name="${spec%%=*}"
  adapter="${spec#*=}"
  [[ -n "$name" && "$adapter" != "$spec" && -d "$adapter" ]] || {
    echo "invalid or missing endpoint: $spec" >&2
    exit 2
  }
  for view in rag global; do
    input="$EXP_DIR/data/dev_${view}_label_free.jsonl"
    output="$PRED_DIR/$MODEL_SIZE/$name/$view.jsonl"
    metrics="$EXP_DIR/evaluation/dev/$MODEL_SIZE/$name/$view.metrics.json"
    scored="$EXP_DIR/evaluation/dev/$MODEL_SIZE/$name/$view.scored.jsonl"
    mkdir -p "$(dirname "$output")" "$(dirname "$metrics")"
    batch="$RAG_BATCH"
    [[ "$view" == global ]] && batch="$GLOBAL_BATCH"
    PYTHON_BIN="$PYTHON_BIN" \
    GPU_LIST="$EVAL_GPU_LIST" \
    BATCH_SIZE="$batch" \
    MAX_NEW_TOKENS=96 \
    SAVE_EVERY=200 \
    SEED=20260802 \
      bash "$RUNNER" "$MODEL" "$adapter" "$input" "$view" "$output" \
        "current-dpo-dev-$MODEL_SIZE-$name-$view" \
        >"${output%.jsonl}.infer.log" 2>&1
    "$PYTHON_BIN" "$EVALUATOR" \
      --predictions "$output" \
      --gold "$EXP_DIR/plan/dev_gold.jsonl" \
      --metrics-out "$metrics" \
      --scored-out "$scored" \
      --run-id "current-dpo-dev-$MODEL_SIZE-$name-$view"
  done
done

echo "current DPO dev evaluation complete model=$MODEL_SIZE endpoints=$#"
