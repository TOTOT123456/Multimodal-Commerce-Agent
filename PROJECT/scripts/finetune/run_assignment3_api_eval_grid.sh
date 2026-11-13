#!/usr/bin/env bash
set -euo pipefail

MODEL_SIZE="${1:?usage: run_assignment3_api_eval_grid.sh <4b|8b> <run-id> <adapter>}"
RUN_ID="${2:?missing run id}"
ADAPTER="${3:?missing adapter path}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
PRED_DIR="$PROJECT_DIR/finetune/outputs/preds/tool_trace/api_native_dev_2400"
DEV_DIR="$PROJECT_DIR/data/tool_call_trace/_aux/assignment3/api_native_dev_2400"
GOLD="$DEV_DIR/dev.jsonl"
mkdir -p "$PRED_DIR"

case "$MODEL_SIZE" in
  4b) MODEL="$PROJECT_DIR/models/Qwen/Qwen3-4B" ;;
  8b) MODEL="$PROJECT_DIR/models/Qwen/Qwen3-8B" ;;
  *) echo "MODEL_SIZE must be 4b or 8b" >&2; exit 2 ;;
esac

declare -a NAMES=(
  "query-only"
  "global"
  "rag-k5"
  "rag-k10"
  "rag-k20"
  "rag-k30"
)
declare -a INPUTS=(
  "$DEV_DIR/queries_label_free_k5.jsonl"
  "$DEV_DIR/queries_label_free_k10.jsonl"
  "$DEV_DIR/rag_k5_label_free.jsonl"
  "$DEV_DIR/rag_k10_label_free.jsonl"
  "$DEV_DIR/rag_k20_label_free.jsonl"
  "$DEV_DIR/rag_k30_label_free.jsonl"
)
declare -a VIEWS=("query_only" "global" "rag" "rag" "rag" "rag")
declare -a GPUS=("0" "1" "2" "3" "4" "5")

pids=()
for index in "${!NAMES[@]}"; do
  name="${NAMES[$index]}"
  output="$PRED_DIR/api-native-$RUN_ID-$name.jsonl"
  PYTHON_BIN="$PYTHON_BIN" \
  GPU_LIST="${GPUS[$index]}" \
  BATCH_SIZE="${BATCH_SIZE:-16}" \
  MAX_NEW_TOKENS=128 \
    bash "$PROJECT_DIR/scripts/finetune/run_assignment3_eval.sh" \
      "$MODEL" "$ADAPTER" "${INPUTS[$index]}" "${VIEWS[$index]}" \
      "$output" "api-native-$RUN_ID-$name" \
      > "$PRED_DIR/api-native-$RUN_ID-$name.infer.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if [[ "$status" -ne 0 ]]; then
  for name in "${NAMES[@]}"; do
      echo "inference failure log: api-native-$RUN_ID-$name.infer.log" >&2
      tail -80 "$PRED_DIR/api-native-$RUN_ID-$name.infer.log" >&2 || true
  done
  exit "$status"
fi

for name in "${NAMES[@]}"; do
  output="$PRED_DIR/api-native-$RUN_ID-$name.jsonl"
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/finetune/evaluate_tool_trace.py" \
    --predictions "$output" \
    --gold "$GOLD" \
    --metrics-out "$PRED_DIR/api-native-$RUN_ID-$name.metrics.json" \
    --scored-out "$PRED_DIR/api-native-$RUN_ID-$name.scored.jsonl" \
    --run-id "api-native-$RUN_ID-$name"
done

echo "API-native dev grid complete: $RUN_ID"
