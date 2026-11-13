#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:?usage: run_assignment3_eval.sh <model> <adapter-or-none> <input> <query_only|global|rag> <output> <run-id>}"
ADAPTER_PATH="${2:?missing adapter path or 'none'}"
INPUT_PATH="${3:?missing label-free input}"
VIEW="${4:?missing view}"
OUTPUT_PATH="${5:?missing output path}"
RUN_ID="${6:?missing run id}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
CANDIDATE_ORDER="${CANDIDATE_ORDER:-original}"
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
SHARD_COUNT="${#GPUS[@]}"
mkdir -p "$(dirname "$OUTPUT_PATH")"

PIDS=()
SHARDS=()
for shard_index in "${!GPUS[@]}"; do
  gpu="${GPUS[$shard_index]}"
  shard_path="${OUTPUT_PATH%.jsonl}.shard${shard_index}.jsonl"
  SHARDS+=("$shard_path")
  command=(
    "$PYTHON_BIN" "$PROJECT_DIR/scripts/finetune/infer_tool_trace.py"
    --input "$INPUT_PATH"
    --view "$VIEW"
    --candidate-order "$CANDIDATE_ORDER"
    --model "$MODEL_PATH"
    --output "$shard_path"
    --run-id "$RUN_ID"
    --model-name "$RUN_ID"
    --batch-size "${BATCH_SIZE:-8}"
    --max-new-tokens "${MAX_NEW_TOKENS:-128}"
    --shard-count "$SHARD_COUNT"
    --shard-index "$shard_index"
    --save-every "${SAVE_EVERY:-200}"
    --seed "${SEED:-20260725}"
    --warmup-batches "${WARMUP_BATCHES:-0}"
  )
  if [[ -n "${LIMIT:-}" ]]; then
    command+=(--limit "$LIMIT")
  fi
  if [[ "$ADAPTER_PATH" != "none" ]]; then
    command+=(--adapter "$ADAPTER_PATH")
  fi
  if [[ "${DO_SAMPLE:-false}" == "true" ]]; then
    command+=(
      --do-sample
      --temperature "${TEMPERATURE:-0.7}"
      --top-p "${TOP_P:-0.8}"
      --top-k "${TOP_K:-20}"
      --samples-per-prompt "${SAMPLES_PER_PROMPT:-2}"
    )
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "${command[@]}" &
  PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
[[ "$status" -eq 0 ]] || { echo "one or more inference shards failed" >&2; exit "$status"; }

if [[ "${SKIP_MERGE:-false}" == "true" ]]; then
  echo "inference shards complete; merge skipped"
  exit 0
fi

"$PYTHON_BIN" "$PROJECT_DIR/scripts/finetune/merge_tool_trace_shards.py" \
  --input "$INPUT_PATH" \
  --shards "${SHARDS[@]}" \
  --output "$OUTPUT_PATH"

echo "predictions: $OUTPUT_PATH"
