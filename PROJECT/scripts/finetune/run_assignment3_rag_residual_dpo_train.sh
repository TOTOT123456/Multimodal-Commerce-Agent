#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 || $# -gt 7 ]]; then
  echo "usage: $0 LR LR_TAG BETA BETA_TAG SEED GPUS [PORT]" >&2
  exit 2
fi

LR="$1"
LR_TAG="$2"
BETA="$3"
BETA_TAG="$4"
SEED="$5"
GPUS="$6"
PORT="${7:-29631}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_ID="${EXP_ID_OVERRIDE:-assignment3_rag_residual_dpo_v2_20260803}"
RUNTIME_DIR="$PROJECT_DIR/finetune/outputs/tooltrace-data-cache/$EXP_ID"
PARENT="${PARENT_OVERRIDE:-$PROJECT_DIR/finetune/outputs/tooltrace-a3-current-dpo-8b-lr1e6-b010-s20260802/checkpoint-100}"
RUNNER="$PROJECT_DIR/scripts/finetune/run_swift_tool_trace_dpo.sh"
PYTHON_ENV="/root/miniconda/envs/qwen-intent-ft/bin"
TRAIN_DATA="$RUNTIME_DIR/8b_train.runtime.jsonl"
MONITOR_DATA="$RUNTIME_DIR/8b_monitor.runtime.jsonl"

TOTAL_PAIRS="${TOTAL_PAIRS:-5000}"
[[ "$TOTAL_PAIRS" =~ ^[0-9]+$ && "$TOTAL_PAIRS" -ge 2000 && "$TOTAL_PAIRS" -le 5000 && $((TOTAL_PAIRS % 10)) -eq 0 ]] || {
  echo "TOTAL_PAIRS must be a multiple of 10 in 2000..5000" >&2
  exit 2
}
MONITOR_PAIRS=$((TOTAL_PAIRS / 10))
TRAIN_PAIRS=$((TOTAL_PAIRS - MONITOR_PAIRS))
[[ "$(wc -l < "$TRAIN_DATA")" -eq "$TRAIN_PAIRS" ]] || {
  echo "expected $TRAIN_PAIRS train pairs: $TRAIN_DATA" >&2
  exit 2
}
[[ "$(wc -l < "$MONITOR_DATA")" -eq "$MONITOR_PAIRS" ]] || {
  echo "expected $MONITOR_PAIRS monitor pairs: $MONITOR_DATA" >&2
  exit 2
}

VARIANT_TAG="${VARIANT_TAG:-}"
[[ -z "$VARIANT_TAG" || "$VARIANT_TAG" =~ ^-[a-z0-9][a-z0-9-]*$ ]] || {
  echo "VARIANT_TAG must be empty or a lowercase dash-prefixed slug" >&2
  exit 2
}
RUN_FAMILY="${RUN_FAMILY:-rag-residual-dpo-v2}"
[[ "$RUN_FAMILY" =~ ^[a-z0-9][a-z0-9-]*$ ]] || {
  echo "RUN_FAMILY must be a lowercase dash slug" >&2
  exit 2
}
RUN_ID="a3-${RUN_FAMILY}-8b-${LR_TAG}-b${BETA_TAG}-s${SEED}${VARIANT_TAG}"
OUT_DIR="$PROJECT_DIR/finetune/outputs/tooltrace-$RUN_ID"
LOG="$PROJECT_DIR/finetune/outputs/tooltrace-$RUN_ID.log"

IFS=',' read -r -a GPU_ARRAY <<< "$GPUS"
[[ "${#GPU_ARRAY[@]}" -eq 3 ]] || {
  echo "residual DPO training requires exactly three GPUs" >&2
  exit 2
}

CUDA_VISIBLE_DEVICES="$GPUS" \
MASTER_PORT="$PORT" \
NPROC_PER_NODE=3 \
BATCH_SIZE=1 \
EVAL_BATCH_SIZE=1 \
GRAD_ACC="${GRAD_ACC:-6}" \
DATASET_NUM_PROC=8 \
DATA_FILE="$TRAIN_DATA" \
VAL_FILE="$MONITOR_DATA" \
STATS_FILE="$PROJECT_DIR/data/tool_call_trace/_aux/assignment3/assignment3_100tool_api/sft_prompt_stats.json" \
MAX_LENGTH=768 \
LR="$LR" \
SEED="$SEED" \
DATA_SEED="$SEED" \
SAVE_STRATEGY=steps \
SAVE_STEPS="${SAVE_STEPS:-0.125}" \
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-20}" \
WARMUP_RATIO="${WARMUP_RATIO:-0.05}" \
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}" \
PATH="$PYTHON_ENV:$PATH" \
OUT_DIR="$OUT_DIR" \
  bash "$RUNNER" 8b "$RUN_ID" "$PARENT" "$BETA" "${EPOCHS:-2}" train \
    >"$LOG" 2>&1

echo "residual DPO training complete: $OUT_DIR"
