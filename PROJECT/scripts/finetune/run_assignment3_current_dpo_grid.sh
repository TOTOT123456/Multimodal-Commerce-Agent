#!/usr/bin/env bash
set -euo pipefail

MODEL_SIZE="${1:?usage: run_assignment3_current_dpo_grid.sh <4b|8b> [seed] [lr5e7|lr1e6] [005|010] [gpus] [port]}"
SEED="${2:-20260802}"
ONLY_LR_TAG="${3:-}"
ONLY_BETA_TAG="${4:-}"
ONLY_GPUS="${5:-0,1,2}"
ONLY_PORT="${6:-29611}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_ID="assignment3_current_api9k_dpo_20260802"
EXP_DIR="$PROJECT_DIR/data/tool_call_trace/_aux/assignment3/$EXP_ID"
RUNTIME_DIR="$PROJECT_DIR/finetune/outputs/tooltrace-data-cache/$EXP_ID"
RUNNER="$PROJECT_DIR/scripts/finetune/run_swift_tool_trace_dpo.sh"
PYTHON_ENV="/root/miniconda/envs/qwen-intent-ft/bin"

case "$MODEL_SIZE" in
  4b)
    PARENT="$PROJECT_DIR/finetune/outputs/tooltrace-a3-ablate-4b-n9000-lr5em5-e3-r32-d05/checkpoint-846"
    BATCH_SIZE=2
    GRAD_ACC=3
    ;;
  8b)
    PARENT="$PROJECT_DIR/finetune/outputs/tooltrace-a3-ablate-8b-n9000-lr5em5-e3-r32-d05/checkpoint-846"
    BATCH_SIZE=1
    GRAD_ACC=6
    ;;
  *)
    echo "MODEL_SIZE must be 4b or 8b" >&2
    exit 2
    ;;
esac

TRAIN_DATA="$RUNTIME_DIR/${MODEL_SIZE}_train.runtime.jsonl"
VAL_DATA="$RUNTIME_DIR/${MODEL_SIZE}_monitor.runtime.jsonl"
[[ "$(wc -l < "$TRAIN_DATA")" -eq 1800 ]] || {
  echo "expected exactly 1800 DPO train pairs: $TRAIN_DATA" >&2
  exit 2
}
[[ "$(wc -l < "$VAL_DATA")" -eq 200 ]] || {
  echo "expected exactly 200 DPO monitor pairs: $VAL_DATA" >&2
  exit 2
}

run_one() {
  local lr="$1"
  local lr_tag="$2"
  local beta="$3"
  local beta_tag="$4"
  local gpus="$5"
  local port="$6"
  local run_id="a3-current-dpo-${MODEL_SIZE}-${lr_tag}-b${beta_tag}-s${SEED}"
  local out_dir="$PROJECT_DIR/finetune/outputs/tooltrace-$run_id"
  if [[ -f "$out_dir/checkpoint-100/adapter_model.safetensors" ]]; then
    echo "reuse complete $run_id"
    return
  fi
  CUDA_VISIBLE_DEVICES="$gpus" \
  MASTER_PORT="$port" \
  NPROC_PER_NODE=3 \
  BATCH_SIZE="$BATCH_SIZE" \
  EVAL_BATCH_SIZE="$BATCH_SIZE" \
  GRAD_ACC="$GRAD_ACC" \
  DATASET_NUM_PROC=8 \
  DATA_FILE="$TRAIN_DATA" \
  VAL_FILE="$VAL_DATA" \
  STATS_FILE="$PROJECT_DIR/data/tool_call_trace/_aux/assignment3/assignment3_100tool_api/sft_prompt_stats.json" \
  MAX_LENGTH=768 \
  LR="$lr" \
  SEED="$SEED" \
  DATA_SEED="$SEED" \
  SAVE_STRATEGY=steps \
  SAVE_STEPS=0.5 \
  PATH="$PYTHON_ENV:$PATH" \
    bash "$RUNNER" "$MODEL_SIZE" "$run_id" "$PARENT" "$beta" 1 train \
      >"$PROJECT_DIR/finetune/outputs/tooltrace-$run_id.log" 2>&1
}

run_wave() {
  run_one "$1" "$2" "$3" "$4" 0,1,2 "$5" &
  local left=$!
  run_one "$1" "$2" "$6" "$7" 3,4,5 "$8" &
  local right=$!
  local status=0
  wait "$left" || status=1
  wait "$right" || status=1
  [[ "$status" -eq 0 ]] || exit "$status"
}

if [[ -n "$ONLY_LR_TAG" || -n "$ONLY_BETA_TAG" ]]; then
  [[ -n "$ONLY_LR_TAG" && -n "$ONLY_BETA_TAG" ]] || {
    echo "selected-recipe mode requires both LR and beta tags" >&2
    exit 2
  }
  case "$ONLY_LR_TAG" in
    lr5e7) only_lr="5e-7" ;;
    lr1e6) only_lr="1e-6" ;;
    *) echo "unsupported LR tag: $ONLY_LR_TAG" >&2; exit 2 ;;
  esac
  case "$ONLY_BETA_TAG" in
    005) only_beta="0.05" ;;
    010) only_beta="0.10" ;;
    *) echo "unsupported beta tag: $ONLY_BETA_TAG" >&2; exit 2 ;;
  esac
  run_one "$only_lr" "$ONLY_LR_TAG" "$only_beta" "$ONLY_BETA_TAG" \
    "$ONLY_GPUS" "$ONLY_PORT"
  echo "current DPO selected recipe complete model=$MODEL_SIZE seed=$SEED"
  exit 0
fi

run_wave 5e-7 lr5e7 0.05 005 29601 0.10 010 29602
run_wave 1e-6 lr1e6 0.05 005 29603 0.10 010 29604

echo "current DPO grid complete model=$MODEL_SIZE seed=$SEED"
