#!/usr/bin/env bash
set -euo pipefail

MODEL_SIZE="${1:?usage: run_assignment3_crossstyle_dpo.sh <4b|8b> <run-id> <sft-checkpoint> <beta>}"
RUN_ID="${2:?missing run id}"
SFT_CHECKPOINT="${3:?missing SFT checkpoint}"
BETA="${4:?missing beta}"
REF_CHECKPOINT="${REF_CHECKPOINT:-$SFT_CHECKPOINT}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME_DIR="$PROJECT_DIR/finetune/outputs/tooltrace-data-cache/runs"
DATA_FILE="${DATA_FILE:-$RUNTIME_DIR/assignment3_crossstyle_dpo_train.runtime.jsonl}"
VAL_FILE="${VAL_FILE:-$RUNTIME_DIR/assignment3_crossstyle_dpo_monitor.runtime.jsonl}"
OUT_DIR="${OUT_DIR:-$PROJECT_DIR/finetune/outputs/tooltrace-$RUN_ID}"
SWIFT_BIN="${SWIFT_BIN:-/root/miniconda/envs/qwen-intent-ft/bin/swift}"
MAX_LENGTH="${MAX_LENGTH:-768}"
EPOCHS="${EPOCHS:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
WORLD_SIZE="${NPROC_PER_NODE:-1}"

case "$MODEL_SIZE" in
  4b)
    MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/models/Qwen/Qwen3-4B}"
    BATCH_SIZE="${BATCH_SIZE:-2}"
    ;;
  8b)
    MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/models/Qwen/Qwen3-8B}"
    BATCH_SIZE="${BATCH_SIZE:-1}"
    ;;
  *)
    echo "MODEL_SIZE must be 4b or 8b" >&2
    exit 2
    ;;
esac

micro_global=$((WORLD_SIZE * BATCH_SIZE))
if (( micro_global <= 0 || GLOBAL_BATCH_SIZE % micro_global != 0 )); then
  echo "GLOBAL_BATCH_SIZE must be divisible by NPROC_PER_NODE * BATCH_SIZE" >&2
  exit 2
fi
GRAD_ACC="${GRAD_ACC:-$((GLOBAL_BATCH_SIZE / micro_global))}"
if (( WORLD_SIZE * BATCH_SIZE * GRAD_ACC != GLOBAL_BATCH_SIZE )); then
  echo "effective global batch mismatch" >&2
  exit 2
fi

[[ -x "$SWIFT_BIN" ]] || { echo "missing MS-SWIFT executable: $SWIFT_BIN" >&2; exit 2; }
[[ -d "$MODEL_PATH" ]] || { echo "missing model: $MODEL_PATH" >&2; exit 2; }
[[ -d "$SFT_CHECKPOINT" ]] || { echo "missing SFT checkpoint: $SFT_CHECKPOINT" >&2; exit 2; }
[[ -d "$REF_CHECKPOINT" ]] || { echo "missing reference checkpoint: $REF_CHECKPOINT" >&2; exit 2; }
[[ -s "$DATA_FILE" ]] || { echo "missing DPO train data: $DATA_FILE" >&2; exit 2; }
[[ -s "$VAL_FILE" ]] || { echo "missing DPO monitor data: $VAL_FILE" >&2; exit 2; }

mkdir -p "$OUT_DIR"
git check-ignore -q "$OUT_DIR/checkpoint-1/state.json" || {
  echo "checkpoint path is not ignored: $OUT_DIR/checkpoint-1/state.json" >&2
  exit 2
}

EXTRA_ARGS=()
if [[ -n "${MAX_STEPS:-}" ]]; then
  EXTRA_ARGS+=(--max_steps "$MAX_STEPS")
fi
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  EXTRA_ARGS+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi
if [[ -n "${RPO_ALPHA:-}" ]]; then
  EXTRA_ARGS+=(--rpo_alpha "$RPO_ALPHA")
fi
if [[ -n "${LABEL_SMOOTHING:-}" ]]; then
  EXTRA_ARGS+=(--label_smoothing "$LABEL_SMOOTHING")
fi

export NPROC_PER_NODE="$WORLD_SIZE"
"$SWIFT_BIN" rlhf \
  --rlhf_type dpo \
  --loss_type "${LOSS_TYPE:-sigmoid}" \
  --model "$MODEL_PATH" \
  --tuner_type lora \
  --adapters "$SFT_CHECKPOINT" \
  --ref_adapters "$REF_CHECKPOINT" \
  --dataset "$DATA_FILE" \
  --val_dataset "$VAL_FILE" \
  --split_dataset_ratio 0 \
  --torch_dtype bfloat16 \
  --num_train_epochs "$EPOCHS" \
  --per_device_train_batch_size "$BATCH_SIZE" \
  --per_device_eval_batch_size "${EVAL_BATCH_SIZE:-1}" \
  --gradient_accumulation_steps "$GRAD_ACC" \
  --learning_rate "${LR:-5e-6}" \
  --beta "$BETA" \
  --max_length "$MAX_LENGTH" \
  --truncation_strategy delete \
  --loss_scale last_round \
  --add_non_thinking_prefix false \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --target_modules all-linear \
  --gradient_checkpointing true \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  --weight_decay 0.1 \
  --max_grad_norm 1.0 \
  --seed "${SEED:-20260729}" \
  --data_seed "${DATA_SEED:-20260729}" \
  --dataset_num_proc "${DATASET_NUM_PROC:-8}" \
  --group_by_length true \
  --eval_strategy "${EVAL_STRATEGY:-epoch}" \
  --save_strategy "${SAVE_STRATEGY:-steps}" \
  --save_steps "${SAVE_STEPS:-100}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT:-3}" \
  --logging_steps 10 \
  --report_to none \
  --output_dir "$OUT_DIR" \
  --add_version false \
  "${EXTRA_ARGS[@]}"

echo "DPO output: $OUT_DIR"
