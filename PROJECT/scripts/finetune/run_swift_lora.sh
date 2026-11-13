#!/usr/bin/env bash
set -euo pipefail

MODEL_SIZE="${1:-0.6b}"
EXP_NAME="${2:-full_lora}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="$ROOT/finetune/data"
OUT_DIR="${OUT_DIR:-$ROOT/finetune/outputs/${MODEL_SIZE}_${EXP_NAME}_$(date +%Y%m%d_%H%M%S)}"

case "$MODEL_SIZE" in
  0.6b)
    MODEL_CANDIDATES=("$ROOT/models/Qwen/Qwen3-0.6B" "$ROOT/models/Qwen/Qwen3-0___6B" "Qwen/Qwen3-0.6B")
    ;;
  1.7b)
    MODEL_CANDIDATES=("$ROOT/models/Qwen/Qwen3-1.7B" "$ROOT/models/Qwen/Qwen3-1___7B" "Qwen/Qwen3-1.7B")
    ;;
  *)
    echo "unknown MODEL_SIZE: $MODEL_SIZE" >&2
    exit 2
    ;;
esac

case "$EXP_NAME" in
  base_lora) DATA_PREFIX="base" ;;
  full_lora) DATA_PREFIX="full" ;;
  balanced_lora) DATA_PREFIX="balanced" ;;
  hard_patch) DATA_PREFIX="hard" ;;
  *) echo "unknown EXP_NAME: $EXP_NAME" >&2; exit 2 ;;
esac

TRAIN_FILE="$DATA_DIR/sft_${DATA_PREFIX}_train.jsonl"
VAL_FILE="$DATA_DIR/sft_${DATA_PREFIX}_val.jsonl"
if [[ ! -f "$TRAIN_FILE" || ! -f "$VAL_FILE" ]]; then
  echo "missing SFT data. Run: python3 PROJECT/scripts/finetune/prepare_sft_data.py" >&2
  exit 2
fi

MODEL_PATH=""
for candidate in "${MODEL_CANDIDATES[@]}"; do
  if [[ -d "$candidate" || "$candidate" == Qwen/* ]]; then
    MODEL_PATH="$candidate"
    break
  fi
done

if ! command -v swift >/dev/null 2>&1; then
  echo "MS-SWIFT is not installed in this environment. Install training deps first, then rerun this script." >&2
  exit 127
fi

BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACC="${GRAD_ACC:-16}"
LR="${LR:-1e-4}"
EPOCHS="${EPOCHS:-2}"
EXTRA_ARGS=()
if [[ -n "${MAX_STEPS:-}" ]]; then
  EXTRA_ARGS+=(--max_steps "$MAX_STEPS")
fi
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  EXTRA_ARGS+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

mkdir -p "$OUT_DIR"
swift sft \
  --model "$MODEL_PATH" \
  --tuner_type lora \
  --dataset "$TRAIN_FILE" \
  --val_dataset "$VAL_FILE" \
  --torch_dtype bfloat16 \
  --num_train_epochs "$EPOCHS" \
  --per_device_train_batch_size "$BATCH_SIZE" \
  --per_device_eval_batch_size "$BATCH_SIZE" \
  --gradient_accumulation_steps "$GRAD_ACC" \
  --learning_rate "$LR" \
  --lora_rank "${LORA_RANK:-16}" \
  --lora_alpha "${LORA_ALPHA:-32}" \
  --target_modules all-linear \
  --max_length "${MAX_LENGTH:-512}" \
  --gradient_checkpointing true \
  --warmup_ratio 0.05 \
  --output_dir "$OUT_DIR" \
  --save_steps 200 \
  --save_total_limit 2 \
  --eval_steps 200 \
  --logging_steps 20 \
  --add_version false \
  "${EXTRA_ARGS[@]}"

echo "training output: $OUT_DIR"
