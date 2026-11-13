#!/usr/bin/env bash
set -euo pipefail

MODEL_SIZE="${1:?usage: run_swift_tool_trace_lora.sh <4b|8b> <run-id> <lr> <epochs>}"
RUN_ID="${2:?missing run id}"
LR="${3:?missing learning rate}"
EPOCHS="${4:?missing epochs}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_FILE="${DATA_FILE:-$PROJECT_DIR/finetune/outputs/tooltrace-data-cache/runs/tool_trace_sft_train.runtime.jsonl}"
OUT_DIR="${OUT_DIR:-$PROJECT_DIR/finetune/outputs/tooltrace-$RUN_ID}"
STATS_FILE="${STATS_FILE:-$PROJECT_DIR/data/tool_call_trace/_aux/assignment3/prompt_length_stats.json}"

case "$MODEL_SIZE" in
  4b) MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/models/Qwen/Qwen3-4B}" ;;
  8b) MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/models/Qwen/Qwen3-8B}" ;;
  *) echo "MODEL_SIZE must be 4b or 8b" >&2; exit 2 ;;
esac

[[ -f "$DATA_FILE" ]] || { echo "missing SFT runtime data: $DATA_FILE" >&2; exit 2; }
[[ -d "$MODEL_PATH" ]] || { echo "missing model: $MODEL_PATH" >&2; exit 2; }
[[ -f "$STATS_FILE" ]] || { echo "missing prompt stats: $STATS_FILE" >&2; exit 2; }
command -v swift >/dev/null || { echo "MS-SWIFT is not available" >&2; exit 127; }

if [[ -z "${MAX_LENGTH:-}" ]]; then
  MAX_LENGTH="$(
    python3 - "$STATS_FILE" "$MODEL_PATH" <<'PY'
import json
import sys
from pathlib import Path

stats_path, model_path = sys.argv[1:]
stats = json.loads(Path(stats_path).read_text(encoding="utf-8"))
tokenizers = stats["tokenizers"]
resolved_model = Path(model_path).resolve()
for key, values in tokenizers.items():
    if Path(key).resolve() == resolved_model:
        print(values["selected_max_length"])
        break
else:
    raise SystemExit(f"model missing from prompt stats: {model_path}")
PY
  )"
fi
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
SAVE_ARGS=(--save_strategy "${SAVE_STRATEGY:-epoch}")
if [[ "${SAVE_STRATEGY:-epoch}" == "steps" ]]; then
  SAVE_ARGS+=(--save_steps "${SAVE_STEPS:-100}")
fi

export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
swift sft \
  --model "$MODEL_PATH" \
  --tuner_type lora \
  --dataset "$DATA_FILE" \
  --split_dataset_ratio 0 \
  --torch_dtype bfloat16 \
  --num_train_epochs "$EPOCHS" \
  --per_device_train_batch_size "${BATCH_SIZE:-1}" \
  --gradient_accumulation_steps "${GRAD_ACC:-4}" \
  --learning_rate "$LR" \
  --lora_rank "${LORA_RANK:-16}" \
  --lora_alpha "${LORA_ALPHA:-32}" \
  --lora_dropout "${LORA_DROPOUT:-0.05}" \
  --target_modules all-linear \
  --max_length "$MAX_LENGTH" \
  --loss_scale last_round \
  --add_non_thinking_prefix false \
  --gradient_checkpointing true \
  --lr_scheduler_type "${LR_SCHEDULER_TYPE:-cosine}" \
  --warmup_ratio "${WARMUP_RATIO:-0.05}" \
  --weight_decay "${WEIGHT_DECAY:-0.1}" \
  --max_grad_norm 1.0 \
  --seed "${SEED:-20260725}" \
  --data_seed "${DATA_SEED:-20260725}" \
  --dataset_num_proc "${DATASET_NUM_PROC:-8}" \
  --group_by_length "${GROUP_BY_LENGTH:-true}" \
  "${SAVE_ARGS[@]}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT:-3}" \
  --logging_steps 10 \
  --report_to none \
  --output_dir "$OUT_DIR" \
  --add_version false \
  "${EXTRA_ARGS[@]}"

echo "SFT output: $OUT_DIR"
