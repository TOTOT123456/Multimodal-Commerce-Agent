#!/usr/bin/env bash
set -euo pipefail

MODEL_SIZE="${1:?usage: run_eval_after_training.sh <0.6b|1.7b> <checkpoint> <run_name>}"
CHECKPOINT="${2:?usage: run_eval_after_training.sh <0.6b|1.7b> <checkpoint> <run_name>}"
RUN_NAME="${3:?usage: run_eval_after_training.sh <0.6b|1.7b> <checkpoint> <run_name>}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python3 "$ROOT/scripts/finetune/infer_finetuned_intent.py" \
  --checkpoint "$CHECKPOINT" \
  --run-name "$RUN_NAME" \
  --input "$ROOT/data/intent_recognition/intent_all.jsonl" \
  --out "$ROOT/finetune/outputs/preds/${RUN_NAME}.jsonl" \
  --metrics-out "$ROOT/finetune/outputs/preds/${RUN_NAME}_metrics.json"
