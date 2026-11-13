#!/usr/bin/env bash
set -euo pipefail

ROLE="${1:?usage: $0 <selection|confirmation> NAME=ADAPTER [...] }"
shift
[[ "$ROLE" == "selection" || "$ROLE" == "confirmation" ]] || {
  echo "role must be selection or confirmation" >&2
  exit 2
}
[[ "$#" -gt 0 ]] || { echo "at least one NAME=ADAPTER is required" >&2; exit 2; }

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_ID="assignment3_rag_residual_dpo_v2_20260803"
EXP_DIR="$PROJECT_DIR/data/tool_call_trace/_aux/assignment3/$EXP_ID"
PANEL_ID="${PANEL_ID:-$ROLE}"
[[ "$PANEL_ID" =~ ^[a-z0-9][a-z0-9_-]*$ ]] || {
  echo "PANEL_ID must be a lowercase slug" >&2
  exit 2
}
PRED_DIR="$PROJECT_DIR/finetune/outputs/preds/tool_trace/$EXP_ID/$PANEL_ID"
MODEL="$PROJECT_DIR/models/Qwen/Qwen3-8B"
RUNNER="$PROJECT_DIR/scripts/finetune/run_assignment3_eval.sh"
EVALUATOR="$PROJECT_DIR/scripts/finetune/evaluate_tool_trace.py"
PYTHON_BIN="/root/miniconda/envs/qwen-intent-ft/bin/python"
INPUT="${INPUT_OVERRIDE:-$EXP_DIR/data/${ROLE}_rag_label_free.jsonl}"
GOLD="${GOLD_OVERRIDE:-$EXP_DIR/plan/${ROLE}_gold.jsonl}"
[[ -f "$INPUT" && -f "$GOLD" ]] || {
  echo "missing panel input or gold: $INPUT / $GOLD" >&2
  exit 2
}

for spec in "$@"; do
  name="${spec%%=*}"
  adapter="${spec#*=}"
  [[ -n "$name" && "$adapter" != "$spec" && -d "$adapter" ]] || {
    echo "invalid endpoint: $spec" >&2
    exit 2
  }
  output="$PRED_DIR/$name/rag.jsonl"
  metrics="$EXP_DIR/evaluation/$PANEL_ID/$name/rag.metrics.json"
  scored="$EXP_DIR/evaluation/$PANEL_ID/$name/rag.scored.jsonl"
  mkdir -p "$(dirname "$output")" "$(dirname "$metrics")"
  PYTHON_BIN="$PYTHON_BIN" \
  GPU_LIST="${GPU_LIST:-0,1,2,3,4,5}" \
  BATCH_SIZE="${BATCH_SIZE:-8}" \
  MAX_NEW_TOKENS=96 \
  SAVE_EVERY=200 \
  SEED=20260803 \
    bash "$RUNNER" "$MODEL" "$adapter" "$INPUT" rag "$output" \
      "residual-dpo-v2-$PANEL_ID-$name" \
      >"${output%.jsonl}.infer.log" 2>&1
  "$PYTHON_BIN" "$EVALUATOR" \
    --predictions "$output" \
    --gold "$GOLD" \
    --metrics-out "$metrics" \
    --scored-out "$scored" \
    --run-id "residual-dpo-v2-$PANEL_ID-$name"
done

echo "residual DPO $PANEL_ID evaluation complete endpoints=$#"
