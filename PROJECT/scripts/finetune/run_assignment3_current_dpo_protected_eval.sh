#!/usr/bin/env bash
set -euo pipefail

MODEL_SIZE="${1:?usage: run_assignment3_current_dpo_protected_eval.sh <4b|8b> <name> <adapter>}"
NAME="${2:?missing endpoint name}"
ADAPTER="${3:?missing adapter}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_ID="assignment3_current_api9k_dpo_20260802"
EXP_DIR="$PROJECT_DIR/data/tool_call_trace/_aux/assignment3/$EXP_ID"
SOURCE_DIR="$PROJECT_DIR/data/tool_call_trace/_aux/assignment3/r20p10_second_rrf_optimization_20260801/protected_after_promotion"
PRED_DIR="$PROJECT_DIR/finetune/outputs/preds/tool_trace/$EXP_ID/protected/$MODEL_SIZE/$NAME"
RUNNER="$PROJECT_DIR/scripts/finetune/run_assignment3_eval.sh"
EVALUATOR="$PROJECT_DIR/scripts/finetune/evaluate_tool_trace.py"
SYSTEM_EVALUATOR="$PROJECT_DIR/scripts/finetune/evaluate_tool_trace_system.py"
INTENT_PREDS="${INTENT_PREDS:-$PROJECT_DIR/finetune/outputs/preds/qwen3-0.6b-full_lora-context_isolated-best.jsonl}"
PYTHON_BIN="/root/miniconda/envs/qwen-intent-ft/bin/python"

[[ "$NAME" =~ ^[a-zA-Z0-9._-]+$ ]] || { echo "invalid endpoint name" >&2; exit 2; }
[[ -d "$ADAPTER" ]] || { echo "missing adapter: $ADAPTER" >&2; exit 2; }
case "$MODEL_SIZE" in
  4b)
    MODEL="$PROJECT_DIR/models/Qwen/Qwen3-4B"
    RAG_BATCH=16
    # Global view exposes all 100 tools and has a materially larger prompt.
    # Batch 8 reached ~22.7 GiB in the full-run probe, so keep headroom.
    GLOBAL_BATCH=6
    ;;
  8b)
    MODEL="$PROJECT_DIR/models/Qwen/Qwen3-8B"
    RAG_BATCH=8
    # Full-run probes at batch 3/4 approached the 24 GiB device limit.
    GLOBAL_BATCH=2
    ;;
  *) echo "MODEL_SIZE must be 4b or 8b" >&2; exit 2 ;;
esac

gold_path() {
  case "$1" in
    legacy6000) printf '%s\n' "$PROJECT_DIR/data/tool_call_trace/test.jsonl" ;;
    api1000) printf '%s\n' "$PROJECT_DIR/data/tool_call_trace/_aux/assignment3/assignment3_100tool_api/test_api/test.jsonl" ;;
  esac
}

input_path() {
  local split="$1" view="$2"
  if [[ "$view" == "rag" ]]; then
    printf '%s\n' "$SOURCE_DIR/$split/inputs/rag_r20p10_w040_k008.jsonl"
  else
    printf '%s\n' "$SOURCE_DIR/$split/inputs/queries_label_free.jsonl"
  fi
}

run_cell() {
  local split="$1" view="$2" gpus="$3" batch="$4"
  local output="$PRED_DIR/$split/$view.jsonl"
  local run_id="current-dpo-protected-$MODEL_SIZE-$NAME-$split-$view"
  mkdir -p "$(dirname "$output")" "$EXP_DIR/evaluation/protected/$MODEL_SIZE/$NAME/$split"
  PYTHON_BIN="$PYTHON_BIN" GPU_LIST="$gpus" BATCH_SIZE="$batch" \
    MAX_NEW_TOKENS=128 SAVE_EVERY=200 SEED=20260725 \
    bash "$RUNNER" "$MODEL" "$ADAPTER" "$(input_path "$split" "$view")" \
      "$view" "$output" "$run_id" >"${output%.jsonl}.infer.log" 2>&1
}

run_cell legacy6000 global 0,1 "$GLOBAL_BATCH" & p0=$!
run_cell legacy6000 rag 2,3 "$RAG_BATCH" & p1=$!
(
  run_cell api1000 global 4,5 "$GLOBAL_BATCH"
  run_cell api1000 rag 4,5 "$RAG_BATCH"
) & p2=$!
status=0
for pid in "$p0" "$p1" "$p2"; do wait "$pid" || status=1; done
[[ "$status" -eq 0 ]] || { echo "protected inference failed" >&2; exit 1; }

for split in legacy6000 api1000; do
  for view in global rag; do
    output="$PRED_DIR/$split/$view.jsonl"
    eval_dir="$EXP_DIR/evaluation/protected/$MODEL_SIZE/$NAME/$split"
    "$PYTHON_BIN" "$EVALUATOR" --predictions "$output" \
      --gold "$(gold_path "$split")" \
      --metrics-out "$eval_dir/$view.metrics.json" \
      --scored-out "$eval_dir/$view.scored.jsonl" \
      --run-id "current-dpo-protected-$MODEL_SIZE-$NAME-$split-$view"
  done
done

"$PYTHON_BIN" "$SYSTEM_EVALUATOR" \
  --intent-predictions "$INTENT_PREDS" \
  --tool-predictions "$PRED_DIR/legacy6000/rag.jsonl" \
  --out "$EXP_DIR/evaluation/protected/$MODEL_SIZE/$NAME/legacy6000/rag.system.jsonl" \
  --metrics-out "$EXP_DIR/evaluation/protected/$MODEL_SIZE/$NAME/legacy6000/rag.system.metrics.json"

echo "current DPO protected evaluation complete model=$MODEL_SIZE endpoint=$NAME"
