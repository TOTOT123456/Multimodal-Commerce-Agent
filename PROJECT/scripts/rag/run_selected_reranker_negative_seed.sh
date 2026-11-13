#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 GPU_ID NEGATIVE_RECIPE SEED" >&2
  exit 2
fi

GPU_ID="$1"
NEGATIVE_RECIPE="$2"
SEED="$3"
if [[ "$NEGATIVE_RECIPE" != "source_mined7" && "$NEGATIVE_RECIPE" != "mixed_typed3_mined4" ]]; then
  echo "negative recipe must be source_mined7 or mixed_typed3_mined4" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PROJECT_DIR="$ROOT_DIR/PROJECT"
RUN_ROOT="$PROJECT_DIR/data/rag/_aux/retrieval_pipeline_reselection"
ARCH_JSON="$RUN_ROOT/downstream/architecture_screen/pipeline_search.json"
SELECTED_GROUP="$(jq -r '.selected.group' "$ARCH_JSON")"
case "$SELECTED_GROUP" in
  existing_task_bge_base)
    MODEL_KEY="bge_base"
    ;;
  bge_base|bge_large|bge_v2_m3|qwen3_0p6b)
    MODEL_KEY="$SELECTED_GROUP"
    ;;
  *)
    echo "unknown selected architecture group: $SELECTED_GROUP" >&2
    exit 1
    ;;
esac

LR_GROUP="$(jq -r '.selected.group' "$RUN_ROOT/downstream/lr_screen/$MODEL_KEY/pipeline_search.json")"
case "$LR_GROUP" in
  *_lr2e5) LEARNING_RATE="2e-5" ;;
  *_lr6e5) LEARNING_RATE="6e-5" ;;
  *_lr5e6) LEARNING_RATE="5e-6" ;;
  *_lr1e5) LEARNING_RATE="1e-5" ;;
  *)
    echo "cannot resolve learning rate from selected group: $LR_GROUP" >&2
    exit 1
    ;;
esac

MINED_PREDS="$RUN_ROOT/downstream/train_mining_source/preds.jsonl"
bash "$PROJECT_DIR/scripts/rag/run_reranker_reselection_seed.sh" \
  "$GPU_ID" "$MODEL_KEY" "$NEGATIVE_RECIPE" "$LEARNING_RATE" "$SEED" "$MINED_PREDS"
bash "$PROJECT_DIR/scripts/rag/run_reranker_reselection_eval.sh" \
  "$GPU_ID" "$MODEL_KEY" "$NEGATIVE_RECIPE" "$LEARNING_RATE" "$SEED"

echo "complete negative recipe=$NEGATIVE_RECIPE model=$MODEL_KEY lr=$LEARNING_RATE seed=$SEED gpu=$GPU_ID"
