#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 GPU_ID SEED" >&2
  exit 2
fi

GPU_ID="$1"
SEED="$2"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PROJECT_DIR="$ROOT_DIR/PROJECT"
RUN_ROOT="$PROJECT_DIR/data/rag/_aux/retrieval_pipeline_reselection"
PLAN="$RUN_ROOT/downstream/negative_interaction_plan.json"

if [[ "$(jq -r '.required' "$PLAN")" != "true" ]]; then
  echo "negative interaction not required"
  exit 0
fi
MODEL_KEY="$(jq -r '.runner_up_model_key' "$PLAN")"
NEGATIVE_RECIPE="$(jq -r '.selected_negative_recipe' "$PLAN")"
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

echo "complete interaction model=$MODEL_KEY recipe=$NEGATIVE_RECIPE seed=$SEED gpu=$GPU_ID"
