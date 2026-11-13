#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 GPU_ID MODEL_KEY SEED" >&2
  exit 2
fi

GPU_ID="$1"
MODEL_KEY="$2"
SEED="$3"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PROJECT_DIR="$ROOT_DIR/PROJECT"
SEARCH_JSON="$PROJECT_DIR/data/rag/_aux/retrieval_pipeline_reselection/downstream/lr_screen/$MODEL_KEY/pipeline_search.json"

if [[ ! -f "$SEARCH_JSON" ]]; then
  echo "missing learning-rate selection: $SEARCH_JSON" >&2
  exit 1
fi
SELECTED_GROUP="$(jq -r '.selected.group' "$SEARCH_JSON")"
case "$SELECTED_GROUP" in
  *_lr2e5)
    LEARNING_RATE="2e-5"
    ;;
  *_lr6e5)
    LEARNING_RATE="6e-5"
    ;;
  *_lr5e6)
    LEARNING_RATE="5e-6"
    ;;
  *_lr1e5)
    LEARNING_RATE="1e-5"
    ;;
  *)
    echo "cannot resolve learning rate from selected group: $SELECTED_GROUP" >&2
    exit 1
    ;;
esac

bash "$PROJECT_DIR/scripts/rag/run_reranker_reselection_seed.sh" \
  "$GPU_ID" "$MODEL_KEY" typed4_random3 "$LEARNING_RATE" "$SEED"
bash "$PROJECT_DIR/scripts/rag/run_reranker_reselection_eval.sh" \
  "$GPU_ID" "$MODEL_KEY" typed4_random3 "$LEARNING_RATE" "$SEED"

echo "complete selected architecture seed model=$MODEL_KEY lr=$LEARNING_RATE seed=$SEED gpu=$GPU_ID"
