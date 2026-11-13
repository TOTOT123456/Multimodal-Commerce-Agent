#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PROJECT_DIR="$ROOT_DIR/PROJECT"
RUN_ROOT="$PROJECT_DIR/data/rag/_aux/retrieval_pipeline_reselection"
ARCH_JSON="$RUN_ROOT/downstream/architecture_screen/pipeline_search.json"
SELECTED_GROUP="$(jq -r '.selected.group' "$ARCH_JSON")"
case "$SELECTED_GROUP" in
  existing_task_bge_base) MODEL_KEY="bge_base" ;;
  bge_base|bge_large|bge_v2_m3|qwen3_0p6b) MODEL_KEY="$SELECTED_GROUP" ;;
  *)
    echo "unknown selected architecture group: $SELECTED_GROUP" >&2
    exit 1
    ;;
esac

LR_GROUP="$(jq -r '.selected.group' "$RUN_ROOT/downstream/lr_screen/$MODEL_KEY/pipeline_search.json")"
case "$LR_GROUP" in
  *_lr2e5) LR_TAG="2em5" ;;
  *_lr6e5) LR_TAG="6em5" ;;
  *_lr5e6) LR_TAG="5em6" ;;
  *_lr1e5) LR_TAG="1em5" ;;
  *)
    echo "cannot resolve learning rate from selected group: $LR_GROUP" >&2
    exit 1
    ;;
esac

SPECS=()
for RECIPE in typed4_random3 source_mined7 mixed_typed3_mined4; do
  for SEED in 42 20260731 20260801; do
    PREDS="$RUN_ROOT/reranker_dev/${MODEL_KEY}-${RECIPE}-lr${LR_TAG}-s${SEED}/preds.jsonl"
    if [[ ! -f "$PREDS" ]]; then
      echo "missing reranker predictions: $PREDS" >&2
      exit 1
    fi
    SPECS+=(--reranker-spec "${RECIPE}|${SEED}|${PREDS}")
  done
done

if [[ "$SELECTED_GROUP" == "existing_task_bge_base" ]]; then
  EXISTING_PREDS="$RUN_ROOT/reranker_dev/existing_task_bge_base_s42/preds.jsonl"
  if [[ ! -f "$EXISTING_PREDS" ]]; then
    echo "missing selected existing task reranker predictions: $EXISTING_PREDS" >&2
    exit 1
  fi
  SPECS+=(--reranker-spec "existing_task_bge_base|42|${EXISTING_PREDS}")
fi

"/root/miniconda/envs/llm_bge/bin/python" \
  "$PROJECT_DIR/scripts/rag/search_retrieval_pipeline.py" \
  --vector-preds "$RUN_ROOT/downstream/dev_vector/preds/vector_selected_embedding.jsonl" \
  --bm25-preds "$RUN_ROOT/downstream/dev_raw/preds/bm25.jsonl" \
  "${SPECS[@]}" \
  --out-dir "$RUN_ROOT/downstream/negative_screen"

echo "complete reranker negative-recipe search"
