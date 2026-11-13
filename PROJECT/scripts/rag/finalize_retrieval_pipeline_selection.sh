#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PROJECT_DIR="$ROOT_DIR/PROJECT"
RUN_ROOT="$PROJECT_DIR/data/rag/_aux/retrieval_pipeline_reselection"
PLAN="$RUN_ROOT/downstream/negative_interaction_plan.json"
NEGATIVE_JSON="$RUN_ROOT/downstream/negative_screen/pipeline_search.json"
FROZEN="$RUN_ROOT/frozen_pipeline.json"

if [[ "$(jq -r '.required' "$PLAN")" != "true" ]]; then
  python3 "$PROJECT_DIR/scripts/rag/freeze_retrieval_pipeline_selection.py" \
    --selection-json "$NEGATIVE_JSON" \
    --out "$FROZEN"
  exit 0
fi

RECIPE="$(jq -r '.selected_negative_recipe' "$PLAN")"
WINNER_MODEL="$(jq -r '.winner_model_key' "$PLAN")"
RUNNER_MODEL="$(jq -r '.runner_up_model_key' "$PLAN")"
LR_GROUP="$(jq -r '.selected.group' "$RUN_ROOT/downstream/lr_screen/$RUNNER_MODEL/pipeline_search.json")"
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
while IFS=$'\t' read -r SEED PREDS; do
  [[ -n "$SEED" && -f "$PREDS" ]] || {
    echo "missing selected winner predictions: $PREDS" >&2
    exit 1
  }
  SPECS+=(--reranker-spec "${WINNER_MODEL}_${RECIPE}|${SEED}|${PREDS}")
done < <(
  jq -r --arg group "$RECIPE" \
    '.reranker_specs[] | select(.group == $group) | [.seed, .path] | @tsv' \
    "$NEGATIVE_JSON"
)

for SEED in 42 20260731 20260801; do
  PREDS="$RUN_ROOT/reranker_dev/${RUNNER_MODEL}-${RECIPE}-lr${LR_TAG}-s${SEED}/preds.jsonl"
  while [[ ! -f "$PREDS" ]]; do sleep 10; done
  SPECS+=(--reranker-spec "${RUNNER_MODEL}_${RECIPE}|${SEED}|${PREDS}")
done

INTERACTION_DIR="$RUN_ROOT/downstream/negative_interaction_screen"
"/root/miniconda/envs/llm_bge/bin/python" \
  "$PROJECT_DIR/scripts/rag/search_retrieval_pipeline.py" \
  --vector-preds "$RUN_ROOT/downstream/dev_vector/preds/vector_selected_embedding.jsonl" \
  --bm25-preds "$RUN_ROOT/downstream/dev_raw/preds/bm25.jsonl" \
  --compare-group-latency \
  "${SPECS[@]}" \
  --out-dir "$INTERACTION_DIR"

python3 "$PROJECT_DIR/scripts/rag/freeze_retrieval_pipeline_selection.py" \
  --selection-json "$INTERACTION_DIR/pipeline_search.json" \
  --out "$FROZEN"

echo "complete final retrieval-pipeline freeze"
