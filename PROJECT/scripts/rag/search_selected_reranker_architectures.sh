#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PROJECT_DIR="$ROOT_DIR/PROJECT"
RUN_ROOT="$PROJECT_DIR/data/rag/_aux/retrieval_pipeline_reselection"

selected_lr_tag() {
  local model_key="$1"
  local search_json="$RUN_ROOT/downstream/lr_screen/$model_key/pipeline_search.json"
  local selected_group
  selected_group="$(jq -r '.selected.group' "$search_json")"
  case "$selected_group" in
    *_lr2e5) echo "2em5" ;;
    *_lr6e5) echo "6em5" ;;
    *_lr5e6) echo "5em6" ;;
    *_lr1e5) echo "1em5" ;;
    *)
      echo "cannot resolve LR tag from $selected_group" >&2
      return 1
      ;;
  esac
}

SPECS=()
for MODEL_KEY in bge_base bge_large bge_v2_m3 qwen3_0p6b; do
  LR_TAG="$(selected_lr_tag "$MODEL_KEY")"
  for SEED in 42 20260731 20260801; do
    PREDS="$RUN_ROOT/reranker_dev/${MODEL_KEY}-typed4_random3-lr${LR_TAG}-s${SEED}/preds.jsonl"
    if [[ ! -f "$PREDS" ]]; then
      echo "missing reranker predictions: $PREDS" >&2
      exit 1
    fi
    SPECS+=(--reranker-spec "${MODEL_KEY}|${SEED}|${PREDS}")
  done
done

EXISTING_PREDS="$RUN_ROOT/reranker_dev/existing_task_bge_base_s42/preds.jsonl"
if [[ ! -f "$EXISTING_PREDS" ]]; then
  echo "missing existing task reranker predictions: $EXISTING_PREDS" >&2
  exit 1
fi
SPECS+=(--reranker-spec "existing_task_bge_base|42|${EXISTING_PREDS}")

"/root/miniconda/envs/llm_bge/bin/python" \
  "$PROJECT_DIR/scripts/rag/search_retrieval_pipeline.py" \
  --vector-preds "$RUN_ROOT/downstream/dev_vector/preds/vector_selected_embedding.jsonl" \
  --bm25-preds "$RUN_ROOT/downstream/dev_raw/preds/bm25.jsonl" \
  --compare-group-latency \
  "${SPECS[@]}" \
  --out-dir "$RUN_ROOT/downstream/architecture_screen"

echo "complete reranker architecture search"
