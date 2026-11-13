#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PROJECT_DIR="$ROOT_DIR/PROJECT"
RUN_ROOT="$PROJECT_DIR/data/tool_call_trace/_aux/assignment3/canonical_planner_candidate_depth_isolated_20260801"
OUT_DIR="$RUN_ROOT/retrieval"
DATASET="$RUN_ROOT/dev.jsonl"
DATA_AUDIT="$RUN_ROOT/data_audit.json"
FROZEN="$PROJECT_DIR/data/rag/_aux/retrieval_pipeline_reselection/frozen_pipeline.json"
DOCS="$PROJECT_DIR/data/rag/tool_corpus.jsonl"
DB_ROOT="$PROJECT_DIR/data/rag/_aux/retrieval_pipeline_reselection/frozen_vector_db"
DB_KEY="frozen_selected_embedding"
DB_PATH="$DB_ROOT/$DB_KEY"
PYTHON_BIN="/root/miniconda/envs/llm_bge/bin/python"
EXPECTED_ROWS=1200

if [[ "$(jq -r '.strict_ok' "$DATA_AUDIT")" != true ]]; then
  echo "isolated development data audit is not strict" >&2
  exit 1
fi
if [[ "$(wc -l < "$DATASET")" -ne "$EXPECTED_ROWS" ]]; then
  echo "isolated development dataset must contain $EXPECTED_ROWS rows" >&2
  exit 1
fi

EMBEDDING="$(jq -r '.embedding.checkpoint' "$FROZEN")"
RERANKER="$(jq -r '.reranker.model' "$FROZEN")"
CANDIDATE_K="$(jq -r '.selected.route.candidate_k' "$FROZEN")"
mkdir -p "$OUT_DIR/logs"

BM25_DIR="$OUT_DIR/bm25"
VECTOR_DIR="$OUT_DIR/vector"
SOURCE_DIR="$OUT_DIR/source"
RERANKER_DIR="$OUT_DIR/reranker"
BM25_PREDS="$BM25_DIR/preds/bm25.jsonl"
VECTOR_PREDS="$VECTOR_DIR/preds/vector_${DB_KEY}.jsonl"
SOURCE_PREDS="$SOURCE_DIR/preds.jsonl"
RERANKER_PREDS="$RERANKER_DIR/preds.jsonl"

complete_jsonl() {
  local path="$1"
  [[ -f "$path" ]] && [[ "$(wc -l < "$path")" -eq "$EXPECTED_ROWS" ]]
}

pids=()
if ! complete_jsonl "$BM25_PREDS"; then
  conda run -n llm_bge python "$PROJECT_DIR/scripts/rag/run_retrieval_baselines.py" \
    --test-file "$DATASET" \
    --output-dir "$BM25_DIR" \
    --compat-metrics-out "$BM25_DIR/compat_metrics.json" \
    --mode bm25 --ks 6 10 20 30 40 50 100 --candidate-k 100 \
    --tokenizer fallback >"$OUT_DIR/logs/bm25.log" 2>&1 &
  pids+=("$!")
fi

if ! complete_jsonl "$VECTOR_PREDS"; then
  (
    export CUDA_VISIBLE_DEVICES="${VECTOR_GPU:-0}"
    if [[ ! -f "$DB_PATH/index.faiss" ]]; then
      conda run -n llm_bge python "$PROJECT_DIR/scripts/rag/build_vector_dbs.py" \
        --models "$DB_KEY" --model-path "$DB_KEY=$EMBEDDING" \
        --output-dir "$DB_ROOT" --device cuda:0 --batch-size 64
    fi
    conda run -n llm_bge python "$PROJECT_DIR/scripts/rag/run_retrieval_baselines.py" \
      --test-file "$DATASET" --vector-db-dir "$DB_ROOT" --models "$DB_KEY" \
      --output-dir "$VECTOR_DIR" \
      --compat-metrics-out "$VECTOR_DIR/compat_metrics.json" \
      --mode vector --ks 6 10 20 30 40 50 100 --candidate-k 100 \
      --batch-size 64 --device cuda:0 --tokenizer fallback
  ) >"$OUT_DIR/logs/vector.log" 2>&1 &
  pids+=("$!")
fi

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "BM25/vector retrieval failed; inspect $OUT_DIR/logs" >&2
  exit 1
fi

for path in "$BM25_PREDS" "$VECTOR_PREDS"; do
  if ! complete_jsonl "$path"; then
    echo "incomplete retrieval input: $path" >&2
    exit 1
  fi
done

"$PYTHON_BIN" "$PROJECT_DIR/scripts/rag/apply_first_stage_route.py" \
  --vector-preds "$VECTOR_PREDS" --bm25-preds "$BM25_PREDS" \
  --route-json "$FROZEN" --route-field selected.route --out-dir "$SOURCE_DIR"

if ! complete_jsonl "$RERANKER_PREDS"; then
  (
    export CUDA_VISIBLE_DEVICES="${RERANKER_GPU:-1}"
    conda run -n llm_bge python "$PROJECT_DIR/scripts/rag/run_reranker_baseline.py" \
      --preds "$SOURCE_PREDS" --docs "$DOCS" --output-dir "$RERANKER_DIR" \
      --model "$RERANKER" --backend sentence-transformers \
      --candidate-k "$CANDIDATE_K" --ks 6 10 20 30 40 50 \
      --query-batch-size 64 --pair-batch-size 64 --max-length 512 \
      --device cuda:0 --use-fp16
  ) >"$OUT_DIR/logs/reranker.log" 2>&1
fi
if ! complete_jsonl "$RERANKER_PREDS"; then
  echo "incomplete reranker output" >&2
  exit 1
fi

"$PYTHON_BIN" "$PROJECT_DIR/scripts/rag/apply_frozen_retrieval_pipeline.py" \
  --vector-preds "$VECTOR_PREDS" --bm25-preds "$BM25_PREDS" \
  --reranker-preds "$RERANKER_PREDS" --route-json "$FROZEN" \
  --dataset "$DATASET" --out-dir "$OUT_DIR/challenger" \
  --complete-ranking-to 50

if [[ "$(jq -r '.overall.min_output_size' "$OUT_DIR/challenger/metrics.json")" -lt 50 ]]; then
  echo "canonical ranking is shorter than K=50 for at least one row" >&2
  exit 1
fi
echo "isolated Planner-K retrieval complete: $OUT_DIR/challenger/metrics.json"
