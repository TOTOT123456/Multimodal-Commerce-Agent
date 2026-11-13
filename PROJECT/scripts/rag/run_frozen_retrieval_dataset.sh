#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "usage: $0 LABEL DATASET EXPECTED_ROWS SELECTED_RERANK_GPU BASELINE_RERANK_GPU SELECTED_VECTOR_GPU BASELINE_VECTOR_GPU" >&2
  exit 2
fi

LABEL="$1"
DATASET="$2"
EXPECTED_ROWS="$3"
SELECTED_RERANK_GPU="$4"
BASELINE_RERANK_GPU="$5"
SELECTED_VECTOR_GPU="$6"
BASELINE_VECTOR_GPU="$7"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PROJECT_DIR="$ROOT_DIR/PROJECT"
RUN_ROOT="$PROJECT_DIR/data/rag/_aux/retrieval_pipeline_reselection"
RAG_PYTHON="/root/miniconda/envs/llm_bge/bin/python"
FROZEN="$RUN_ROOT/frozen_pipeline.json"
DOCS="$PROJECT_DIR/data/rag/tool_corpus.jsonl"
ORIGINAL_ROUTE="$RUN_ROOT/bs64_control_route.snapshot.json"
ORIGINAL_DB_ROOT="$PROJECT_DIR/data/rag/vector_dbs_finetuned"
ORIGINAL_DB_KEY="bge-large-mnrl-e1-lr2e-5-bs64"
ORIGINAL_RERANKER="$PROJECT_DIR/data/rag/_aux/reranker_finetune/bge-reranker-base-e1-g8-bs16"
SELECTED_DB_ROOT="$RUN_ROOT/frozen_vector_db"
SELECTED_DB_KEY="frozen_selected_embedding"
SELECTED_DB="$SELECTED_DB_ROOT/$SELECTED_DB_KEY"
OUT_DIR="$RUN_ROOT/evaluation/$LABEL"

if [[ ! -f "$FROZEN" ]]; then
  echo "pipeline must be frozen before evaluating $LABEL" >&2
  exit 1
fi
if [[ "$(jq -r '.status' "$FROZEN")" != "frozen_before_round_final" ]]; then
  echo "invalid frozen pipeline status" >&2
  exit 1
fi

SELECTED_EMBEDDING="$(jq -r '.embedding.checkpoint' "$FROZEN")"
SELECTED_RERANKER="$(jq -r '.reranker.model' "$FROZEN")"
SELECTED_GROUP="$(jq -r '.reranker.group' "$FROZEN")"
SELECTED_CANDIDATE_K="$(jq -r '.selected.route.candidate_k' "$FROZEN")"
BASELINE_CANDIDATE_K="$(jq -r '.route.candidate_k' "$ORIGINAL_ROUTE")"
SELECTED_PAIR_BATCH=64
if [[ "$SELECTED_GROUP" == qwen3_0p6b* ]]; then
  SELECTED_PAIR_BATCH=32
fi

mkdir -p "$OUT_DIR/logs"
python3 "$PROJECT_DIR/scripts/rag/record_frozen_retrieval_evaluation.py" \
  --stage begin \
  --label "$LABEL" \
  --dataset "$DATASET" \
  --expected-rows "$EXPECTED_ROWS" \
  --frozen "$FROZEN" \
  --out-dir "$OUT_DIR"

complete_jsonl() {
  local path="$1"
  [[ -f "$path" ]] && [[ "$(wc -l < "$path")" -eq "$EXPECTED_ROWS" ]]
}

BM25_OUT="$OUT_DIR/bm25"
SELECTED_VECTOR_OUT="$OUT_DIR/vector_selected"
BASELINE_VECTOR_OUT="$OUT_DIR/vector_baseline"
SELECTED_SOURCE_OUT="$OUT_DIR/source_selected"
BASELINE_SOURCE_OUT="$OUT_DIR/source_baseline"
SELECTED_RERANK_OUT="$OUT_DIR/reranker_selected"
BASELINE_RERANK_OUT="$OUT_DIR/reranker_baseline"
BM25_PREDS="$BM25_OUT/preds/bm25.jsonl"
SELECTED_VECTOR_PREDS="$SELECTED_VECTOR_OUT/preds/vector_${SELECTED_DB_KEY}.jsonl"
BASELINE_VECTOR_PREDS="$BASELINE_VECTOR_OUT/preds/vector_${ORIGINAL_DB_KEY}.jsonl"
SELECTED_SOURCE_PREDS="$SELECTED_SOURCE_OUT/preds.jsonl"
BASELINE_SOURCE_PREDS="$BASELINE_SOURCE_OUT/preds.jsonl"
SELECTED_RERANK_PREDS="$SELECTED_RERANK_OUT/preds.jsonl"
BASELINE_RERANK_PREDS="$BASELINE_RERANK_OUT/preds.jsonl"

pids=()
if ! complete_jsonl "$BM25_PREDS"; then
  conda run -n llm_bge python "$PROJECT_DIR/scripts/rag/run_retrieval_baselines.py" \
    --test-file "$DATASET" \
    --output-dir "$BM25_OUT" \
    --compat-metrics-out "$BM25_OUT/compat_metrics.json" \
    --mode bm25 \
    --ks 6 10 20 30 40 50 100 \
    --candidate-k 100 \
    --tokenizer fallback \
    >"$OUT_DIR/logs/bm25.log" 2>&1 &
  pids+=("$!")
fi

if ! complete_jsonl "$SELECTED_VECTOR_PREDS"; then
  (
    export CUDA_VISIBLE_DEVICES="$SELECTED_VECTOR_GPU"
    if [[ ! -f "$SELECTED_DB/index.faiss" ]]; then
      conda run -n llm_bge python "$PROJECT_DIR/scripts/rag/build_vector_dbs.py" \
        --models "$SELECTED_DB_KEY" \
        --model-path "$SELECTED_DB_KEY=$SELECTED_EMBEDDING" \
        --output-dir "$SELECTED_DB_ROOT" \
        --device cuda:0 \
        --batch-size 64
    fi
    conda run -n llm_bge python "$PROJECT_DIR/scripts/rag/run_retrieval_baselines.py" \
      --test-file "$DATASET" \
      --vector-db-dir "$SELECTED_DB_ROOT" \
      --models "$SELECTED_DB_KEY" \
      --output-dir "$SELECTED_VECTOR_OUT" \
      --compat-metrics-out "$SELECTED_VECTOR_OUT/compat_metrics.json" \
      --mode vector \
      --ks 6 10 20 30 40 50 100 \
      --candidate-k 100 \
      --batch-size 64 \
      --device cuda:0 \
      --tokenizer fallback
  ) >"$OUT_DIR/logs/vector_selected.log" 2>&1 &
  pids+=("$!")
fi

if ! complete_jsonl "$BASELINE_VECTOR_PREDS"; then
  (
    export CUDA_VISIBLE_DEVICES="$BASELINE_VECTOR_GPU"
    conda run -n llm_bge python "$PROJECT_DIR/scripts/rag/run_retrieval_baselines.py" \
      --test-file "$DATASET" \
      --vector-db-dir "$ORIGINAL_DB_ROOT" \
      --models "$ORIGINAL_DB_KEY" \
      --output-dir "$BASELINE_VECTOR_OUT" \
      --compat-metrics-out "$BASELINE_VECTOR_OUT/compat_metrics.json" \
      --mode vector \
      --ks 6 10 20 30 40 50 100 \
      --candidate-k 100 \
      --batch-size 64 \
      --device cuda:0 \
      --tokenizer fallback
  ) >"$OUT_DIR/logs/vector_baseline.log" 2>&1 &
  pids+=("$!")
fi

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "one or more frozen first-stage workers failed; inspect $OUT_DIR/logs" >&2
  exit 1
fi

for path in "$BM25_PREDS" "$SELECTED_VECTOR_PREDS" "$BASELINE_VECTOR_PREDS"; do
  if ! complete_jsonl "$path"; then
    echo "incomplete evaluation input: $path" >&2
    exit 1
  fi
done

"$RAG_PYTHON" "$PROJECT_DIR/scripts/rag/apply_first_stage_route.py" \
  --vector-preds "$SELECTED_VECTOR_PREDS" \
  --bm25-preds "$BM25_PREDS" \
  --route-json "$FROZEN" \
  --route-field selected.route \
  --out-dir "$SELECTED_SOURCE_OUT"

"$RAG_PYTHON" "$PROJECT_DIR/scripts/rag/apply_first_stage_route.py" \
  --vector-preds "$BASELINE_VECTOR_PREDS" \
  --bm25-preds "$BM25_PREDS" \
  --route-json "$ORIGINAL_ROUTE" \
  --route-field route \
  --out-dir "$BASELINE_SOURCE_OUT"

rerank_pids=()
if ! complete_jsonl "$SELECTED_RERANK_PREDS" \
   || [[ ! -f "$SELECTED_RERANK_OUT/metrics.json" ]] \
   || [[ "$(jq -r '.candidate_k' "$SELECTED_RERANK_OUT/metrics.json")" -ne "$SELECTED_CANDIDATE_K" ]]; then
  (
    export CUDA_VISIBLE_DEVICES="$SELECTED_RERANK_GPU"
    conda run -n llm_bge python "$PROJECT_DIR/scripts/rag/run_reranker_baseline.py" \
      --preds "$SELECTED_SOURCE_PREDS" \
      --docs "$DOCS" \
      --output-dir "$SELECTED_RERANK_OUT" \
      --model "$SELECTED_RERANKER" \
      --backend sentence-transformers \
      --candidate-k "$SELECTED_CANDIDATE_K" \
      --ks 6 10 20 30 40 50 \
      --query-batch-size 64 \
      --pair-batch-size "$SELECTED_PAIR_BATCH" \
      --max-length 512 \
      --device cuda:0 \
      --use-fp16
  ) >"$OUT_DIR/logs/reranker_selected.log" 2>&1 &
  rerank_pids+=("$!")
fi

if ! complete_jsonl "$BASELINE_RERANK_PREDS" \
   || [[ ! -f "$BASELINE_RERANK_OUT/metrics.json" ]] \
   || [[ "$(jq -r '.candidate_k' "$BASELINE_RERANK_OUT/metrics.json")" -ne "$BASELINE_CANDIDATE_K" ]]; then
  (
    export CUDA_VISIBLE_DEVICES="$BASELINE_RERANK_GPU"
    conda run -n llm_bge python "$PROJECT_DIR/scripts/rag/run_reranker_baseline.py" \
      --preds "$BASELINE_SOURCE_PREDS" \
      --docs "$DOCS" \
      --output-dir "$BASELINE_RERANK_OUT" \
      --model "$ORIGINAL_RERANKER" \
      --backend sentence-transformers \
      --candidate-k "$BASELINE_CANDIDATE_K" \
      --ks 6 10 20 30 40 50 \
      --query-batch-size 64 \
      --pair-batch-size 128 \
      --max-length 512 \
      --device cuda:0 \
      --use-fp16
  ) >"$OUT_DIR/logs/reranker_baseline.log" 2>&1 &
  rerank_pids+=("$!")
fi

rerank_failed=0
for pid in "${rerank_pids[@]}"; do
  if ! wait "$pid"; then
    rerank_failed=1
  fi
done
if [[ "$rerank_failed" -ne 0 ]]; then
  echo "one or more frozen candidate-only reranker workers failed; inspect $OUT_DIR/logs" >&2
  exit 1
fi

for path in "$SELECTED_SOURCE_PREDS" "$BASELINE_SOURCE_PREDS" "$SELECTED_RERANK_PREDS" "$BASELINE_RERANK_PREDS"; do
  if ! complete_jsonl "$path"; then
    echo "incomplete candidate-only evaluation input: $path" >&2
    exit 1
  fi
done

"$RAG_PYTHON" "$PROJECT_DIR/scripts/rag/apply_frozen_retrieval_pipeline.py" \
  --vector-preds "$SELECTED_VECTOR_PREDS" \
  --bm25-preds "$BM25_PREDS" \
  --reranker-preds "$SELECTED_RERANK_PREDS" \
  --route-json "$FROZEN" \
  --dataset "$DATASET" \
  --out-dir "$OUT_DIR/challenger"

"$RAG_PYTHON" "$PROJECT_DIR/scripts/rag/apply_frozen_retrieval_pipeline.py" \
  --vector-preds "$BASELINE_VECTOR_PREDS" \
  --bm25-preds "$BM25_PREDS" \
  --reranker-preds "$BASELINE_RERANK_PREDS" \
  --route-json "$ORIGINAL_ROUTE" \
  --dataset "$DATASET" \
  --out-dir "$OUT_DIR/baseline"

"$RAG_PYTHON" "$PROJECT_DIR/scripts/rag/compare_retrieval_pipeline_predictions.py" \
  --baseline "$OUT_DIR/baseline/preds.jsonl" \
  --challenger "$OUT_DIR/challenger/preds.jsonl" \
  --dataset "$DATASET" \
  --out "$OUT_DIR/comparison.json" \
  --bootstrap-samples 20000 \
  --seed 20260731 \
  --noninferiority-margin -0.002

python3 "$PROJECT_DIR/scripts/rag/record_frozen_retrieval_evaluation.py" \
  --stage complete \
  --label "$LABEL" \
  --dataset "$DATASET" \
  --expected-rows "$EXPECTED_ROWS" \
  --frozen "$FROZEN" \
  --out-dir "$OUT_DIR"

echo "complete frozen retrieval evaluation label=$LABEL rows=$EXPECTED_ROWS"
