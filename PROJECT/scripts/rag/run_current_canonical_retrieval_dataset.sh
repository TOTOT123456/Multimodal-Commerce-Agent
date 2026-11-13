#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 LABEL DATASET EXPECTED_ROWS GPU[,GPU...]" >&2
  exit 2
fi

LABEL="$1"
DATASET="$2"
EXPECTED_ROWS="$3"
GPU_SPEC="$4"
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_SPEC"
[[ "${#GPU_ARRAY[@]}" -ge 1 ]] || { echo "at least one GPU is required" >&2; exit 2; }
for gpu in "${GPU_ARRAY[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "invalid GPU index: $gpu" >&2; exit 2; }
done
VECTOR_GPU="${GPU_ARRAY[0]}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="/root/miniconda/envs/llm_bge/bin/python"
ROUTE="$PROJECT_DIR/configs/retrieval_canonical_route.json"
RUN_ROOT="$PROJECT_DIR/data/rag/_aux/current_canonical_dataset_retrieval/$LABEL"
DOCS="$PROJECT_DIR/data/rag/tool_corpus.jsonl"
DB_ROOT="$PROJECT_DIR/data/rag/_aux/retrieval_pipeline_reselection/frozen_vector_db"
DB_KEY="frozen_selected_embedding"
RERANKER="$(jq -r '.reranker.checkpoint' "$ROUTE")"
RERANK_K="$(jq -r '.reranker.candidate_k' "$ROUTE")"
if [[ "$RERANKER" != /* ]]; then
  RERANKER="$PROJECT_DIR/${RERANKER#PROJECT/}"
fi

[[ -f "$DATASET" ]] || { echo "missing dataset: $DATASET" >&2; exit 2; }
[[ "$(wc -l < "$DATASET")" -eq "$EXPECTED_ROWS" ]] || {
  echo "dataset row count differs from $EXPECTED_ROWS: $DATASET" >&2
  exit 2
}
[[ -f "$ROUTE" && "$(jq -r '.status' "$ROUTE")" == "canonical" ]] || {
  echo "canonical route is missing or unlocked" >&2
  exit 2
}
[[ -d "$RERANKER" ]] || { echo "missing reranker: $RERANKER" >&2; exit 2; }
[[ -f "$DB_ROOT/$DB_KEY/index.faiss" ]] || {
  echo "missing canonical embedding index: $DB_ROOT/$DB_KEY" >&2
  exit 2
}

mkdir -p "$RUN_ROOT/logs"
BM25_DIR="$RUN_ROOT/bm25"
VECTOR_DIR="$RUN_ROOT/vector"
SOURCE_DIR="$RUN_ROOT/first_stage"
RERANK_DIR="$RUN_ROOT/reranker"
FINAL_DIR="$RUN_ROOT/final"
BM25_PREDS="$BM25_DIR/preds/bm25.jsonl"
VECTOR_PREDS="$VECTOR_DIR/preds/vector_${DB_KEY}.jsonl"
SOURCE_PREDS="$SOURCE_DIR/preds.jsonl"
RERANK_PREDS="$RERANK_DIR/preds.jsonl"

complete_jsonl() {
  local path="$1"
  [[ -f "$path" && "$(wc -l < "$path")" -eq "$EXPECTED_ROWS" ]]
}

pids=()
if ! complete_jsonl "$BM25_PREDS"; then
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/rag/run_retrieval_baselines.py" \
    --test-file "$DATASET" \
    --output-dir "$BM25_DIR" \
    --compat-metrics-out "$BM25_DIR/compat_metrics.json" \
    --mode bm25 \
    --ks 6 10 20 30 40 50 100 \
    --candidate-k 100 \
    --tokenizer fallback \
    >"$RUN_ROOT/logs/bm25.log" 2>&1 &
  pids+=("$!")
fi
if ! complete_jsonl "$VECTOR_PREDS"; then
  (
    export CUDA_VISIBLE_DEVICES="$VECTOR_GPU"
    "$PYTHON_BIN" "$PROJECT_DIR/scripts/rag/run_retrieval_baselines.py" \
      --test-file "$DATASET" \
      --vector-db-dir "$DB_ROOT" \
      --models "$DB_KEY" \
      --output-dir "$VECTOR_DIR" \
      --compat-metrics-out "$VECTOR_DIR/compat_metrics.json" \
      --mode vector \
      --ks 6 10 20 30 40 50 100 \
      --candidate-k 100 \
      --batch-size 64 \
      --device cuda:0 \
      --tokenizer fallback
  ) >"$RUN_ROOT/logs/vector.log" 2>&1 &
  pids+=("$!")
fi
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
[[ "$status" -eq 0 ]] || { echo "first-stage baseline failed" >&2; exit 1; }
complete_jsonl "$BM25_PREDS" || { echo "incomplete BM25 output" >&2; exit 1; }
complete_jsonl "$VECTOR_PREDS" || { echo "incomplete vector output" >&2; exit 1; }

"$PYTHON_BIN" "$PROJECT_DIR/scripts/rag/apply_first_stage_route.py" \
  --vector-preds "$VECTOR_PREDS" \
  --bm25-preds "$BM25_PREDS" \
  --route-json "$ROUTE" \
  --route-field first_stage \
  --out-dir "$SOURCE_DIR"

if ! complete_jsonl "$RERANK_PREDS"; then
  if [[ "${#GPU_ARRAY[@]}" -eq 1 ]]; then
    CUDA_VISIBLE_DEVICES="$VECTOR_GPU" "$PYTHON_BIN" \
      "$PROJECT_DIR/scripts/rag/run_reranker_baseline.py" \
      --preds "$SOURCE_PREDS" \
      --docs "$DOCS" \
      --output-dir "$RERANK_DIR" \
      --model "$RERANKER" \
      --backend sentence-transformers \
      --candidate-k "$RERANK_K" \
      --ks 6 10 20 30 40 50 \
      --query-batch-size 64 \
      --pair-batch-size 64 \
      --max-length 512 \
      --device cuda:0 \
      --use-fp16 \
      >"$RUN_ROOT/logs/reranker.log" 2>&1
  else
    pids=()
    shard_args=()
    shard_count="${#GPU_ARRAY[@]}"
    for shard_index in "${!GPU_ARRAY[@]}"; do
      shard_dir="$RERANK_DIR/shard$shard_index"
      shard_pred="$shard_dir/preds.jsonl"
      shard_args+=(--reranker-shard "$shard_pred")
      if [[ ! -s "$shard_pred" ]]; then
        (
          export CUDA_VISIBLE_DEVICES="${GPU_ARRAY[$shard_index]}"
          "$PYTHON_BIN" "$PROJECT_DIR/scripts/rag/run_reranker_baseline.py" \
            --preds "$SOURCE_PREDS" \
            --docs "$DOCS" \
            --output-dir "$shard_dir" \
            --model "$RERANKER" \
            --backend sentence-transformers \
            --candidate-k "$RERANK_K" \
            --ks 6 10 20 30 40 50 \
            --query-batch-size 64 \
            --pair-batch-size 64 \
            --max-length 512 \
            --device cuda:0 \
            --use-fp16 \
            --shard-count "$shard_count" \
            --shard-index "$shard_index"
        ) >"$RUN_ROOT/logs/reranker_shard${shard_index}.log" 2>&1 &
        pids+=("$!")
      fi
    done
    status=0
    for pid in "${pids[@]}"; do
      wait "$pid" || status=1
    done
    [[ "$status" -eq 0 ]] || { echo "reranker shard failed" >&2; exit 1; }
    python3 "$PROJECT_DIR/scripts/finetune/prepare_assignment3_current_dpo_data.py" \
      --stage merge-reranker \
      --output-dir "$RUN_ROOT" \
      --first-stage-preds "$SOURCE_PREDS" \
      "${shard_args[@]}" \
      --merged-reranker-out "$RERANK_PREDS"
  fi
fi
complete_jsonl "$RERANK_PREDS" || { echo "incomplete reranker output" >&2; exit 1; }

"$PYTHON_BIN" "$PROJECT_DIR/scripts/rag/apply_frozen_retrieval_pipeline.py" \
  --vector-preds "$VECTOR_PREDS" \
  --bm25-preds "$BM25_PREDS" \
  --reranker-preds "$RERANK_PREDS" \
  --route-json "$ROUTE" \
  --dataset "$DATASET" \
  --out-dir "$FINAL_DIR"

python3 "$PROJECT_DIR/scripts/finetune/sanitize_tool_trace_eval_inputs.py" \
  --split test \
  --input "$DATASET" \
  --rankings "$FINAL_DIR/preds.jsonl" \
  --candidate-k 10 \
  --query-out "$RUN_ROOT/queries_label_free.jsonl" \
  --rag-out "$RUN_ROOT/rag10_label_free.jsonl" \
  --audit-out "$RUN_ROOT/label_free_audit.json"

echo "current canonical retrieval complete label=$LABEL rows=$EXPECTED_ROWS"
