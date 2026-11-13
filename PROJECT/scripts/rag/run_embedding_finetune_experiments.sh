#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PROJECT_DIR="$ROOT_DIR/PROJECT"

SPECS="${SPECS:-\
bge-base-mnrl-e0.25-lr2e-5-bs64|BAAI/bge-base-zh-v1.5|0.25|2e-5|64|1 \
bge-base-mnrl-e0.5-lr2e-5-bs64|BAAI/bge-base-zh-v1.5|0.5|2e-5|64|1 \
bge-base-mnrl-e1-lr1e-5-bs64|BAAI/bge-base-zh-v1.5|1|1e-5|64|1 \
bge-base-mnrl-e1-lr2e-5-bs64|BAAI/bge-base-zh-v1.5|1|2e-5|64|1 \
bge-base-mnrl-e1-lr3e-5-bs64|BAAI/bge-base-zh-v1.5|1|3e-5|64|1 \
bge-base-mnrl-e1.5-lr2e-5-bs64|BAAI/bge-base-zh-v1.5|1.5|2e-5|64|1 \
bge-large-mnrl-e0.5-lr2e-5-bs64|BAAI/bge-large-zh-v1.5|0.5|2e-5|64|1 \
bge-large-mnrl-e1-lr2e-5-bs64|BAAI/bge-large-zh-v1.5|1|2e-5|64|1 \
qwen3-embedding-0.6b-mnrl-e0.5-lr2e-5-bs32|Qwen/Qwen3-Embedding-0.6B|0.5|2e-5|32|1}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-256}"
DEVICE="${DEVICE:-cuda}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-64}"
BF16_FLAG="${BF16_FLAG:---bf16}"
KS=(${KS:-6 7 8 9 10 15 20 30 50})
CANDIDATE_NS=(${CANDIDATE_NS:-10 20 30 50})
RERANK_BEST="${RERANK_BEST:-1}"
BEST_RUN="${BEST_RUN:-bge-large-mnrl-e1-lr2e-5-bs64}"
RERANKER_MODEL="${RERANKER_MODEL:-BAAI/bge-reranker-base}"
RERANKER_MODEL_CLASS="${RERANKER_MODEL_CLASS:-encoder-only-base}"

train_eval_one() {
  local run_name="$1"
  local model="$2"
  local epochs="$3"
  local lr="$4"
  local train_batch_size="$5"
  local grad_acc="$6"
  local run_dir="$PROJECT_DIR/data/rag/_aux/embedding_finetune/$run_name"
  local model_out="$run_dir/model"
  local vector_db_dir="$PROJECT_DIR/data/rag/vector_dbs_finetuned"
  local eval_out="$run_dir/retrieval"

  conda run -n llm_bge python "$PROJECT_DIR/scripts/rag/train_embedding.py" \
    --run \
    --train "$PROJECT_DIR/data/rag/embedding_grouped_train.jsonl" \
    --model "$model" \
    --out-dir "$model_out" \
    --epochs "$epochs" \
    --learning-rate "$lr" \
    --per-device-train-batch-size "$train_batch_size" \
    --gradient-accumulation-steps "$grad_acc" \
    --max-seq-length "$MAX_SEQ_LENGTH" \
    --device "$DEVICE" \
    $BF16_FLAG

  conda run -n llm_bge python "$PROJECT_DIR/scripts/rag/build_vector_dbs.py" \
    --models "$run_name" \
    --model-path "$run_name=$model_out/final" \
    --output-dir "$vector_db_dir" \
    --device "$DEVICE" \
    --batch-size "$ENCODE_BATCH_SIZE" \
    --force

  conda run -n llm_bge python "$PROJECT_DIR/scripts/rag/run_retrieval_baselines.py" \
    --mode all \
    --vector-db-dir "$vector_db_dir" \
    --models "$run_name" \
    --output-dir "$eval_out" \
    --compat-metrics-out "$run_dir/retrieval_metrics.json" \
    --device "$DEVICE" \
    --batch-size "$ENCODE_BATCH_SIZE" \
    --ks "${KS[@]}"

  python3 "$PROJECT_DIR/scripts/rag/analyze_candidate_depth_ablation.py" \
    --baseline-dir "$eval_out" \
    --candidate-ns "${CANDIDATE_NS[@]}" \
    --ks 6 10 30
}

for spec in $SPECS; do
  IFS="|" read -r run_name model epochs lr train_batch_size grad_acc <<<"$spec"
  train_eval_one "$run_name" "$model" "$epochs" "$lr" "$train_batch_size" "$grad_acc"
done

if [[ "$RERANK_BEST" == "1" ]]; then
  best_dir="$PROJECT_DIR/data/rag/_aux/embedding_finetune/$BEST_RUN/retrieval"
  python3 "$PROJECT_DIR/scripts/rag/analyze_candidate_depth_ablation.py" \
    --baseline-dir "$best_dir" \
    --candidate-ns "${CANDIDATE_NS[@]}" \
    --ks 6 10 30 \
    --write-preds \
    --pred-models "$BEST_RUN" \
    --pred-strategies rrf

  for spec in "10 20" "20 40" "30 60" "50 100"; do
    set -- $spec
    n="$1"
    candidate_k="$2"
    conda run -n llm_bge python "$PROJECT_DIR/scripts/rag/run_reranker_baseline.py" \
      --preds "$best_dir/preds_candidate_depth/hybrid_${BEST_RUN}_rrf_n${n}.jsonl" \
      --output-dir "$PROJECT_DIR/data/rag/_aux/embedding_finetune/$BEST_RUN/reranker_candidate_depth/${BEST_RUN}_rrf_n${n}" \
      --model "$RERANKER_MODEL" \
      --model-class "$RERANKER_MODEL_CLASS" \
      --candidate-k "$candidate_k" \
      --ks 6 10 30 50 \
      --query-batch-size 64 \
      --pair-batch-size 256
  done

  python3 "$PROJECT_DIR/scripts/rag/summarize_reranker_candidate_depth.py" \
    --candidate-ablation "$best_dir/candidate_depth_ablation.json" \
    --reranker-dir "$PROJECT_DIR/data/rag/_aux/embedding_finetune/$BEST_RUN/reranker_candidate_depth" \
    --model "$BEST_RUN" \
    --strategy rrf \
    --output-json "$best_dir/reranker_candidate_depth_ablation.json" \
    --output-md "$best_dir/reranker_candidate_depth_ablation.md"
fi
