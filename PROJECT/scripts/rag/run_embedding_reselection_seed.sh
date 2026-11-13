#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 GPU_ID RECIPE SEED" >&2
  exit 2
fi

GPU_ID="$1"
RECIPE="$2"
SEED="$3"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PROJECT_DIR="$ROOT_DIR/PROJECT"
DEV_FILE="$PROJECT_DIR/data/tool_call_trace/_aux/assignment3/assignment3_crossstyle_rag10/dpo_v2/eval_protocol/dev.jsonl"
RUN_ROOT="$PROJECT_DIR/data/rag/_aux/retrieval_pipeline_reselection"
TRAIN_FILE="$PROJECT_DIR/data/rag/embedding_grouped_train.jsonl"
VECTOR_ROOT="$RUN_ROOT/vector_dbs"

case "$RECIPE" in
  bs64_none)
    NEGATIVE_MODE="none"
    TRAIN_BATCH=64
    GRAD_ACC=1
    MEMORY_ARGS=(--gradient-checkpointing --precompute-no-duplicate-hashes)
    ;;
  bs8_control)
    NEGATIVE_MODE="none"
    TRAIN_BATCH=8
    GRAD_ACC=8
    MEMORY_ARGS=()
    ;;
  easy2)
    NEGATIVE_MODE="easy"
    TRAIN_BATCH=8
    GRAD_ACC=8
    MEMORY_ARGS=()
    ;;
  hard2)
    NEGATIVE_MODE="hard"
    TRAIN_BATCH=8
    GRAD_ACC=8
    MEMORY_ARGS=()
    ;;
  all4)
    NEGATIVE_MODE="all"
    TRAIN_BATCH=8
    GRAD_ACC=8
    MEMORY_ARGS=()
    ;;
  *)
    echo "unknown recipe: $RECIPE" >&2
    exit 2
    ;;
esac

RUN_NAME="bge-large-reselection-${RECIPE}-s${SEED}"
RUN_DIR="$RUN_ROOT/embedding_train/$RUN_NAME"
MODEL_DIR="$RUN_DIR/model"
DB_DIR="$VECTOR_ROOT/$RUN_NAME"
EVAL_DIR="$RUN_ROOT/embedding_dev/seed_${SEED}/$RECIPE"

export CUDA_VISIBLE_DEVICES="$GPU_ID"

if [[ ! -f "$MODEL_DIR/final/model.safetensors" ]]; then
  TRAIN_ARGS=(
    --run
    --train "$TRAIN_FILE"
    --model BAAI/bge-large-zh-v1.5
    --out-dir "$MODEL_DIR"
    --loss mnrl
    --negative-mode "$NEGATIVE_MODE"
    --epochs 1
    --learning-rate 2e-5
    --per-device-train-batch-size "$TRAIN_BATCH"
    --gradient-accumulation-steps "$GRAD_ACC"
    --max-seq-length 256
    --save-steps 0
    --seed "$SEED"
    --device cuda:0
    --fp16
  )
  TRAIN_ARGS+=("${MEMORY_ARGS[@]}")
  conda run -n llm_bge python "$PROJECT_DIR/scripts/rag/train_embedding.py" "${TRAIN_ARGS[@]}"
fi

if [[ ! -f "$DB_DIR/index.faiss" ]]; then
  conda run -n llm_bge python "$PROJECT_DIR/scripts/rag/build_vector_dbs.py" \
    --models "$RUN_NAME" \
    --model-path "$RUN_NAME=$MODEL_DIR/final" \
    --output-dir "$VECTOR_ROOT" \
    --device cuda:0 \
    --batch-size 64
fi

if [[ ! -f "$EVAL_DIR/metrics.json" ]]; then
  conda run -n llm_bge python "$PROJECT_DIR/scripts/rag/run_retrieval_baselines.py" \
    --test-file "$DEV_FILE" \
    --vector-db-dir "$VECTOR_ROOT" \
    --models "$RUN_NAME" \
    --output-dir "$EVAL_DIR" \
    --compat-metrics-out "$EVAL_DIR/compat_metrics.json" \
    --mode all \
    --ks 6 10 20 30 50 \
    --candidate-k 50 \
    --batch-size 64 \
    --device cuda:0 \
    --tokenizer fallback
fi

echo "complete recipe=$RECIPE seed=$SEED gpu=$GPU_ID"
