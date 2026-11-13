#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo "usage: $0 GPU_ID MODEL_KEY NEGATIVE_RECIPE LEARNING_RATE SEED [MINED_PREDS]" >&2
  exit 2
fi

GPU_ID="$1"
MODEL_KEY="$2"
NEGATIVE_RECIPE="$3"
LEARNING_RATE="$4"
SEED="$5"
MINED_PREDS="${6:-}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PROJECT_DIR="$ROOT_DIR/PROJECT"
RUN_ROOT="$PROJECT_DIR/data/rag/_aux/retrieval_pipeline_reselection/reranker_train"

case "$MODEL_KEY" in
  bge_base)
    MODEL="BAAI/bge-reranker-base"
    MODEL_KIND="encoder"
    GROUP_BATCH=8
    GRAD_ACC=2
    MEMORY_ARGS=()
    ;;
  bge_large)
    MODEL="BAAI/bge-reranker-large"
    MODEL_KIND="encoder"
    GROUP_BATCH=2
    GRAD_ACC=8
    MEMORY_ARGS=(--gradient-checkpointing)
    ;;
  bge_v2_m3)
    MODEL="BAAI/bge-reranker-v2-m3"
    MODEL_KIND="encoder"
    GROUP_BATCH=2
    GRAD_ACC=8
    MEMORY_ARGS=(--gradient-checkpointing)
    ;;
  qwen3_0p6b)
    MODEL="Qwen/Qwen3-Reranker-0.6B"
    MODEL_KIND="qwen_causal"
    GROUP_BATCH=2
    GRAD_ACC=8
    MEMORY_ARGS=(--gradient-checkpointing)
    ;;
  *)
    echo "unknown model key: $MODEL_KEY" >&2
    exit 2
    ;;
esac

if [[ "$NEGATIVE_RECIPE" != "typed4_random3" && -z "$MINED_PREDS" ]]; then
  echo "$NEGATIVE_RECIPE requires MINED_PREDS" >&2
  exit 2
fi

LR_TAG="${LEARNING_RATE//./p}"
LR_TAG="${LR_TAG//-/m}"
RUN_NAME="${MODEL_KEY}-${NEGATIVE_RECIPE}-lr${LR_TAG}-s${SEED}"
OUT_DIR="$RUN_ROOT/$RUN_NAME"
MINED_ARGS=()
if [[ -n "$MINED_PREDS" ]]; then
  MINED_ARGS=(--mined-preds "$MINED_PREDS")
fi

mkdir -p "$OUT_DIR"
exec 9>"$OUT_DIR/.train.lock"
flock 9
if [[ -f "$OUT_DIR/training_manifest.json" ]] \
  && [[ "$(jq -r '.status' "$OUT_DIR/training_manifest.json")" == "complete" ]] \
  && [[ -f "$OUT_DIR/model.safetensors" ]]; then
  echo "skip complete reranker: $RUN_NAME"
  exit 0
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"
TRAIN_ARGS=(
  --run
  --train "$PROJECT_DIR/data/rag/embedding_grouped_train.jsonl"
  --docs "$PROJECT_DIR/data/rag/tool_corpus.jsonl"
  --model "$MODEL"
  --model-kind "$MODEL_KIND"
  --negative-recipe "$NEGATIVE_RECIPE"
  --out-dir "$OUT_DIR"
  --group-size 8
  --per-device-group-batch-size "$GROUP_BATCH"
  --gradient-accumulation-steps "$GRAD_ACC"
  --epochs 1
  --learning-rate "$LEARNING_RATE"
  --warmup-ratio 0.1
  --weight-decay 0.01
  --max-length 512
  --seed "$SEED"
  --device cuda:0
  --fp16
)
TRAIN_ARGS+=("${MEMORY_ARGS[@]}")
TRAIN_ARGS+=("${MINED_ARGS[@]}")
conda run -n llm_bge python "$PROJECT_DIR/scripts/rag/train_reranker_groupwise.py" "${TRAIN_ARGS[@]}"

echo "complete model=$MODEL_KEY negative=$NEGATIVE_RECIPE lr=$LEARNING_RATE seed=$SEED gpu=$GPU_ID"
