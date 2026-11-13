#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 GPU_ID MODEL_KEY NEGATIVE_RECIPE LEARNING_RATE SEED" >&2
  exit 2
fi

GPU_ID="$1"
MODEL_KEY="$2"
NEGATIVE_RECIPE="$3"
LEARNING_RATE="$4"
SEED="$5"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PROJECT_DIR="$ROOT_DIR/PROJECT"
RUN_ROOT="$PROJECT_DIR/data/rag/_aux/retrieval_pipeline_reselection"
LR_TAG="${LEARNING_RATE//./p}"
LR_TAG="${LR_TAG//-/m}"
RUN_NAME="${MODEL_KEY}-${NEGATIVE_RECIPE}-lr${LR_TAG}-s${SEED}"
MODEL_DIR="$RUN_ROOT/reranker_train/$RUN_NAME"
OUT_DIR="$RUN_ROOT/reranker_dev/$RUN_NAME"
CANDIDATES="$RUN_ROOT/reranker_dev/all_tool_candidates.jsonl"

if [[ ! -f "$MODEL_DIR/model.safetensors" ]]; then
  echo "missing trained reranker: $MODEL_DIR/model.safetensors" >&2
  exit 1
fi
mkdir -p "$OUT_DIR"
exec 9>"$OUT_DIR/.eval.lock"
flock 9
if [[ -f "$OUT_DIR/metrics.json" && -f "$OUT_DIR/preds.jsonl" ]] \
  && [[ "$(wc -l < "$OUT_DIR/preds.jsonl")" -eq 1200 ]]; then
  echo "skip complete reranker evaluation: $RUN_NAME"
  exit 0
fi

case "$MODEL_KEY" in
  qwen3_0p6b)
    PAIR_BATCH=32
    ;;
  bge_base|bge_large|bge_v2_m3)
    PAIR_BATCH=64
    ;;
  *)
    echo "unknown model key: $MODEL_KEY" >&2
    exit 2
    ;;
esac

export CUDA_VISIBLE_DEVICES="$GPU_ID"
conda run -n llm_bge python "$PROJECT_DIR/scripts/rag/run_reranker_baseline.py" \
  --preds "$CANDIDATES" \
  --docs "$PROJECT_DIR/data/rag/tool_corpus.jsonl" \
  --output-dir "$OUT_DIR" \
  --model "$MODEL_DIR" \
  --backend sentence-transformers \
  --candidate-k 100 \
  --ks 6 10 20 30 40 50 \
  --query-batch-size 24 \
  --pair-batch-size "$PAIR_BATCH" \
  --max-length 512 \
  --device cuda:0 \
  --use-fp16

echo "complete reranker evaluation=$RUN_NAME gpu=$GPU_ID"
