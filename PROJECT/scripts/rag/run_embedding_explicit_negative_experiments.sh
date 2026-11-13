#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PROJECT_DIR="$ROOT_DIR/PROJECT"

SPECS="${SPECS:-\
bge-large-mnrl-control-e1-lr2e-5-bs8-g8|none \
bge-large-mnrl-explicit-hard2-e1-lr2e-5-bs8-g8|hard \
bge-large-mnrl-explicit-easy2-e1-lr2e-5-bs8-g8|easy \
bge-large-mnrl-explicit-all4-e1-lr2e-5-bs8-g8|all}"
MODEL="${MODEL:-BAAI/bge-large-zh-v1.5}"
EPOCHS="${EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
GRAD_ACC="${GRAD_ACC:-8}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-256}"
DEVICE="${DEVICE:-cuda}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-64}"
PRECISION_FLAG="${PRECISION_FLAG:---fp16}"
RETRAIN="${RETRAIN:-0}"
REEVALUATE="${REEVALUATE:-0}"
KS=(${KS:-6 7 8 9 10 15 20 30 50})
read -r -a precision_args <<<"$PRECISION_FLAG"

train_eval_one() {
  local run_name="$1"
  local negative_mode="$2"
  local run_dir="$PROJECT_DIR/data/rag/_aux/embedding_finetune/$run_name"
  local model_out="$run_dir/model"
  local vector_db_dir="$PROJECT_DIR/data/rag/vector_dbs_finetuned"
  local eval_out="$run_dir/retrieval"
  local overwrite_args=()

  if [[ "$RETRAIN" == "1" ]]; then
    overwrite_args+=(--overwrite-output-dir)
  fi

  if [[ "$RETRAIN" == "1" || ! -d "$model_out/final" ]]; then
    conda run -n llm_bge python "$PROJECT_DIR/scripts/rag/train_embedding.py" \
      --run \
      --train "$PROJECT_DIR/data/rag/embedding_grouped_train.jsonl" \
      --model "$MODEL" \
      --out-dir "$model_out" \
      --loss mnrl \
      --negative-mode "$negative_mode" \
      --epochs "$EPOCHS" \
      --learning-rate "$LEARNING_RATE" \
      --per-device-train-batch-size "$TRAIN_BATCH_SIZE" \
      --gradient-accumulation-steps "$GRAD_ACC" \
      --max-seq-length "$MAX_SEQ_LENGTH" \
      --save-steps 0 \
      --device "$DEVICE" \
      "${precision_args[@]}" \
      "${overwrite_args[@]}"
  fi

  if [[ "$REEVALUATE" == "1" || ! -f "$eval_out/metrics.json" ]]; then
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
  fi
}

for spec in $SPECS; do
  IFS="|" read -r run_name negative_mode <<<"$spec"
  train_eval_one "$run_name" "$negative_mode"
done

python3 "$PROJECT_DIR/scripts/report/make_embedding_negative_report.py"
