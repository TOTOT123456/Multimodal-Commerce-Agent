#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PROJECT_DIR="$ROOT_DIR/PROJECT"
RUN_ROOT="$PROJECT_DIR/data/rag/_aux/retrieval_pipeline_reselection"
FROZEN="$RUN_ROOT/frozen_pipeline.json"
FINAL_DATASET="$PROJECT_DIR/data/tool_call_trace/_aux/assignment3/assignment3_crossstyle_rag10/dpo_v2/eval_protocol/final.jsonl"
LEGACY_DATASET="$PROJECT_DIR/data/tool_call_trace/test.jsonl"
API_DATASET="$PROJECT_DIR/data/tool_call_trace/_aux/assignment3/assignment3_100tool_api/test_api/test.jsonl"

while [[ ! -f "$FROZEN" ]]; do sleep 10; done

bash "$PROJECT_DIR/scripts/rag/run_frozen_retrieval_dataset.sh" \
  round_final "$FINAL_DATASET" 1200 0 1 2 3

SELECTED_DB="$RUN_ROOT/frozen_vector_db/frozen_selected_embedding"
SELECTED_RERANKER="$(jq -r '.reranker.model' "$FROZEN")"
SELECTED_GROUP="$(jq -r '.reranker.group' "$FROZEN")"
case "$SELECTED_GROUP" in
  qwen3_0p6b*) SELECTED_PAIR_BATCH=32 ;;
  bge_large*|bge_v2_m3*) SELECTED_PAIR_BATCH=64 ;;
  *) SELECTED_PAIR_BATCH=128 ;;
esac
BENCH_DIR="$RUN_ROOT/evaluation/round_final/benchmark"
mkdir -p "$BENCH_DIR"

(
  export CUDA_VISIBLE_DEVICES=0
  conda run -n llm_bge python "$PROJECT_DIR/scripts/rag/benchmark_retrieval_pipeline.py" \
    --input "$FINAL_DATASET" \
    --reference "$RUN_ROOT/evaluation/round_final/challenger/preds.jsonl" \
    --fine-db "$SELECTED_DB" \
    --reranker "$SELECTED_RERANKER" \
    --route-json "$FROZEN" \
    --device cuda:0 \
    --query-batch-size 64 \
    --pair-batch-size "$SELECTED_PAIR_BATCH" \
    --tokenizer fallback \
    --out "$BENCH_DIR/challenger.json"
) >"$BENCH_DIR/challenger.log" 2>&1 &
challenger_benchmark_pid="$!"

(
  export CUDA_VISIBLE_DEVICES=1
  conda run -n llm_bge python "$PROJECT_DIR/scripts/rag/benchmark_retrieval_pipeline.py" \
    --input "$FINAL_DATASET" \
    --reference "$RUN_ROOT/evaluation/round_final/baseline/preds.jsonl" \
    --fine-db "$PROJECT_DIR/data/rag/vector_dbs_finetuned/bge-large-mnrl-e1-lr2e-5-bs64" \
    --reranker "$PROJECT_DIR/data/rag/_aux/reranker_finetune/bge-reranker-base-e1-g8-bs16" \
    --route-json "$RUN_ROOT/bs64_control_route.snapshot.json" \
    --device cuda:0 \
    --query-batch-size 64 \
    --pair-batch-size 128 \
    --tokenizer fallback \
    --out "$BENCH_DIR/baseline.json"
) >"$BENCH_DIR/baseline.log" 2>&1 &
baseline_benchmark_pid="$!"

benchmark_failed=0
if ! wait "$challenger_benchmark_pid"; then benchmark_failed=1; fi
if ! wait "$baseline_benchmark_pid"; then benchmark_failed=1; fi
if [[ "$benchmark_failed" -ne 0 ]]; then
  echo "end-to-end benchmark failed; inspect $BENCH_DIR/*.log" >&2
  exit 1
fi

bash "$PROJECT_DIR/scripts/rag/run_frozen_retrieval_dataset.sh" \
  legacy_frozen_6k "$LEGACY_DATASET" 6000 0 1 4 5 &
legacy_pid="$!"
bash "$PROJECT_DIR/scripts/rag/run_frozen_retrieval_dataset.sh" \
  api_frozen_1k "$API_DATASET" 1000 2 3 4 5 &
api_pid="$!"

regression_failed=0
if ! wait "$legacy_pid"; then regression_failed=1; fi
if ! wait "$api_pid"; then regression_failed=1; fi
if [[ "$regression_failed" -ne 0 ]]; then
  echo "one or more frozen regression evaluations failed" >&2
  exit 1
fi

echo "complete round-final gate, efficiency benchmark, and frozen regressions"
