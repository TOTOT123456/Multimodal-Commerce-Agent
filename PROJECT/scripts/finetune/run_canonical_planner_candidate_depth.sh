#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PROJECT_DIR="$ROOT_DIR/PROJECT"
RUN_ROOT="$PROJECT_DIR/data/tool_call_trace/_aux/assignment3/canonical_planner_candidate_depth_isolated_20260801"
PRED_ROOT="$PROJECT_DIR/finetune/outputs/preds/tool_trace/canonical_planner_candidate_depth_isolated_20260801"
DEV_DATASET="$RUN_ROOT/dev.jsonl"
DEV_RANKING="$RUN_ROOT/retrieval/challenger/preds.jsonl"
LEGACY_DATASET="$PROJECT_DIR/data/tool_call_trace/test.jsonl"
API_DATASET="$PROJECT_DIR/data/tool_call_trace/_aux/assignment3/assignment3_100tool_api/test_api/test.jsonl"
RETRIEVAL_EVAL_ROOT="$PROJECT_DIR/data/rag/_aux/retrieval_pipeline_reselection/evaluation"
FROZEN_PIPELINE="$PROJECT_DIR/data/rag/_aux/retrieval_pipeline_reselection/frozen_pipeline.json"
MODEL_4B="$PROJECT_DIR/models/Qwen/Qwen3-4B"
MODEL_8B="$PROJECT_DIR/models/Qwen/Qwen3-8B"
ADAPTER_4B="$PROJECT_DIR/finetune/outputs/tooltrace-a3-crossstyle-rag10-sft4-e1-lr5e5-bs16/checkpoint-2429"
ADAPTER_8B="$PROJECT_DIR/finetune/outputs/tooltrace-a3-crossstyle-rag10-sft8-e1-lr5e5-bs16/checkpoint-2429"
INTENT_PREDS="${INTENT_PREDS:-$PROJECT_DIR/finetune/outputs/preds/qwen3-0.6b-full_lora-context_isolated-best.jsonl}"
PYTHON_BIN="/root/miniconda/envs/llm_bge/bin/python"
ANALYZER="$PROJECT_DIR/scripts/report/summarize_canonical_planner_candidate_depth.py"
SANITIZER="$PROJECT_DIR/scripts/finetune/sanitize_tool_trace_eval_inputs.py"
EVALUATOR="$PROJECT_DIR/scripts/finetune/evaluate_tool_trace.py"
APPLY_RETRIEVAL="$PROJECT_DIR/scripts/rag/apply_frozen_retrieval_pipeline.py"

mkdir -p "$RUN_ROOT/inputs" "$RUN_ROOT/dev" "$RUN_ROOT/protected" \
  "$RUN_ROOT/logs" "$PRED_ROOT/dev" "$PRED_ROOT/protected"

"$PYTHON_BIN" "$ANALYZER" --stage begin

for candidate_k in 5 10 20 30 50; do
  "$PYTHON_BIN" "$SANITIZER" \
    --split test \
    --input "$DEV_DATASET" \
    --rankings "$DEV_RANKING" \
    --candidate-k "$candidate_k" \
    --query-out "$RUN_ROOT/inputs/dev_queries_label_free.jsonl" \
    --rag-out "$RUN_ROOT/inputs/dev_rag_k${candidate_k}_label_free.jsonl" \
    --audit-out "$RUN_ROOT/inputs/dev_k${candidate_k}_audit.json"
done

run_dev() {
  local model_size="$1" candidate_k="$2" gpu="$3" batch_size="$4"
  local model adapter stem
  if [[ "$model_size" == 4b ]]; then
    model="$MODEL_4B"
    adapter="$ADAPTER_4B"
  else
    model="$MODEL_8B"
    adapter="$ADAPTER_8B"
  fi
  stem="canonical-isolated-sft-${model_size}-dev1200-k${candidate_k}"
  GPU_LIST="$gpu" BATCH_SIZE="$batch_size" PYTHON_BIN="$PYTHON_BIN" \
    bash "$PROJECT_DIR/scripts/finetune/run_assignment3_eval.sh" \
      "$model" "$adapter" \
      "$RUN_ROOT/inputs/dev_rag_k${candidate_k}_label_free.jsonl" \
      rag "$PRED_ROOT/dev/$stem.jsonl" "$stem"
}

# Five 8B cells start together; GPU 5 clears the short 4B cells while two
# completed 8B queues pick up the remaining 4B work.
(run_dev 8b 5 0 4 && run_dev 4b 30 0 16) >"$RUN_ROOT/logs/dev-gpu0.log" 2>&1 & p0=$!
(run_dev 8b 10 1 4 && run_dev 4b 50 1 8) >"$RUN_ROOT/logs/dev-gpu1.log" 2>&1 & p1=$!
(run_dev 8b 20 2 4) >"$RUN_ROOT/logs/dev-8b-k20.log" 2>&1 & p2=$!
(run_dev 8b 30 3 4) >"$RUN_ROOT/logs/dev-8b-k30.log" 2>&1 & p3=$!
(run_dev 8b 50 4 2) >"$RUN_ROOT/logs/dev-8b-k50.log" 2>&1 & p4=$!
(run_dev 4b 5 5 16 && run_dev 4b 10 5 16 && run_dev 4b 20 5 16) >"$RUN_ROOT/logs/dev-gpu5.log" 2>&1 & p5=$!

failed=0
for pid in "$p0" "$p1" "$p2" "$p3" "$p4" "$p5"; do
  if ! wait "$pid"; then failed=1; fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "one or more development candidate-depth workers failed" >&2
  exit 1
fi

for model_size in 4b 8b; do
  for candidate_k in 5 10 20 30 50; do
    stem="canonical-isolated-sft-${model_size}-dev1200-k${candidate_k}"
    "$PYTHON_BIN" "$EVALUATOR" \
      --predictions "$PRED_ROOT/dev/$stem.jsonl" \
      --gold "$DEV_DATASET" \
      --metrics-out "$RUN_ROOT/dev/$stem.metrics.json" \
      --scored-out "$RUN_ROOT/dev/$stem.scored.jsonl" \
      --run-id "$stem"
  done
done

"$PYTHON_BIN" "$ANALYZER" --stage select
selected_k="$(jq -r '.selected_k' "$RUN_ROOT/selection.lock.json")"

if [[ "$selected_k" != 10 ]]; then
  for split in legacy6000 api1000; do
    if [[ "$split" == legacy6000 ]]; then
      dataset="$LEGACY_DATASET"
      retrieval_dir="$RETRIEVAL_EVAL_ROOT/legacy_frozen_6k"
    else
      dataset="$API_DATASET"
      retrieval_dir="$RETRIEVAL_EVAL_ROOT/api_frozen_1k"
    fi
    completed_dir="$RUN_ROOT/protected_rankings/$split"
    "$PYTHON_BIN" "$APPLY_RETRIEVAL" \
      --vector-preds "$retrieval_dir/vector_selected/preds/vector_frozen_selected_embedding.jsonl" \
      --bm25-preds "$retrieval_dir/bm25/preds/bm25.jsonl" \
      --reranker-preds "$retrieval_dir/reranker_selected/preds.jsonl" \
      --route-json "$FROZEN_PIPELINE" --dataset "$dataset" \
      --out-dir "$completed_dir" --complete-ranking-to 50
    ranking="$completed_dir/preds.jsonl"
    "$PYTHON_BIN" "$SANITIZER" \
      --split test \
      --input "$dataset" \
      --rankings "$ranking" \
      --candidate-k "$selected_k" \
      --query-out "$RUN_ROOT/inputs/${split}_queries_label_free.jsonl" \
      --rag-out "$RUN_ROOT/inputs/${split}_rag_k${selected_k}_label_free.jsonl" \
      --audit-out "$RUN_ROOT/inputs/${split}_k${selected_k}_audit.json"
  done

  run_protected() {
    local model_size="$1" split="$2" gpu_list="$3" batch_size="$4"
    local model adapter stem input
    if [[ "$model_size" == 4b ]]; then
      model="$MODEL_4B"
      adapter="$ADAPTER_4B"
    else
      model="$MODEL_8B"
      adapter="$ADAPTER_8B"
    fi
    stem="canonical-sft-${model_size}-${split}-k${selected_k}"
    input="$RUN_ROOT/inputs/${split}_rag_k${selected_k}_label_free.jsonl"
    GPU_LIST="$gpu_list" BATCH_SIZE="$batch_size" PYTHON_BIN="$PYTHON_BIN" \
      bash "$PROJECT_DIR/scripts/finetune/run_assignment3_eval.sh" \
        "$model" "$adapter" "$input" rag \
        "$PRED_ROOT/protected/$stem.jsonl" "$stem"
  }

  protected_8b_batch=4
  protected_4b_batch=16
  if [[ "$selected_k" -eq 50 ]]; then
    protected_8b_batch=2
    protected_4b_batch=8
  fi
  run_protected 8b legacy6000 0,1 "$protected_8b_batch" >"$RUN_ROOT/logs/protected-8b-legacy.log" 2>&1 & q0=$!
  run_protected 4b legacy6000 2 "$protected_4b_batch" >"$RUN_ROOT/logs/protected-4b-legacy.log" 2>&1 & q1=$!
  run_protected 8b api1000 3 "$protected_8b_batch" >"$RUN_ROOT/logs/protected-8b-api.log" 2>&1 & q2=$!
  run_protected 4b api1000 4 "$protected_4b_batch" >"$RUN_ROOT/logs/protected-4b-api.log" 2>&1 & q3=$!
  protected_failed=0
  for pid in "$q0" "$q1" "$q2" "$q3"; do
    if ! wait "$pid"; then protected_failed=1; fi
  done
  if [[ "$protected_failed" -ne 0 ]]; then
    echo "one or more protected candidate-depth workers failed" >&2
    exit 1
  fi

  run_protected_base() {
    local model_size="$1" split="$2" gpu_list="$3" batch_size="$4"
    local model stem input
    if [[ "$model_size" == 4b ]]; then model="$MODEL_4B"; else model="$MODEL_8B"; fi
    stem="canonical-base-${model_size}-${split}-rag-k${selected_k}"
    input="$RUN_ROOT/inputs/${split}_rag_k${selected_k}_label_free.jsonl"
    GPU_LIST="$gpu_list" BATCH_SIZE="$batch_size" PYTHON_BIN="$PYTHON_BIN" \
      bash "$PROJECT_DIR/scripts/finetune/run_assignment3_eval.sh" \
        "$model" none "$input" rag \
        "$PRED_ROOT/protected/$stem.jsonl" "$stem"
  }

  # Keep the previously completed Base/SFT x Wo-RAG/RAG matrix current if
  # development selects a Planner depth other than the existing K=10.
  run_protected_base 8b legacy6000 0,1 "$protected_8b_batch" >"$RUN_ROOT/logs/protected-base-8b-legacy.log" 2>&1 & b0=$!
  run_protected_base 4b legacy6000 2 "$protected_4b_batch" >"$RUN_ROOT/logs/protected-base-4b-legacy.log" 2>&1 & b1=$!
  run_protected_base 8b api1000 3 "$protected_8b_batch" >"$RUN_ROOT/logs/protected-base-8b-api.log" 2>&1 & b2=$!
  run_protected_base 4b api1000 4 "$protected_4b_batch" >"$RUN_ROOT/logs/protected-base-4b-api.log" 2>&1 & b3=$!
  base_failed=0
  for pid in "$b0" "$b1" "$b2" "$b3"; do
    if ! wait "$pid"; then base_failed=1; fi
  done
  if [[ "$base_failed" -ne 0 ]]; then
    echo "one or more protected Base candidate-depth workers failed" >&2
    exit 1
  fi

  for model_size in 4b 8b; do
    for split in legacy6000 api1000; do
      if [[ "$split" == legacy6000 ]]; then dataset="$LEGACY_DATASET"; else dataset="$API_DATASET"; fi
      for stage in sft base; do
        if [[ "$stage" == sft ]]; then
          stem="canonical-sft-${model_size}-${split}-k${selected_k}"
        else
          stem="canonical-base-${model_size}-${split}-rag-k${selected_k}"
        fi
        "$PYTHON_BIN" "$EVALUATOR" \
          --predictions "$PRED_ROOT/protected/$stem.jsonl" \
          --gold "$dataset" \
          --metrics-out "$RUN_ROOT/protected/$stem.metrics.json" \
          --scored-out "$RUN_ROOT/protected/$stem.scored.jsonl" \
          --run-id "$stem"
      done
    done
    stem="canonical-sft-${model_size}-legacy6000-k${selected_k}"
    "$PYTHON_BIN" "$PROJECT_DIR/scripts/finetune/evaluate_tool_trace_system.py" \
      --intent-predictions "$INTENT_PREDS" \
      --tool-predictions "$PRED_ROOT/protected/$stem.jsonl" \
      --out "$RUN_ROOT/protected/$stem.system.jsonl" \
      --metrics-out "$RUN_ROOT/protected/$stem.system.metrics.json"
  done
fi

"$PYTHON_BIN" "$ANALYZER" --stage complete
echo "canonical Planner candidate-depth experiment complete: $RUN_ROOT/summary.json"
