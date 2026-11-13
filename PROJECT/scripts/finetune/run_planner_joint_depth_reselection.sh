#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PROJECT_DIR="$ROOT_DIR/PROJECT"
RUN_ROOT="$PROJECT_DIR/data/tool_call_trace/_aux/assignment3/planner_joint_depth_reselection_20260801"
PRED_ROOT="$PROJECT_DIR/finetune/outputs/preds/tool_trace/planner_joint_depth_reselection_20260801"
RAG_ROOT="$PROJECT_DIR/data/rag/_aux/retrieval_pipeline_reselection"
FROZEN="$RAG_ROOT/frozen_pipeline.json"
ROUTE_CONFIG="$PROJECT_DIR/configs/retrieval_canonical_route.json"
DOCS="$PROJECT_DIR/data/rag/tool_corpus.jsonl"
DB_ROOT="$RAG_ROOT/frozen_vector_db"
DB_KEY="frozen_selected_embedding"
MODEL="$PROJECT_DIR/models/Qwen/Qwen3-8B"
ADAPTER="$PROJECT_DIR/finetune/outputs/tooltrace-a3-crossstyle-rag10-sft8-e1-lr5e5-bs16/checkpoint-2429"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda/envs/llm_bge/bin/python}"
ANALYZER="$PROJECT_DIR/scripts/report/summarize_planner_joint_depth_reselection.py"
SANITIZER="$PROJECT_DIR/scripts/finetune/sanitize_tool_trace_eval_inputs.py"
EVALUATOR="$PROJECT_DIR/scripts/finetune/evaluate_tool_trace.py"
ROUTE_BUILDER="$PROJECT_DIR/scripts/rag/build_planner_joint_depth_routes.py"
FIRST_STAGE_APPLIER="$PROJECT_DIR/scripts/rag/apply_first_stage_route.py"
RUNNER="$PROJECT_DIR/scripts/finetune/run_assignment3_eval.sh"

mkdir -p "$RUN_ROOT/logs" "$PRED_ROOT/selection" "$PRED_ROOT/confirmation"

complete_jsonl() {
  local path="$1" expected="$2"
  [[ -f "$path" ]] && [[ "$(wc -l < "$path")" -eq "$expected" ]]
}

prepare_retrieval_split() {
  local split="$1" expected="$2" vector_gpu="$3"
  local dataset="$RUN_ROOT/${split}.jsonl"
  local root="$RUN_ROOT/${split}/retrieval"
  local bm25="$root/bm25/preds/bm25.jsonl"
  local vector="$root/vector/preds/vector_${DB_KEY}.jsonl"
  local source="$root/source/preds.jsonl"
  mkdir -p "$root/logs"

  local pids=()
  if ! complete_jsonl "$bm25" "$expected"; then
    conda run -n llm_bge python "$PROJECT_DIR/scripts/rag/run_retrieval_baselines.py" \
      --test-file "$dataset" --output-dir "$root/bm25" \
      --compat-metrics-out "$root/bm25/compat_metrics.json" \
      --mode bm25 --ks 6 10 20 30 40 50 100 --candidate-k 100 \
      --tokenizer fallback >"$root/logs/bm25.log" 2>&1 &
    pids+=("$!")
  fi
  if ! complete_jsonl "$vector" "$expected"; then
    (
      export CUDA_VISIBLE_DEVICES="$vector_gpu"
      conda run -n llm_bge python "$PROJECT_DIR/scripts/rag/run_retrieval_baselines.py" \
        --test-file "$dataset" --vector-db-dir "$DB_ROOT" --models "$DB_KEY" \
        --output-dir "$root/vector" \
        --compat-metrics-out "$root/vector/compat_metrics.json" \
        --mode vector --ks 6 10 20 30 40 50 100 --candidate-k 100 \
        --batch-size 64 --device cuda:0 --tokenizer fallback
    ) >"$root/logs/vector.log" 2>&1 &
    pids+=("$!")
  fi
  local failed=0
  for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
  [[ "$failed" -eq 0 ]] || return "$failed"
  for path in "$bm25" "$vector"; do
    complete_jsonl "$path" "$expected" || {
      echo "incomplete retrieval artifact: $path" >&2
      return 1
    }
  done
  "$PYTHON_BIN" "$FIRST_STAGE_APPLIER" \
    --vector-preds "$vector" --bm25-preds "$bm25" \
    --route-json "$ROUTE_CONFIG" --route-field first_stage --complete-to 50 \
    --out-dir "$root/source"
  complete_jsonl "$source" "$expected" || {
    echo "incomplete first-stage artifact: $source" >&2
    return 1
  }
}

run_reranker_depth() {
  local split="$1" depth="$2" gpu="$3" expected="$4"
  local root="$RUN_ROOT/${split}/retrieval"
  local out_dir="$root/reranker_k${depth}"
  local output="$out_dir/preds.jsonl"
  if complete_jsonl "$output" "$expected" \
    && [[ "$(jq -r '.candidate_k // 0' "$out_dir/metrics.json" 2>/dev/null || true)" == "$depth" ]]; then
    return 0
  fi
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    conda run -n llm_bge python "$PROJECT_DIR/scripts/rag/run_reranker_baseline.py" \
      --preds "$root/source/preds.jsonl" --docs "$DOCS" --output-dir "$out_dir" \
      --model "$(jq -r '.reranker.model' "$FROZEN")" \
      --backend sentence-transformers --candidate-k "$depth" \
      --ks 6 10 20 30 50 --query-batch-size 24 --pair-batch-size 64 \
      --max-length 512 --device cuda:0 --use-fp16
  ) >"$root/logs/reranker-k${depth}.log" 2>&1
  complete_jsonl "$output" "$expected"
}

run_planner_cell() {
  local split="$1" route_id="$2" gpu="$3"
  local planner_depth="${route_id##*planner}"
  local batch_size=4
  if [[ "$planner_depth" -eq 50 ]]; then batch_size=2; fi
  local input="$RUN_ROOT/$split/inputs/${route_id}.jsonl"
  local stem="joint-sft8-${split}-${route_id}"
  GPU_LIST="$gpu" BATCH_SIZE="$batch_size" PYTHON_BIN="$PYTHON_BIN" \
    bash "$RUNNER" "$MODEL" "$ADAPTER" "$input" rag \
      "$PRED_ROOT/$split/$stem.jsonl" "$stem"
}

evaluate_cell() {
  local split="$1" route_id="$2"
  local stem="joint-sft8-${split}-${route_id}"
  "$PYTHON_BIN" "$EVALUATOR" \
    --predictions "$PRED_ROOT/$split/$stem.jsonl" \
    --gold "$RUN_ROOT/${split}.jsonl" \
    --metrics-out "$RUN_ROOT/$split/eval/$stem.metrics.json" \
    --scored-out "$RUN_ROOT/$split/eval/$stem.scored.jsonl" \
    --run-id "$stem"
}

"$PYTHON_BIN" "$ANALYZER" --stage begin

prepare_retrieval_split selection 2400 0 \
  >"$RUN_ROOT/logs/retrieval-selection.log" 2>&1 & p_selection=$!
prepare_retrieval_split confirmation 1200 1 \
  >"$RUN_ROOT/logs/retrieval-confirmation.log" 2>&1 & p_confirmation=$!
failed=0
if ! wait "$p_selection"; then failed=1; fi
if ! wait "$p_confirmation"; then failed=1; fi
[[ "$failed" -eq 0 ]] || { echo "retrieval preparation failed" >&2; exit 1; }

selection_retrieval="$RUN_ROOT/selection/retrieval"
run_reranker_depth selection 10 0 2400 & r10=$!
run_reranker_depth selection 20 1 2400 & r20=$!
run_reranker_depth selection 30 2 2400 & r30=$!
run_reranker_depth selection 50 3 2400 & r50=$!
failed=0
for pid in "$r10" "$r20" "$r30" "$r50"; do
  if ! wait "$pid"; then failed=1; fi
done
[[ "$failed" -eq 0 ]] || { echo "exact-depth selection reranking failed" >&2; exit 1; }

"$PYTHON_BIN" "$ROUTE_BUILDER" \
  --dataset "$RUN_ROOT/selection.jsonl" \
  --vector-preds "$selection_retrieval/vector/preds/vector_${DB_KEY}.jsonl" \
  --bm25-preds "$selection_retrieval/bm25/preds/bm25.jsonl" \
  --reranker-preds-at-depth "10=$selection_retrieval/reranker_k10/preds.jsonl" \
  --reranker-preds-at-depth "20=$selection_retrieval/reranker_k20/preds.jsonl" \
  --reranker-preds-at-depth "30=$selection_retrieval/reranker_k30/preds.jsonl" \
  --reranker-preds-at-depth "50=$selection_retrieval/reranker_k50/preds.jsonl" \
  --route-json "$ROUTE_CONFIG" \
  --out-dir "$selection_retrieval"

route_ids=(
  rerank10_planner5 rerank10_planner10
  rerank20_planner5 rerank20_planner10 rerank20_planner20
  rerank30_planner5 rerank30_planner10 rerank30_planner20 rerank30_planner30
  rerank50_planner5 rerank50_planner10 rerank50_planner20 rerank50_planner30 rerank50_planner50
)
mkdir -p "$RUN_ROOT/selection/inputs" "$RUN_ROOT/selection/eval"
for route_id in "${route_ids[@]}"; do
  planner_depth="${route_id##*planner}"
  "$PYTHON_BIN" "$SANITIZER" --split test \
    --input "$RUN_ROOT/selection.jsonl" \
    --rankings "$selection_retrieval/rankings/${route_id}.jsonl" \
    --candidate-k "$planner_depth" \
    --query-out "$RUN_ROOT/selection/inputs/queries_label_free.jsonl" \
    --rag-out "$RUN_ROOT/selection/inputs/${route_id}.jsonl" \
    --audit-out "$RUN_ROOT/selection/inputs/${route_id}.audit.json"
done

(run_planner_cell selection rerank50_planner50 0 && run_planner_cell selection rerank10_planner5 0) >"$RUN_ROOT/logs/planner-selection-gpu0.log" 2>&1 & q0=$!
(run_planner_cell selection rerank30_planner30 1 && run_planner_cell selection rerank10_planner10 1 && run_planner_cell selection rerank20_planner5 1) >"$RUN_ROOT/logs/planner-selection-gpu1.log" 2>&1 & q1=$!
(run_planner_cell selection rerank50_planner30 2 && run_planner_cell selection rerank20_planner10 2 && run_planner_cell selection rerank30_planner5 2) >"$RUN_ROOT/logs/planner-selection-gpu2.log" 2>&1 & q2=$!
(run_planner_cell selection rerank20_planner20 3 && run_planner_cell selection rerank30_planner10 3 && run_planner_cell selection rerank50_planner5 3) >"$RUN_ROOT/logs/planner-selection-gpu3.log" 2>&1 & q3=$!
(run_planner_cell selection rerank30_planner20 4 && run_planner_cell selection rerank50_planner10 4) >"$RUN_ROOT/logs/planner-selection-gpu4.log" 2>&1 & q4=$!
(run_planner_cell selection rerank50_planner20 5) >"$RUN_ROOT/logs/planner-selection-gpu5.log" 2>&1 & q5=$!
failed=0
for pid in "$q0" "$q1" "$q2" "$q3" "$q4" "$q5"; do
  if ! wait "$pid"; then failed=1; fi
done
[[ "$failed" -eq 0 ]] || { echo "selection Planner inference failed" >&2; exit 1; }

for route_id in "${route_ids[@]}"; do evaluate_cell selection "$route_id"; done
"$PYTHON_BIN" "$ANALYZER" --stage select

confirmation_retrieval="$RUN_ROOT/confirmation/retrieval"
selected_id="$(jq -r '.selected_route_id' "$RUN_ROOT/selection.lock.json")"
selected_reranker_depth="${selected_id%%_*}"
selected_reranker_depth="${selected_reranker_depth#rerank}"
run_reranker_depth confirmation 10 0 1200 & cr10=$!
if [[ "$selected_reranker_depth" != 10 ]]; then
  run_reranker_depth confirmation "$selected_reranker_depth" 1 1200 & cr_selected=$!
else
  cr_selected=""
fi
wait "$cr10"
if [[ -n "$cr_selected" ]]; then wait "$cr_selected"; fi

confirmation_ids=(rerank10_planner10)
if [[ "$selected_id" != rerank10_planner10 ]]; then confirmation_ids+=("$selected_id"); fi
confirmation_reranker_args=(
  --reranker-preds-at-depth "10=$confirmation_retrieval/reranker_k10/preds.jsonl"
)
if [[ "$selected_reranker_depth" != 10 ]]; then
  confirmation_reranker_args+=(
    --reranker-preds-at-depth "$selected_reranker_depth=$confirmation_retrieval/reranker_k${selected_reranker_depth}/preds.jsonl"
  )
fi
confirmation_route_args=()
for route_id in "${confirmation_ids[@]}"; do
  confirmation_route_args+=(--route-id "$route_id")
done
"$PYTHON_BIN" "$ROUTE_BUILDER" \
  --dataset "$RUN_ROOT/confirmation.jsonl" \
  --vector-preds "$confirmation_retrieval/vector/preds/vector_${DB_KEY}.jsonl" \
  --bm25-preds "$confirmation_retrieval/bm25/preds/bm25.jsonl" \
  "${confirmation_reranker_args[@]}" \
  --route-json "$ROUTE_CONFIG" \
  --frozen-route-specs "$selection_retrieval/routes.json" \
  "${confirmation_route_args[@]}" \
  --out-dir "$confirmation_retrieval"

mkdir -p "$RUN_ROOT/confirmation/inputs" "$RUN_ROOT/confirmation/eval"
for route_id in "${confirmation_ids[@]}"; do
  planner_depth="${route_id##*planner}"
  "$PYTHON_BIN" "$SANITIZER" --split test \
    --input "$RUN_ROOT/confirmation.jsonl" \
    --rankings "$confirmation_retrieval/rankings/${route_id}.jsonl" \
    --candidate-k "$planner_depth" \
    --query-out "$RUN_ROOT/confirmation/inputs/queries_label_free.jsonl" \
    --rag-out "$RUN_ROOT/confirmation/inputs/${route_id}.jsonl" \
    --audit-out "$RUN_ROOT/confirmation/inputs/${route_id}.audit.json"
done

run_planner_cell confirmation rerank10_planner10 0 \
  >"$RUN_ROOT/logs/planner-confirmation-anchor.log" 2>&1 & c0=$!
if [[ "$selected_id" != rerank10_planner10 ]]; then
  run_planner_cell confirmation "$selected_id" 1 \
    >"$RUN_ROOT/logs/planner-confirmation-selected.log" 2>&1 & c1=$!
else
  c1=""
fi
wait "$c0"
if [[ -n "$c1" ]]; then wait "$c1"; fi
for route_id in "${confirmation_ids[@]}"; do evaluate_cell confirmation "$route_id"; done
"$PYTHON_BIN" "$ANALYZER" --stage confirm
echo "joint depth reselection complete: $RUN_ROOT/summary.json"
