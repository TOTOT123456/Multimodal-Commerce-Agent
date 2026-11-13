#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PROJECT_DIR="$ROOT_DIR/PROJECT"
RUN_ROOT="$PROJECT_DIR/data/tool_call_trace/_aux/assignment3/planner_joint_depth_reselection_20260801"
PRED_ROOT="$PROJECT_DIR/finetune/outputs/preds/tool_trace/planner_joint_depth_reselection_20260801/protected"
RAG_ROOT="$PROJECT_DIR/data/rag/_aux/retrieval_pipeline_reselection"
FROZEN="$RAG_ROOT/frozen_pipeline.json"
ROUTE_CONFIG="$PROJECT_DIR/configs/retrieval_canonical_route.json"
DOCS="$PROJECT_DIR/data/rag/tool_corpus.jsonl"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda/envs/llm_bge/bin/python}"
ROUTE_BUILDER="$PROJECT_DIR/scripts/rag/build_planner_joint_depth_routes.py"
FIRST_STAGE_APPLIER="$PROJECT_DIR/scripts/rag/apply_first_stage_route.py"
SANITIZER="$PROJECT_DIR/scripts/finetune/sanitize_tool_trace_eval_inputs.py"
RUNNER="$PROJECT_DIR/scripts/finetune/run_assignment3_eval.sh"
EVALUATOR="$PROJECT_DIR/scripts/finetune/evaluate_tool_trace.py"
SYSTEM_EVALUATOR="$PROJECT_DIR/scripts/finetune/evaluate_tool_trace_system.py"
BASE_4B="$PROJECT_DIR/models/Qwen/Qwen3-4B"
BASE_8B="$PROJECT_DIR/models/Qwen/Qwen3-8B"
SFT_4B="$PROJECT_DIR/finetune/outputs/tooltrace-a3-crossstyle-rag10-sft4-e1-lr5e5-bs16/checkpoint-2429"
SFT_8B="$PROJECT_DIR/finetune/outputs/tooltrace-a3-crossstyle-rag10-sft8-e1-lr5e5-bs16/checkpoint-2429"
INTENT_PREDS="${INTENT_PREDS:-$PROJECT_DIR/finetune/outputs/preds/qwen3-0.6b-full_lora-context_isolated-best.jsonl}"

[[ -f "$RUN_ROOT/summary.json" ]] || { echo "joint-depth selection is incomplete" >&2; exit 2; }
route_id="$(jq -r '.final_route_id' "$RUN_ROOT/summary.json")"
reranker_depth="${route_id%%_*}"
reranker_depth="${reranker_depth#rerank}"
planner_depth="${route_id##*planner}"
[[ "$planner_depth" -le "$reranker_depth" ]] || { echo "incoherent final depths" >&2; exit 2; }

mkdir -p "$RUN_ROOT/protected/logs" "$PRED_ROOT"

dataset_path() {
  case "$1" in
    legacy6000) printf '%s\n' "$PROJECT_DIR/data/tool_call_trace/test.jsonl" ;;
    api1000) printf '%s\n' "$PROJECT_DIR/data/tool_call_trace/_aux/assignment3/assignment3_100tool_api/test_api/test.jsonl" ;;
    *) echo "unsupported split: $1" >&2; return 2 ;;
  esac
}

retrieval_eval_root() {
  case "$1" in
    legacy6000) printf '%s\n' "$RAG_ROOT/evaluation/legacy_frozen_6k" ;;
    api1000) printf '%s\n' "$RAG_ROOT/evaluation/api_frozen_1k" ;;
    *) echo "unsupported split: $1" >&2; return 2 ;;
  esac
}

expected_rows() {
  case "$1" in legacy6000) printf '6000\n' ;; api1000) printf '1000\n' ;; esac
}

complete_jsonl() {
  local path="$1" expected="$2"
  [[ -f "$path" ]] && [[ "$(wc -l < "$path")" -eq "$expected" ]]
}

prepare_protected_retrieval() {
  local split="$1" gpu="$2"
  local expected source_root vector bm25 reranker_out
  expected="$(expected_rows "$split")"
  source_root="$(retrieval_eval_root "$split")"
  vector="$source_root/vector_selected/preds/vector_frozen_selected_embedding.jsonl"
  bm25="$source_root/bm25/preds/bm25.jsonl"
  reranker_out="$RUN_ROOT/protected/$split/retrieval/reranker_k${reranker_depth}"
  mkdir -p "$RUN_ROOT/protected/$split/retrieval/logs"
  "$PYTHON_BIN" "$FIRST_STAGE_APPLIER" \
    --vector-preds "$vector" --bm25-preds "$bm25" \
    --route-json "$ROUTE_CONFIG" --route-field first_stage \
    --complete-to "$reranker_depth" \
    --out-dir "$RUN_ROOT/protected/$split/retrieval/source"
  if ! complete_jsonl "$reranker_out/preds.jsonl" "$expected" \
    || [[ "$(jq -r '.candidate_k // 0' "$reranker_out/metrics.json" 2>/dev/null || true)" != "$reranker_depth" ]]; then
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      conda run -n llm_bge python "$PROJECT_DIR/scripts/rag/run_reranker_baseline.py" \
        --preds "$RUN_ROOT/protected/$split/retrieval/source/preds.jsonl" \
        --docs "$DOCS" --output-dir "$reranker_out" \
        --model "$(jq -r '.reranker.model' "$FROZEN")" \
        --backend sentence-transformers --candidate-k "$reranker_depth" \
        --ks 5 6 10 20 30 50 --query-batch-size 24 --pair-batch-size 64 \
        --max-length 512 --device cuda:0 --use-fp16
    ) >"$RUN_ROOT/protected/$split/retrieval/logs/reranker-k${reranker_depth}.log" 2>&1
  fi
  complete_jsonl "$reranker_out/preds.jsonl" "$expected"
  "$PYTHON_BIN" "$ROUTE_BUILDER" \
    --dataset "$(dataset_path "$split")" \
    --vector-preds "$vector" --bm25-preds "$bm25" \
    --reranker-preds-at-depth "$reranker_depth=$reranker_out/preds.jsonl" \
    --route-json "$ROUTE_CONFIG" \
    --frozen-route-specs "$RUN_ROOT/selection/retrieval/routes.json" \
    --route-id "$route_id" \
    --out-dir "$RUN_ROOT/protected/$split/retrieval"
  mkdir -p "$RUN_ROOT/protected/$split/inputs" "$RUN_ROOT/protected/$split/eval"
  "$PYTHON_BIN" "$SANITIZER" --split test \
    --input "$(dataset_path "$split")" \
    --rankings "$RUN_ROOT/protected/$split/retrieval/rankings/${route_id}.jsonl" \
    --candidate-k "$planner_depth" \
    --query-out "$RUN_ROOT/protected/$split/inputs/queries_label_free.jsonl" \
    --rag-out "$RUN_ROOT/protected/$split/inputs/rag_${route_id}.jsonl" \
    --audit-out "$RUN_ROOT/protected/$split/inputs/rag_${route_id}.audit.json"
}

prepare_protected_retrieval legacy6000 0 >"$RUN_ROOT/protected/logs/retrieval-legacy6000.log" 2>&1 & p0=$!
prepare_protected_retrieval api1000 1 >"$RUN_ROOT/protected/logs/retrieval-api1000.log" 2>&1 & p1=$!
failed=0
if ! wait "$p0"; then failed=1; fi
if ! wait "$p1"; then failed=1; fi
[[ "$failed" -eq 0 ]] || { echo "protected retrieval failed" >&2; exit 1; }

model_path() {
  case "$1" in 4b) printf '%s\n' "$BASE_4B" ;; 8b) printf '%s\n' "$BASE_8B" ;; esac
}

adapter_path() {
  case "$1/$2" in
    base/*) printf 'none\n' ;;
    sft/4b) printf '%s\n' "$SFT_4B" ;;
    sft/8b) printf '%s\n' "$SFT_8B" ;;
    *) echo "unsupported stage/model: $1/$2" >&2; return 2 ;;
  esac
}

run_cell() {
  local stage="$1" model="$2" split="$3" gpus="$4"
  local stem="joint-final-${stage}-${model}-${split}-${route_id}"
  local batch=4
  if [[ "$model" == 4b ]]; then batch=8; fi
  if [[ "$planner_depth" -ge 50 && "$model" == 8b ]]; then batch=2; fi
  PYTHON_BIN="$PYTHON_BIN" GPU_LIST="$gpus" BATCH_SIZE="$batch" \
    bash "$RUNNER" "$(model_path "$model")" "$(adapter_path "$stage" "$model")" \
      "$RUN_ROOT/protected/$split/inputs/rag_${route_id}.jsonl" rag \
      "$PRED_ROOT/$stem.jsonl" "$stem"
}

(run_cell base 4b legacy6000 0 && run_cell base 4b api1000 0) >"$RUN_ROOT/protected/logs/planner-base4.log" 2>&1 & q0=$!
(run_cell sft 4b legacy6000 1 && run_cell sft 4b api1000 1) >"$RUN_ROOT/protected/logs/planner-sft4.log" 2>&1 & q1=$!
(run_cell base 8b legacy6000 2,3 && run_cell base 8b api1000 2,3) >"$RUN_ROOT/protected/logs/planner-base8.log" 2>&1 & q2=$!
(run_cell sft 8b legacy6000 4,5 && run_cell sft 8b api1000 4,5) >"$RUN_ROOT/protected/logs/planner-sft8.log" 2>&1 & q3=$!
failed=0
for pid in "$q0" "$q1" "$q2" "$q3"; do if ! wait "$pid"; then failed=1; fi; done
[[ "$failed" -eq 0 ]] || { echo "protected Planner inference failed" >&2; exit 1; }

for split in legacy6000 api1000; do
  for stage in base sft; do
    for model in 4b 8b; do
      stem="joint-final-${stage}-${model}-${split}-${route_id}"
      "$PYTHON_BIN" "$EVALUATOR" \
        --predictions "$PRED_ROOT/$stem.jsonl" --gold "$(dataset_path "$split")" \
        --metrics-out "$RUN_ROOT/protected/$split/eval/$stem.metrics.json" \
        --scored-out "$RUN_ROOT/protected/$split/eval/$stem.scored.jsonl" \
        --run-id "$stem"
    done
  done
done

for model in 4b 8b; do
  stem="joint-final-sft-${model}-legacy6000-${route_id}"
  "$PYTHON_BIN" "$SYSTEM_EVALUATOR" \
    --intent-predictions "$INTENT_PREDS" \
    --tool-predictions "$PRED_ROOT/$stem.jsonl" \
    --out "$RUN_ROOT/protected/legacy6000/eval/$stem.system.jsonl" \
    --metrics-out "$RUN_ROOT/protected/legacy6000/eval/$stem.system.metrics.json"
done

# Re-report the selected route on the earlier strict SFT-combination-isolated
# development set.  This is diagnostic only: its K=10/K=50 results were known
# before preregistration and cannot alter the locked route.
ISO_SOURCE="$PROJECT_DIR/data/tool_call_trace/_aux/assignment3/canonical_planner_candidate_depth_isolated_20260801"
ISO_ROOT="$RUN_ROOT/strict_isolated_diagnostic"
ISO_PRED_ROOT="$PROJECT_DIR/finetune/outputs/preds/tool_trace/planner_joint_depth_reselection_20260801/strict_isolated_diagnostic"
mkdir -p "$ISO_ROOT/retrieval/logs" "$ISO_ROOT/inputs" "$ISO_ROOT/eval" "$ISO_PRED_ROOT"
"$PYTHON_BIN" "$FIRST_STAGE_APPLIER" \
  --vector-preds "$ISO_SOURCE/retrieval/vector/preds/vector_frozen_selected_embedding.jsonl" \
  --bm25-preds "$ISO_SOURCE/retrieval/bm25/preds/bm25.jsonl" \
  --route-json "$ROUTE_CONFIG" --route-field first_stage \
  --complete-to "$reranker_depth" --out-dir "$ISO_ROOT/retrieval/source"

iso_depths=(10)
if [[ "$reranker_depth" != 10 ]]; then iso_depths+=("$reranker_depth"); fi
iso_reranker_args=()
iso_pids=()
iso_gpu=0
for depth in "${iso_depths[@]}"; do
  out_dir="$ISO_ROOT/retrieval/reranker_k${depth}"
  if ! complete_jsonl "$out_dir/preds.jsonl" 1200 \
    || [[ "$(jq -r '.candidate_k // 0' "$out_dir/metrics.json" 2>/dev/null || true)" != "$depth" ]]; then
    (
      export CUDA_VISIBLE_DEVICES="$iso_gpu"
      conda run -n llm_bge python "$PROJECT_DIR/scripts/rag/run_reranker_baseline.py" \
        --preds "$ISO_ROOT/retrieval/source/preds.jsonl" --docs "$DOCS" \
        --output-dir "$out_dir" --model "$(jq -r '.reranker.model' "$FROZEN")" \
        --backend sentence-transformers --candidate-k "$depth" \
        --ks 5 6 10 20 30 50 --query-batch-size 24 --pair-batch-size 64 \
        --max-length 512 --device cuda:0 --use-fp16
    ) >"$ISO_ROOT/retrieval/logs/reranker-k${depth}.log" 2>&1 &
    iso_pids+=("$!")
  fi
  iso_reranker_args+=(--reranker-preds-at-depth "$depth=$out_dir/preds.jsonl")
  iso_gpu=$((iso_gpu + 1))
done
for pid in "${iso_pids[@]}"; do wait "$pid"; done

iso_route_ids=(rerank10_planner10)
if [[ "$route_id" != rerank10_planner10 ]]; then iso_route_ids+=("$route_id"); fi
iso_route_args=()
for id in "${iso_route_ids[@]}"; do iso_route_args+=(--route-id "$id"); done
"$PYTHON_BIN" "$ROUTE_BUILDER" \
  --dataset "$ISO_SOURCE/dev.jsonl" \
  --vector-preds "$ISO_SOURCE/retrieval/vector/preds/vector_frozen_selected_embedding.jsonl" \
  --bm25-preds "$ISO_SOURCE/retrieval/bm25/preds/bm25.jsonl" \
  "${iso_reranker_args[@]}" --route-json "$ROUTE_CONFIG" \
  --frozen-route-specs "$RUN_ROOT/selection/retrieval/routes.json" \
  "${iso_route_args[@]}" --out-dir "$ISO_ROOT/retrieval"

for id in "${iso_route_ids[@]}"; do
  depth="${id##*planner}"
  "$PYTHON_BIN" "$SANITIZER" --split test --input "$ISO_SOURCE/dev.jsonl" \
    --rankings "$ISO_ROOT/retrieval/rankings/${id}.jsonl" --candidate-k "$depth" \
    --query-out "$ISO_ROOT/inputs/queries_label_free.jsonl" \
    --rag-out "$ISO_ROOT/inputs/${id}.jsonl" \
    --audit-out "$ISO_ROOT/inputs/${id}.audit.json"
done

run_iso_cell() {
  local id="$1" gpu="$2" depth="${1##*planner}"
  local stem="joint-isolated-sft8-${id}" batch=4
  if [[ "$depth" -ge 50 ]]; then batch=2; fi
  PYTHON_BIN="$PYTHON_BIN" GPU_LIST="$gpu" BATCH_SIZE="$batch" \
    bash "$RUNNER" "$BASE_8B" "$SFT_8B" "$ISO_ROOT/inputs/${id}.jsonl" rag \
      "$ISO_PRED_ROOT/$stem.jsonl" "$stem"
}
run_iso_cell rerank10_planner10 0 >"$ISO_ROOT/retrieval/logs/planner-anchor.log" 2>&1 & ia=$!
if [[ "$route_id" != rerank10_planner10 ]]; then
  run_iso_cell "$route_id" 1 >"$ISO_ROOT/retrieval/logs/planner-selected.log" 2>&1 & ib=$!
else
  ib=""
fi
wait "$ia"
if [[ -n "$ib" ]]; then wait "$ib"; fi
for id in "${iso_route_ids[@]}"; do
  stem="joint-isolated-sft8-${id}"
  "$PYTHON_BIN" "$EVALUATOR" --predictions "$ISO_PRED_ROOT/$stem.jsonl" \
    --gold "$ISO_SOURCE/dev.jsonl" --metrics-out "$ISO_ROOT/eval/$stem.metrics.json" \
    --scored-out "$ISO_ROOT/eval/$stem.scored.jsonl" --run-id "$stem"
done

"$PYTHON_BIN" "$PROJECT_DIR/scripts/report/summarize_planner_joint_depth_reselection.py" \
  --stage report

echo "protected joint-depth evaluation complete: route=$route_id"
