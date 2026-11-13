#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PROJECT_DIR="$ROOT_DIR/PROJECT"
REPLAY_ROOT="$PROJECT_DIR/data/tool_call_trace/_aux/assignment3/retrieval_reselection_planner_replay_20260731"
MATRIX_ROOT="$PROJECT_DIR/data/tool_call_trace/_aux/assignment3/canonical_base_sft_matrix_20260801"
PRED_ROOT="$PROJECT_DIR/finetune/outputs/preds/tool_trace/canonical_base_sft_matrix_20260801"
EVAL_ROOT="$MATRIX_ROOT/eval"
LOG_ROOT="$MATRIX_ROOT/logs"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda/envs/llm_bge/bin/python}"

BASE_4B="$PROJECT_DIR/models/Qwen/Qwen3-4B"
BASE_8B="$PROJECT_DIR/models/Qwen/Qwen3-8B"
SFT_4B="$PROJECT_DIR/finetune/outputs/tooltrace-a3-crossstyle-rag10-sft4-e1-lr5e5-bs16/checkpoint-2429"
SFT_8B="$PROJECT_DIR/finetune/outputs/tooltrace-a3-crossstyle-rag10-sft8-e1-lr5e5-bs16/checkpoint-2429"
LEGACY_GOLD="$PROJECT_DIR/data/tool_call_trace/test.jsonl"
API_GOLD="$PROJECT_DIR/data/tool_call_trace/_aux/assignment3/assignment3_100tool_api/test_api/test.jsonl"
RUNNER="$PROJECT_DIR/scripts/finetune/run_assignment3_eval.sh"
EVALUATOR="$PROJECT_DIR/scripts/finetune/evaluate_tool_trace.py"
SUMMARIZER="$PROJECT_DIR/scripts/report/summarize_current_canonical_planner_matrix.py"
DEPTH_ROOT="$PROJECT_DIR/data/tool_call_trace/_aux/assignment3/canonical_planner_candidate_depth_isolated_20260801"
DEPTH_PRED_ROOT="$PROJECT_DIR/finetune/outputs/preds/tool_trace/canonical_planner_candidate_depth_isolated_20260801"
CURRENT_K="$(jq -r '.planner.candidate_k' "$PROJECT_DIR/configs/retrieval_canonical_route.json")"
CURRENT_R="$(jq -r '.reranker.candidate_k' "$PROJECT_DIR/configs/retrieval_canonical_route.json")"
JOINT_ROOT="$PROJECT_DIR/data/tool_call_trace/_aux/assignment3/planner_joint_depth_reselection_20260801"
JOINT_PRED_ROOT="$PROJECT_DIR/finetune/outputs/preds/tool_trace/planner_joint_depth_reselection_20260801/protected"
JOINT_ROUTE_ID="rerank${CURRENT_R}_planner${CURRENT_K}"
SECOND_STAGE_ROOT="$PROJECT_DIR/data/tool_call_trace/_aux/assignment3/r20p10_second_rrf_optimization_20260801"
SECOND_STAGE_PRED_ROOT="$PROJECT_DIR/finetune/outputs/preds/tool_trace/r20p10_second_rrf_optimization_20260801"
CANONICAL_REVISION="$(jq -r '.canonical_revision // ""' "$PROJECT_DIR/configs/retrieval_canonical_route.json")"

if [[ "$CANONICAL_REVISION" == r20p10_w040_k008 ]]; then
  for split in legacy6000 api1000; do
    for stage in sft base; do
      for model in 4b 8b; do
        stem="${stage}${model%b}_${CANONICAL_REVISION}"
        for required in \
          "$SECOND_STAGE_PRED_ROOT/protected_after_promotion/$split/$stem.jsonl" \
          "$SECOND_STAGE_ROOT/protected_after_promotion/$split/eval/$stem.metrics.json" \
          "$SECOND_STAGE_ROOT/protected_after_promotion/$split/eval/$stem.scored.jsonl"; do
          [[ -f "$required" ]] || { echo "missing post-promotion matrix artifact: $required" >&2; exit 2; }
        done
      done
    done
  done
  "$PYTHON_BIN" "$SUMMARIZER"
  echo "current canonical Base/SFT matrix summarized at $CANONICAL_REVISION"
  exit 0
fi

if [[ -f "$JOINT_ROOT/final_report.json" ]] \
  && jq -e --arg route "$JOINT_ROUTE_ID" \
    '.status == "complete_after_locked_route_on_protected_splits" and .route_id == $route' \
    "$JOINT_ROOT/final_report.json" >/dev/null; then
  for split in legacy6000 api1000; do
    for stage in sft base; do
      for model in 4b 8b; do
        stem="joint-final-${stage}-${model}-${split}-${JOINT_ROUTE_ID}"
        for required in \
          "$JOINT_PRED_ROOT/$stem.jsonl" \
          "$JOINT_ROOT/protected/$split/eval/$stem.metrics.json" \
          "$JOINT_ROOT/protected/$split/eval/$stem.scored.jsonl"; do
          [[ -f "$required" ]] || { echo "missing joint canonical matrix artifact: $required" >&2; exit 2; }
        done
      done
    done
  done
  "$PYTHON_BIN" "$SUMMARIZER"
  echo "current canonical Base/SFT matrix summarized at $JOINT_ROUTE_ID"
  exit 0
fi

if [[ "$CURRENT_K" != 10 ]]; then
  jq -e --argjson k "$CURRENT_K" \
    '.status == "complete_no_post_protected_split_tuning" and .final_planner_candidate_k == $k' \
    "$DEPTH_ROOT/summary.json" >/dev/null
  for split in legacy6000 api1000; do
    for stage in sft base; do
      for model in 4b 8b; do
        if [[ "$stage" == sft ]]; then
          stem="canonical-sft-${model}-${split}-k${CURRENT_K}"
        else
          stem="canonical-base-${model}-${split}-rag-k${CURRENT_K}"
        fi
        for required in \
          "$DEPTH_PRED_ROOT/protected/$stem.jsonl" \
          "$DEPTH_ROOT/protected/$stem.metrics.json" \
          "$DEPTH_ROOT/protected/$stem.scored.jsonl"; do
          [[ -f "$required" ]] || { echo "missing current K=$CURRENT_K matrix artifact: $required" >&2; exit 2; }
        done
      done
    done
  done
  "$PYTHON_BIN" "$SUMMARIZER"
  echo "current canonical Base/SFT matrix summarized at Planner K=$CURRENT_K"
  exit 0
fi

for required in \
  "$PROJECT_DIR/configs/retrieval_canonical_route.json" \
  "$BASE_4B" "$BASE_8B" "$SFT_4B" "$SFT_8B" \
  "$LEGACY_GOLD" "$API_GOLD" \
  "$REPLAY_ROOT/legacy_queries_label_free.jsonl" \
  "$REPLAY_ROOT/legacy_rag_k10_label_free.jsonl" \
  "$REPLAY_ROOT/api1000_queries_label_free.jsonl" \
  "$REPLAY_ROOT/api1000_rag_k10_label_free.jsonl"; do
  [[ -e "$required" ]] || { echo "missing required artifact: $required" >&2; exit 2; }
done

jq -e '
  .status == "canonical"
  and .first_stage.candidate_n_per_source == 40
  and .reranker.candidate_k == 10
  and .second_stage.output_k == 10
  and .planner.candidate_k == 10
  and .planner.decoding == "greedy"
' "$PROJECT_DIR/configs/retrieval_canonical_route.json" >/dev/null
jq -e '.strict_ok == true and .candidate_k == 10' \
  "$REPLAY_ROOT/legacy_label_free_k10_audit.json" >/dev/null
jq -e '.strict_ok == true and .candidate_k == 10' \
  "$REPLAY_ROOT/api1000_label_free_k10_audit.json" >/dev/null

mkdir -p "$PRED_ROOT" "$EVAL_ROOT" "$LOG_ROOT"

model_path() {
  case "$1" in
    4b) printf '%s\n' "$BASE_4B" ;;
    8b) printf '%s\n' "$BASE_8B" ;;
    *) echo "unsupported model size: $1" >&2; return 2 ;;
  esac
}

adapter_path() {
  local stage="$1" model="$2"
  if [[ "$stage" == base ]]; then
    printf '%s\n' none
  elif [[ "$model" == 4b ]]; then
    printf '%s\n' "$SFT_4B"
  elif [[ "$model" == 8b ]]; then
    printf '%s\n' "$SFT_8B"
  else
    echo "unsupported stage/model: $stage/$model" >&2
    return 2
  fi
}

input_path() {
  local split="$1" view="$2"
  case "$split/$view" in
    legacy6000/global) printf '%s\n' "$REPLAY_ROOT/legacy_queries_label_free.jsonl" ;;
    legacy6000/rag) printf '%s\n' "$REPLAY_ROOT/legacy_rag_k10_label_free.jsonl" ;;
    api1000/global) printf '%s\n' "$REPLAY_ROOT/api1000_queries_label_free.jsonl" ;;
    api1000/rag) printf '%s\n' "$REPLAY_ROOT/api1000_rag_k10_label_free.jsonl" ;;
    *) echo "unsupported split/view: $split/$view" >&2; return 2 ;;
  esac
}

gold_path() {
  case "$1" in
    legacy6000) printf '%s\n' "$LEGACY_GOLD" ;;
    api1000) printf '%s\n' "$API_GOLD" ;;
    *) echo "unsupported split: $1" >&2; return 2 ;;
  esac
}

batch_size() {
  local model="$1" view="$2"
  case "$model/$view" in
    4b/global) printf '8\n' ;;
    4b/rag) printf '16\n' ;;
    8b/global|8b/rag) printf '4\n' ;;
    *) echo "unsupported model/view: $model/$view" >&2; return 2 ;;
  esac
}

run_cell() {
  local stage="$1" model="$2" split="$3" view="$4" gpu_list="$5"
  local stem="canonical-${stage}-${model}-${split}-${view}"
  PYTHON_BIN="$PYTHON_BIN" GPU_LIST="$gpu_list" \
    BATCH_SIZE="$(batch_size "$model" "$view")" \
    bash "$RUNNER" \
      "$(model_path "$model")" \
      "$(adapter_path "$stage" "$model")" \
      "$(input_path "$split" "$view")" \
      "$view" \
      "$PRED_ROOT/$stem.jsonl" \
      "$stem"
}

run_logged() {
  local stage="$1" model="$2" split="$3" view="$4" gpu_list="$5"
  local stem="canonical-${stage}-${model}-${split}-${view}"
  run_cell "$stage" "$model" "$split" "$view" "$gpu_list" \
    >"$LOG_ROOT/$stem.infer.log" 2>&1
}

# Long full-catalog Legacy cells dominate runtime.  Give both 8B cells two
# shards, then immediately reuse each released GPU pair for the shorter cells.
run_logged base 4b legacy6000 global 0 & P_BASE4_GLOBAL=$!
run_logged base 8b legacy6000 global 1,2 & P_BASE8_GLOBAL=$!
run_logged sft 4b legacy6000 global 3 & P_SFT4_GLOBAL=$!
run_logged sft 8b legacy6000 global 4,5 & P_SFT8_GLOBAL=$!

wait_one() {
  local pid="$1" label="$2"
  if ! wait "$pid"; then
    echo "inference failed: $label" >&2
    return 1
  fi
}

wait_one "$P_BASE8_GLOBAL" base-8b-legacy6000-global
(
  run_logged base 8b legacy6000 rag 1
  run_logged base 8b api1000 rag 1
) & P_BASE8_RAG=$!
run_logged sft 8b api1000 global 2 & P_SFT8_API_GLOBAL=$!

wait_one "$P_SFT8_GLOBAL" sft-8b-legacy6000-global
(
  run_logged base 4b legacy6000 rag 4
  run_logged base 4b api1000 rag 4
) & P_BASE4_RAG=$!
run_logged sft 4b api1000 global 5 & P_SFT4_API_GLOBAL=$!

status=0
for item in \
  "$P_BASE4_GLOBAL:base-4b-legacy6000-global" \
  "$P_SFT4_GLOBAL:sft-4b-legacy6000-global" \
  "$P_BASE8_RAG:base-8b-rag" \
  "$P_SFT8_API_GLOBAL:sft-8b-api1000-global" \
  "$P_BASE4_RAG:base-4b-rag" \
  "$P_SFT4_API_GLOBAL:sft-4b-api1000-global"; do
  pid="${item%%:*}"
  label="${item#*:}"
  wait_one "$pid" "$label" || status=1
done
[[ "$status" -eq 0 ]] || exit "$status"

for split in legacy6000 api1000; do
  gold="$(gold_path "$split")"
  for stage in base sft; do
    for model in 4b 8b; do
      for view in global rag; do
        # Two route-equivalent cells already have complete, audited evidence:
        # API Base/global is retrieval-independent; SFT/RAG is the canonical
        # replay that established the current endpoint.
        if [[ "$split/$stage/$view" == api1000/base/global \
           || "$stage/$view" == sft/rag ]]; then
          continue
        fi
        stem="canonical-${stage}-${model}-${split}-${view}"
        "$PYTHON_BIN" "$EVALUATOR" \
          --predictions "$PRED_ROOT/$stem.jsonl" \
          --gold "$gold" \
          --metrics-out "$EVAL_ROOT/$stem.metrics.json" \
          --scored-out "$EVAL_ROOT/$stem.scored.jsonl" \
          --run-id "$stem"
      done
    done
  done
done

"$PYTHON_BIN" "$SUMMARIZER"
echo "current canonical Base/SFT matrix complete: $MATRIX_ROOT/summary.json"
