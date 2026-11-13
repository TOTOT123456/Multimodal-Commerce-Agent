#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PROJECT_DIR="$ROOT_DIR/PROJECT"
RUN_ROOT="$PROJECT_DIR/data/rag/_aux/retrieval_pipeline_reselection"
REPLAY_ROOT="$PROJECT_DIR/data/tool_call_trace/_aux/assignment3/retrieval_reselection_planner_replay_20260731"
PRED_ROOT="$PROJECT_DIR/finetune/outputs/preds/tool_trace/retrieval_reselection_planner_replay_20260731"
EVAL_ROOT="$REPLAY_ROOT/eval"
BASE_MODEL_4B="$PROJECT_DIR/models/Qwen/Qwen3-4B"
BASE_MODEL_8B="$PROJECT_DIR/models/Qwen/Qwen3-8B"
ADAPTER_4B="$PROJECT_DIR/finetune/outputs/tooltrace-a3-crossstyle-rag10-sft4-e1-lr5e5-bs16/checkpoint-2429"
ADAPTER_8B="$PROJECT_DIR/finetune/outputs/tooltrace-a3-crossstyle-rag10-sft8-e1-lr5e5-bs16/checkpoint-2429"
LEGACY="$PROJECT_DIR/data/tool_call_trace/test.jsonl"
API="$PROJECT_DIR/data/tool_call_trace/_aux/assignment3/assignment3_100tool_api/test_api/test.jsonl"
INTENT_PREDS="${INTENT_PREDS:-$PROJECT_DIR/finetune/outputs/preds/qwen3-0.6b-full_lora-context_isolated-best.jsonl}"
PYTHON_BIN="/root/miniconda/envs/llm_bge/bin/python"

while [[ ! -f "$RUN_ROOT/final_summary.json" \
      || ! -f "$RUN_ROOT/evaluation/legacy_frozen_6k/evaluation_manifest.json" \
      || ! -f "$RUN_ROOT/evaluation/api_frozen_1k/evaluation_manifest.json" ]]; do
  sleep 10
done
if [[ "$(jq -r '.status' "$RUN_ROOT/evaluation/legacy_frozen_6k/evaluation_manifest.json")" != "complete_no_post_evaluation_tuning" \
   || "$(jq -r '.status' "$RUN_ROOT/evaluation/api_frozen_1k/evaluation_manifest.json")" != "complete_no_post_evaluation_tuning" ]]; then
  echo "retrieval regressions are not complete" >&2
  exit 1
fi

mkdir -p "$REPLAY_ROOT" "$PRED_ROOT" "$EVAL_ROOT" "$REPLAY_ROOT/logs"
python3 "$PROJECT_DIR/scripts/report/analyze_retrieval_reselection_planner_replay.py" --stage begin

python3 "$PROJECT_DIR/scripts/finetune/sanitize_tool_trace_eval_inputs.py" \
  --split test \
  --input "$LEGACY" \
  --rankings "$RUN_ROOT/evaluation/legacy_frozen_6k/challenger/preds.jsonl" \
  --candidate-k 10 \
  --query-out "$REPLAY_ROOT/legacy_queries_label_free.jsonl" \
  --rag-out "$REPLAY_ROOT/legacy_rag_k10_label_free.jsonl" \
  --audit-out "$REPLAY_ROOT/legacy_label_free_k10_audit.json"

python3 "$PROJECT_DIR/scripts/finetune/sanitize_tool_trace_eval_inputs.py" \
  --split test \
  --input "$API" \
  --rankings "$RUN_ROOT/evaluation/api_frozen_1k/challenger/preds.jsonl" \
  --candidate-k 10 \
  --query-out "$REPLAY_ROOT/api1000_queries_label_free.jsonl" \
  --rag-out "$REPLAY_ROOT/api1000_rag_k10_label_free.jsonl" \
  --audit-out "$REPLAY_ROOT/api1000_label_free_k10_audit.json"

run_inference() {
  local model="$1" adapter="$2" input="$3" output="$4" run_id="$5"
  local gpu_list="$6" batch_size="$7"
  GPU_LIST="$gpu_list" BATCH_SIZE="$batch_size" PYTHON_BIN="$PYTHON_BIN" \
    bash "$PROJECT_DIR/scripts/finetune/run_assignment3_eval.sh" \
      "$model" "$adapter" "$input" rag "$output" "$run_id"
}

run_inference "$BASE_MODEL_4B" "$ADAPTER_4B" \
  "$REPLAY_ROOT/legacy_rag_k10_label_free.jsonl" \
  "$PRED_ROOT/reselection-crossstyle-4b-legacy6000.jsonl" \
  reselection-crossstyle-4b-legacy6000 0,1 16 \
  >"$REPLAY_ROOT/logs/infer_4b_legacy.log" 2>&1 & P1=$!
run_inference "$BASE_MODEL_8B" "$ADAPTER_8B" \
  "$REPLAY_ROOT/legacy_rag_k10_label_free.jsonl" \
  "$PRED_ROOT/reselection-crossstyle-8b-legacy6000.jsonl" \
  reselection-crossstyle-8b-legacy6000 2,3 4 \
  >"$REPLAY_ROOT/logs/infer_8b_legacy.log" 2>&1 & P2=$!
run_inference "$BASE_MODEL_4B" "$ADAPTER_4B" \
  "$REPLAY_ROOT/api1000_rag_k10_label_free.jsonl" \
  "$PRED_ROOT/reselection-crossstyle-4b-api1000.jsonl" \
  reselection-crossstyle-4b-api1000 4 16 \
  >"$REPLAY_ROOT/logs/infer_4b_api.log" 2>&1 & P3=$!
run_inference "$BASE_MODEL_8B" "$ADAPTER_8B" \
  "$REPLAY_ROOT/api1000_rag_k10_label_free.jsonl" \
  "$PRED_ROOT/reselection-crossstyle-8b-api1000.jsonl" \
  reselection-crossstyle-8b-api1000 5 4 \
  >"$REPLAY_ROOT/logs/infer_8b_api.log" 2>&1 & P4=$!

failed=0
for pid in "$P1" "$P2" "$P3" "$P4"; do
  if ! wait "$pid"; then failed=1; fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "one or more Planner replay inference workers failed" >&2
  exit 1
fi

for model in 4 8; do
  for split in legacy6000 api1000; do
    if [[ "$split" == legacy6000 ]]; then gold="$LEGACY"; else gold="$API"; fi
    stem="reselection-crossstyle-${model}b-${split}"
    "$PYTHON_BIN" "$PROJECT_DIR/scripts/finetune/evaluate_tool_trace.py" \
      --predictions "$PRED_ROOT/$stem.jsonl" \
      --gold "$gold" \
      --metrics-out "$EVAL_ROOT/$stem.metrics.json" \
      --scored-out "$EVAL_ROOT/$stem.scored.jsonl" \
      --run-id "$stem"
  done
  stem="reselection-crossstyle-${model}b-legacy6000"
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/finetune/evaluate_tool_trace_system.py" \
    --intent-predictions "$INTENT_PREDS" \
    --tool-predictions "$PRED_ROOT/$stem.jsonl" \
    --out "$EVAL_ROOT/$stem.system.jsonl" \
    --metrics-out "$EVAL_ROOT/$stem.system.metrics.json"
done

python3 "$PROJECT_DIR/scripts/report/analyze_retrieval_reselection_planner_replay.py" --stage complete
echo "complete frozen retrieval Planner replay"
