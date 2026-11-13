#!/usr/bin/env bash
set -euo pipefail

ADAPTER_4B="${1:?usage: run_assignment3_100tool_dpo_mining.sh <selected-4b-adapter> <early-4b-adapter> <selected-8b-adapter>}"
EARLY_ADAPTER_4B="${2:?missing early 4B adapter}"
ADAPTER_8B="${3:?missing selected 8B adapter}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda/envs/qwen-intent-ft/bin/python}"
EXP_DIR="$PROJECT_DIR/data/tool_call_trace/_aux/assignment3/assignment3_100tool_api"
PRED_DIR="$PROJECT_DIR/finetune/outputs/preds/tool_trace/assignment3_100tool_dpo_mining"
RUNTIME_DIR="$PROJECT_DIR/finetune/outputs/tooltrace-data-cache/runs"
INPUT="$EXP_DIR/train_rag_mining_label_free.jsonl"
RUNNER="$PROJECT_DIR/scripts/finetune/run_assignment3_eval.sh"
BUILDER="$PROJECT_DIR/scripts/finetune/build_dpo_preferences.py"
mkdir -p "$PRED_DIR" "$RUNTIME_DIR"

run_sampled() {
  local model="$1"
  local adapter="$2"
  local name="$3"
  local gpu_list="$4"
  local batch_size="$5"
  local seed="$6"
  PYTHON_BIN="$PYTHON_BIN" \
  GPU_LIST="$gpu_list" \
  BATCH_SIZE="$batch_size" \
  MAX_NEW_TOKENS=128 \
  DO_SAMPLE=true \
  TEMPERATURE=0.7 \
  TOP_P=0.8 \
  TOP_K=20 \
  SAMPLES_PER_PROMPT=1 \
  SAVE_EVERY=400 \
  SEED="$seed" \
    bash "$RUNNER" "$model" "$adapter" "$INPUT" rag \
      "$PRED_DIR/$name.jsonl" "$name" \
      >"$PRED_DIR/$name.infer.log" 2>&1
}

run_sampled \
  "$PROJECT_DIR/models/Qwen/Qwen3-4B" \
  "$ADAPTER_4B" \
  a3-100-train-selected_sft4-rag-sampled-s0 \
  0,1 32 20260731 &
p0="$!"
run_sampled \
  "$PROJECT_DIR/models/Qwen/Qwen3-4B" \
  "$EARLY_ADAPTER_4B" \
  a3-100-train-early_sft4-step20-rag-sampled-s0 \
  2,3 32 20260801 &
p1="$!"
run_sampled \
  "$PROJECT_DIR/models/Qwen/Qwen3-8B" \
  "$ADAPTER_8B" \
  a3-100-train-selected_sft8-rag-sampled-s0 \
  4,5,6,7 16 20260802 &
p2="$!"

status=0
wait "$p0" || status=1
wait "$p1" || status=1
wait "$p2" || status=1
[[ "$status" -eq 0 ]] || {
  echo "Assignment3 DPO mining inference failed; inspect per-source logs" >&2
  exit "$status"
}

"$PYTHON_BIN" "$BUILDER" \
  --train "$EXP_DIR/train_api/train.jsonl" \
  --dev "$PROJECT_DIR/data/tool_call_trace/_aux/assignment3/api_native_dev_2400/dev.jsonl" \
  --frozen "$PROJECT_DIR/data/tool_call_trace/test.jsonl" \
  --extra-blocked "$EXP_DIR/test_api/test.jsonl" \
  --predictions "selected_sft4_sampled_s0=$PRED_DIR/a3-100-train-selected_sft4-rag-sampled-s0.jsonl" \
  --predictions "early_sft4_step20_sampled_s0=$PRED_DIR/a3-100-train-early_sft4-step20-rag-sampled-s0.jsonl" \
  --predictions "selected_sft8_sampled_s0=$PRED_DIR/a3-100-train-selected_sft8-rag-sampled-s0.jsonl" \
  --out "$EXP_DIR/dpo_preferences_compact.jsonl" \
  --audit-out "$EXP_DIR/dpo_data_audit.json" \
  --runtime-all-out "$RUNTIME_DIR/assignment3_100tool_dpo_all.runtime.jsonl" \
  --runtime-train-out "$RUNTIME_DIR/assignment3_100tool_dpo_train.runtime.jsonl" \
  --runtime-monitor-out "$RUNTIME_DIR/assignment3_100tool_dpo_monitor.runtime.jsonl" \
  --min-pairs 5000 \
  --max-pairs 5000 \
  --max-per-query 2 \
  --min-core-ratio 0.70 \
  --no-allow-synthetic

"$PYTHON_BIN" - "$EXP_DIR/dpo_data_audit.json" <<'PY'
import json
import sys
from pathlib import Path

audit = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if not audit.get("strict_ok") or audit.get("final_pairs") != 5000:
    raise SystemExit(
        "DPO mining did not reach exactly 5000 audited real-error pairs; "
        "run another real-model sampling seed before training"
    )
PY

echo "Assignment3 100-tool DPO mining complete"
