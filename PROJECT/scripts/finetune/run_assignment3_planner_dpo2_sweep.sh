#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_ID="assignment3_planner_dpo2_20260808"
CACHE_DIR="$PROJECT_DIR/finetune/outputs/tooltrace-data-cache/$EXP_ID"
PREFERENCE_DATA="$PROJECT_DIR/data/tool_call_trace/_aux/assignment3/$EXP_ID/preferences/preferences.jsonl"
PARENT="$PROJECT_DIR/finetune/outputs/tooltrace-a3-crossstyle-simple-dpo8-lr1e6-b010-s20260808/checkpoint-29"
RUNNER="$PROJECT_DIR/scripts/finetune/run_assignment3_crossstyle_dpo.sh"
LOG_DIR="$PROJECT_DIR/data/tool_call_trace/_aux/assignment3/$EXP_ID/training_logs"
SEED="${SEED:-20260817}"

[[ -s "$CACHE_DIR/train.runtime.jsonl" ]] || { echo "missing DPO train data" >&2; exit 2; }
[[ -s "$CACHE_DIR/monitor.runtime.jsonl" ]] || { echo "missing DPO monitor data" >&2; exit 2; }
[[ -d "$PARENT" ]] || { echo "missing parent adapter" >&2; exit 2; }
mkdir -p "$LOG_DIR"

require_sha256() {
  local path="$1" expected="$2" actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || {
    echo "SHA256 mismatch: $path expected=$expected actual=$actual" >&2
    exit 2
  }
}
require_sha256 "$PREFERENCE_DATA" "8a27860b346d2a7e37e772031004b1c85a85de8771a086a2f6920a1f38604253"
require_sha256 "$CACHE_DIR/train.runtime.jsonl" "168eac613a8d617e8dcfddf843197a5bdae9bed8f04eb5adc0ede68732dd7e74"
require_sha256 "$CACHE_DIR/monitor.runtime.jsonl" "02d95a0af24c696336c5af639082bd405b387abe185faf222f7654de1a9fad7e"

names=(
  "a3-planner-dpo2-dpo-lr5e7-s${SEED}"
  "a3-planner-dpo2-rpo050-lr5e7-s${SEED}"
  "a3-planner-dpo2-rpo100-lr5e7-s${SEED}"
  "a3-planner-dpo2-rpo050-lr1e6-s${SEED}"
)
lrs=(5e-7 5e-7 5e-7 1e-6)
rpo_alphas=(none 0.5 1.0 0.5)
ports=(29500 29501 29502 29503)

for name in "${names[@]}"; do
  output="$PROJECT_DIR/finetune/outputs/tooltrace-$name"
  if [[ -f "$output/args.json" && "${ALLOW_EXISTING:-false}" != true ]]; then
    echo "existing run requires ALLOW_EXISTING=true: $output" >&2
    exit 2
  fi
done

run_one() {
  local gpu="$1" name="$2" lr="$3" rpo_alpha="$4" port="$5"
  common=(
    CUDA_VISIBLE_DEVICES="$gpu"
    MASTER_PORT="$port"
    DATA_FILE="$CACHE_DIR/train.runtime.jsonl"
    VAL_FILE="$CACHE_DIR/monitor.runtime.jsonl"
    BATCH_SIZE=2
    EVAL_BATCH_SIZE=1
    GLOBAL_BATCH_SIZE=16
    NPROC_PER_NODE=1
    LR="$lr"
    LOSS_TYPE=sigmoid
    EPOCHS=1
    SAVE_STEPS=0.125
    SAVE_TOTAL_LIMIT=8
    SEED="$SEED"
    DATA_SEED="$SEED"
    DATASET_NUM_PROC=4
  )
  if [[ "$rpo_alpha" == none ]]; then
    env "${common[@]}" bash "$RUNNER" 8b "$name" "$PARENT" 0.1
  else
    env "${common[@]}" RPO_ALPHA="$rpo_alpha" \
      bash "$RUNNER" 8b "$name" "$PARENT" 0.1
  fi
}

pids=()
for index in 0 1 2 3; do
  run_one \
    "$index" "${names[$index]}" "${lrs[$index]}" \
    "${rpo_alphas[$index]}" "${ports[$index]}" \
    >"$LOG_DIR/${names[$index]}.log" 2>&1 &
  pids+=("$!")
done

status=0
for index in 0 1 2 3; do
  if wait "${pids[$index]}"; then
    echo "complete ${names[$index]}"
  else
    echo "failed ${names[$index]}" >&2
    status=1
  fi
done
exit "$status"
