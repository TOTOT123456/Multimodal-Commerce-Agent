#!/usr/bin/env bash
set -euo pipefail

SFT_4B="${1:?usage: run_assignment3_100tool_dpo_grid.sh <selected-4b-sft> <selected-8b-sft>}"
SFT_8B="${2:?missing selected 8B SFT checkpoint}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda/envs/qwen-intent-ft/bin/python}"
EXP_DIR="$PROJECT_DIR/data/tool_call_trace/_aux/assignment3/assignment3_100tool_api"
RUNTIME_DIR="$PROJECT_DIR/finetune/outputs/tooltrace-data-cache/runs"
PRED_DIR="$PROJECT_DIR/finetune/outputs/preds/tool_trace/assignment3_100tool_dpo_dev_selection"
TRAIN_DATA="$RUNTIME_DIR/assignment3_100tool_dpo_train.runtime.jsonl"
VAL_DATA="$RUNTIME_DIR/assignment3_100tool_dpo_monitor.runtime.jsonl"
STATS_FILE="$EXP_DIR/sft_prompt_stats.json"
RUNNER="$PROJECT_DIR/scripts/finetune/run_assignment3_eval.sh"
EVALUATOR="$PROJECT_DIR/scripts/finetune/evaluate_tool_trace.py"
SELECTOR="$PROJECT_DIR/scripts/finetune/select_assignment3_100tool_checkpoint.py"
DPO_RUNNER="$PROJECT_DIR/scripts/finetune/run_swift_tool_trace_dpo.sh"
DEV_DIR="$PROJECT_DIR/data/tool_call_trace/_aux/assignment3/api_native_dev_2400"
mkdir -p "$PRED_DIR"

run_dpo() {
  local model_size="$1"
  local run_id="$2"
  local sft_checkpoint="$3"
  local beta="$4"
  local gpu_list="$5"
  local nproc="$6"
  local batch_size="$7"
  local grad_acc="$8"
  local eval_batch="$9"
  local master_port="${10}"
  CUDA_VISIBLE_DEVICES="$gpu_list" \
  MASTER_PORT="$master_port" \
  NPROC_PER_NODE="$nproc" \
  BATCH_SIZE="$batch_size" \
  EVAL_BATCH_SIZE="$eval_batch" \
  GRAD_ACC="$grad_acc" \
  DATASET_NUM_PROC=8 \
  DATA_FILE="$TRAIN_DATA" \
  VAL_FILE="$VAL_DATA" \
  STATS_FILE="$STATS_FILE" \
  MAX_LENGTH=2048 \
  LR=5e-6 \
  PATH="/root/miniconda/envs/qwen-intent-ft/bin:$PATH" \
    bash "$DPO_RUNNER" "$model_size" "$run_id" "$sft_checkpoint" "$beta" 1 train \
      >"$PROJECT_DIR/finetune/outputs/tooltrace-$run_id.log" 2>&1
}

run_dpo 4b a3-100-dpo4-beta005-e1 "$SFT_4B" 0.05 0,1,2,3 4 2 2 2 29541 &
p0="$!"
run_dpo 4b a3-100-dpo4-beta010-e1 "$SFT_4B" 0.10 4,5,6,7 4 2 2 2 29542 &
p1="$!"
status=0
wait "$p0" || status=1
wait "$p1" || status=1
[[ "$status" -eq 0 ]] || {
  echo "4B DPO beta grid failed" >&2
  exit "$status"
}

endpoint() {
  local output_dir="$1"
  local selected
  selected="$(find "$output_dir" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)"
  [[ -n "$selected" && -f "$selected/adapter_model.safetensors" ]] || {
    echo "missing DPO endpoint: $output_dir" >&2
    return 1
  }
  printf '%s\n' "$selected"
}

OUT_005="$PROJECT_DIR/finetune/outputs/tooltrace-a3-100-dpo4-beta005-e1"
OUT_010="$PROJECT_DIR/finetune/outputs/tooltrace-a3-100-dpo4-beta010-e1"
CP_005="$(endpoint "$OUT_005")"
CP_010="$(endpoint "$OUT_010")"

run_dev() {
  local name="$1"
  local adapter="$2"
  local gpu_list="$3"
  local run_id="a3-100-${name}-dev-rag-k10"
  PYTHON_BIN="$PYTHON_BIN" GPU_LIST="$gpu_list" BATCH_SIZE=16 \
    bash "$RUNNER" \
      "$PROJECT_DIR/models/Qwen/Qwen3-4B" "$adapter" \
      "$DEV_DIR/rag_k10_label_free.jsonl" rag \
      "$PRED_DIR/$run_id.jsonl" "$run_id" \
      >"$PRED_DIR/$run_id.infer.log" 2>&1
}

run_dev dpo4-beta005 "$CP_005" 0,1,2,3 &
p0="$!"
run_dev dpo4-beta010 "$CP_010" 4,5,6,7 &
p1="$!"
status=0
wait "$p0" || status=1
wait "$p1" || status=1
[[ "$status" -eq 0 ]] || {
  echo "4B DPO development inference failed" >&2
  exit "$status"
}

for name in dpo4-beta005 dpo4-beta010; do
  run_id="a3-100-${name}-dev-rag-k10"
  "$PYTHON_BIN" "$EVALUATOR" \
    --predictions "$PRED_DIR/$run_id.jsonl" \
    --gold "$DEV_DIR/dev.jsonl" \
    --metrics-out "$PRED_DIR/$run_id.metrics.json" \
    --scored-out "$PRED_DIR/$run_id.scored.jsonl" \
    --run-id "$run_id"
done

SELECTION="$EXP_DIR/dpo4_beta_selection.json"
"$PYTHON_BIN" "$SELECTOR" \
  --candidate "beta005=$CP_005=$PRED_DIR/a3-100-dpo4-beta005-dev-rag-k10.metrics.json" \
  --candidate "beta010=$CP_010=$PRED_DIR/a3-100-dpo4-beta010-dev-rag-k10.metrics.json" \
  --out "$SELECTION"
SELECTED_4B="$("$PYTHON_BIN" - "$SELECTION" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["selected"]["adapter"])
PY
)"
if [[ "$SELECTED_4B" == "$CP_005" ]]; then
  SELECTED_BETA=0.05
  BETA_TAG=005
elif [[ "$SELECTED_4B" == "$CP_010" ]]; then
  SELECTED_BETA=0.10
  BETA_TAG=010
else
  echo "selected adapter is not part of the fixed beta grid" >&2
  exit 2
fi

RUN_8B="a3-100-dpo8-beta${BETA_TAG}-e1"
run_dpo 8b "$RUN_8B" "$SFT_8B" "$SELECTED_BETA" 0,1,2,3,4,5,6,7 8 1 2 1 29543
OUT_8B="$PROJECT_DIR/finetune/outputs/tooltrace-$RUN_8B"
CP_8B="$(endpoint "$OUT_8B")"

"$PYTHON_BIN" "$PROJECT_DIR/scripts/finetune/audit_tool_trace_dpo_runs.py" \
  --run "4b_beta005=$OUT_005" \
  --run "4b_beta010=$OUT_010" \
  --run "8b_selected_beta=$OUT_8B" \
  --out "$EXP_DIR/dpo_training_runs_audit.json"

"$PYTHON_BIN" - "$SELECTION" "$CP_8B" "$SELECTED_BETA" "$EXP_DIR/dpo_final_selection.json" <<'PY'
import json
import sys
from pathlib import Path
from hashlib import sha256

selection_path, checkpoint_8b, beta, output_path = sys.argv[1:]
selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))

def digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

checkpoint_4b = Path(selection["selected"]["adapter"])
checkpoint_8b_path = Path(checkpoint_8b)
payload = {
    "status": "complete",
    "selection_split": "assignment3_api_native_dev_2400",
    "official_test_used": False,
    "selected_beta": float(beta),
    "selected_4b": {
        **selection["selected"],
        "adapter_model_sha256": digest(checkpoint_4b / "adapter_model.safetensors"),
    },
    "selected_8b": {
        "adapter": str(checkpoint_8b_path),
        "adapter_model_sha256": digest(checkpoint_8b_path / "adapter_model.safetensors"),
        "beta_inherited_from_4b_dev_grid": float(beta),
    },
    "beta_grid": selection["ranked_candidates"],
}
Path(output_path).write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY

echo "Assignment3 DPO grid complete selected_4b=$SELECTED_4B selected_8b=$CP_8B beta=$SELECTED_BETA"
