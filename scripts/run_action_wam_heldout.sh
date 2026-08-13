#!/usr/bin/env bash
# Offline Action-Policy eval on held-out SO-101:
#   prepare inputs -> export -> WAM (action+vision) -> forward_dynamics (vision) -> score
set -euo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$LAB_ROOT/scripts/env.sh"

JOB_NAME="${JOB_NAME:-stack3cam_action_policy_edge_2000}"
ITER="${ITER:-2000}"
EPISODES="${EPISODES:-111}"
CHUNK_LENGTH="${CHUNK_LENGTH:-32}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
FPS="${FPS:-30}"
# Default: pick max-motion window (idle starts give vacuous action MSE).
AUTO_MOTION="${AUTO_MOTION:-1}"
START_FRAME="${START_FRAME:-}"

EVAL_ROOT="${EVAL_ROOT:-$LAB_ROOT/outputs/eval_action_wam}"
INPUT_DIR="$EVAL_ROOT/inputs"
WAM_OUT="$EVAL_ROOT/wam_out"
FD_OUT="$EVAL_ROOT/fd_out"
EXPORT="${EXPORT:-$LAB_ROOT/outputs/export/action_${JOB_NAME}}"
LOG_DIR="$LAB_ROOT/outputs/logs"
mkdir -p "$LOG_DIR" "$INPUT_DIR" "$WAM_OUT" "$FD_OUT"

PREP_ARGS=(
  --root "$SO101_ROOT"
  --out-dir "$INPUT_DIR"
  --episodes $EPISODES
  --chunk-length "$CHUNK_LENGTH"
  --image-size "$IMAGE_SIZE"
  --fps "$FPS"
)
if [[ -n "$START_FRAME" ]]; then
  PREP_ARGS+=(--start-frame "$START_FRAME" --no-auto-motion-window)
elif [[ "$AUTO_MOTION" == "1" ]]; then
  PREP_ARGS+=(--auto-motion-window)
else
  PREP_ARGS+=(--start-frame 0 --no-auto-motion-window)
fi

echo ">>> [1/4] prepare held-out inputs  episodes=$EPISODES"
# shellcheck disable=SC2086
"$COSMOS_FRAMEWORK_ROOT/.venv/bin/python" \
  "$LAB_ROOT/scripts/prepare_action_wam_eval_inputs.py" \
  "${PREP_ARGS[@]}"

echo ">>> [2/4] export (skip if present)"
bash "$LAB_ROOT/scripts/export_action_policy_2000.sh"

echo ">>> [3/4] WAM + forward_dynamics inference"
activate_fw() {
  # shellcheck disable=SC1091
  source "$COSMOS_FRAMEWORK_ROOT/.venv/bin/activate"
  _VENV_NVIDIA_LIB="$(echo "$COSMOS_FRAMEWORK_ROOT"/.venv/lib/python*/site-packages/nvidia/cu13/lib)"
  [[ -d $_VENV_NVIDIA_LIB ]] && export LD_LIBRARY_PATH="$_VENV_NVIDIA_LIB"
}

run_inference() {
  local mode="$1"
  local in_glob="$2"
  local out_dir="$3"
  local log="$4"
  echo ">>> inference mode=$mode  -> $out_dir"
  (
    cd "$COSMOS_FRAMEWORK_ROOT"
    activate_fw
    # shellcheck disable=SC2086
    python -m cosmos_framework.scripts.inference \
      --parallelism-preset=latency \
      --no-use-torch-compile \
      --no-use-cuda-graphs \
      --no-guardrails \
      --checkpoint-path "$EXPORT" \
      -i $in_glob \
      -o "$out_dir" \
      --resolution "$IMAGE_SIZE" \
      --fps "$FPS" \
      --seed 0
  ) 2>&1 | tee "$log"
}

# Build per-episode input globs from prepared dirs.
WAM_INPUTS=()
FD_INPUTS=()
for ep in $EPISODES; do
  ep_tag=$(printf '%03d' "$ep")
  wam_json="$INPUT_DIR/ep${ep_tag}/wam_ep${ep_tag}.json"
  fd_json="$INPUT_DIR/ep${ep_tag}/fd_ep${ep_tag}.json"
  [[ -f "$wam_json" ]] || { echo "ERROR: missing $wam_json" >&2; exit 1; }
  [[ -f "$fd_json" ]] || { echo "ERROR: missing $fd_json" >&2; exit 1; }
  WAM_INPUTS+=("$wam_json")
  FD_INPUTS+=("$fd_json")
done

GPU_MON_DIR=""
if [[ "${MONITOR_GPU:-1}" == "1" ]]; then
  GPU_MON_DIR=$(bash "$LAB_ROOT/scripts/monitor_gpu.sh" start --tag action_wam_eval)
  trap 'bash "$LAB_ROOT/scripts/monitor_gpu.sh" stop --tag action_wam_eval --out-dir "$GPU_MON_DIR" >/dev/null || true' EXIT
fi

run_inference wam "${WAM_INPUTS[*]}" "$WAM_OUT" "$LOG_DIR/action_wam_heldout.log"
run_inference forward_dynamics "${FD_INPUTS[*]}" "$FD_OUT" "$LOG_DIR/action_fd_heldout.log"

if [[ -n "$GPU_MON_DIR" ]]; then
  trap - EXIT
  bash "$LAB_ROOT/scripts/monitor_gpu.sh" stop --tag action_wam_eval --out-dir "$GPU_MON_DIR" >/dev/null || true
  echo ">>> GPU monitor: $GPU_MON_DIR/summary.json"
fi

echo ">>> [4/4] score"
"$COSMOS_FRAMEWORK_ROOT/.venv/bin/python" \
  "$LAB_ROOT/scripts/score_action_wam_eval.py" \
  --eval-root "$EVAL_ROOT" \
  --episodes $EPISODES \
  --stats-path "$COSMOS_FRAMEWORK_ROOT/cosmos_framework/data/generator/action/normalizer_stats/so101_stack_3cam_meanstd.json"

echo ">>> DONE"
echo "  inputs:  $INPUT_DIR"
echo "  export:  $EXPORT"
echo "  wam_out: $WAM_OUT"
echo "  fd_out:  $FD_OUT"
echo "  metrics: $EVAL_ROOT/metrics.json"
