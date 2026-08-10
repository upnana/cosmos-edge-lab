#!/usr/bin/env bash
# Vision SFT 500 iter → export → held-out ep111 I2V.
set -euo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$LAB_ROOT/scripts/env.sh"

LOG_DIR="$LAB_ROOT/outputs/logs"
mkdir -p "$LOG_DIR"
TRAIN_LOG="$LOG_DIR/vision_sft_500.log"
EXPORT_LOG="$LOG_DIR/export_vision_500.log"
I2V_LOG="$LOG_DIR/i2v_vision_heldout_ep111.log"

JOB_NAME="${JOB_NAME:-stack3cam_vision_sft_edge_500}"
MAX_ITER="${MAX_ITER:-500}"
SAVE_ITER="${SAVE_ITER:-100}"

echo ">>> [1/3] Vision SFT  job=$JOB_NAME  max_iter=$MAX_ITER"
export EXTRA_TAIL_OVERRIDES="job.name=${JOB_NAME} trainer.max_iter=${MAX_ITER} checkpoint.save_iter=${SAVE_ITER}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
bash "$LAB_ROOT/scripts/launch_vision_sft.sh" 2>&1 | tee "$TRAIN_LOG"

RUN_DIR="$LAB_ROOT/outputs/cosmos3/sft/${JOB_NAME}"
CKPT="$RUN_DIR/checkpoints/iter_$(printf '%09d' "$MAX_ITER")"
if [[ ! -d "$CKPT" ]]; then
  # fallback: latest iter_* under checkpoints/
  CKPT="$(ls -d "$RUN_DIR"/checkpoints/iter_* 2>/dev/null | sort | tail -1 || true)"
fi
[[ -d "$CKPT" ]] || { echo "ERROR: no checkpoint under $RUN_DIR/checkpoints" >&2; exit 1; }

EXPORT="$LAB_ROOT/outputs/export/vision_${JOB_NAME}"
echo ">>> [2/3] export  $CKPT -> $EXPORT"
(
  cd "$COSMOS_FRAMEWORK_ROOT"
  # shellcheck disable=SC1091
  source .venv/bin/activate
  _VENV_NVIDIA_LIB="$(echo .venv/lib/python*/site-packages/nvidia/cu13/lib)"
  [[ -d $_VENV_NVIDIA_LIB ]] && export LD_LIBRARY_PATH="$_VENV_NVIDIA_LIB"
  python -m cosmos_framework.scripts.export_model \
    --checkpoint-path "$CKPT" \
    --config-file "$RUN_DIR/config.yaml" \
    --parallelism-preset=latency \
    --no-use-torch-compile \
    --no-use-cuda-graphs \
    -o "$EXPORT"
) 2>&1 | tee "$EXPORT_LOG"

EVAL="$LAB_ROOT/outputs/eval_vision_heldout"
IN="$EVAL/i2v_heldout_ep111.json"
OUT="$EVAL/i2v_out"
mkdir -p "$OUT"
echo ">>> [3/3] held-out I2V  ep111  -> $OUT"
(
  cd "$COSMOS_FRAMEWORK_ROOT"
  # shellcheck disable=SC1091
  source .venv/bin/activate
  _VENV_NVIDIA_LIB="$(echo .venv/lib/python*/site-packages/nvidia/cu13/lib)"
  [[ -d $_VENV_NVIDIA_LIB ]] && export LD_LIBRARY_PATH="$_VENV_NVIDIA_LIB"
  python -m cosmos_framework.scripts.inference \
    --parallelism-preset=latency \
    --no-use-torch-compile \
    --no-use-cuda-graphs \
    --no-guardrails \
    --checkpoint-path "$EXPORT" \
    -i "$IN" \
    -o "$OUT" \
    --resolution 480 \
    --num-frames 49 \
    --fps 24 \
    --shift 5.0 \
    --guidance 6.0 \
    --seed 0
) 2>&1 | tee "$I2V_LOG"

echo ">>> DONE"
echo "  ckpt:   $CKPT"
echo "  export: $EXPORT"
echo "  pred:   $OUT (look for vision.mp4)"
echo "  gt:     $EVAL/assets/gt_front_2s.mp4"
