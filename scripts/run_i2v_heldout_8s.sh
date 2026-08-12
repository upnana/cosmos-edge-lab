#!/usr/bin/env bash
# Longer held-out I2V (~8s) from Vision 500 export.
set -euo pipefail
LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$LAB_ROOT/scripts/env.sh"

EXPORT="${EXPORT:-$LAB_ROOT/outputs/export/vision_stack3cam_vision_sft_edge_500}"
IN="${IN:-$LAB_ROOT/outputs/eval_vision_heldout/i2v_heldout_ep111_8s.json}"
OUT="${OUT:-$LAB_ROOT/outputs/eval_vision_heldout/i2v_out_8s}"
NUM_FRAMES="${NUM_FRAMES:-193}"   # 4n+1 @24fps ≈ 8.04s
FPS="${FPS:-24}"
LOG="$LAB_ROOT/outputs/logs/i2v_heldout_ep111_${NUM_FRAMES}f.log"

mkdir -p "$OUT"
cd "$COSMOS_FRAMEWORK_ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate
_VENV_NVIDIA_LIB="$(echo .venv/lib/python*/site-packages/nvidia/cu13/lib)"
[[ -d $_VENV_NVIDIA_LIB ]] && export LD_LIBRARY_PATH="$_VENV_NVIDIA_LIB"

echo ">>> I2V longer  frames=$NUM_FRAMES fps=$FPS  -> $OUT"
python -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  --no-use-torch-compile \
  --no-use-cuda-graphs \
  --no-guardrails \
  --checkpoint-path "$EXPORT" \
  -i "$IN" \
  -o "$OUT" \
  --resolution 480 \
  --num-frames "$NUM_FRAMES" \
  --fps "$FPS" \
  --shift 5.0 \
  --guidance 6.0 \
  --seed 0 \
  2>&1 | tee "$LOG"

echo ">>> DONE  look for vision.mp4 under $OUT"
