#!/usr/bin/env bash
# Export action-policy DCP (iter_000002000) -> HF safetensors for offline inference.
set -euo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$LAB_ROOT/scripts/env.sh"

JOB_NAME="${JOB_NAME:-stack3cam_action_policy_edge_2000}"
ITER="${ITER:-2000}"
RUN_DIR="${RUN_DIR:-$LAB_ROOT/outputs/cosmos3/action_sft/${JOB_NAME}}"
CKPT="${CKPT:-$RUN_DIR/checkpoints/iter_$(printf '%09d' "$ITER")}"
EXPORT="${EXPORT:-$LAB_ROOT/outputs/export/action_${JOB_NAME}}"
LOG_DIR="$LAB_ROOT/outputs/logs"
mkdir -p "$LOG_DIR" "$(dirname "$EXPORT")"
LOG="$LOG_DIR/export_action_policy_${ITER}.log"

if [[ ! -d "$CKPT" ]]; then
  echo "ERROR: checkpoint not found: $CKPT" >&2
  exit 1
fi
if [[ ! -f "$RUN_DIR/config.yaml" ]]; then
  echo "ERROR: missing config.yaml under $RUN_DIR" >&2
  exit 1
fi

if [[ "${FORCE_EXPORT:-0}" != "1" && -f "$EXPORT/config.json" ]]; then
  echo ">>> export already exists, skip: $EXPORT  (FORCE_EXPORT=1 to redo)"
  exit 0
fi

echo ">>> export action-policy  $CKPT -> $EXPORT"
# SO101 uses PackingDataLoader; upstream export_model metadata helper expects
# LIBERO/DROID dataloader layout. Use lab wrapper + Hydra experiment module.
(
  cd "$COSMOS_FRAMEWORK_ROOT"
  # shellcheck disable=SC1091
  source .venv/bin/activate
  _VENV_NVIDIA_LIB="$(echo .venv/lib/python*/site-packages/nvidia/cu13/lib)"
  [[ -d $_VENV_NVIDIA_LIB ]] && export LD_LIBRARY_PATH="$_VENV_NVIDIA_LIB"
  python "$LAB_ROOT/scripts/export_action_policy_so101.py" \
    --checkpoint-path "$CKPT" \
    --config-file cosmos_framework/configs/base/config.py \
    --experiment action_policy_so101_edge \
    --experiment-overrides \
      "model.config.tokenizer.vae_path=${WAN_VAE_PATH}" \
      "checkpoint.load_path=${CKPT}" \
    --parallelism-preset=latency \
    --no-use-torch-compile \
    --no-use-cuda-graphs \
    -o "$EXPORT"
) 2>&1 | tee "$LOG"

echo ">>> DONE export"
echo "  ckpt:   $CKPT"
echo "  export: $EXPORT"
echo "  log:    $LOG"
