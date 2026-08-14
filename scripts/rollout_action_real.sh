#!/usr/bin/env bash
# SO-101 closed-loop rollout — wires hooks via scripts/so101_rollout_driver.py
#
# This GPU host usually has no arm/cameras. Use --dry-run here; run --force-real
# on the bench PC. See docs/REAL_ROBOT_EVAL_CHECKLIST.md.
#
# Usage:
#   source scripts/env.sh
#   DRY_RUN=1 N_TRIALS=1 POLICY=zeros bash scripts/rollout_action_real.sh
#   DRY_RUN=1 N_TRIALS=1 POLICY=cosmos bash scripts/rollout_action_real.sh   # slow
#   FORCE_REAL=1 POLICY=cosmos N_TRIALS=20 bash scripts/rollout_action_real.sh
#   FORCE_REAL=1 POLICY=pi0    N_TRIALS=20 bash scripts/rollout_action_real.sh
set -euo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$LAB_ROOT/scripts/env.sh"

POLICY="${POLICY:-zeros}"
N_TRIALS="${N_TRIALS:-20}"
DRY_RUN="${DRY_RUN:-0}"
FORCE_REAL="${FORCE_REAL:-0}"
TIMEOUT_S="${TIMEOUT_S:-300}"
LAYOUT_CYCLE="${LAYOUT_CYCLE:-5}"
CHUNK="${CHUNK:-32}"
EXECUTE_STEPS="${EXECUTE_STEPS:-16}"
MAX_CHUNKS="${MAX_CHUNKS:-20}"
INTERACTIVE="${INTERACTIVE:-0}"
CONTROL_HZ="${CONTROL_HZ:-30}"
OUT_ROOT="${OUT_ROOT:-$LAB_ROOT/outputs/rollout_real}"
EXPORT="${EXPORT:-$LAB_ROOT/outputs/export/action_stack3cam_action_policy_edge_2000}"
PI0_CKPT="${PI0_CKPT:-/home/july/lerobot_alohamini/outputs/train/pi0_stack_white_blue_black_3cam/checkpoints/080000/pretrained_model}"
# Bench PC (wenxingnan) defaults; override on other hosts.
if [[ -d /home/rxn/lerobot_alohamini/src ]]; then
  LEROBOT_SRC="${LEROBOT_SRC:-/home/rxn/lerobot_alohamini/src}"
else
  LEROBOT_SRC="${LEROBOT_SRC:-/home/july/lerobot_alohamini/src}"
fi
# Prefer alohamini conda (has so_follower) for real/π0; cosmos subprocess uses FRAMEWORK_PYTHON.
if [[ -x /home/rxn/miniconda3/envs/lerobot_alohamini/bin/python ]]; then
  LEROBOT_PYTHON="${LEROBOT_PYTHON:-/home/rxn/miniconda3/envs/lerobot_alohamini/bin/python}"
else
  LEROBOT_PYTHON="${LEROBOT_PYTHON:-/home/july/miniconda3/envs/lerobot_alohamini/bin/python}"
fi
FRAMEWORK_PYTHON="${FRAMEWORK_PYTHON:-$COSMOS_FRAMEWORK_ROOT/.venv/bin/python}"
SO101_PORT="${SO101_PORT:-/dev/ttyUSB0}"
SO101_ID="${SO101_ID:-so101_follower}"
CAM_FRONT="${CAM_FRONT:-0}"
CAM_WRIST="${CAM_WRIST:-1}"
# Action-policy only uses front+wrist. Leave CAM_SIDE empty unless you need side
# (do NOT reuse the wrist index — double-open causes OpenCV abort / exit 134).
CAM_SIDE="${CAM_SIDE:-}"
PROMPT="${PROMPT:-stack the blocks from bottom to top white then blue then black}"
# 1 = keep Cosmos weights resident across chunks (default). 0 = cold CLI each chunk.
COSMOS_WARM="${COSMOS_WARM:-1}"

# Normalize policy aliases for the driver
case "$POLICY" in
  cosmos_action_2000) POLICY=cosmos ;;
  pi0_80k) POLICY=pi0 ;;
esac

STAMP=$(date +%Y%m%d_%H%M%S)_$$
OUT_DIR="$OUT_ROOT/${POLICY}_${STAMP}"
mkdir -p "$OUT_DIR"

echo "=== SO-101 closed-loop rollout ==="
echo "POLICY=$POLICY  N_TRIALS=$N_TRIALS  DRY_RUN=$DRY_RUN  FORCE_REAL=$FORCE_REAL  COSMOS_WARM=$COSMOS_WARM"
echo "OUT_DIR=$OUT_DIR"
echo "EXPORT=$EXPORT"
echo "PI0_CKPT=$PI0_CKPT"
echo "SO101_PORT=$SO101_PORT  cams=$CAM_FRONT/$CAM_WRIST/$CAM_SIDE"
echo

MODE_ARGS=()
if [[ "$DRY_RUN" == "1" ]]; then
  MODE_ARGS+=(--dry-run)
elif [[ "$FORCE_REAL" == "1" ]]; then
  MODE_ARGS+=(--force-real)
  echo "SAFETY: e-stop must be reachable. max_relative_target=15° per step."
else
  echo "ERROR: set DRY_RUN=1 or FORCE_REAL=1" >&2
  exit 2
fi

INTERACTIVE_ARGS=()
if [[ "$INTERACTIVE" == "1" ]]; then
  INTERACTIVE_ARGS+=(--interactive)
else
  INTERACTIVE_ARGS+=(--no-interactive)
fi

WARM_ARGS=()
if [[ "$COSMOS_WARM" == "1" ]]; then
  WARM_ARGS+=(--cosmos-warm)
else
  WARM_ARGS+=(--no-cosmos-warm)
fi

# π0 / live robot: lerobot_alohamini python; cosmos cold-infer still uses FRAMEWORK_PYTHON inside driver.
DRIVER_PY="$LEROBOT_PYTHON"
if [[ "$POLICY" == "cosmos" || "$POLICY" == "zeros" || "$POLICY" == "hold" ]]; then
  # zeros/cosmos driver itself does not need PI0; cosmos subprocess uses FRAMEWORK_PYTHON.
  # Still use LEROBOT_PYTHON so SO101Follower import works when FORCE_REAL=1.
  if [[ ! -x "$DRIVER_PY" ]]; then
    DRIVER_PY="$FRAMEWORK_PYTHON"
  fi
fi

export PYTHONPATH="${LEROBOT_SRC}:${PYTHONPATH:-}"

exec "$DRIVER_PY" "$LAB_ROOT/scripts/so101_rollout_driver.py" \
  "${MODE_ARGS[@]}" \
  "${INTERACTIVE_ARGS[@]}" \
  "${WARM_ARGS[@]}" \
  --policy "$POLICY" \
  --n-trials "$N_TRIALS" \
  --timeout-s "$TIMEOUT_S" \
  --layout-cycle "$LAYOUT_CYCLE" \
  --chunk "$CHUNK" \
  --execute-steps "$EXECUTE_STEPS" \
  --max-chunks "$MAX_CHUNKS" \
  --control-hz "$CONTROL_HZ" \
  --out-dir "$OUT_DIR" \
  --export "$EXPORT" \
  --pi0-ckpt "$PI0_CKPT" \
  --framework-python "$FRAMEWORK_PYTHON" \
  --lerobot-src "$LEROBOT_SRC" \
  --port "$SO101_PORT" \
  --robot-id "$SO101_ID" \
  --cam-front "$CAM_FRONT" \
  --cam-wrist "$CAM_WRIST" \
  --cam-side "$CAM_SIDE" \
  --prompt "$PROMPT"
