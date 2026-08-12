#!/usr/bin/env bash
# Real-robot (or dry-run) SO-101 stack rollout skeleton.
#
# Does NOT talk to hardware yet — fills trial logs + hooks you must wire to
# LeRobot / your SO-101 driver. See docs/REAL_ROBOT_EVAL_CHECKLIST.md.
#
# Usage:
#   source scripts/env.sh
#   DRY_RUN=1 N_TRIALS=2 bash scripts/rollout_action_real.sh
#   POLICY=cosmos_action_2000 N_TRIALS=20 bash scripts/rollout_action_real.sh
#   POLICY=pi0_80k N_TRIALS=20 bash scripts/rollout_action_real.sh
set -euo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$LAB_ROOT/scripts/env.sh"

POLICY="${POLICY:-cosmos_action_2000}"
N_TRIALS="${N_TRIALS:-20}"
DRY_RUN="${DRY_RUN:-0}"
TIMEOUT_S="${TIMEOUT_S:-90}"
LAYOUT_CYCLE="${LAYOUT_CYCLE:-5}"   # layout ids 0..LAYOUT_CYCLE-1
OUT_ROOT="${OUT_ROOT:-$LAB_ROOT/outputs/rollout_real}"
EXPORT="${EXPORT:-$LAB_ROOT/outputs/export/action_stack3cam_action_policy_edge_2000}"
CHUNK="${CHUNK:-32}"
PROMPT="${PROMPT:-stack the blocks from bottom to top white then blue then black}"

STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="$OUT_ROOT/${POLICY}_${STAMP}"
VIDEO_DIR="$OUT_DIR/videos"
mkdir -p "$OUT_DIR" "$VIDEO_DIR"

echo "=== real rollout skeleton ==="
echo "POLICY=$POLICY  N_TRIALS=$N_TRIALS  DRY_RUN=$DRY_RUN  TIMEOUT_S=$TIMEOUT_S"
echo "OUT_DIR=$OUT_DIR"
echo "EXPORT=$EXPORT"
echo
echo "SAFETY: confirm e-stop is reachable before DRY_RUN=0."
echo

# ---------------------------------------------------------------------------
# HOOKS — replace these functions with your robot stack
# ---------------------------------------------------------------------------
robot_connect() {
  echo "[hook] robot_connect — TODO: LeRobot SO-101 / serial / ROS"
  if [[ "$DRY_RUN" != "1" && "${FORCE_REAL:-0}" != "1" ]]; then
    echo "ERROR: refusing real mode without FORCE_REAL=1 (skeleton safety)." >&2
    echo "  Set FORCE_REAL=1 only after hooks are implemented and e-stop tested." >&2
    exit 2
  fi
}

robot_home() {
  echo "[hook] robot_home — TODO: go to home joints"
}

robot_reset_scene() {
  local layout_id="$1"
  echo "[hook] robot_reset_scene layout=$layout_id — TODO: human places blocks / fixture"
  # In practice: pause for operator Enter
  if [[ "${INTERACTIVE:-1}" == "1" ]]; then
    read -r -p "  Scene ready for layout=$layout_id? Enter to continue / Ctrl-C abort: " _
  fi
}

capture_obs_concat() {
  # Should write front|wrist 256 concat frame/video path compatible with policy
  local out_path="$1"
  echo "[hook] capture_obs_concat -> $out_path — TODO: grab cameras, resize, hstack"
  # placeholder black frame list marker
  echo "placeholder" >"$out_path.stub"
}

policy_infer_chunk() {
  # Args: obs_stub_path -> prints path to actions_raw.json [T,6] degrees
  local obs_stub="$1"
  local actions_out="$2"
  echo "[hook] policy_infer_chunk policy=$POLICY chunk=$CHUNK"
  case "$POLICY" in
    cosmos_action_2000|cosmos*)
      echo "  TODO: call cosmos inference (model_mode=wam or policy) with EXPORT=$EXPORT"
      echo "  TODO: denorm meanstd -> raw using so101_stack_3cam_meanstd.json"
      # Offline fallback demo: copy zeros so the logging path works in dry-run
      "$COSMOS_FRAMEWORK_ROOT/.venv/bin/python" - <<PY
import json, numpy as np
from pathlib import Path
T, D = $CHUNK, 6
Path("$actions_out").write_text(json.dumps(np.zeros((T, D)).tolist()) + "\n")
print("$actions_out")
PY
      ;;
    pi0*|π0*)
      echo "  TODO: call your π0 LeRobot serve / predict entry"
      "$COSMOS_FRAMEWORK_ROOT/.venv/bin/python" - <<PY
import json, numpy as np
from pathlib import Path
Path("$actions_out").write_text(json.dumps(np.zeros(($CHUNK, 6)).tolist()) + "\n")
print("$actions_out")
PY
      ;;
    *)
      echo "ERROR: unknown POLICY=$POLICY" >&2
      exit 1
      ;;
  esac
}

robot_execute_chunk() {
  local actions_json="$1"
  echo "[hook] robot_execute_chunk $actions_json — TODO: stream joints at control Hz"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  DRY_RUN=1 — not sending commands"
  fi
}

start_video() {
  local path="$1"
  echo "[hook] start_video $path — TODO: ffmpeg/opencv record front"
}

stop_video() {
  echo "[hook] stop_video"
}

# ---------------------------------------------------------------------------
# Trial loop
# ---------------------------------------------------------------------------
robot_connect
robot_home

TRIALS_JSONL="$OUT_DIR/trials.jsonl"
: >"$TRIALS_JSONL"

for i in $(seq 1 "$N_TRIALS"); do
  layout=$(( (i - 1) % LAYOUT_CYCLE ))
  trial_id=$(printf "%s_t%03d" "$POLICY" "$i")
  echo
  echo ">>> trial $i/$N_TRIALS  id=$trial_id  layout=$layout"

  robot_home
  robot_reset_scene "$layout"

  vid="$VIDEO_DIR/${trial_id}.mp4"
  start_video "$vid"

  t0=$(date +%s)
  obs="$OUT_DIR/${trial_id}_obs"
  act="$OUT_DIR/${trial_id}_actions_raw.json"
  capture_obs_concat "$obs"
  policy_infer_chunk "$obs" "$act"
  robot_execute_chunk "$act"
  # TODO: closed-loop: while not terminal: capture -> infer -> execute until
  # success / fail / timeout. Skeleton runs a single chunk for wiring tests.

  t1=$(date +%s)
  dur=$((t1 - t0))
  stop_video

  success=""
  failure=""
  if [[ "${INTERACTIVE:-1}" == "1" ]]; then
    read -r -p "  Label success? [y/n/skip]: " ans || true
    case "${ans:-skip}" in
      y|Y|yes) success=true ;;
      n|N|no)
        success=false
        read -r -p "  failure_code (grasp_miss|wrong_order|knock_over|timeout|other): " failure
        ;;
      *) success="" ;;
    esac
  fi

  "$COSMOS_FRAMEWORK_ROOT/.venv/bin/python" - <<PY
import json
from pathlib import Path
rec = {
  "trial_id": "$trial_id",
  "policy": "$POLICY",
  "mode": "dry_run" if "$DRY_RUN" == "1" else "real",
  "prompt": """$PROMPT""",
  "layout_id": str($layout),
  "success": None if "$success" == "" else ("$success" == "true"),
  "failure_code": "$failure" or None,
  "duration_s": float($dur),
  "notes": "skeleton single-chunk; wire closed-loop in hooks",
  "video_path": "$vid",
  "action_raw_path": "$act",
  "meta": {"timeout_s": $TIMEOUT_S, "chunk": $CHUNK, "export": "$EXPORT"},
}
Path("$TRIALS_JSONL").open("a").write(json.dumps(rec, ensure_ascii=False) + "\n")
print(json.dumps(rec, ensure_ascii=False))
PY
done

"$COSMOS_FRAMEWORK_ROOT/.venv/bin/python" - <<PY
import json
from pathlib import Path
import sys
sys.path.insert(0, "$LAB_ROOT/scripts")
from rollout_common import TrialLogger
# rebuild summary from jsonl
logger = TrialLogger(Path("$OUT_DIR"))
# TrialLogger.summarize reads trials.jsonl we already wrote
print(json.dumps(logger.summarize(), indent=2))
PY

echo
echo "=== done ==="
echo "trials:  $TRIALS_JSONL"
echo "summary: $OUT_DIR/summary.json"
echo "Next: implement hooks, FORCE_REAL=1, fill docs/REAL_ROBOT_EVAL_CHECKLIST.md"
