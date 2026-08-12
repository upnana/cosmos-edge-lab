#!/usr/bin/env bash
# Wait for GPU headroom, then run multi-ep WAM/FD held-out eval.
set -euo pipefail
LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$LAB_ROOT/outputs/logs/action_wam_more_when_free.log"
NEED_FREE_MIB="${NEED_FREE_MIB:-50000}"
POLL_SEC="${POLL_SEC:-60}"

mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo ">>> $(date -Is) waiting for GPU free>=${NEED_FREE_MIB}MiB"
while true; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  echo "  $(date -Is) used=${used}MiB free=${free}MiB"
  if [[ "${free:-0}" -ge "$NEED_FREE_MIB" ]]; then
    break
  fi
  sleep "$POLL_SEC"
done

echo ">>> $(date -Is) GPU free enough — launching WAM/FD multi-ep"
source "$LAB_ROOT/scripts/env.sh"
export EPISODES="3 22 47 69 99"
export CHUNK_LENGTH=32
export EVAL_ROOT="$LAB_ROOT/outputs/eval_action_wam_more"
export JOB_NAME=stack3cam_action_policy_edge_2000
export ITER=2000
# Inputs already prepared; script will refresh them (ok) then export-skip + infer + score
bash "$LAB_ROOT/scripts/run_action_wam_heldout.sh"
bash "$LAB_ROOT/scripts/finalize_action_wam_more.sh"
echo ">>> $(date -Is) DONE multi-ep WAM/FD"
