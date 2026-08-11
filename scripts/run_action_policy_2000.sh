#!/usr/bin/env bash
# Non-smoke Action-Policy SFT (showcase track) — then stop; rollout is separate.
set -euo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$LAB_ROOT/scripts/env.sh"

LOG_DIR="$LAB_ROOT/outputs/logs"
mkdir -p "$LOG_DIR"
TRAIN_LOG="$LOG_DIR/action_policy_2000.log"

JOB_NAME="${JOB_NAME:-stack3cam_action_policy_edge_2000}"
MAX_ITER="${MAX_ITER:-2000}"
SAVE_ITER="${SAVE_ITER:-200}"
# Smoke used 2; TOML default 8. Use 4 as a 1xH100-safe showcase default.
MAX_SAMPLES="${MAX_SAMPLES:-4}"

echo ">>> Action-Policy SFT  job=$JOB_NAME  max_iter=$MAX_ITER  max_samples_per_batch=$MAX_SAMPLES"
export EXTRA_TAIL_OVERRIDES="job.name=${JOB_NAME} trainer.max_iter=${MAX_ITER} checkpoint.save_iter=${SAVE_ITER} dataloader_train.max_samples_per_batch=${MAX_SAMPLES}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

bash "$LAB_ROOT/scripts/launch_action_policy.sh" 2>&1 | tee "$TRAIN_LOG"

echo ">>> DONE training"
echo "  log:  $TRAIN_LOG"
echo "  ckpt: $LAB_ROOT/outputs/cosmos3/action_sft/${JOB_NAME}/checkpoints/"
echo "  next: offline/real rollout + vs π0 (not part of this script)"
