#!/usr/bin/env bash
# Vision (world-model) SFT on Cosmos3-Edge — front-camera clips from stack3cam.
set -euo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$LAB_ROOT/scripts/env.sh"

export DATASET_PATH="${DATASET_PATH:-$LAB_ROOT/data/processed/stack3cam_vision_sft}"
export BASE_CHECKPOINT_PATH
export WAN_VAE_PATH
export NPROC_PER_NODE
export IMAGINAIRE_OUTPUT_ROOT
export OUTPUT_ROOT="${OUTPUT_ROOT:-$LAB_ROOT/outputs/train}"

# Prefer lab TOML; fall back to framework copy if needed.
TOML_FILE="$LAB_ROOT/configs/vision_sft_edge.toml"
# Framework launcher anchors WORKDIR to cosmos-framework — copy TOML there for this run.
FW_TOML="$COSMOS_FRAMEWORK_ROOT/examples/toml/sft_config/vision_sft_edge_lab.toml"
cp -f "$TOML_FILE" "$FW_TOML"

EXTRA_DATASET_CHECK='[[ -f "$DATASET_PATH/train/video_dataset_file.jsonl" ]] || { echo "ERROR: missing vision jsonl — run scripts/prepare_vision_data.sh" >&2; exit 1; }'

TAIL_OVERRIDES=(
  trainer.grad_accum_iter=4
  trainer.max_iter=500
  checkpoint.save_iter=100
  model.compile.enabled=false
  job.name=stack3cam_vision_sft_edge
  ${EXTRA_TAIL_OVERRIDES:-}
)

TOML_FILE="examples/toml/sft_config/vision_sft_edge_lab.toml"
cd "$COSMOS_FRAMEWORK_ROOT"
# shellcheck disable=SC1091
source "$COSMOS_FRAMEWORK_ROOT/examples/_sft_launcher_common.sh"
