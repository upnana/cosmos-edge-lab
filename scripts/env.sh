#!/usr/bin/env bash
# Shared environment for cosmos-edge-lab launches.
# Source from repo root:  source scripts/env.sh

set -euo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LAB_ROOT

# Local NVIDIA cosmos-framework checkout (engine). Override if needed.
: "${COSMOS_FRAMEWORK_ROOT:=$(cd "$LAB_ROOT/../cosmos-framework" 2>/dev/null && pwd || true)}"
if [[ -z "${COSMOS_FRAMEWORK_ROOT:-}" || ! -d "$COSMOS_FRAMEWORK_ROOT" ]]; then
  echo "ERROR: set COSMOS_FRAMEWORK_ROOT to your cosmos-framework install" >&2
  return 1 2>/dev/null || exit 1
fi
export COSMOS_FRAMEWORK_ROOT

# Activate framework venv by default.
# shellcheck disable=SC1091
source "$COSMOS_FRAMEWORK_ROOT/.venv/bin/activate"

# Bare-metal: expose venv NVIDIA cu13 libs for torchcodec/NPP; avoid host bleed-through.
_VENV_NVIDIA_LIB="$(echo "$COSMOS_FRAMEWORK_ROOT"/.venv/lib/python*/site-packages/nvidia/cu13/lib)"
if [[ -d $_VENV_NVIDIA_LIB ]]; then
  export LD_LIBRARY_PATH="$_VENV_NVIDIA_LIB"
else
  export LD_LIBRARY_PATH=
fi

# Defaults for this lab's first experiment.
: "${SO101_ROOT:=/home/july/datasets/stack_3blocks_white_blue_black_3cam}"
: "${EDGE_HF:=/home/july/models/Cosmos3-Edge}"
: "${BASE_CHECKPOINT_PATH:=$LAB_ROOT/checkpoints/Cosmos3-Edge-dcp}"
: "${WAN_VAE_PATH:=$COSMOS_FRAMEWORK_ROOT/examples/checkpoints/wan22_vae/Wan2.2_VAE.pth}"
: "${IMAGINAIRE_OUTPUT_ROOT:=$LAB_ROOT/outputs}"
: "${NPROC_PER_NODE:=1}"

export SO101_ROOT EDGE_HF BASE_CHECKPOINT_PATH WAN_VAE_PATH IMAGINAIRE_OUTPUT_ROOT NPROC_PER_NODE
export DATASET_PATH="${DATASET_PATH:-$LAB_ROOT/data/processed/stack3cam_vision_sft}"

echo "LAB_ROOT=$LAB_ROOT"
echo "COSMOS_FRAMEWORK_ROOT=$COSMOS_FRAMEWORK_ROOT"
echo "SO101_ROOT=$SO101_ROOT"
