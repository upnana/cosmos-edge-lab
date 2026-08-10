#!/usr/bin/env bash
# Action-policy (WAM) SFT on Cosmos3-Edge — SO-101 6D + front/wrist.
set -euo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$LAB_ROOT/scripts/env.sh"

# Ensure SO101 adapters are present in the framework checkout.
bash "$LAB_ROOT/scripts/sync_patches.sh"

export SO101_ROOT
export DATASET_PATH="$SO101_ROOT"
export BASE_CHECKPOINT_PATH
export WAN_VAE_PATH
export NPROC_PER_NODE
export IMAGINAIRE_OUTPUT_ROOT
export OUTPUT_ROOT="${OUTPUT_ROOT:-$LAB_ROOT/outputs/train}"

FW_TOML="$COSMOS_FRAMEWORK_ROOT/examples/toml/sft_config/action_policy_so101_edge_lab.toml"
cp -f "$LAB_ROOT/configs/action_policy_so101_edge.toml" "$FW_TOML"

EXTRA_DATASET_CHECK='[[ -f "$SO101_ROOT/meta/info.json" ]] || { echo "ERROR: missing $SO101_ROOT/meta/info.json" >&2; exit 1; }'

TAIL_OVERRIDES=(
  job.name=stack3cam_action_policy_edge
  ${EXTRA_TAIL_OVERRIDES:-}
)

TOML_FILE="examples/toml/sft_config/action_policy_so101_edge_lab.toml"
cd "$COSMOS_FRAMEWORK_ROOT"
# shellcheck disable=SC1091
source "$COSMOS_FRAMEWORK_ROOT/examples/_sft_launcher_common.sh"
