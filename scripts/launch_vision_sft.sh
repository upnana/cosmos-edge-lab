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

# Framework launcher anchors WORKDIR to the parent of examples/ — so the
# actual torchrun entry must live under COSMOS_FRAMEWORK_ROOT/examples/.
FW_TOML="$COSMOS_FRAMEWORK_ROOT/examples/toml/sft_config/vision_sft_edge_lab.toml"
cp -f "$LAB_ROOT/configs/vision_sft_edge.toml" "$FW_TOML"

FW_LAUNCH="$COSMOS_FRAMEWORK_ROOT/examples/launch_sft_vision_edge_lab.sh"
cat > "$FW_LAUNCH" <<EOF
#!/usr/bin/env bash
set -euo pipefail
TOML_FILE="examples/toml/sft_config/vision_sft_edge_lab.toml"
: "\${DATASET_PATH:=$DATASET_PATH}"
: "\${BASE_CHECKPOINT_PATH:=$BASE_CHECKPOINT_PATH}"
: "\${WAN_VAE_PATH:=$WAN_VAE_PATH}"
: "\${NPROC_PER_NODE:=$NPROC_PER_NODE}"
: "\${OUTPUT_ROOT:=$OUTPUT_ROOT}"
: "\${IMAGINAIRE_OUTPUT_ROOT:=$IMAGINAIRE_OUTPUT_ROOT}"
EXTRA_DATASET_CHECK='[[ -f "\$DATASET_PATH/train/video_dataset_file.jsonl" ]] || { echo "ERROR: missing vision jsonl — run scripts/prepare_vision_data.sh" >&2; exit 1; }'
TAIL_OVERRIDES=(
  trainer.grad_accum_iter=4
  trainer.max_iter=500
  checkpoint.save_iter=100
  job.name=stack3cam_vision_sft_edge
  \${EXTRA_TAIL_OVERRIDES:-}
)
source "\$(dirname "\${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
EOF
chmod +x "$FW_LAUNCH"

echo ">>> launching Vision SFT via $FW_LAUNCH"
if [[ "${MONITOR_GPU:-1}" == "1" ]]; then
  bash "$LAB_ROOT/scripts/monitor_gpu.sh" wrap --tag vision_sft -- bash "$FW_LAUNCH"
else
  bash "$FW_LAUNCH"
fi
