#!/usr/bin/env bash
# Convert LeRobot v3 stack3cam -> Vision-SFT JSONL under data/processed/.
set -euo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$LAB_ROOT/scripts/env.sh"

OUT="${1:-$LAB_ROOT/data/processed/stack3cam_vision_sft}"
CAM="${CAMERA:-observation.images.front}"

mkdir -p "$OUT"
python "$LAB_ROOT/scripts/convert_lerobot_to_vision_sft.py" \
  --dataset-root "$SO101_ROOT" \
  --output-root "$OUT" \
  --camera "$CAM"

echo ">>> vision data: $OUT"
ls -la "$OUT/train" 2>/dev/null | head -20
