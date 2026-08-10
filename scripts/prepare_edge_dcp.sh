#!/usr/bin/env bash
# One-time: HF Cosmos3-Edge weights -> DCP under lab checkpoints/.
set -euo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$LAB_ROOT/scripts/env.sh"

OUT="${1:-$BASE_CHECKPOINT_PATH}"
mkdir -p "$OUT"

if [[ ! -d "$EDGE_HF" ]]; then
  echo "ERROR: EDGE_HF not found: $EDGE_HF" >&2
  exit 1
fi

echo ">>> converting $EDGE_HF -> $OUT"
cd "$COSMOS_FRAMEWORK_ROOT"
python -m cosmos_framework.scripts.convert_model_to_dcp \
  --checkpoint-path "$EDGE_HF" \
  -o "$OUT"

echo ">>> DCP ready: $OUT"
