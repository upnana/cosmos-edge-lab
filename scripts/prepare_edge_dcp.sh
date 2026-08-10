#!/usr/bin/env bash
# One-time: HF Cosmos3-Edge weights -> DCP under lab checkpoints/.
#
# Important: must use registry name "Cosmos3-Edge" so Cosmos3-Edge.yaml
# (includes ema, etc.) is loaded. Local path alone breaks convert (Missing key ema).
# We seed checkpoint_hf._path to EDGE_HF to avoid re-downloading.
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

echo ">>> converting registry Cosmos3-Edge (weights from $EDGE_HF) -> $OUT"
cd "$COSMOS_FRAMEWORK_ROOT"

# Local Wan VAE so convert does not re-download Wan-AI/Wan2.2-TI2V-5B.
mkdir -p pretrained/tokenizers/video/wan2pt2
ln -sfn "$WAN_VAE_PATH" pretrained/tokenizers/video/wan2pt2/Wan2.2_VAE.pth

# Proxy for any remaining HF calls inside isolated uv hf CLI.
export https_proxy="${https_proxy:-http://127.0.0.1:7897}"
export http_proxy="${http_proxy:-http://127.0.0.1:7897}"
export HTTPS_PROXY="$https_proxy" HTTP_PROXY="$http_proxy"
export HF_HUB_DISABLE_XET=1
export PYTHONUNBUFFERED=1

# Match convert_model_to_dcp.py: init_script before heavy imports.
EDGE_HF="$EDGE_HF" OUT="$OUT" WAN_VAE_PATH="$WAN_VAE_PATH" python - <<'PY'
from cosmos_framework.inference.common.init import init_script

init_script(env={"COSMOS_DEVICE": "cpu"})

import math
import os
import shutil
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.filesystem import FileSystemWriter
from torch.distributed.checkpoint.state_dict import get_model_state_dict

from cosmos_framework.checkpoint.dcp import CustomSavePlanner
from cosmos_framework.inference.args import OmniSetupOverrides
from cosmos_framework.inference.common.args import CheckpointOverrides
from cosmos_framework.inference.common.public_model_config import build_public_model_config
from cosmos_framework.inference.model import Cosmos3OmniConfig, Cosmos3OmniModel
from cosmos_framework.scripts.convert_model_to_dcp import _redirect_avae_to_local

local = Path(os.environ["EDGE_HF"]).resolve()
out = Path(os.environ["OUT"]).resolve()
out.mkdir(parents=True, exist_ok=True)

overrides = CheckpointOverrides(checkpoint_path="Cosmos3-Edge")
checkpoint_config = overrides.build_checkpoint(checkpoints=OmniSetupOverrides.CHECKPOINTS)
assert checkpoint_config.checkpoint_hf is not None
checkpoint_config.checkpoint_hf._path = str(local)

# Tokenizer / processor also call CheckpointDirHf.download("nvidia/Cosmos3-Edge").
# Force those to the local snapshot so convert works offline.
from cosmos_framework.utils.checkpoint_db import CheckpointDirHf, CheckpointFileHf

_orig_dir_download = CheckpointDirHf.download
_orig_file_download = CheckpointFileHf.download
wan_vae = Path(os.environ["WAN_VAE_PATH"]).resolve()

def _dir_download_prefer_local(self):
    if getattr(self, "repository", None) == "nvidia/Cosmos3-Edge" and local.is_dir():
        self._path = str(local)
        return self._path
    return _orig_dir_download(self)

def _file_download_prefer_local(self):
    if getattr(self, "filename", None) == "Wan2.2_VAE.pth" and wan_vae.is_file():
        self._path = str(wan_vae)
        return self._path
    return _orig_file_download(self)

CheckpointDirHf.download = _dir_download_prefer_local  # type: ignore[method-assign]
CheckpointFileHf.download = _file_download_prefer_local  # type: ignore[method-assign]

print("Loading model from", local)
hf_path = checkpoint_config.download_checkpoint()
_redirect_avae_to_local(hf_path)
model_dict = checkpoint_config.load_model_config_dict()
hf_config = Cosmos3OmniConfig(model=build_public_model_config(model_dict))
hf_model = Cosmos3OmniModel.from_pretrained_dcp(hf_path, config=hf_config)
state_dict = get_model_state_dict(hf_model.model)

max_shard_size = 5 * 1024**3
model_size = sum(p.numel() * p.element_size() for p in state_dict.values() if isinstance(p, torch.Tensor))
thread_count = max(1, math.ceil(model_size / max_shard_size))

print("Saving model...")
model_dir = out / "model"
if model_dir.exists():
    shutil.rmtree(model_dir)
storage_writer = FileSystemWriter(model_dir, thread_count=thread_count)
dcp.save(state_dict=state_dict, storage_writer=storage_writer, planner=CustomSavePlanner())
source_checkpoint_json = hf_path / "checkpoint.json"
if source_checkpoint_json.exists():
    shutil.copy(source_checkpoint_json, out / "checkpoint.json")
hf_config.save_pretrained(model_dir)
print(f"Saved checkpoint to {out}")
PY

echo ">>> DCP ready: $OUT"
ls -la "$OUT" | head
du -sh "$OUT"
