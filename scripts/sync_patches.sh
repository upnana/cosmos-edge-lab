#!/usr/bin/env bash
# Copy lab-owned SO101 / WAM adapters into a local cosmos-framework checkout,
# then ensure domain registry + experiment imports are wired.
set -euo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$LAB_ROOT/scripts/env.sh"

PATCH_DIR="$LAB_ROOT/patches/cosmos-framework"
MANIFEST="$PATCH_DIR/MANIFEST.txt"

echo ">>> syncing patches into $COSMOS_FRAMEWORK_ROOT"

while IFS= read -r line || [[ -n "$line" ]]; do
  [[ -z "$line" || "$line" =~ ^# ]] && continue
  src="${line%%|*}"
  dst="${line#*|}"
  src="$(echo "$src" | xargs)"
  dst="$(echo "$dst" | xargs)"
  mkdir -p "$(dirname "$COSMOS_FRAMEWORK_ROOT/$dst")"
  cp -f "$PATCH_DIR/$src" "$COSMOS_FRAMEWORK_ROOT/$dst"
  echo "  + $dst"
done < "$MANIFEST"

python3 - <<'PY'
from pathlib import Path
import os
import re

root = Path(os.environ["COSMOS_FRAMEWORK_ROOT"])

# 1) domain registry
du = root / "cosmos_framework/data/generator/action/domain_utils.py"
text = du.read_text()
changed = False
if '"so101_follower": 22' not in text:
    text = text.replace(
        '"drawanything": 21,\n}',
        '"drawanything": 21,\n    "so101_follower": 22,\n}',
        1,
    )
    if '"so101_follower": 22' not in text:
        raise SystemExit("Failed to insert so101_follower domain id")
    changed = True
if '"so101_follower": 6' not in text:
    text = text.replace(
        '"drawanything": 3,\n',
        '"drawanything": 3,\n    "so101_follower": 6,\n',
        1,
    )
    if '"so101_follower": 6' not in text:
        raise SystemExit("Failed to insert so101_follower action dim")
    changed = True
if changed:
    du.write_text(text)
    print("  ~ domain_utils.py (so101_follower)")
else:
    print("  = domain_utils.py already has so101_follower")

# 2) experiment import in config.py
cfg = root / "cosmos_framework/configs/base/config.py"
ct = cfg.read_text()
imp = (
    "import cosmos_framework.configs.base.experiment.action.posttrain_config"
    ".action_policy_so101_edge  # noqa: F401"
)
if imp not in ct and "action_policy_so101_edge" not in ct:
    m = re.search(
        r"^import cosmos_framework\.configs\.base\.experiment\.action\.posttrain_config\.[^\n]+$",
        ct,
        flags=re.M,
    )
    if m:
        insert_at = m.end()
        ct = ct[:insert_at] + "\n" + imp + ct[insert_at:]
    else:
        ct = ct.rstrip() + "\n" + imp + "\n"
    cfg.write_text(ct)
    print("  ~ config.py (register action_policy_so101_edge)")
else:
    print("  = config.py already registers action_policy_so101_edge")

# 3) get_action_so101_sft_dataset wiring
ads = root / "cosmos_framework/data/generator/action/datasets/action_sft_dataset.py"
at = ads.read_text()
orig = at
need_import = (
    "from cosmos_framework.data.generator.action.datasets.so101_lerobot_dataset "
    "import SO101LeRobotDataset"
)
if need_import not in at:
    droid_imp = (
        "from cosmos_framework.data.generator.action.datasets.droid_lerobot_dataset "
        "import DROIDLeRobotDataset"
    )
    if droid_imp in at:
        at = at.replace(droid_imp, droid_imp + "\n" + need_import, 1)
    else:
        at = need_import + "\n" + at
    print("  ~ action_sft_dataset.py (import SO101LeRobotDataset)")
if "def get_action_so101_sft_dataset" not in at:
    at = at.rstrip() + """

def get_action_so101_sft_dataset(
    *,
    root: str,
    fps: float = 30.0,
    chunk_length: int = 32,
    mode: str = "wam",
    use_state: bool = True,
    action_normalization: str | None = "meanstd",
    viewpoint: str = "concat_view",
    image_size: int = 256,
    resolution: str | int = "256",
    max_action_dim: int = 64,
    tokenizer_config: dict | None = None,
    cfg_dropout_rate: float = 0.1,
    append_viewpoint_info: bool = True,
    append_duration_fps_timestamps: bool = True,
    append_resolution_info: bool = True,
    append_idle_frames: bool = False,
    format_prompt_as_json: bool = True,
    iterable_shuffle: bool = False,
    episode_shuffle_seed: int = 42,
    sample_stride: int = 1,
) -> Dataset:
    \"\"\"Build SO-101 stack/3-cam action-policy SFT dataset (absolute 6D joints).\"\"\"
    dataset = SO101LeRobotDataset(
        root=root,
        fps=fps,
        chunk_length=chunk_length,
        mode=mode,
        use_state=use_state,
        action_normalization=action_normalization,
        viewpoint=viewpoint,
        image_size=image_size,
        sample_stride=sample_stride,
    )
    transform = ActionTransformPipeline(
        tokenizer_config=tokenizer_config,
        cfg_dropout_rate=cfg_dropout_rate,
        max_action_dim=max_action_dim,
        append_viewpoint_info=append_viewpoint_info,
        append_duration_fps_timestamps=append_duration_fps_timestamps,
        append_resolution_info=append_resolution_info,
        append_idle_frames=append_idle_frames,
        format_prompt_as_json=format_prompt_as_json,
    )
    wrapped = ActionSFTDataset(dataset, transform, resolution)
    if iterable_shuffle:
        return ActionIterableShuffleDataset(wrapped, seed=episode_shuffle_seed)
    return wrapped
"""
    print("  ~ action_sft_dataset.py (added get_action_so101_sft_dataset)")
elif at == orig:
    print("  = action_sft_dataset.py already has get_action_so101_sft_dataset")
if at != orig:
    ads.write_text(at)

print(">>> patches synced")
PY

echo "Done. Re-run after pulling a fresh cosmos-framework."
