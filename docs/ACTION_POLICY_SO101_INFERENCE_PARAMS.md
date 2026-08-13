# SO-101 Action Policy (`action_policy_so101_edge`) — Parameters for Real-Robot Inference

Summary of the SO-101 stack3cam action-policy experiment and the exact parameter
values needed to run **real-robot (closed-loop) inference** with the trained
checkpoint. All values below were verified against the training run artifacts on
the training machine (paths are absolute paths on that machine).

## 1. Experiment layout

| What | Path |
| --- | --- |
| Experiment patch (Hydra `experiment=action_policy_so101_edge`) | `patches/cosmos-framework/action_policy_so101_edge.py` |
| SO-101 dataset adapter (front+wrist concat) | `patches/cosmos-framework/so101_lerobot_dataset.py` |
| Action mean/std stats | `patches/cosmos-framework/so101_stack_3cam_meanstd.json` |
| Patch manifest (copy targets into the framework checkout) | `patches/cosmos-framework/MANIFEST.txt` |
| Lab launch TOML (training) | `configs/action_policy_so101_edge.toml` |
| Trained DCP (2000 iters) | `/home/july/cosmos-edge-lab/outputs/cosmos3/action_sft/stack3cam_action_policy_edge_2000/checkpoints/iter_000002000` |
| Exported HF checkpoint | `/home/july/cosmos-edge-lab/outputs/export/action_stack3cam_action_policy_edge_2000` |
| Dataset root | `/home/july/datasets/stack_3blocks_white_blue_black_3cam` |

## 2. Domain ID mapping (`domain_utils.py`)

The dataset passes `domain_name="so101_follower"` and `ActionBaseDataset` looks it
up in `EMBODIMENT_TO_DOMAIN_ID` / `EMBODIMENT_TO_RAW_ACTION_DIM`
(`cosmos_framework/data/generator/action/domain_utils.py`). Inference performs the
same lookup, so these two entries are required:

```python
EMBODIMENT_TO_DOMAIN_ID["so101_follower"] = 22       # domain id used by the model
EMBODIMENT_TO_RAW_ACTION_DIM["so101_follower"] = 6   # raw 6D joint action width
```

> Note: these two lines are **not** part of the upstream `domain_utils.py` and are
> **not** shipped in `patches/` (the manifest only covers the dataset adapter,
> stats, and experiment patch). They were applied directly to the framework
> checkout on the training machine and were recovered from the runtime `.pyc`
> (`cosmos_framework/data/generator/action/__pycache__/domain_utils.cpython-313.pyc`).
> Re-add them to `cosmos_framework/data/generator/action/domain_utils.py` on any
> new machine before training or inference.

## 3. Viewpoint / camera concatenation

Training config sets `viewpoint="concat_view"`, `image_size=256`, `fps=30`,
`mode="wam"`, `use_state=True`, `action_normalization="meanstd"`,
`chunk_length=32` (see `patches/cosmos-framework/action_policy_so101_edge.py`).

In `SO101LeRobotDataset._load_video` (`patches/cosmos-framework/so101_lerobot_dataset.py`):

```python
if self._viewpoint == "third_person_view":
    keys = [_FRONT]                          # observation.images.front
elif self._viewpoint == "wrist_view":
    keys = [_WRIST]                          # observation.images.wrist
else:                                        # concat_view
    keys = [_FRONT, _WRIST]                  # front + wrist

# per key: decode_video_frames(..., backend="pyav") -> [T,C,H,W],
# F.interpolate(..., size=(image_size, image_size), bilinear)
# final: torch.cat(clips, dim=-1)  -> horizontal concat, front | wrist
```

Result: each camera is resized to 256×256 and concatenated along the width,
giving a **512×256** observation (`[T, C, H, 2W]`). The caption carries
`additional_view_description` = "The left half shows the front third-person
camera; the right half shows the wrist-mounted camera."

> The raw dataset has **three** cameras (`front`, `wrist`, `side`) — hence the
> "stack3cam" name — but the **action policy consumes only front + wrist**.
> `side` is unused by training and eval. (The vision SFT
> `stack3cam_vision_sft_edge` used only the front camera.)

## 4. Real-robot inference parameters

Input JSON for `python -m cosmos_framework.scripts.inference`:

```json
{
  "domain_name": "so101_follower",
  "view_point": "concat_view",
  "model_mode": "wam",
  "action_chunk_size": 32,
  "fps": 30,
  "image_size": 256,
  "resolution": "256",
  "seed": 0,
  "prompt": "stack the blocks from bottom to top white then blue then black",
  "vision_path": "concat_obs.mp4"
}
```

Notes:

- `vision_path` is the live `front|wrist` concatenated video (each 256×256,
  30 fps), path relative to the JSON file. On the bench PC, build it exactly as
  the eval prep does: `np.concatenate([front, wrist], axis=1)`
  (`scripts/prepare_action_wam_eval_inputs.py`).
- `action_chunk_size=32` is the training value (~1.07 s at 30 fps); held-out eval
  also used 240 (~8 s). Pick per control frequency.
- Output is a 6-D joint-position chunk:
  `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`
  (degrees; gripper 0–72 in the raw data).
- Actions are trained in **mean/std normalized** space; de-normalize with the
  stats in `so101_stack_3cam_meanstd.json` before sending to the robot.

Run:

```bash
export LD_LIBRARY_PATH=''   # NGC/PyTorch container requirement
python -m cosmos_framework.scripts.inference \
  -i <input.json> -o outputs/ \
  --checkpoint-path /home/july/cosmos-edge-lab/outputs/export/action_stack3cam_action_policy_edge_2000 \
  --seed 0
```

## 5. Reference eval inputs

Generated eval samples (WAM/FD JSONs + concat videos) live under
`outputs/eval_action_wam*/inputs/`; e.g.
`outputs/eval_action_wam_more_8s/inputs/ep003/wam_ep003.json`.
