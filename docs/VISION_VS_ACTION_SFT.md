# Vision SFT vs Action-Policy SFT

Both post-train **Cosmos3-Edge** on the same SO-101 stack-3cam teleop set.
They learn different things.

## Comparison

| | Vision SFT | Action-Policy SFT |
|---|---|---|
| Goal | World dynamics (“how the scene moves”) | Executable joints (“how the arm moves”) |
| Inputs | Front-cam clips + captions | Front+wrist images + proprio → action chunk |
| Outputs | Video / world-model generation | Absolute **6D** joint actions |
| Data shape | Convert to JSONL + mp4 first | Read LeRobot v3 directly via SO101 adapter |
| Cameras (v1) | `observation.images.front` only | Front + wrist concat (side unused) |

They are **two recipes**, not one automatic pipeline. You can run Vision only,
Action only, or both as parallel tracks.

## Vision SFT — how it is “trained”

1. **Convert** LeRobot episodes → short clips + caption JSONL  
   (`scripts/prepare_vision_data.sh` → `scripts/convert_lerobot_to_vision_sft.py`)
2. **Load** Cosmos3-Edge DCP + Wan VAE
3. **Optimize** generation-side modules (see `configs/vision_sft_edge.toml`
   `keys_to_select`: `moe_gen`, `time_embedder`, `vae2llm`, `llm2vae`,
   `k_norm_und_for_gen`)
4. Launch: `scripts/launch_vision_sft.sh`

Intuition: teach the model the desk / block stacking visual dynamics
(white → blue → black). Does **not** emit joint angles by itself.

## Action-Policy SFT — how it is “trained”

1. **No JSONL convert** — `SO101LeRobotDataset` reads  
   `/home/july/datasets/stack_3blocks_white_blue_black_3cam` (or `SO101_ROOT`)
2. **Normalize** absolute 6D joints with mean/std from dataset stats  
   (`patches/.../so101_stack_3cam_meanstd.json`)
3. **Train** with `action_gen=True`; action heads
   (`action2llm` / `llm2action` / …) plus WAM-style batching
4. Launch: `scripts/launch_action_policy.sh` (runs `sync_patches.sh` first)

Intuition: closer to “can move”; comparable *task* to π0 on the same data,
different model family.

## Shared prerequisites

1. Wan2.2 VAE downloaded under the framework tree  
2. HF Cosmos3-Edge → DCP (`scripts/prepare_edge_dcp.sh`; needs free GPU)  
3. Do not fight a long-running π0 job for the same H100 without planning

## Lab entrypoints

| Step | Script |
|------|--------|
| Vision data | `scripts/prepare_vision_data.sh` |
| Vision train | `scripts/launch_vision_sft.sh` |
| Action train | `scripts/launch_action_policy.sh` |
| Patch sync | `scripts/sync_patches.sh` |
