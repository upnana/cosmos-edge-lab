# stack3cam_wam

Personal WAM track on **Cosmos3-Edge** using my SO-101 stack-3-blocks 3-cam dataset.

## Data

- Raw LeRobot v3: `/home/july/datasets/stack_3blocks_white_blue_black_3cam`
- Vision processed: `data/processed/stack3cam_vision_sft/` (gitignored)
- Action reads LeRobot root directly via SO101 adapter (patched into framework)

## Recipes

| Track | Config | Launch |
|-------|--------|--------|
| Vision SFT | `configs/vision_sft_edge.toml` | `scripts/launch_vision_sft.sh` |
| Action policy | `configs/action_policy_so101_edge.toml` | `scripts/launch_action_policy.sh` |

## Run log (fill as you go)

| Date | Track | Steps | Notes |
|------|-------|-------|-------|
| | | | |

Baseline π0 (same task): Hugging Face `upna/pi0_stack_white_blue_black_3cam_060000` (and later 80k).
