# Where the code lives (and what to edit)

Lab scripts are **entrypoints**. Real logic sits in lab Python/TOML/patches;
`cosmos-framework` is the runtime engine.

## `prepare_vision_data.sh`

Path: `scripts/prepare_vision_data.sh`

| Role | Path |
|------|------|
| Entrypoint | `scripts/prepare_vision_data.sh` |
| Converter (edit this) | `scripts/convert_lerobot_to_vision_sft.py` |
| Upstream reference | `cosmos-framework/tools/lerobot3cam_to_vision_sft.py` |
| Default input | `SO101_ROOT` → stack3cam LeRobot root |
| Default output | `data/processed/stack3cam_vision_sft/` |

Edit tips:

- Camera / paths → env (`CAMERA`, `SO101_ROOT`) or the shell script  
- Clip length, captions, splits → **`convert_lerobot_to_vision_sft.py`**

## `launch_vision_sft.sh`

| Role | Path |
|------|------|
| Entrypoint | `scripts/launch_vision_sft.sh` |
| Hyperparams | `configs/vision_sft_edge.toml` |
| Runtime | Copies TOML into framework, sources `_sft_launcher_common.sh` |

Vision does **not** use SO101 patches; it uses the stock `vision_sft_edge`
experiment + our data/TOML.

## `launch_action_policy.sh`

Path: `scripts/launch_action_policy.sh`

Flow: `sync_patches.sh` → copy lab TOML into framework → `_sft_launcher_common.sh`.

| Role | Edit in lab |
|------|-------------|
| Hyperparams | `configs/action_policy_so101_edge.toml` |
| Dataset reader (cams, 6D, concat) | `patches/cosmos-framework/so101_lerobot_dataset.py` |
| Action mean/std | `patches/cosmos-framework/so101_stack_3cam_meanstd.json` |
| Experiment (WAM dataloader) | `patches/cosmos-framework/action_policy_so101_edge.py` |

After editing patches:

```bash
bash scripts/sync_patches.sh
```

Synced destinations inside `COSMOS_FRAMEWORK_ROOT`:

```
cosmos_framework/data/generator/action/datasets/so101_lerobot_dataset.py
cosmos_framework/data/generator/action/normalizer_stats/so101_stack_3cam_meanstd.json
cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_so101_edge.py
```

`sync_patches.sh` also ensures `domain_utils.py` has `so101_follower`,
`action_sft_dataset.py` has `get_action_so101_sft_dataset`, and `config.py`
registers the experiment.

**Rule:** change lab patch sources, then sync. Avoid hand-editing only the
framework copy (easy to lose on pull/reinstall).

## Ownership diagram

```
cosmos-edge-lab/                          ← edit here
├── scripts/prepare_vision_data.sh
├── scripts/convert_lerobot_to_vision_sft.py
├── scripts/launch_vision_sft.sh
├── scripts/launch_action_policy.sh
├── configs/*.toml
└── patches/cosmos-framework/*
         │ sync_patches.sh
         ▼
cosmos-framework/                         ← engine
└── examples/_sft_launcher_common.sh      ← torchrun train
```
