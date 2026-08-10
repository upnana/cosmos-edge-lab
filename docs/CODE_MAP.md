# 代码在哪、该改哪里

Lab 脚本是**入口**。真正逻辑在 lab 的 Python / TOML / patches；
`cosmos-framework` 是运行时引擎。

## `prepare_vision_data.sh`

路径：`scripts/prepare_vision_data.sh`

| 角色 | 路径 |
|------|------|
| 入口 | `scripts/prepare_vision_data.sh` |
| 转换器（改这里） | `scripts/convert_lerobot_to_vision_sft.py` |
| 上游参考 | `cosmos-framework/tools/lerobot3cam_to_vision_sft.py` |
| 默认输入 | `SO101_ROOT` → stack3cam LeRobot 根目录 |
| 默认输出 | `data/processed/stack3cam_vision_sft/` |

修改提示：

- 相机 / 路径 → 环境变量（`CAMERA`、`SO101_ROOT`）或 shell 脚本  
- clip 长度、caption、划分 → **`convert_lerobot_to_vision_sft.py`**

## `launch_vision_sft.sh`

| 角色 | 路径 |
|------|------|
| 入口 | `scripts/launch_vision_sft.sh` |
| 超参 | `configs/vision_sft_edge.toml` |
| 运行时 | 把 TOML 复制进 framework，再 source `_sft_launcher_common.sh` |

Vision **不**依赖 SO101 patches；用官方 `vision_sft_edge` experiment + 我们的数据/TOML。

## `launch_action_policy.sh`

路径：`scripts/launch_action_policy.sh`

流程：`sync_patches.sh` → 把 lab TOML 拷进 framework → `_sft_launcher_common.sh`。

| 角色 | 在 lab 里改 |
|------|-------------|
| 超参 | `configs/action_policy_so101_edge.toml` |
| 数据集读取（相机、6D、拼接） | `patches/cosmos-framework/so101_lerobot_dataset.py` |
| Action mean/std | `patches/cosmos-framework/so101_stack_3cam_meanstd.json` |
| Experiment（WAM dataloader） | `patches/cosmos-framework/action_policy_so101_edge.py` |

改完 patches 后：

```bash
bash scripts/sync_patches.sh
```

同步到 `COSMOS_FRAMEWORK_ROOT` 内的目标：

```
cosmos_framework/data/generator/action/datasets/so101_lerobot_dataset.py
cosmos_framework/data/generator/action/normalizer_stats/so101_stack_3cam_meanstd.json
cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_so101_edge.py
```

`sync_patches.sh` 还会确保 `domain_utils.py` 有 `so101_follower`、
`action_sft_dataset.py` 有 `get_action_so101_sft_dataset`，以及 `config.py` 注册该 experiment。

**原则：** 改 lab 里的 patch 源，再 sync。不要只手改 framework 副本（一 pull/重装就丢）。

## 所有权示意

```
cosmos-edge-lab/                          ← 在这里改
├── scripts/prepare_vision_data.sh
├── scripts/convert_lerobot_to_vision_sft.py
├── scripts/launch_vision_sft.sh
├── scripts/launch_action_policy.sh
├── configs/*.toml
└── patches/cosmos-framework/*
         │ sync_patches.sh
         ▼
cosmos-framework/                         ← 引擎
└── examples/_sft_launcher_common.sh      ← torchrun train
```
