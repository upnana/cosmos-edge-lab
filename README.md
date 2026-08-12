# cosmos-edge-lab

**基于 Cosmos3-Edge 的个人 World-Action-Model（WAM）实验仓库** —— 不是别人配方的简单拷贝。

所有者：[upnana](https://github.com/upnana)  
远程仓库：https://github.com/upnana/cosmos-edge-lab

本仓库负责：

- 实验问题、假设与运行记录
- SO-101 / stack-3cam 数据适配与归一化
- 面向 **1×H100 + Cosmos3-Edge** 调好的 Vision SFT / Action-policy TOML
- 启动 / 转换脚本

**不**包含训练引擎核心。训练仍通过本地
[NVIDIA cosmos-framework](https://github.com/NVIDIA/cosmos-framework)
（默认：`../cosmos-framework`）运行。Lab 补丁用 `scripts/sync_patches.sh` 同步进去。

```
cosmos-edge-lab/          ← 你的实验科学
../cosmos-framework/      ← 引擎（外部依赖）
```

## 首个实验：`stack3cam_wam`

数据：SO-101 遥操作 `stack_3blocks_white_blue_black_3cam`（LeRobot v3，3 相机，6D 关节）。

同一底座（**Cosmos3-Edge**）上两条线：

1. **Vision SFT** —— 前视相机世界 / 视频动力学  
2. **Action-policy SFT** —— WAM 风格动作块（前视+腕部）

### 文档

| 文档 | 主题 |
|------|------|
| [`docs/RESULTS_SUMMARY.md`](docs/RESULTS_SUMMARY.md) | **结果汇总**（Vision I2V + Action WAM/FD） |
| [`docs/ACTION_POLICY_OFFLINE_WAM.md`](docs/ACTION_POLICY_OFFLINE_WAM.md) | Action-Policy 离线 WAM/FD（ep111） |
| [`docs/ACTION_POLICY_WAM_MORE.md`](docs/ACTION_POLICY_WAM_MORE.md) | 更多 held-out WAM/FD（1s/3s/8s） |
| [`docs/VISION_500_HELDOUT_I2V.md`](docs/VISION_500_HELDOUT_I2V.md) | Vision 500 held-out I2V |
| [`docs/EXPERIMENT_DESIGN.md`](docs/EXPERIMENT_DESIGN.md) | 假设与成功标准 |
| [`docs/COSMOS_ARCHITECTURE.md`](docs/COSMOS_ARCHITECTURE.md) | Cosmos3 MoT / Edge 网络结构 |
| [`docs/MOT_ATTENTION_AND_VISION_SFT.md`](docs/MOT_ATTENTION_AND_VISION_SFT.md) | MoT / und·gen / `*_moe_gen` / two_way / Vision 训哪些 |
| [`docs/VISION_VS_ACTION_SFT.md`](docs/VISION_VS_ACTION_SFT.md) | Vision 与 Action 如何训练 |
| [`docs/CODE_MAP.md`](docs/CODE_MAP.md) | 代码位置 / 改哪里 |
| [`docs/FULL_SFT_VS_LORA.md`](docs/FULL_SFT_VS_LORA.md) | 模块级 Full SFT vs LoRA（~120 ep） |
| [`docs/VISION_SMOKE_EXPORT_I2V.md`](docs/VISION_SMOKE_EXPORT_I2V.md) | Vision smoke→导出→I2V（打开文档页即可内嵌预览 pred/GT） |
| [`docs/VISION_500_HELDOUT_I2V.md`](docs/VISION_500_HELDOUT_I2V.md) | Vision 500iter→导出→held-out ep111 I2V（含 pred/GT 并排） |
| [`docs/ACTION_POLICY_OFFLINE_WAM.md`](docs/ACTION_POLICY_OFFLINE_WAM.md) | Action-Policy 2000→WAM/FD 离线评测（ep111，含 1s/3s 预览） |
| [`notes/analysis_2026-08-10.md`](notes/analysis_2026-08-10.md) | 会话纪要 |

配方目录：[`experiments/stack3cam_wam/`](experiments/stack3cam_wam/)

对照基线（同任务、不同模型族）：π0 3-cam on LeRobot（`upna/pi0_stack_white_blue_black_3cam_*`）。

## 环境准备

```bash
# 1) 引擎（一次性）
#    另处 clone + uv sync cosmos-framework，例如 ../cosmos-framework

# 2) Lab 环境
cd /path/to/cosmos-edge-lab
export COSMOS_FRAMEWORK_ROOT=/path/to/cosmos-framework   # 若已是 ../cosmos-framework 可省略
source scripts/env.sh

# 3) 把 SO101 适配器注入引擎
bash scripts/sync_patches.sh
```

机器上还需一次性准备：

- Wan2.2 VAE：`$COSMOS_FRAMEWORK_ROOT/examples/checkpoints/wan22_vae/Wan2.2_VAE.pth`
- HF 权重：`/home/july/models/Cosmos3-Edge`（可用 `EDGE_HF` 覆盖）
- 转 DCP：`bash scripts/prepare_edge_dcp.sh`

## 运行手册

```bash
source scripts/env.sh

# Vision 数据（前视 clip → JSONL）
bash scripts/prepare_vision_data.sh

# Vision smoke（GPU 空闲时）
export EXTRA_TAIL_OVERRIDES="trainer.max_iter=10 checkpoint.save_iter=10"
bash scripts/launch_vision_sft.sh

# Action smoke
export EXTRA_TAIL_OVERRIDES="trainer.max_iter=10 checkpoint.save_iter=10 dataloader_train.max_samples_per_batch=2"
bash scripts/launch_action_policy.sh

# 更长训练：取消 EXTRA_TAIL_OVERRIDES 再跑
```

产物在 `outputs/`（已 gitignore）；检查点在 `checkpoints/`。

## 目录结构

```
configs/                 # 你拥有的 TOML 配方
scripts/                 # 转换 / 同步 / 启动
patches/cosmos-framework # SO101 数据集 + action experiment 源码
experiments/stack3cam_wam
docs/                    # 设计文档
notes/                   # 工作假设 / 对比记录
data/ processed/         # 仅本机
```

## 许可说明

`patches/` 下保留的上游 NVIDIA 文件及部分配置仍带 SPDX 头（OpenMDW）。
本仓库的实验叙述与适配器属于个人 lab；再分发时请遵守上游许可。
