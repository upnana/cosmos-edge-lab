# Bench PC 离线 WAM 推理（wenxingnan）

> 日期：2026-08-13  
> 机器：`wenxingnan`（2×RTX 3090，真机旁 bench / 推理机）  
> 权重：HF [`upna/action_stack3cam_action_policy_edge_2000`](https://huggingface.co/upna/action_stack3cam_action_policy_edge_2000)  
> 引擎：本机 `~/cosmos-framework`（非 H100 训练机）  
> 下一步：真机闭环 → [`REAL_ROBOT_EVAL_CHECKLIST.md`](REAL_ROBOT_EVAL_CHECKLIST.md)

训练机上的 held-out WAM/FD 结果见 [`ACTION_POLICY_OFFLINE_WAM.md`](ACTION_POLICY_OFFLINE_WAM.md)。  
**本文记录：把同一份 Action export 搬到 bench PC，先离线冒烟，再上真机。**

---

## 0. 目标

在 **不接臂** 的情况下，用与训练一致的配置跑通：

```text
front|wrist concat @256 → WAM → action [T,6]（meanstd）→ denorm → 关节角（度）
```

通了之后，再按 checklist 接 SO-101 client。

---

## 1. 推理参数（与训练 / 评测对齐）

| 项 | 值 |
|----|-----|
| `domain_name` | `so101_follower` |
| `EMBODIMENT_TO_DOMAIN_ID` | **22** |
| `EMBODIMENT_TO_RAW_ACTION_DIM` | **6** |
| `view_point` | `concat_view` |
| `camera_keys` | `observation.images.front`, `observation.images.wrist` |
| 拼接 | **front \| wrist**，各 **256×256** → **256×512** |
| `action_chunk_size` | **32**（≈1.07s @30fps；真机按控制频率定） |
| `fps` | 30 |
| `image_size` / `resolution` | 256 |
| `model_mode` | `wam` |
| 动作空间 | 6D 关节位置（度）；gripper 约 0–72 |
| 归一化 | **meanstd**；stats：`so101_stack_3cam_meanstd.json` |

离线 sample JSON 结构：

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

`vision_path` 相对 JSON 目录；内容为已拼好的 front|wrist 视频。

---

## 2. Bench PC 上已做的准备

路径均相对 `~/cosmos-framework`（本机实际布局）。

### 2.1 权重与 VAE

```bash
# HF export（含 vision_encoder / tokenizer）
outputs/export/action_stack3cam_action_policy_edge_2000/

# Wan2.2 VAE（export 里曾写死 /home/july/...，已改成本机路径）
examples/checkpoints/wan22_vae/Wan2.2_VAE.pth
```

`config.json` 中：

```text
tokenizer.vae_path = /home/rxn/cosmos-framework/examples/checkpoints/wan22_vae/Wan2.2_VAE.pth
```

### 2.2 引擎补丁（本机 `cosmos-framework`）

`cosmos_framework/data/generator/action/domain_utils.py`：

```python
"so101_follower": 22,  # EMBODIMENT_TO_DOMAIN_ID（与 behavior1k 同号，跟训练一致）
"so101_follower": 6,   # EMBODIMENT_TO_RAW_ACTION_DIM
```

归一化 stats（由本机数据集 `meta/stats.json` 写入；若与训练机文件有出入，以训练机为准覆盖）：

```text
cosmos_framework/data/generator/action/normalizer_stats/so101_stack_3cam_meanstd.json
```

### 2.3 离线输入

```text
inputs/omni/action_policy_so101_offline.json
inputs/omni/so101_offline/concat_obs.mp4   # front|wrist @256, 33 frames, 30fps
```

观测来自本机数据集：

```text
/home/rxn/datasets/stack_3blocks_white_blue_black_3cam
```

---

## 3. 环境注意（宿主机，非 NGC）

| 问题 | 处理 |
|------|------|
| Guardrail 拉 `Cosmos-Guardrail1` / `HF_HUB_OFFLINE` | 加 `--no-guardrails` |
| `torchcodec` 找不到 `libnppicc.so.13` | **不要** `LD_LIBRARY_PATH=''`；改为指向 venv 内 cu13：见下方 |
| NGC 容器里的空 `LD_LIBRARY_PATH` | 仅容器需要；本机宿主机不要清空 |

```bash
cd ~/cosmos-framework
source .venv/bin/activate
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.13/site-packages/nvidia/cu13/lib"
```

---

## 4. 离线 WAM 命令

```bash
cd ~/cosmos-framework
source .venv/bin/activate
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.13/site-packages/nvidia/cu13/lib"

python -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  --no-guardrails \
  -i inputs/omni/action_policy_so101_offline.json \
  -o outputs/action_policy_so101_offline \
  --checkpoint-path outputs/export/action_stack3cam_action_policy_edge_2000 \
  --seed=0
```

预期产物：

| 文件 | 含义 |
|------|------|
| `.../sample_outputs.json` | `action`：`[T,6]`，**meanstd 空间** |
| `.../vision.mp4` | WAM 未来视觉 rollout（辅助看，不直接下发） |

### 反归一化（真机执行前必做）

```text
degrees = action * std + mean
```

可用引擎内：

```python
from cosmos_framework.data.generator.action.action_normalization import (
    denormalize_action,
    load_action_stats,
)
stats = load_action_stats(
    "cosmos_framework/data/generator/action/normalizer_stats/so101_stack_3cam_meanstd.json"
)
# degrees = denormalize_action(action_tensor, "meanstd", stats_as_tensors)
```

---

## 5. 与「已跑通的 Bridge 冒烟」的区别

此前用 `inputs/omni/action_policy_robot.json`（`bridge_orig_lerobot`）只证明 **引擎 + GPU + export 能跑**。  
**不能**把那次 `action` 接到 SO-101。

本步才是 SO-101 配置：`domain=22`、front|wrist@256、chunk=32、meanstd denorm。

---

## 6. 完成后勾选

- [x] HF export 落到 bench PC  
- [x] VAE 本机路径可用  
- [x] `so101_follower` 注册（id=22, dim=6）  
- [x] meanstd stats 文件就位  
- [x] front\|wrist concat 离线输入准备好  
- [ ] 本命令离线 WAM 跑通（看 `sample_outputs.json`）  
- [ ] denorm 后关节角量级与遥操作接近  
- [ ] 进入 [`REAL_ROBOT_EVAL_CHECKLIST.md`](REAL_ROBOT_EVAL_CHECKLIST.md) §4–§5  

---

## 7. 相关链接

- Lab 仓库：<https://github.com/upnana/cosmos-edge-lab>  
- 权重：<https://huggingface.co/upna/action_stack3cam_action_policy_edge_2000>  
- 数据集：本机 `/home/rxn/datasets/stack_3blocks_white_blue_black_3cam`  
- 真机清单：[`REAL_ROBOT_EVAL_CHECKLIST.md`](REAL_ROBOT_EVAL_CHECKLIST.md)  
- 训练机离线 metrics：[`ACTION_POLICY_OFFLINE_WAM.md`](ACTION_POLICY_OFFLINE_WAM.md)  
