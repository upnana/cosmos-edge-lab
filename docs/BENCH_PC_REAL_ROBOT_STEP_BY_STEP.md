# Bench PC 真机闭环逐步手册（wenxingnan）

> 机器：`wenxingnan`（2×RTX 3090）  
> 任务：`stack the blocks from bottom to top white then blue then black`  
> 权重：Hugging Face `upna/action_stack3cam_action_policy_edge_2000`  
> 仓库：[cosmos-edge-lab](https://github.com/upnana/cosmos-edge-lab)

本文按 **真实排障顺序** 写：从 HF 下载 → 环境 → `domain_utils` / 相机 → dry-run → 冷/热启动 → 踩过的坑 → denorm / 动作变换 → eval 视频含义。

相关：[`BENCH_PC_OFFLINE_INFERENCE.md`](BENCH_PC_OFFLINE_INFERENCE.md) · [`REAL_ROBOT_EVAL_CHECKLIST.md`](REAL_ROBOT_EVAL_CHECKLIST.md) · [`RESOLUTION_CONFIG.md`](RESOLUTION_CONFIG.md)

---

## 总览

```text
HF export
  → 注册 so101_follower (id=22, dim=6) + meanstd
  → front|wrist @256 concat
  → DRY_RUN（Mock 臂）
  → FORCE_REAL 冷启动（每 chunk 重启 inference）
  → FORCE_REAL 热启动（模型常驻）+ 录 eval 视频
  → denorm：meanstd → 关节角度(°) → SO-101 下发
```

| 产物 | 是什么 |
|------|--------|
| `*_chunk*_actions.json` | 模型输出的 **动作**（度） |
| `videos/*_eval.mp4` | 真机相机录的 **执行画面**（不是模型生成视频） |
| `cosmos_steps/.../vision.mp4` | WAM 推理时可选写出的预测视觉（离线路径更常见） |

---

## Step 1 — 从 Hugging Face 下载 Action-Policy

```bash
export COSMOS_FRAMEWORK_ROOT=/home/rxn/cosmos-framework   # 按本机改
export EXPORT=$COSMOS_FRAMEWORK_ROOT/outputs/export/action_stack3cam_action_policy_edge_2000

# 需要 huggingface-cli / hf
hf download upna/action_stack3cam_action_policy_edge_2000 \
  --local-dir "$EXPORT"
```

同时准备 Wan2.2 VAE（export 的 `config.json` 里 `vae_path` 必须指向本机文件）：

```bash
# 示例路径
ls $COSMOS_FRAMEWORK_ROOT/examples/checkpoints/wan22_vae/Wan2.2_VAE.pth
# 若 config 仍写 /home/july/... ，改成上面的本机路径
```

Torch / torchcodec 依赖 CUDA 库时，bench 上常见：

```bash
export LD_LIBRARY_PATH=$COSMOS_FRAMEWORK_ROOT/.venv/lib/python3.13/site-packages/nvidia/cu13/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
```

（空 `LD_LIBRARY_PATH` 在 NGC 容器里修 torch 导入；本机 venv 则常要补 nvidia cu13 lib。）

---

## Step 2 — 环境变量与双 Python

真机驱动需要 **两套 Python**：

| 用途 | 变量 | 本机示例 |
|------|------|----------|
| 连臂 / 相机（LeRobot） | `LEROBOT_PYTHON` | `/home/rxn/miniconda3/envs/lerobot_alohamini/bin/python` |
| Cosmos 推理 | `FRAMEWORK_PYTHON` | `$COSMOS_FRAMEWORK_ROOT/.venv/bin/python` |
| LeRobot 源码 | `LEROBOT_SRC` | `/home/rxn/lerobot_alohamini/src`（含 `so_follower`） |

```bash
cd /home/rxn/cosmos-edge-lab
export COSMOS_FRAMEWORK_ROOT=/home/rxn/cosmos-framework
export EXPORT=$COSMOS_FRAMEWORK_ROOT/outputs/export/action_stack3cam_action_policy_edge_2000
export OUT_ROOT=$COSMOS_FRAMEWORK_ROOT/outputs/rollout_real
export FRAMEWORK_PYTHON=$COSMOS_FRAMEWORK_ROOT/.venv/bin/python
export LEROBOT_PYTHON=/home/rxn/miniconda3/envs/lerobot_alohamini/bin/python
export LEROBOT_SRC=/home/rxn/lerobot_alohamini/src
export LD_LIBRARY_PATH=$COSMOS_FRAMEWORK_ROOT/.venv/lib/python3.13/site-packages/nvidia/cu13/lib
source scripts/env.sh
```

> 不要用「只有 stock `lerobot`、没有 `draccus` / `so_follower`」的环境跑 driver。

把 SO101 适配补丁同步进 framework（若尚未）：

```bash
bash scripts/sync_patches.sh
```

---

## Step 3 — `domain_utils.py`：domain id 与动作维

Action 训练/推理都依赖：

| 键 | 值 | 含义 |
|----|-----|------|
| `so101_follower` domain id | **22** | `EMBODIMENT_TO_DOMAIN_ID` |
| raw action dim | **6** | 肩×2 + 肘 + 腕×2 + 夹爪 |

位置（framework）：

```text
cosmos_framework/data/generator/action/domain_utils.py
```

Lab 的 `scripts/sync_patches.sh` 会插入这两行。本机也可手改后确认：

```python
assert get_domain_id("so101_follower") == 22
assert get_action_dim("so101_follower") == 6
```

归一化统计（meanstd）：

```text
cosmos_framework/data/generator/action/normalizer_stats/so101_stack_3cam_meanstd.json
```

`rollout_common.DEFAULT_STATS` 通过 `COSMOS_FRAMEWORK_ROOT` 解析，勿写死 `/home/july/...`。

---

## Step 4 — 相机与串口配置

### 训练 / 策略约定（Action-policy）

| 项 | 值 |
|----|-----|
| 相机 | **front + wrist only**（不用 side） |
| 每路分辨率 | **256×256** → 横拼 **512×256** |
| `viewpoint` | `concat_view` |
| `action_chunk_size` | 训练 32；真机默认 chunk=32，执行前 `EXECUTE_STEPS=16` 再 replan |
| 原始采集 | front/wrist 640×480（进模前 resize） |

### wenxingnan 实测映射

| 设备 | 角色 |
|------|------|
| `/dev/ttyACM2` | SO-101 follower（拔插验证） |
| `/dev/video0` | front（俯视桌面） |
| `/dev/video2` | wrist（夹爪近景） |
| `/dev/video4` | side 800×480 → **Action 不用** |

```bash
export SO101_PORT=/dev/ttyACM2
export CAM_FRONT=0
export CAM_WRIST=2
export CAM_SIDE=   # 必须空！不要默认成 2（会与 wrist 抢设备）
```

预览：

```bash
# lerobot 环境
lerobot-find-cameras opencv
```

---

## Step 5 — Dry-run（不上臂）

验：拼图 →（可选）Cosmos → denorm → MockRobot。

```bash
cd ~/cosmos-edge-lab
source scripts/env.sh   # 环境变量同上

# 1) 零动作链路
DRY_RUN=1 N_TRIALS=1 POLICY=zeros \
  OUT_ROOT="$OUT_ROOT" bash scripts/rollout_action_real.sh

# 2) Cosmos（会调 FRAMEWORK_PYTHON；现默认热启动）
DRY_RUN=1 N_TRIALS=1 POLICY=cosmos \
  EXPORT="$EXPORT" OUT_ROOT="$OUT_ROOT" \
  COSMOS_WARM=1 \
  bash scripts/rollout_action_real.sh
```

成功标志：写出 `OUT_ROOT/zeros_*` 或 `cosmos_*`，`sr: null`（mock）正常。

---

## Step 6 — 冷启动真机（历史路径）

**冷启动** = 每个 chunk 都 `python -m cosmos_framework.scripts.inference ...` 再起进程：读盘 → 建模 → 上 GPU → 推理 → 退出。

```bash
FORCE_REAL=1 INTERACTIVE=0 N_TRIALS=1 POLICY=cosmos \
  COSMOS_WARM=0 \
  TIMEOUT_S=600 MAX_CHUNKS=20 \
  SO101_PORT=/dev/ttyACM2 CAM_FRONT=0 CAM_WRIST=2 CAM_SIDE= \
  EXPORT="$EXPORT" OUT_ROOT="$OUT_ROOT" \
  bash scripts/rollout_action_real.sh
```

现象：每段动作前空等很久（多为重新 load，而非采样本身）。

---

## Step 7 — 热启动真机（推荐）

**热启动** = `scripts/cosmos_wam_worker.py` 常驻：权重只加载一次，之后 stdin JSONL 请求推理。

```bash
FORCE_REAL=1 INTERACTIVE=0 N_TRIALS=1 POLICY=cosmos \
  COSMOS_WARM=1 \
  TIMEOUT_S=600 MAX_CHUNKS=20 EXECUTE_STEPS=16 \
  SO101_PORT=/dev/ttyACM2 CAM_FRONT=0 CAM_WRIST=2 CAM_SIDE= \
  LEROBOT_PYTHON="$LEROBOT_PYTHON" LEROBOT_SRC="$LEROBOT_SRC" \
  FRAMEWORK_PYTHON="$FRAMEWORK_PYTHON" \
  EXPORT="$EXPORT" OUT_ROOT="$OUT_ROOT" \
  bash scripts/rollout_action_real.sh 2>&1 | tee /tmp/force_real_warm.log
```

成功标志：

1. `[policy:cosmos] warm worker ready`（第一次慢）  
2. 多次 `[policy:cosmos:warm] infer id=N`，后续约 **1–2 s/chunk**  
3. `[video] wrote .../*_eval.mp4`  
4. 结束行：`NOTE: Cosmos used warm resident worker`

示例一次成功 run（20 chunks，~66s，340 帧 eval）：

```text
outputs/rollout_real/cosmos_20260814_112525_3523384/
```

仓库内展示用（色通道已校正；**打开下方 Markdown 页即可播放**，勿点裸 mp4 blob）：

- 预览页：[`assets/real_robot_eval/README.md`](assets/real_robot_eval/README.md)
- GIF：[`assets/real_robot_eval/cosmos_action_warm_t001_eval.gif`](assets/real_robot_eval/cosmos_action_warm_t001_eval.gif)
- MP4（H.264）：[`assets/real_robot_eval/cosmos_action_warm_t001_eval_web.mp4`](assets/real_robot_eval/cosmos_action_warm_t001_eval_web.mp4)

![warm eval](assets/real_robot_eval/cosmos_action_warm_t001_eval.gif)

<video src="assets/real_robot_eval/cosmos_action_warm_t001_eval_web.mp4" controls width="720" preload="metadata">
  <a href="assets/real_robot_eval/cosmos_action_warm_t001_eval_web.mp4">cosmos_action_warm_t001_eval_web.mp4</a>
</video>

---

## Step 8 — 踩过的坑（Bug 清单）

| # | 现象 | 原因 | 处理 |
|---|------|------|------|
| 1 | `ModuleNotFoundError: draccus` | 用了 framework `.venv` 跑 driver | `LEROBOT_PYTHON=.../lerobot_alohamini` |
| 2 | `No module named lerobot.robots.so_follower` | stock `lerobot` 只有 `so101_follower` | 用 alohamini 源码；driver 已 dual-import |
| 3 | exit **134** / `Failed to open OpenCVCamera(2)` | 默认 `CAM_SIDE=2` 与 `CAM_WRIST=2` 抢同一路 | `CAM_SIDE=` 留空 |
| 4 | 必须在 lab 目录跑 | 在 `cosmos-framework` 下找不到 `scripts/env.sh` | `cd ~/cosmos-edge-lab` |
| 5 | eval 里积木变「红」 | RGB 误当 BGR 写入 `VideoWriter` | 写入前 `COLOR_RGB2BGR`；旧视频可用 `*_colorfixed.mp4` |
| 6 | Cursor「bash exit 1/134」 | 只是终端退出码，看 Traceback / tee 日志 | `tee /tmp/*.log` |
| 7 | `max_relative_target` clamp 警告 | 单步 >15° 被安全夹紧 | 正常保护，不是崩溃 |

---

## Step 9 — Denorm 与动作变换（eval 时）

### 9.1 模型输出

WAM 样本 JSON 里 `domain_name=so101_follower`，模型吐出的是 **meanstd 空间** 的 action chunk（长度≈`action_chunk_size`，宽=6）。

### 9.2 Denorm（度）

```text
degrees = action_norm * std + mean
```

`mean` / `std` 来自 `so101_stack_3cam_meanstd.json`（与数据集 `meta/stats.json` 一致）。

Driver 内：`extract_wam_action` → `denorm_actions` → `*_actions.json`（已是度）。

### 9.3 下发 SO-101

```text
关节顺序：
  shoulder_pan, shoulder_lift, elbow_flex,
  wrist_flex, wrist_roll, gripper

每步：send_action({f"{j}.pos": deg})
控制频率：CONTROL_HZ=30（默认）
每 chunk 只执行前 EXECUTE_STEPS=16 步，再采图 replan
```

`max_relative_target=15`：相对当前位姿每关节最多约 15°，过大则 clamp。

### 9.4 Eval 视频 ≠ action

- **action**：json 里的角度序列  
- **eval.mp4**：执行过程中相机帧（front\|wrist @256 concat）  
- 看行为看视频；对数值开 json / 画曲线

---

## Step 10 — 闭环逻辑（一句话）

```text
home → [ 采 front+wrist → 256 concat →（热）WAM chunk=32
       → denorm → 执行 16 步并录像 → 重复至 MAX_CHUNKS/timeout ]
```

`INTERACTIVE=1` 时每 chunk 后问 `c/y/n/q` 标成功；`0` 则自动连跑，`success=null`（需事后标 SR）。

---

## 命令速查（复制用）

```bash
cd ~/cosmos-edge-lab
export COSMOS_FRAMEWORK_ROOT=/home/rxn/cosmos-framework
export EXPORT=$COSMOS_FRAMEWORK_ROOT/outputs/export/action_stack3cam_action_policy_edge_2000
export OUT_ROOT=$COSMOS_FRAMEWORK_ROOT/outputs/rollout_real
export FRAMEWORK_PYTHON=$COSMOS_FRAMEWORK_ROOT/.venv/bin/python
export LEROBOT_PYTHON=/home/rxn/miniconda3/envs/lerobot_alohamini/bin/python
export LEROBOT_SRC=/home/rxn/lerobot_alohamini/src
export LD_LIBRARY_PATH=$COSMOS_FRAMEWORK_ROOT/.venv/lib/python3.13/site-packages/nvidia/cu13/lib
export SO101_PORT=/dev/ttyACM2 CAM_FRONT=0 CAM_WRIST=2 CAM_SIDE=
source scripts/env.sh

FORCE_REAL=1 INTERACTIVE=0 N_TRIALS=1 POLICY=cosmos COSMOS_WARM=1 \
  TIMEOUT_S=600 MAX_CHUNKS=20 \
  bash scripts/rollout_action_real.sh 2>&1 | tee /tmp/force_real_warm.log
```

---

## 脚本索引

| 文件 | 作用 |
|------|------|
| `scripts/rollout_action_real.sh` | 环境包装、`COSMOS_WARM`、相机口 |
| `scripts/so101_rollout_driver.py` | 闭环、录像、denorm、双 import |
| `scripts/cosmos_wam_worker.py` | 热启动常驻推理 |
| `scripts/rollout_common.py` | meanstd / TrialLogger |

---

## 安全提醒

急停 = 电源或 USB（`/dev/ttyACM2`）可立刻断开；先 `N_TRIALS=1`；人站在能伸手断电的位置。
