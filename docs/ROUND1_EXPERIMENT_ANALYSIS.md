# 第一轮实验分析（Vision SFT + Action-Policy）

> 日期：2026-08-16  
> 底座：Cosmos3-Edge @ 1×H100  
> 数据：SO-101 `stack_3blocks_white_blue_black_3cam`（120 ep，~79k frames，fps=30）  
> 结论导向：**离线有信号；Action 训练量按 epoch 明显偏少；真机 SR 尚未在本机完成。**

---

## 1. 本轮做了什么

| 轨 | 训练 | 评测 | 产物 |
|----|------|------|------|
| Vision SFT | `stack3cam_vision_sft_edge_500`，**500 iter** | held-out I2V（ep111 等） | `outputs/export/vision_*`，`docs/assets/vision_*` |
| Action-Policy | `stack3cam_action_policy_edge_2000`，**2000 iter** | 离线 WAM + Forward Dynamics | `outputs/export/action_stack3cam_action_policy_edge_2000`，`docs/assets/action_wam_*` |
| 对照 | π0 3-cam 已训到 **80k** | 真机 SR 协议已写，闭环驱动已接 | ckpt 在 `lerobot_alohamini/.../080000` |

详表与视频资产见 [`RESULTS_SUMMARY.md`](RESULTS_SUMMARY.md)。

---

## 2. 训练量：steps ↔ epochs（重点）

### 2.1 实际跑的是 steps，不是 epochs

Cosmos SFT 以 `trainer.max_iter` 停训，**没有**按 epoch 调度。

| 项 | Action-Policy 本轮 |
|----|-------------------|
| `max_iter` | **2000** |
| ckpt | `iter_000002000` |
| `max_samples_per_batch` | **4** |
| `chunk_length` | 32 |
| `sample_stride` | 2 |
| 约 windows / epoch | ~**37797**（按 ep 长度、stride=2） |

换算：

\[
\text{steps/epoch} \approx \frac{N_{\text{windows}}}{4} \approx 9450
\qquad
\text{epochs} \approx \frac{\text{steps}\times 4}{N_{\text{windows}}}
\]

| 解读 | 数值 |
|------|------|
| **本轮 2000 steps** | ≈ **0.21 epoch**（不到一整轮数据） |
| 1 epoch | ≈ **9450 steps** |
| 10 epochs | ≈ **9.5 万 steps** |
| 同任务 π0 | **80k steps** |

### 2.2 Vision 侧

Vision SFT：**500 iter**（smoke/showcase 档）。分辨率 **640×480**，前视，fps=30。  
同样是短程 iter 预算，目标是验证 I2V / 世界动力学通路，不是刷满 epoch。

### 2.3 判断：**按 epoch 看，Action 训练偏少**

- 离线 WAM/FD **已有可测信号**（动作 L1、PSNR 非随机），说明不是「完全没学」。
- 但 **0.2 epoch vs π0 80k**，不宜把本轮 Action ckpt 当成最终真机对照模型。
- 建议下一档：**~10k–20k steps（约 1–2 epoch）**，再考虑往 π0 量级靠拢后比 closed-loop SR。

---

## 3. 离线证据（摘要）

### Vision I2V

- held-out / 更多 val 的 pred–GT 并排已落盘（须 **同 episode** 对齐）。
- 文档：[`VISION_500_HELDOUT_I2V.md`](VISION_500_HELDOUT_I2V.md)

### Action WAM + FD

- ep111 与更多 held-out @ 1s / 3s / 8s；指标见 [`ACTION_POLICY_OFFLINE_WAM.md`](ACTION_POLICY_OFFLINE_WAM.md)、[`ACTION_POLICY_WAM_MORE.md`](ACTION_POLICY_WAM_MORE.md)。
- **不要**用离线 Action L1 / PSNR 代替真机叠块 SR。

### 与论文/官方数字

- 官方 Edge SR 主战场是 RoboLab 等 **仿真**；本仓库任务是 **SO-101 真机叠块**，数字不可直接横比。

---

## 4. 真机闭环（状态）

| 项 | 状态 |
|----|------|
| 协议 / checklist | [`REAL_ROBOT_EVAL_CHECKLIST.md`](REAL_ROBOT_EVAL_CHECKLIST.md) |
| 驱动 | `scripts/so101_rollout_driver.py` + `rollout_action_real.sh` |
| H100 训练机 | **无**臂 / 相机 → 训练与离线评测 |
| bench PC（wenxingnan，2×3090） | **热启动闭环 smoke 已通**（2026-08-14，1 trial，SR 未标） |
| 正式 SR vs π0 | **待做**（建议各 ≥20 trials） |

Action export HF：`upna/action_stack3cam_action_policy_edge_2000`  
真机 smoke 摘要见 [`RESULTS_SUMMARY.md`](RESULTS_SUMMARY.md) §0。

---

## 5. GPU 监控

自 `a3879f1` 起，训练/评测默认 `MONITOR_GPU=1`，写入 `outputs/gpu_monitor/`。  
**本轮 Vision 500 / Action 2000 训练当时未系统采样**；之后 launch 会自动记 util / 显存。见 [`GPU_MONITOR.md`](GPU_MONITOR.md)。

---

## 6. 第一轮结论（一句话）

> Vision / Action **通路跑通**，离线有定性/半定量证据；Action **2000 steps ≈ 0.2 epoch，训练量不足**；真机 **热启动 smoke 已通但 SR 未标**。与 π0 的公平对比取决于 **续训 + 标注 SR**，而不是再堆离线 PSNR。

### 建议下一轮

1. Action 续训到 ≥1 epoch（~10k steps）或更高，保留中间 ckpt。  
2. 在 bench PC 上按 checklist 跑 Cosmos vs π0 各 ≥20 trials SR。  
3. 全程挂上 `monitor_gpu.sh`，把峰值显存 / util 写进结果表。

---

## 7. 相关提交

| commit | 内容 |
|--------|------|
| `41450d8` 等 | Vision I2V + Action WAM/FD 结果汇总 |
| `2055825` | 真机评测 checklist + rollout 骨架 |
| `3bfcf5d` | SO-101 闭环驱动接线 |
| `a3879f1` | GPU 采样 |
| `6241032` | Action 推理参数说明 |
