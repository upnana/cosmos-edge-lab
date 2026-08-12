# Cosmos-Edge Lab — 结果汇总

> 更新：2026-08-12  
> 底座：Cosmos3-Edge；数据：SO-101 `stack_3blocks_white_blue_black_3cam`

本文汇总本仓库已落盘的 **Vision I2V** 与 **Action-Policy WAM/FD** 证据。  
不宣称对 π0 / 官方 RoboLab 的 SOTA。

真机 closed-loop SR（vs π0）：驱动已接好（`scripts/so101_rollout_driver.py`），但 **本 H100 无臂/相机**，SR 数字需在 bench PC 上 `FORCE_REAL=1` 跑完后填入。协议见 [`REAL_ROBOT_EVAL_CHECKLIST.md`](REAL_ROBOT_EVAL_CHECKLIST.md)。

---

## 1. Vision SFT → I2V（世界动力学）

训练：`stack3cam_vision_sft_edge_500` → 导出 `outputs/export/vision_stack3cam_vision_sft_edge_500`。

| 集合 | Episode | 时长 | 并排资产 |
|------|---------|------|----------|
| held-out | 111 | ~2s | [`assets/vision_heldout_ep111/`](assets/vision_heldout_ep111/) |
| held-out | 111 | ~8s | [`assets/vision_heldout_ep111_8s/`](assets/vision_heldout_ep111_8s/) |
| train 对照 | 119 | ~8s | [`assets/vision_ep119_8s/`](assets/vision_ep119_8s/) |
| 更多 val | 3 / 22 / 47 / 69 / 99 | ~8s | [`assets/vision_more_examples/`](assets/vision_more_examples/) |

说明：pred 与 GT **必须同 episode**；此前混用 ep111 pred + ep119 GT 会造成「对不齐」假象。

文档：[`VISION_500_HELDOUT_I2V.md`](VISION_500_HELDOUT_I2V.md)

---

## 2. Action-Policy → WAM + Forward Dynamics

训练：`stack3cam_action_policy_edge_2000`（`iter_000002000`）  
设定：front\|wrist concat @256，fps=30，auto-motion 窗口。

### 2.1 ep111（单集，多时长）

详见 [`ACTION_POLICY_OFFLINE_WAM.md`](ACTION_POLICY_OFFLINE_WAM.md) / [`assets/action_wam_ep111/`](assets/action_wam_ep111/)

| 时长 | Action L1 (raw°) | WAM PSNR | FD PSNR |
|------|------------------|----------|---------|
| ~1.1s (chunk=32) | **7.62** | **16.9** | **17.3** |
| ~3.2s (chunk=96) | 17.7 | 16.9 | 16.8 |
| ~8.0s (chunk=240) | 15.2 | 16.0 | 16.1 |

### 2.2 更多 held-out（3 / 22 / 47 / 69 / 99）

详见 [`ACTION_POLICY_WAM_MORE.md`](ACTION_POLICY_WAM_MORE.md) / [`assets/action_wam_more/`](assets/action_wam_more/)

| 时长 | Mean Action L1 | Mean WAM | Mean FD |
|------|----------------|----------|---------|
| ~1.1s | 14.4° | 17.8 dB | 17.9 dB |
| ~3.2s | 22.2° | 16.9 dB | 17.1 dB |
| ~8.0s | 20.2° | 15.1 dB | 15.1 dB |

**解读（离线 open-loop）：**

- **Action L1**：关节角平均绝对误差（度）；越小越好；训练 chunk=32 上最好，加长后变粗属预期。  
- **WAM PSNR**：同时预测动作+视频，相对 GT 的视频相似度。  
- **FD PSNR**：给定 GT 动作只预测视频（更偏世界模型）。  
- 与官方 Cosmos action-cond 公开 PSNR（约 21–25 dB）**不可直接对标**（数据/模型/机臂不同）。  
- **真机 closed-loop SR** 仍待做；官方 Edge-Policy 主报 **RoboLab 仿真 SR ~22.9%**，非本 SO-101 任务。

---

## 3. 指标含义速查

| 指标 | 验什么 |
|------|--------|
| Vision I2V pred vs GT | 纯视觉世界动力学 |
| Action L1 | 会不会动手（open-loop） |
| WAM PSNR | 边动手边想象画面 |
| FD PSNR | 用对的动作推画面 |
| 真机 SR（未做） | 叠块成不成功；对标 π0 |

---

## 4. 复现入口

```bash
source scripts/env.sh

# Vision held-out I2V（示例）
# 见 scripts/run_i2v_heldout_8s.sh / docs/VISION_500_HELDOUT_I2V.md

# Action WAM/FD
EPISODES=111 bash scripts/run_action_wam_heldout.sh
EPISODES="3 22 47 69 99" CHUNK_LENGTH=32 EVAL_ROOT=$LAB_ROOT/outputs/eval_action_wam_more \
  bash scripts/run_action_wam_heldout.sh
```

训练笔记：[`notes/action_policy_2000_showcase.md`](../notes/action_policy_2000_showcase.md)  
架构 / Vision vs Action：[`VISION_VS_ACTION_SFT.md`](VISION_VS_ACTION_SFT.md)
