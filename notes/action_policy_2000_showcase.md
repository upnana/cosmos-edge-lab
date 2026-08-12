# Action-Policy 2000 — 展示向训练

- 目标：真机/离线叠块展示 + 对标 π0（同 SO-101 stack3cam 任务）
- job：`stack3cam_action_policy_edge_2000`
- 配置：`configs/action_policy_so101_edge.toml`
- 启动：`scripts/run_action_policy_2000.sh`
- 日志：`outputs/logs/action_policy_2000.log`

## 训练设定

| 项 | 值 |
|----|-----|
| max_iter | 2000 |
| save_iter | 200 |
| max_samples_per_batch | 4（1×H100 稳妥；TOML 默认 8） |
| lr | 1e-4；action 头 ×5 |
| keys | moe_gen + vae 桥 + action2llm/llm2action/… |

## 训练完成后（展示清单）

1. 选 ckpt（如 `iter_000002000` 或 loss/视频最好的中间档）  
2. 离线或真机 rollout，录成功/失败短视频  
3. 同任务对比 π0 80k（`upna/pi0_stack_white_blue_black_3cam_*`）  
4. 记成功率 + 典型失败模式到本 notes / GitHub docs  

## 离线评测（世界模型 + Action）

入口：`scripts/run_action_wam_heldout.sh`（默认 held-out **ep111**，自动选最大运动窗口）。

| 步骤 | 脚本 / 产物 |
|------|-------------|
| 准备 concat obs + GT actions | `scripts/prepare_action_wam_eval_inputs.py` → `outputs/eval_action_wam/inputs/` |
| Export DCP→HF | `scripts/export_action_policy_2000.sh` → `outputs/export/action_stack3cam_action_policy_edge_2000` |
| WAM（动作+预测视频） | `outputs/eval_action_wam/wam_out/` |
| forward_dynamics（GT action→视频） | `outputs/eval_action_wam/fd_out/` |
| 打分 / pred-GT 预览 | `scripts/score_action_wam_eval.py` → `outputs/eval_action_wam/metrics.json` + `previews/` |

对齐训练：`domain=so101_follower`，`chunk=32`，`fps=30`，`image_size=256`，front\|wrist concat。  
注意：WAM 输出在 meanstd 空间；打分脚本会 denorm 后再算 raw L1/MSE。FD 条件动作用 `actions_norm.json`。

```bash
source scripts/env.sh
EPISODES=111 bash scripts/run_action_wam_heldout.sh
```

### 结果（ep111 / iter_000002000，2026-08-11）

窗口：`start_frame=645`（max-motion，mean step L2≈5.29）；idle `start=0` 不具参考价值。

| 指标 | 值 |
|------|-----|
| Action L1（raw deg，denorm） | **7.62** |
| Action MSE（raw） | 106.5 |
| Action L1（meanstd） | **0.41** |
| WAM vision PSNR vs GT | **16.9** dB |
| FD vision PSNR vs GT | **17.3** dB |

产物路径：

- WAM pred：`outputs/eval_action_wam/wam_out/wam_ep111/vision.mp4` + `sample_outputs.json`
- FD pred：`outputs/eval_action_wam/fd_out/fd_ep111/vision.mp4`
- 并排预览：`outputs/eval_action_wam/previews/ep111_gt_wam_fd.mp4`
- 完整 metrics：`outputs/eval_action_wam/metrics.json`

解读（离线 open-loop）：动作块有信号但还粗；世界模型在 WAM/FD 上都有可辨视频动力学（~17 dB），FD 略优于 WAM。真机 closed-loop 与 π0 对比仍待做。

### 加长 ~3.2s（chunk=96，同日）

窗口：`start_frame=274`。训练只见过 32，动作误差上升属预期。

| 指标 | 值 |
|------|-----|
| Action L1（raw / meanstd） | 17.7 / 0.82 |
| WAM / FD PSNR | 16.9 / 16.8 dB |

GitHub 可读页（含 GIF/mp4）：[`docs/ACTION_POLICY_OFFLINE_WAM.md`](../docs/ACTION_POLICY_OFFLINE_WAM.md)

### 加长 ~8.0s（chunk=240，2026-08-12）

窗口：`start_frame=273`；241 帧 @30fps。

| 指标 | 值 |
|------|-----|
| Action L1（raw / meanstd） | 15.2 / 0.71 |
| WAM / FD PSNR | 16.0 / 16.1 dB |

## 与 Vision 的关系

Vision 500 已作世界模型证据；本轨是 **会动手** 的主展示。Action-policy 的 WAM / forward_dynamics 离线评测提供同一底座上的联合证据。
