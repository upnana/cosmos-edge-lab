# 真机评测 Checklist — SO-101 叠块（vs π0）

> 任务：`stack the blocks from bottom to top white then blue then black`  
> 策略：Cosmos3-Edge Action-Policy `stack3cam_action_policy_edge_2000`  
> 对照：π0 3-cam（`upna/pi0_stack_white_blue_black_3cam_*`）  
> 离线 WAM/FD 已完成 → 本清单补 **closed-loop SR**。

相关脚本：

- 离线：`scripts/rollout_action_offline.py`
- 真机闭环驱动：`scripts/so101_rollout_driver.py`（LeRobot SO101Follower + Cosmos/π0）
- 包装：`scripts/rollout_action_real.sh`
- 公共：`scripts/rollout_common.py`

> **硬件注意：** 当前 H100 训练机通常 **无** `/dev/ttyUSB*` / `/dev/video*`。  
> 在此机用 `DRY_RUN=1` 验链路；**真机 SR** 须在接好臂与相机的 bench PC 上 `FORCE_REAL=1` 跑。

---

## 0. 成功定义（先写死，再开测）

| 项 | v1 约定 |
|----|---------|
| 成功 | 白→蓝→黑 **稳定叠成三层**，终态保持 ≥2s，无需人手扶 |
| 失败 | 碰倒 / 抓空 / 顺序错 / 超时 / 急停 / 出工作区 |
| 超时 | 单 trial ≤ **90s**（可改，两模型必须相同） |
| 初始条件 | 三块分散在桌面固定区；臂回 home；灯光/相机与采集时接近 |
| 最少 trial | 每策略 **≥20**（建议 30）；报告 **SR = 成功数 / 总数** |

可选二级分：抓到目标块 / 放到正确相对位置（用于失败分析，不算主 SR）。

---

## 1. 硬件与安全（真机必过）

- [ ] SO-101 follower 上电；急停可达、已测试
- [ ] 工作区清空；线缆不缠臂
- [ ] 相机：`front` + `wrist` 可用（与训练一致）；对焦/曝光正常
- [ ] 关节限位 / 速度上限已设；禁止未限速全功率
- [ ] **人在环**：每次 trial 可立刻急停；先 dry-run（不下发或极低速）
- [ ] 记录固件 / LeRobot 版本 / 相机设备名到 `notes/rollout_trials.md`

---

## 2. 软件前置

- [ ] `source scripts/env.sh`；`SO101_ROOT` 正确
- [ ] Action export 存在：`outputs/export/action_stack3cam_action_policy_edge_2000`
- [ ] mean/std：`so101_stack_3cam_meanstd.json`（与训练同一份）
- [ ] `domain=so101_follower`，`chunk=32`，`image_size=256`，front\|wrist concat
- [ ] 离线骨架跑通：`python scripts/rollout_action_offline.py --help`
- [ ] π0 推理入口可复现（路径/ckpt 写进 notes，避免口口相传）

---

## 3. 评测协议（两模型必须对齐）

| 规则 | 说明 |
|------|------|
| 同一桌面布局族 | 可用固定 5 套初始布局，循环抽 |
| 同一提示词 | 与训练 caption 一致 |
| 同一超时 / 成功定义 | 见 §0 |
| 交替或分块 | 避免「先测 A 全成功再测 B」的疲劳/温度偏置 |
| 禁止中途调参 | 改超参算新实验，另开一行 notes |
| 录像 | 每 trial 至少 front；文件名含 `policy_trial_ok/fail` |

---

## 4. 离线 rollout（真机前）

目的：不接臂，验证 **动作块 → denorm → 记录格式**；可选 open-loop 与 GT 比 L1。

```bash
source scripts/env.sh
python scripts/rollout_action_offline.py \
  --eval-root outputs/eval_action_wam \
  --episodes 111 \
  --out-dir outputs/rollout_offline \
  --write-trial-log
```

- [ ] 能读 WAM `sample_outputs.json` 的 `action` `[T,6]`（meanstd）
- [ ] denorm 后关节角量级合理（与 teleop 同量级）
- [ ] 写出 `trials.jsonl` + 摘要 SR 占位（离线默认无 success 字段，仅诊断）

---

## 5. 真机 rollout

```bash
source scripts/env.sh

# 本机（无臂）先 dry-run 验日志链路
DRY_RUN=1 N_TRIALS=1 POLICY=zeros bash scripts/rollout_action_real.sh

# bench PC（有 SO-101 + front/wrist[/side]）——确认急停后
export SO101_PORT=/dev/ttyUSB0   # 或 lerobot-find-port 结果
export CAM_FRONT=0 CAM_WRIST=1 CAM_SIDE=2
FORCE_REAL=1 INTERACTIVE=1 POLICY=cosmos N_TRIALS=20 bash scripts/rollout_action_real.sh
FORCE_REAL=1 INTERACTIVE=1 POLICY=pi0    N_TRIALS=20 bash scripts/rollout_action_real.sh
```

环境：`LEROBOT_PYTHON` 默认 `miniconda3/envs/lerobot_alohamini`；`LEROBOT_SRC` 默认 `lerobot_alohamini/src`。

每 trial：

1. `home` → 人摆块（layout ID）
2. 闭环：`get_obs → policy.infer_chunk → execute_chunk` 直到成功/失败/超时
3. 人标 `success` + 失败码（`INTERACTIVE=1`）
4. 日志：`outputs/rollout_real/<policy>_<stamp>/trials.jsonl`

- [ ] Cosmos 完成 ≥20 trials  
- [ ] π0 完成 ≥20 trials  
- [ ] 失败码分布表（抓空 / 碰撞 / 超时 / …）  
- [ ] 各剪 2–3 条成功 + 2–3 条失败进 `docs/assets/`（小 mp4/GIF）

---

## 6. 报告要写什么

写入 `notes/rollout_trials.md` 与（可选）`docs/RESULTS_SUMMARY.md`：

| 字段 | 例 |
|------|-----|
| 日期 / 操作者 | |
| ckpt | `iter_000002000` / π0 step |
| N / 成功数 / **SR%** | 20 / 6 / **30%** |
| 平均完成时间 | |
| Top-3 失败模式 | |
| 相对 π0 | 高/低/接近 + 一句话 |
| 视频链接 | GitHub assets 路径 |

**不要**用离线 Action L1 / PSNR 代替 SR。

---

## 7. 明确不做（本轮）

- 不改成功定义中途放水
- 不把 RoboLab 官方 SR 说成本 SO-101 真机结果
- 不强制一次跑满上百 trial（先 20，稳了再加）

---

## 8. 完成后勾选

- [ ] 离线骨架验证通过  
- [ ] 真机安全检查通过  
- [ ] Cosmos SR 记入 notes  
- [ ] π0 SR 记入 notes  
- [ ] 成功/失败视频入库并（可选）push GitHub  
