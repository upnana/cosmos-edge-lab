# Rollout trials log — SO-101 stack vs π0

> 协议见 [`docs/REAL_ROBOT_EVAL_CHECKLIST.md`](../docs/REAL_ROBOT_EVAL_CHECKLIST.md)  
> 驱动：`scripts/so101_rollout_driver.py`  
> 包装：`scripts/rollout_action_real.sh`  
> 离线：`scripts/rollout_action_offline.py`

## 本机状态（H100）

| 项 | 值 |
|----|-----|
| `/dev/ttyUSB*` | 无 |
| `/dev/video*` | 无 |
| dry-run 链路 | 可（`POLICY=zeros`） |
| 真机 SR | **blocked — 需 bench PC** |

## 环境

| 项 | 值 |
|----|-----|
| 日期 | |
| 操作者 | |
| Cosmos ckpt | `stack3cam_action_policy_edge_2000` / `iter_000002000` |
| π0 ckpt | `upna/pi0_stack_white_blue_black_3cam_*` step= |
| 超时 | 90s |
| 成功定义 | 白→蓝→黑稳定叠成 |

## 离线诊断（非 SR）

```bash
source scripts/env.sh
python scripts/rollout_action_offline.py \
  --eval-root outputs/eval_action_wam \
  --episodes 111 \
  --out-dir outputs/rollout_offline
```

| Ep | Action L1 raw (°) | 备注 |
|----|-------------------|------|
| 111 | | |

## 真机结果

### Cosmos Action-Policy 2000

| N | 成功 | SR | 平均时长 | Top 失败 |
|---|------|-----|----------|----------|
|  |  |  |  |  |

trials 目录：`outputs/rollout_real/cosmos_action_2000_*/`

### π0

| N | 成功 | SR | 平均时长 | Top 失败 |
|---|------|-----|----------|----------|
|  |  |  |  |  |

## 结论（一句话）

- 

## 视频

- 成功：
- 失败：
