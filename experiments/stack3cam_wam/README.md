# stack3cam_wam

基于 **Cosmos3-Edge** 的个人 WAM 轨道，使用我的 SO-101 三色叠块 3 相机数据集。

## 数据

- 原始 LeRobot v3：`/home/july/datasets/stack_3blocks_white_blue_black_3cam`
- Vision 处理后：`data/processed/stack3cam_vision_sft/`（已 gitignore）
- Action 经 SO101 适配器（patch 进 framework）直接读 LeRobot 根目录

## 配方

| 轨道 | 配置 | 启动 |
|------|------|------|
| Vision SFT | `configs/vision_sft_edge.toml` | `scripts/launch_vision_sft.sh` |
| Action policy | `configs/action_policy_so101_edge.toml` | `scripts/launch_action_policy.sh` |

## 运行日志（边跑边填）

| 日期 | 轨道 | 步数 | 备注 |
|------|------|------|------|
| 2026-08-10 | Vision smoke | 10 | loss≈1.56；已导出并跑 I2V，见 `docs/VISION_SMOKE_EXPORT_I2V.md` |
| 2026-08-10 | Action smoke | 10 | loss≈20→15.6；ckpt `iter_000000010` |
| 2026-08-10 | Vision SFT | 500 | job `stack3cam_vision_sft_edge_500`；held-out ep111 I2V，见 `docs/VISION_500_HELDOUT_I2V.md` |

基线 π0（同任务）：Hugging Face `upna/pi0_stack_white_blue_black_3cam_060000`（及后续 80k）。
