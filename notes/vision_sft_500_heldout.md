# Vision SFT 500 iter + held-out I2V

- 启动：2026-08-10 ~19:24（流水线 PID 见 `outputs/logs/vision_500_pipeline.nohup.log`）
- job：`stack3cam_vision_sft_edge_500`
- 入口脚本：`scripts/run_vision_500_then_heldout_i2v.sh`（训完自动 export → I2V）
- 数据：既有 `train/` 108 / `val/` 12；**不用训过的 ep119 做评测**
- Held-out 评测 episode：**111**（在 val 集合内）
- 评测素材：`outputs/eval_vision_heldout/`
- 预计：~4h 训练（~28s/iter）+ 导出 + I2V

## 日志 / 产物

| 项 | 路径 |
|----|------|
| 训练日志 | `outputs/logs/vision_sft_500.log` |
| 流水线日志 | `outputs/logs/vision_500_pipeline.nohup.log` |
| ckpt | `outputs/cosmos3/sft/stack3cam_vision_sft_edge_500/checkpoints/` |
| 导出（完成后） | `outputs/export/vision_stack3cam_vision_sft_edge_500/` |
| I2V 预测（完成后） | `outputs/eval_vision_heldout/i2v_out/` |
| GT 2s / 8s | `outputs/eval_vision_heldout/assets/gt_front_*.mp4` |

Val（held-out）episode 列表：`3,22,25,30,47,57,59,69,73,99,103,111`
