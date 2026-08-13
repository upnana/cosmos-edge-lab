# GPU 使用情况记录

训练 / 评测默认开启采样（`MONITOR_GPU=1`）。

## 产物

`outputs/gpu_monitor/<tag>_<stamp>/`

| 文件 | 内容 |
|------|------|
| `gpu.csv` | 每 `GPU_MONITOR_INTERVAL` 秒一行（util / 显存 / 功耗 / 温度） |
| `summary.json` | mean / max / p95 util，峰值显存 |
| `meta.json` | tag、host、开始时间 |

## 自动挂载

| 脚本 | tag |
|------|-----|
| `launch_vision_sft.sh` | `vision_sft` |
| `launch_action_policy.sh` | `action_policy` |
| `run_vision_500_then_heldout_i2v.sh`（I2V 段） | `vision_i2v_eval` |
| `run_action_wam_heldout.sh`（推理段） | `action_wam_eval` |

关闭：`MONITOR_GPU=0 bash scripts/...`

## 手动包一层任意命令

```bash
bash scripts/monitor_gpu.sh wrap --tag my_job -- sleep 30
# 或
DIR=$(bash scripts/monitor_gpu.sh start --tag my_job)
# ... your command ...
bash scripts/monitor_gpu.sh stop --tag my_job --out-dir "$DIR"
```

采样间隔：`GPU_MONITOR_INTERVAL=2`（秒，默认 5）。
