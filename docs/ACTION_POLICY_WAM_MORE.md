# Action-Policy WAM/FD — more held-out (1s / 3s / 8s)

> job `stack3cam_action_policy_edge_2000` / `iter_000002000`  
> episodes: 3, 22, 47, 69, 99  
> front|wrist @256, fps=30, auto-motion

## 1s (~1.1s, chunk=32)

**Mean:** Action L1 raw=14.41°, norm=0.612, WAM=17.8 dB, FD=17.9 dB

| Ep | Action L1 (raw°) | Action L1 (norm) | WAM PSNR | FD PSNR |
|----|------------------|------------------|----------|---------|
| 3 | 18.60 | 0.771 | 17.1 | 17.3 |
| 22 | 9.19 | 0.442 | 19.1 | 19.0 |
| 47 | 11.79 | 0.571 | 19.9 | 19.3 |
| 69 | 18.84 | 0.666 | 17.9 | 17.7 |
| 99 | 13.62 | 0.612 | 15.3 | 16.2 |

### Preview (ep022)

![ep022 1s](assets/action_wam_more/1s/ep022_gt_wam_fd.gif)

<video src="assets/action_wam_more/1s/ep022_gt_wam_fd.mp4" controls width="960" preload="metadata"></video>

## 3s (~3.2s, chunk=96)

**Mean:** Action L1 raw=22.21°, norm=0.855, WAM=16.9 dB, FD=17.1 dB

| Ep | Action L1 (raw°) | Action L1 (norm) | WAM PSNR | FD PSNR |
|----|------------------|------------------|----------|---------|
| 3 | 19.32 | 0.885 | 17.2 | 17.3 |
| 22 | 28.17 | 1.037 | 16.8 | 17.8 |
| 47 | 21.62 | 0.867 | 17.8 | 17.7 |
| 69 | 16.31 | 0.654 | 16.7 | 16.7 |
| 99 | 25.63 | 0.832 | 15.8 | 15.9 |

### Preview (ep022)

![ep022 3s](assets/action_wam_more/3s/ep022_gt_wam_fd.gif)

<video src="assets/action_wam_more/3s/ep022_gt_wam_fd.mp4" controls width="960" preload="metadata"></video>

## 8s (~8.0s, chunk=240)

**Mean:** Action L1 raw=20.23°, norm=0.795, WAM=15.1 dB, FD=15.1 dB

| Ep | Action L1 (raw°) | Action L1 (norm) | WAM PSNR | FD PSNR |
|----|------------------|------------------|----------|---------|
| 3 | 33.84 | 1.033 | 16.1 | 16.1 |
| 22 | 18.20 | 0.756 | 14.1 | 14.1 |
| 47 | 17.44 | 0.834 | 15.6 | 15.5 |
| 69 | 18.77 | 0.756 | 15.9 | 16.1 |
| 99 | 12.87 | 0.595 | 13.8 | 13.8 |

### Preview (ep022)

![ep022 8s](assets/action_wam_more/8s/ep022_gt_wam_fd.gif)

<video src="assets/action_wam_more/8s/ep022_gt_wam_fd.mp4" controls width="960" preload="metadata"></video>

## 下载

`outputs/bundles/action_wam_more.tar.gz`（含 1s/3s/8s）

## 复现

```bash
EPISODES="3 22 47 69 99" CHUNK_LENGTH=96 EVAL_ROOT=$LAB_ROOT/outputs/eval_action_wam_more_3s bash scripts/run_action_wam_heldout.sh
EPISODES="3 22 47 69 99" CHUNK_LENGTH=240 EVAL_ROOT=$LAB_ROOT/outputs/eval_action_wam_more_8s bash scripts/run_action_wam_heldout.sh
```
