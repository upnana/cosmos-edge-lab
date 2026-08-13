# 分辨率配置说明（采集 / 策略输入 / 仿真）

> 适用于 `stack3cam_wam` Action-policy（`action_policy_so101_edge`）。  
> 写论文或接仿真/真机时，请区分 **渲染或采集分辨率** 与 **策略输入分辨率**。

相关：[`ACTION_POLICY_SO101_INFERENCE_PARAMS.md`](ACTION_POLICY_SO101_INFERENCE_PARAMS.md) · [`BENCH_PC_OFFLINE_INFERENCE.md`](BENCH_PC_OFFLINE_INFERENCE.md)

---

## 1. 本实验实际配置（权威）

| 层级 | 分辨率 | 说明 |
|------|--------|------|
| **原始采集**（LeRobot 真机遥操作） | front/wrist **640×480**；side **800×480** | 数据集 `stack_3blocks_white_blue_black_3cam` |
| **Action-policy 输入** | 每路 **256×256**，front\|wrist 横拼 → **512×256** | `viewpoint=concat_view`，`image_size=256` |
| **Vision SFT** | 主要用 **front**（见 Vision 配方） | 与 Action 相机子集不同 |
| **侧视 side** | 在数据集里，**不进 Action-policy** | “stack3cam” 指数据有三路，不是策略吃三路 |

训练侧在 `SO101LeRobotDataset._load_video` 中：

1. 取 `observation.images.front` + `observation.images.wrist`
2. `F.interpolate(..., size=(256, 256), mode=bilinear)`
3. `torch.cat(..., dim=-1)` → `[T, C, 256, 512]`

推理（离线 WAM / 真机）必须做 **同一套 resize + concat**，不能直接喂 640×480。

---

## 2. 写论文时建议怎么写

建议同时写清两层，避免审稿人把「相机原生分辨率」和「网络输入」混为一谈：

**推荐表述（可直接改）：**

> Cameras are recorded at 640×480 (front, wrist). For the action policy we
> resize each view to 256×256 and concatenate them horizontally
> (front | wrist → 512×256), matching training (`image_size=256`,
> `viewpoint=concat_view`). The side camera is present in the dataset but
> unused by the action policy.

若报告仿真评测，同样声明：仿真渲染分辨率（若有）→ resize 到 **256×256 per view** 再 concat。

---

## 3. 仿真环境一般怎么配（文献常见做法）

多数 manipulation / VLA 论文里：

| 层级 | 常见做法 |
|------|----------|
| 仿真渲染 | 480p / 640×480 或更高（引擎默认） |
| 策略输入 | 固定小图：84 / 128 / **224** / **256** 等 |

原则是：

```text
sim render 或 real capture（可较大）
        ↓  resize
policy input（与训练 image_size 一致）
```

本仓库当前 **Action-policy 评测主线是真机/离线 WAM**，不是「在 Isaac 里用另一套分辨率重训」。  
本地 Isaac Sim SO101（`isaac-sim-so101`）主要用于场景与关节 bring-up；若日后用仿真闭环跑本策略，**仍须 front|wrist → 256×256 → concat**，与真机一致。

---

## 4. 与 π0 对照（同任务数据）

同数据集上的 π0 3-cam 基线记录的是 **原始特征尺寸**（front/wrist `[3,480,640]`，side `[3,480,800]`），见 pi0 训练笔记。  
Cosmos Action-policy **额外**规定策略输入为 256；对比实验时不要把「π0 数据分辨率」直接写成「Cosmos 策略输入分辨率」。

---

## 5. 检查清单

- [ ] 论文图表写明：**capture 640×480** vs **policy 256×256（concat 512×256）**
- [ ] 真机/仿真 client 与训练相同的相机集合：**front + wrist only**
- [ ] 归一化仍用 `so101_stack_3cam_meanstd.json`（与分辨率无关，但常一起漏配）

---

## 6. 代码 / 文档锚点

| 内容 | 位置 |
|------|------|
| `image_size=256`, `concat_view` | `patches/cosmos-framework/action_policy_so101_edge.py` + `so101_lerobot_dataset.py` |
| 推理参数汇总 | [`ACTION_POLICY_SO101_INFERENCE_PARAMS.md`](ACTION_POLICY_SO101_INFERENCE_PARAMS.md) |
| Bench PC 离线 | [`BENCH_PC_OFFLINE_INFERENCE.md`](BENCH_PC_OFFLINE_INFERENCE.md) |
| 真机 checklist | [`REAL_ROBOT_EVAL_CHECKLIST.md`](REAL_ROBOT_EVAL_CHECKLIST.md) |
