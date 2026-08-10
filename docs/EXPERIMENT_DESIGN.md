# 实验设计 — Stack3Cam WAM on Cosmos3-Edge

## 问题

基于 **Cosmos3-Edge** 的 **World-Action Model**，在我自己的 SO-101 遥操作数据
（`stack_3blocks_white_blue_black_3cam`）上后训练之后，能否同时学到：

1. **世界动力学**（前视相机视频 / Vision SFT），以及
2. **动作策略**（6D 绝对关节，前视+腕部），

并足够支撑后续边缘部署方向 —— 而不是「又一个上游配方的克隆」？

这个问题由本 lab 拥有。NVIDIA `cosmos-framework` 只是**训练引擎**。

## 假设

- 在前视 clip 上做 Vision SFT，能教会白→蓝→黑叠块任务的场景 / 物体动态。
- 以 WAM 模式做 Action-policy SFT（视频 + 状态 → 动作块），能教会可执行的关节轨迹。
- 双轨后训练，对后续 sim/real 闭环，比纯策略（π0）或纯视觉更有用。

## 变量（本仓库掌控）

| 旋钮 | v1 选择 | 原因 |
|------|---------|------|
| 底座 | Cosmos3-Edge | 相对 Nano/Super 更小、偏边缘的 WAM |
| 机器人 | SO-101 follower，6D 绝对关节 | 与我的遥操作一致 |
| 相机 | Vision：前视；Action：前视+腕部拼接 | 侧视留给后续消融 |
| 归一化 | 来自数据集 `stats.json` 的 mean/std | 稳定绝对关节 |
| 硬件 | 1×H100 | TOML 里的 lab 默认 |
| 对照 | π0 3-cam（LeRobot）作基线 | 同数据、不同模型族 |

## 非目标（v1）

- 本仓库暂不交付完整 RDK / HBM 部署流水线。
- 不宣称对 π0 的 SOTA —— 只做可对比的个人 WAM 轨道。
- 不 fork cosmos-framework；补丁放在 `patches/`，再 sync 进去。

## 成功标准（v1）

1. Vision smoke：约 100 iter 内 loss 下降，checkpoint 写到 `outputs/`。
2. Action smoke：10-iter 跑通，动作头有效。
3. 较长 run 结束并保存 DCP；在 `notes/` 留下定性视频 / rollout 记录。
4. 在 `notes/` 写下与 π0 3-cam（同叠块任务）的文字对比。

## 实验目录

具体配方指针与运行日志模板见 `experiments/stack3cam_wam/`。
