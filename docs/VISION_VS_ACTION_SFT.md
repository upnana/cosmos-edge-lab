# Vision SFT vs Action-Policy SFT

两者都在同一套 SO-101 stack-3cam 遥操作数据上后训练 **Cosmos3-Edge**，
但学的东西不同。

## 对比

| | Vision SFT | Action-Policy SFT |
|---|---|---|
| 目标 | 世界动力学（「场景怎么动」） | 可执行关节（「手臂怎么动」） |
| 输入 | 前视 clip + caption | 前视+腕部图像 + 本体感觉 → 动作块 |
| 输出 | 视频 / 世界模型生成 | 绝对 **6D** 关节动作 |
| 数据形态 | 先转成 JSONL + mp4 | 经 SO101 适配器直接读 LeRobot v3 |
| 相机（v1） | 仅 `observation.images.front` | 前视 + 腕部拼接（侧视暂不用） |

这是 **两条配方**，不是一条自动流水线。可以只跑 Vision、只跑 Action，或并行。

## Vision SFT —— 怎么「训」

1. **转换** LeRobot episode → 短 clip + caption JSONL  
   （`scripts/prepare_vision_data.sh` → `scripts/convert_lerobot_to_vision_sft.py`）
2. **加载** Cosmos3-Edge DCP + Wan VAE
3. **优化** 生成侧模块（见 `configs/vision_sft_edge.toml` 的
   `keys_to_select`：`moe_gen`、`time_embedder`、`vae2llm`、`llm2vae`、
   `k_norm_und_for_gen`）
4. 启动：`scripts/launch_vision_sft.sh`

直觉：教会模型桌面 / 叠块视觉动态（白 → 蓝 → 黑）。
本身**不会**吐出关节角。

## Action-Policy SFT —— 怎么「训」

1. **不做 JSONL 转换** —— `SO101LeRobotDataset` 直接读  
   `/home/july/datasets/stack_3blocks_white_blue_black_3cam`（或 `SO101_ROOT`）
2. **归一化** 绝对 6D 关节，用数据集 stats 的 mean/std  
   （`patches/.../so101_stack_3cam_meanstd.json`）
3. **训练** `action_gen=True`；动作头
   （`action2llm` / `llm2action` / …）+ WAM 风格 batching
4. 启动：`scripts/launch_action_policy.sh`（会先跑 `sync_patches.sh`）

直觉：更接近「能动手」；与 π0 在同数据上的*任务*可比，模型族不同。

## 共同前置

1. Wan2.2 VAE 已下载到 framework 树下  
2. HF Cosmos3-Edge → DCP（`scripts/prepare_edge_dcp.sh`；需要空闲 GPU）  
3. 不要和长时间跑着的 π0 抢同一块 H100（除非提前规划）

## Lab 入口

| 步骤 | 脚本 |
|------|------|
| Vision 数据 | `scripts/prepare_vision_data.sh` |
| Vision 训练 | `scripts/launch_vision_sft.sh` |
| Action 训练 | `scripts/launch_action_policy.sh` |
| Patch 同步 | `scripts/sync_patches.sh` |
