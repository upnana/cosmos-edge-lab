# 工作笔记

## 为什么不只 clone cosmos-framework？

上游配方面向 DROID / LIBERO + Nano 默认。我的设定是：

- SO-101 绝对 6D
- 3 相机遥操作（前视 / 腕部 / 侧视）
- Cosmos3-Edge + 1×H100
- 目标：个人 WAM 实验叙事，可与我的 π0 叠块 run 对照

因此本仓库拥有 **适配器、归一化、TOML、启动器与研究日志**。
Framework 仍是 `../cosmos-framework` 下的依赖。

## 开放问题

1. 仅前视的 Vision SFT 是否帮助 Action-policy 微调，还是只训 Action 就够？
2. 前视+腕部是否足够，侧视对叠块是否重要？
3. Edge WAM SFT 之后，通往 RDK / 量化部署的最短路径是什么？
