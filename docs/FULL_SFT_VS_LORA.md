# ~120 个 episode：Full SFT 还是 LoRA？

针对本 lab 的 Cosmos3-Edge + stack3cam（约 120 episode）设定。

## 短答

**不是自动「上 LoRA」。**  
数据少 → 要防**过拟合**（短跑、早停）。  
这不等于「Full SFT 就不对」。

## 这里的「Full SFT」指什么

我们的 Edge Vision 配方（`configs/vision_sft_edge.toml`）：

- **没有** `lora_enabled`
- 通过 `keys_to_select` 训选定模块  
  （`moe_gen`、`time_embedder`、`vae2llm`、`llm2vae`、`k_norm_und_for_gen`）
- 官方注释：Edge/Nano 风格配方用 `lr=1e-4` 做 full fine-tune

这是 **生成通路上的模块级 Full FT**，不是「解冻整个 4B，把 120 ep 倒进所有权重」。

官方 **LoRA** 是 Super 档视觉配方（`vision_sft_super`：
`lora_enabled=true`，`keys_to_select=["lora_"]`），因为 backbone 太大。

## 决策表

| 情形 | 优先 |
|------|------|
| Edge Vision v1 / smoke（我们默认） | 模块级 **Full SFT**（当前 TOML） |
| Super 32B Vision | **LoRA**（上游默认） |
| 训练 loss 塌、生成只会复述训练集 | 先缩短 `max_iter`、降 lr、早停 → 再考虑对 `moe_gen` 上 LoRA |
| Action-Policy 头（`action2llm` / …） | 训这些模块（常新初始化 / skip-load）；LoRA 不是首选 |
| ~120 episodes | 限制步数；用定性 rollout / held-out clip 验证 |

## Vision（~120 ep）

- 继续用 Edge **模块级 Full SFT** 合理。
- 主要风险：iter 太多 → 死记硬背。
- LoRA 有正则化作用，但在小 Edge 上也可能 **欠拟合**。
- 不要只因为 episode≈120 就改成 LoRA。

## Action-Policy（~120 ep）

- 动作头要学新映射；checkpoint 常 **跳过加载** 这些张量，从初始化训起。
- 优先动作模块 FT（+ 配方里的 gen keys），预算从小/smoke 起步。
- 同样：先短 smoke（10 → 数百 iter），再拉长。

## Lab 实践

1. v1：保持 Edge 模块级 Full SFT（Vision）与 SO101 action 配方。  
2. 预算：Vision 先数百 iter；Action 先 10-iter smoke 再放大。  
3. 若过拟合：先砍步数 / lr；再考虑 Vision LoRA。  
4. 结果记到 `experiments/stack3cam_wam/README.md` 与 `notes/`。

**结论：** episode 少 → 少训、盯泛化；  
**不要**只因 N≈120 就在 Edge 上默认 LoRA。
