# Full SFT vs LoRA for ~120 episodes

Analysis for this lab’s Cosmos3-Edge + stack3cam (~120 episodes) setting.

## Short answer

**Not automatically “use LoRA.”**  
Small data → fight **overfitting** (short runs, early stop).  
That is not the same as “full SFT is wrong.”

## What “full SFT” means here

Our Edge Vision recipe (`configs/vision_sft_edge.toml`):

- **No** `lora_enabled`
- Trains selected modules via `keys_to_select`
  (`moe_gen`, `time_embedder`, `vae2llm`, `llm2vae`, `k_norm_und_for_gen`)
- Official comment: full fine-tune at `lr=1e-4` for Edge/Nano-style recipes

This is **module-level full FT of the gen pathway**, not “unfreeze the entire 4B
and dump 120 episodes into every weight.”

Official **LoRA** is the Super-tier vision recipe (`vision_sft_super`:
`lora_enabled=true`, `keys_to_select=["lora_"]`) because the backbone is huge.

## Decision table

| Situation | Prefer |
|-----------|--------|
| Edge Vision v1 / smoke (our default) | Module-level **full SFT** (current TOML) |
| Super 32B Vision | **LoRA** (upstream default) |
| Train loss collapses, generations only replay train set | Shorter `max_iter`, lower lr, early stop → then try LoRA on `moe_gen` |
| Action-Policy heads (`action2llm` / …) | Train those modules (often freshly init / skip-load); LoRA is a weaker default |
| ~120 episodes | Cap steps; validate with qualitative rollouts / held-out clips |

## Vision (~120 ep)

- Staying on Edge **module full SFT** is reasonable.
- Main risk: too many iterations → memorization.
- LoRA can help regularization but can also **underfit** on a small Edge model.
- Do not switch to LoRA only because episode count is ~120.

## Action-Policy (~120 ep)

- Action heads need to learn a new mapping; checkpoint often **skips loading**
  those tensors so they train from init.
- Prefer action-module FT (+ recipe gen keys) with small/smoke budgets.
- Same rule: short smoke (10 → hundreds of iters) before long runs.

## Lab practice

1. v1: keep Edge module-level full SFT (Vision) and SO101 action recipe.  
2. Budget: Vision hundreds of iters to start; Action 10-iter smoke then scale.  
3. If overfit: cut steps / lr first; consider Vision LoRA second.  
4. Log outcomes in `experiments/stack3cam_wam/README.md` and `notes/`.

**Bottom line:** few episodes → train less and watch generalization;  
**do not default to LoRA** on Edge just because N≈120.
