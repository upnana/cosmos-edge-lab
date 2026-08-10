# Experiment design — Stack3Cam WAM on Cosmos3-Edge

## Question

Can a **World-Action Model** based on **Cosmos3-Edge**, post-trained on my own
SO-101 teleop (`stack_3blocks_white_blue_black_3cam`), learn both:

1. **World dynamics** (front-camera video / Vision SFT), and
2. **Action policy** (6D absolute joints, front+wrist),

well enough to transfer toward edge deployment — without being “just another
clone of an upstream recipe”?

This lab owns that question. NVIDIA `cosmos-framework` is only the **trainer
engine**.

## Hypothesis

- Vision SFT on front-cam clips teaches scene / object dynamics for the
  white→blue→black stack task.
- Action-policy SFT in WAM mode (video + state → action chunks) teaches
  executable joint trajectories.
- Dual post-training is more useful for later sim/real loop than policy-only
  (π0) or vision-only.

## Variables (owned by this repo)

| Knob | v1 choice | Why |
|------|-----------|-----|
| Base | Cosmos3-Edge | Smaller edge-oriented WAM vs Nano/Super |
| Robot | SO-101 follower, 6D abs joints | Matches my teleop |
| Cameras | Vision: front; Action: front+wrist concat | Side kept for later ablation |
| Norm | mean/std from dataset `stats.json` | Stable absolute joints |
| Hardware | 1×H100 | Lab defaults in TOMLs |
| Control | π0 3-cam (LeRobot) as baseline | Same dataset, different family |

## Non-goals (v1)

- Not shipping a full RDK / HBM deploy pipeline in this repo yet.
- Not claiming SOTA vs π0 — only a comparable personal WAM track.
- Not forking cosmos-framework; patches stay under `patches/` and sync in.

## Success criteria (v1)

1. Vision smoke: loss decreases over ~100 iters, checkpoints write under `outputs/`.
2. Action smoke: 10-iter run completes with valid action heads.
3. Full-ish runs finish with saved DCP; qualitative video / rollout notes in `notes/`.
4. Written comparison vs π0 3-cam (same stack task) in `notes/`.

## Experiment folder

See `experiments/stack3cam_wam/` for the concrete recipe pointers and run log template.
