# Working notes

## Why not only clone cosmos-framework?

Upstream recipes target DROID / LIBERO + Nano defaults. My setup is:

- SO-101 absolute 6D
- 3-cam teleop (front / wrist / side)
- Cosmos3-Edge + 1×H100
- Goal: personal WAM experiment story, comparable to my π0 stack runs

So this repo owns **adapters, norms, TOMLs, launchers, and the research log**.
Framework stays a dependency under `../cosmos-framework`.

## Open questions

1. Does front-only vision SFT help action-policy finetune, or should we train action only?
2. Is front+wrist enough, or does side view matter for stacking?
3. After Edge WAM SFT, what’s the smallest path to RDK / quantized deploy?
