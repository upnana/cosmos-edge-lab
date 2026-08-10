# cosmos-edge-lab

**Personal World-Action-Model (WAM) experiments on Cosmos3-Edge** — not a fork of someone else’s recipe dump.

Owner: [upnana](https://github.com/upnana)  
Remote: https://github.com/upnana/cosmos-edge-lab

This repo owns:

- experiment questions, hypotheses, and run logs
- SO-101 / stack-3cam data adapters and normalizers
- Vision SFT + Action-policy TOMLs tuned for **1×H100 + Cosmos3-Edge**
- launch / convert scripts

It does **not** own the trainer core. Training still runs through a local
[NVIDIA cosmos-framework](https://github.com/NVIDIA/cosmos-framework) install
(default: `../cosmos-framework`). Lab patches sync in via `scripts/sync_patches.sh`.

```
cosmos-edge-lab/          ← your science
../cosmos-framework/      ← engine (external)
```

## First experiment: `stack3cam_wam`

Dataset: SO-101 teleop `stack_3blocks_white_blue_black_3cam` (LeRobot v3, 3 cams, 6D joints).

Two tracks on the same base (**Cosmos3-Edge**):

1. **Vision SFT** — front-camera world / video dynamics  
2. **Action-policy SFT** — WAM-style action chunks (front+wrist)

### Docs

| Doc | Topic |
|-----|--------|
| [`docs/EXPERIMENT_DESIGN.md`](docs/EXPERIMENT_DESIGN.md) | Hypothesis & success criteria |
| [`docs/COSMOS_ARCHITECTURE.md`](docs/COSMOS_ARCHITECTURE.md) | Cosmos3 MoT / Edge network |
| [`docs/VISION_VS_ACTION_SFT.md`](docs/VISION_VS_ACTION_SFT.md) | How Vision vs Action are trained |
| [`docs/CODE_MAP.md`](docs/CODE_MAP.md) | Where code lives / what to edit |
| [`docs/FULL_SFT_VS_LORA.md`](docs/FULL_SFT_VS_LORA.md) | Full module SFT vs LoRA (~120 ep) |
| [`notes/analysis_2026-08-10.md`](notes/analysis_2026-08-10.md) | Session summary |

Recipe folder: [`experiments/stack3cam_wam/`](experiments/stack3cam_wam/)

Baseline (same task, different family): π0 3-cam on LeRobot (`upna/pi0_stack_white_blue_black_3cam_*`).

## Setup

```bash
# 1) Engine (once)
#    clone + uv sync cosmos-framework elsewhere, e.g. ../cosmos-framework

# 2) Lab env
cd /path/to/cosmos-edge-lab
export COSMOS_FRAMEWORK_ROOT=/path/to/cosmos-framework   # optional if ../cosmos-framework
source scripts/env.sh

# 3) Inject SO101 adapters into the engine
bash scripts/sync_patches.sh
```

Also once on the machine:

- Wan2.2 VAE at `$COSMOS_FRAMEWORK_ROOT/examples/checkpoints/wan22_vae/Wan2.2_VAE.pth`
- HF weights `/home/july/models/Cosmos3-Edge` (override `EDGE_HF`)
- Convert DCP: `bash scripts/prepare_edge_dcp.sh`

## Runbook

```bash
source scripts/env.sh

# Vision data (front cam clips → JSONL)
bash scripts/prepare_vision_data.sh

# Smoke Vision (when GPU free)
export EXTRA_TAIL_OVERRIDES="trainer.max_iter=10 checkpoint.save_iter=10"
bash scripts/launch_vision_sft.sh

# Smoke Action
export EXTRA_TAIL_OVERRIDES="trainer.max_iter=10 checkpoint.save_iter=10 dataloader_train.max_samples_per_batch=2"
bash scripts/launch_action_policy.sh

# Full-ish: unset EXTRA_TAIL_OVERRIDES and re-run
```

Outputs land under `outputs/` (gitignored). Checkpoints under `checkpoints/`.

## Layout

```
configs/                 # TOML recipes you own
scripts/                 # convert / sync / launch
patches/cosmos-framework # SO101 dataset + action experiment sources
experiments/stack3cam_wam
docs/                    # design docs
notes/                   # working hypotheses / comparisons
data/ processed/         # local only
```

## License note

Upstream NVIDIA files kept under `patches/` and some configs retain their
SPDX headers (OpenMDW). Your experiment narrative and adapters in this repo
are part of the personal lab; respect upstream licenses when redistributing.
