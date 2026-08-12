#!/usr/bin/env python3
"""Offline Action-Policy rollout skeleton (no robot).

Uses existing WAM held-out outputs:
  - denorm predicted action chunks
  - compare open-loop L1 vs GT
  - write trials.jsonl / summary.json (success left null — offline)

Example:
  source scripts/env.sh
  python scripts/rollout_action_offline.py \\
    --eval-root outputs/eval_action_wam \\
    --episodes 111 \\
    --out-dir outputs/rollout_offline

Part of upnana/cosmos-edge-lab.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_LAB = Path(__file__).resolve().parents[1]
if str(_LAB / "scripts") not in sys.path:
    sys.path.insert(0, str(_LAB / "scripts"))

from rollout_common import (  # noqa: E402
    DEFAULT_PROMPT,
    DEFAULT_STATS,
    TrialLogger,
    TrialRecord,
    action_l1,
    denorm_actions,
    extract_wam_action,
    load_meanstd,
    now_trial_id,
)


def _ep_tag(ep: int) -> str:
    return f"{ep:03d}"


def run_episode(
    eval_root: Path,
    episode: int,
    mean: np.ndarray,
    std: np.ndarray,
    logger: TrialLogger,
    policy: str,
) -> dict:
    ep = _ep_tag(episode)
    inp = eval_root / "inputs" / f"ep{ep}"
    wam_out = eval_root / "wam_out" / f"wam_ep{ep}"
    sample = wam_out / "sample_outputs.json"
    gt_path = inp / "actions.json"
    if not sample.exists():
        raise FileNotFoundError(f"missing {sample} — run run_action_wam_heldout.sh first")
    if not gt_path.exists():
        raise FileNotFoundError(gt_path)

    pred_n = extract_wam_action(sample)
    gt = np.asarray(json.loads(gt_path.read_text()), dtype=np.float64)
    if gt.ndim != 2:
        raise ValueError(f"GT actions shape {gt.shape}")
    pred_raw = denorm_actions(pred_n, mean, std)
    # GT in prepare script is already raw absolute degrees
    l1_raw = action_l1(gt, pred_raw)

    out_dir = logger.out_dir / f"ep{ep}"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "pred_actions_raw.json"
    raw_path.write_text(json.dumps(pred_raw.tolist()) + "\n")

    meta = {
        "episode": episode,
        "pred_shape": list(pred_raw.shape),
        "gt_shape": list(gt.shape),
        "action_l1_raw_deg": l1_raw,
        "pred_meanstd_sample0": pred_n[0].tolist(),
        "pred_raw_sample0": pred_raw[0].tolist(),
        "wam_sample_outputs": str(sample),
    }
    rec = TrialRecord(
        trial_id=now_trial_id(f"offline_ep{ep}"),
        policy=policy,
        mode="offline",
        prompt=DEFAULT_PROMPT,
        layout_id=f"heldout_ep{ep}",
        success=None,  # offline: no stack success label
        notes=f"open-loop L1 raw={l1_raw:.3f} deg (not SR)",
        action_raw_path=str(raw_path),
        meta=meta,
    )
    logger.append(rec)
    return meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-root", type=Path, required=True, help="e.g. outputs/eval_action_wam")
    p.add_argument("--episodes", type=int, nargs="+", default=[111])
    p.add_argument("--out-dir", type=Path, default=_LAB / "outputs" / "rollout_offline")
    p.add_argument("--stats-path", type=Path, default=DEFAULT_STATS)
    p.add_argument("--policy", default="cosmos_action_2000")
    p.add_argument("--write-trial-log", action="store_true", default=True)
    args = p.parse_args()

    mean, std = load_meanstd(args.stats_path)
    logger = TrialLogger(args.out_dir)
    results = []
    for ep in args.episodes:
        print(f">>> offline ep{ep:03d}")
        results.append(run_episode(args.eval_root, ep, mean, std, logger, args.policy))
        print(f"    action L1 raw={results[-1]['action_l1_raw_deg']:.3f}°")

    summary = logger.summarize()
    (args.out_dir / "offline_diagnostics.json").write_text(
        json.dumps({"episodes": results, "trial_summary": summary}, indent=2) + "\n"
    )
    print(f">>> wrote {logger.path}")
    print(f">>> summary {args.out_dir / 'summary.json'}")
    print("NOTE: offline does not compute stack SR — use rollout_action_real.sh for that.")


if __name__ == "__main__":
    main()
