#!/usr/bin/env python3
"""Shared helpers for offline / real SO-101 rollout logging.

Part of upnana/cosmos-edge-lab. Skeleton — fill robot I/O in real path.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_STATS = Path(
    "/home/july/cosmos-framework/cosmos_framework/data/generator/action/"
    "normalizer_stats/so101_stack_3cam_meanstd.json"
)

DEFAULT_PROMPT = "stack the blocks from bottom to top white then blue then black"

# Human-readable failure codes for real trials
FAILURE_CODES = (
    "grasp_miss",
    "wrong_order",
    "knock_over",
    "timeout",
    "out_of_workspace",
    "estop",
    "hardware",
    "other",
)


def load_meanstd(stats_path: Path = DEFAULT_STATS) -> tuple[np.ndarray, np.ndarray]:
    stats = json.loads(stats_path.read_text())
    mean = np.asarray(stats["mean"], dtype=np.float64)
    std = np.asarray(stats["std"], dtype=np.float64)
    return mean, std


def denorm_actions(actions: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """meanstd space -> raw joint degrees (absolute 6D)."""
    a = np.asarray(actions, dtype=np.float64)
    return a * np.clip(std, 1e-6, None) + mean


def norm_actions(actions: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    a = np.asarray(actions, dtype=np.float64)
    return (a - mean) / np.clip(std, 1e-6, None)


def extract_wam_action(sample_outputs_path: Path) -> np.ndarray:
    """Read [T, 6] action chunk from cosmos inference sample_outputs.json (meanstd)."""
    data = json.loads(sample_outputs_path.read_text())
    outputs = data.get("outputs") or []
    if not outputs:
        raise ValueError(f"empty outputs in {sample_outputs_path}")
    content = outputs[0].get("content") or {}
    if "action" not in content:
        raise KeyError(f"no action in {sample_outputs_path}")
    arr = np.asarray(content["action"], dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 6:
        raise ValueError(f"expected [T,6+], got {arr.shape}")
    return arr[:, :6]


@dataclass
class TrialRecord:
    trial_id: str
    policy: str
    mode: str  # offline | real | dry_run
    prompt: str = DEFAULT_PROMPT
    layout_id: str | None = None
    success: bool | None = None
    failure_code: str | None = None
    duration_s: float | None = None
    notes: str = ""
    video_path: str | None = None
    action_raw_path: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class TrialLogger:
    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.out_dir / "trials.jsonl"

    def append(self, record: TrialRecord) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(record.to_json() + "\n")

    def summarize(self) -> dict[str, Any]:
        rows = []
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        by_policy: dict[str, list] = {}
        for r in rows:
            by_policy.setdefault(r.get("policy", "?"), []).append(r)
        summary: dict[str, Any] = {"n_total": len(rows), "policies": {}}
        for policy, items in by_policy.items():
            labeled = [x for x in items if x.get("success") is not None]
            n_ok = sum(1 for x in labeled if x["success"])
            n = len(labeled)
            summary["policies"][policy] = {
                "n_labeled": n,
                "n_success": n_ok,
                "sr": (n_ok / n) if n else None,
                "n_unlabeled": len(items) - n,
            }
        out = self.out_dir / "summary.json"
        out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return summary


def action_l1(gt: np.ndarray, pred: np.ndarray) -> float:
    t = min(len(gt), len(pred))
    d = min(gt.shape[1], pred.shape[1])
    return float(np.mean(np.abs(gt[:t, :d] - pred[:t, :d])))


def now_trial_id(prefix: str = "t") -> str:
    return f"{prefix}_{int(time.time() * 1000)}"
