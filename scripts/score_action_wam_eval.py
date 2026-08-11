#!/usr/bin/env python3
"""Score offline Action-Policy WAM / forward_dynamics held-out eval.

Computes action L1/MSE (raw + mean/std-normalized) and optional vision PSNR,
writes side-by-side pred/GT videos and metrics.json.

Part of upnana/cosmos-edge-lab.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _extract_pred_action(sample_outputs: dict) -> np.ndarray:
    outputs = sample_outputs.get("outputs") or []
    if not outputs:
        raise ValueError("sample_outputs.json has empty outputs")
    content = outputs[0].get("content") or {}
    if "action" not in content:
        raise KeyError("sample_outputs content missing 'action'")
    arr = np.asarray(content["action"], dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"expected action [T,D], got shape {arr.shape}")
    return arr


def _action_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    t = min(gt.shape[0], pred.shape[0])
    d = min(gt.shape[1], pred.shape[1])
    g = gt[:t, :d]
    p = pred[:t, :d]
    diff = p - g
    mse = float(np.mean(diff**2))
    l1 = float(np.mean(np.abs(diff)))
    per_dim_mse = np.mean(diff**2, axis=0).tolist()
    per_dim_l1 = np.mean(np.abs(diff), axis=0).tolist()
    return {
        "T": int(t),
        "D": int(d),
        "mse": mse,
        "l1": l1,
        "rmse": float(math.sqrt(mse)),
        "per_dim_mse": per_dim_mse,
        "per_dim_l1": per_dim_l1,
    }


def _normalize(actions: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (actions - mean) / np.clip(std, 1e-6, None)


def _denormalize(actions: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return actions * np.clip(std, 1e-6, None) + mean


def _read_video(path: Path, max_frames: int | None = None) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            if max_frames is not None and len(frames) >= max_frames:
                break
    finally:
        cap.release()
    return frames


def _resize(frame: np.ndarray, wh: tuple[int, int]) -> np.ndarray:
    w, h = wh
    return cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)
    mse = np.mean((a_f - b_f) ** 2)
    if mse <= 1e-12:
        return 99.0
    return float(20.0 * math.log10(255.0) - 10.0 * math.log10(mse))


def _write_side_by_side(
    out_path: Path,
    panels: list[tuple[str, list[np.ndarray]]],
    fps: float,
) -> dict:
    """Write horizontally stacked labeled panels; returns mean PSNR vs first panel if >=2."""
    if not panels or not panels[0][1]:
        raise ValueError("empty panels")
    n = min(len(p[1]) for p in panels)
    # Unify height to min height; scale width proportionally.
    target_h = min(p[1][0].shape[0] for p in panels)
    resized: list[list[np.ndarray]] = []
    for _, frames in panels:
        row = []
        for fr in frames[:n]:
            h, w = fr.shape[:2]
            tw = max(1, int(round(w * (target_h / h))))
            row.append(_resize(fr, (tw, target_h)))
        resized.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel_w = [r[0].shape[1] for r in resized]
    total_w = int(sum(panel_w))
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (total_w, target_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot open VideoWriter {out_path}")

    psnrs: list[float] = []
    try:
        for i in range(n):
            tiles = []
            for pi, (label, _) in enumerate(panels):
                fr = resized[pi][i].copy()
                cv2.putText(
                    fr,
                    label,
                    (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                tiles.append(fr)
            canvas = np.concatenate(tiles, axis=1)
            writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
            if len(resized) >= 2:
                # Compare panel1 vs panel0 (GT) after matching size.
                gt = resized[0][i]
                pred = _resize(resized[1][i], (gt.shape[1], gt.shape[0]))
                psnrs.append(_psnr(gt, pred))
    finally:
        writer.release()

    stats: dict = {"path": str(out_path), "num_frames": n}
    if psnrs:
        stats["psnr_mean_vs_gt"] = float(np.mean(psnrs))
        stats["psnr_min_vs_gt"] = float(np.min(psnrs))
    return stats


def _find_vision(sample_dir: Path) -> Path | None:
    for name in ("vision.mp4", "vision.jpg"):
        p = sample_dir / name
        if p.is_file():
            return p
    # Fallback: any mp4
    mp4s = sorted(sample_dir.glob("*.mp4"))
    return mp4s[0] if mp4s else None


def score_episode(
    *,
    eval_root: Path,
    episode: int,
    mean: np.ndarray,
    std: np.ndarray,
    fps: float,
) -> dict:
    ep_tag = f"{episode:03d}"
    inp = eval_root / "inputs" / f"ep{ep_tag}"
    gt_actions = np.asarray(_load_json(inp / "actions.json"), dtype=np.float64)
    gt_video = _read_video(inp / "concat_obs.mp4")

    wam_dir = eval_root / "wam_out" / f"wam_ep{ep_tag}"
    fd_dir = eval_root / "fd_out" / f"fd_ep{ep_tag}"
    preview_dir = eval_root / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    result: dict = {"episode": episode, "input_dir": str(inp)}

    # --- Action metrics from WAM ---
    # Offline inference leaves action_normalizer=None, so WAM preds are in
    # meanstd-normalized space (same as training targets). Denorm for raw metrics.
    so_path = wam_dir / "sample_outputs.json"
    if so_path.is_file():
        pred_norm = _extract_pred_action(_load_json(so_path))
        d = min(gt_actions.shape[1], pred_norm.shape[1], mean.shape[0])
        pred_raw = _denormalize(pred_norm[:, :d], mean[:d], std[:d])
        raw = _action_metrics(gt_actions[:, :d], pred_raw)
        g_n = _normalize(gt_actions[: raw["T"], :d], mean[:d], std[:d])
        p_n = pred_norm[: raw["T"], :d]
        norm = _action_metrics(g_n, p_n)
        result["action"] = {
            "raw": raw,
            "normalized_meanstd": norm,
            "pred_shape": list(pred_norm.shape),
            "gt_shape": list(gt_actions.shape),
            "pred_space": "meanstd_then_denorm_for_raw",
            "sample_outputs": str(so_path),
            "start_frame": (_load_json(inp / "meta.json").get("start_frame") if (inp / "meta.json").is_file() else None),
            "mean_step_motion": (
                _load_json(inp / "meta.json").get("mean_step_motion") if (inp / "meta.json").is_file() else None
            ),
        }
    else:
        result["action"] = {"error": f"missing {so_path}"}

    # --- Vision previews ---
    wam_vision = _find_vision(wam_dir) if wam_dir.is_dir() else None
    fd_vision = _find_vision(fd_dir) if fd_dir.is_dir() else None
    vision_stats: dict = {}

    if wam_vision is not None:
        wam_frames = _read_video(wam_vision)
        vision_stats["wam"] = _write_side_by_side(
            preview_dir / f"ep{ep_tag}_gt_vs_wam.mp4",
            [("GT", gt_video), ("WAM", wam_frames)],
            fps=fps,
        )
        vision_stats["wam"]["pred"] = str(wam_vision)

    if fd_vision is not None:
        fd_frames = _read_video(fd_vision)
        vision_stats["forward_dynamics"] = _write_side_by_side(
            preview_dir / f"ep{ep_tag}_gt_vs_fd.mp4",
            [("GT", gt_video), ("FD", fd_frames)],
            fps=fps,
        )
        vision_stats["forward_dynamics"]["pred"] = str(fd_vision)

    if wam_vision is not None and fd_vision is not None:
        vision_stats["triple"] = _write_side_by_side(
            preview_dir / f"ep{ep_tag}_gt_wam_fd.mp4",
            [("GT", gt_video), ("WAM", _read_video(wam_vision)), ("FD", _read_video(fd_vision))],
            fps=fps,
        )

    result["vision"] = vision_stats
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=Path("/home/july/cosmos-edge-lab/outputs/eval_action_wam"),
    )
    parser.add_argument("--episodes", type=int, nargs="+", default=[111])
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=Path(
            "/home/july/cosmos-framework/cosmos_framework/data/generator/action/"
            "normalizer_stats/so101_stack_3cam_meanstd.json"
        ),
    )
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    stats = _load_json(args.stats_path)
    mean = np.asarray(stats["mean"], dtype=np.float64)
    std = np.asarray(stats["std"], dtype=np.float64)

    episodes = []
    for ep in args.episodes:
        print(f">>> scoring ep{ep:03d}")
        episodes.append(
            score_episode(
                eval_root=args.eval_root,
                episode=ep,
                mean=mean,
                std=std,
                fps=args.fps,
            )
        )

    summary = {
        "eval_root": str(args.eval_root),
        "stats_path": str(args.stats_path),
        "episodes": episodes,
    }
    # Compact top-line metrics
    lines = []
    for ep in episodes:
        tag = f"ep{ep['episode']:03d}"
        act = ep.get("action", {})
        if "raw" in act:
            lines.append(
                {
                    "episode": ep["episode"],
                    "action_l1": act["raw"]["l1"],
                    "action_mse": act["raw"]["mse"],
                    "action_l1_norm": act["normalized_meanstd"]["l1"],
                    "action_mse_norm": act["normalized_meanstd"]["mse"],
                    "wam_psnr": (ep.get("vision") or {}).get("wam", {}).get("psnr_mean_vs_gt"),
                    "fd_psnr": (ep.get("vision") or {})
                    .get("forward_dynamics", {})
                    .get("psnr_mean_vs_gt"),
                }
            )
            print(
                f"  {tag}  action L1={act['raw']['l1']:.4f} MSE={act['raw']['mse']:.4f}  "
                f"normL1={act['normalized_meanstd']['l1']:.4f}  "
                f"wamPSNR={(ep.get('vision') or {}).get('wam', {}).get('psnr_mean_vs_gt')}  "
                f"fdPSNR={(ep.get('vision') or {}).get('forward_dynamics', {}).get('psnr_mean_vs_gt')}"
            )
        else:
            print(f"  {tag}  action error: {act}")
            lines.append({"episode": ep["episode"], "error": act.get("error")})

    summary["summary"] = lines
    out = args.eval_root / "metrics.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f">>> wrote {out}")


if __name__ == "__main__":
    main()
