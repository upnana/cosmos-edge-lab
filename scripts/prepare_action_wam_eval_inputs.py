#!/usr/bin/env python3
"""Build held-out SO-101 WAM / forward_dynamics inference inputs.

Reads LeRobot v3 stack3cam data, writes:
  - front|wrist concat observation mp4 (256x256 each)
  - GT absolute 6D actions JSON (chunk_length steps)
  - sample JSON for model_mode=wam and forward_dynamics

Part of upnana/cosmos-edge-lab.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq


DEFAULT_EPISODES = [111, 3, 22, 47, 69, 99]
DEFAULT_PROMPT = "stack the blocks from bottom to top white then blue then black"
DEFAULT_STATS = Path(
    "/home/july/cosmos-framework/cosmos_framework/data/generator/action/"
    "normalizer_stats/so101_stack_3cam_meanstd.json"
)


def _load_meanstd(stats_path: Path) -> tuple[np.ndarray, np.ndarray]:
    stats = json.loads(stats_path.read_text())
    return np.asarray(stats["mean"], dtype=np.float64), np.asarray(stats["std"], dtype=np.float64)


def _best_motion_start(actions: np.ndarray, chunk_length: int) -> int:
    """Pick start frame maximizing mean per-step L2 motion over the chunk."""
    if len(actions) <= chunk_length:
        return 0
    best_i, best_m = 0, -1.0
    for i in range(0, len(actions) - chunk_length + 1):
        w = actions[i : i + chunk_length]
        motion = float(np.mean(np.linalg.norm(np.diff(w, axis=0), axis=1)))
        if motion > best_m:
            best_m, best_i = motion, i
    return best_i


def _load_tasks(root: Path) -> dict[int, str]:
    table = pq.read_table(root / "meta" / "tasks.parquet")
    cols = set(table.column_names)
    task_index = table.column("task_index").to_pylist()
    if "task" in cols:
        texts = [str(x) for x in table.column("task").to_pylist()]
    else:
        # LeRobot often stores task text as the parquet index column.
        idx_col = "__index_level_0__"
        if idx_col not in cols:
            raise KeyError(f"tasks.parquet missing task text column; cols={sorted(cols)}")
        texts = [str(x) for x in table.column(idx_col).to_pylist()]
    return {int(i): t for i, t in zip(task_index, texts)}


def _load_episode_rows(root: Path, episode: int) -> list[dict]:
    rows: list[dict] = []
    for path in sorted((root / "data").glob("chunk-*/file-*.parquet")):
        table = pq.read_table(
            path,
            columns=[
                "action",
                "observation.state",
                "timestamp",
                "frame_index",
                "episode_index",
                "task_index",
            ],
        )
        subset = table.filter(pc.equal(table.column("episode_index"), episode))
        if subset.num_rows == 0:
            continue
        rows.extend(subset.to_pylist())
    if not rows:
        raise ValueError(f"episode {episode} not found under {root}/data")
    rows.sort(key=lambda r: int(r["frame_index"]))
    return rows


def _video_path(root: Path, camera: str, episode: int) -> Path:
    # video_path pattern: videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4
    # For this dataset file_index == episode_index within chunk-000.
    chunk = episode // 1000
    path = root / "videos" / f"observation.images.{camera}" / f"chunk-{chunk:03d}" / f"file-{episode:03d}.mp4"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _read_frames(path: Path, n: int, size: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    frames: list[np.ndarray] = []
    try:
        while len(frames) < n:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
            frames.append(rgb)
    finally:
        cap.release()
    if len(frames) < n:
        if not frames:
            raise RuntimeError(f"no frames decoded from {path}")
        while len(frames) < n:
            frames.append(frames[-1].copy())
    return frames


def _write_mp4(path: Path, frames: list[np.ndarray], fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot open VideoWriter for {path}")
    try:
        for rgb in frames:
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def prepare_episode(
    *,
    root: Path,
    out_dir: Path,
    episode: int,
    chunk_length: int,
    image_size: int,
    fps: float,
    tasks: dict[int, str],
    mean: np.ndarray,
    std: np.ndarray,
    start_frame: int | None = 0,
    auto_motion_window: bool = False,
) -> dict:
    rows = _load_episode_rows(root, episode)
    need = chunk_length + 1
    all_actions = np.asarray([row["action"] for row in rows], dtype=np.float64)
    if auto_motion_window or start_frame is None:
        start_frame = _best_motion_start(all_actions, chunk_length)
    assert start_frame is not None
    if start_frame + need > len(rows):
        raise ValueError(
            f"ep{episode}: need {need} frames from start={start_frame}, have {len(rows)}"
        )
    window = rows[start_frame : start_frame + need]
    task_index = int(window[0].get("task_index", 0))
    prompt = tasks.get(task_index, DEFAULT_PROMPT)

    front = _read_frames(_video_path(root, "front", episode), start_frame + need, image_size)
    wrist = _read_frames(_video_path(root, "wrist", episode), start_frame + need, image_size)
    front = front[start_frame : start_frame + need]
    wrist = wrist[start_frame : start_frame + need]
    concat = [np.concatenate([f, w], axis=1) for f, w in zip(front, wrist)]

    ep_dir = out_dir / f"ep{episode:03d}"
    ep_dir.mkdir(parents=True, exist_ok=True)
    concat_path = ep_dir / "concat_obs.mp4"
    actions_path = ep_dir / "actions.json"
    actions_norm_path = ep_dir / "actions_norm.json"
    gt_state_path = ep_dir / "initial_state.json"
    meta_path = ep_dir / "meta.json"

    _write_mp4(concat_path, concat, fps=fps)

    actions = np.asarray(
        [list(map(float, row["action"])) for row in window[:chunk_length]],
        dtype=np.float64,
    )
    actions_norm = (actions - mean) / np.clip(std, 1e-6, None)
    motion = float(np.mean(np.linalg.norm(np.diff(actions, axis=0), axis=1))) if len(actions) > 1 else 0.0
    initial_state = list(map(float, window[0]["observation.state"]))
    actions_path.write_text(json.dumps(actions.tolist(), indent=2))
    actions_norm_path.write_text(json.dumps(actions_norm.tolist(), indent=2))
    gt_state_path.write_text(json.dumps(initial_state, indent=2))

    # Paths in sample JSON are relative to the JSON file location.
    common = {
        "domain_name": "so101_follower",
        "view_point": "concat_view",
        "action_chunk_size": chunk_length,
        "fps": int(fps),
        "image_size": image_size,
        "resolution": str(image_size),
        # Omit aspect_ratio: concat front|wrist is ~2:1 and is not in
        # VIDEO_RES_SIZE_INFO; action inference pads via find_closest_target_size.
        "seed": 0,
        "prompt": prompt,
        "vision_path": "concat_obs.mp4",
    }

    wam_name = f"wam_ep{episode:03d}"
    fd_name = f"fd_ep{episode:03d}"
    wam_json = {
        **common,
        "model_mode": "wam",
        "name": wam_name,
        # Optional GT for scoring / smoke checks (not consumed by WAM forward).
        "action_path": "actions.json",
        "extra": {
            "episode": episode,
            "start_frame": start_frame,
            "golden_action_path": "actions.json",
            "golden_action_norm_path": "actions_norm.json",
            "initial_state_path": "initial_state.json",
        },
    }
    fd_json = {
        **common,
        "model_mode": "forward_dynamics",
        "name": fd_name,
        # Training used meanstd; offline FD must condition in the same space.
        "action_path": "actions_norm.json",
        "extra": {
            "episode": episode,
            "start_frame": start_frame,
            "action_space": "meanstd",
        },
    }
    (ep_dir / f"{wam_name}.json").write_text(json.dumps(wam_json, indent=2) + "\n")
    (ep_dir / f"{fd_name}.json").write_text(json.dumps(fd_json, indent=2) + "\n")

    meta = {
        "episode": episode,
        "start_frame": start_frame,
        "num_frames": need,
        "chunk_length": chunk_length,
        "fps": fps,
        "image_size": image_size,
        "prompt": prompt,
        "concat_obs": str(concat_path),
        "actions": str(actions_path),
        "actions_norm": str(actions_norm_path),
        "mean_step_motion": motion,
        "wam_json": str(ep_dir / f"{wam_name}.json"),
        "fd_json": str(ep_dir / f"{fd_name}.json"),
        "action_dim": int(actions.shape[1]) if actions.size else 0,
        "n_actions": int(actions.shape[0]),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/home/july/datasets/stack_3blocks_white_blue_black_3cam"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/home/july/cosmos-edge-lab/outputs/eval_action_wam/inputs"),
    )
    parser.add_argument("--episodes", type=int, nargs="+", default=DEFAULT_EPISODES)
    parser.add_argument("--chunk-length", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--start-frame", type=int, default=None, help="Fixed start; default with --auto-motion-window")
    parser.add_argument(
        "--auto-motion-window",
        action="store_true",
        default=True,
        help="Pick start frame with max mean step motion (default on)",
    )
    parser.add_argument(
        "--no-auto-motion-window",
        action="store_false",
        dest="auto_motion_window",
    )
    parser.add_argument("--stats-path", type=Path, default=DEFAULT_STATS)
    args = parser.parse_args()

    if args.start_frame is not None:
        args.auto_motion_window = False

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tasks = _load_tasks(args.root)
    mean, std = _load_meanstd(args.stats_path)
    metas = []
    for ep in args.episodes:
        meta = prepare_episode(
            root=args.root,
            out_dir=args.out_dir,
            episode=ep,
            chunk_length=args.chunk_length,
            image_size=args.image_size,
            fps=args.fps,
            tasks=tasks,
            mean=mean,
            std=std,
            start_frame=args.start_frame,
            auto_motion_window=args.auto_motion_window,
        )
        metas.append(meta)
        print(
            f">>> prepared ep{ep:03d}  start={meta['start_frame']}  "
            f"motion={meta['mean_step_motion']:.3f}  actions={meta['n_actions']}  -> {meta['wam_json']}"
        )

    index = {
        "root": str(args.root),
        "chunk_length": args.chunk_length,
        "image_size": args.image_size,
        "fps": args.fps,
        "start_frame": args.start_frame,
        "auto_motion_window": args.auto_motion_window,
        "stats_path": str(args.stats_path),
        "episodes": metas,
    }
    (args.out_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f">>> wrote {args.out_dir / 'index.json'}")


if __name__ == "__main__":
    main()
