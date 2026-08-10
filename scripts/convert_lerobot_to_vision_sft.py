#!/usr/bin/env python3
"""Lab-owned LeRobot v3 -> Cosmos3 Vision-SFT JSONL converter.

Part of upnana/cosmos-edge-lab (personal WAM experiments).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def _load_tasks(root: Path) -> dict[int, str]:
    tasks_df = pd.read_parquet(root / "meta" / "tasks.parquet")
    task_texts = tasks_df["task"] if "task" in tasks_df.columns else tasks_df.index
    return {int(task_index): str(task) for task, task_index in zip(task_texts, tasks_df["task_index"])}


def _load_episodes(root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
        rows.extend(pq.read_table(path).to_pylist())
    rows.sort(key=lambda r: int(r["episode_index"]))
    return rows


def _probe_video(path: Path) -> tuple[int, int, float]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    out = subprocess.check_output(cmd, text=True)
    stream = json.loads(out)["streams"][0]
    num, den = stream["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    return int(stream["width"]), int(stream["height"]), fps


def _extract_clip(src: Path, dst: Path, start_s: float, end_s: float) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_s:.3f}",
        "-to",
        f"{end_s:.3f}",
        "-i",
        str(src),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(dst),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _caption_json(task: str, width: int, height: int, duration_s: float, fps: float) -> dict:
    dur = max(1, int(round(duration_s)))
    return {
        "subjects": [
            {
                "description": "A SO-101 robotic arm with a parallel gripper",
                "action": task,
            }
        ],
        "background_setting": "An indoor tabletop workspace with colored stacking blocks",
        "cinematography": {
            "camera_motion": "static",
            "framing": "medium shot",
            "camera_angle": "third-person front view",
        },
        "actions": [{"time": f"0:00-0:{dur:02d}", "description": task}],
        "temporal_caption": (
            f"A SO-101 robotic arm performs the task: {task}. "
            "The camera is static and shows the arm manipulating colored blocks on a tabletop."
        ),
        "resolution": {"H": height, "W": width},
        "aspect_ratio": f"{width},{height}",
        "duration": f"{dur}s",
        "fps": int(round(fps)),
    }


def convert(dataset_root: Path, output_root: Path, camera: str, val_ratio: float, seed: int) -> None:
    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    fps_meta = float(info.get("fps", 30))
    tasks = _load_tasks(dataset_root)
    episodes = _load_episodes(dataset_root)

    rng = random.Random(seed)
    indices = list(range(len(episodes)))
    rng.shuffle(indices)
    n_val = max(1, int(round(len(indices) * val_ratio))) if len(indices) > 1 else 0
    val_set = set(indices[:n_val])

    splits = {"train": [], "val": []}
    for i, ep in enumerate(episodes):
        split = "val" if i in val_set else "train"
        ep_idx = int(ep["episode_index"])
        uuid = f"episode_{ep_idx:06d}_clip000"

        chunk = int(ep[f"videos/{camera}/chunk_index"])
        file_i = int(ep[f"videos/{camera}/file_index"])
        start_s = float(ep[f"videos/{camera}/from_timestamp"])
        end_s = float(ep[f"videos/{camera}/to_timestamp"])
        src = dataset_root / "videos" / camera / f"chunk-{chunk:03d}" / f"file-{file_i:03d}.mp4"
        if not src.exists():
            raise FileNotFoundError(src)

        rel_video = f"videos/{uuid}.mp4"
        dst = output_root / split / rel_video
        _extract_clip(src, dst, start_s, end_s)

        width, height, fps = _probe_video(dst)
        duration_s = max(1e-3, end_s - start_s)
        # Prefer dataset fps when probe is noisy.
        fps = fps_meta or fps
        n_frames = max(1, int(round(duration_s * fps)))

        task_list = ep.get("tasks") or []
        if isinstance(task_list, list) and task_list:
            task = str(task_list[0])
        else:
            # Fallback via task_index stats if present.
            task = next(iter(tasks.values()))

        cap = _caption_json(task, width, height, duration_s, fps)
        sample = {
            "uuid": uuid,
            "duration": duration_s,
            "width": width,
            "height": height,
            "vision_path": rel_video,
            "t2w_windows": [
                {
                    "start_frame": 0,
                    "end_frame": n_frames - 1,
                    "temporal_interval": 1,
                    "caption_json": cap,
                    "caption": cap["temporal_caption"],
                }
            ],
        }
        splits[split].append(sample)
        print(f"[{split}] {uuid}  {duration_s:.2f}s  {width}x{height}")

    for split, samples in splits.items():
        out_dir = output_root / split
        out_dir.mkdir(parents=True, exist_ok=True)
        jsonl = out_dir / "video_dataset_file.jsonl"
        with jsonl.open("w") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"wrote {jsonl} ({len(samples)} samples)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--camera", default="observation.images.front")
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    convert(args.dataset_root, args.output_root, args.camera, args.val_ratio, args.seed)


if __name__ == "__main__":
    main()
