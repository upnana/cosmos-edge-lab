# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""SO-101 LeRobot v3 dataset for Cosmos3 action-policy SFT.

Absolute 6D joint actions + proprioceptive state, dual-view concat
(front + wrist). Designed for stack-blocks style teleop datasets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F

from cosmos_framework.data.generator.action.action_spec import ActionSpec, Gripper, Joint, build_action_spec
from cosmos_framework.data.generator.action.datasets.base_dataset import ActionBaseDataset

Viewpoint = Literal["concat_view", "third_person_view", "wrist_view"]

_FRONT = "observation.images.front"
_WRIST = "observation.images.wrist"
_STATE = "observation.state"
_ACTION = "action"

_NORMALIZER_PATH = Path(__file__).parent.parent / "normalizer_stats/so101_stack_3cam_meanstd.json"


class SO101LeRobotDataset(ActionBaseDataset):
    """SO-101 follower: absolute joint positions ``[6]`` + dual camera concat."""

    def __init__(
        self,
        root: str,
        fps: float = 30.0,
        chunk_length: int = 32,
        mode: str = "wam",
        pose_convention: str = "backward_framewise",
        tolerance_s: float = 2e-2,
        viewpoint: Viewpoint = "concat_view",
        action_normalization: str | None = "meanstd",
        use_state: bool = True,
        image_size: int = 256,
        sample_stride: int = 1,
    ) -> None:
        super().__init__(
            root=root,
            domain_name="so101_follower",
            fps=fps,
            chunk_length=chunk_length,
            mode=mode,
            pose_convention=pose_convention,
            tolerance_s=tolerance_s,
            viewpoint=viewpoint,
            action_normalization=action_normalization,
            sample_stride=sample_stride,
        )
        self._use_state = bool(use_state)
        self._image_size = int(image_size)

    @property
    def action_dim(self) -> int:
        return 6

    def _action_spec(self) -> ActionSpec:
        # 5 arm joints + 1 gripper (idle detection uses JOINT / GRIPPER rules).
        return build_action_spec(Joint(n=5, label="arm"), Gripper())

    @classmethod
    def _stats_path(cls) -> Path:
        return _NORMALIZER_PATH

    def get_shuffle_blocks(self) -> list[tuple[int, int]]:
        """Per-episode ``(start, length)`` flat-index blocks for iterable shuffle."""
        n = len(self)
        if n <= 0:
            return []
        blocks: list[tuple[int, int]] = []
        cur_ep: int | None = None
        start = 0
        length = 0
        for idx in range(n):
            row_idx = int(idx) * self._sample_stride
            ep = int(self._rows[row_idx]["episode_index"])
            if cur_ep is None:
                cur_ep = ep
                start = idx
                length = 1
            elif ep == cur_ep:
                length += 1
            else:
                blocks.append((start, length))
                cur_ep = ep
                start = idx
                length = 1
        if length > 0:
            blocks.append((start, length))
        return blocks

    def __getitem__(self, idx: int) -> dict[str, Any]:
        mode = self._choose_mode()
        row_idx = int(idx) * self._sample_stride
        observation_rows = self._rows[row_idx : row_idx + self._chunk_length + 1]
        action_rows = observation_rows[: self._chunk_length]

        episode = self._episodes[int(observation_rows[0]["episode_index"])]
        task_index = int(observation_rows[0].get("task_index", 0))
        ai_caption = self._tasks.get(task_index, "stack the blocks")

        video = self._load_video(episode, observation_rows)
        action = torch.from_numpy(
            np.asarray([row[_ACTION] for row in action_rows], dtype=np.float32)
        ).float()  # [T, 6]

        extras: dict[str, Any] = {
            "additional_view_description": (
                "The left half shows the front third-person camera; "
                "the right half shows the wrist-mounted camera."
            )
        }
        if self._use_state:
            initial_state = torch.from_numpy(np.asarray(observation_rows[0][_STATE], dtype=np.float32)).float()
            action = torch.cat([initial_state.unsqueeze(0), action], dim=0)
            extras["initial_pose"] = torch.eye(4)

        return self._build_result(
            mode=mode,
            video=video,
            action=action,
            ai_caption=ai_caption,
            **extras,
        )

    def _load_video(self, episode: dict[str, Any], observation_rows: list[dict[str, Any]]) -> torch.Tensor:
        from lerobot.datasets.video_utils import decode_video_frames

        timestamps = [float(row["timestamp"]) for row in observation_rows]
        keys: list[str]
        if self._viewpoint == "third_person_view":
            keys = [_FRONT]
        elif self._viewpoint == "wrist_view":
            keys = [_WRIST]
        else:
            keys = [_FRONT, _WRIST]

        clips = []
        for key in keys:
            base = float(episode.get(f"videos/{key}/from_timestamp", 0.0))
            frames = decode_video_frames(
                self._video_path(episode, key),
                [base + ts for ts in timestamps],
                self._tolerance_s,
                backend="pyav",  # avoid torchcodec + system NPP path issues on bare metal
            )  # [T, C, H, W] float 0-1
            frames = F.interpolate(
                frames,
                size=(self._image_size, self._image_size),
                mode="bilinear",
                align_corners=False,
            )
            clips.append(frames)

        if len(clips) == 1:
            return clips[0]  # [T, C, H, W]; _build_result permutes to [C, T, H, W]
        # Horizontal concat -> [T, C, H, 2W]
        return torch.cat(clips, dim=-1)
