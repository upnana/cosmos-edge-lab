#!/usr/bin/env python3
"""SO-101 closed-loop rollout driver (Cosmos Action-Policy vs π0).

Wires the hooks that used to be TODOs in rollout_action_real.sh:

  connect / home / capture / infer / execute / log trials

Hardware: LeRobot ``SO101Follower`` via ``lerobot_alohamini`` (see --lerobot-src).
This H100 training host typically has **no** /dev/ttyUSB* or /dev/video* —
real SR must run on the bench PC with arm + cameras attached.

Modes:
  --dry-run     mock robot; no serial; policies may still run if weights exist
  --force-real  require live SO-101 (refuses if port/cameras missing)

Examples:
  # Safe smoke on GPU host (mock robot, zero actions)
  python scripts/so101_rollout_driver.py --dry-run --policy zeros --n-trials 1

  # Mock + Cosmos (warm worker loads once; subsequent chunks are fast)
  python scripts/so101_rollout_driver.py --dry-run --policy cosmos --n-trials 1

  # Bench PC real eval
  python scripts/so101_rollout_driver.py --force-real --policy cosmos --n-trials 20
  python scripts/so101_rollout_driver.py --force-real --policy pi0 --n-trials 20

Part of upnana/cosmos-edge-lab.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_LAB = Path(__file__).resolve().parents[1]
if str(_LAB / "scripts") not in sys.path:
    sys.path.insert(0, str(_LAB / "scripts"))

from rollout_common import (  # noqa: E402
    DEFAULT_PROMPT,
    DEFAULT_STATS,
    TrialLogger,
    TrialRecord,
    denorm_actions,
    extract_wam_action,
    load_meanstd,
    now_trial_id,
)

JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

# Rest-ish pose from held-out ep111 initial_state (degrees).
DEFAULT_HOME = [
    -8.017492294311523,
    -6.137786865234375,
    -4.915102958679199,
    97.05280303955078,
    -5.964010238647463,
    56.3197021484375,
]


def _probe_hardware(port: str) -> dict[str, Any]:
    videos = sorted(Path("/dev").glob("video*"))
    serials = []
    for pat in ("ttyUSB*", "ttyACM*", "serial/by-id/*"):
        serials.extend(sorted(Path("/dev").glob(pat)))
    port_ok = Path(port).exists() if port else False
    return {
        "port": port,
        "port_exists": port_ok,
        "videos": [str(p) for p in videos],
        "serials": [str(p) for p in serials],
        "ready": port_ok and len(videos) >= 2,
    }


# ---------------------------------------------------------------------------
# Robot backends
# ---------------------------------------------------------------------------


class RobotBackend:
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def home(self, pose: list[float]) -> None: ...
    def get_obs(self) -> dict[str, Any]: ...
    def execute_chunk(
        self,
        actions_deg: np.ndarray,
        hz: float,
        on_step: Any | None = None,
    ) -> None: ...


class MockRobot(RobotBackend):
    """No hardware — returns black frames + home joints."""

    def __init__(self, image_size: int = 256):
        self.image_size = image_size
        self._state = np.asarray(DEFAULT_HOME, dtype=np.float64)
        self._connected = False

    def connect(self) -> None:
        self._connected = True
        print("[robot] MockRobot connected (dry-run)")

    def disconnect(self) -> None:
        self._connected = False

    def home(self, pose: list[float]) -> None:
        self._state = np.asarray(pose, dtype=np.float64)
        print(f"[robot] MockRobot home -> {self._state.round(1).tolist()}")

    def get_obs(self) -> dict[str, Any]:
        h = w = self.image_size
        black = np.zeros((h, w, 3), dtype=np.uint8)
        side = np.zeros((h, int(w * 800 / 640), 3), dtype=np.uint8)
        obs = {
            "front": black.copy(),
            "wrist": black.copy(),
            "side": side,
            "state": self._state.copy(),
        }
        for i, j in enumerate(JOINTS):
            obs[f"{j}.pos"] = float(self._state[i])
        return obs

    def execute_chunk(
        self,
        actions_deg: np.ndarray,
        hz: float,
        on_step: Any | None = None,
    ) -> None:
        dt = 1.0 / max(hz, 1e-3)
        for row in actions_deg:
            self._state = np.asarray(row, dtype=np.float64)
            if on_step is not None:
                on_step(self.get_obs())
            time.sleep(min(dt, 0.05))


class SO101Robot(RobotBackend):
    """Live LeRobot SO101Follower."""

    def __init__(
        self,
        port: str,
        robot_id: str,
        cam_front: int | str,
        cam_wrist: int | str,
        cam_side: int | str | None,
        use_degrees: bool = True,
        max_relative_target: float | None = 15.0,
        calibrate: bool = False,
    ):
        self.port = port
        self.robot_id = robot_id
        self.cam_front = cam_front
        self.cam_wrist = cam_wrist
        self.cam_side = cam_side
        self.use_degrees = use_degrees
        self.max_relative_target = max_relative_target
        self.calibrate = calibrate
        self.robot = None

    def connect(self) -> None:
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

        # alohamini fork: lerobot.robots.so_follower
        # stock lerobot:  lerobot.robots.so101_follower
        try:
            from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        except ModuleNotFoundError:
            from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

        cameras = {
            "front": OpenCVCameraConfig(
                index_or_path=self.cam_front, width=640, height=480, fps=30
            ),
            "wrist": OpenCVCameraConfig(
                index_or_path=self.cam_wrist, width=640, height=480, fps=30
            ),
        }
        if self.cam_side is not None and str(self.cam_side) != "":
            if str(self.cam_side) in {str(self.cam_front), str(self.cam_wrist)}:
                raise ValueError(
                    f"CAM_SIDE={self.cam_side} collides with front/wrist "
                    f"({self.cam_front}/{self.cam_wrist}). Leave CAM_SIDE empty "
                    "for action-policy, or set it to the side camera (e.g. 4)."
                )
            cameras["side"] = OpenCVCameraConfig(
                index_or_path=self.cam_side, width=800, height=480, fps=30
            )

        cfg = SO101FollowerConfig(
            port=self.port,
            id=self.robot_id,
            cameras=cameras,
            use_degrees=self.use_degrees,
            max_relative_target=self.max_relative_target,
        )
        self.robot = SO101Follower(cfg)
        self.robot.connect(calibrate=self.calibrate)
        print(f"[robot] SO101Follower connected port={self.port} id={self.robot_id}")

    def disconnect(self) -> None:
        if self.robot is not None:
            self.robot.disconnect()
            self.robot = None

    def home(self, pose: list[float]) -> None:
        action = {f"{j}.pos": float(v) for j, v in zip(JOINTS, pose)}
        self.robot.send_action(action)
        time.sleep(1.5)

    def get_obs(self) -> dict[str, Any]:
        raw = self.robot.get_observation()
        state = np.asarray([float(raw[f"{j}.pos"]) for j in JOINTS], dtype=np.float64)
        out: dict[str, Any] = {"state": state, **raw}
        # Normalize camera key aliases
        for k in ("front", "wrist", "side"):
            if k in raw:
                out[k] = raw[k]
        return out

    def execute_chunk(
        self,
        actions_deg: np.ndarray,
        hz: float,
        on_step: Any | None = None,
    ) -> None:
        dt = 1.0 / max(hz, 1e-3)
        for row in actions_deg:
            action = {f"{j}.pos": float(v) for j, v in zip(JOINTS, row)}
            self.robot.send_action(action)
            if on_step is not None:
                on_step(self.get_obs())
            time.sleep(dt)


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------


def resize_rgb(img: np.ndarray, size: int) -> np.ndarray:
    """Resize HxWx3 camera frame. LeRobot OpenCV default is RGB."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def front_wrist_concat(obs: dict[str, Any], size: int = 256) -> np.ndarray:
    """Return front|wrist concat in RGB (matches training / cosmos decode)."""
    front = resize_rgb(np.asarray(obs["front"]), size)
    wrist = resize_rgb(np.asarray(obs["wrist"]), size)
    return np.concatenate([front, wrist], axis=1)


def write_concat_mp4(path: Path, frame_rgb: np.ndarray, fps: int = 30, n_frames: int = 8) -> None:
    """Write RGB frame(s) to mp4. Converts to BGR for OpenCV VideoWriter."""
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frame_rgb.shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    for _ in range(n_frames):
        writer.write(bgr)
    writer.release()


def write_frames_mp4(path: Path, frames_rgb: list[np.ndarray], fps: int = 30) -> None:
    """Write a list of HxWx3 RGB uint8 frames to mp4 (BGR for VideoWriter)."""
    if not frames_rgb:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames_rgb[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (w, h),
    )
    for fr in frames_rgb:
        if fr.shape[0] != h or fr.shape[1] != w:
            fr = cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA)
        writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"[video] wrote {path}  frames={len(frames_rgb)}  {w}x{h}@{fps}fps")


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


class PolicyBackend:
    name: str = "base"

    def infer_chunk(self, obs: dict[str, Any], out_actions_path: Path) -> np.ndarray:
        raise NotImplementedError


class ZerosPolicy(PolicyBackend):
    name = "zeros"

    def __init__(self, chunk: int = 32):
        self.chunk = chunk

    def infer_chunk(self, obs: dict[str, Any], out_actions_path: Path) -> np.ndarray:
        state = np.asarray(obs["state"], dtype=np.float64)
        actions = np.repeat(state[None, :], self.chunk, axis=0)
        out_actions_path.write_text(json.dumps(actions.tolist()) + "\n")
        return actions


class CosmosWAMPolicy(PolicyBackend):
    """Cosmos Action WAM — warm resident worker by default (cold CLI optional)."""

    name = "cosmos"

    def __init__(
        self,
        export_dir: Path,
        framework_python: Path,
        chunk: int = 32,
        image_size: int = 256,
        fps: int = 30,
        prompt: str = DEFAULT_PROMPT,
        stats_path: Path = DEFAULT_STATS,
        work_dir: Path | None = None,
        warm: bool = True,
    ):
        self.export_dir = export_dir
        self.framework_python = framework_python
        self.chunk = chunk
        self.image_size = image_size
        self.fps = fps
        self.prompt = prompt
        self.mean, self.std = load_meanstd(stats_path)
        self.work_dir = work_dir or (_LAB / "outputs" / "rollout_cosmos_tmp")
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.warm = warm
        self._n = 0
        self._proc: subprocess.Popen[str] | None = None
        self._worker_script = _LAB / "scripts" / "cosmos_wam_worker.py"

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if proc.stdin and proc.poll() is None:
                proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                proc.stdin.flush()
                proc.wait(timeout=30)
        except Exception as e:  # noqa: BLE001
            print(f"[policy:cosmos] worker shutdown: {e}")
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass

    def _ensure_worker(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        if not self._worker_script.is_file():
            raise FileNotFoundError(f"warm worker missing: {self._worker_script}")
        scratch = self.work_dir / "warm_scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(self.framework_python),
            "-u",
            str(self._worker_script),
            "--checkpoint-path",
            str(self.export_dir),
            "--scratch-dir",
            str(scratch),
            "--resolution",
            str(self.image_size),
            "--fps",
            str(self.fps),
        ]
        print(f"[policy:cosmos] starting warm worker (load once):\n  {' '.join(cmd)}")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # inherit → user sees load progress
            text=True,
            bufsize=1,
            env=env,
        )
        assert self._proc.stdout is not None
        # Wait until ready (or error). Model load can take minutes.
        while True:
            line = self._proc.stdout.readline()
            if not line:
                code = self._proc.poll()
                raise RuntimeError(f"warm worker exited before ready (code={code})")
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                # Non-protocol noise should not appear on stdout; ignore if any.
                print(f"[policy:cosmos] worker stdout (ignored): {line.rstrip()}")
                continue
            if msg.get("event") == "ready":
                print("[policy:cosmos] warm worker ready")
                return
            if msg.get("event") == "error" or msg.get("ok") is False:
                raise RuntimeError(f"warm worker failed to start: {msg}")

    def _infer_cold(self, sample_path: Path, out_dir: Path) -> Path:
        cmd = [
            str(self.framework_python),
            "-m",
            "cosmos_framework.scripts.inference",
            "--checkpoint-path",
            str(self.export_dir),
            "-i",
            str(sample_path),
            "-o",
            str(out_dir),
            "--parallelism-preset",
            "latency",
            "--no-guardrails",
            "--resolution",
            str(self.image_size),
            "--fps",
            str(self.fps),
        ]
        print(f"[policy:cosmos:cold] {' '.join(cmd)}")
        subprocess.run(cmd, check=True, env=os.environ.copy())
        so = next(out_dir.rglob("sample_outputs.json"), None)
        if so is None:
            raise FileNotFoundError(f"no sample_outputs.json under {out_dir}")
        return so

    def _infer_warm(self, sample_path: Path, out_dir: Path) -> Path:
        self._ensure_worker()
        assert self._proc is not None and self._proc.stdin and self._proc.stdout
        rid = f"{self._n}"
        req = {
            "cmd": "infer",
            "id": rid,
            "input": str(sample_path),
            "output_dir": str(out_dir),
        }
        print(f"[policy:cosmos:warm] infer id={rid} input={sample_path}")
        self._proc.stdin.write(json.dumps(req) + "\n")
        self._proc.stdin.flush()
        while True:
            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError("warm worker died during infer")
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                print(f"[policy:cosmos] worker stdout (ignored): {line.rstrip()}")
                continue
            if msg.get("event") != "done" or str(msg.get("id")) != rid:
                print(f"[policy:cosmos] unexpected worker msg: {msg}")
                continue
            if not msg.get("ok"):
                raise RuntimeError(f"warm infer failed: {msg.get('error')}")
            return Path(msg["sample_outputs"])

    def infer_chunk(self, obs: dict[str, Any], out_actions_path: Path) -> np.ndarray:
        self._n += 1
        step_dir = self.work_dir / f"step_{self._n:04d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        concat = front_wrist_concat(obs, self.image_size)
        # cosmos / opencv: write BGR as-is (matches prior offline path using cv2)
        vision = step_dir / "concat_obs.mp4"
        write_concat_mp4(vision, concat, fps=self.fps, n_frames=max(8, self.chunk // 4))
        sample = {
            "domain_name": "so101_follower",
            "view_point": "concat_view",
            "action_chunk_size": self.chunk,
            "fps": self.fps,
            "image_size": self.image_size,
            "resolution": str(self.image_size),
            "seed": 0,
            "prompt": self.prompt,
            "vision_path": "concat_obs.mp4",
            "model_mode": "wam",
            "name": f"live_{self._n:04d}",
        }
        sample_path = step_dir / f"wam_live_{self._n:04d}.json"
        sample_path.write_text(json.dumps(sample, indent=2) + "\n")
        out_dir = step_dir / "wam_out"
        out_dir.mkdir(exist_ok=True)

        if self.warm:
            so = self._infer_warm(sample_path, out_dir)
        else:
            so = self._infer_cold(sample_path, out_dir)

        pred_n = extract_wam_action(so)
        pred = denorm_actions(pred_n, self.mean, self.std)
        if len(pred) > self.chunk:
            pred = pred[: self.chunk]
        out_actions_path.write_text(json.dumps(pred.tolist()) + "\n")
        return pred


class Pi0PolicyBackend(PolicyBackend):
    name = "pi0"

    def __init__(
        self,
        ckpt: Path,
        prompt: str = DEFAULT_PROMPT,
        device: str = "cuda",
        chunk: int = 50,
    ):
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy

        self.torch = torch
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.prompt = prompt
        self.chunk = chunk
        print(f"[policy:pi0] loading {ckpt} on {self.device}")
        self.model = PI0Policy.from_pretrained(str(ckpt))
        self.model.to(self.device)
        self.model.eval()
        self.preprocess, self.postprocess = make_pre_post_processors(
            self.model.config,
            str(ckpt),
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )

    def infer_chunk(self, obs: dict[str, Any], out_actions_path: Path) -> np.ndarray:
        import torch
        from lerobot.policies.utils import build_inference_frame

        # Build dataset-style observation expected by this stack3cam ckpt.
        hw_obs = {
            "observation.state": np.asarray(obs["state"], dtype=np.float32),
            "observation.images.front": np.asarray(obs["front"]),
            "observation.images.wrist": np.asarray(obs["wrist"]),
        }
        if "side" in obs:
            hw_obs["observation.images.side"] = np.asarray(obs["side"])
        else:
            # Fallback: duplicate wrist resized — better to provide a real side cam.
            hw_obs["observation.images.side"] = np.asarray(obs["wrist"])

        # Prefer official helper when features known; else manual batch.
        try:
            from lerobot.datasets.utils import hw_to_dataset_features

            # Minimal feature map for stack3cam pi0
            action_features = {f"action.{j}": float for j in JOINTS}
            # build_inference_frame path varies by lerobot version — fall through
            _ = action_features
            frame = {
                "observation.state": torch.as_tensor(hw_obs["observation.state"], dtype=torch.float32),
                "observation.images.front": _image_to_torch(hw_obs["observation.images.front"]),
                "observation.images.wrist": _image_to_torch(hw_obs["observation.images.wrist"]),
                "observation.images.side": _image_to_torch(hw_obs["observation.images.side"]),
                "task": self.prompt,
            }
        except Exception:
            frame = {
                "observation.state": torch.as_tensor(hw_obs["observation.state"], dtype=torch.float32),
                "observation.images.front": _image_to_torch(hw_obs["observation.images.front"]),
                "observation.images.wrist": _image_to_torch(hw_obs["observation.images.wrist"]),
                "observation.images.side": _image_to_torch(hw_obs["observation.images.side"]),
                "task": self.prompt,
            }

        # batch dim
        batch = {
            k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else v) for k, v in frame.items()
        }
        batch = self.preprocess(batch)
        with torch.inference_mode():
            if hasattr(self.model, "predict_action_chunk"):
                actions = self.model.predict_action_chunk(batch)
            else:
                # select_action repeatedly
                acts = []
                for _ in range(self.chunk):
                    a = self.model.select_action(batch)
                    acts.append(a)
                actions = torch.stack(acts, dim=1)
            actions = self.postprocess(actions)

        if isinstance(actions, dict):
            # flatten joint dicts if needed
            arr = np.stack(
                [np.asarray(actions[f"{j}.pos"] if f"{j}.pos" in actions else actions.get(j)) for j in JOINTS],
                axis=-1,
            )
        else:
            arr = actions.detach().cpu().numpy()
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim == 3:
            arr = arr[0]
        if arr.ndim == 1:
            arr = arr[None, :]
        out_actions_path.write_text(json.dumps(arr.tolist()) + "\n")
        return arr


def _image_to_torch(img: np.ndarray):
    import torch

    x = np.asarray(img)
    if x.dtype != np.uint8:
        x = np.clip(x, 0, 255).astype(np.uint8)
    # HWC -> CHW float 0-1
    t = torch.from_numpy(x).permute(2, 0, 1).float() / 255.0
    return t


# ---------------------------------------------------------------------------
# Closed loop
# ---------------------------------------------------------------------------


@dataclass
class LoopConfig:
    n_trials: int
    timeout_s: float
    control_hz: float
    layout_cycle: int
    interactive: bool
    execute_steps: int  # how many actions of each chunk to execute before replan
    max_chunks: int


def run_trial(
    *,
    robot: RobotBackend,
    policy: PolicyBackend,
    logger: TrialLogger,
    trial_idx: int,
    cfg: LoopConfig,
    prompt: str,
    home: list[float],
    video_dir: Path,
    mode: str,
) -> TrialRecord:
    layout = (trial_idx - 1) % cfg.layout_cycle
    trial_id = now_trial_id(f"{policy.name}_t{trial_idx:03d}")
    print(f"\n>>> trial {trial_idx} id={trial_id} layout={layout} policy={policy.name}")

    robot.home(home)
    if cfg.interactive:
        input(f"  Scene ready for layout={layout}? Enter to continue / Ctrl-C abort: ")

    video_path = video_dir / f"{trial_id}_eval.mp4"
    # Eval video: front|wrist concat recorded at each control step during execute.
    eval_frames: list[np.ndarray] = []
    eval_size = 256  # match policy view; easy to eyeball

    def _record_obs(o: dict[str, Any]) -> None:
        try:
            eval_frames.append(front_wrist_concat(o, eval_size))
        except Exception as e:  # noqa: BLE001 — never abort motion for video
            print(f"[video] skip frame: {e}")

    t0 = time.time()
    success: bool | None = None
    failure: str | None = None
    n_chunks = 0
    last_actions_path: str | None = None

    try:
        while True:
            if time.time() - t0 > cfg.timeout_s:
                success = False
                failure = "timeout"
                break
            if n_chunks >= cfg.max_chunks:
                break

            obs = robot.get_obs()
            _record_obs(obs)

            actions_path = logger.out_dir / f"{trial_id}_chunk{n_chunks:02d}_actions.json"
            actions = policy.infer_chunk(obs, actions_path)
            last_actions_path = str(actions_path)
            n_exec = min(cfg.execute_steps, len(actions))
            robot.execute_chunk(actions[:n_exec], cfg.control_hz, on_step=_record_obs)
            n_chunks += 1

            if cfg.interactive:
                ans = input("  continue chunk / mark [c=continue, y=success, n=fail, q=stop trial]: ").strip().lower()
                if ans == "y":
                    success = True
                    break
                if ans == "n":
                    success = False
                    failure = input("  failure_code: ").strip() or "other"
                    break
                if ans == "q":
                    break
                # else continue
    except KeyboardInterrupt:
        print("  interrupted")
        failure = failure or "estop"

    write_frames_mp4(video_path, eval_frames, fps=int(cfg.control_hz))

    dur = time.time() - t0
    if success is None and cfg.interactive:
        ans = input("  Final label success? [y/n/skip]: ").strip().lower()
        if ans in ("y", "yes"):
            success = True
        elif ans in ("n", "no"):
            success = False
            failure = failure or (input("  failure_code: ").strip() or "other")

    rec = TrialRecord(
        trial_id=trial_id,
        policy=policy.name,
        mode=mode,
        prompt=prompt,
        layout_id=str(layout),
        success=success,
        failure_code=failure,
        duration_s=dur,
        notes=f"chunks={n_chunks} eval_frames={len(eval_frames)}",
        video_path=str(video_path),
        action_raw_path=last_actions_path,
        meta={"n_chunks": n_chunks, "timeout_s": cfg.timeout_s, "eval_frames": len(eval_frames)},
    )
    logger.append(rec)
    return rec


def build_policy(args: argparse.Namespace, work_dir: Path) -> PolicyBackend:
    if args.policy in ("zeros", "hold"):
        return ZerosPolicy(chunk=args.chunk)
    if args.policy in ("cosmos", "cosmos_action_2000"):
        fw_py = Path(args.framework_python)
        return CosmosWAMPolicy(
            export_dir=Path(args.export),
            framework_python=fw_py,
            chunk=args.chunk,
            prompt=args.prompt,
            stats_path=Path(args.stats_path),
            work_dir=work_dir / "cosmos_steps",
            warm=bool(args.cosmos_warm),
        )
    if args.policy in ("pi0", "pi0_80k"):
        return Pi0PolicyBackend(
            ckpt=Path(args.pi0_ckpt),
            prompt=args.prompt,
            device=args.device,
            chunk=args.chunk,
        )
    raise SystemExit(f"unknown --policy {args.policy}")


def build_robot(args: argparse.Namespace) -> RobotBackend:
    if args.dry_run:
        return MockRobot(image_size=args.image_size)
    return SO101Robot(
        port=args.port,
        robot_id=args.robot_id,
        cam_front=_parse_cam(args.cam_front),
        cam_wrist=_parse_cam(args.cam_wrist),
        cam_side=_parse_cam(args.cam_side) if args.cam_side != "" else None,
        use_degrees=True,
        max_relative_target=args.max_relative_target,
        calibrate=args.calibrate,
    )


def _parse_cam(v: str) -> int | str:
    return int(v) if v.isdigit() else v


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="Mock robot; no serial I/O")
    p.add_argument("--force-real", action="store_true", help="Require live SO-101 hardware")
    p.add_argument("--policy", default="zeros", choices=["zeros", "hold", "cosmos", "cosmos_action_2000", "pi0", "pi0_80k"])
    p.add_argument("--n-trials", type=int, default=1)
    p.add_argument("--timeout-s", type=float, default=90.0)
    p.add_argument("--control-hz", type=float, default=30.0)
    p.add_argument("--chunk", type=int, default=32)
    p.add_argument("--execute-steps", type=int, default=16, help="Steps of each chunk to execute before replan")
    p.add_argument("--max-chunks", type=int, default=20)
    p.add_argument("--layout-cycle", type=int, default=5)
    p.add_argument("--interactive", action="store_true", default=False)
    p.add_argument("--no-interactive", action="store_true")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--port", default=os.environ.get("SO101_PORT", "/dev/ttyUSB0"))
    p.add_argument("--robot-id", default=os.environ.get("SO101_ID", "so101_follower"))
    p.add_argument("--cam-front", default=os.environ.get("CAM_FRONT", "0"))
    p.add_argument("--cam-wrist", default=os.environ.get("CAM_WRIST", "1"))
    # Empty = do not open side (Cosmos action-policy is front|wrist only).
    p.add_argument("--cam-side", default=os.environ.get("CAM_SIDE", ""))
    p.add_argument("--max-relative-target", type=float, default=15.0)
    p.add_argument("--calibrate", action="store_true")
    p.add_argument(
        "--export",
        default=str(_LAB / "outputs/export/action_stack3cam_action_policy_edge_2000"),
    )
    p.add_argument(
        "--pi0-ckpt",
        default=str(
            Path(
                "/home/july/lerobot_alohamini/outputs/train/pi0_stack_white_blue_black_3cam/"
                "checkpoints/080000/pretrained_model"
            )
        ),
    )
    p.add_argument(
        "--framework-python",
        default=str(Path(os.environ.get("COSMOS_FRAMEWORK_ROOT", "/home/july/cosmos-framework")) / ".venv/bin/python"),
    )
    p.add_argument("--stats-path", default=str(DEFAULT_STATS))
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--cosmos-warm",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("COSMOS_WARM", "1") not in ("0", "false", "False"),
        help="Keep Cosmos weights resident across chunks (default on). Use --no-cosmos-warm for cold CLI.",
    )
    p.add_argument(
        "--lerobot-src",
        default=os.environ.get("LEROBOT_SRC", "/home/july/lerobot_alohamini/src"),
        help="Prepend to sys.path for SO101Follower / PI0Policy",
    )
    p.add_argument("--home-json", type=Path, default=None, help="Optional 6D home pose JSON")
    args = p.parse_args()

    if args.no_interactive:
        args.interactive = False

    if args.force_real and args.dry_run:
        raise SystemExit("use either --dry-run or --force-real, not both")

    if args.lerobot_src:
        src = str(Path(args.lerobot_src))
        if src not in sys.path:
            sys.path.insert(0, src)

    probe = _probe_hardware(args.port)
    print("[hw-probe]", json.dumps(probe))

    if args.force_real:
        if not probe["ready"]:
            print(
                "ERROR: --force-real but SO-101 hardware not ready on this machine.\n"
                f"  port={args.port} exists={probe['port_exists']}\n"
                f"  videos={probe['videos']}\n"
                f"  serials={probe['serials']}\n"
                "This looks like a GPU training host. Run closed-loop SR on the bench PC "
                "with arm + front/wrist cameras attached, then re-run with --force-real.",
                file=sys.stderr,
            )
            raise SystemExit(3)
    elif not args.dry_run:
        # default safety: require explicit mode
        print("ERROR: specify --dry-run (mock) or --force-real (live SO-101).", file=sys.stderr)
        raise SystemExit(2)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (_LAB / "outputs" / "rollout_real" / f"{args.policy}_{stamp}")
    out_dir = Path(out_dir)
    video_dir = out_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    home = list(DEFAULT_HOME)
    if args.home_json and args.home_json.exists():
        home = list(map(float, json.loads(args.home_json.read_text())))

    robot = build_robot(args)
    policy = build_policy(args, out_dir)
    logger = TrialLogger(out_dir)
    mode = "dry_run" if args.dry_run else "real"
    if isinstance(policy, CosmosWAMPolicy):
        print(f"[policy:cosmos] warm={policy.warm}")

    loop = LoopConfig(
        n_trials=args.n_trials,
        timeout_s=args.timeout_s,
        control_hz=args.control_hz,
        layout_cycle=args.layout_cycle,
        interactive=args.interactive,
        execute_steps=args.execute_steps,
        max_chunks=args.max_chunks if not args.dry_run else min(args.max_chunks, 2),
    )

    robot.connect()
    try:
        for i in range(1, args.n_trials + 1):
            run_trial(
                robot=robot,
                policy=policy,
                logger=logger,
                trial_idx=i,
                cfg=loop,
                prompt=args.prompt,
                home=home,
                video_dir=video_dir,
                mode=mode,
            )
    finally:
        robot.disconnect()
        if hasattr(policy, "close"):
            policy.close()  # type: ignore[attr-defined]

    summary = logger.summarize()
    (out_dir / "hw_probe.json").write_text(json.dumps(probe, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f">>> trials:  {logger.path}")
    print(f">>> summary: {out_dir / 'summary.json'}")
    if args.dry_run:
        print("NOTE: dry-run mock robot — success rate is not real SO-101 SR.")
    if isinstance(policy, CosmosWAMPolicy) and policy.warm:
        print("NOTE: Cosmos used warm resident worker (load once, replan per chunk).")


if __name__ == "__main__":
    main()
