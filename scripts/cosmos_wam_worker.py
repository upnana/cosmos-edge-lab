#!/usr/bin/env python3
"""Warm Cosmos WAM inference worker (model resident).

Runs under cosmos-framework Python. Loads the checkpoint once, then serves
newline-delimited JSON requests on stdin and replies on stdout (logs → stderr).

Protocol
--------
stdout (startup)::
  {"event":"ready","checkpoint":"..."}

stdin request::
  {"cmd":"infer","id":"1","input":"/path/sample.json","output_dir":"/path/out"}
  {"cmd":"shutdown"}

stdout reply::
  {"event":"done","id":"1","ok":true,"sample_outputs":"/path/.../sample_outputs.json"}
  {"event":"done","id":"1","ok":false,"error":"..."}
  {"event":"bye"}

Used by ``so101_rollout_driver.py`` CosmosWAMPolicy warm path.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


def _emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _load_pipe(*, checkpoint: Path, scratch_dir: Path, resolution: str, fps: int):
    from cosmos_framework.inference.common.init import init_output_dir, init_script

    init_script()

    from cosmos_framework.inference.args import OmniSampleOverrides, OmniSetupOverrides
    from cosmos_framework.inference.common.inference import sync_distributed_errors

    scratch_dir.mkdir(parents=True, exist_ok=True)
    setup_ov = OmniSetupOverrides(
        checkpoint_path=str(checkpoint),
        output_dir=scratch_dir,
        guardrails=False,
        parallelism_preset="latency",
        sample_overrides=OmniSampleOverrides(
            resolution=resolution,  # type: ignore[arg-type]
            fps=fps,
        ),
    )
    with sync_distributed_errors():
        setup_args = setup_ov.build_setup()
        init_output_dir(setup_args.output_dir)
    pipe = setup_args.get_inference_cls().create(setup_args)
    return pipe, setup_args


def _run_one(pipe, setup_args, *, input_path: Path, output_dir: Path) -> Path:
    from cosmos_framework.inference.common.inference import sync_distributed_errors

    output_dir.mkdir(parents=True, exist_ok=True)
    with sync_distributed_errors():
        ovs = setup_args.get_sample_overrides_cls().from_files(
            [input_path],
            overrides=setup_args.sample_overrides,
        )
        if len(ovs) != 1:
            raise ValueError(f"expected 1 sample in {input_path}, got {len(ovs)}")
        ov = ovs[0]
        assert ov.name
        ov.output_dir = output_dir / ov.name
        ov.download(ov.output_dir / "inputs")
        sample = ov.build_sample(model_config=pipe.model_config)
    pipe.generate([sample])
    so = ov.output_dir / "sample_outputs.json"
    if not so.is_file():
        found = next(ov.output_dir.rglob("sample_outputs.json"), None)
        if found is None:
            raise FileNotFoundError(f"no sample_outputs.json under {ov.output_dir}")
        so = found
    return so


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-path", type=Path, required=True)
    p.add_argument("--scratch-dir", type=Path, required=True)
    p.add_argument("--resolution", default="256")
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()

    print(
        f"[cosmos_wam_worker] loading once: {args.checkpoint_path}",
        file=sys.stderr,
        flush=True,
    )
    try:
        pipe, setup_args = _load_pipe(
            checkpoint=args.checkpoint_path,
            scratch_dir=args.scratch_dir,
            resolution=str(args.resolution),
            fps=int(args.fps),
        )
    except Exception as e:
        _emit({"event": "error", "ok": False, "error": f"load failed: {e}"})
        traceback.print_exc(file=sys.stderr)
        raise SystemExit(1) from e

    _emit({"event": "ready", "checkpoint": str(args.checkpoint_path)})
    print("[cosmos_wam_worker] ready (warm)", file=sys.stderr, flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            _emit({"event": "done", "id": None, "ok": False, "error": f"bad json: {e}"})
            continue

        cmd = req.get("cmd")
        if cmd == "shutdown":
            _emit({"event": "bye"})
            break
        if cmd != "infer":
            _emit({"event": "done", "id": req.get("id"), "ok": False, "error": f"unknown cmd {cmd}"})
            continue

        rid = req.get("id")
        try:
            so = _run_one(
                pipe,
                setup_args,
                input_path=Path(req["input"]),
                output_dir=Path(req["output_dir"]),
            )
            _emit({"event": "done", "id": rid, "ok": True, "sample_outputs": str(so)})
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            _emit({"event": "done", "id": rid, "ok": False, "error": str(e)})

    print("[cosmos_wam_worker] exit", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
