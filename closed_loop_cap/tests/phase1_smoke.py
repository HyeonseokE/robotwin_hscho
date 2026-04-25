"""Phase 1 smoke test — exercises the env wrapper without any VLM.

Run from repo root:
    python -m closed_loop_cap.tests.phase1_smoke --task shake_bottle --seed 0

Success criteria:
    1. make_env returns an EnvHandle with task_env.plan_success initially True.
    2. capture_rgb returns a uint8 HxWx3 array, saved to disk for eyeball.
    3. snapshot_robot_state returns two 7-D poses + two gripper scalars.
    4. close_env runs without raising.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import imageio
import numpy as np
import yaml

# Make repo root importable when run as a module from elsewhere.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from closed_loop_cap.env import (
    capture_rgb,
    close_env,
    is_task_success,
    make_env,
    snapshot_robot_state,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="shake_bottle")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "closed_loop_cap" / "configs" / "default.yaml"),
    )
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "closed_loop_cap" / "output" / "_phase1_smoke"),
    )
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Retry a few seeds — some tasks (shake_bottle etc.) reject unstable
    # initial object placements via UnStableError. That's not an install bug.
    handle = None
    last_err = None
    for s in range(args.seed, args.seed + 30):
        try:
            handle = make_env(args.task, s, config)
            if s != args.seed:
                print(f"[smoke] first stable seed found at s={s}")
            break
        except Exception as exc:
            last_err = exc
            msg = str(exc)
            if "UnStableError" in type(exc).__name__ or "unstable" in msg.lower():
                print(f"[smoke] seed={s} unstable, trying next")
                continue
            raise
    if handle is None:
        raise SystemExit(f"[smoke] no stable seed found; last error: {last_err}")
    try:
        rgb = capture_rgb(handle)
        assert rgb.dtype == np.uint8, f"expected uint8, got {rgb.dtype}"
        assert rgb.ndim == 3 and rgb.shape[2] == 3, f"unexpected shape {rgb.shape}"
        img_path = out_dir / f"{args.task}_seed{args.seed}_initial.png"
        imageio.imwrite(img_path, rgb)
        print(f"[smoke] saved initial RGB to {img_path} (shape={rgb.shape})")

        state = snapshot_robot_state(handle)
        assert state.left_ee_pose.shape == (7,), state.left_ee_pose.shape
        assert state.right_ee_pose.shape == (7,), state.right_ee_pose.shape
        print(
            f"[smoke] robot state: "
            f"L_ee={state.left_ee_pose[:3].round(3)} L_grip={state.left_gripper:.3f} "
            f"R_ee={state.right_ee_pose[:3].round(3)} R_grip={state.right_gripper:.3f}"
        )

        # check_success is expected to be False on a fresh scene — we just want it callable.
        success = is_task_success(handle)
        print(f"[smoke] is_task_success (pre-execution) = {success}")
    finally:
        close_env(handle)
        print("[smoke] close_env done")

    print("[smoke] PHASE 1 OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
