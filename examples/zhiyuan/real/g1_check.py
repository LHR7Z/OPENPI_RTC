"""Read-only G1 check for the pi05 policy (python 3.10): sends no commands and loads no model.

Checks that all 3 cameras deliver fresh frames of the training resolution, prints the 20-dim state in training
units next to the training range, and saves one frame per camera interpreted as RGB. Open the PNGs: if red and
blue look swapped, the SDK delivers BGR for this robot mode -> pass --swap-rb to g1_check.py and g1_run.py.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import cv2
from g1_robot import CAMERAS
from g1_robot import STATE_NAMES
from g1_robot import TRAIN_IMAGE_SHAPES
from g1_robot import G1Robot
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--norm-stats", default="/checkpoint/assets/group1/task15_train/norm_stats.json")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--swap-rb", action="store_true")
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("/work/my_assets/real_check"))
    args = parser.parse_args()
    args.out = args.out.resolve()  # G1Robot changes the cwd

    stats = json.loads(pathlib.Path(args.norm_stats).read_text())["norm_stats"]["state"]
    mean, q01, q99 = (np.asarray(stats[k]) for k in ("mean", "q01", "q99"))

    robot = G1Robot(swap_rb=args.swap_rb)
    ok = True
    try:
        fresh = dict.fromkeys(CAMERAS.values(), 0)
        last_stamps: dict = {}
        obs = None
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            new = robot.observe()
            if new is not None:
                obs = new
                for cam, stamp in obs["stamps"].items():
                    fresh[cam] += int(cam in last_stamps and stamp != last_stamps[cam])
                last_stamps = obs["stamps"]
            time.sleep(0.1)
        if obs is None:
            print("No complete observation during the check window")
            return 1

        print("\n[cameras]")
        args.out.mkdir(parents=True, exist_ok=True)
        for key, cam in CAMERAS.items():
            frame = obs["images"][key]
            shape_ok = tuple(frame.shape[:2]) == TRAIN_IMAGE_SHAPES[cam]
            ok &= shape_ok and fresh[cam] >= 3
            path = args.out / f"{cam}.png"
            cv2.imwrite(str(path), np.ascontiguousarray(frame[..., ::-1]))  # frame treated as RGB; cv2 writes BGR
            print(
                f"  {cam:10} -> {key:17} shape {frame.shape} {frame.dtype} "
                f"(train {TRAIN_IMAGE_SHAPES[cam]}: {'OK' if shape_ok else 'MISMATCH'}), "
                f"fresh updates {fresh[cam]}, mean RGB {frame.reshape(-1, 3).mean(0).round(1)}, saved {path}"
            )

        print("\n[state, training units]   live      train_mean   train_q01   train_q99")
        state = obs["state"]
        for i, name in enumerate(STATE_NAMES):
            span = q99[i] - q01[i]
            margin = 0.1 * span + 0.02
            inside = q01[i] - margin <= state[i] <= q99[i] + margin
            ok &= bool(inside)
            print(
                f"  {i:2d} {name:14} {state[i]:10.4f} {mean[i]:12.4f} {q01[i]:11.4f} {q99[i]:11.4f}"
                f"{'' if inside else '   <-- outside training range'}"
            )
    finally:
        robot.close()

    print("\nCHECK_OK" if ok else "\nCHECK_FAILED: fix the flagged items (pose, units or camera mode) before running")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
