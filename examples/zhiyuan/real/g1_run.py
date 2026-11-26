"""Run the pi05 policy on the G1 (python 3.10 side). DRY RUN unless --execute is given.

Synchronous chunking: observe -> get a (50, 22) chunk from g1_policy_server.py -> execute the first --exec-steps
actions at --hz -> observe again. Only the 14 arm joints and 2 grippers are commanded; the right arm is held at its
starting pose by default, head/waist/base are never commanded. Each control tick moves a joint by at most
--max-joint-step rad. Any failed safety check stops sending commands immediately (the arm holds its last target).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

from common import PROMPT
from common import SOCKET
from common import PolicyClient
from g1_robot import G1Robot
from g1_robot import ReplayRobot
import numpy as np

LEFT_ARM = slice(0, 7)
RIGHT_ARM = slice(7, 14)


class SafetyStopError(Exception):
    pass


def check_chunk(actions: np.ndarray, state: np.ndarray, steps: int, lo: np.ndarray, hi: np.ndarray, args) -> float:
    start_gap = float(np.abs(actions[0, LEFT_ARM] - state[LEFT_ARM]).max())
    if start_gap > args.max_start_gap:
        raise SafetyStopError(f"first action is {start_gap:.3f} rad from the current left arm (> {args.max_start_gap})")
    targets = actions[:steps, LEFT_ARM]
    out = (targets < lo - args.range_margin) | (targets > hi + args.range_margin)
    if out.any():
        step, joint = np.argwhere(out)[0]
        raise SafetyStopError(
            f"left arm joint {joint} target {targets[step, joint]:.3f} at step {step} is outside the "
            f"training range [{lo[joint]:.3f}, {hi[joint]:.3f}] +- {args.range_margin}"
        )
    return start_gap


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--execute", action="store_true", help="Actually send commands (default: dry run)")
    parser.add_argument("--yes", action="store_true", help="Skip the interactive confirmation in --execute mode")
    parser.add_argument("--exec-steps", type=int, default=25, help="Actions executed per chunk before re-observing")
    parser.add_argument("--hz", type=float, default=30.0, help="Command rate (training data is 30 fps)")
    parser.add_argument("--max-joint-step", type=float, default=0.03, help="Max rad per joint per control tick")
    parser.add_argument("--max-start-gap", type=float, default=0.2, help="Max rad between chunk[0] and current arm")
    parser.add_argument("--range-margin", type=float, default=0.1, help="Rad beyond the training joint range")
    parser.add_argument("--max-tracking-error", type=float, default=0.25, help="Max rad commanded-vs-measured")
    parser.add_argument("--move-right-arm", action="store_true", help="Follow the policy for the right arm too")
    parser.add_argument("--max-chunks", type=int, default=0, help="Stop after N chunks (0 = until Ctrl+C)")
    parser.add_argument("--swap-rb", action="store_true", help="Camera frames are BGR (see g1_check.py)")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--norm-stats", default="/checkpoint/assets/group1/task15_train/norm_stats.json")
    parser.add_argument("--socket", default=SOCKET)
    parser.add_argument("--log-dir", type=pathlib.Path, default=pathlib.Path("/work/my_assets/real_runs"))
    parser.add_argument("--replay-npz", help="Offline test: use a recorded NPZ observation instead of the robot")
    args = parser.parse_args()
    # G1Robot changes the cwd, so pin paths first.
    args.log_dir, args.socket = args.log_dir.resolve(), os.path.abspath(args.socket)
    if not 1 <= args.exec_steps <= 50:
        parser.error("--exec-steps must be in [1, 50]")

    stats = json.loads(pathlib.Path(args.norm_stats).read_text())["norm_stats"]["state"]
    lo, hi = np.asarray(stats["q01"])[LEFT_ARM], np.asarray(stats["q99"])[LEFT_ARM]

    mode = "EXECUTE" if args.execute else "DRY RUN (no commands sent)"
    print(f"Mode: {mode}; exec {args.exec_steps} steps/chunk at {args.hz} Hz, max {args.max_joint_step} rad/tick")
    if args.execute and not args.yes and input("Robot may move immediately. E-stop ready? Type GO: ").strip() != "GO":
        print("Aborted")
        return 1

    robot = ReplayRobot(args.replay_npz) if args.replay_npz else G1Robot(swap_rb=args.swap_rb)
    client = PolicyClient(args.socket)
    run_dir = args.log_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{'exec' if args.execute else 'dry'}_{os.getpid()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Policy server: {client.metadata}; logging to {run_dir}")

    exit_code = 0
    last_cmd = None
    hold_right = None
    chunk_idx = 0
    try:
        while args.max_chunks == 0 or chunk_idx < args.max_chunks:
            obs = robot.observe()
            if obs is None:
                raise SafetyStopError("lost observation (camera or joint states)")
            state = obs["state"]
            if hold_right is None:
                hold_right = state[RIGHT_ARM].copy()
            if last_cmd is not None:
                tracking = float(np.abs(last_cmd - state[:14]).max())
                if tracking > args.max_tracking_error:
                    raise SafetyStopError(
                        f"arm is {tracking:.3f} rad behind the last command (blocked or not following)"
                    )

            actions, infer_ms = client.infer(obs["images"], state, args.prompt)
            start_gap = check_chunk(actions, state, args.exec_steps, lo, hi, args)
            arm_targets = actions[: args.exec_steps, :14].copy()
            if not args.move_right_arm:
                arm_targets[:, RIGHT_ARM] = hold_right
            grippers = np.clip(actions[: args.exec_steps, 14:16], 0.0, 1.0)

            sent = []
            if args.execute:
                if last_cmd is None:
                    last_cmd = state[:14].astype(np.float64)
                next_tick = time.monotonic()
                for i in range(args.exec_steps):
                    last_cmd = last_cmd + np.clip(arm_targets[i] - last_cmd, -args.max_joint_step, args.max_joint_step)
                    robot.command(last_cmd, grippers[i])
                    sent.append(np.concatenate([last_cmd, grippers[i]]))
                    next_tick += 1.0 / args.hz
                    time.sleep(max(0.0, next_tick - time.monotonic()))
            np.savez(
                run_dir / f"chunk_{chunk_idx:04d}.npz",
                state=state,
                actions=actions,
                infer_ms=infer_ms,
                sent=np.asarray(sent),
                executed=args.execute,
            )
            reach = float(np.abs(arm_targets[-1, LEFT_ARM] - state[LEFT_ARM]).max())
            print(
                f"chunk {chunk_idx:4d}: infer {infer_ms:6.0f} ms | start gap {start_gap:.3f} rad | left arm moves "
                f"up to {reach:.3f} rad over {args.exec_steps} steps | left gripper "
                f"{grippers[0, 0]:.2f}->{grippers[-1, 0]:.2f}{'' if args.execute else ' | (dry run)'}",
                flush=True,
            )
            chunk_idx += 1
    except SafetyStopError as e:
        print(f"SAFETY STOP: {e}. No further commands sent.")
        exit_code = 2
    except KeyboardInterrupt:
        print("Stopped by user. No further commands sent.")
    finally:
        client.close()
        robot.close()
    print(f"{chunk_idx} chunks, logs in {run_dir}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
