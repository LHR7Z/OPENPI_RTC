"""Offline Real-Time Chunking (RTC) check on the G1 dataset, without a robot.

Simulates one chunk switch per sample on recorded episodes:
  - chunk A is generated from the observation at t - s and has executed s steps by time t;
  - chunk B is generated from the observation at t; while it is computed (d steps), A keeps executing;
  - at t + d the robot switches from A[s + d - 1] to B[d].
Plain chunking ignores A when generating B; RTC guides B toward A[s:]. Both use the same noise (paired).

Metrics on the left arm (dims 0-6, rad) and left gripper (dim 14):
  - switch_jump:  |B[d] - A[s+d-1]|, the step taken at the switch; compare with gt_step (typical step in the data)
  - disagreement: |B[d] - A[s+d]|, how much the two chunks disagree at the switch time
  - mae_exec:     |B[d:d+s] - ground truth|, accuracy over the window B executes before the next switch
The dataset has no held-out episodes, so these are training-set episodes.
"""

import argparse
import collections
import json
from pathlib import Path
import time

import numpy as np

from examples.zhiyuan import infer_g1
from examples.zhiyuan.eval_offline import episode_actions
from examples.zhiyuan.eval_offline import load_frame

DIMS = {"left_arm": list(range(7)), "left_gripper": [14]}


def gt_at(gt_all: np.ndarray, t: int, offset) -> np.ndarray:
    return gt_all[np.clip(t + offset, 0, len(gt_all) - 1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--asset", default="group1/task15_train")
    parser.add_argument("--repo-id", default="group1/task15_train")
    parser.add_argument(
        "--exec-horizon", type=int, default=25, help="s: steps of a chunk executed before the next inference starts"
    )
    parser.add_argument("--delays", type=int, nargs="+", default=[5, 10], help="d values (inference delay in steps)")
    parser.add_argument("--max-guidance-weight", type=float, default=5.0)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.3, 0.6])
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    s, horizon = args.exec_horizon, infer_g1.ACTION_HORIZON
    if max(args.delays) > horizon - s:
        raise ValueError(f"Each delay must be <= horizon - exec_horizon = {horizon - s}")

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(args.repo_id)
    prompt = ds.meta.tasks[0]
    policy, stats = infer_g1.load_policy(args.checkpoint, args.asset)

    variants = ["plain"] + [f"rtc_d{d}" for d in args.delays]
    records = collections.defaultdict(list)  # (variant, delay) -> list of per-sample metric dicts
    latency = {"plain": [], "rtc": []}
    num_eps = ds.meta.total_episodes if args.max_episodes is None else min(args.max_episodes, ds.meta.total_episodes)
    sample_id = 0
    for ep in range(num_eps):
        gt_all = episode_actions(ds, ep)
        start, length = int(ds.episode_data_index["from"][ep]), len(gt_all)
        for frac in args.fractions:
            t = max(int(frac * length), s)
            rng = np.random.default_rng(args.seed + sample_id)
            noise_a, noise_b = (rng.standard_normal((horizon, 32)).astype(np.float32) for _ in range(2))
            sample_id += 1

            images, state = load_frame(ds, start + t - s)
            chunk_a = infer_g1.predict(policy, stats, images, state, prompt, noise=noise_a)
            images, state = load_frame(ds, start + t)
            t0 = time.perf_counter()
            plain_b = infer_g1.predict(policy, stats, images, state, prompt, noise=noise_b)
            latency["plain"].append(time.perf_counter() - t0)
            chunks_b = {"plain": dict.fromkeys(args.delays, plain_b)}
            for d in args.delays:
                t0 = time.perf_counter()
                chunks_b.setdefault(f"rtc_d{d}", {})[d] = infer_g1.predict(
                    policy,
                    stats,
                    images,
                    state,
                    prompt,
                    noise=noise_b,
                    prev_actions=chunk_a[s:],
                    inference_delay=d,
                    max_guidance_weight=args.max_guidance_weight,
                )
                latency["rtc"].append(time.perf_counter() - t0)

            for v in variants:
                for d, b in chunks_b[v].items():
                    window = gt_at(gt_all, t, np.arange(d, d + s))
                    step = gt_at(gt_all, t, d) - gt_at(gt_all, t, d - 1)
                    records[v, d].append(
                        {
                            name: {
                                "switch_jump": np.abs(b[d, dims] - chunk_a[s + d - 1, dims]).mean(),
                                "disagreement": np.abs(b[d, dims] - chunk_a[s + d, dims]).mean(),
                                "mae_exec": np.abs(b[d : d + s][:, dims] - window[:, dims]).mean(),
                                "gt_step": np.abs(step[dims]).mean(),
                            }
                            for name, dims in DIMS.items()
                        }
                    )
        print(f"episode {ep:3d} done ({sample_id} samples)", flush=True)

    summary = {
        "checkpoint": str(args.checkpoint),
        "num_samples": sample_id,
        "exec_horizon": s,
        "max_guidance_weight": args.max_guidance_weight,
        "latency_ms": {
            k: {"mean": float(np.mean(v[1:]) * 1000), "first_call": float(v[0] * 1000)} for k, v in latency.items()
        },
        "results": {},
    }
    for d in args.delays:
        for v in ("plain", f"rtc_d{d}"):
            key = f"d={d}/{'rtc' if v != 'plain' else 'plain'}"
            summary["results"][key] = {
                name: {m: float(np.mean([r[name][m] for r in records[v, d]])) for m in records[v, d][0][name]}
                for name in DIMS
            }
    args.out.write_text(json.dumps(summary, indent=2))

    print(
        f"\nlatency (ms, after warmup): plain {summary['latency_ms']['plain']['mean']:.0f}, "
        f"rtc {summary['latency_ms']['rtc']['mean']:.0f}"
    )
    for name in DIMS:
        print(f"\n[{name}] mean over {sample_id} samples")
        print(f"{'':16}{'switch_jump':>14}{'gt_step':>10}{'disagreement':>14}{'mae_exec':>10}")
        for key, res in summary["results"].items():
            r = res[name]
            print(f"{key:16}{r['switch_jump']:14.4f}{r['gt_step']:10.4f}{r['disagreement']:14.4f}{r['mae_exec']:10.4f}")


if __name__ == "__main__":
    main()
