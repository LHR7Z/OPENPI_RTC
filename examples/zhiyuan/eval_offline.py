"""Offline check of a G1 pi05 checkpoint against ground-truth action chunks from the LeRobot dataset.

Note: the course dataset has no held-out episodes, so this measures fit on training data and verifies the
inference pipeline (normalization, delta->absolute, image preprocessing); it is not a generalization test.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq

from examples.zhiyuan import infer_g1
from openpi.policies.g1_policy import _parse_image

CAMERA_KEYS = {
    "base_0_rgb": "observation.images.top_head",
    "left_wrist_0_rgb": "observation.images.hand_left",
    "right_wrist_0_rgb": "observation.images.hand_right",
}
GROUPS = {"left_arm_0_6": list(range(7)), "left_gripper_14": [14], "right_arm_7_13": list(range(7, 14)),
          "head_waist_16_19": list(range(16, 20))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--asset", default="group1/task15_train")
    parser.add_argument("--repo-id", default="group1/task15_train")
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.1, 0.4, 0.7],
                        help="Positions within each episode to evaluate")
    parser.add_argument("--out", type=Path, required=True, help="JSON file for the summary")
    parser.add_argument("--save-npz", type=Path, help="Also dump the first sample as an infer_g1.py input NPZ")
    args = parser.parse_args()

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(args.repo_id)
    prompt = ds.meta.tasks[0]
    policy, stats = infer_g1.load_policy(args.checkpoint, args.asset)

    errs = {k: [] for k in ("pred", "hold", "no_delta_restore")}
    latencies = []
    for ep in range(ds.meta.total_episodes):
        ep_actions = np.stack(pq.read_table(
            ds.root / ds.meta.get_data_file_path(ep), columns=["action"]).column("action").to_pylist())
        start = int(ds.episode_data_index["from"][ep])
        length = len(ep_actions)
        for frac in args.fractions:
            t = int(frac * length)
            item = ds[start + t]
            images = {name: _parse_image(item[key].numpy()) for name, key in CAMERA_KEYS.items()}
            state = item["observation.state"].numpy()
            gt = ep_actions[np.clip(np.arange(t, t + infer_g1.ACTION_HORIZON), 0, length - 1)]

            if args.save_npz is not None and not latencies:
                np.savez(args.save_npz, prompt=np.str_(prompt), state=state,
                         **{n: infer_g1.letterbox_224(images[n]) for n in infer_g1.CAMERAS})
                np.save(args.save_npz.with_name(args.save_npz.stem + "_gt_actions.npy"), gt)

            t0 = time.perf_counter()
            pred = infer_g1.predict(policy, stats, images, state, prompt)
            latencies.append(time.perf_counter() - t0)

            masked_state = np.where(infer_g1.DELTA_MASK, state, 0)
            hold = pred.copy()
            hold[:, :20] = np.where(infer_g1.DELTA_MASK, state, pred[:, :20])
            no_restore = pred.copy()
            no_restore[:, :20] -= masked_state
            for k, p in (("pred", pred), ("hold", hold), ("no_delta_restore", no_restore)):
                errs[k].append(np.abs(p - gt))
        print(f"episode {ep:3d} done, last latency {latencies[-1] * 1000:.0f} ms", flush=True)

    summary = {"checkpoint": str(args.checkpoint), "num_samples": len(latencies),
               "latency_ms_after_warmup": {"mean": float(np.mean(latencies[1:]) * 1000),
                                           "p50": float(np.median(latencies[1:]) * 1000),
                                           "first_call": float(latencies[0] * 1000)}}
    for k, e in errs.items():
        e = np.stack(e)  # (N, 50, 22)
        summary[k] = {g: {"mae_all_steps": float(e[:, :, d].mean()), "mae_step0": float(e[:, 0, d].mean()),
                          "mae_step24": float(e[:, 24, d].mean()), "mae_step49": float(e[:, 49, d].mean())}
                      for g, d in GROUPS.items()}
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
