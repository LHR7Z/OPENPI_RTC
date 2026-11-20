"""Convert course raw episodes directly to LeRobot v2.1 dataset (3 real cameras only)."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

FEATURES = {
    "observation.images.top_head": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": 30.0,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "observation.images.hand_left": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": 30.0,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "observation.images.hand_right": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": 30.0,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (20,),
    },
    "action": {
        "dtype": "float32",
        "shape": (22,),
    },
}


def load_episode(ep_dir: Path):
    with h5py.File(ep_dir / "aligned_joints.h5", "r") as f:
        st_joint = np.array(f["state/joint/position"], dtype=np.float32)
        st_left_eff = np.array(f["state/left_effector/position"], dtype=np.float32)
        st_right_eff = np.array(f["state/right_effector/position"], dtype=np.float32)
        st_head = np.array(f["state/head/position"], dtype=np.float32)
        st_waist = np.array(f["state/waist/position"], dtype=np.float32)
        ac_joint = np.array(f["action/joint/position"], dtype=np.float32)
        ac_left_eff = np.array(f["action/left_effector/position"], dtype=np.float32)
        ac_right_eff = np.array(f["action/right_effector/position"], dtype=np.float32)
        ac_head = np.array(f["action/head/position"], dtype=np.float32)
        ac_waist = np.array(f["action/waist/position"], dtype=np.float32)
        ac_vel = np.array(f["action/robot/velocity"], dtype=np.float32)
        n_frames = f["timestamp"].shape[0]

    state = np.hstack([st_joint, st_left_eff, st_right_eff, st_head, st_waist])
    action = np.hstack([ac_joint, ac_left_eff, ac_right_eff, ac_head, ac_waist, ac_vel])
    assert state.shape == (n_frames, 20), f"state shape {state.shape}"
    assert action.shape == (n_frames, 22), f"action shape {action.shape}"

    cam_dir = ep_dir / "camera"
    frame_dirs = sorted(
        [d for d in cam_dir.iterdir() if d.is_dir()],
        key=lambda p: int(p.name),
    )
    assert len(frame_dirs) == n_frames, f"frames {len(frame_dirs)} != h5 {n_frames}"

    return state, action, frame_dirs, n_frames


def get_task_text(ep_dir: Path) -> str:
    meta = json.loads((ep_dir / "meta_info.json").read_text(encoding="utf-8"))
    try:
        text = json.loads(meta.get("text", "{}"))
        desc = text.get("description", "")
        extras = text.get("extra", [])
        if extras:
            return "; ".join(extras)
        return desc or f"task_{meta.get('task_id', 'unknown')}"
    except (json.JSONDecodeError, AttributeError):
        return f"task_{meta.get('task_id', 'unknown')}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-list", required=True, help="Text file with one episode dir per line")
    parser.add_argument("--repo-id", required=True, help="LeRobot dataset repo_id (e.g. group1/task15_train)")
    parser.add_argument("--out", required=True, help="Output root dir for LeRobot dataset")
    parser.add_argument("--debug", action="store_true", help="Only convert first 2 episodes")
    args = parser.parse_args()

    eps = []
    for line in Path(args.input_list).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            eps.append(Path(line))
    if args.debug:
        eps = eps[:2]

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    out_root = Path(args.out) / args.repo_id
    if out_root.exists():
        import shutil
        shutil.rmtree(out_root)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=out_root,
        fps=30,
        robot_type="a2d",
        features=FEATURES,
        use_videos=True,
    )

    for ep_dir in tqdm(eps, desc="Converting episodes"):
        state, action, frame_dirs, n_frames = load_episode(ep_dir)
        task_text = get_task_text(ep_dir)

        for i in range(n_frames):
            head_img = Image.open(frame_dirs[i] / "head_color.jpg")
            left_img = Image.open(frame_dirs[i] / "hand_left_color.jpg")
            right_img = Image.open(frame_dirs[i] / "hand_right_color.jpg")

            frame = {
                "observation.images.top_head": head_img,
                "observation.images.hand_left": left_img,
                "observation.images.hand_right": right_img,
                "observation.state": state[i],
                "action": action[i],
                "task": task_text,
            }
            dataset.add_frame(frame)

        dataset.save_episode()
        print(f"  ep {ep_dir.name}: {n_frames} frames, task='{task_text}'")

    dataset.stop_image_writer()
    print(f"\nDone. Dataset at {out_root}")
    print(f"Episodes: {len(eps)}, repo_id: {args.repo_id}")


if __name__ == "__main__":
    main()
