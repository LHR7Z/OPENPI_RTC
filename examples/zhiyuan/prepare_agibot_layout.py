#!/usr/bin/env python3
"""prepare_agibot_layout.py — Adapter: course-exported episodes -> official AgiBot World layout

The official AgiBot converter convert_to_lerobot.py (HF AgiBotWorld-Alpha scripts/) expects
input in the AgiBot World Alpha layout. This script adapts the course export (per-frame jpg
directories + a single aligned_joints.h5) into that layout, after which the **official script
can be run unmodified** to produce a LeRobot dataset:

    python convert_to_lerobot.py --src_path <output root of this script> \
        --task_id <task_id> --tgt_path <LeRobot output root> [--debug]

Adaptation rules (matching the read logic of the official load_local_dataset):
  observations/<task_id>/<episode_id>/videos/*.mp4
      The three real cameras are encoded from camera/<frame_idx>/*.jpg (ffmpeg, 30fps);
      the 5 fisheye cameras required by the official layout are absent in the course data,
      so black placeholder videos are generated (do not select them for training);
      same for the head_depth placeholder video (the official script also requires depth
      png files, so black images are generated as well).
  observations/<task_id>/<episode_id>/depth/head_depth_%06d.png
      The course has no depth export; black images are generated, one per frame
      (the official script asserts the count matches).
  proprio_stats/<task_id>/<episode_id>/proprio_stats.h5
      state: joint(14)+effector(2 = left 1 + right 1)+head(2)+waist(2) = 20 dims
      action: joint(14)+effector(2)+head(2)+waist(2)+robot/velocity(2) = 22 dims
  task_info/task_<task_id>.json
      [{"task_name": ..., "init_scene_text": ...}] (the official script concatenates these into the prompt)
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from typing import Optional
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

OFFICIAL_VIDEO_KEYS = (
    "head_color", "hand_left_color", "hand_right_color",
    "head_center_fisheye_color", "head_left_fisheye_color",
    "head_right_fisheye_color", "back_left_fisheye_color",
    "back_right_fisheye_color",
)
PLACEHOLDER_CAMS = ("head_center_fisheye_color", "head_left_fisheye_color",
                    "head_right_fisheye_color", "back_left_fisheye_color",
                    "back_right_fisheye_color")
W, H, FPS = 640, 480, 30


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"[Error] {' '.join(cmd[:6])}...\n{r.stderr[-800:]}")


def jpgs_to_mp4(frame_files: list[Path], out_mp4: Path):
    with tempfile.NamedTemporaryFile("w", suffix=".ffconcat", delete=False) as f:
        f.write("ffconcat version 1.0\n")
        dt = 1.0 / FPS
        for i, p in enumerate(frame_files):
            f.write(f"file '{p.resolve()}'\nduration {dt:.6f}\n")
        f.write(f"file '{frame_files[-1].resolve()}'\n")  # The last frame must be repeated once
        lst = f.name
    run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst,
         "-r", str(FPS), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_mp4)])
    Path(lst).unlink()


def black_mp4(duration_s: float, out_mp4: Path):
    run(["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", f"color=c=black:s={W}x{H}:r={FPS}",
         "-t", f"{max(duration_s, 1.0 / FPS):.6f}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_mp4)])


def convert_episode(ep_dir: Path, agibot_root: Path, depth_size: Optional[tuple]):
    meta = json.loads((ep_dir / "meta_info.json").read_text(encoding="utf-8"))
    task_id, episode_id = int(meta["task_id"]), int(meta["episode_id"])

    with h5py.File(ep_dir / "aligned_joints.h5", "r") as f:
        st_joint = np.array(f["state/joint/position"], dtype=np.float32)
        st_eff = np.hstack([np.array(f["state/left_effector/position"], dtype=np.float32),
                            np.array(f["state/right_effector/position"], dtype=np.float32)])
        st_head = np.array(f["state/head/position"], dtype=np.float32)
        st_waist = np.array(f["state/waist/position"], dtype=np.float32)
        ac_joint = np.array(f["action/joint/position"], dtype=np.float32)
        ac_eff = np.hstack([np.array(f["action/left_effector/position"], dtype=np.float32),
                            np.array(f["action/right_effector/position"], dtype=np.float32)])
        ac_head = np.array(f["action/head/position"], dtype=np.float32)
        ac_waist = np.array(f["action/waist/position"], dtype=np.float32)
        ac_vel = np.array(f["action/robot/velocity"], dtype=np.float32)
        n_frames = f["timestamp"].shape[0]

    frames = sorted((d for d in (ep_dir / "camera").iterdir() if d.is_dir()),
                    key=lambda p: int(p.name))
    frame_files = {cam: [fr / f"{cam}.jpg" for fr in frames]
                   for cam in ("head_color", "hand_left_color", "hand_right_color")}
    for cam, files in frame_files.items():
        missing = [str(p) for p in files if not p.exists()]
        if missing:
            sys.exit(f"[Error] episode {episode_id} is missing {cam} frames, e.g. {missing[0]}")

    # Observation videos (3 real cameras + placeholder streams required by the official layout)
    t = n_frames / float(FPS)
    vid_dir = agibot_root / "observations" / str(task_id) / str(episode_id) / "videos"
    vid_dir.mkdir(parents=True, exist_ok=True)
    for cam in OFFICIAL_VIDEO_KEYS:
        out = vid_dir / f"{cam}.mp4"
        if out.exists():
            continue
        if cam in PLACEHOLDER_CAMS:
            black_mp4(t, out)
        else:
            jpgs_to_mp4(frame_files[cam], out)

    # Depth placeholder png files (official script asserts count == number of frames)
    size = depth_size or (W, H)
    depth_dir = agibot_root / "observations" / str(task_id) / str(episode_id) / "depth"
    depth_dir.mkdir(parents=True, exist_ok=True)
    black = Image.new("L", size, 0)
    for i in range(n_frames):
        black.save(depth_dir / f"head_depth_{i:06d}.png")

    # proprio_stats.h5 (exact key names read by the official load_local_dataset)
    pro_dir = agibot_root / "proprio_stats" / str(task_id) / str(episode_id)
    pro_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(pro_dir / "proprio_stats.h5", "w") as f:
        for name, arr in [
            ("state/joint/position", st_joint), ("state/effector/position", st_eff),
            ("state/head/position", st_head), ("state/waist/position", st_waist),
            ("action/joint/position", ac_joint), ("action/effector/position", ac_eff),
            ("action/head/position", ac_head), ("action/waist/position", ac_waist),
            ("action/robot/velocity", ac_vel),
        ]:
            f.create_dataset(name, data=arr)

    # task_info/task_<id>.json (written once per task; appended with de-duplication)
    raw = meta.get("text", "")
    try:
        t = json.loads(raw) if isinstance(raw, str) else raw
        scene = t.get("description", "")
    except (json.JSONDecodeError, AttributeError):
        scene = ""
    info_dir = agibot_root / "task_info"
    info_dir.mkdir(parents=True, exist_ok=True)
    info_path = info_dir / f"task_{task_id}.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else []
    if not info:
        info = [{"task_name": f"task_{task_id}", "init_scene_text": scene}]
        info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[Done] episode {episode_id}: {n_frames} frames -> {vid_dir.parent}")
    return task_id


def main():
    ap = argparse.ArgumentParser(description="Course-exported episodes -> official AgiBot World layout")
    ap.add_argument("--input", action="append", default=[],
                    help="Episode directory (can be passed multiple times)")
    ap.add_argument("--input-list", help="Batch mode: text file with one episode directory per line")
    ap.add_argument("--out", required=True, help="Output root directory for the AgiBot layout")
    ap.add_argument("--depth-size", default="640x480", help="Depth placeholder image size WxH (default 640x480)")
    args = ap.parse_args()

    eps: list[Path] = list(Path(p).expanduser() for p in args.input)
    if args.input_list:
        for line in Path(args.input_list).read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                eps.append(Path(line.strip()).expanduser())
    if not eps:
        ap.error("Either --input or --input-list is required")
    w, h = (int(x) for x in args.depth_size.lower().split("x"))
    agibot_root = Path(args.out).expanduser()

    task_ids = set()
    for ep in eps:
        task_ids.add(convert_episode(ep, agibot_root, (w, h)))
    print(f"\n[Next step] Run the official converter (once per task_id):")
    for tid in sorted(task_ids):
        print(f"  python convert_to_lerobot.py --src_path {agibot_root} "
              f"--task_id {tid} --tgt_path <LeRobot output root>")


if __name__ == "__main__":
    main()
