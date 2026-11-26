"""G1 robot I/O for the pi05 policy (python 3.10, a2d_sdk over cosine_bus; no ROS or forwarder needed).

Maps SDK readings to the 20-dim training state and policy actions to SDK commands. Conventions checked against
a2d_sdk 1.5.0 and the training data:
  - state = arm 14 (rad, Joint1_l..7_l, Joint1_r..7_r) + grippers 2 (raw 0-120) + head 2 + waist 2 (pitch, lift);
    `head_joint_states()` returns degrees, training used radians -> converted here.
  - `move_gripper` treats values > 1.0 as raw 35-120 units, so gripper commands are clipped to [0, 1].
  - head, waist and base are never commanded (fixed in this task; `move_head`/`move_waist` are blocking services).
"""

from __future__ import annotations

import os
import time

import numpy as np

# a2d_sdk writes camera.txt, hal.txt and glog files into the cwd at import time.
SDK_LOG_DIR = "/tmp/g1_sdk_logs"

# policy camera key -> SDK camera name
CAMERAS = {"base_0_rgb": "head", "left_wrist_0_rgb": "hand_left", "right_wrist_0_rgb": "hand_right"}
# Frame sizes (H, W) the policy was trained on.
TRAIN_IMAGE_SHAPES = {"head": (800, 1280), "hand_left": (480, 848), "hand_right": (480, 848)}
STATE_NAMES = (
    [f"left_arm_{i}" for i in range(1, 8)]
    + [f"right_arm_{i}" for i in range(1, 8)]
    + ["left_gripper", "right_gripper", "head_0", "head_1", "waist_pitch", "waist_lift"]
)


class G1Robot:
    def __init__(self, *, swap_rb: bool = False, timeout: float = 15.0):
        """Changes the process cwd to SDK_LOG_DIR (see above); resolve relative paths before constructing."""
        # Without these, cosine_bus calls exit(1) ("aorta domain init failed"), which cannot be caught.
        missing = [k for k in ("LOCATOR_IP", "AORTA_DISCOVERY_URI") if not os.environ.get(k)]
        if missing:
            raise RuntimeError(f"{missing} not set: start the container with container.sh and `source env.sh`.")
        os.makedirs(SDK_LOG_DIR, exist_ok=True)
        os.chdir(SDK_LOG_DIR)
        from a2d_sdk.robot import CosineCamera
        from a2d_sdk.robot import RobotDds

        self._camera = CosineCamera(list(CAMERAS.values()))
        self._robot = RobotDds()
        self._swap_rb = swap_rb
        deadline = time.monotonic() + timeout
        while self.observe() is None:
            if time.monotonic() > deadline:
                self.close()
                raise TimeoutError(
                    "No complete observation (3 cameras + arm/gripper/head/waist states). Check LOCATOR_IP, "
                    "AORTA_DISCOVERY_URI, the robot-side mode switch server and `robot-service -s ... --no-ros`."
                )
            time.sleep(0.2)

    def observe(self) -> dict | None:
        images, stamps = {}, {}
        for key, cam in CAMERAS.items():
            frame, stamp = self._camera.get_latest_image(cam)
            if frame is None or np.ndim(frame) != 3 or frame.shape[2] != 3:
                return None
            images[key] = np.ascontiguousarray(frame[..., ::-1]) if self._swap_rb else frame
            stamps[cam] = stamp
        readings = [
            self._robot.arm_joint_states()[0],
            self._robot.gripper_states()[0],
            self._robot.head_joint_states()[0],
            self._robot.waist_joint_states()[0],
        ]
        if any(r is None or len(r) == 0 or any(v is None for v in r) for r in readings):
            return None
        arm, grip, head_deg, waist = (np.asarray(list(r), np.float64) for r in readings)
        state = np.concatenate([arm[:14], grip[:2], np.deg2rad(head_deg[:2]), waist[:2]]).astype(np.float32)
        if state.shape != (20,) or not np.isfinite(state).all():
            return None
        return {"images": images, "state": state, "stamps": stamps}

    def command(self, arm: np.ndarray, gripper: np.ndarray) -> None:
        self._robot.move_arm([float(x) for x in arm])
        self._robot.move_gripper([float(x) for x in np.clip(gripper, 0.0, 1.0)])

    def close(self) -> None:
        self._camera.close()
        self._robot.shutdown()


class ReplayRobot:
    """Offline stand-in: replays one NPZ observation and tracks commanded arm joints perfectly."""

    def __init__(self, npz_path: str):
        with np.load(npz_path, allow_pickle=False) as f:
            self._images = {key: f[key] for key in CAMERAS}
            self._state = f["state"].astype(np.float32)
        self.commands: list[np.ndarray] = []

    def observe(self) -> dict:
        return {"images": self._images, "state": self._state.copy(), "stamps": {}}

    def command(self, arm: np.ndarray, gripper: np.ndarray) -> None:
        self._state[:14] = arm
        self.commands.append(np.concatenate([arm, np.clip(gripper, 0.0, 1.0)]))

    def close(self) -> None:
        pass
