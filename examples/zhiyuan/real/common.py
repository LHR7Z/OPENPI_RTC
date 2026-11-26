"""Shared by the policy server (python 3.11) and the robot side (python 3.10, a2d_sdk): stdlib + numpy only.

Both processes run in the same course container, so a Unix socket under the container's own /tmp is private to it.
"""

from __future__ import annotations

from multiprocessing.connection import Client
import time

import numpy as np

SOCKET = "/tmp/g1_policy.sock"
AUTHKEY = b"g1-pi05-policy"
# Must match the training task string exactly.
PROMPT = "夹起玩偶; 将玩偶移动到左边区域; 放下玩偶"
ACTION_HORIZON = 50
ACTION_DIM = 22


class PolicyClient:
    def __init__(self, socket_path: str = SOCKET, wait_seconds: float = 300.0):
        deadline = time.monotonic() + wait_seconds
        while True:
            try:
                self._conn = Client(socket_path, family="AF_UNIX", authkey=AUTHKEY)
                break
            except (FileNotFoundError, ConnectionRefusedError):
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"Policy server not reachable at {socket_path}; is g1_policy_server.py running?"
                    ) from None
                time.sleep(1.0)
        self.metadata = self._call({"type": "ping"})["metadata"]

    def infer(self, images: dict, state: np.ndarray, prompt: str = PROMPT) -> tuple[np.ndarray, float]:
        """Returns a (50, 22) chunk of absolute action targets and the server-side inference time in ms."""
        resp = self._call({"type": "infer", "images": images, "state": np.asarray(state, np.float32), "prompt": prompt})
        actions = np.asarray(resp["actions"])
        if actions.shape != (ACTION_HORIZON, ACTION_DIM) or not np.isfinite(actions).all():
            raise RuntimeError(f"Bad action chunk from server: shape {actions.shape}")
        return actions, resp["infer_ms"]

    def _call(self, request: dict) -> dict:
        self._conn.send(request)
        resp = self._conn.recv()
        if "error" in resp:
            raise RuntimeError(f"Policy server error:\n{resp['error']}")
        return resp

    def close(self) -> None:
        self._conn.close()
