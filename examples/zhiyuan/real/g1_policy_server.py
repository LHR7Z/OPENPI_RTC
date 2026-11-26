"""Serves the G1 pi05 policy (python 3.11, openpi) over a Unix socket to the a2d_sdk side (python 3.10).

Run inside the course container:  python examples/zhiyuan/real/g1_policy_server.py --checkpoint /checkpoint
Requests carry raw camera frames (any size; letterboxed to 224 like training), the 20-dim state and the prompt;
responses carry a (50, 22) chunk of absolute action targets.
"""

import argparse
from multiprocessing.connection import Listener
import os
import pathlib
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # examples/zhiyuan, for infer_g1
import common
import infer_g1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=pathlib.Path, default=pathlib.Path("/checkpoint"))
    parser.add_argument("--asset", default="group1/task15_train")
    parser.add_argument("--socket", default=common.SOCKET)
    args = parser.parse_args()

    policy, stats = infer_g1.load_policy(args.checkpoint, args.asset)
    blank = {name: np.zeros((224, 224, 3), np.uint8) for name in infer_g1.CAMERAS}
    t0 = time.perf_counter()
    infer_g1.predict(policy, stats, blank, np.zeros(infer_g1.STATE_DIM, np.float32), common.PROMPT)
    print(f"Warmup (compile) took {time.perf_counter() - t0:.1f} s", flush=True)

    if os.path.exists(args.socket):
        os.unlink(args.socket)
    metadata = {"checkpoint": str(args.checkpoint), "asset": args.asset}
    with Listener(args.socket, family="AF_UNIX", authkey=common.AUTHKEY) as listener:
        print(f"READY on {args.socket}", flush=True)
        while True:
            with listener.accept() as conn:
                print("client connected", flush=True)
                while True:
                    try:
                        req = conn.recv()
                    except EOFError:
                        print("client disconnected", flush=True)
                        break
                    try:
                        if req["type"] == "ping":
                            conn.send({"metadata": metadata})
                            continue
                        t0 = time.perf_counter()
                        actions = infer_g1.predict(policy, stats, req["images"], req["state"], req["prompt"])
                        conn.send({"actions": actions, "infer_ms": (time.perf_counter() - t0) * 1000})
                    except Exception:
                        conn.send({"error": traceback.format_exc()})


if __name__ == "__main__":
    main()
