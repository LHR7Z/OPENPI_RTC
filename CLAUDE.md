# Notes

## Docker (course image `course/pi05-g1:2.0`, already loaded)
- This machine runs **rootless docker** (root dir `~/.local/share/docker`). Do NOT pass `--user "$(id -u):$(id -g)"` (the course README does): the mapped uid can't write bind mounts. Without `--user`, container root == host `zzh`, files come out owned by zzh.
- GPU: `--gpus device=N` works. CPU-only (TA's test): drop `--gpus`, add `-e JAX_PLATFORMS=cpu`.
- `import openpi` inside the image loads upstream `/opt/openpi/src` (commit 215abfb), NOT the mounted `/work/src`. Our modules (`g1_policy`, `pi05_course` config) are missing there unless `-e PYTHONPATH=/work/src`.
- `python` = 3.11 (OpenPI/JAX 0.5.3); `python3.10` = G1 SDK (`a2d_sdk`). Importing `a2d_sdk` writes `camera.txt`, `hal.txt`, glog files into cwd → `cd /tmp` first.
- Inference for our checkpoint: `examples/zhiyuan/infer_g1.py` (self-contained, works in the image). Course `infer.py` skips delta→absolute, wrong for this model.
- RTC (`predict(..., prev_actions=...)`) needs this repo's `pi0.py`/`policy.py` → in the image add `-e PYTHONPATH=/work/src`. Plain inference doesn't.
- Real-robot gotchas (from `a2d_sdk/robot.py` + data): `move_gripper` treats values >1.0 as raw 35–120 units → always clip model gripper output to [0,1]. Training images were raw res (head 1280×800, hands 848×480; info.json's 480×640 is wrong); SDK camera_config says head 720×1280 → verify live shape/RGB order. Dataset start pose ≠ SDK `reset()` default pose → reset to dataset first-frame mean, verify joint order/sign on robot. `move_head` is a blocking service call (not for 30 Hz). Model inference: GPU ~0.1–0.15 s, CPU ~10 s (unusable for real time). py3.10 side has no openpi_client/websockets/msgpack; numpy 1.26.4 on both pythons.
- The docker daemon is shared with other workloads of this account (eval-harness containers `*env-main*`/`*verifier*`); when busy, `docker run` can queue for 10+ min. Don't touch those containers.

```bash
CKPT=$PWD/checkpoints/pi05_course/g1_bs32_fix/2999
docker run --rm -e HOME=/tmp --gpus device=0 -v "$PWD:/work" -v "$CKPT:/checkpoint:ro" course/pi05-g1:2.0 \
  python examples/zhiyuan/infer_g1.py /checkpoint my_assets/inference/obs_ep0.npz out.npy --asset group1/task15_train
```
