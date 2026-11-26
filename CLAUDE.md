# Notes

## Docker (course image `course/pi05-g1:2.0`, already loaded)
- This **training server** runs **rootless docker** (root dir `~/.local/share/docker`; the 4090 deploy workstation is rootful). Here, do NOT pass `--user "$(id -u):$(id -g)"` (the course README does): the mapped uid can't write bind mounts. Without `--user`, container root == host `zzh`, files come out owned by zzh.
- GPU: `--gpus device=N` works. CPU-only (TA's test): drop `--gpus`, add `-e JAX_PLATFORMS=cpu`.
- `import openpi` inside the image loads upstream `/opt/openpi/src` (commit 215abfb), NOT the mounted `/work/src`. Our modules (`g1_policy`, `pi05_course` config) are missing there unless `-e PYTHONPATH=/work/src`.
- `python` = 3.11 (OpenPI/JAX 0.5.3); `python3.10` = G1 SDK (`a2d_sdk`). Importing `a2d_sdk` writes `camera.txt`, `hal.txt`, glog files into cwd → `cd /tmp` first.
- Inference for our checkpoint: `examples/zhiyuan/infer_g1.py` (self-contained, works in the image). Course `infer.py` skips delta→absolute, wrong for this model.
- RTC (`predict(..., prev_actions=...)`) needs this repo's `pi0.py`/`policy.py` → in the image add `-e PYTHONPATH=/work/src`. Plain inference doesn't.
- Real-robot gotchas (from `a2d_sdk/robot.py` + data): `move_gripper` treats values >1.0 as raw 35–120 units → always clip model gripper output to [0,1]. Training images were raw res (head 1280×800, hands 848×480; info.json's 480×640 is wrong); SDK camera_config says head 720×1280 → verify live shape/RGB order. Dataset start pose ≠ SDK `reset()` default pose → reset to dataset first-frame mean, verify joint order/sign on robot. `move_head` is a blocking service call (not for 30 Hz). Model inference: GPU ~0.1–0.15 s, CPU ~10 s (unusable for real time). py3.10 side has no openpi_client/websockets/msgpack; numpy 1.26.4 on both pythons.
- Single-container real-robot path: `examples/zhiyuan/real/` (course image only, no GDK/forwarder). SDK uses cosine_bus, not ROS; `robot-service -s -c vr.pbtxt --no-ros` does the mode switch (skips only the forwarder). Without `AORTA_DISCOVERY_URI`/`LOCATOR_IP`, cosine_bus `exit(1)`s ("aorta domain init failed") — uncatchable. Can only be tested offline here (rootless docker can't reach the robot).
- The docker daemon is shared with other workloads of this account (eval-harness containers `*env-main*`/`*verifier*`); when busy, `docker run` can queue for 10+ min. Don't touch those containers.

```bash
CKPT=$PWD/checkpoints/pi05_course/g1_bs32_fix/2999
docker run --rm -e HOME=/tmp --gpus device=0 -v "$PWD:/work" -v "$CKPT:/checkpoint:ro" course/pi05-g1:2.0 \
  python examples/zhiyuan/infer_g1.py /checkpoint my_assets/inference/obs_ep0.npz out.npy --asset group1/task15_train
```

## Real-robot runbook (4090 workstation wired to the robot; RDT guide `my_assets/RDT-部署推理实验指导书.md` 7.4–7.5)
Not yet run on the robot. Same robot/task as the RDT lab (A2D0013B200209, episode 1015 is in our data).

0. **Copy to the 4090**: `pi05-g1-2.0.tar.zst` + `SHA256SUMS` (`sha256sum -c SHA256SUMS && zstd -dc pi05-g1-2.0.tar.zst | docker load`), checkpoint step dir `g1_bs32_fix/2999` (`params/` ~12 GB + `assets/`), this repo.
   4090 docker is **rootful** (confirmed by user), unlike this server: `container.sh` passes `--user uid:gid` (correct there). Image store is machine-wide → `docker images course/pi05-g1` first; ID `a0dbba32a616` means already loaded, don't reload/remove others' images. Only one stack may command the robot: `container.sh` refuses while `genie_sim_deploy` runs `realworld/infer.py`; also don't share the GPU with a running RDT. GDK container uses `--privileged`; ours doesn't — if the SDK shows device/clock (PTP) errors, rerun with `G1_DOCKER_ARGS=--privileged bash .../container.sh ...`. Stale own container ⇒ `docker rm g1_pi05_$(id -un)`.
1. **Offline first (no robot)**: run `infer_g1.py` on `obs_ep0.npz`, then server + `g1_run.py --replay-npz my_assets/inference/obs_ep0.npz` (dry and `--execute --yes`) → note 4090 latency (H100: ~50–100 ms).
2. **Network (guide 7.4.2)**: wired NIC → robot back-panel **Debug** port; static IPv4 `10.42.0.10–99/24` (assigned, no conflict); `ip route get 10.42.0.101` shows that NIC/src; ping both ways. Robot on, VR teleop exited, arms safe, e-stop person present.
3. **Terminal R (robot, keep running)**: `ssh agi@10.42.0.101`, then `cd /home/agi && source app/a2d_sdk/a2d_env.sh && python3 app/a2d_sdk/scripts/a2d_mode_switch_server.py`.
4. **Terminal C (host)**: `bash examples/zhiyuan/real/container.sh <gpu> <host_debug_ip> <ckpt_step_dir>`; inside: `source examples/zhiyuan/real/env.sh` (~10–40 s; should end with "success" / "Robot is ready").
5. **Read-only check**: `python3.10 examples/zhiyuan/real/g1_check.py` → `CHECK_OK`. Open `my_assets/real_check/*.png`: red/blue swapped ⇒ add `--swap-rb` to every g1_* call. Shapes must be head 800×1280, hands 480×848. State flags outside training range ⇒ fix posture before running. Collection posture: head ≈ [0, 0.436] rad, waist ≈ [0.852, 0.469], right arm ≈ [1.08, −0.61, −0.28, 1.27, −0.73, −1.49, 0.19]; left arm start ≈ dataset first-frame mean [−1.17, 0.66, 0.45, −1.17, 0.71, 1.43, 0.02].
6. **Policy server (2nd shell)**: `docker exec -it g1_pi05_$(id -un) bash`, then `python examples/zhiyuan/real/g1_policy_server.py`; wait for `READY` (~1 min load + ~12 s compile).
7. **Dry run**: `python3.10 examples/zhiyuan/real/g1_run.py --max-chunks 3`. Expect start gap ≲ 0.1 rad, left arm moving ≲ 0.7 rad per 25 steps, gripper in [0,1]. Anything odd ⇒ stop here.
8. **Execute** (doll/target placed as in collection, e-stop in hand): `python3.10 examples/zhiyuan/real/g1_run.py --execute --max-joint-step 0.02 --max-chunks 2` first, then drop `--max-chunks`; raise `--max-joint-step` to 0.03–0.04 once stable (demo max ≈ 0.036 rad/step). Ctrl+C stops sending. If the arm ignores `move_arm` ⇒ fallback: Dockerfile `FROM course/pi05-g1:2.0` adding the GDK forwarder + ROS publish to `/wbc/arm_command` (genie_sim `robot_interface.py` way), or the GDK container.
9. **Stop**: Ctrl+C `g1_run.py`, Ctrl+C server, `exit` (container is `--rm`), Ctrl+C terminal R. Logs: `my_assets/real_runs/<time>_<dry|exec>_<pid>/chunk_*.npz` (state, chunk, sent commands). Success = picked up, moved left of the yellow line, released; record ≥ 3 attempts.
