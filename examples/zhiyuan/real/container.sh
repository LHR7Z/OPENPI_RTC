#!/bin/bash
# Host: start the course image for a real-robot session (GPU + host network for the robot's cosine/aorta bus).
# Usage: bash examples/zhiyuan/real/container.sh <gpu_index> <host_debug_ip 10.42.0.x> <checkpoint_step_dir>
# Extra docker flags via G1_DOCKER_ARGS, e.g. G1_DOCKER_ARGS=--privileged (what the GDK container uses) if the
# SDK reports device/clock errors.
set -euo pipefail

if [ $# -ne 3 ]; then
    echo "Usage: $0 <gpu_index> <host_debug_ip> <checkpoint_step_dir>" >&2
    exit 1
fi
GPU=$1
HOST_IP=$2
CKPT=$(realpath "$3")
REPO=$(cd "$(dirname "$0")/../../.." && pwd)

if docker info -f '{{json .SecurityOptions}}' | grep -q rootless; then
    echo "Rootless docker: --network host is not the real host network, so the SDK cannot reach the robot." >&2
    echo "Use the workstation's regular (rootful) docker." >&2
    exit 1
fi
if ! ip -o -4 addr | grep -q " ${HOST_IP}/"; then
    echo "${HOST_IP} is not configured on any interface of this host (see guide 7.4.2)." >&2
    exit 1
fi
if [ ! -d "$CKPT/params" ] || [ ! -f "$CKPT/assets/group1/task15_train/norm_stats.json" ]; then
    echo "$CKPT is not an openpi checkpoint step dir with params/ and the task norm stats." >&2
    exit 1
fi
# Only one stack may command the robot: refuse while the RDT wrapper is running in the GDK container.
if [ -n "$(docker ps -q -f name='^genie_sim_deploy$')" ] &&
    docker exec genie_sim_deploy pgrep -af realworld/infer.py >/dev/null 2>&1; then
    echo "RDT inference (wrapper/realworld/infer.py) is running in genie_sim_deploy; stop it first (guide 7.5.5)." >&2
    exit 1
fi

# shellcheck disable=SC2086  # G1_DOCKER_ARGS is intentionally word-split
exec docker run -it --rm --name "g1_pi05_$(id -un)" ${G1_DOCKER_ARGS:-} \
    --user "$(id -u):$(id -g)" -e HOME=/tmp \
    --gpus "device=${GPU}" --network host \
    -e LOCATOR_IP="$HOST_IP" -e AORTA_DISCOVERY_URI=http://10.42.0.101:2379 \
    -v "$REPO:/work" -v "$CKPT:/checkpoint:ro" -w /work \
    course/pi05-g1:2.0 bash
