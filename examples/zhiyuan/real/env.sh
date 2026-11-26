# Source inside the course container before using a2d_sdk:  source examples/zhiyuan/real/env.sh
# Mirrors the GDK env.sh without ROS/forwarder: our code talks to the robot through cosine_bus only.
# Needs the robot-side mode switch server running (ssh agi@10.42.0.101; a2d_mode_switch_server.py).

if [ -z "${LOCATOR_IP:-}" ]; then
    echo "LOCATOR_IP is not set: pass this host's 10.42.0.x debug address (container.sh does)." >&2
    return 1 2>/dev/null || exit 1
fi
export AORTA_DISCOVERY_URI="${AORTA_DISCOVERY_URI:-http://10.42.0.101:2379}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export DYLOG_log_dir=/tmp/g1_sdk_logs DYLOG_LOG_SIZE=20000 DYLOG_DEFAULT_LEVEL=FATAL
mkdir -p "$DYLOG_log_dir"
export PATH="/usr/local/lib/python3.10/dist-packages/bin:$PATH"

# Switch the robot to COPILOT / cosine-SDK mode and wait until joint states and cameras are up (~10-40 s).
(cd /tmp && robot-service -s -c "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/vr.pbtxt" --no-ros)
