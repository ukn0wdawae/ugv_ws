#!/usr/bin/env bash
# One-shot: build Humble+Gazebo image, start container, build ugv_gazebo, launch grass sim.
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="ugv-humble-gazebo:latest"
NAME="ugv-gazebo"
MODEL="${UGV_MODEL:-ugv_rover}"

cd "$WS"
chmod +x docker/*.sh

echo "==> Allow X11 for Docker"
xhost +local:root >/dev/null 2>&1 || true

echo "==> Build image (first time can take several minutes)"
docker build -t "$IMAGE" "$WS/docker"

echo "==> Recreate container"
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" \
  --network host \
  --ipc host \
  -e DISPLAY="${DISPLAY:-:0}" \
  -e LIBGL_ALWAYS_SOFTWARE=1 \
  -e XDG_RUNTIME_DIR=/tmp/runtime-root \
  -e UGV_MODEL="$MODEL" \
  -e GAZEBO_MODEL_DATABASE_URI="" \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$WS":/root/ugv_ws \
  -w /root/ugv_ws \
  "$IMAGE" sleep infinity

echo "==> Install meshes + build ugv_gazebo"
docker exec "$NAME" bash -lc '
  set -e
  source /opt/ros/humble/setup.bash
  mkdir -p /root/.gazebo/models
  unzip -qo /root/ugv_ws/ugv_description.zip -d /root/.gazebo/models
  cd /root/ugv_ws
  colcon build --packages-select ugv_gazebo --symlink-install
'

echo "==> Launch grass world + ${MODEL}"
docker exec -it \
  -e DISPLAY="${DISPLAY:-:0}" \
  -e LIBGL_ALWAYS_SOFTWARE=1 \
  -e UGV_MODEL="$MODEL" \
  -e GAZEBO_MODEL_DATABASE_URI="" \
  -e GAZEBO_MODEL_PATH="/root/.gazebo/models:/root/ugv_ws/src/ugv_main/ugv_gazebo/models" \
  "$NAME" bash -lc '
    source /opt/ros/humble/setup.bash
    source /root/ugv_ws/install/setup.bash
    export UGV_MODEL="${UGV_MODEL:-ugv_rover}"
    export GAZEBO_MODEL_DATABASE_URI=""
    export GAZEBO_MODEL_PATH="/root/.gazebo/models:/root/ugv_ws/src/ugv_main/ugv_gazebo/models:${GAZEBO_MODEL_PATH:-}"
    ros2 launch ugv_gazebo bringup_grass.launch.py
  '
