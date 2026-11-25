#!/usr/bin/env bash
# One-shot: build Humble+Gazebo image, start container, build ugv_gazebo, launch grass sim.
#
#   ./legacy_code/launch_docker_sim.sh          # NVIDIA GPU rendering (default, GPU=1)
#   GPU=0 ./legacy_code/launch_docker_sim.sh    # CPU software rendering
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="ugv-humble-gazebo:latest"
NAME="ugv-gazebo"
MODEL="${UGV_MODEL:-ugv_rover}"
GPU="${GPU:-1}"

if [ "$GPU" = "1" ]; then
  GPU_ARGS=(
    --gpus all
    --device /dev/dri
    -e NVIDIA_DRIVER_CAPABILITIES=all
    -e NVIDIA_VISIBLE_DEVICES=all
    -e __NV_PRIME_RENDER_OFFLOAD=1
    -e __GLX_VENDOR_LIBRARY_NAME=nvidia
  )
  EXEC_GPU_ARGS=(
    -e __NV_PRIME_RENDER_OFFLOAD=1
    -e __GLX_VENDOR_LIBRARY_NAME=nvidia
  )
else
  GPU_ARGS=(-e LIBGL_ALWAYS_SOFTWARE=1)
  EXEC_GPU_ARGS=(-e LIBGL_ALWAYS_SOFTWARE=1)
fi

cd "$WS"
chmod +x docker/*.sh

echo "==> Allow X11 for Docker"
xhost +local:root >/dev/null 2>&1 || true

echo "==> Build image (first time can take several minutes)"
docker build -t "$IMAGE" "$WS/docker"

echo "==> Recreate container (GPU=$GPU)"
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" \
  --init \
  --label "ugv.gpu=$GPU" \
  --network host \
  --shm-size=2g \
  "${GPU_ARGS[@]}" \
  -e DISPLAY="${DISPLAY:-:0}" \
  -e XDG_RUNTIME_DIR=/tmp/runtime-root \
  -e UGV_MODEL="$MODEL" \
  -e GAZEBO_MODEL_DATABASE_URI="" \
  -e ALSOFT_DRIVERS=null \
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
  cp -f /root/ugv_ws/src/ugv_main/ugv_description/meshes/ugv_rover/base_link_chassis.stl \
    /root/.gazebo/models/ugv_description/meshes/ugv_rover/ 2>/dev/null || true
  cd /root/ugv_ws
  colcon build --packages-select ugv_gazebo --symlink-install
'

echo "==> Launch grass world + ${MODEL}"
docker exec -it \
  -e DISPLAY="${DISPLAY:-:0}" \
  "${EXEC_GPU_ARGS[@]}" \
  -e UGV_MODEL="$MODEL" \
  -e GAZEBO_MODEL_DATABASE_URI="" \
  -e GAZEBO_MODEL_PATH="/root/.gazebo/models:/root/ugv_ws/src/ugv_main/ugv_gazebo/models:/root/ugv_ws/src/ugv_main/ugv_gazebo/legacy_code/models:/usr/share/gazebo-11/models" \
  "$NAME" bash -lc '
    source /opt/ros/humble/setup.bash
    source /root/ugv_ws/install/setup.bash
    export UGV_MODEL="${UGV_MODEL:-ugv_rover}"
    export GAZEBO_MODEL_DATABASE_URI=""
    export GAZEBO_MODEL_PATH="/root/.gazebo/models:/root/ugv_ws/src/ugv_main/ugv_gazebo/models:/root/ugv_ws/src/ugv_main/ugv_gazebo/legacy_code/models:${GAZEBO_MODEL_PATH:-}:/usr/share/gazebo-11/models"
    ros2 launch ugv_gazebo bringup_grass.launch.py
  '
