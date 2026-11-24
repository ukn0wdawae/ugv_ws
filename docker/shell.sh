#!/usr/bin/env bash
# Open a shell in the ugv Humble+Gazebo container (creates it on first run).
#
#   ./docker/shell.sh            # NVIDIA GPU rendering (default, GPU=1)
#   GPU=0 ./docker/shell.sh      # CPU software rendering (llvmpipe), no NVIDIA needed
#
# GPU mode needs nvidia-container-toolkit on the host. The GPU mode is fixed
# when the container is created; to switch, run:  docker rm -f ugv-gazebo
set -euo pipefail

IMAGE="ugv-humble-gazebo:latest"
NAME="ugv-gazebo"
WS_HOST="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="${GPU:-1}"

xhost +local:root >/dev/null 2>&1 || true

if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  EXISTING_GPU="$(docker inspect -f '{{index .Config.Labels "ugv.gpu"}}' "$NAME" 2>/dev/null || true)"
  if [ "${EXISTING_GPU:-0}" != "$GPU" ]; then
    echo "Container '$NAME' was created with GPU=${EXISTING_GPU:-0}, but you asked for GPU=$GPU." >&2
    echo "Recreate it with:  docker rm -f $NAME && GPU=$GPU ./docker/shell.sh" >&2
    exit 1
  fi
  EXISTING_IPC="$(docker inspect -f '{{.HostConfig.IpcMode}}' "$NAME" 2>/dev/null || true)"
  if [ "$EXISTING_IPC" = "host" ]; then
    echo "Note: '$NAME' uses legacy host IPC mode. Recreate once with 'docker rm -f $NAME && ./docker/shell.sh' for --init + isolated /dev/shm." >&2
  fi
  docker start "$NAME" >/dev/null 2>&1 || true
  exec docker exec -it \
    -e DISPLAY="${DISPLAY:-:0}" \
    -e UGV_MODEL="${UGV_MODEL:-ugv_pro_audio}" \
    -e ALSOFT_DRIVERS="${ALSOFT_DRIVERS:-null}" \
    "$NAME" bash
fi

if [ "$GPU" = "1" ]; then
  # Real OpenGL on the NVIDIA GPU (gzclient + camera/depth/gpu_ray sensors).
  # No LIBGL_ALWAYS_SOFTWARE here.
  GPU_ARGS=(
    --gpus all
    --device /dev/dri
    -e NVIDIA_DRIVER_CAPABILITIES=all
    -e NVIDIA_VISIBLE_DEVICES=all
    -e __NV_PRIME_RENDER_OFFLOAD=1
    -e __GLX_VENDOR_LIBRARY_NAME=nvidia
  )
else
  # Works without NVIDIA passthrough (Mesa llvmpipe on the CPU).
  GPU_ARGS=(-e LIBGL_ALWAYS_SOFTWARE=1)
fi

exec docker run -it \
  --name "$NAME" \
  --init \
  --label "ugv.gpu=$GPU" \
  --network host \
  --shm-size=2g \
  "${GPU_ARGS[@]}" \
  -e DISPLAY="${DISPLAY:-:0}" \
  -e XDG_RUNTIME_DIR=/tmp/runtime-root \
  -e UGV_MODEL="${UGV_MODEL:-ugv_pro_audio}" \
  -e GAZEBO_MODEL_DATABASE_URI="" \
  -e ALSOFT_DRIVERS=null \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /dev/input:/dev/input \
  --device-cgroup-rule='c 13:* rmw' \
  -v "$WS_HOST":/root/ugv_ws \
  -w /root/ugv_ws \
  "$IMAGE" bash
