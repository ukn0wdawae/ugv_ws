#!/usr/bin/env bash
# Open a shell in the ugv Humble+Gazebo container (creates it on first run).
set -euo pipefail

IMAGE="ugv-humble-gazebo:latest"
NAME="ugv-gazebo"
WS_HOST="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

xhost +local:root >/dev/null 2>&1 || true

if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  docker start "$NAME" >/dev/null 2>&1 || true
  exec docker exec -it \
    -e DISPLAY="${DISPLAY:-:0}" \
    -e UGV_MODEL="${UGV_MODEL:-ugv_rover}" \
    "$NAME" bash
fi

# LIBGL_ALWAYS_SOFTWARE=1 : works without NVIDIA passthrough (llvmpipe)
exec docker run -it \
  --name "$NAME" \
  --network host \
  --ipc host \
  -e DISPLAY="${DISPLAY:-:0}" \
  -e LIBGL_ALWAYS_SOFTWARE=1 \
  -e XDG_RUNTIME_DIR=/tmp/runtime-root \
  -e UGV_MODEL="${UGV_MODEL:-ugv_rover}" \
  -e GAZEBO_MODEL_DATABASE_URI="" \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$WS_HOST":/root/ugv_ws \
  -w /root/ugv_ws \
  "$IMAGE" bash
