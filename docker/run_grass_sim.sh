#!/usr/bin/env bash
# One-shot: build packages (if needed) and launch grass world + ugv_rover.
# Run inside the container (after ./docker/shell.sh).
set -eo pipefail

cd /root/ugv_ws
source /opt/ros/humble/setup.bash

export UGV_MODEL="${UGV_MODEL:-ugv_rover}"
export GAZEBO_MODEL_DATABASE_URI=""
export GAZEBO_MODEL_PATH="/root/.gazebo/models:/root/ugv_ws/src/ugv_main/ugv_gazebo/models:${GAZEBO_MODEL_PATH:-}"

# Mesh package expected by model://ugv_description/...
if [ ! -d /root/.gazebo/models/ugv_description ]; then
  echo "Installing ugv_description meshes into ~/.gazebo/models ..."
  mkdir -p /root/.gazebo/models
  unzip -qo /root/ugv_ws/ugv_description.zip -d /root/.gazebo/models
fi

if [ ! -f /root/ugv_ws/install/setup.bash ]; then
  echo "Building ugv_gazebo (first time) ..."
  colcon build --packages-select ugv_gazebo --symlink-install
fi

source /root/ugv_ws/install/setup.bash

echo "Launching grass world + model=${UGV_MODEL}"
exec ros2 launch ugv_gazebo bringup_grass.launch.py
