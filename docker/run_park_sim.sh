#!/usr/bin/env bash
# Combined launch: build packages and launch park world + UGV (ugv_rover 6-wheeled robot by default).
# Run inside the container (after ./docker/shell.sh).
set -eo pipefail

cd /root/ugv_ws
source /opt/ros/humble/setup.bash

export UGV_MODEL="ugv_pro_audio"
export GAZEBO_MODEL_DATABASE_URI=""
export GAZEBO_MODEL_PATH="/root/ugv_ws/gazebo_models:/root/.gazebo/models:/root/ugv_ws/src/ugv_main/ugv_gazebo/models:${GAZEBO_MODEL_PATH:-}:/usr/share/gazebo-11/models"

if [ ! -f /root/.gazebo/models/ugv_description/meshes/ugv_rover/base_link_chassis.stl ]; then
  echo "Installing ugv_description meshes into ~/.gazebo/models ..."
  mkdir -p /root/.gazebo/models
  unzip -qo /root/ugv_ws/ugv_description.zip -d /root/.gazebo/models
  cp -f /root/ugv_ws/src/ugv_main/ugv_description/meshes/ugv_rover/base_link_chassis.stl \
    /root/.gazebo/models/ugv_description/meshes/ugv_rover/ 2>/dev/null || true
fi

echo "Building ugv_description and ugv_gazebo ..."
colcon build --packages-select ugv_description ugv_gazebo --symlink-install
source /root/ugv_ws/install/setup.bash

echo "Launching park world + model=${UGV_MODEL}"
exec ros2 launch ugv_gazebo bringup_park.launch.py "$@"
