#!/usr/bin/env bash
# Step 1: Build packages and launch ONLY the park world in Gazebo (no robot spawned yet).
# Run inside the container (after ./docker/shell.sh).
# Once the world is open, run ./docker/legacy_code/spawn_ugv_in_world.sh park in a second container shell.
set -eo pipefail

cd /root/ugv_ws
source /opt/ros/humble/setup.bash

export GAZEBO_MODEL_DATABASE_URI=""
export GAZEBO_MODEL_PATH="/root/ugv_ws/gazebo_models:/root/.gazebo/models:/root/ugv_ws/src/ugv_main/ugv_gazebo/models:/root/ugv_ws/src/ugv_main/ugv_gazebo/legacy_code/models:${GAZEBO_MODEL_PATH:-}:/usr/share/gazebo-11/models"

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

echo "Launching park.world (world only, spawn_robot:=false) ..."
exec ros2 launch ugv_gazebo bringup_park.launch.py spawn_robot:=false "$@"
