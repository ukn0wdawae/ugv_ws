#!/usr/bin/env bash
# Step 1: Build packages and launch ONLY the forest world in Gazebo (no robot spawned yet).
# Run inside the container (after ./docker/shell.sh).
# Once the world is open, run ./docker/spawn_ugv_in_world.sh forest in a second container shell.
set -eo pipefail

cd /root/ugv_ws
source /opt/ros/humble/setup.bash

export GAZEBO_MODEL_DATABASE_URI=""
export GAZEBO_MODEL_PATH="/root/ugv_ws/gazebo_models:/root/.gazebo/models:/root/ugv_ws/src/ugv_main/ugv_gazebo/models:${GAZEBO_MODEL_PATH:-}:/usr/share/gazebo-11/models"

echo "Building ugv_description and ugv_gazebo ..."
colcon build --packages-select ugv_description ugv_gazebo --symlink-install
source /root/ugv_ws/install/setup.bash

echo "Launching forest.world (world only, spawn_robot:=false) ..."
exec ros2 launch ugv_gazebo bringup_forest.launch.py spawn_robot:=false "$@"
