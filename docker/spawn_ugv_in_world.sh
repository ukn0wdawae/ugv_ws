#!/usr/bin/env bash
# Step 2: Spawn the UGV (default: ugv_beast_pro) + robot_state_publisher into an already-running Gazebo world.
# Usage (inside the container):
#   ./docker/spawn_ugv_in_world.sh park          # Spawn at park spot 1 (x=47, y=1, z=4, yaw=3.05)
#   ./docker/spawn_ugv_in_world.sh lake          # Spawn at lake spot 1 (x=-47, y=-15, z=4, yaw=0)
#   ./docker/spawn_ugv_in_world.sh hill 1        # Spawn at hill spot 1 (x=43, y=-13, z=3, yaw=-3.0)
#   ./docker/spawn_ugv_in_world.sh hill 2        # Spawn at hill spot 2 (x=45, y=12, z=14, yaw=0.0)
#   ./docker/spawn_ugv_in_world.sh forest 1      # Spawn at forest spot 1 (x=48, y=2.1, z=4, yaw=-3.13)
#   ./docker/spawn_ugv_in_world.sh forest 2      # Spawn at forest spot 2 (x=-37, y=-22.75, z=4, yaw=0.0)
set -eo pipefail

ENV_NAME="${1:-park}"
SPOT="${2:-1}"
shift $(( $# >= 2 ? 2 : ($# >= 1 ? 1 : 0) ))

cd /root/ugv_ws
source /opt/ros/humble/setup.bash

# Default to ugv_rover (6-wheeled robot with 3D LiDAR + RGB-D + GPS) unless UGV_MODEL is set
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

source /root/ugv_ws/install/setup.bash

echo "Spawning model=${UGV_MODEL} into env=${ENV_NAME} (spot=${SPOT}) ..."
exec ros2 launch ugv_gazebo spawn_ugv_env.launch.py env:="${ENV_NAME}" spot:="${SPOT}" "$@"
