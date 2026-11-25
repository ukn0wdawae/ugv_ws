#!/usr/bin/env bash
# Build packages and launch one of the 4 compact Coqui environments (park, lake, hill, forest)
# with the `ugv_pro_audio` robot (equipped with the 5-channel 3D microphone array & PTZ camera).
#
# Usage inside container (./docker/shell.sh):
#   ./docker/legacy_code/run_coqui_sim.sh                  # defaults to env:=park, model=ugv_pro_audio
#   ./docker/legacy_code/run_coqui_sim.sh env:=lake
#   ./docker/legacy_code/run_coqui_sim.sh env:=hill
#   ./docker/legacy_code/run_coqui_sim.sh env:=forest
set -eo pipefail

cd /root/ugv_ws
source /opt/ros/humble/setup.bash

export UGV_MODEL="ugv_pro_audio"
export GAZEBO_MODEL_DATABASE_URI=""
export GAZEBO_MODEL_PATH="/root/ugv_ws/gazebo_models:/root/.gazebo/models:/root/ugv_ws/src/ugv_main/ugv_gazebo/models:/root/ugv_ws/src/ugv_main/ugv_gazebo/legacy_code/models:${GAZEBO_MODEL_PATH:-}:/usr/share/gazebo-11/models"

if [ ! -f /root/.gazebo/models/ugv_description/meshes/ugv_rover/base_link_chassis.stl ]; then
  echo "Installing ugv_description meshes into ~/.gazebo/models ..."
  mkdir -p /root/.gazebo/models
  unzip -qo /root/ugv_ws/ugv_description.zip -d /root/.gazebo/models
  cp -f /root/ugv_ws/src/ugv_main/ugv_description/meshes/ugv_rover/base_link_chassis.stl \
    /root/.gazebo/models/ugv_description/meshes/ugv_rover/ 2>/dev/null || true
fi

echo "Building ugv_description, ugv_gazebo, ugv_audio, ugv_coqui_behavior ..."
colcon build --packages-select ugv_description ugv_gazebo ugv_audio ugv_coqui_behavior --symlink-install
source /root/ugv_ws/install/setup.bash

echo "Launching compact Coqui environment with model=${UGV_MODEL} ..."
exec ros2 launch ugv_gazebo bringup_coqui.launch.py "$@"

