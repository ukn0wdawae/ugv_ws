#!/usr/bin/env bash
# Launch the full Coqui Acoustic Localization + 3D/2.5D RTAB-Map SLAM + Nav2 + RViz + Hunter Behavior.
# All 5 Coqui frogs in the environment (open, behind a tree, inside an untraversable bush,
# behind an object, and perched on a tree) are active on the map without any `scenario` tag.
#
# Prerequisite (Terminal 1 already running):
#   ./docker/run_coqui_sim.sh env:=park    (or env:=lake / env:=hill / env:=forest)
#
# Usage inside container (Terminal 2):
#   ./docker/run_coqui_behavior.sh env:=park
#   ./docker/run_coqui_behavior.sh env:=lake
#   ./docker/run_coqui_behavior.sh env:=hill
#   ./docker/run_coqui_behavior.sh env:=forest
set -eo pipefail

cd /root/ugv_ws
source /opt/ros/humble/setup.bash

export UGV_MODEL="ugv_pro_audio"
export GAZEBO_MODEL_DATABASE_URI=""
export GAZEBO_MODEL_PATH="/root/ugv_ws/gazebo_models:/root/.gazebo/models:/root/ugv_ws/src/ugv_main/ugv_gazebo/models:${GAZEBO_MODEL_PATH:-}:/usr/share/gazebo-11/models"

echo "Building ugv_description, ugv_gazebo, ugv_audio, ugv_coqui_behavior ..."
colcon build --packages-select ugv_description ugv_gazebo ugv_audio ugv_coqui_behavior --symlink-install
source /root/ugv_ws/install/setup.bash

echo "Launching Coqui 3D/2.5D SLAM + Nav2 + RViz (view_coqui_3d.rviz) + Acoustic Hunter Behavior ..."
exec ros2 launch ugv_coqui_behavior coqui_behavior.launch.py use_sim_time:=true "$@"
