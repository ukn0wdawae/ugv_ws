#!/usr/bin/env bash
# Autonomous frontier exploration (explore_lite) on RTAB-Map /map + Nav2.
# Run inside the container (after ./docker/shell.sh).
#
# Prerequisite terminals already running:
#   1) ./docker/legacy_code/run_park_sim.sh   (or run_lake_sim.sh / run_hill_sim.sh / run_forest_sim.sh)
#   2) ./docker/legacy_code/run_rtabmap_slam_rviz.sh
#   3) ros2 launch nav2_bringup navigation_launch.py \
#        use_sim_time:=true \
#        params_file:=/root/ugv_ws/src/ugv_main/ugv_gazebo/param/legacy_code/slam_nav.yaml
#
# Hill world (uneven terrain):
#   1) ./docker/legacy_code/run_hill_sim.sh
#   2) ./docker/legacy_code/run_rtabmap_slam_rviz.sh force_3dof:=false
#   3) ros2 launch nav2_bringup navigation_launch.py \
#        use_sim_time:=true \
#        params_file:=/root/ugv_ws/src/ugv_main/ugv_gazebo/param/legacy_code/slam_nav_terrain.yaml
#
# Usage:
#   ./docker/legacy_code/run_explore_lite.sh
set -eo pipefail

cd /root/ugv_ws
source /opt/ros/humble/setup.bash

if [ ! -d /root/ugv_ws/install/explore_lite ]; then
  echo "Building explore_lite ..."
  colcon build --packages-select explore_lite --symlink-install
fi

source /root/ugv_ws/install/setup.bash
echo "Launching explore_lite frontier exploration ..."
exec ros2 launch explore_lite explore.launch.py use_sim_time:=true "$@"

