#!/usr/bin/env bash
# RTAB-Map RGB-D SLAM + RViz for ugv_world.
# Run inside the container (after ./docker/shell.sh).
#
# Prerequisite (other terminal, already running):
#   ./docker/run_ugv_world_sim.sh
#   or: ros2 launch ugv_gazebo bringup.launch.py
# Gazebo must already have the rover spawned and publishing
# /odom /scan /camera/image_raw /camera/depth/image_raw /camera/camera_info.
#
# This script is SLAM (mapping), not localization. rtabmap_rgbd.launch.py
# runs with -d and deletes ~/.ros/rtabmap.db on every start.
# Mapping is slam_toolbox-like: ICP lidar scan match, 20 m grid, strict visual loops.
#
# Flat floor (default):
#   ./docker/run_rtabmap_slam_rviz.sh
# Static hills / elevation (6DoF + IMU, still a 2D /map for Nav2):
#   ./docker/run_rtabmap_slam_rviz.sh force_3dof:=false
# A ground plane that moves is still not supported.
# Drive slowly. Do not start explore_lite until rooms are sketched in.
#
# Do NOT start a map_server / AMCL stack on the other side.
# Those fight RTAB-Map for /map and the map->odom TF.
#   BAD:  ros2 launch ugv_gazebo nav_rtabmap.launch.py
#   BAD:  ros2 launch nav2_bringup bringup_launch.py
#   OK:   ros2 launch nav2_bringup navigation_launch.py \
#           use_sim_time:=true \
#           params_file:=/root/ugv_ws/src/ugv_main/ugv_gazebo/param/slam_nav.yaml
# With force_3dof:=false (hills) use param/slam_nav_terrain.yaml instead; it reads
# /obstacles_cloud + /ground_cloud from obstacles_detection (started by this launch).
# After Nav2 is up (optional, for explore_lite):
#   ros2 launch explore_lite explore.launch.py use_sim_time:=true
#
# Also needs: ros-humble-rtabmap-ros  (rtabmap_slam, rtabmap_viz)
# RViz robot meshes need the ugv_description ROS package built
# (package://ugv_description/...), which is different from the
# Gazebo unzip into ~/.gazebo/models (model://ugv_description/...).
#
# Usage:
#   ./docker/run_rtabmap_slam_rviz.sh
#   ./docker/run_rtabmap_slam_rviz.sh force_3dof:=false  # static hills
#   ./docker/run_rtabmap_slam_rviz.sh use_rviz:=false    # rtabmap_viz instead
set -eo pipefail

cd /root/ugv_ws
source /opt/ros/humble/setup.bash

export UGV_MODEL="${UGV_MODEL:-ugv_rover}"
export GAZEBO_MODEL_DATABASE_URI=""
export GAZEBO_MODEL_PATH="/root/ugv_ws/gazebo_models:/root/.gazebo/models:/root/ugv_ws/src/ugv_main/ugv_gazebo/models:${GAZEBO_MODEL_PATH:-}:/usr/share/gazebo-11/models"

if ! ros2 pkg prefix rtabmap_slam >/dev/null 2>&1; then
  echo "rtabmap_slam is not installed. In the container run:"
  echo "  apt-get update && apt-get install -y ros-humble-rtabmap-ros"
  exit 1
fi

if [ ! -d /root/.gazebo/models/ugv_description ]; then
  echo "Installing ugv_description meshes into ~/.gazebo/models ..."
  mkdir -p /root/.gazebo/models
  unzip -qo /root/ugv_ws/ugv_description.zip -d /root/.gazebo/models
fi

# RViz RobotModel uses package://ugv_description/... — must be built.
NEED_BUILD=()
if [ ! -f /root/ugv_ws/install/ugv_gazebo/share/ugv_gazebo/package.sh ] \
   && [ ! -d /root/ugv_ws/install/ugv_gazebo ] \
   || [ ! -f "/root/ugv_ws/install/ugv_gazebo/share/ugv_gazebo/urdf/${UGV_MODEL}.urdf" ]; then
  NEED_BUILD+=(ugv_gazebo)
fi
if [ ! -d /root/ugv_ws/install/ugv_description ] \
   || [ ! -f "/root/ugv_ws/install/ugv_description/share/ugv_description/urdf/${UGV_MODEL}.urdf" ]; then
  NEED_BUILD+=(ugv_description)
fi
if [ "${#NEED_BUILD[@]}" -gt 0 ]; then
  echo "Building: ${NEED_BUILD[*]}"
  colcon build --packages-select "${NEED_BUILD[@]}" --symlink-install
fi

source /root/ugv_ws/install/setup.bash

echo "Launching RTAB-Map SLAM + RViz (map_server must stay OFF in other terminals)"
exec ros2 launch ugv_gazebo rtabmap_rgbd.launch.py use_rviz:=true "$@"