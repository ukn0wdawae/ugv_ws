#!/usr/bin/env bash
set -e

source /opt/ros/humble/setup.bash

if [ -f /root/ugv_ws/install/setup.bash ]; then
  source /root/ugv_ws/install/setup.bash
fi

export UGV_MODEL="${UGV_MODEL:-ugv_rover}"
export GAZEBO_MODEL_DATABASE_URI=""
export ALSOFT_DRIVERS="${ALSOFT_DRIVERS:-null}"
export GAZEBO_MODEL_PATH="/root/ugv_ws/gazebo_models:/root/.gazebo/models:/root/ugv_ws/src/ugv_main/ugv_gazebo/models:${GAZEBO_MODEL_PATH:-}:/usr/share/gazebo-11/models"

exec "$@"
