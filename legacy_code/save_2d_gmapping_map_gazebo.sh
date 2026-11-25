#!/bin/bash
cd /home/ws/ugv_ws/src/ugv_main/ugv_gazebo/legacy_code/maps
ros2 run nav2_map_server map_saver_cli -f ./map
cd -

