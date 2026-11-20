#!/usr/bin/env bash
# Isolated headless hill test (own Gazebo port + ROS domain). Usage: .hilltest.sh <world file>
WORLD="$1"
export GAZEBO_MASTER_URI=http://localhost:11499
export ROS_DOMAIN_ID=77
cd /root/ugv_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export GAZEBO_MODEL_DATABASE_URI=""
export GAZEBO_MODEL_PATH="/root/ugv_ws/gazebo_models:/root/.gazebo/models:/root/ugv_ws/src/ugv_main/ugv_gazebo/models:/usr/share/gazebo-11/models"
export UGV_MODEL=ugv_rover

gzserver "$WORLD" -slibgazebo_ros_init.so -slibgazebo_ros_factory.so > /tmp/hilltest_gz.log 2>&1 &
GZ=$!
ros2 run gazebo_ros spawn_entity.py -entity ugv_rover -timeout 300 \
  -file src/ugv_main/ugv_gazebo/models/ugv_rover/model.sdf \
  -x 43.0 -y -12.88 -z 2.36 -Y -3.0 > /tmp/hilltest_spawn.log 2>&1
for i in $(seq 1 60); do gz model -m ugv_rover -p >/dev/null 2>&1 && break; sleep 1; done

pose() { gz model -m ugv_rover -p; }
place() { gz model -m ugv_rover -x "$1" -y "$2" -z "$3" -R 0 -P 0 -Y "$4"; sleep 2; }
drive() {  # linear angular seconds
  timeout "$3" ros2 topic pub -r 20 /cmd_vel geometry_msgs/msg/Twist \
    "{linear: {x: $1}, angular: {z: $2}}" >/dev/null 2>&1
  timeout 2 ros2 topic pub -r 20 -t 10 /cmd_vel geometry_msgs/msg/Twist "{}" >/dev/null 2>&1
}

echo "== 1) idle at spawn (43, -12.88), 8 s  [x y z roll pitch yaw]"
place 43.0 -12.88 2.36 -3.0
for i in 1 2 3 4 5 6 7 8; do pose; sleep 1; done
echo "== 2) cross north crack at x=43 heading +y, 0.2 m/s 6 s"
place 43.0 -12.7 2.36 1.5708; pose; drive 0.2 0 6; pose
echo "== 3) cross south crack at x=44 heading -y, 0.2 m/s 6 s"
place 44.0 -13.0 2.37 -1.5708; pose; drive 0.2 0 6; pose
echo "== 4) east into boundary wall (inner face x=49.0), 0.26 m/s 12 s"
place 47.5 -12.9 2.25 0.0; pose; drive 0.26 0 12; pose
echo "== 5) uphill along path from spawn (heading -x), 0.26 m/s 15 s"
place 43.0 -12.88 2.36 -3.0; pose; drive 0.26 0 15; pose

kill $GZ; sleep 2; kill -9 $GZ 2>/dev/null
echo done
