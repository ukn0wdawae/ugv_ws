#!/usr/bin/env bash
# Pooled from docker/.hilltest.sh: isolated headless hill terrain contact test
# (idle at spawn, north/south cracks, boundary wall, uphill climb along switchback path).
# Run inside Docker container from ugv_ws root.
set -eo pipefail
WS_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WORLD="${1:-$WS_ROOT/src/ugv_main/ugv_gazebo/worlds/hill_coqui_survey.world}"
OUT_DIR="${2:-$WS_ROOT/tests/results/hill_terrain_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_DIR"

export GAZEBO_MASTER_URI="${GAZEBO_MASTER_URI:-http://localhost:11498}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-82}"
source /opt/ros/humble/setup.bash
source "$WS_ROOT/install/setup.bash"
export GAZEBO_MODEL_DATABASE_URI=""
export GAZEBO_MODEL_PATH="$WS_ROOT/gazebo_models:/root/.gazebo/models:$WS_ROOT/src/ugv_main/ugv_gazebo/models:/usr/share/gazebo-11/models"
export UGV_MODEL=ugv_pro_audio

gzserver "$WORLD" -slibgazebo_ros_init.so -slibgazebo_ros_factory.so > "$OUT_DIR/gz.log" 2>&1 &
GZ=$!
trap 'kill $GZ 2>/dev/null || true; sleep 2; kill -9 $GZ 2>/dev/null || true' EXIT

ros2 run gazebo_ros spawn_entity.py -entity ugv_pro_audio -timeout 300 \
  -file "$WS_ROOT/src/ugv_main/ugv_gazebo/models/ugv_pro_audio/model.sdf" \
  -x 43.0 -y -12.88 -z 2.45 -Y -3.0 > "$OUT_DIR/spawn.log" 2>&1
for _ in $(seq 1 60); do gz model -m ugv_pro_audio -p >/dev/null 2>&1 && break; sleep 1; done

pose() { gz model -m ugv_pro_audio -p | tr '\n' ' '; echo; }
place() { gz model -m ugv_pro_audio -x "$1" -y "$2" -z "$3" -R 0 -P 0 -Y "$4"; sleep 2; }
drive() {
  timeout "$3" ros2 topic pub -r 20 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: $1}, angular: {z: $2}}" >/dev/null 2>&1 || true
  timeout 2 ros2 topic pub -r 20 -t 10 /cmd_vel geometry_msgs/msg/Twist "{}" >/dev/null 2>&1 || true
}

{
  echo "== 1) idle at spawn (43, -12.88), 6 s [x y z roll pitch yaw]"
  place 43.0 -12.88 2.45 -3.0
  for _ in 1 2 3 4 5 6; do pose; sleep 1; done
  echo "== 2) cross north crack at x=43 heading +y, 0.2 m/s 6 s"
  place 43.0 -12.7 2.45 1.5708; pose; drive 0.2 0 6; pose
  echo "== 3) cross south crack at x=44 heading -y, 0.2 m/s 6 s"
  place 44.0 -13.0 2.45 -1.5708; pose; drive 0.2 0 6; pose
  echo "== 4) east into boundary wall (inner face x=49.0), 0.26 m/s 10 s"
  place 47.5 -12.9 2.35 0.0; pose; drive 0.26 0 10; pose
  echo "== 5) uphill along path from spawn (heading -x), 0.26 m/s 12 s"
  place 43.0 -12.88 2.45 -3.0; pose; drive 0.26 0 12; pose
} | tee "$OUT_DIR/hill_terrain.txt"
echo "Hill terrain test complete: $OUT_DIR/hill_terrain.txt"
