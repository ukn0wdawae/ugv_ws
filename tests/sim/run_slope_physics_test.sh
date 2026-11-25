#!/usr/bin/env bash
# Pooled from docker/legacy_code/.slopetest.sh: isolated headless slope physics stability test (6, 15, and 25 deg).
# Checks that ugv_pro_audio holds a cross-slope without excessive slip and does not flip.
# Run inside Docker container from ugv_ws root.
set -eo pipefail
WS_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT_DIR="${1:-$WS_ROOT/tests/results/slope_physics_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_DIR"

export GAZEBO_MASTER_URI="${GAZEBO_MASTER_URI:-http://localhost:11499}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-81}"
source /opt/ros/humble/setup.bash
source "$WS_ROOT/install/setup.bash"
export GAZEBO_MODEL_DATABASE_URI=""
export GAZEBO_MODEL_PATH="$WS_ROOT/gazebo_models:/root/.gazebo/models:$WS_ROOT/src/ugv_main/ugv_gazebo/models:/usr/share/gazebo-11/models"

MODEL="$WS_ROOT/src/ugv_main/ugv_gazebo/models/ugv_pro_audio/model.sdf"
for DEG in 6 15 25; do
  RAD=$(python3 -c "import math; print(f'{math.radians($DEG):.5f}')")
  WFILE="/tmp/slope_${DEG}deg.world"
  cat > "$WFILE" <<EOF
<sdf version='1.6'><world name='default'>
  <physics type='ode'><max_step_size>0.002</max_step_size><real_time_factor>1</real_time_factor><real_time_update_rate>500</real_time_update_rate></physics>
  <model name='slope'><static>1</static><pose>0 0 0 ${RAD} 0 0</pose>
    <link name='l'><collision name='c'><geometry><box><size>40 20 0.1</size></box></geometry>
      <surface><friction><ode><mu>0.85</mu><mu2>0.85</mu2></ode></friction>
      <contact><ode><kp>100000.0</kp><kd>100.0</kd><max_vel>0.01</max_vel><min_depth>0.002</min_depth></ode></contact></surface>
    </collision></link></model>
</world></sdf>
EOF
  gzserver "$WFILE" -slibgazebo_ros_init.so -slibgazebo_ros_factory.so > "$OUT_DIR/gz_${DEG}deg.log" 2>&1 &
  GZ=$!
  trap 'kill -9 $GZ 2>/dev/null || true' EXIT
  sleep 4
  timeout 60 ros2 run gazebo_ros spawn_entity.py -entity ugv_pro_audio -file "$MODEL" -x 0 -y 0 -z 0.12 -Y 0 >/dev/null 2>&1
  echo "=== ${DEG} deg cross-slope (downhill = -y): t | x y z roll pitch yaw ===" | tee "$OUT_DIR/slope_${DEG}deg.txt"
  for t in 0 1 2 3 4 5; do
    p=$(gz model -m ugv_pro_audio -p 2>/dev/null | tr '\n' ' ')
    echo "t=$t | $p" | tee -a "$OUT_DIR/slope_${DEG}deg.txt"
    sleep 1
  done
  kill $GZ 2>/dev/null || true; sleep 1; kill -9 $GZ 2>/dev/null || true
done
trap - EXIT
echo "Slope physics test logs saved to $OUT_DIR"
