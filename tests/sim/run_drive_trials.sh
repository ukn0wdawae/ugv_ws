#!/usr/bin/env bash
# Pooled from docs/forest_traversability/test_harness/run_world_trials.sh:
# Runs all 24 forest drive trials (edge_off, edge_on, open, vegetation) from tests/cases/forest_drive_trials.json
# using src/ugv_main/ugv_gazebo/scripts/forest_traversability/sim_drive_test.py.
# Run inside Docker container from ugv_ws root.
set -eo pipefail
WS_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT_DIR="${1:-$WS_ROOT/tests/results/drive_trials_$(date +%Y%m%d_%H%M%S)}"
shift || true
mkdir -p "$OUT_DIR"

export GAZEBO_MASTER_URI="${GAZEBO_MASTER_URI:-http://localhost:11497}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-83}"
source /opt/ros/humble/setup.bash
source "$WS_ROOT/install/setup.bash"
export UGV_MODEL=ugv_pro_audio GAZEBO_MODEL_DATABASE_URI=""
export GAZEBO_MODEL_PATH="$WS_ROOT/gazebo_models:/root/.gazebo/models:$WS_ROOT/src/ugv_main/ugv_gazebo/models:/usr/share/gazebo-11/models"

SRC_WORLD="$WS_ROOT/src/ugv_main/ugv_gazebo/worlds/forest_coqui_survey.world"
TMP_WORLD="$OUT_DIR/forest_with_state.world"
python3 - "$SRC_WORLD" "$TMP_WORLD" <<'PY'
import re, sys
t = open(sys.argv[1], encoding='utf-8').read()
plug = ("<plugin name='gazebo_ros_state' filename='libgazebo_ros_state.so'>"
        "<ros><namespace>/</namespace></ros><update_rate>50</update_rate></plugin>")
if 'libgazebo_ros_state.so' not in t:
    t = re.sub(r"(<world\s+name=['\"][^'\"]+['\"]>)", r"\1" + plug, t, count=1)
open(sys.argv[2], 'w', encoding='utf-8').write(t)
PY

setsid ros2 launch gazebo_ros gzserver.launch.py world:="$TMP_WORLD" verbose:=false > "$OUT_DIR/gz.log" 2>&1 &
GZ_PID=$!
trap 'kill -INT -- -$GZ_PID 2>/dev/null || true; sleep 3; kill -9 -- -$GZ_PID 2>/dev/null || true' EXIT
sleep 5
ros2 launch ugv_gazebo spawn_ugv.launch.py x_pose:=20.0 y_pose:=3.7 z_pose:=2.86 yaw_pose:=1.57 start_ekf:=false > "$OUT_DIR/spawn.log" 2>&1 &
sleep 8

python3 "$WS_ROOT/src/ugv_main/ugv_gazebo/scripts/forest_traversability/sim_drive_test.py" \
  "$WS_ROOT/tests/cases/forest_drive_trials.json" "$OUT_DIR/results.json" --entity ugv_pro_audio --z-key z_fixed "$@" | tee "$OUT_DIR/summary.jsonl"
echo "Drive trials saved to $OUT_DIR/results.json"
