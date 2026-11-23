#!/usr/bin/env bash
# Headless Coqui survey trial: world + robot + frog calls + full robot stack, no GUI, then grade it.
# Runs in its own ROS domain and Gazebo port so it does not interfere with an interactive session.
# Run inside the container (./docker/shell.sh), from /root/ugv_ws.
#
#   ./docker/run_coqui_survey_trial.sh [run_id] [seed] [max_mission_sec] [localization]
#
# Output: /root/ugv_ws/coqui_surveys/<run_id>/ (survey.yaml, frogs.geojson, photos/, events.jsonl,
#         launch.log, grade.json)
set -eo pipefail
RUN=${1:-trial_$(date +%H%M%S)}
SEED=${2:-0}
MAXSEC=${3:-2400}
LOC=${4:-perfect}
OUT=/root/ugv_ws/coqui_surveys/$RUN
mkdir -p "$OUT"

source /opt/ros/humble/setup.bash
source /root/ugv_ws/install/setup.bash
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-77}
export GAZEBO_MASTER_URI=${GAZEBO_MASTER_URI:-http://localhost:11399}
export UGV_MODEL=ugv_pro_audio GAZEBO_MODEL_DATABASE_URI=""

echo "trial $RUN seed=$SEED max=${MAXSEC}s localization=$LOC domain=$ROS_DOMAIN_ID"
setsid ros2 launch ugv_coqui_behavior coqui_survey.launch.py sim:=true gui:=false seed:="$SEED" \
  run_id:="$RUN" max_mission_sec:="$MAXSEC" localization:="$LOC" > "$OUT/launch.log" 2>&1 &
PID=$!
trap 'kill -INT -- -$PID 2>/dev/null; sleep 5; kill -9 -- -$PID 2>/dev/null; true' EXIT

# wait for the mission to finish (DONE in survey.yaml) or the wall-clock budget (sim may run < 1x)
WALL_LIMIT=$(( MAXSEC * 3 + 600 ))
T0=$(date +%s)
while true; do
  sleep 20
  if grep -Eq "result: '?(area covered|time limit|no reachable|aborted|crashed)" "$OUT/survey.yaml" 2>/dev/null; then
    echo "mission finished: $(grep -m1 'result:' "$OUT/survey.yaml")"; break
  fi
  if ! kill -0 $PID 2>/dev/null; then echo "launch exited"; break; fi
  if (( $(date +%s) - T0 > WALL_LIMIT )); then echo "wall-clock limit"; break; fi
  tail -n 1 "$OUT/events.jsonl" 2>/dev/null | cut -c1-200 || true
done
ros2 run ugv_coqui_behavior eval_survey "$OUT/survey.yaml" --json "$OUT/grade.json" | tee "$OUT/grade.txt"
