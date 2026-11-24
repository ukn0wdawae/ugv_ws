#!/usr/bin/env bash
# Headless Coqui survey trial: world + robot + frog calls + full robot stack, no GUI, then grade it.
# Runs in its own ROS domain and Gazebo port so it does not interfere with an interactive session.
# Run inside the container (./docker/shell.sh), from /root/ugv_ws.
#
#   ./docker/run_coqui_survey_trial.sh [run_id] [seed] [max_mission_sec] [localization] [profile] [randomize] [extra launch args...]
#
# Examples:
#   ./docker/run_coqui_survey_trial.sh smoke1 0 600 lidar forest false
#   ./docker/run_coqui_survey_trial.sh rand_s3 3 2400 lidar forest true
#   ./docker/run_coqui_survey_trial.sh custom_nav 0 1200 lidar hill false nav2_params_file:=/path/to/my_nav2.yaml
#
# Output: /root/ugv_ws/coqui_surveys/<run_id>/ (survey.yaml, frogs.geojson, photos/, events.jsonl,
#         launch.log, grade.json, truth_used.yaml)
set -eo pipefail
RUN=${1:-trial_$(date +%H%M%S)}
SEED=${2:-0}
MAXSEC=${3:-2400}
LOC=${4:-lidar}
PROFILE=${5:-forest}
RANDOMIZE=${6:-false}
shift $(( $# > 6 ? 6 : $# ))
OUT=/root/ugv_ws/coqui_surveys/$RUN
mkdir -p "$OUT"

source /opt/ros/humble/setup.bash
source /root/ugv_ws/install/setup.bash
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-77}
export GAZEBO_MASTER_URI=${GAZEBO_MASTER_URI:-http://localhost:11399}
export UGV_MODEL=ugv_pro_audio GAZEBO_MODEL_DATABASE_URI=""

# Refuse to run if stale nodes exist in this trial ROS domain
STALE=$(timeout 4 ros2 node list 2>/dev/null | grep -E '^/(coqui_sound_emitter|coqui_survey|bt_navigator)$' || true)
if [ -n "$STALE" ]; then
  echo "Stale nodes running in ROS_DOMAIN_ID=${ROS_DOMAIN_ID}: $STALE"
  exit 1
fi

TRUTH_ARGS=()
GRADE_TRUTH_ARGS=(--profile "$PROFILE")
if [ "$RANDOMIZE" = "true" ] && [ "$SEED" != "0" ]; then
  TRUTH_OUT="$OUT/truth_used.yaml"
  TRUTH_ARGS+=(randomize:=true "truth_out:=$TRUTH_OUT")
  GRADE_TRUTH_ARGS=(--truth "$TRUTH_OUT")
  # The randomized spawn is where the operator puts the robot down: tell the robot that, not the profile's trailhead
  # (otherwise its whole map, and every frog in it, is shifted by the difference).
  TRUTH_ARGS+=($(python3 - "$PROFILE" "$SEED" <<'PY'
import os, sys, yaml
from ament_index_python.packages import get_package_share_directory as share
from ugv_audio.audio_core import randomize_truth_scenario
profile, seed = sys.argv[1], int(sys.argv[2])
truth = yaml.safe_load(open(os.path.join(share('ugv_audio'), 'config', 'sim_truth', f'{profile}_survey.yaml')))
prof = yaml.safe_load(open(os.path.join(share('ugv_coqui_behavior'), 'config', 'profiles', f'{profile}.yaml')))
area = prof['survey']['coqui_survey']['ros__parameters']['survey_area']
s = randomize_truth_scenario(truth, seed, survey_area=area)['spawn']
print(f'init_x:={s[0]} init_y:={s[1]} init_yaw:={s[3]}')
PY
))
fi

echo "trial $RUN profile=$PROFILE seed=$SEED randomize=$RANDOMIZE max=${MAXSEC}s localization=$LOC domain=$ROS_DOMAIN_ID"
setsid ros2 launch ugv_coqui_behavior coqui_survey.launch.py profile:="$PROFILE" sim:=true gui:=false seed:="$SEED" \
  run_id:="$RUN" max_mission_sec:="$MAXSEC" localization:="$LOC" "${TRUTH_ARGS[@]}" "$@" > "$OUT/launch.log" 2>&1 &
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
ros2 run ugv_coqui_behavior eval_survey "$OUT/survey.yaml" "${GRADE_TRUTH_ARGS[@]}" --json "$OUT/grade.json" | tee "$OUT/grade.txt"
