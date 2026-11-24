#!/usr/bin/env bash
# Terminal 1 of the Coqui survey: the SIMULATED WORLD (forest or hill + 6 coqui frogs + realistic audio),
# with the Gazebo window opening behind the robot and following it.
# Run inside the container (./docker/shell.sh).
#
#   ./docker/latest-scripts/run_coqui_survey_sim.sh                      # forest, fresh random frog locations each run
#   ./docker/latest-scripts/run_coqui_survey_sim.sh profile:=hill        # hill switchback world (fresh random frogs)
#   ./docker/latest-scripts/run_coqui_survey_sim.sh follow:=false        # Gazebo GUI, free camera
#   ./docker/latest-scripts/run_coqui_survey_sim.sh gui:=false           # headless Gazebo
#   ./docker/latest-scripts/run_coqui_survey_sim.sh show_truth:=false    # hide true frog markers in RViz
#   ./docker/latest-scripts/run_coqui_survey_sim.sh seed:=42             # replay a specific randomized frog layout
#   ./docker/latest-scripts/run_coqui_survey_sim.sh randomize:=false     # fixed canonical frog layout (seed=0)
#
# Then in terminal 2: ./docker/latest-scripts/run_coqui_survey.sh
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_cleanup_lib.sh"

cd /root/ugv_ws
source /opt/ros/humble/setup.bash
export UGV_MODEL=ugv_pro_audio GAZEBO_MODEL_DATABASE_URI="" ALSOFT_DRIVERS="${ALSOFT_DRIVERS:-null}"
DOMAIN=${ROS_DOMAIN_ID:-0}
SESSION_FILE="/tmp/coqui_sim_session_${DOMAIN}.env"

# Parse profile / seed / randomize from launch args (defaults: fresh random frog layout on every run!)
PROFILE="forest"
SEED=""
RANDOMIZE="true"
HAS_SEED=0
HAS_RANDOMIZE=0
HAS_TRUTH_OUT=0
for arg in "$@"; do
  case "$arg" in
    profile:=*)   PROFILE="${arg#profile:=}" ;;
    seed:=*)      SEED="${arg#seed:=}"; HAS_SEED=1 ;;
    randomize:=*) RANDOMIZE="${arg#randomize:=}"; HAS_RANDOMIZE=1 ;;
    truth_out:=*) TRUTH_OUT="${arg#truth_out:=}"; HAS_TRUTH_OUT=1 ;;
  esac
done

case "${RANDOMIZE,,}" in
  false|0|no)
    RANDOMIZE="false"
    [ "$HAS_SEED" -eq 0 ] && SEED="0"
    ;;
  *)
    RANDOMIZE="true"
    if [ "$HAS_SEED" -eq 1 ] && [ "$SEED" = "0" ] && [ "$HAS_RANDOMIZE" -eq 0 ]; then
      # Explicit seed:=0 without randomize:=true selects the canonical static baseline
      RANDOMIZE="false"
    elif [ -z "$SEED" ] || [ "${SEED,,}" = "auto" ] || [ "$SEED" = "0" ]; then
      SEED="$(python3 -c 'import secrets; print(secrets.randbelow(900000) + 100000)')"
    fi
    ;;
esac
[ "$HAS_TRUTH_OUT" -eq 0 ] && TRUTH_OUT="/tmp/coqui_latest_truth_${PROFILE}_d${DOMAIN}.yaml"

# 1) Check Gazebo port is free
MASTER=${GAZEBO_MASTER_URI:-http://localhost:11345}
PORT=${MASTER##*:}
if (exec 3<>"/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null; then
  echo "A Gazebo is already running on port ${PORT}."
  echo "Stop it first (Ctrl+C in its terminal), and also stop anything started for it (RTAB-Map, RViz, Nav2),"
  echo "then run ./docker/latest-scripts/run_coqui_survey_sim.sh again."
  exit 1
fi

# 2) Check for leftover sound emitter or survey nodes in this ROS domain
STALE_NODES=$(timeout 4 ros2 node list 2>/dev/null | grep -E '^/(coqui_sound_emitter|coqui_survey|bt_navigator|terrain_traversability|rtabmap)$' || true)
if [ -n "$STALE_NODES" ]; then
  echo "Stale ROS nodes are still running in ROS_DOMAIN_ID=${DOMAIN}:"
  echo "$STALE_NODES"
  echo "Stop those processes before starting a fresh simulation."
  exit 1
fi

if [ ! -d /root/.gazebo/models/ugv_description ]; then
  echo "Installing ugv_description meshes into ~/.gazebo/models ..."
  mkdir -p /root/.gazebo/models
  unzip -qo /root/ugv_ws/ugv_description.zip -d /root/.gazebo/models
fi

PKGS=(ugv_description ugv_gazebo ugv_audio ugv_traversability ugv_coqui_behavior)
if [ -d /root/ugv_ws/src/ugv_else/robot_pose_publisher ]; then
  PKGS+=(robot_pose_publisher)
fi
echo "Building ${PKGS[*]} ..."
colcon build --symlink-install --packages-select "${PKGS[@]}"
source /root/ugv_ws/install/setup.bash

ugv_init_launch_cleanup "coqui_sim" "$DOMAIN" "$SESSION_FILE"
cat > "$SESSION_FILE" <<EOF
SIM_PROFILE=${PROFILE}
SIM_SEED=${SEED}
SIM_RANDOMIZE=${RANDOMIZE}
SIM_TRUTH_FILE=${TRUTH_OUT}
SIM_DOMAIN=${DOMAIN}
SIM_GAZEBO_MASTER=${MASTER}
SIM_PID=$$
EOF

echo
echo "Gazebo opens behind the robot and follows it once it has spawned (profile=${PROFILE}, seed=${SEED}, randomize=${RANDOMIZE})."
echo "Active ground-truth answer key (for eval_survey): ${TRUTH_OUT}"
echo "To replay this exact frog layout later, pass: seed:=${SEED}"
echo "Esc in the Gazebo view stops following; right-click the robot > Follow follows it again."
echo "When the robot is on the ground, start terminal 2:"
echo "  ./docker/latest-scripts/run_coqui_survey.sh profile:=${PROFILE}"
echo
LAUNCH_EXTRA=("seed:=${SEED}" "randomize:=${RANDOMIZE}")
[ "$HAS_TRUTH_OUT" -eq 0 ] && LAUNCH_EXTRA+=("truth_out:=${TRUTH_OUT}")
ugv_run_foreground ros2 launch ugv_coqui_behavior coqui_survey_sim.launch.py gui:=true "${LAUNCH_EXTRA[@]}" "$@"
