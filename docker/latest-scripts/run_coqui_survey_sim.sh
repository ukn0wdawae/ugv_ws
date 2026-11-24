#!/usr/bin/env bash
# Terminal 1 of the Coqui survey: the SIMULATED WORLD (forest or hill + 6 coqui frogs + realistic audio),
# with the Gazebo window opening behind the robot and following it.
# Run inside the container (./docker/shell.sh).
#
#   ./docker/latest-scripts/run_coqui_survey_sim.sh                      # forest, Gazebo GUI follows robot
#   ./docker/latest-scripts/run_coqui_survey_sim.sh profile:=hill        # hill switchback world
#   ./docker/latest-scripts/run_coqui_survey_sim.sh follow:=false        # Gazebo GUI, free camera
#   ./docker/latest-scripts/run_coqui_survey_sim.sh gui:=false           # headless Gazebo
#   ./docker/latest-scripts/run_coqui_survey_sim.sh show_truth:=false    # hide true frog markers in RViz
#   ./docker/latest-scripts/run_coqui_survey_sim.sh seed:=3 randomize:=true
#
# Then in terminal 2: ./docker/latest-scripts/run_coqui_survey.sh
set -eo pipefail
cd /root/ugv_ws
source /opt/ros/humble/setup.bash
export UGV_MODEL=ugv_pro_audio GAZEBO_MODEL_DATABASE_URI=""
DOMAIN=${ROS_DOMAIN_ID:-0}
SESSION_FILE="/tmp/coqui_sim_session_${DOMAIN}.env"

# Parse profile / seed / randomize from launch args for Terminal 2 verification
PROFILE="forest"
SEED="0"
RANDOMIZE="false"
for arg in "$@"; do
  case "$arg" in
    profile:=*)   PROFILE="${arg#profile:=}" ;;
    seed:=*)      SEED="${arg#seed:=}" ;;
    randomize:=*) RANDOMIZE="${arg#randomize:=}" ;;
  esac
done

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

echo "Building ugv_description ugv_gazebo ugv_audio ugv_traversability ugv_coqui_behavior ..."
colcon build --symlink-install --packages-select ugv_description ugv_gazebo ugv_audio ugv_traversability ugv_coqui_behavior
source /root/ugv_ws/install/setup.bash

cat > "$SESSION_FILE" <<EOF
SIM_PROFILE=${PROFILE}
SIM_SEED=${SEED}
SIM_RANDOMIZE=${RANDOMIZE}
SIM_DOMAIN=${DOMAIN}
SIM_GAZEBO_MASTER=${MASTER}
SIM_PID=$$
EOF
trap 'rm -f "$SESSION_FILE"' EXIT

echo
echo "Gazebo opens behind the robot and follows it once it has spawned (profile=${PROFILE}, seed=${SEED}, randomize=${RANDOMIZE})."
echo "Esc in the Gazebo view stops following; right-click the robot > Follow follows it again."
echo "When the robot is on the ground, start terminal 2:"
echo "  ./docker/latest-scripts/run_coqui_survey.sh profile:=${PROFILE}"
echo
ros2 launch ugv_coqui_behavior coqui_survey_sim.launch.py gui:=true "$@"
