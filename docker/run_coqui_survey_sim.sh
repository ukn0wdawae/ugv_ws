#!/usr/bin/env bash
# Terminal 1 of the Coqui survey: the SIMULATED WORLD (big forest + 6 coqui frogs + their calls), with the
# Gazebo window opening behind the robot and following it.
# Run inside the container (./docker/shell.sh).
#
#   ./docker/run_coqui_survey_sim.sh                     # Gazebo GUI, view follows the robot
#   ./docker/run_coqui_survey_sim.sh follow:=false       # Gazebo GUI, free camera
#   ./docker/run_coqui_survey_sim.sh gui:=false          # no Gazebo window
#   ./docker/run_coqui_survey_sim.sh show_truth:=false   # do not draw the true frog positions in RViz
#   ./docker/run_coqui_survey_sim.sh seed:=3             # another calling schedule
#
# Then in terminal 2: ./docker/run_coqui_survey.sh   (Nav2, microphones, camera, mission + RViz)
set -eo pipefail
cd /root/ugv_ws
source /opt/ros/humble/setup.bash
export UGV_MODEL=ugv_pro_audio GAZEBO_MODEL_DATABASE_URI=""

# Only one Gazebo can own a master port. Starting a second one fails half-way with "Address already in use".
MASTER=${GAZEBO_MASTER_URI:-http://localhost:11345}
PORT=${MASTER##*:}
if (exec 3<>"/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null; then
  echo "A Gazebo is already running on port ${PORT} (another sim, e.g. run_forest_sim.sh)."
  echo "Stop it first (Ctrl+C in its terminal), and also stop anything started for it (RTAB-Map, RViz, Nav2),"
  echo "then run this script again."
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

echo
echo "Gazebo opens behind the robot and follows it once it has spawned. Esc in the Gazebo view stops following;"
echo "right-click the robot > Follow follows it again."
echo "When the robot is on the ground, start terminal 2:  ./docker/run_coqui_survey.sh"
echo
exec ros2 launch ugv_coqui_behavior coqui_survey_sim.launch.py gui:=true "$@"
