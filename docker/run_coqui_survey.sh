#!/usr/bin/env bash
# Terminal 2 of the Coqui survey: the ROBOT (traversability + Nav2 + microphone array + pan-tilt camera
# + survey mission) and RViz showing every sensor and what the mission is doing.
# Start ./docker/run_coqui_survey_sim.sh first and wait for the robot to appear in Gazebo.
# Run inside the container (./docker/shell.sh).
#
#   ./docker/run_coqui_survey.sh                                  # no SLAM (sim only: odom is exact)
#   ./docker/run_coqui_survey.sh mission:=false                   # everything but the mission: drive it yourself
#   ./docker/run_coqui_survey.sh run_id:=day1
#   ./docker/run_coqui_survey.sh resume_survey:=/root/ugv_ws/coqui_surveys/day1/survey.yaml
#   ./docker/run_coqui_survey.sh localization:=slam               # with RTAB-Map (not tested yet)
#
# Results: /root/ugv_ws/coqui_surveys/<run_id>/  (survey.yaml, frogs.geojson, photos/, events.jsonl)
# Grade a finished run against the simulation truth:
#   ros2 run ugv_coqui_behavior eval_survey /root/ugv_ws/coqui_surveys/<run_id>/survey.yaml
set -eo pipefail
cd /root/ugv_ws
source /opt/ros/humble/setup.bash
source /root/ugv_ws/install/setup.bash
export UGV_MODEL=ugv_pro_audio

# Is the simulation (terminal 1) up? The robot's TF comes from Gazebo.
if ! timeout 15 ros2 topic echo --once /clock rosgraph_msgs/msg/Clock >/dev/null 2>&1; then
  echo "No /clock: start the simulation first (terminal 1: ./docker/run_coqui_survey_sim.sh)."
  exit 1
fi
# Without SLAM this launch publishes map->odom itself; a leftover RTAB-Map would publish a second one.
if [[ " $* " != *"localization:=slam"* ]] && ros2 node list 2>/dev/null | grep -q '/rtabmap$'; then
  echo "An RTAB-Map node is still running (it publishes map->odom too). Stop it first, or use localization:=slam."
  exit 1
fi

exec ros2 launch ugv_coqui_behavior coqui_survey.launch.py rviz:=true "$@"
