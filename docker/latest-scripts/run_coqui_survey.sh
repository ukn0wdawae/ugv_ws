#!/usr/bin/env bash
# Terminal 2 of the Coqui survey: the ROBOT (EKF/SLAM + traversability + Nav2 + collision monitor
# + microphone array + pan-tilt camera + survey mission) and RViz.
# Start ./docker/latest-scripts/run_coqui_survey_sim.sh first and wait for the robot to appear in Gazebo.
# Run inside the container (./docker/shell.sh).
#
#   ./docker/latest-scripts/run_coqui_survey.sh                       # lidar localization, GPS-denied (default)
#   ./docker/latest-scripts/run_coqui_survey.sh localization:=ekf     # dead reckoning only (drifts)
#   ./docker/latest-scripts/run_coqui_survey.sh localization:=slam    # EKF + RTAB-Map 3D SLAM
#   ./docker/latest-scripts/run_coqui_survey.sh localization:=perfect # ground-truth ablation mode
#   ./docker/latest-scripts/run_coqui_survey.sh mission:=false        # drive manually
#   ./docker/latest-scripts/run_coqui_survey.sh run_id:=day1
#   ./docker/latest-scripts/run_coqui_survey.sh resume_survey:=/root/ugv_ws/coqui_surveys/day1/survey.yaml
#
# Results: /root/ugv_ws/coqui_surveys/<run_id>/  (survey.yaml, frogs.geojson, photos/, events.jsonl)
# Grade a finished run against the simulation truth:
#   ros2 run ugv_coqui_behavior eval_survey /root/ugv_ws/coqui_surveys/<run_id>/survey.yaml --profile <profile>
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_cleanup_lib.sh"

cd /root/ugv_ws
source /opt/ros/humble/setup.bash
source /root/ugv_ws/install/setup.bash
export UGV_MODEL=ugv_pro_audio ALSOFT_DRIVERS="${ALSOFT_DRIVERS:-null}"
DOMAIN=${ROS_DOMAIN_ID:-0}
SESSION_FILE="/tmp/coqui_sim_session_${DOMAIN}.env"

# 1) Check simulation (Terminal 1) is up
if ! timeout 15 ros2 topic echo --once /clock rosgraph_msgs/msg/Clock >/dev/null 2>&1; then
  echo "No /clock in ROS_DOMAIN_ID=${DOMAIN}: start the simulation first (terminal 1):"
  echo "  ./docker/latest-scripts/run_coqui_survey_sim.sh"
  exit 1
fi

# 2) Verify profile matches Terminal 1 (or auto-adopt Terminal 1's profile if omitted)
USER_PROFILE=""
HAS_PROFILE=0
for arg in "$@"; do
  case "$arg" in
    profile:=*) USER_PROFILE="${arg#profile:=}"; HAS_PROFILE=1 ;;
  esac
done

EXTRA_ARGS=()
if [ -f "$SESSION_FILE" ]; then
  # shellcheck disable=SC1090
  source "$SESSION_FILE"
  if [ "$HAS_PROFILE" -eq 1 ] && [ -n "${SIM_PROFILE:-}" ] && [ "$USER_PROFILE" != "$SIM_PROFILE" ]; then
    echo "Profile mismatch! Terminal 1 is running profile:=${SIM_PROFILE}, but you passed profile:=${USER_PROFILE}."
    echo "Use: ./docker/latest-scripts/run_coqui_survey.sh profile:=${SIM_PROFILE}"
    exit 1
  elif [ "$HAS_PROFILE" -eq 0 ] && [ -n "${SIM_PROFILE:-}" ]; then
    echo "Auto-selecting profile:=${SIM_PROFILE} to match Terminal 1 (seed=${SIM_SEED:-?}, randomize=${SIM_RANDOMIZE:-?})."
    EXTRA_ARGS+=("profile:=${SIM_PROFILE}")
  fi
fi

# 3) Check no stale robot-side nodes are already running in this ROS domain
STALE=$(timeout 4 ros2 node list 2>/dev/null | grep -E '^/(coqui_survey|bt_navigator|terrain_traversability|lidar_localizer|ekf_filter_node|rtabmap)$' || true)
if [ -n "$STALE" ]; then
  echo "Stale robot-side ROS nodes are still running in ROS_DOMAIN_ID=${DOMAIN}:"
  echo "$STALE"
  echo "Stop them first before starting Terminal 2."
  exit 1
fi

# Ensure robot_pose_publisher is built if localization:=slam is requested
for arg in "$@"; do
  if [ "$arg" = "localization:=slam" ] && ! ros2 pkg prefix robot_pose_publisher >/dev/null 2>&1; then
    echo "Building robot_pose_publisher for localization:=slam ..."
    colcon build --symlink-install --packages-select robot_pose_publisher
    source /root/ugv_ws/install/setup.bash
    break
  fi
done

WATCH_SESSION=""
[ -f "$SESSION_FILE" ] && WATCH_SESSION="$SESSION_FILE"
ugv_init_launch_cleanup "coqui_bot" "$DOMAIN" "" "$WATCH_SESSION"
ugv_run_foreground ros2 launch ugv_coqui_behavior coqui_survey.launch.py rviz:=true "${EXTRA_ARGS[@]}" "$@"
