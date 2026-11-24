#!/usr/bin/env bash
# Master runner for the pooled ugv_ws /tests suite.
# Test ANY new navigation style against every offline, ROS 2, and Gazebo regression case:
#
#   ./tests/run_all.sh --offline
#   ./tests/run_all.sh --offline --style slam_nav_traversability
#   ./tests/run_all.sh --offline --nav2-params path/to/new_nav2.yaml --bt-xml path/to/new_bt.xml
#   ./tests/run_all.sh --ros     --nav2-params path/to/new_nav2.yaml
#   ./tests/run_all.sh --sim     --suite smoke,slope --nav2-params path/to/new_nav2.yaml
set -eo pipefail
WS_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WS_ROOT"

MODE="offline"
STYLE="survey_dwb_default"
NAV2_PARAMS=""
BT_XML=""
TRAV_PARAMS=""
SUITE="smoke,slope"
LOC="ekf"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --offline) MODE="offline"; shift ;;
    --ros)     MODE="ros"; shift ;;
    --sim)     MODE="sim"; shift ;;
    --all)     MODE="all"; shift ;;
    --style)   STYLE="$2"; shift 2 ;;
    --nav2-params) NAV2_PARAMS="$2"; shift 2 ;;
    --bt-xml)  BT_XML="$2"; shift 2 ;;
    --traversability-params) TRAV_PARAMS="$2"; shift 2 ;;
    --suite)   SUITE="$2"; shift 2 ;;
    --localization) LOC="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1"; exit 2 ;;
  esac
done

run_offline() {
  echo "=== [1/2] Package unit tests (ugv_coqui_behavior + ugv_traversability) ==="
  python3 -m pytest -q src/ugv_main/ugv_coqui_behavior/test src/ugv_main/ugv_traversability/test
  echo "=== [2/2] Pooled /tests/offline regression suite (style=$STYLE) ==="
  PYTEST_ARGS=(tests/offline --style "$STYLE" -v)
  [ -n "$NAV2_PARAMS" ] && PYTEST_ARGS+=(--nav2-params "$NAV2_PARAMS")
  [ -n "$BT_XML" ] && PYTEST_ARGS+=(--bt-xml "$BT_XML")
  [ -n "$TRAV_PARAMS" ] && PYTEST_ARGS+=(--traversability-params "$TRAV_PARAMS")
  python3 -m pytest "${PYTEST_ARGS[@]}"
}

run_ros() {
  echo "=== Live ROS 2 Humble Nav2 costmap + launch resolution tests ==="
  source /opt/ros/humble/setup.bash
  [ -f "$WS_ROOT/install/setup.bash" ] && source "$WS_ROOT/install/setup.bash"
  export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-89}"
  local n2="${NAV2_PARAMS:-$WS_ROOT/src/ugv_main/ugv_coqui_behavior/config/nav2_survey_base.yaml}"
  python3 "$WS_ROOT/tests/ros/test_nav2_costmap_layers.py" --nav2-params "$n2"
  python3 "$WS_ROOT/tests/ros/test_launch_resolution.py"
}

run_sim() {
  echo "=== Headless Gazebo benchmark suite ($SUITE) ==="
  "$WS_ROOT/tests/sim/run_nav_benchmark.sh" --suite "$SUITE" --localization "$LOC" \
    ${NAV2_PARAMS:+--nav2-params "$NAV2_PARAMS"} \
    ${BT_XML:+--bt-xml "$BT_XML"} \
    ${TRAV_PARAMS:+--traversability-params "$TRAV_PARAMS"}
}

case "$MODE" in
  offline) run_offline ;;
  ros)     run_ros ;;
  sim)     run_sim ;;
  all)     run_offline; run_ros; run_sim ;;
esac
