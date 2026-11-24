#!/usr/bin/env bash
# Unified Gazebo navigation benchmark runner for testing ANY navigation style against all pooled
# historical scenarios (smoke, slope, hill, forest, drive_trials, slope_physics, hill_terrain).
#
# Run inside the Docker container (./docker/shell.sh) from ugv_ws root:
#   ./tests/sim/run_nav_benchmark.sh --suite smoke,slope --localization ekf
#   ./tests/sim/run_nav_benchmark.sh --suite all --nav2-params /path/to/new_nav2.yaml --bt-xml /path/to/new_bt.xml
set -eo pipefail
WS_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SUITE="smoke,slope"
LOC="ekf"
NAV2_PARAMS=""
BT_XML=""
TRAV_PARAMS=""
TAG="nav_$(date +%Y%m%d_%H%M%S)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite)       SUITE="$2"; shift 2 ;;
    --localization) LOC="$2"; shift 2 ;;
    --nav2-params) NAV2_PARAMS="$2"; shift 2 ;;
    --bt-xml)      BT_XML="$2"; shift 2 ;;
    --traversability-params) TRAV_PARAMS="$2"; shift 2 ;;
    --tag)         TAG="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 2 ;;
  esac
done

OUT_BASE="$WS_ROOT/tests/results/$TAG"
mkdir -p "$OUT_BASE"
EXTRA_LAUNCH=()
[ -n "$NAV2_PARAMS" ] && EXTRA_LAUNCH+=("nav2_params_file:=$NAV2_PARAMS")
[ -n "$BT_XML" ] && EXTRA_LAUNCH+=("bt_xml:=$BT_XML")
[ -n "$TRAV_PARAMS" ] && EXTRA_LAUNCH+=("traversability_params:=$TRAV_PARAMS")

run_trial() {
  local id="$1" profile="$2" seed="$3" maxsec="$4" rand="$5"
  shift 5
  echo "============================================================"
  echo "[BENCHMARK] $id (profile=$profile loc=$LOC seed=$seed rand=$rand max=${maxsec}s)"
  echo "============================================================"
  "$WS_ROOT/docker/run_coqui_survey_trial.sh" "${TAG}_${id}" "$seed" "$maxsec" "$LOC" "$profile" "$rand" \
    "${EXTRA_LAUNCH[@]}" "$@" | tee "$OUT_BASE/${id}.log" || true
}

IFS=',' read -ra PARTS <<< "$SUITE"
for part in "${PARTS[@]}"; do
  case "$part" in
    smoke)
      run_trial smoke1_forest_short forest 0 300 false
      run_trial smoke2_forest_rand1 forest 1 480 true
      ;;
    slope)
      run_trial slope_test_1 hill 0 240 false x:=-11.219 y:=3.479 z:=12.597 yaw:=0.257
      run_trial slope_test_2 hill 0 240 false x:=-15.633 y:=6.319 z:=12.232 yaw:=2.622
      run_trial slope_test_3_fall_guard hill 0 240 false x:=-11.201 y:=10.497 z:=12.366 yaw:=-1.561
      run_trial bush_slope_test hill 0 300 false x:=-3.125 y:=-9.014 z:=3.390 yaw:=2.985
      ;;
    hill)
      run_trial hill_v1_switchback hill 0 1200 false
      ;;
    forest)
      run_trial full6_forest forest 0 2400 false
      ;;
    drive_trials)
      "$WS_ROOT/tests/sim/run_drive_trials.sh" "$OUT_BASE/drive_trials"
      ;;
    slope_physics)
      "$WS_ROOT/tests/sim/run_slope_physics_test.sh" "$OUT_BASE/slope_physics"
      ;;
    hill_terrain)
      "$WS_ROOT/tests/sim/run_hill_terrain_test.sh" "$WS_ROOT/src/ugv_main/ugv_gazebo/worlds/hill_coqui_survey.world" "$OUT_BASE/hill_terrain"
      ;;
    all)
      "$0" --suite smoke,slope,hill,drive_trials,slope_physics,hill_terrain \
        --localization "$LOC" --tag "$TAG" \
        ${NAV2_PARAMS:+--nav2-params "$NAV2_PARAMS"} \
        ${BT_XML:+--bt-xml "$BT_XML"} \
        ${TRAV_PARAMS:+--traversability-params "$TRAV_PARAMS"}
      exit 0
      ;;
    *) echo "Unknown suite part: $part (use smoke,slope,hill,forest,drive_trials,slope_physics,hill_terrain,all)"; exit 2 ;;
  esac
done

echo "Benchmark finished. Logs in $OUT_BASE"
