# UGV Navigation & Coqui Survey Regression Test Suite (`/tests`)

Pooled regression suite combining every historical test case (smoke runs, steep slope tests,
hill switchback climbs, 24 forest drive trials, traversability realism checks, and all recorded
navigation/mission failure cases) so **every new navigation style** (controller, planner, critic,
behavior tree, or traversability tuning) can be validated consistently before deploying to sim or hardware.

## Quick Start

```bash
# 1. Fast offline regression suite (no ROS/Gazebo needed, ~5-10 s)
./tests/run_all.sh --offline

# 2. Test a NEW navigation style (custom Nav2 YAML / BT XML / traversability YAML)
./tests/run_all.sh --offline \
  --nav2-params path/to/new_nav2.yaml \
  --bt-xml path/to/new_bt.xml \
  --traversability-params path/to/new_trav.yaml

# 3. In-container ROS 2 live Nav2 costmap & launch verification (no Gazebo GUI, ~15 s)
./tests/run_all.sh --ros --nav2-params src/ugv_main/ugv_coqui_behavior/config/nav2_survey_base.yaml

# 4. Headless Gazebo benchmark (smoke, slope, hill, forest, drive_trials, slope_physics, hill_terrain)
# Run inside the Docker container (./docker/shell.sh):
./tests/sim/run_nav_benchmark.sh --suite smoke,slope --localization ekf
./tests/sim/run_nav_benchmark.sh --suite all --nav2-params path/to/new_nav2.yaml --bt-xml path/to/new_bt.xml
```

## Directory Layout

| Path | Purpose |
|---|---|
| `nav_styles.yaml` | Named navigation configurations (`survey_dwb_default`, `slam_nav_traversability`, add new styles here) |
| `cases/scenarios.yaml` | Pooled catalog of every smoke, slope, hill, forest, and failure scenario with coordinates & acceptance criteria |
| `cases/forest_drive_trials.json` | 24 closed-loop forest drive trials (`edge_off`, `edge_on`, `open`, `vegetation`) |
| `offline/` | Fast pure-Python pytest suites covering Nav2 config, BT rules, safety watchdog, traversability attitude noise, mission failures, audio realism, and launch/script hygiene |
| `ros/` | Live ROS 2 Humble checks (real `nav2_costmap_2d` layer combination + live launch argument resolution) |
| `sim/` | Headless Gazebo runners (`run_nav_benchmark.sh`, `run_slope_physics_test.sh`, `run_hill_terrain_test.sh`, `run_drive_trials.sh`, `nav_goal_runner.py`) |
| `results/` | Timestamped benchmark outputs (gitignored) |
