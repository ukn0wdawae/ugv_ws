# Autonomous Off-Road UGV Navigation & 3D Acoustic Coquí Survey (`ugv_ws`)

An end-to-end **ROS 2 Humble + Nav2 + Gazebo Classic 11** robotics stack for **GPS-denied autonomous exploration, 3D terrain traversability analysis, off-road navigation, and multi-microphone 3D acoustic wildlife localization** in unstructured outdoor environments — built using pure geometry, signal processing, filtering, and graph search (no machine learning or prior maps).

> **Upstream Base Repository**: This workspace was originally cloned from [**waveshareteam/ugv_ws**](https://github.com/waveshareteam/ugv_ws). See the upstream repository for stock Waveshare **UGV Rover / UGV Beast / RaspRover** hardware bringup, ESP32 sub-controller serial drivers (`ugv_base_node`, `ugv_bringup`), AprilTag demos (`ugv_vision`), and web/chat utilities (`ugv_web_app`, `ugv_chat_ai`). **This README documents the custom robot platform (`ugv_pro_audio`), outdoor simulation worlds, 3D acoustic pipeline, GPS-denied LiDAR localization, 3D terrain traversability, slope-safe navigation, autonomous Coquí survey behavior, Docker isolation, and regression test framework built on top of that base.**

---

## Table of Contents

1. [Project Highlights](#1-project-highlights)
2. [Quick Start (Docker & Survey Scripts)](#2-quick-start-docker--survey-scripts)
3. [New Robot Model: `ugv_pro_audio` & Physics Realism](#3-new-robot-model-ugv_pro_audio--physics-realism)
4. [Outdoor Simulation Worlds & 3D Habitat Frog Randomization](#4-outdoor-simulation-worlds--3d-habitat-frog-randomization)
5. [3D Acoustic Synthesis & SRP-PHAT Localization (`ugv_audio`)](#5-3d-acoustic-synthesis--srp-phat-localization-ugv_audio)
6. [GPS-Denied 3D Localization, Traversability & Off-Road Navigation](#6-gps-denied-3d-localization-traversability--off-road-navigation)
7. [Autonomous Coquí Frog Survey Mission (`ugv_coqui_behavior`)](#7-autonomous-coquí-frog-survey-mission-ugv_coqui_behavior)
8. [Docker Container Architecture & Clean Process Lifecycle](#8-docker-container-architecture--clean-process-lifecycle)
9. [Pooled Regression & Gazebo Benchmark Suite (`/tests`)](#9-pooled-regression--gazebo-benchmark-suite-tests)
10. [Workspace Package Map & Upstream Reference](#10-workspace-package-map--upstream-reference)

---

## 1. Project Highlights

- **Custom 6-Wheel Acoustic Survey Rover (`ugv_pro_audio`)**: Extends the 6-wheel skid-steer UGV Rover with an omnidirectional **5-capsule 3D microphone array**, **16-channel 360° 3D LiDAR**, **2-axis pan-tilt RGB camera turret**, **forward RGB-D camera**, **realistic noisy MEMS IMU**, and **anisotropic Coulomb tire friction** with true wheel-encoder odometry.
- **Four Unstructured Outdoor Worlds + 3D Habitat Randomization**: Custom Gazebo worlds (`park`, `lake`, `forest`, `hill`) with repaired terrain step meshes, passable non-solid grass tufts, surface-specific traction, and automatic per-run **3D frog placement** inside dense bushes, on top of pine crowns and fallen logs, perched up standing pine bark, and across tall grass and forest glades.
- **Physics-Based Chorus Audio & 3D SRP-PHAT (`ugv_audio`)**: Real-time `16 kHz` 5-channel wave propagation with FFT sub-sample fractional delays, asynchronous multi-frog chorus scheduling, per-frog/path directional loudness variation, motor self-noise, multipath echoes, harmonic-free guard-band SNR, and coarse-to-fine 3D SRP-PHAT with single-source wave coherence gating.
- **GPS-Denied 3D LiDAR Structure Localization**: Combines a 3D wheel+IMU EKF (`wheel_odom_relay` + Madgwick AHRS + `robot_localization`) with `lidar_localizer`, which matches 3D tree trunks and bush structure across scans without GPS or a prior map — cutting 330 m forest position drift from **4.7 m** (dead reckoning) down to **<= 0.4 m** (with RTAB-Map 3D ICP SLAM also supported).
- **3D Terrain Traversability & Steep-Slope Safety (`ugv_traversability`, `slope_recovery.py`)**: Real-time elevation mapping scoring slope, trimmed step height, and roughness while ignoring overhanging branches above chassis clearance, paired with Nav2 footprint-aware DWB, collision monitoring, IMU steep-slope stall recovery (controlled downhill gravity-line descent around boulders), and a 10 Hz mission safety watchdog.
- **Full Autonomous Survey Mission (`ugv_coqui_behavior`)**: Frontier exploration with acoustic whisper bias, a persistent lead book, chorus-resilient MSAC angular triangulation with loudness consistency, active baseline stepping for parallax, 3D pan-tilt visual servoing & frog photography, strict simulation truth isolation, and staged breadcrumb return-home.
- **230 Automated Offline Tests + Live ROS & Gazebo Benchmarks**: `147` package unit tests, `83` pooled offline regression tests, live ROS 2 launch/costmap verification, and headless Gazebo benchmark suites (`12/12` frogs located across forest missions with `0` false positives and `0.17 m` mean horizontal error; `24/24` forest drive trials passed).

---

## 2. Quick Start (Docker & Survey Scripts)

The simulation stack targets **ROS 2 Humble + Gazebo Classic 11** and runs cleanly on Ubuntu 22.04 / 24.04 hosts inside Docker with X11 GUI and GPU acceleration.

### Build & Enter the Container (Host)

```bash
./docker/build_image.sh
./docker/shell.sh
```

### Run the Interactive Coquí Survey (2 Terminals Inside `./docker/shell.sh`)

**Terminal 1 — Simulated World, Robot & Calling Frogs:**

```bash
# Forest world (default): opens Gazebo following the robot and spawns 6 frogs at fresh
# randomized 3D bush, tree-top, bark-perch, tall-grass, and glade positions every run
./docker/latest-scripts/run_coqui_survey_sim.sh

# Useful options:
./docker/latest-scripts/run_coqui_survey_sim.sh profile:=hill        # steep mountain switchback world (+11 m climb)
./docker/latest-scripts/run_coqui_survey_sim.sh seed:=42             # replay a specific randomized frog layout
./docker/latest-scripts/run_coqui_survey_sim.sh randomize:=false     # fixed canonical frog layout (seed=0)
./docker/latest-scripts/run_coqui_survey_sim.sh gui:=false           # headless gzserver only
./docker/latest-scripts/run_coqui_survey_sim.sh follow:=false        # free Gazebo camera
./docker/latest-scripts/run_coqui_survey_sim.sh show_truth:=false    # hide /sim/coqui_truth_markers in RViz
```

**Terminal 2 — Localization, Traversability, Nav2, Acoustic Localizer, Mission & RViz:**

```bash
# Default: GPS-denied 3D EKF + 3D LiDAR structure localizer + full autonomous survey + RViz
./docker/latest-scripts/run_coqui_survey.sh

# Useful options:
./docker/latest-scripts/run_coqui_survey.sh localization:=slam       # 3D EKF + RTAB-Map 3D LiDAR SLAM
./docker/latest-scripts/run_coqui_survey.sh localization:=ekf        # 3D wheel + IMU dead reckoning only
./docker/latest-scripts/run_coqui_survey.sh localization:=perfect    # ground-truth pose ablation mode
./docker/latest-scripts/run_coqui_survey.sh mission:=false           # full nav stack without autonomous mission (teleop/RViz goals)
./docker/latest-scripts/run_coqui_survey.sh run_id:=forest_day1      # custom output folder under coqui_surveys/
```

### Grade a Completed Survey or Run a Headless Trial

Survey outputs (`survey.yaml`, `frogs.geojson`, `events.jsonl`, and annotated `photos/*.jpg`) are written to `coqui_surveys/<run_id>/`:

```bash
# Grade an interactive run (automatically resolves the active session's randomized truth YAML)
ros2 run ugv_coqui_behavior eval_survey /root/ugv_ws/coqui_surveys/<run_id>/survey.yaml --profile forest

# Single-command headless trial (isolated ROS_DOMAIN_ID & Gazebo port + automatic grading)
./docker/run_coqui_survey_trial.sh smoke1 0 600 lidar forest false
./docker/run_coqui_survey_trial.sh rand_s3 3 2400 lidar forest true
```

---

## 3. New Robot Model: `ugv_pro_audio` & Physics Realism

Defined in `src/ugv_main/ugv_gazebo/models/ugv_pro_audio/model.sdf` and `src/ugv_main/ugv_description/urdf/ugv_pro_audio.urdf` (alongside `ugv_beast_pro`), `ugv_pro_audio` upgrades the base Waveshare 6-wheel rover into a physically faithful off-road acoustic/visual survey platform:

| Subsystem | Implementation Details |
|---|---|
| **6-Wheel Skid-Steer Chassis** | 3 driven wheel pairs (`0.175 m` track width, `0.080 m` spherical tire collision, realistic `1.2 N·m` max wheel torque & `3.5 rad/s²` max wheel acceleration to prevent unrealistic wheelies), raised `6 cm` ground-clearance belly collision box, and upper-deck roll-bar collision. |
| **Anisotropic Coulomb Tire Friction** | Wheel-link axle alignment (`<fdir1>0 0 1</fdir1>`, `slip1 = slip2 = 0.0`) decouples **lateral shoulder grip** (`mu = 0.85`, holding steep side-hills without creeping while still scrubbing cleanly under turn torque) from **longitudinal tread grip** (`mu2 = 1.05`, governed by each terrain material's friction via Gazebo's `min(mu_wheel, mu_ground)` rule). |
| **Honest Encoder Odometry** | `libgazebo_ros_diff_drive.so` configured with `<odometry_source>0</odometry_source>` (wheel encoders on `/wheel/odom`, `publish_odom_tf=false`), never leaking Gazebo world pose into robot localization. |
| **5-Capsule 3D Microphone Array** | `audio_array_base_link` (`z = +11.5 cm`) with 4 planar capsules (`±9.5 cm × ±7.5 cm`) plus an elevated top-center capsule (`-6.5 cm, 0, +4.0 cm`) breaking planar elevation ambiguity for full 3D azimuth + elevation DOA. |
| **16-Channel 360° 3D LiDAR** | `lidar_3d_link` publishing `sensor_msgs/PointCloud2` on `/scan_cloud` at `10 Hz` (`360 × 16` samples, `±16°` vertical FOV, `0.22–30.0 m` range, `8 mm` Gaussian noise). |
| **2-Axis Pan-Tilt Camera Turret** | Revolute pan (`±180°`) and tilt (`-37°..+90°`) turret carrying a `640×480` RGB camera (`/pt_camera/image_raw`). Driven by `pt_camera_controller`, which only dispatches `JointTrajectory` updates when pan/tilt changes so `ODEJoint::SetPosition` never perturbs wheel-ground contact while driving. |
| **Forward RGB-D Camera** | `320×240` depth camera (`3d_camera_link`) with `<far>30.0</far>` matched to the 3D LiDAR range — fixing a Gazebo Classic OGRE render-target bug where a `5 m` depth clip plane leaked fake `4.75 m` walls into `gpu_ray` LiDAR scans. |
| **Noisy 6-Axis MEMS IMU** | `50 Hz` gyroscope and accelerometer on `/imu/raw` with Gaussian white noise and `300 s` bias random walk, fused by `imu_filter_madgwick` into `/imu`. |

---

## 4. Outdoor Simulation Worlds & 3D Habitat Frog Randomization

### Custom Worlds (`src/ugv_main/ugv_gazebo/worlds/`)

- **`forest.world` / `forest_coqui_survey.world`**: Dense forest (`100 m × 50 m`, El Yunque GPS reference, `32 m × 30 m` primary survey box) with rolling dirt terrain, winding trails, 100+ pine trees (`pine_1..5`), dense pine bushes (`bush_pine*`), fallen logs, and 260+ tall-grass tufts (`altaniv*`).
  - **Simulator Physics Repairs**: Original forest trail meshes had `4–9 cm` sharp step lips (lethal cliffs for `8 cm` wheels) and rock-solid collision meshes on grass tufts that capsized the rover. We built `forest_terrain_traversable`, stripped non-physical grass collisions across worlds, removed invisible collision pillars on `coqui_frog/model.sdf`, and tuned physics to `500 Hz` (`max_step_size = 0.002 s`) — taking closed-loop forest drive trials from **8/24 → 24/24**.
- **`hill.world` / `hill_coqui_survey.world`**: Mountain switchback road climbing **+11 m** vertically (`z ~ 2.4 m` to `13.0 m`, `10°–22°` grades) lined with boulders (`roca3`), dry trees (`tree_8`, `arbol3_dry`, `arbol5`), dry bushes (`dry_bush`, `arbusto3_dry`), and ridge plateaus.
- **`park.world` / `park_coqui.world` & `lake.world` / `lake_coqui.world`**: Open park obstacle course and sloped/slippery lakeshore worlds, plus compact `*_tagged.world` variants.

### 3D Terrain- & Habitat-Aware Frog Randomization

In `forest_coqui_survey.world`, the pine and bush `.obj` assets (`pine_1..6`) carry `10–37 m` scene-space vertex offsets and `+90°` link rotations relative to `<model><pose>`. Naive 2D XY jitter left frogs floating in empty air or buried under rolling terrain.

- **Exact 3D Scene Extraction (`extract_world_scenes.py` → `world_scenes.npz`)**: Transforms every COLLADA, OBJ, and STL terrain/vegetation mesh through Gazebo's `<state>`, `<link>`, `<visual>`, and `<scale>` matrices to precompute millimeter-accurate elevation grids (`ground_elevation_z`) and verified 3D habitat catalogs for both `forest` and `hill`.
- **Realistic Per-Run Diversity (`randomize_truth_scenario` in `audio_core.py`)**: Every run of `run_coqui_survey_sim.sh` generates a new seed and places the 6 Coquí frogs across distinct 3D micro-habitats (with shuffled IDs, `>= 3.2 m` pairwise separation, and `>= 4.0 m` spawn clearance):
  - **2 frogs inside dense bushes** (`bush_pine*` interior with overhead canopy, `attenuation: 0.75–0.83`),
  - **2 elevated frogs on trees** — at least **1 literally on top of a young pine crown or fallen pine trunk** (`+1.1..2.5 m`) and **1 perched up a standing pine trunk** (`+1.45..2.20 m`, oriented outward toward clear camera vantages),
  - **1 frog inside a tall-grass patch** (`altaniv*`) and **1 frog in an open forest glade or tucked behind a pine trunk**.
- **Strict Answer-Key Isolation**: Ground-truth frog coordinates live exclusively on the simulation/grading side (`coqui_sound_emitter`, `_rewrite_world_frogs`, `eval_survey`). `test_answer_key_separation.py` statically scans every robot-side package and config to guarantee no robot node ever reads truth files, Gazebo model states, or `/sim/*` topics.

---

## 5. 3D Acoustic Synthesis & SRP-PHAT Localization (`ugv_audio`)

### Multi-Source Acoustic Simulator (`coqui_sound_emitter`, `audio_core.py`)

- **Biologically Grounded Call Synthesis**: Generates `16 kHz` (`2048`-sample, `128 ms`) frames of the two-note male *Eleutherodactylus coqui* call — a constant `~1160 Hz` *"Co"* territorial note followed by an upward `1820 → 2180 Hz` *"Quí"* sweep — with per-frog pitch variation.
- **Asynchronous Chorus & Call Variation**:
  - `ChorusSchedule`: Every frog calls on its own natural rhythm (`2.0–3.6 s` intervals in `45–120 s` bouts separated by `8–35 s` pauses), holding off briefly when a neighbor is mid-call except for configurable simultaneous call collisions (`overlap_prob = 0.25`).
  - `CallVariation` & `make_mic_error`: Models per-frog source level scatter, call-to-call amplitude variance, per-meter foliage loss, directional cover shadowing (muffling behind tree trunks/bushes), and fixed per-capsule microphone position (`1.5 mm`) and gain (`0.7 dB`) tolerances.
- **Wave Propagation & Clutter**: Applies exact Fourier linear-phase sub-sample time delays across all 5 capsule world coordinates, multipath foliage echoes, `3.3–5.2 kHz` forest insect trills, and speed/yaw-rate-coupled DC gearmotor + tire rumble.

### Harmonic-Free Detection & Coarse-to-Fine 3D SRP-PHAT (`mic_array_localizer`)

- **Harmonic-Free SNR & Calibrated Band Level**: Compares Co + Quí narrowband energy against median noise power in harmonic-free guard bands (`550–920 Hz`, `1380–1620 Hz`, `4800–6500 Hz`), preserving `>25 dB` dynamic range (`1 m` to `22 m`) without harmonic self-masking or motor-whine false triggers, while `call_band_level_db` estimates noise-subtracted received loudness (`0 dB` at `1 m` nominal) for range estimation.
- **Coarse-to-Fine 3D SRP-PHAT (`estimate_doa`)**: Evaluates magnitude-weighted phase transform (`|X|^0.3` weighting) on a coarse `72 × 17` azimuth-elevation grid followed by a `25 × 25` local `±6°` refinement (`~0.5°` resolution), publishing 3D DOA bearings plus **single-plane-wave coherence** (`>= 0.97` for one frog vs `0.60–0.95` when two calls collide) so blended bearings are rejected before corrupting triangulation.

---

## 6. GPS-Denied 3D Localization, Traversability & Off-Road Navigation

### State Estimation Modes (`coqui_survey.launch.py`)

1. **`localization:=lidar` *(Default — GPS-Denied, No Prior Map)***:
   - `wheel_odom_relay`: Converts `/wheel/odom` into `/wheel/odom_3d`, enforcing soft lateral (`vy = 0`) and tight vertical (`vz = 0`) non-holonomic constraints, scaling `vx` variance with speed and turn rate to reflect 6-wheel skid-steer scrub, and dropping wheel yaw rate in favor of the IMU gyroscope.
   - `robot_localization` 3D EKF: Fuses `/wheel/odom_3d` + Madgwick `/imu` into smooth 3D `odom -> base_footprint`.
   - `lidar_localizer` (`lidar_localizer_core.py`): Extracts non-ground tree trunks, bushes, and vertical structure from `/scan_cloud` matched at exact scan timestamps (`tf2` lookup) against accumulated local structure, publishing rate-limited (`CorrectionSmoother`) `map -> odom` transforms (~`0.1–0.4 m` accuracy across full `330 m` forest missions).
2. **`localization:=slam`**: 3D EKF + **RTAB-Map 3D LiDAR ICP SLAM** (`rtabmap_3d_lidar.launch.py`) with tuned point-to-plane ICP and ground/obstacle segmentation (`rtabmap_slam.yaml`).
3. **`localization:=ekf`**: Pure 3D wheel+IMU dead reckoning with static `map -> odom` at the profile trailhead.
4. **`localization:=perfect`**: Ground-truth `/ugv_gt_odom` bridge for controlled perception/planning ablations.

### 3D Terrain Traversability (`ugv_traversability`)

- **Drift-Anchored Elevation Mapping (`elevation_map.py`, `traversability_node.py`)**: Integrates 3D LiDAR scans into a rolling `5 cm`-class elevation grid, anchoring vertical odometry drift to measured ground under the footprint (`set_footprint` never artificially raises ground into a fake step behind the wheels) and shifting published clouds (`height_shift`) to stay locked to the robot's TF frame.
- **Geometric Terrain Scoring (`features.py`)**: Scores local **slope**, **trimmed step height** (catching `>= 5 cm` curbs/trunks while ignoring sparse outlier points), and **roughness**, while filtering overhead branches/canopy above chassis height (`max_obstacle_height`) so the rover can drive beneath tree crowns.
- **Dual Nav2 Feeds**: Publishes `/terrain_traversability/grid` (global non-lethal slope costmap preserving `-1` unknown cells so frontier exploration sees unexplored boundaries) and `/terrain_traversability/obstacle_grid` + robot-frame ground/obstacle `PointCloud2` streams for Nav2 raytracing.

### Slope-Safe Off-Road Nav2 & Recovery (`ugv_coqui_behavior`)

- **Footprint-Aware Planning & Control (`nav2_survey_base.yaml`, `bt_commit_to_path.xml`)**: Uses `GridBased` global planning with slope-weighted path selection (`path_terrain_cost`), footprint-aware DWB local trajectory critics (replacing center-point-only scoring), `nav2_collision_monitor` (`PolygonStop` + `Approach` slowdown on `/cmd_vel`), and a custom commit-to-path Behavior Tree that recovers locally without ever wiping the global costmap.
- **IMU Steep-Slope Stall Recovery (`slope_recovery.py`)**: Monitors IMU pitch/roll (`>18°–28°`) and actual chassis progress; if skid-steer turning stalls or side-hill roll approaches rollover (`>= 22°`), cancels Nav2 and drives smoothly down the IMU gravity fall-line (forward or reverse, never spinning in place on a steep bank) while steering around `/terrain_traversability/obstacle_grid` boulders until flat ground (`<= 14°`) is confirmed.
- **10 Hz Safety Monitor**: Continuously checks for vertical drops (`>= 1.5 m`), excessive downward velocity, capsizing (`>= 65°`), runaway speed, tumble rate, and stale TF/IMU/LiDAR/costmap streams in every mission state.

---

## 7. Autonomous Coquí Frog Survey Mission (`ugv_coqui_behavior`)

`coqui_survey_node.py`, `survey_planning.py`, and `acoustic_hunt.py` implement a complete autonomous wildlife survey state machine (`EXPLORE` → `HUNT` → `VANTAGE` → `PHOTO` → `RECORD` → `HOME`):

1. **Frontier Exploration + Lead Book**: Explores unknown frontiers inside `survey_area` with preemptive goal handoff (`frontier_handoff_m = 2.0 m`) so the rover rolls smoothly without stop-and-go pauses, biased toward faint distant calls (`whisper_bias_m`), falling back to a `6 m` listening-station grid to guarantee 100% area coverage. Unresolved acoustic detections are banked in a persistent **Lead Book** for later retry from better geometry.
2. **Chorus-Resilient Single-Frog Hunting (`AcousticHunt`, `robust_triangulate`)**:
   - Anchors on a stationary call and filters incoming bearings by **coherence**, **anchor alignment**, **calling-bout grouping**, and **calibrated source loudness consistency** (distinguishing two collinear frogs at different distances and preventing two side-by-side frogs from creating a phantom intersection).
   - Refines 3D positions via MSAC + angular Gauss-Newton least squares + median ray elevation, planning Nav2-verified baseline steps (`ComputePathToPose`) outside frog keep-outs to maximize angular parallax (`>= 15°`, `>= 3` listening places, `<= 0.6 m` uncertainty).
3. **3D Viewpoint Selection & Pan-Tilt Visual Servoing (`survey_vision.py`)**:
   - Classifies targets as `ground` or `elevated` (`>= 0.8 m` above local ground for bark-perched / tree-top frogs) and navigates to an obstacle-free viewpoint (`1.4 m` ground / `2.2 m` elevated standoff) with costmap line-of-sight.
   - Aims `/pt_camera`, scans local angular offsets, detects the lime-green body + yellow vocal sac in HSV, centers the camera via pinhole intrinsics (`CameraInfo`), verifies acoustic presence at the viewpoint, and rejects empty clear-LOS open-ground false fixes while accepting confirmed foliage-occluded bush/grass fixes.
4. **Survey Persistence, Live RViz HUD & Staged Return-Home**:
   - Writes `survey.yaml`, `frogs.geojson` (local ENU -> WGS84), `events.jsonl`, and annotated bounding-box photos in `coqui_surveys/<run_id>/`, publishes live RViz markers (`/coqui_survey/markers`, `/coqui_survey/hearing`, `/coqui_survey/photo`, `survey_viz_aids`), and drives back to the trailhead in rolling-window breadcrumb stages (`next_home_waypoints`).

---

## 8. Docker Container Architecture & Clean Process Lifecycle

Located in `docker/` (`Dockerfile`, `shell.sh`, `entrypoint.sh`, `latest-scripts/`):

- **Ubuntu 22.04 + ROS 2 Humble + Gazebo Classic 11 Image**: Pre-installs Nav2, `robot_localization`, Madgwick IMU filter, RTAB-Map, OpenCV, and NVIDIA/Mesa GLX X11 support (`launch_docker_sim.sh` and `docker/shell.sh`).
- **Shared-Memory & Zombie Isolation**: Containers run with `--init` (`docker-init` PID 1 zombie reaping), private `--shm-size=2g` (preventing FastDDS `fastrtps_*` shared-memory segment leaks onto the host), and `ALSOFT_DRIVERS=null`.
- **Zero-Orphan Teardown (`docker/latest-scripts/_cleanup_lib.sh`)**: Stamps every launch process tree with a unique `UGV_LAUNCH_TAG` in `/proc/<pid>/environ` and pairs it with a detached session watchdog (`SESSION_FILE` peer-monitoring). Pressing `Ctrl+C`, double-`Ctrl+C`, or closing a terminal tab cleanly terminates `gzserver`, `gzclient`, `rviz2` (`1 s` `sigterm_timeout`), `rtabmap`, Nav2 lifecycle servers, and idle `ros2-daemon` processes within ~1–2 seconds with zero leftover CPU/VRAM load.

---

## 9. Pooled Regression & Gazebo Benchmark Suite (`/tests`)

All historical bug fixes, physics edge cases, chorus phantom scenarios, and navigation configurations are pooled under [`tests/`](tests/README.md):

```bash
# 1. Fast ROS-free offline suite: 147 package unit tests + 83 pooled regressions (~80 s)
./tests/run_all.sh --offline

# 2. Validate a custom Nav2 / Behavior Tree / Traversability configuration offline
./tests/run_all.sh --offline \
  --nav2-params path/to/custom_nav2.yaml \
  --bt-xml path/to/custom_bt.xml \
  --traversability-params path/to/custom_trav.yaml

# 3. Live ROS 2 Humble costmap layer & launch-graph resolution tests (inside container)
./tests/run_all.sh --ros

# 4. Headless Gazebo closed-loop benchmarks (smoke, slope, hill, forest, drive_trials)
./tests/sim/run_nav_benchmark.sh --suite smoke,slope --localization lidar
```

---

## 10. Workspace Package Map & Upstream Reference

### Custom & Extended Packages in This Repository

| Path | Role |
|---|---|
| `src/ugv_main/ugv_audio` | **New**: 5-mic 3D acoustic wave synthesis, chorus & call variation, harmonic-free SNR, 3D SRP-PHAT localizer, and 3D terrain/habitat frog randomizer (`world_scenes.npz`). |
| `src/ugv_main/ugv_traversability` | **New**: 3D LiDAR elevation mapping, height-drift anchoring, slope/step/roughness scoring, canopy filtering, and Nav2 traversability grids/clouds. |
| `src/ugv_main/ugv_coqui_behavior` | **New**: Autonomous Coquí survey node, chorus-resilient acoustic hunt, GPS-denied 3D `lidar_localizer`, `wheel_odom_relay`, `pt_camera_controller`, IMU `slope_recovery` & safety monitor, RViz HUD, and `eval_survey` grader. |
| `src/ugv_main/ugv_gazebo` | **Extended**: `ugv_pro_audio`, `ugv_beast_pro`, `coqui_frog`, `forest_terrain_traversable`, outdoor survey worlds (`forest`, `hill`, `park`, `lake`), 3D scene/world generators (`scripts/coqui/`), clean spawner, and 3D LiDAR RTAB-Map launch. |
| `src/ugv_main/ugv_description` | **Extended**: `ugv_pro_audio.urdf` and `ugv_beast_pro.urdf` robot descriptions. |
| `docker/` & `docker/latest-scripts/` | **New**: ROS 2 Humble + Gazebo Classic 11 container build, isolated shell launcher, process-reaping `_cleanup_lib.sh`, 2-terminal survey scripts, and headless trial runner. |
| `tests/` | **New**: Offline, ROS 2, and Gazebo regression & navigation style benchmark framework (`230` automated unit/offline tests + recorded field/sim scenarios). |
| `poster/` | **New**: Research poster (`main.tex`), mission trajectory & localization drift datasets (`poster/data/`), and figure generators (`poster/scripts/`). |

### Stock Upstream Waveshare Packages

Inherited from [**waveshareteam/ugv_ws**](https://github.com/waveshareteam/ugv_ws) for physical Waveshare UGV Rover / UGV Beast / RaspRover hardware bringup and 2D indoor demos:
- **Hardware & Teleop**: `ugv_base_node`, `ugv_bringup`, `ugv_interface`, `ugv_tools`
- **Stock 2D SLAM / Nav / Vision / Web**: `ugv_nav`, `ugv_slam`, `ugv_vision`, `ugv_web_app`, `ugv_chat_ai`, and third-party dependencies under `src/ugv_else/` (`apriltag_ros`, `cartographer`, `costmap_converter`, `emcl2_ros2`, `explore_lite`, `gmapping`, `ldlidar`, `rf2o_laser_odometry`, `robot_pose_publisher`, `teb_local_planner`, `vizanti`).

For physical car wiring, ESP32 UART JSON protocol, and stock 2D indoor Gmapping/Cartographer/AprilTag tutorials, refer directly to [**https://github.com/waveshareteam/ugv_ws**](https://github.com/waveshareteam/ugv_ws).
