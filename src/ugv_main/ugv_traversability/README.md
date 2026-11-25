# ugv_traversability

Terrain traversability estimation for the Waveshare UGV (`ugv_pro_audio`, `ugv_rover`).

The node fuses 3D lidar (and optionally depth-camera) point clouds into a robot-centric 2.5D
**elevation map** and scores every 5 cm cell against what this robot can physically drive over.

| Feature | Window | Safe → cost 0 | Lethal (100) | Why |
|---|---|---|---|---|
| **Step height** – spread of slope-corrected heights | ±0.10 m | < 2 cm | ≥ 4.5 cm | wheel radius is 4 cm |
| **Slope** – least-squares plane | ±0.15 m | < 12° | ≥ 25° | traction μ≈0.75, small top-heavy robot |
| **Roughness** – plane residual std | ±0.15 m | < 1 cm | ≥ 3 cm | 6 cm chassis clearance |

Between safe and lethal the cost rises linearly (0–99, max over the three features). The step uses a
*median* gradient as its slope reference, so a 5 cm curb is still detected at full height on a 10° hill,
and a smooth 20° slope does not count as a step.

## Run

```bash
ros2 launch ugv_traversability traversability.launch.py            # sim (use_sim_time:=true default)
ros2 launch ugv_traversability traversability.launch.py rviz:=true # + RViz view
```

### Topics
| Topic | Type | Content |
|---|---|---|
| `/terrain_traversability/grid` | `nav_msgs/OccupancyGrid` (odom) | cost 0..100, −1 unknown (latched) |
| `/terrain_traversability/elevation_cloud` | `PointCloud2` (odom) | x, y, elevation, intensity = cost (RViz) |
| `/terrain_traversability/obstacles` | `PointCloud2` (base_footprint) | lethal cells → Nav2 obstacle_layer *marking* |
| `/terrain_traversability/ground` | `PointCloud2` (base_footprint) | traversable cells → Nav2 obstacle_layer *clearing* |

Inputs: `/scan_cloud` (16-ch lidar), `/camera/points` (depth camera, stride 4), `/imu`, TF `odom→base_footprint`.

### Per-sensor settings (`config/traversability.yaml`)
| Parameter | Default | Why |
|---|---|---|
| `cloud_topics` | `[/scan_cloud, /camera/points]` | lidar for range, depth camera for the lidar's near-field gap |
| `cloud_strides` | `[1, 4]` | depth cloud subsampled 4×4 → ~19k points |
| `cloud_qos` | `[best_effort, reliable]` | a depth cloud is 9.8 MB; with best-effort one lost UDP fragment drops the whole message |
| `cloud_max_ranges` | `[0.0, 2.0]` (0 = `max_range`) | the low camera sees the ground at a grazing angle: beyond ~2 m its samples are sparse and its height error grows (+1…3 cm measured), which created fake steps where camera and lidar cells meet |

The 10 s status line reports clouds processed **per sensor frame**, so a silent sensor is easy to spot:
`clouds 250 (...), maps 50 (62 ms), known 61 %, ... per frame {'lidar_3d_link': 100, '3d_camera_link': 150}`.

### Nav2
`ugv_gazebo/param/legacy_code/slam_nav_traversability.yaml` is `slam_nav_terrain.yaml` with the obstacle-layer
sources switched to the two clouds above, plus a graded `traversability_layer` (StaticLayer on the grid)
in the local costmap:

```bash
ros2 launch ugv_gazebo slam_nav.launch.py \
  params_file:=$(ros2 pkg prefix ugv_gazebo)/share/ugv_gazebo/param/slam_nav_traversability.yaml
```

**Set `max_obstacle_height` on the obstacle layer itself**, not only per source. Nav2's ObstacleLayer has
its own `max_obstacle_height` (default 2.0 m) that it applies to each point's height *in the costmap frame*.
This forest sits 2.6–3.3 m above the world origin, so with the default every obstacle point was silently
dropped. Only ground clearing (which has no height check) reached the costmaps, and they showed open
ground where there were logs and bushes. Both costmaps in `slam_nav_traversability.yaml` (and the Coqui
survey's Nav2 config) now set `max_obstacle_height: 1000.0` at the layer level. `slam_nav_terrain.yaml`
and the other original configs have the same latent problem whenever the terrain is more than 2 m above
the map origin.

### Pose handling (added for the Coqui survey)
* `exact_pose: true`: each cloud is placed with the robot pose **at the cloud's timestamp**. If that pose
  is not in the TF buffer yet, the cloud is parked and retried every 20 ms, and dropped after
  `pose_wait_s` (0.3 s). It is never placed with the latest pose instead: while Nav2 turns the robot on
  the spot, a cloud placed with a newer heading smears the ground into fake steps. Dropped clouds are
  counted in the status line.
* `footprint_follow_slope: true`: the footprint prior (cells under the robot are ground) lies on the
  robot's tilted base plane rather than a flat disc at the robot's height. On a 10° side slope the flat
  disc is ~2 cm off at its rim.

### 2D odometry
Real wheel odometry is planar (no z / roll / pitch). With `use_imu_attitude: true` (default) the node
rebuilds a 3D pose: roll/pitch from `/imu`, height integrated from wheel travel × sin(pitch).
If the `odom` TF turns out to be 3D already (roll/pitch > 0.5° or z change > 1 cm — e.g. Gazebo's
`diff_drive` with `odometry_source: world`, robot_localization 3D, RTAB-Map) the compensation switches
itself off and logs it once, so height is never counted twice.

## Limits
* Geometry only. **Water, puddles, mud** look flat to a lidar (and often return nothing), so they can't be
  detected from height. Use a camera terrain classifier (the `*_tagged.world` variants render per-class
  colours you can use as labels) or wheel-slip detection (wheel odometry vs IMU / visual odometry) for those.
* **Tall grass looks like a wall to a lidar.** In the fixed forest world the robot can push through the
  `altaniv` grass, but the node still marks it lethal. Telling grass from rock needs semantics.
* The 16-ch lidar only sees ground beyond ~0.85 m. Near-field coverage comes from the depth camera and
  from the footprint prior (cells the robot stands on are ground).
* Gazebo depth cameras need GPU rendering. In a container with software GL (llvmpipe) the forest renders
  at ~0.03 Hz, so the node effectively runs lidar-only there (it keeps working, just with less near-field data).
* Map update cost is ~30 ms (numpy 2.x) / ~60 ms (Humble's numpy 1.21) for the 8×8 m grid; lower
  `publish_rate` or `map_length` on a small onboard computer.

## Measured in simulation (fixed forest world, scripted 30 m drive, 7 snapshots)
Compared cell by cell with world ground truth (tree/bush collision < 30 cm, grass patches, steps ≥ 4.5 cm):

| Sensors | obstacle cells found | lethal cells that are real | free ground marked lethal | map known |
|---|---|---|---|---|
| lidar only | 92.9 % | 89.6 % | 2.9 % | 51 % |
| lidar + camera (to 6 m) | 94.0 % | 79.5 % | 5.0 % | 62 % |
| **lidar + camera (to 2 m, default)** | **92.6 %** | **91.1 %** | **2.1 %** | 53 % |

Height error of the map on bare ground: lidar −1.0 cm median, 0.2–0.4 cm spread; camera (≤ 2 m) ~0 cm, ~0.8 cm spread.

## Tests
```bash
python3 -m pytest -q src/ugv_main/ugv_traversability/test   # no ROS needed (numpy only)
```
