# Waveshare UGV — Gazebo Classic in Docker (Ubuntu 24 host)

Host has ROS 2 Jazzy + modern Gazebo (Ignition). This stack needs **ROS 2 Humble + Gazebo Classic 11**, so we run it in Docker with X11 GUI.

## 1) Install Docker (host — approve these)

```bash
sudo apt-get update
sudo apt-get install -y docker.io
sudo usermod -aG docker "$USER"
newgrp docker
docker run --rm hello-world
```

If `newgrp` is awkward, log out/in once so group `docker` applies.

## 2) Build the image (host)

```bash
cd /home/dark-tiger-ubun/projects/ugv_ws
chmod +x docker/*.sh
./docker/build_image.sh
```

## 3) Enter the container (host)

```bash
cd /home/dark-tiger-ubun/projects/ugv_ws
export UGV_MODEL=ugv_rover   # or ugv_beast / rasp_rover
./docker/shell.sh
```

## 4) Inside the container — setup once, then launch

```bash
# first time only
chmod +x docker/run_grass_sim.sh
./docker/run_grass_sim.sh
```

Or do the steps manually:

```bash
cd /root/ugv_ws
source /opt/ros/humble/setup.bash
export UGV_MODEL=ugv_rover
export GAZEBO_MODEL_DATABASE_URI=""
export GAZEBO_MODEL_PATH=/root/.gazebo/models:/root/ugv_ws/src/ugv_main/ugv_gazebo/models:$GAZEBO_MODEL_PATH

mkdir -p /root/.gazebo/models
unzip -qo ugv_description.zip -d /root/.gazebo/models

colcon build --packages-select ugv_gazebo --symlink-install
source install/setup.bash

ros2 launch ugv_gazebo bringup_grass.launch.py
```

Gazebo GUI should open on your Ubuntu desktop (X11). You should see a **grass plane**, a wood box, a cylinder, and the **ugv_rover**.

## 5) Drive it (second terminal)

```bash
cd /home/dark-tiger-ubun/projects/ugv_ws
./docker/shell.sh
# inside:
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

Or Waveshare keyboard node (if you build `ugv_tools`):

```bash
ros2 run ugv_tools keyboard_ctrl
```

## Gazebo CLI alternatives (inside container)

With gzserver already running from the launch file, you usually do **not** need these. Useful for debugging:

```bash
# List models
gz model -m

# Spawn robot manually (if launch spawn failed)
gz model --spawn-file=/root/ugv_ws/src/ugv_main/ugv_gazebo/models/ugv_rover/model.sdf \
  --model-name=ugv_rover -x 0 -y 0 -z 0.15

# Delete / respawn
gz model -m ugv_rover -d

# World info
gz stats
gz world -w
```

Pure Gazebo Classic (no ROS) smoke test:

```bash
gazebo --verbose /root/ugv_ws/src/ugv_main/ugv_gazebo/worlds/grass.world
```

Then insert the model from the GUI: **Insert** → navigate to `/root/ugv_ws/src/ugv_main/ugv_gazebo/models/ugv_rover`.

## Model options

| Env | Robot |
|-----|--------|
| `UGV_MODEL=ugv_rover` | Waveshare UGV Rover (default) |
| `UGV_MODEL=ugv_beast` | UGV Beast |
| `UGV_MODEL=rasp_rover` | Rasp Rover |

## Troubleshooting

- **Black / blank Gazebo window**: container already sets `LIBGL_ALWAYS_SOFTWARE=1`. Try `glxinfo -B` inside container.
- **`UGV_MODEL` KeyError**: export it before launch.
- **Missing meshes**: re-run the `unzip ugv_description.zip` step.
- **Rebuild image**: `docker rm -f ugv-gazebo && ./docker/build_image.sh && ./docker/shell.sh`
