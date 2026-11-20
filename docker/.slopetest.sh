#!/usr/bin/env bash
# Narrow down the rover sliding on a 6 deg cross-slope (isolated port/domain).
export GAZEBO_MASTER_URI=http://localhost:11499
export ROS_DOMAIN_ID=77
cd /root/ugv_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export GAZEBO_MODEL_DATABASE_URI=""
export GAZEBO_MODEL_PATH="/root/ugv_ws/gazebo_models:/root/.gazebo/models:/root/ugv_ws/src/ugv_main/ugv_gazebo/models:/usr/share/gazebo-11/models"

if pgrep -f "gzserver /tmp/slope.world" >/dev/null; then echo "stale test gzserver running, abort"; exit 1; fi

cat > /tmp/slope.world <<'EOF'
<sdf version='1.6'><world name='default'>
  <physics type='ode'><max_step_size>0.001</max_step_size><real_time_factor>1</real_time_factor><real_time_update_rate>1000</real_time_update_rate></physics>
  <model name='slope'><static>1</static><pose>0 0 0 0.1047 0 0</pose>
    <link name='l'><collision name='c'><geometry><box><size>40 20 0.1</size></box></geometry>
      <surface><friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction>
      <contact><ode><kp>100000.0</kp><kd>100.0</kd><max_vel>0.01</max_vel><min_depth>0.002</min_depth></ode></contact></surface>
    </collision></link></model>
  <model name='box'><pose>-4 0 0.3 0.1047 0 0</pose>
    <link name='l'><inertial><mass>3</mass><inertia><ixx>0.02</ixx><iyy>0.02</iyy><izz>0.02</izz></inertia></inertial>
      <collision name='c'><geometry><box><size>0.2 0.2 0.1</size></box></geometry>
      <surface><friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction></surface></collision></link></model>
</world></sdf>
EOF

gzserver /tmp/slope.world -slibgazebo_ros_init.so -slibgazebo_ros_factory.so > /tmp/slope_gz.log 2>&1 &
GZ=$!
sleep 5
X=0
for v in head index now nowbar; do
  timeout 60 ros2 run gazebo_ros spawn_entity.py -entity v_$v -file /root/ugv_ws/docker/.v_$v.sdf \
    -x $X -y 0 -z 0.06 -Y 0 >/dev/null 2>&1
  X=$((X + 3))
done
echo "downhill = -y.  per model: y z roll   (head=git HEAD model, index=staged, now=current w/o roll bar, nowbar=current)"
for t in 0 1 2 3 4; do
  line="t=$t"
  for m in box v_head v_index v_now v_nowbar; do
    p=$(gz model -m $m -p | awk '{printf "%7.3f %6.3f %6.3f", $2, $3, $4}')
    line="$line | $m $p"
  done
  echo "$line"; sleep 1
done
kill -9 $GZ; sleep 1; pgrep -f "gzserver /tmp/slope.world" && echo "WARN gzserver still alive"; echo done
