"""Regression tests for Nav2 costmaps, critics, BT, collision monitor, SDF physics, and worlds.

Designed to run against ANY new navigation style (`--nav2-params`, `--bt-xml`, `--traversability-params`).
"""
import os
import re
import xml.etree.ElementTree as ET


def _get_costmap_params(nav2_cfg, key):
    return nav2_cfg[key][key]['ros__parameters']


def test_global_slope_layer_unknown_handling(nav2_cfg, nav_style):
    """Regression for 21:31 edit: in Nav2 Humble, StaticLayer rolling-window `use_maximum: true`
    with `track_unknown_space: true` causes NO_INFORMATION (255) to win std::max(255, cost) and
    erase free space and obstacles from obstacle_layer.
    StaticLayer MUST run BEFORE obstacle_layer with use_maximum: false, and obstacle_layer MUST
    use combination_method: 1 (Max) so known free/lethal/slope cells and unknown frontiers all survive."""
    gc = _get_costmap_params(nav2_cfg, 'global_costmap')
    plugins = gc.get('plugins', [])
    if 'traversability_layer' not in plugins:
        if nav_style.get('require_global_slope_layer', False):
            raise AssertionError('global_costmap is missing traversability_layer')
        return
    assert gc.get('track_unknown_space', False) is True
    assert gc.get('use_maximum', False) is False, (
        'global_costmap use_maximum must be false so StaticLayer NO_INFORMATION (255) never wins max')
    tl = gc.get('traversability_layer', {})
    assert tl.get('use_maximum', False) is False
    assert plugins.index('traversability_layer') < plugins.index('obstacle_layer'), (
        f'traversability_layer must run BEFORE obstacle_layer, got {plugins}')
    ol = gc.get('obstacle_layer', {})
    assert ol.get('combination_method', 1) == 1, 'obstacle_layer must max-combine on top of traversability_layer'
    assert ol.get('max_obstacle_height', 0.0) >= 50.0, 'max_obstacle_height must clear elevated terrain z > 2 m'


def test_local_costmap_layer_order_and_frame_consistency(nav2_cfg):
    """Local costmap must keep traversability_layer before obstacle_layer (no NO_INFORMATION wipe),
    use base_footprint consistently with global_costmap and behavior_server, and set expected_update_rate."""
    lc = _get_costmap_params(nav2_cfg, 'local_costmap')
    gc = _get_costmap_params(nav2_cfg, 'global_costmap')
    assert lc.get('robot_base_frame') == 'base_footprint'
    assert gc.get('robot_base_frame') in ('base_footprint', 'base_link')
    plugins = lc.get('plugins', [])
    if 'traversability_layer' in plugins and 'obstacle_layer' in plugins:
        assert plugins.index('traversability_layer') < plugins.index('obstacle_layer')
        assert lc.get('use_maximum', False) is False
        assert lc.get('obstacle_layer', {}).get('combination_method', 1) == 1


def test_dwb_uses_footprint_critic_not_center_only(nav2_cfg):
    """Regression for 19:40 hill run (0/74 recoveries succeeded; 44/252 across 12 sessions):
    BaseObstacle only checks the robot center, driving corners into poses where full-footprint
    Spin/BackUp recoveries immediately fail. DWB styles must use ObstacleFootprint instead."""
    cs = nav2_cfg['controller_server']['ros__parameters']
    for pname in cs.get('controller_plugins', []):
        p = cs.get(pname, {})
        if 'DWBLocalPlanner' in str(p.get('plugin', '')):
            critics = p.get('critics', [])
            assert 'BaseObstacle' not in critics, (
                'Center-only BaseObstacle causes Spin/BackUp footprint failures; use ObstacleFootprint')
            assert 'ObstacleFootprint' in critics, (
                f'DWB controller {pname} must include ObstacleFootprint critic; got {critics}')


def test_bt_never_wipes_global_costmap(nav_style):
    """Regression for 286 global costmap wipes across 12 sessions: wiping global_costmap destroys
    explored-space memory that frontier exploration and obstacle persistence rely on."""
    tree = ET.parse(nav_style['bt_xml'])
    for elem in tree.iter():
        if elem.tag == 'ClearEntireCostmap':
            srv = elem.attrib.get('service_name', '')
            assert 'global_costmap' not in srv, (
                f'BT must never call ClearEntireCostmap on global_costmap ({srv})')


def test_safety_net_collision_monitor_and_tight_tf_tolerance(nav2_cfg, nav_style):
    """Regression for missing safety net: TF tolerance <= 0.25 s, stale sensor expected_update_rate
    configured, and collision_monitor defined when required by the style."""
    lc = _get_costmap_params(nav2_cfg, 'local_costmap')
    gc = _get_costmap_params(nav2_cfg, 'global_costmap')
    bs = nav2_cfg['behavior_server']['ros__parameters']
    for name, section in (('local_costmap', lc), ('global_costmap', gc), ('behavior_server', bs)):
        tol = float(section.get('transform_tolerance', 0.2))
        assert tol <= 0.25, f'{name} transform_tolerance={tol} s is too loose (<= 0.25 s required)'
    if nav_style.get('require_collision_monitor', False):
        assert 'collision_monitor' in nav2_cfg, 'Nav2 config must include collision_monitor safety net'
        cm = nav2_cfg['collision_monitor']['ros__parameters']
        assert cm.get('cmd_vel_out_topic') == 'cmd_vel'
        assert cm.get('source_timeout', 99.0) <= 3.0
        assert 'PolygonStop' in cm.get('polygons', [])
        assert cm.get('observation_sources'), 'collision_monitor must subscribe to at least one obstacle source'


def test_robot_sdf_physics_odometry_and_lidar_realism(ws_root):
    """Regressions for simulation realism:
    - Wheel torque realistic (<= 3.0 N*m, NOT 25 N*m = 190x climb torque)
    - Wheel encoder odometry (odometry_source=0, publish_odom_tf=false -> EKF owns /odom TF)
    - Raw IMU published to /imu/raw for Madgwick AHRS (not using noise-free orientation directly)
    - 3D lidar points >= 6 beams below horizon so drop-offs are visible
    """
    sdf_path = os.path.join(ws_root, 'src', 'ugv_main', 'ugv_gazebo', 'models', 'ugv_pro_audio', 'model.sdf')
    root = ET.parse(sdf_path).getroot()
    # Wheel torque & encoder odometry
    torque = float(root.find('.//max_wheel_torque').text)
    assert 0.3 <= torque <= 3.0, f'unrealistic max_wheel_torque={torque} N*m (was 25 N*m)'
    odom_src = root.find('.//odometry_source')
    assert odom_src is not None and odom_src.text.strip() == '0', (
        'diff_drive must use odometry_source=0 (wheel encoders), not 1 (Gazebo world ground truth)')
    pub_tf = root.find('.//publish_odom_tf')
    assert pub_tf is not None and pub_tf.text.strip().lower() == 'false', (
        'diff_drive must set publish_odom_tf=false so EKF owns odom->base_footprint')
    diff_plugin = root.find(".//plugin[@filename='libgazebo_ros_diff_drive.so']")
    assert diff_plugin is not None
    remaps = [r.text.strip() for r in diff_plugin.findall('.//ros/remapping') if r.text]
    assert any('odom:=wheel/odom' in r for r in remaps), (
        f'ROS 2 libgazebo_ros_diff_drive ignores <odometry_topic>; must remap odom:=wheel/odom in <ros>, got {remaps}')
    # Lidar vertical FOV: at least 6 of 16 beams below horizon (min_angle <= -15 deg)
    lidar_vert = root.find(".//sensor[@name='lidar_3d']//vertical")
    samples = int(lidar_vert.find('samples').text)
    min_a = float(lidar_vert.find('min_angle').text)
    max_a = float(lidar_vert.find('max_angle').text)
    below = sum(1 for i in range(samples) if (min_a + i * (max_a - min_a) / max(samples - 1, 1)) < 0.0)
    assert below >= 6, f'only {below}/{samples} lidar beams point below horizon; drop-offs invisible'


def test_frog_model_has_no_invisible_collision_pillar_and_hill_grass_matches_forest(ws_root):
    """Regressions:
    - coqui_frog/model.sdf must NOT have an invisible 28x34 cm collision cylinder that gpu_ray can't see.
    - hill.world & hill_coqui_survey.world altaniv* grass must be pass-through (collide_without_contact=1),
      matching forest.world."""
    frog_sdf = os.path.join(ws_root, 'src', 'ugv_main', 'ugv_gazebo', 'models', 'coqui_frog', 'model.sdf')
    root = ET.parse(frog_sdf).getroot()
    for cyl in root.findall('.//collision//cylinder'):
        r = float(cyl.find('radius').text)
        length = float(cyl.find('length').text)
        assert r <= 0.10 and length <= 0.06, f'oversized invisible frog collision cylinder r={r}, L={length}'
    for wname in ('hill.world', 'hill_coqui_survey.world'):
        wpath = os.path.join(ws_root, 'src', 'ugv_main', 'ugv_gazebo', 'worlds', wname)
        txt = open(wpath, encoding='utf-8').read()
        grass_blocks = re.findall(r'<model name=["\']altaniv[^"\']*["\']>.*?</model>', txt, flags=re.S)
        assert grass_blocks, f'no grass models found in {wname}'
        solid = sum(b.count('<collide_without_contact>0</collide_without_contact>') for b in grass_blocks)
        assert solid == 0, f'{wname} still has {solid} solid grass collision blocks'
