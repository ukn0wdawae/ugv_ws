"""ROBOT side of the Coqui survey: localization, traversability, Nav2 (+ collision monitor),
microphone array, pan-tilt camera, and the survey mission.

    ros2 launch ugv_coqui_behavior coqui_survey.launch.py [profile:=forest|hill]
         [localization:=ekf|slam|perfect] [use_sim_time:=true|false]
         [collision_monitor:=true|false] [rviz:=true|false]
         [nav2_params_file:=/path/to/custom_nav2.yaml] [bt_xml:=/path/to/custom_bt.xml]
         [traversability_params:=/path/to/custom_trav.yaml]
         [run_id:=...] [resume_survey:=/path/to/survey.yaml] [sim:=true gui:=true]

localization
  ekf      (default) Wheel encoder odometry (/wheel/odom) + Madgwick-filtered noisy IMU (/imu)
           fused by robot_localization EKF -> /odom + odom->base_footprint, plus static map->odom.
  slam     Same EKF odometry + RTAB-Map 3D lidar SLAM publishing map->odom (what the real robot needs).
  perfect  Simulation ablation mode: ground-truth /ugv_gt_odom -> /odom + identity map->odom.

Parameters are layered: config/survey_base.yaml + config/nav2_survey_base.yaml (or `nav2_params_file`),
then the `survey:` and `nav2:` sections of config/profiles/<profile>.yaml on top. The merged files are
written to /tmp and their paths are printed.
"""
import copy
import os
import tempfile

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def set_use_sim_time_recursive(doc, use_sim_time_bool):
    """Rewrite every `use_sim_time` key in a ROS 2 YAML parameter tree so sim time is never hard-coded."""
    if isinstance(doc, dict):
        return {
            k: (use_sim_time_bool if k == 'use_sim_time' else set_use_sim_time_recursive(v, use_sim_time_bool))
            for k, v in doc.items()
        }
    return doc


def merged_file(base_path, overlay, name, use_sim_time_bool=True):
    doc = deep_merge(yaml.safe_load(open(base_path)), overlay)
    doc = set_use_sim_time_recursive(doc, use_sim_time_bool)
    fd, path = tempfile.mkstemp(prefix=f'coqui_{name}_', suffix='.yaml')
    with os.fdopen(fd, 'w') as f:
        yaml.safe_dump(doc, f, sort_keys=False)
    return path, doc


def _build_nav2_nodes(nav2_yaml, use_sim_time_bool, use_collision_monitor):
    """Start Nav2 lifecycle nodes with velocity_smoother -> collision_monitor -> /cmd_vel safety net."""
    remappings = [('/tf', 'tf'), ('/tf_static', 'tf_static')]
    vel_out = 'cmd_vel_smoothed' if use_collision_monitor else 'cmd_vel'
    lifecycle_nodes = [
        'controller_server',
        'smoother_server',
        'planner_server',
        'behavior_server',
        'bt_navigator',
        'waypoint_follower',
        'velocity_smoother',
    ]
    nodes = [
        Node(
            package='nav2_controller',
            executable='controller_server',
            output='screen',
            parameters=[nav2_yaml],
            remappings=remappings + [('cmd_vel', 'cmd_vel_nav')],
        ),
        Node(
            package='nav2_smoother',
            executable='smoother_server',
            name='smoother_server',
            output='screen',
            parameters=[nav2_yaml],
            remappings=remappings,
        ),
        Node(
            package='nav2_planner',
            executable='planner_server',
            name='planner_server',
            output='screen',
            parameters=[nav2_yaml],
            remappings=remappings,
        ),
        Node(
            package='nav2_behaviors',
            executable='behavior_server',
            name='behavior_server',
            output='screen',
            parameters=[nav2_yaml],
            remappings=remappings + ([('cmd_vel', 'cmd_vel_smoothed')] if use_collision_monitor else []),
        ),
        Node(
            package='nav2_bt_navigator',
            executable='bt_navigator',
            name='bt_navigator',
            output='screen',
            parameters=[nav2_yaml],
            remappings=remappings,
        ),
        Node(
            package='nav2_waypoint_follower',
            executable='waypoint_follower',
            name='waypoint_follower',
            output='screen',
            parameters=[nav2_yaml],
            remappings=remappings,
        ),
        Node(
            package='nav2_velocity_smoother',
            executable='velocity_smoother',
            name='velocity_smoother',
            output='screen',
            parameters=[nav2_yaml],
            remappings=remappings + [('cmd_vel', 'cmd_vel_nav'), ('cmd_vel_smoothed', vel_out)],
        ),
    ]
    if use_collision_monitor:
        nodes.append(Node(
            package='nav2_collision_monitor',
            executable='collision_monitor',
            name='collision_monitor',
            output='screen',
            parameters=[nav2_yaml],
            remappings=remappings,
        ))
        lifecycle_nodes.append('collision_monitor')
    nodes.append(Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time_bool,
            'autostart': True,
            'node_names': lifecycle_nodes,
        }],
    ))
    return nodes


def setup(context):
    lc = lambda n: LaunchConfiguration(n).perform(context)  # noqa: E731
    me = get_package_share_directory('ugv_coqui_behavior')
    cfg = os.path.join(me, 'config')
    use_sim_time_str = lc('use_sim_time').strip().lower()
    use_sim_time_bool = use_sim_time_str in ('true', '1', 'yes')
    use_collision_monitor = lc('collision_monitor').strip().lower() in ('true', '1', 'yes')

    profile_path = lc('profile')
    if not os.path.isabs(profile_path):
        profile_path = os.path.join(cfg, 'profiles', profile_path + '.yaml')
    profile = yaml.safe_load(open(profile_path)) or {}

    survey_over = profile.get('survey', {})
    run_over = {}
    if lc('run_id'):
        run_over['run_id'] = lc('run_id')
    if lc('resume_survey'):
        run_over['resume_survey'] = lc('resume_survey')
    if lc('max_mission_sec'):
        run_over['max_mission_sec'] = float(lc('max_mission_sec'))
    if run_over:
        survey_over = deep_merge(survey_over, {'coqui_survey': {'ros__parameters': run_over}})
    survey_yaml, _ = merged_file(os.path.join(cfg, 'survey_base.yaml'), survey_over, 'survey', use_sim_time_bool)

    bt_xml_path = lc('bt_xml').strip() or os.path.join(cfg, 'bt_commit_to_path.xml')
    nav2_base_path = lc('nav2_params_file').strip() or os.path.join(cfg, 'nav2_survey_base.yaml')
    nav2_over = deep_merge(
        {'bt_navigator': {'ros__parameters': {
            'default_nav_to_pose_bt_xml': bt_xml_path,
        }}},
        profile.get('nav2', {}),
    )
    nav2_yaml, _ = merged_file(nav2_base_path, nav2_over, 'nav2', use_sim_time_bool)
    ekf_yaml, _ = merged_file(os.path.join(cfg, 'ekf_odom.yaml'), {}, 'ekf', use_sim_time_bool)
    imu_yaml, _ = merged_file(os.path.join(cfg, 'imu_filter.yaml'), {}, 'imu', use_sim_time_bool)

    acts = [LogInfo(msg=f'[coqui_survey] profile {profile_path} (localization={lc("localization")}, '
                        f'use_sim_time={use_sim_time_bool}, collision_monitor={use_collision_monitor})\n'
                        f'  survey params: {survey_yaml}\n  nav2 params:   {nav2_yaml}\n  bt xml:        {bt_xml_path}')]
    if lc('sim') == 'true':
        acts.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(me, 'launch', 'coqui_survey_sim.launch.py')),
            launch_arguments={'profile': lc('profile'), 'gui': lc('gui'), 'follow': lc('follow'),
                              'show_truth': lc('show_truth'), 'seed': lc('seed'),
                              'randomize': lc('randomize'),
                              'x': lc('x'), 'y': lc('y'), 'z': lc('z'), 'yaw': lc('yaw')}.items()))

    loc = lc('localization').strip().lower()
    if loc in ('ekf', 'slam'):
        # Realistic state estimation: Madgwick AHRS on noisy /imu/raw -> /imu,
        # fused with encoder /wheel/odom in robot_localization EKF -> /odom + odom->base_footprint
        acts += [
            Node(
                package='imu_filter_madgwick',
                executable='imu_filter_madgwick_node',
                name='imu_filter_madgwick',
                output='screen',
                parameters=[imu_yaml],
                remappings=[('imu/data_raw', '/imu/raw'), ('imu/data', '/imu')],
            ),
            Node(
                package='robot_localization',
                executable='ekf_node',
                name='ekf_filter_node',
                output='screen',
                parameters=[ekf_yaml],
                remappings=[('odometry/filtered', '/odom')],
            ),
        ]
        if loc == 'ekf':
            ip = profile.get('initial_pose', [0.0, 0.0, 0.0, 0.0])
            ix = lc('init_x') or lc('x') or str(ip[0])
            iy = lc('init_y') or lc('y') or str(ip[1])
            iz = lc('init_z') or lc('z') or str(ip[2])
            iyaw = lc('init_yaw') or lc('yaw') or str(ip[3])
            acts.append(Node(
                package='tf2_ros', executable='static_transform_publisher', name='map_to_odom_identity',
                arguments=[
                    '--x', ix, '--y', iy, '--z', iz,
                    '--yaw', iyaw, '--pitch', '0', '--roll', '0',
                    '--frame-id', 'map', '--child-frame-id', 'odom',
                ],
                parameters=[{'use_sim_time': use_sim_time_bool}],
            ))
        else:
            acts.append(IncludeLaunchDescription(
                PythonLaunchDescriptionSource(os.path.join(
                    get_package_share_directory('ugv_gazebo'), 'launch', 'slam', 'rtabmap_3d_lidar.launch.py')),
                launch_arguments={
                    'force_3dof': 'false',
                    'use_sim_time': use_sim_time_str,
                    'use_rviz': 'false',
                    'use_rtabmap_viz': 'false',
                    'rtabmap_localization': 'false',
                }.items(),
            ))
    elif loc == 'perfect':
        acts += [
            Node(
                package='ugv_coqui_behavior', executable='gt_odom_bridge', name='gt_odom_bridge',
                output='screen', parameters=[{'use_sim_time': use_sim_time_bool}],
            ),
            Node(
                package='tf2_ros', executable='static_transform_publisher', name='map_to_odom_identity',
                arguments=['--frame-id', 'map', '--child-frame-id', 'odom'],
                parameters=[{'use_sim_time': use_sim_time_bool}],
            ),
        ]
    else:
        raise RuntimeError(f'localization must be ekf, slam, or perfect, not {loc!r}')

    trav_args = {'use_sim_time': use_sim_time_str, 'rviz': 'false'}
    if lc('traversability_params').strip():
        trav_args['traversability_params'] = lc('traversability_params').strip()
    acts.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('ugv_traversability'), 'launch', 'traversability.launch.py')),
        launch_arguments=trav_args.items(),
    ))
    acts += _build_nav2_nodes(nav2_yaml, use_sim_time_bool, use_collision_monitor)
    acts += [
        Node(package='ugv_audio', executable='mic_array_localizer', name='mic_array_localizer', output='screen',
             parameters=[survey_yaml]),
        Node(package='ugv_coqui_behavior', executable='pt_camera_controller', name='pt_camera_controller',
             output='screen', parameters=[{'use_sim_time': use_sim_time_bool}]),
    ]
    if lc('mission') == 'true':
        acts.append(Node(package='ugv_coqui_behavior', executable='coqui_survey', name='coqui_survey',
                         output='screen', parameters=[survey_yaml]))
    viz_aids = lc('viz_aids') or lc('rviz')
    if viz_aids == 'true':
        acts.append(Node(package='ugv_coqui_behavior', executable='survey_viz_aids', name='survey_viz_aids',
                         output='screen', parameters=[{'use_sim_time': use_sim_time_bool}]))
    if lc('rviz') == 'true':
        acts.append(Node(package='rviz2', executable='rviz2', name='coqui_survey_rviz', output='log',
                         arguments=['-d', os.path.join(me, 'rviz', 'coqui_survey.rviz')],
                         parameters=[{'use_sim_time': use_sim_time_bool}]))
    return acts


def generate_launch_description():
    os.environ['UGV_MODEL'] = 'ugv_pro_audio'
    return LaunchDescription([
        DeclareLaunchArgument('profile', default_value='forest',
                              description='profile name in config/profiles/ or a path to a profile yaml'),
        DeclareLaunchArgument('localization', default_value='ekf',
                              description='ekf (encoder+IMU EKF) | slam (EKF+RTAB-Map 3D) | perfect (GT ablation)'),
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              description='true in Gazebo simulation, false on real robot hardware'),
        DeclareLaunchArgument('collision_monitor', default_value='true',
                              description='enable nav2_collision_monitor safety stop/slowdown on /cmd_vel'),
        DeclareLaunchArgument('nav2_params_file', default_value='',
                              description='optional override base Nav2 YAML (for testing new navigation styles)'),
        DeclareLaunchArgument('bt_xml', default_value='',
                              description='optional override Nav2 behavior tree XML'),
        DeclareLaunchArgument('traversability_params', default_value='',
                              description='optional override traversability YAML'),
        DeclareLaunchArgument('rviz', default_value='false'),
        DeclareLaunchArgument('run_id', default_value=''),
        DeclareLaunchArgument('resume_survey', default_value='',
                              description='survey.yaml of an earlier run: its frogs are treated as known'),
        DeclareLaunchArgument('max_mission_sec', default_value=''),
        DeclareLaunchArgument('sim', default_value='false', description='also start the simulation side'),
        DeclareLaunchArgument('gui', default_value='false', description='sim:=true only: Gazebo GUI'),
        DeclareLaunchArgument('follow', default_value='true',
                              description='sim:=true gui:=true only: the Gazebo view follows the robot'),
        DeclareLaunchArgument('show_truth', default_value='true',
                              description='sim:=true only: publish true frog positions for RViz viewer'),
        DeclareLaunchArgument('viz_aids', default_value='',
                              description='true/false: survey_viz_aids (default: on when rviz is on)'),
        DeclareLaunchArgument('seed', default_value='0'),
        DeclareLaunchArgument('randomize', default_value='false',
                              description='sim:=true only: randomize frog positions and spawn pose from seed'),
        DeclareLaunchArgument('mission', default_value='true',
                              description='false: everything but the survey mission (drive with teleop / Nav2 goals)'),
        DeclareLaunchArgument('x', default_value='', description='override spawn pose when sim:=true'),
        DeclareLaunchArgument('y', default_value=''),
        DeclareLaunchArgument('z', default_value=''),
        DeclareLaunchArgument('yaw', default_value=''),
        DeclareLaunchArgument('init_x', default_value='', description='override initial map->odom trailhead x'),
        DeclareLaunchArgument('init_y', default_value=''),
        DeclareLaunchArgument('init_z', default_value=''),
        DeclareLaunchArgument('init_yaw', default_value=''),
        OpaqueFunction(function=setup),
    ])
