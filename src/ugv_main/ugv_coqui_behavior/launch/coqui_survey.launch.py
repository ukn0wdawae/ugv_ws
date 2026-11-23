"""ROBOT side of the Coqui survey: localization, traversability, Nav2, microphone array, pan-tilt camera
and the survey mission.

    ros2 launch ugv_coqui_behavior coqui_survey.launch.py [profile:=forest] [localization:=perfect|slam]
         [rviz:=true] [run_id:=...] [resume_survey:=/path/to/survey.yaml] [sim:=true gui:=true]

localization
  perfect  map->odom is a static identity transform. Only valid in Gazebo, where /odom is ground truth.
           Use it to test the behaviour and navigation without SLAM errors.
  slam     RTAB-Map 3D lidar SLAM publishes map->odom (what the real robot needs).
sim:=true  also starts coqui_survey_sim.launch.py (world + robot + frog calls) in the same launch.

Parameters are layered: config/survey_base.yaml + config/nav2_survey_base.yaml, then the `survey:` and
`nav2:` sections of config/profiles/<profile>.yaml on top. The merged files are written to /tmp and
their paths are printed.
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


def merged_file(base_path, overlay, name):
    doc = deep_merge(yaml.safe_load(open(base_path)), overlay)
    fd, path = tempfile.mkstemp(prefix=f'coqui_{name}_', suffix='.yaml')
    with os.fdopen(fd, 'w') as f:
        yaml.safe_dump(doc, f, sort_keys=False)
    return path, doc


def setup(context):
    lc = lambda n: LaunchConfiguration(n).perform(context)  # noqa: E731
    me = get_package_share_directory('ugv_coqui_behavior')
    cfg = os.path.join(me, 'config')
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
    survey_yaml, _ = merged_file(os.path.join(cfg, 'survey_base.yaml'), survey_over, 'survey')
    nav2_yaml, _ = merged_file(os.path.join(cfg, 'nav2_survey_base.yaml'), profile.get('nav2', {}), 'nav2')

    acts = [LogInfo(msg=f'[coqui_survey] profile {profile_path}\n  survey params: {survey_yaml}\n'
                        f'  nav2 params:   {nav2_yaml}')]
    if lc('sim') == 'true':
        acts.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(me, 'launch', 'coqui_survey_sim.launch.py')),
            launch_arguments={'gui': lc('gui'), 'follow': lc('follow'), 'show_truth': lc('show_truth'),
                              'seed': lc('seed'), 'x': lc('x'), 'y': lc('y'), 'z': lc('z'),
                              'yaw': lc('yaw')}.items()))

    loc = lc('localization')
    if loc == 'perfect':
        acts.append(Node(package='tf2_ros', executable='static_transform_publisher', name='map_to_odom_identity',
                         arguments=['--frame-id', 'map', '--child-frame-id', 'odom'],
                         parameters=[{'use_sim_time': True}]))
    elif loc == 'slam':
        acts.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(get_package_share_directory('ugv_gazebo'), 'launch', 'slam',
                                                       'rtabmap_3d_lidar.launch.py')),
            launch_arguments={'force_3dof': 'false', 'use_sim_time': 'true', 'use_rviz': 'false'}.items()))
    else:
        raise RuntimeError(f'localization must be perfect or slam, not {loc!r}')

    acts += [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(get_package_share_directory('ugv_traversability'),
                                                       'launch', 'traversability.launch.py')),
            launch_arguments={'use_sim_time': 'true'}.items()),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(get_package_share_directory('nav2_bringup'),
                                                       'launch', 'navigation_launch.py')),
            launch_arguments={'use_sim_time': 'true', 'params_file': nav2_yaml, 'autostart': 'true'}.items()),
        Node(package='ugv_audio', executable='mic_array_localizer', name='mic_array_localizer', output='screen',
             parameters=[survey_yaml]),
        Node(package='ugv_coqui_behavior', executable='pt_camera_controller', name='pt_camera_controller',
             output='screen', parameters=[{'use_sim_time': True}]),
    ]
    if lc('mission') == 'true':
        acts.append(Node(package='ugv_coqui_behavior', executable='coqui_survey', name='coqui_survey',
                         output='screen', parameters=[survey_yaml]))
    viz_aids = lc('viz_aids') or lc('rviz')
    if viz_aids == 'true':
        # RViz helpers (no SLAM here): accumulated lidar map + costmaps lifted to the ground height
        acts.append(Node(package='ugv_coqui_behavior', executable='survey_viz_aids', name='survey_viz_aids',
                         output='screen', parameters=[{'use_sim_time': True}]))
    if lc('rviz') == 'true':
        acts.append(Node(package='rviz2', executable='rviz2', name='coqui_survey_rviz', output='log',
                         arguments=['-d', os.path.join(me, 'rviz', 'coqui_survey.rviz')],
                         parameters=[{'use_sim_time': True}]))
    return acts


def generate_launch_description():
    os.environ['UGV_MODEL'] = 'ugv_pro_audio'
    return LaunchDescription([
        DeclareLaunchArgument('profile', default_value='forest',
                              description='profile name in config/profiles/ or a path to a profile yaml'),
        DeclareLaunchArgument('localization', default_value='perfect', description='perfect | slam'),
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
                              description='sim:=true only: publish the true frog positions for RViz '
                                          '(/sim/coqui_truth_markers, viewer only; the robot never subscribes)'),
        DeclareLaunchArgument('viz_aids', default_value='',
                              description='true/false: survey_viz_aids (seen-map cloud, lifted costmaps). '
                                          'Default: on when rviz is on'),
        DeclareLaunchArgument('seed', default_value='0'),
        DeclareLaunchArgument('mission', default_value='true',
                              description='false: everything but the survey mission (drive with teleop / Nav2 goals)'),
        DeclareLaunchArgument('x', default_value='48.0', description='spawn pose when sim:=true'),
        DeclareLaunchArgument('y', default_value='2.1'),
        DeclareLaunchArgument('z', default_value='2.75'),
        DeclareLaunchArgument('yaw', default_value='-3.13'),
        OpaqueFunction(function=setup),
    ])
