#!/usr/bin/env python3
"""Launch Gazebo Classic with the hill world and spawn `ugv_pro_audio` (by default)."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def _ensure_gazebo_env(ugv_model: str):
    pkg_share = get_package_share_directory('ugv_gazebo')
    pkg_desc = get_package_share_directory('ugv_description')
    ws_root = os.path.abspath(os.path.join(pkg_share, '..', '..', '..', '..'))
    extra_paths = [
        os.path.dirname(pkg_desc),
        os.path.join(ws_root, 'gazebo_models'),
        '/root/ugv_ws/gazebo_models',
        os.path.join(pkg_share, 'models'),
    ]
    current = os.environ.get('GAZEBO_MODEL_PATH', '')
    parts = [p for p in current.split(':') if p]
    for p in reversed(extra_paths):
        if os.path.isdir(p) and p not in parts:
            parts.insert(0, p)
    os.environ['GAZEBO_MODEL_PATH'] = ':'.join(parts)
    os.environ['GAZEBO_MODEL_DATABASE_URI'] = ''
    os.environ['UGV_MODEL'] = ugv_model


def _launch_setup(context, *args, **kwargs):
    ugv_model = LaunchConfiguration('ugv_model').perform(context).strip() or 'ugv_pro_audio'
    _ensure_gazebo_env(ugv_model)

    launch_file_dir = os.path.join(
        get_package_share_directory('ugv_gazebo'), 'launch', 'bringup'
    )
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')

    use_sim_time = LaunchConfiguration('use_sim_time')
    tagged = LaunchConfiguration('tagged').perform(context).strip().lower() in ('true', '1', 'yes')
    gui = LaunchConfiguration('gui').perform(context).strip().lower() in ('true', '1', 'yes')
    spawn_robot = LaunchConfiguration('spawn_robot').perform(context).strip().lower() in ('true', '1', 'yes')

    world_name = 'hill_tagged.world' if tagged else 'hill.world'
    world = os.path.join(
        get_package_share_directory('ugv_gazebo'),
        'worlds',
        world_name,
    )

    actions = [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
            ),
            launch_arguments={
                'world': world,
                'verbose': 'true',
            }.items(),
        )
    ]

    if gui:
        actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
                )
            )
        )

    if spawn_robot:
        actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(launch_file_dir, 'robot_state_publisher.launch.py')
                ),
                launch_arguments={'use_sim_time': use_sim_time}.items(),
            )
        )
        actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(launch_file_dir, 'spawn_ugv.launch.py')
                ),
                launch_arguments={
                    'x_pose': LaunchConfiguration('x_pose'),
                    'y_pose': LaunchConfiguration('y_pose'),
                    'z_pose': LaunchConfiguration('z_pose'),
                    'yaw_pose': LaunchConfiguration('yaw_pose'),
                }.items(),
            )
        )

    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'ugv_model',
            default_value='ugv_pro_audio',
            description='UGV model to spawn (default: ugv_pro_audio)',
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use Gazebo simulation clock',
        ),
        DeclareLaunchArgument(
            'gui',
            default_value='true',
            description='Start Gazebo GUI client (gzclient)',
        ),
        DeclareLaunchArgument(
            'spawn_robot',
            default_value='true',
            description='Spawn the UGV and robot_state_publisher (set false for world-only)',
        ),
        DeclareLaunchArgument(
            'tagged',
            default_value='false',
            description='Use semantic-tagged world variant (hill_tagged.world)',
        ),
        DeclareLaunchArgument(
            'x_pose',
            default_value='43.0',
            description='Initial x pose of the robot (spot 1: 43.0, spot 2: 45.0)',
        ),
        DeclareLaunchArgument(
            'y_pose',
            default_value='-13.0',
            description='Initial y pose of the robot (spot 1: -13.0, spot 2: 12.0)',
        ),
        DeclareLaunchArgument(
            'z_pose',
            default_value='2.45',
            description='Initial z pose of the robot (spot 1: 2.45, spot 2: 13.05)',
        ),
        DeclareLaunchArgument(
            'yaw_pose',
            default_value='-3.0',
            description='Initial yaw orientation of the robot (spot 1: -3.0, spot 2: 0.0)',
        ),
        OpaqueFunction(function=_launch_setup),
    ])
