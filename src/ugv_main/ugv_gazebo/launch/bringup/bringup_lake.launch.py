#!/usr/bin/env python3
"""Launch Gazebo Classic with the lake (lakeshore) world (and optionally spawn the UGV).

Migrated from ROS 1 `create_lake.launch` + `add_husky_lake_1.launch` to ROS 2 Humble.
Use `spawn_robot:=false` to launch ONLY the world first.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def _ensure_gazebo_env():
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
    if True:
        os.environ['UGV_MODEL'] = 'ugv_pro_audio'


def _launch_setup(context, *args, **kwargs):
    _ensure_gazebo_env()

    launch_file_dir = os.path.join(
        get_package_share_directory('ugv_gazebo'), 'launch', 'bringup'
    )
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')

    use_sim_time = LaunchConfiguration('use_sim_time')
    tagged = LaunchConfiguration('tagged').perform(context).strip().lower() in ('true', '1', 'yes')
    gui = LaunchConfiguration('gui').perform(context).strip().lower() in ('true', '1', 'yes')
    spawn_robot = LaunchConfiguration('spawn_robot').perform(context).strip().lower() in ('true', '1', 'yes')

    world_name = 'lake_tagged.world' if tagged else 'lake.world'
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
            description='Use semantic-tagged world variant (lake_tagged.world)',
        ),
        DeclareLaunchArgument(
            'x_pose',
            default_value='-47.0',
            description='Initial x pose of the robot',
        ),
        DeclareLaunchArgument(
            'y_pose',
            default_value='-15.0',
            description='Initial y pose of the robot',
        ),
        DeclareLaunchArgument(
            'z_pose',
            default_value='4.0',
            description='Initial z pose of the robot',
        ),
        DeclareLaunchArgument(
            'yaw_pose',
            default_value='0.0',
            description='Initial yaw orientation of the robot',
        ),
        OpaqueFunction(function=_launch_setup),
    ])
