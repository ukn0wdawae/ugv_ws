#!/usr/bin/env python3
"""Launch Gazebo Classic with `ugv_world.world` and spawn `ugv_pro_audio`
(6-wheeled UGV Rover with true 6-wheel skid-steer physics, 16-channel 3D LiDAR,
2-axis Pan-Tilt camera, and omnidirectional 5-microphone 3D array).

Usage:
  ros2 launch ugv_gazebo bringup_ugv_pro_audio.launch.py
  ros2 launch ugv_gazebo bringup_ugv_pro_audio.launch.py gui:=false
  ros2 launch ugv_gazebo bringup_ugv_pro_audio.launch.py x_pose:=0.0 y_pose:=0.0 z_pose:=0.08
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def _configure_gazebo_env(ugv_model: str):
    pkg_ugv_gazebo = get_package_share_directory('ugv_gazebo')
    pkg_ugv_desc = get_package_share_directory('ugv_description')
    ws_root = os.path.abspath(os.path.join(pkg_ugv_gazebo, '..', '..', '..', '..'))

    # Include `share` parent of `ugv_description` so `model://ugv_description/meshes/...`
    # always resolves directly from the workspace install tree, plus `ugv_gazebo/models`.
    model_paths = [
        os.path.dirname(pkg_ugv_desc),
        os.path.join(pkg_ugv_gazebo, 'models'),
        os.path.join(ws_root, 'gazebo_models'),
        os.path.expanduser('~/.gazebo/models'),
        '/usr/share/gazebo-11/models',
    ]
    existing = [p for p in os.environ.get('GAZEBO_MODEL_PATH', '').split(':') if p]
    for p in reversed(model_paths):
        if os.path.isdir(p) and p not in existing:
            existing.insert(0, p)

    os.environ['GAZEBO_MODEL_PATH'] = ':'.join(existing)
    os.environ['GAZEBO_MODEL_DATABASE_URI'] = ''
    os.environ['UGV_MODEL'] = ugv_model


def _launch_setup(context, *args, **kwargs):
    ugv_model = LaunchConfiguration('ugv_model').perform(context).strip() or 'ugv_pro_audio'
    _configure_gazebo_env(ugv_model)

    pkg_ugv_gazebo = get_package_share_directory('ugv_gazebo')
    launch_file_dir = os.path.join(pkg_ugv_gazebo, 'launch', 'bringup')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')

    use_sim_time = LaunchConfiguration('use_sim_time')
    gui = LaunchConfiguration('gui').perform(context).strip().lower() in ('true', '1', 'yes')
    world_arg = LaunchConfiguration('world').perform(context).strip()
    world_path = world_arg or os.path.join(pkg_ugv_gazebo, 'worlds', 'ugv_world.world')

    actions = [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
            ),
            launch_arguments={
                'world': world_path,
                'verbose': 'true',
            }.items(),
        ),
    ]

    if gui:
        actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
                )
            )
        )

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
            description='Launch Gazebo GUI client (gzclient)',
        ),
        DeclareLaunchArgument(
            'world',
            default_value='',
            description='Full path to world file (defaults to ugv_world.world)',
        ),
        DeclareLaunchArgument('x_pose', default_value='0.0'),
        DeclareLaunchArgument('y_pose', default_value='0.0'),
        DeclareLaunchArgument('z_pose', default_value='0.08'),
        DeclareLaunchArgument('yaw_pose', default_value='0.0'),
        OpaqueFunction(function=_launch_setup),
    ])
