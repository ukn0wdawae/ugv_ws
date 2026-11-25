#!/usr/bin/env python3
"""Spawn the UGV + robot_state_publisher into an already-running Gazebo natural environment.

Migrated from ROS 1 `add_husky_<environment>_<spot>.launch` to ROS 2 Humble.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

# Presets migrated from natural_environments/launch/add_husky_<env>_<spot>.launch
ENV_SPOTS = {
    'park': {
        '1': ('47.0', '1.0', '4.0', '3.05'),
        '2': ('47.0', '1.0', '4.0', '3.05'),
    },
    'lake': {
        '1': ('-47.0', '-15.0', '4.0', '0.0'),
        '2': ('-47.0', '-15.0', '4.0', '0.0'),
    },
    'hill': {
        '1': ('43.0', '-13.0', '2.45', '-3.0'),
        '2': ('45.0', '12.0', '13.05', '0.0'),
    },
    'forest': {
        '1': ('48.0', '2.1', '4.0', '-3.13'),
        '2': ('-37.0', '-22.75', '4.0', '0.0'),
    },
}


def _launch_setup(context, *args, **kwargs):
    if 'UGV_MODEL' not in os.environ or not os.environ['UGV_MODEL']:
        os.environ['UGV_MODEL'] = 'ugv_rover'

    launch_file_dir = os.path.join(
        get_package_share_directory('ugv_gazebo'), 'launch', 'bringup'
    )

    env_name = LaunchConfiguration('env').perform(context).strip().lower()
    spot = LaunchConfiguration('spot').perform(context).strip()
    default_x, default_y, default_z, default_yaw = ENV_SPOTS.get(
        env_name, ENV_SPOTS['park']
    ).get(spot, ENV_SPOTS.get(env_name, ENV_SPOTS['park'])['1'])

    x_pose = LaunchConfiguration('x_pose').perform(context).strip() or default_x
    y_pose = LaunchConfiguration('y_pose').perform(context).strip() or default_y
    z_pose = LaunchConfiguration('z_pose').perform(context).strip() or default_z
    yaw_pose = LaunchConfiguration('yaw_pose').perform(context).strip() or default_yaw
    use_sim_time = LaunchConfiguration('use_sim_time')

    robot_state_publisher_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, 'robot_state_publisher.launch.py')
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items(),
    )

    spawn_ugv_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, 'spawn_ugv.launch.py')
        ),
        launch_arguments={
            'x_pose': x_pose,
            'y_pose': y_pose,
            'z_pose': z_pose,
            'yaw_pose': yaw_pose,
        }.items(),
    )

    return [robot_state_publisher_cmd, spawn_ugv_cmd]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use Gazebo simulation clock',
        ),
        DeclareLaunchArgument(
            'env',
            default_value='park',
            description='Target environment preset: park, lake, hill, or forest',
        ),
        DeclareLaunchArgument(
            'spot',
            default_value='1',
            description='Spawn location preset (1 or 2) from natural_environments',
        ),
        DeclareLaunchArgument(
            'x_pose',
            default_value='',
            description='Override x pose (leave empty to use env/spot preset)',
        ),
        DeclareLaunchArgument(
            'y_pose',
            default_value='',
            description='Override y pose (leave empty to use env/spot preset)',
        ),
        DeclareLaunchArgument(
            'z_pose',
            default_value='',
            description='Override z pose (leave empty to use env/spot preset)',
        ),
        DeclareLaunchArgument(
            'yaw_pose',
            default_value='',
            description='Override yaw pose in radians (leave empty to use env/spot preset)',
        ),
        OpaqueFunction(function=_launch_setup),
    ])
