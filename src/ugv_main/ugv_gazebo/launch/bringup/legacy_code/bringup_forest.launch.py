#!/usr/bin/env python3
"""Launch Gazebo Classic with the forest world (and optionally spawn the UGV).

Migrated from ROS 1 `create_forest.launch` + `add_husky_forest_1.launch` to ROS 2 Humble.
Use `spawn_robot:=false` to launch ONLY the world first.
"""

import os
import shutil

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def _sync_ugv_description_meshes(pkg_desc: str, gz_models_home: str) -> bool:
    src_meshes = os.path.join(pkg_desc, 'meshes')
    dst_root = os.path.join(gz_models_home, 'ugv_description')
    if not os.path.isdir(src_meshes):
        return os.path.isdir(dst_root)
    try:
        os.makedirs(dst_root, exist_ok=True)
        dst_cfg = os.path.join(dst_root, 'model.config')
        src_cfg = os.path.join(pkg_desc, 'model.config')
        if not os.path.isfile(dst_cfg) and os.path.isfile(src_cfg):
            shutil.copy2(src_cfg, dst_cfg)
        for root, _, files in os.walk(src_meshes):
            rel = os.path.relpath(root, src_meshes)
            target_dir = os.path.join(dst_root, 'meshes') if rel == '.' else os.path.join(dst_root, 'meshes', rel)
            os.makedirs(target_dir, exist_ok=True)
            for fn in files:
                s_file = os.path.join(root, fn)
                d_file = os.path.join(target_dir, fn)
                if not os.path.isfile(d_file) or os.path.getsize(s_file) != os.path.getsize(d_file):
                    shutil.copy2(s_file, d_file)
        return True
    except OSError:
        return False


def _ensure_gazebo_env():
    pkg_share = get_package_share_directory('ugv_gazebo')
    pkg_desc = get_package_share_directory('ugv_description')
    ws_root = os.path.abspath(os.path.join(pkg_share, '..', '..', '..', '..'))
    gz_models_home = os.path.expanduser('~/.gazebo/models')
    synced = _sync_ugv_description_meshes(pkg_desc, gz_models_home)
    extra_paths = [
        gz_models_home,
        os.path.join(ws_root, 'gazebo_models'),
        '/root/ugv_ws/gazebo_models',
        os.path.join(pkg_share, 'models'),
    ]
    if not synced:
        extra_paths.insert(0, os.path.dirname(pkg_desc))
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

    world_name = 'forest_tagged.world' if tagged else 'forest.world'
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
            description='Use semantic-tagged world variant (forest_tagged.world)',
        ),
        DeclareLaunchArgument(
            'x_pose',
            default_value='48.0',
            description='Initial x pose of the robot (spot 1: 48.0, spot 2: -37.0)',
        ),
        DeclareLaunchArgument(
            'y_pose',
            default_value='2.1',
            description='Initial y pose of the robot (spot 1: 2.1, spot 2: -22.75)',
        ),
        DeclareLaunchArgument(
            'z_pose',
            default_value='4.0',
            description='Initial z pose of the robot',
        ),
        DeclareLaunchArgument(
            'yaw_pose',
            default_value='-3.13',
            description='Initial yaw orientation of the robot (spot 1: -3.13, spot 2: 0.0)',
        ),
        OpaqueFunction(function=_launch_setup),
    ])
