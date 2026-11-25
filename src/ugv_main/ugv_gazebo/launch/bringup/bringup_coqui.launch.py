#!/usr/bin/env python3
"""Launch Gazebo Classic with one of the 4 extended compact Coqui frog worlds
(park_coqui, lake_coqui, hill_coqui, forest_coqui) and spawn `ugv_pro_audio`
(equipped with the 5-channel 3D microphone array and 2-axis Pan-Tilt camera).

Examples:
  ros2 launch ugv_gazebo bringup_coqui.launch.py env:=park
  ros2 launch ugv_gazebo bringup_coqui.launch.py env:=lake
  ros2 launch ugv_gazebo bringup_coqui.launch.py env:=hill
  ros2 launch ugv_gazebo bringup_coqui.launch.py env:=forest
"""

import os
import shutil
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


COQUI_ENV_PRESETS = {
    'park': ('park_coqui.world', '47.0', '1.0', '3.03', '3.05'),
    'lake': ('lake_coqui.world', '-47.0', '-15.0', '3.85', '0.0'),
    'hill': ('hill_coqui.world', '43.0', '-13.0', '2.45', '-3.0'),
    'forest': ('forest_coqui.world', '48.0', '2.1', '2.75', '-3.13'),
}


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
    # Default to ugv_pro_audio for Coqui acoustic missions
    os.environ['UGV_MODEL'] = 'ugv_pro_audio'


def _launch_setup(context, *args, **kwargs):
    _ensure_gazebo_env()

    pkg_ugv_gazebo = get_package_share_directory('ugv_gazebo')
    launch_file_dir = os.path.join(pkg_ugv_gazebo, 'launch', 'bringup')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')

    env_name = LaunchConfiguration('env').perform(context).strip().lower()
    gui = LaunchConfiguration('gui').perform(context).strip().lower() in ('true', '1', 'yes')
    spawn_robot = LaunchConfiguration('spawn_robot').perform(context).strip().lower() in ('true', '1', 'yes')
    use_sim_time = LaunchConfiguration('use_sim_time')

    world_file, def_x, def_y, def_z, def_yaw = COQUI_ENV_PRESETS.get(
        env_name, COQUI_ENV_PRESETS['park']
    )
    x_pose = LaunchConfiguration('x_pose').perform(context).strip() or def_x
    y_pose = LaunchConfiguration('y_pose').perform(context).strip() or def_y
    z_pose = LaunchConfiguration('z_pose').perform(context).strip() or def_z
    yaw_pose = LaunchConfiguration('yaw_pose').perform(context).strip() or def_yaw

    world_path = os.path.join(pkg_ugv_gazebo, 'worlds', world_file)

    actions = [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
            ),
            launch_arguments={
                'world': world_path,
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
                    'x_pose': x_pose,
                    'y_pose': y_pose,
                    'z_pose': z_pose,
                    'yaw_pose': yaw_pose,
                }.items(),
            )
        )

    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument('spawn_robot', default_value='true'),
        DeclareLaunchArgument(
            'env',
            default_value='park',
            description='Compact Coqui environment to launch: park, lake, hill, or forest',
        ),
        DeclareLaunchArgument('x_pose', default_value=''),
        DeclareLaunchArgument('y_pose', default_value=''),
        DeclareLaunchArgument('z_pose', default_value=''),
        DeclareLaunchArgument('yaw_pose', default_value=''),
        OpaqueFunction(function=_launch_setup),
    ])

