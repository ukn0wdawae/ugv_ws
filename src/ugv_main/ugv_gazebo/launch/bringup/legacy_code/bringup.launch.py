#!/usr/bin/env python3
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


def _configure_gazebo_env(ugv_model: str):
    pkg_ugv_gazebo = get_package_share_directory('ugv_gazebo')
    pkg_ugv_desc = get_package_share_directory('ugv_description')
    ws_root = os.path.abspath(os.path.join(pkg_ugv_gazebo, '..', '..', '..', '..'))
    gz_models_home = os.path.expanduser('~/.gazebo/models')
    synced = _sync_ugv_description_meshes(pkg_ugv_desc, gz_models_home)

    model_paths = [
        gz_models_home,
        os.path.join(pkg_ugv_gazebo, 'models'),
        os.path.join(ws_root, 'gazebo_models'),
        '/usr/share/gazebo-11/models',
    ]
    if not synced:
        model_paths.insert(0, os.path.dirname(pkg_ugv_desc))
    existing = [p for p in os.environ.get('GAZEBO_MODEL_PATH', '').split(':') if p]
    for p in reversed(model_paths):
        if os.path.isdir(p) and p not in existing:
            existing.insert(0, p)

    os.environ['GAZEBO_MODEL_PATH'] = ':'.join(existing)
    os.environ['GAZEBO_MODEL_DATABASE_URI'] = ''
    os.environ['UGV_MODEL'] = ugv_model


def _launch_setup(context, *args, **kwargs):
    default_model = os.environ.get('UGV_MODEL', 'ugv_pro_audio')
    ugv_model = LaunchConfiguration('ugv_model').perform(context).strip() or default_model
    _configure_gazebo_env(ugv_model)

    launch_file_dir = os.path.join(get_package_share_directory('ugv_gazebo'), 'launch', 'bringup')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    gui = LaunchConfiguration('gui').perform(context).strip().lower() in ('true', '1', 'yes')

    world = os.path.join(
        get_package_share_directory('ugv_gazebo'),
        'worlds',
        'ugv_world.world'
    )

    actions = [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
            ),
            launch_arguments={'world': world}.items()
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

    actions.append(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(launch_file_dir, 'robot_state_publisher.launch.py')
            ),
            launch_arguments={'use_sim_time': use_sim_time}.items()
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
            default_value=os.environ.get('UGV_MODEL', 'ugv_pro_audio'),
            description='UGV model to spawn (default: ugv_pro_audio)',
        ),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument('x_pose', default_value='0.0'),
        DeclareLaunchArgument('y_pose', default_value='0.0'),
        DeclareLaunchArgument('z_pose', default_value='0.08'),
        DeclareLaunchArgument('yaw_pose', default_value='0.0'),
        OpaqueFunction(function=_launch_setup),
    ])
