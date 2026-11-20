#!/usr/bin/env python3
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
