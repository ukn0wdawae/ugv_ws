import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    UGV_MODEL = os.environ.get('UGV_MODEL', 'ugv_pro_audio')

    sdf_path = os.path.join(
        get_package_share_directory('ugv_gazebo'),
        'models',
        UGV_MODEL,
        'model.sdf'
    )

    declare_x_position_cmd = DeclareLaunchArgument(
        'x_pose', default_value='0.0',
        description='Initial x pose of the robot')

    declare_y_position_cmd = DeclareLaunchArgument(
        'y_pose', default_value='0.0',
        description='Initial y pose of the robot')

    declare_z_position_cmd = DeclareLaunchArgument(
        'z_pose', default_value='0.08',
        description='Initial z pose of the robot (clearance above ground)')

    declare_yaw_position_cmd = DeclareLaunchArgument(
        'yaw_pose', default_value='0.0',
        description='Initial yaw orientation (radians) of the robot')

    start_gazebo_ros_spawner_cmd = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', UGV_MODEL,
            '-file', sdf_path,
            '-x', LaunchConfiguration('x_pose'),
            '-y', LaunchConfiguration('y_pose'),
            '-z', LaunchConfiguration('z_pose'),
            '-Y', LaunchConfiguration('yaw_pose'),
            '-timeout', '120.0',
        ],
        output='screen',
    )

    ld = LaunchDescription()
    ld.add_action(declare_x_position_cmd)
    ld.add_action(declare_y_position_cmd)
    ld.add_action(declare_z_position_cmd)
    ld.add_action(declare_yaw_position_cmd)
    ld.add_action(start_gazebo_ros_spawner_cmd)

    return ld
