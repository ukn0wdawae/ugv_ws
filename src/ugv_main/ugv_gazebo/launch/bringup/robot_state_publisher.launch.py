import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()

    UGV_MODEL = os.environ.get('UGV_MODEL', 'ugv_pro_audio')
    urdf_file_name = UGV_MODEL + '.urdf'
    urdf_model_path = os.path.join(
        get_package_share_directory('ugv_gazebo'),
        'urdf',
        urdf_file_name
    )

    with open(urdf_model_path, 'r') as infp:
        robot_desc = infp.read()

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_description': robot_desc,
        }],
    )

    ld.add_action(use_sim_time_arg)
    ld.add_action(robot_state_publisher_node)

    # Note: For ugv_pro_audio, Gazebo's libgazebo_ros_joint_state_publisher.so
    # already publishes all 6 wheel joints + both Pan-Tilt joints at 30 Hz on /joint_states.
    # Only launch joint_state_publisher for legacy models whose Pan-Tilt joints are not in Gazebo.
    if UGV_MODEL != 'ugv_pro_audio':
        joint_state_publisher_node = Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            name='joint_state_publisher',
            arguments=[urdf_model_path],
            parameters=[{
                'use_sim_time': use_sim_time,
                'source_list': ['ugv/joint_states'],
            }],
        )
        ld.add_action(joint_state_publisher_node)

    return ld
