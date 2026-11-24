"""Start the terrain traversability node.

  ros2 launch ugv_traversability traversability.launch.py [use_sim_time:=true] [traversability_params:=...] [rviz:=true]

The params argument is deliberately NOT called `params_file`: launch configurations are global, and
gazebo_ros' gzserver.launch.py declares `params_file` with an empty default, which would silently
replace ours when both are included in one launch.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('ugv_traversability')
    use_sim_time = LaunchConfiguration('use_sim_time')
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('traversability_params', default_value=os.path.join(share, 'config', 'traversability.yaml')),
        DeclareLaunchArgument('rviz', default_value='false'),
        Node(
            package='ugv_traversability',
            executable='traversability_node',
            name='terrain_traversability',
            output='screen',
            parameters=[LaunchConfiguration('traversability_params'), {'use_sim_time': use_sim_time}],
        ),
        Node(
            package='rviz2', executable='rviz2', name='traversability_rviz',
            arguments=['-d', os.path.join(share, 'rviz', 'traversability.rviz')],
            parameters=[{'use_sim_time': use_sim_time}],
            sigterm_timeout='1', sigkill_timeout='2',
            condition=IfCondition(LaunchConfiguration('rviz')),
        ),
    ])
