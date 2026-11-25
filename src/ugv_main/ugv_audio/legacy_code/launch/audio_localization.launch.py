#!/usr/bin/env python3
"""Launch the Coqui Multi-Frog 3D Acoustic Emitter and 5-Microphone Array Localizer nodes."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    env = LaunchConfiguration('env')
    start_delay_sec = LaunchConfiguration('start_delay_sec')

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument(
            'env',
            default_value='park',
            description='Environment preset: park, lake, hill, or forest',
        ),
        DeclareLaunchArgument(
            'start_delay_sec',
            default_value='4.5',
            description='Seconds after launch before Coqui frogs begin calling (lets robot start exploring first)',
        ),
        Node(
            package='ugv_audio',
            executable='coqui_sound_emitter',
            name='coqui_sound_emitter',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'env': env,
                'start_delay_sec': start_delay_sec,
            }],
        ),
        Node(
            package='ugv_audio',
            executable='mic_array_localizer',
            name='mic_array_localizer',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
            }],
        ),
    ])
