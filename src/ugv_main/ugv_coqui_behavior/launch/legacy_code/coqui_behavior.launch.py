#!/usr/bin/env python3
"""Launch the complete Coqui Hunter stack:
  1. 3D/2.5D RTAB-Map RGB-D + 3D LiDAR SLAM (`/map`, `/mapData`, `map->odom` TF)
  2. 3D/2.5D Nav2 Navigation Stack (`NavigateToPose`, `/global_costmap/costmap`, `/plan`)
  3. RViz2 (`view_coqui_3d.rviz`) showing what the robot sees, acoustic 3D DOA rays,
     triangulated frog positions, and sound-based navigation goals sent to Nav2
  4. `ugv_audio` (`coqui_sound_emitter` + `mic_array_localizer`)
  5. `ugv_coqui_behavior` (`pt_camera_controller` + `coqui_hunter_node`)
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _as_bool(val: str) -> bool:
    return str(val).strip().lower() in ('true', '1', 'yes')


def _launch_setup(context, *args, **kwargs):
    use_sim_time = LaunchConfiguration('use_sim_time')
    env_name = LaunchConfiguration('env').perform(context).strip().lower()
    start_delay_sec = LaunchConfiguration('start_delay_sec')
    start_audio = _as_bool(LaunchConfiguration('start_audio').perform(context))
    start_slam = _as_bool(LaunchConfiguration('start_slam').perform(context))
    start_nav2 = _as_bool(LaunchConfiguration('start_nav2').perform(context))
    use_rviz = _as_bool(LaunchConfiguration('use_rviz').perform(context))
    single_frog = _as_bool(LaunchConfiguration('single_frog_mission').perform(context))

    pkg_ugv_gazebo = get_package_share_directory('ugv_gazebo')
    pkg_ugv_audio = get_package_share_directory('ugv_audio')

    rviz_coqui_config = os.path.join(pkg_ugv_gazebo, 'rviz', 'view_coqui_3d.rviz')
    audio_launch = os.path.join(pkg_ugv_audio, 'launch', 'audio_localization.launch.py')
    rtabmap_launch = os.path.join(pkg_ugv_gazebo, 'launch', 'slam', 'rtabmap_3d_lidar.launch.py')

    # Use 3DoF on flat park world; 6DoF + normal ground segmentation on sloped terrain worlds
    force_3dof = 'true' if env_name == 'park' else 'false'
    terrain_nav_yaml = os.path.join(pkg_ugv_gazebo, 'param', 'slam_nav_terrain.yaml')
    flat_nav_yaml = os.path.join(pkg_ugv_gazebo, 'param', 'slam_nav_3d.yaml')
    nav_params_file = (
        terrain_nav_yaml
        if (force_3dof == 'false' and os.path.exists(terrain_nav_yaml))
        else flat_nav_yaml
    )

    actions = []

    # 1. 3D/2.5D RTAB-Map SLAM + RViz (`view_coqui_3d.rviz`)
    if start_slam:
        actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(rtabmap_launch),
                launch_arguments={
                    'use_sim_time': use_sim_time,
                    'use_rviz': 'true' if use_rviz else 'false',
                    'rviz_config': rviz_coqui_config,
                    'force_3dof': force_3dof,
                }.items(),
            )
        )
    elif use_rviz:
        actions.append(
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                output='screen',
                arguments=['-d', rviz_coqui_config],
                parameters=[{'use_sim_time': use_sim_time}],
            )
        )

    # 2. 3D/2.5D Nav2 Navigation Stack (`NavigateToPose`)
    if start_nav2:
        try:
            pkg_nav2_bringup = get_package_share_directory('nav2_bringup')
            nav2_launch = os.path.join(pkg_nav2_bringup, 'launch', 'navigation_launch.py')
            actions.append(
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(nav2_launch),
                    launch_arguments={
                        'use_sim_time': 'true',
                        'params_file': nav_params_file,
                    }.items(),
                )
            )
        except Exception:
            pass

    # 3. Multi-Frog 3D Acoustic Emitter + 5-Channel Microphone Array Localizer
    if start_audio:
        actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(audio_launch),
                launch_arguments={
                    'use_sim_time': use_sim_time,
                    'env': env_name,
                    'start_delay_sec': start_delay_sec,
                }.items(),
            )
        )

    # 4. 2-Axis Pan-Tilt Camera Controller + Coqui Hunter Behavior Node
    actions.append(
        Node(
            package='ugv_coqui_behavior',
            executable='pt_camera_controller',
            name='pt_camera_controller',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}],
        )
    )
    actions.append(
        Node(
            package='ugv_coqui_behavior',
            executable='coqui_hunter_node',
            name='coqui_hunter_node',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'use_nav2_if_available': start_nav2,
                'single_frog_mission': single_frog,
            }],
        )
    )

    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument(
            'env',
            default_value='park',
            description='Compact Coqui environment preset: park, lake, hill, or forest',
        ),
        DeclareLaunchArgument(
            'start_delay_sec',
            default_value='4.5',
            description='Seconds before Coqui frogs begin calling',
        ),
        DeclareLaunchArgument(
            'start_audio',
            default_value='true',
            description='Launch ugv_audio multi-frog emitter and 5-mic array localizer',
        ),
        DeclareLaunchArgument(
            'start_slam',
            default_value='true',
            description='Launch RTAB-Map 3D/2.5D SLAM (/map and /mapData)',
        ),
        DeclareLaunchArgument(
            'start_nav2',
            default_value='true',
            description='Launch 3D/2.5D Nav2 navigation stack (NavigateToPose)',
        ),
        DeclareLaunchArgument(
            'use_rviz',
            default_value='true',
            description='Launch RViz2 with view_coqui_3d.rviz',
        ),
        DeclareLaunchArgument(
            'single_frog_mission',
            default_value='false',
            description='If true, stop after spotting 1 frog and returning home; if false, hunt all frogs on map returning home after each',
        ),
        OpaqueFunction(function=_launch_setup),
    ])
