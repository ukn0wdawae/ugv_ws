"""RTAB-Map SLAM: 2D LiDAR (/scan) + RGB-D depth camera (original ugv_rover setup).

Publishes:
  * /map                  -> 2D occupancy grid from 2D LiDAR (/scan) for Nav2 + explore_lite
  * /rtabmap_3d/cloud_map -> 3D voxel cloud built from the depth camera
  * /mapData              -> per-node RGB-D frames for RViz MapCloud plugin

Usage (with ./docker/run_ugv_world_sim.sh or ./docker/run_grass_sim.sh running):
  ros2 launch ugv_gazebo rtabmap_rgbd_2d.launch.py use_sim_time:=true reg_strategy:=1 use_rviz:=true
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch_setup(context, *args, **kwargs):
    use_sim_time = LaunchConfiguration('use_sim_time')
    qos = LaunchConfiguration('qos')
    localization = LaunchConfiguration('localization')
    reg_strategy = str(LaunchConfiguration('reg_strategy').perform(context)).strip()
    cloud_range = str(LaunchConfiguration('cloud_range').perform(context)).strip()
    rviz_config = LaunchConfiguration('rviz_config').perform(context).strip()

    parameters = {
        'frame_id': 'base_footprint',
        'use_sim_time': use_sim_time,
        'subscribe_depth': True,
        'subscribe_rgb': True,
        'subscribe_scan': True,
        'use_action_for_goal': True,
        'qos_image': qos,
        'qos_camera_info': qos,
        'qos_scan': qos,
        'qos_odom': qos,
        'qos_imu': qos,
        'Reg/Force3DoF': 'true',
        'Optimizer/GravitySigma': '0',

        'approx_sync': True,
        'sync_queue_size': 30,
        'topic_queue_size': 30,
        'queue_size': 30,
        'map_always_update': True,

        # 0=visual (original default), 1=ICP on the 2D scan
        'Reg/Strategy': reg_strategy,
        'RGBD/NeighborLinkRefining': 'true',
        'RGBD/ProximityBySpace': 'true',
        'Icp/VoxelSize': '0.05',
        'Icp/MaxCorrespondenceDistance': '0.1',

        'Rtabmap/DetectionRate': '2',
        'RGBD/LinearUpdate': '0.05',
        'RGBD/AngularUpdate': '0.05',

        # /map for Nav2 + explore_lite: 2D lidar occupancy grid with ray tracing
        'RGBD/CreateOccupancyGrid': 'true',
        'Grid/Sensor': '0',
        'Grid/3D': 'false',
        'Grid/RayTracing': 'true',
        'Grid/RangeMax': '12.0',
        'Grid/CellSize': '0.05',

        'Mem/ImagePreDecimation': '1',
        'Mem/ImagePostDecimation': '1',
    }

    remappings = [
        ('rgb/image', '/camera/image_raw'),
        ('rgb/camera_info', '/camera/camera_info'),
        ('depth/image', '/camera/depth/image_raw'),
        ('scan', '/scan'),
        ('odom', '/odom'),
    ]

    rtabmap_slam = Node(
        condition=UnlessCondition(localization),
        package='rtabmap_slam',
        executable='rtabmap',
        output='screen',
        parameters=[parameters],
        remappings=remappings,
        arguments=['-d'],
    )

    rtabmap_localization = Node(
        condition=IfCondition(localization),
        package='rtabmap_slam',
        executable='rtabmap',
        output='screen',
        parameters=[
            parameters,
            {'Mem/IncrementalMemory': 'False', 'Mem/InitWMWithAllNodes': 'True'},
        ],
        remappings=remappings,
    )

    map_assembler_3d = Node(
        package='rtabmap_util',
        executable='map_assembler',
        name='map_assembler_3d',
        namespace='rtabmap_3d',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'Grid/Sensor': '1',
            'Grid/3D': 'true',
            'Grid/RayTracing': 'true',
            'Grid/CellSize': '0.05',
            'Grid/RangeMax': cloud_range,
            'Grid/DepthDecimation': '2',
            'Grid/NormalsSegmentation': 'false',
            'Grid/MaxGroundHeight': '0.05',
            'Grid/MaxObstacleHeight': '3.0',
            'Grid/MinClusterSize': '3',
            'Grid/NoiseFilteringRadius': '0.0',
            'regenerate_local_grids': True,
        }],
        remappings=[('mapData', '/mapData')],
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(LaunchConfiguration('use_rviz')),
    )

    rtabmap_viz = Node(
        package='rtabmap_viz',
        executable='rtabmap_viz',
        output='screen',
        parameters=[parameters],
        remappings=remappings,
        condition=UnlessCondition(LaunchConfiguration('use_rviz')),
    )

    robot_pose_publisher = Node(
        package='robot_pose_publisher',
        executable='robot_pose_publisher',
        name='robot_pose_publisher',
        output='screen',
        emulate_tty=True,
        parameters=[
            {'use_sim_time': True},
            {'is_stamped': True},
            {'map_frame': 'map'},
            {'base_frame': 'base_footprint'},
        ],
    )

    return [
        rtabmap_slam,
        rtabmap_localization,
        map_assembler_3d,
        rviz,
        rtabmap_viz,
        robot_pose_publisher,
    ]


def generate_launch_description():
    default_rviz = os.path.join(
        get_package_share_directory('ugv_gazebo'), 'rviz', 'view_slam_3d.rviz'
    )
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument(
            'qos', default_value='2', description='QoS for sensor topics (2 = best effort)'
        ),
        DeclareLaunchArgument('localization', default_value='false'),
        DeclareLaunchArgument(
            'use_rviz', default_value='true', description='RViz2 (true) or rtabmap_viz (false)'
        ),
        DeclareLaunchArgument('rviz_config', default_value=default_rviz),
        DeclareLaunchArgument(
            'reg_strategy',
            default_value='1',
            description='0=visual (original repo), 1=ICP on 2D scan',
        ),
        DeclareLaunchArgument(
            'cloud_range',
            default_value='4.0',
            description='Max depth (m) used for the 3D cloud_map',
        ),
        OpaqueFunction(function=_launch_setup),
    ])
