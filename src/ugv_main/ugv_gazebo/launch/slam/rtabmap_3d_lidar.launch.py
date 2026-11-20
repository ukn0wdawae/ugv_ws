from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def _as_bool(value):
    return str(value).strip().lower() in ('true', '1', 'yes')


def launch_setup(context, *args, **kwargs):
    use_sim_time = LaunchConfiguration('use_sim_time')
    queue_size = LaunchConfiguration('queue_size')
    qos = LaunchConfiguration('qos')
    localization = LaunchConfiguration('localization')
    force_3dof = _as_bool(LaunchConfiguration('force_3dof').perform(context))

    param_file = os.path.join(
        get_package_share_directory('ugv_gazebo'), 'param', 'rtabmap_slam.yaml')
    default_rviz_config = os.path.join(
        get_package_share_directory('ugv_gazebo'), 'rviz', 'view_slam_3d_lidar.rviz')
    rviz_slam_3d_config = LaunchConfiguration('rviz_config').perform(context).strip() or default_rviz_config

    # Flat floor: 3D lidar + depth camera grid (Grid/Sensor=2) + 3DoF, like slam_toolbox.
    # Static hills: 6DoF + IMU gravity + normals ground segmentation on fused 3D lidar + depth cloud
    # so a tilted sensor slice does not paint the floor as occupied.
    terrain = {
        'Reg/Force3DoF': 'true' if force_3dof else 'false',
        'Optimizer/GravitySigma': '0' if force_3dof else '0.3',
        'subscribe_imu': True,
        'wait_imu_to_init': False if force_3dof else True,
        'Grid/Sensor': '2',
        'Grid/RangeMax': '20.0',
        'Grid/NormalsSegmentation': 'false' if force_3dof else 'true',
        'Grid/MaxGroundHeight': '0.15' if force_3dof else '0.25',
    }

    parameters = [
        param_file,
        {
            'use_sim_time': use_sim_time,
            'queue_size': queue_size,
            'sync_queue_size': queue_size,
            'qos_image': qos,
            'qos_camera_info': qos,
            'qos_imu': qos,
            'qos_scan': qos,
            'qos_odom': qos,
            'qos_gps': qos,
            'subscribe_scan': False,
            'subscribe_scan_cloud': True,
        },
        terrain,
    ]

    remappings = [
        ('rgb/image', '/camera/image_raw'),
        ('rgb/camera_info', '/camera/camera_info'),
        ('depth/image', '/camera/depth/image_raw'),
        ('scan_cloud', '/scan_cloud'),
        ('gps/fix', '/gps/fix'),
        ('odom', '/odom'),
        ('imu', '/imu'),
    ]

    rgbd_sync_node = Node(
        package='rtabmap_sync',
        executable='rgbd_sync',
        output='screen',
        parameters=[{
            'approx_sync': True,
            'approx_sync_max_interval': 0.05,
            'use_sim_time': use_sim_time,
            'qos': qos,
        }],
        remappings=remappings,
    )

    rtabmap_slam_node_slam = Node(
        condition=UnlessCondition(localization),
        package='rtabmap_slam',
        executable='rtabmap',
        output='screen',
        parameters=parameters,
        remappings=remappings,
        arguments=['-d'],
    )

    rtabmap_slam_node_localization = Node(
        condition=IfCondition(localization),
        package='rtabmap_slam',
        executable='rtabmap',
        output='screen',
        parameters=parameters + [
            {'Mem/IncrementalMemory': 'False',
             'Mem/InitWMWithAllNodes': 'True'},
        ],
        remappings=remappings,
    )

    rviz2_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_slam_3d_config],
        condition=IfCondition(LaunchConfiguration('use_rviz')),
    )

    rtabmap_viz_node = Node(
        package='rtabmap_viz',
        executable='rtabmap_viz',
        output='screen',
        parameters=parameters,
        remappings=remappings,
        condition=UnlessCondition(LaunchConfiguration('use_rviz')),
    )

    robot_pose_publisher_node = Node(
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

    nodes = [
        rgbd_sync_node,
        rtabmap_slam_node_slam,
        rtabmap_slam_node_localization,
        rviz2_node,
        rtabmap_viz_node,
        robot_pose_publisher_node,
    ]

    # Static hills: split the 3D lidar cloud into ground / obstacles by surface normals
    # so Nav2 costmaps (param/slam_nav_terrain.yaml) do not treat slopes as walls.
    if not force_3dof:
        nodes.append(Node(
            package='rtabmap_util',
            executable='obstacles_detection',
            name='obstacles_detection',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'frame_id': 'base_footprint',
                'Grid/NormalsSegmentation': 'true',
                'Grid/MaxGroundAngle': '35',
                'Grid/MaxGroundHeight': '0.0',
                'Grid/MaxObstacleHeight': '1.0',
                'Grid/RangeMax': '5.0',
                'Grid/RangeMin': '0.2',
                'Grid/ClusterRadius': '0.1',
                'Grid/MinClusterSize': '5',
            }],
            remappings=[
                ('cloud', '/scan_cloud'),
                ('obstacles', '/obstacles_cloud'),
                ('ground', '/ground_cloud'),
            ],
        ))

    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time', default_value='true',
            description='Use simulation (Gazebo) clock if true'),
        DeclareLaunchArgument(
            'queue_size', default_value='30',
            description='Sync / topic queue size'),
        DeclareLaunchArgument(
            'qos', default_value='2',
            description='QoS used for input sensor topics'),
        DeclareLaunchArgument(
            'localization', default_value='false',
            description='Launch in localization mode.'),
        DeclareLaunchArgument(
            'use_rviz', default_value='false',
            description='Whether to launch RViz2'),
        DeclareLaunchArgument(
            'rviz_config', default_value='',
            description='Optional custom RViz2 config file path'),
        DeclareLaunchArgument(
            'force_3dof', default_value='true',
            description='true: slam_toolbox-like planar mapping. '
                        'false: 6DoF + IMU for static elevation changes. '
                        'Moving ground is still not supported.'),
        OpaqueFunction(function=launch_setup),
    ])
