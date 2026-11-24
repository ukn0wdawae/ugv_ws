import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _setup(context):
    UGV_MODEL = os.environ.get('UGV_MODEL', 'ugv_pro_audio')
    sdf_path = os.path.join(
        get_package_share_directory('ugv_gazebo'),
        'models',
        UGV_MODEL,
        'model.sdf',
    )
    use_sim_time = LaunchConfiguration('use_sim_time').perform(context).strip().lower() in ('true', '1', 'yes')
    start_ekf = LaunchConfiguration('start_ekf').perform(context).strip().lower() in ('true', '1', 'yes')

    acts = [
        Node(
            package='ugv_gazebo',
            executable='spawn_entity_clean.py',
            arguments=[
                '-entity', UGV_MODEL,
                '-file', sdf_path,
                '-x', LaunchConfiguration('x_pose').perform(context),
                '-y', LaunchConfiguration('y_pose').perform(context),
                '-z', LaunchConfiguration('z_pose').perform(context),
                '-Y', LaunchConfiguration('yaw_pose').perform(context),
                '-timeout', '120.0',
            ],
            output='screen',
        ),
    ]
    # For ugv_pro_audio (which publishes /wheel/odom + /imu/raw with publish_odom_tf=false),
    # start Madgwick IMU filter + EKF automatically in standalone bringup launches.
    # coqui_survey_sim passes start_ekf:=false because coqui_survey.launch.py manages localization.
    if UGV_MODEL == 'ugv_pro_audio' and start_ekf:
        beh_cfg = os.path.join(get_package_share_directory('ugv_coqui_behavior'), 'config')
        acts += [
            Node(
                package='imu_filter_madgwick',
                executable='imu_filter_madgwick_node',
                name='imu_filter_madgwick',
                output='screen',
                parameters=[os.path.join(beh_cfg, 'imu_filter.yaml'), {'use_sim_time': use_sim_time}],
                remappings=[('imu/data_raw', '/imu/raw'), ('imu/data', '/imu')],
            ),
            Node(
                package='robot_localization',
                executable='ekf_node',
                name='ekf_filter_node',
                output='screen',
                parameters=[os.path.join(beh_cfg, 'ekf_odom.yaml'), {'use_sim_time': use_sim_time}],
                remappings=[('odometry/filtered', '/odom')],
            ),
        ]
    return acts


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('x_pose', default_value='0.0', description='Initial x pose of the robot'),
        DeclareLaunchArgument('y_pose', default_value='0.0', description='Initial y pose of the robot'),
        DeclareLaunchArgument('z_pose', default_value='0.08', description='Initial z pose of the robot'),
        DeclareLaunchArgument('yaw_pose', default_value='0.0', description='Initial yaw orientation (radians)'),
        DeclareLaunchArgument('use_sim_time', default_value='true', description='Use simulation clock'),
        DeclareLaunchArgument('start_ekf', default_value='true',
                              description='Start Madgwick IMU + EKF for ugv_pro_audio standalone bringup'),
        OpaqueFunction(function=_setup),
    ])
