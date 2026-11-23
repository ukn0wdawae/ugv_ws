"""SIMULATION side of the Coqui survey: Gazebo world with the frogs + robot + the frogs' calls.

    ros2 launch ugv_coqui_behavior coqui_survey_sim.launch.py [gui:=true] [follow:=false] [show_truth:=false]
                                                              [x:= y:= z:= yaw:=]

This is "the world". The robot stack (coqui_survey.launch.py) runs separately and never reads the
truth file used here, nor anything published under /sim/. With gui:=true the Gazebo window opens
behind the robot spawn and, with follow:=true, its camera follows the robot once the robot is there.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ROBOT_MODEL = 'ugv_pro_audio'   # UGV_MODEL = the robot's Gazebo model (entity) name, see spawn_ugv.launch.py
FOLLOW_TIMEOUT_SEC = 180        # give up waiting for the robot and the Gazebo GUI after this long

# bash, args $1 model $2 timeout [s]: once the robot model and the Gazebo GUI camera both exist, make the GUI camera
# follow the robot. gzclient (Gazebo 11) names its view camera 'gzclient_camera'; 'user_camera' is only the name in
# the world file's <gui>. 'gz model -i' exits 0 even for an unknown model, so its output is checked. The gz tools
# talk to the Gazebo master of GAZEBO_MASTER_URI (inherited environment).
FOLLOW_SCRIPT = r"""
trap 'exit 0' INT TERM
deadline=$((SECONDS + $2))
while [ "$SECONDS" -lt "$deadline" ]; do
  sleep 2
  timeout 10 gz model -m "$1" -i 2>/dev/null | grep -q "^name: \"$1\"" || continue
  cam=$(timeout 10 gz camera -l 2>/dev/null | grep -m1 -E '^(user_camera|gzclient_camera[0-9]*)$') || continue
  sleep 2
  timeout 10 gz camera -c "$cam" -f "$1" > /dev/null 2>&1 || exit 0
  echo "Gazebo view now follows $1. To stop: click into the Gazebo 3D view and press Esc" \
       "(right-click the robot > Follow to follow it again)."
  exit 0
done
exit 0
"""


def setup(context):
    lc = lambda n: LaunchConfiguration(n).perform(context)  # noqa: E731
    ugv = get_package_share_directory('ugv_gazebo')
    gz = get_package_share_directory('gazebo_ros')
    bring = os.path.join(ugv, 'launch', 'bringup')

    # model paths (same as bringup_forest.launch.py)
    ws_root = os.path.abspath(os.path.join(ugv, '..', '..', '..', '..'))
    extra = [os.path.dirname(get_package_share_directory('ugv_description')),
             os.path.join(ws_root, 'gazebo_models'), '/root/ugv_ws/gazebo_models',
             os.path.expanduser('~/.gazebo/models'), os.path.join(ugv, 'models')]
    parts = [p for p in os.environ.get('GAZEBO_MODEL_PATH', '').split(':') if p]
    for p in reversed(extra):
        if os.path.isdir(p) and p not in parts:
            parts.insert(0, p)
    os.environ['GAZEBO_MODEL_PATH'] = ':'.join(parts)
    os.environ['GAZEBO_MODEL_DATABASE_URI'] = ''

    world = lc('world') or os.path.join(ugv, 'worlds', 'forest_coqui_survey.world')
    acts = [
        IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(gz, 'launch', 'gzserver.launch.py')),
                                 launch_arguments={'world': world, 'verbose': 'false'}.items()),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(bring, 'robot_state_publisher.launch.py')),
                                 launch_arguments={'use_sim_time': 'true'}.items()),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(bring, 'spawn_ugv.launch.py')),
                                 launch_arguments={'x_pose': lc('x'), 'y_pose': lc('y'), 'z_pose': lc('z'),
                                                   'yaw_pose': lc('yaw')}.items()),
        Node(package='ugv_audio', executable='coqui_sound_emitter', name='coqui_sound_emitter', output='screen',
             parameters=[{'use_sim_time': True, 'mode': 'survey', 'truth_file': lc('truth_file'),
                          'seed': int(lc('seed')), 'start_delay_sec': 8.0,
                          'publish_truth_markers': lc('show_truth') == 'true'}]),
    ]
    if lc('gui') == 'true':
        acts.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(gz, 'launch', 'gzclient.launch.py'))))
        if lc('follow') == 'true':
            acts.append(ExecuteProcess(
                cmd=['bash', '-c', FOLLOW_SCRIPT, 'gazebo_follow', ROBOT_MODEL, str(FOLLOW_TIMEOUT_SEC)],
                name='gazebo_follow', output='screen'))
    return acts


def generate_launch_description():
    os.environ['UGV_MODEL'] = ROBOT_MODEL   # read by spawn_ugv / robot_state_publisher launch files
    return LaunchDescription([
        DeclareLaunchArgument('world', default_value='', description='default: forest_coqui_survey.world'),
        DeclareLaunchArgument('truth_file', default_value='',
                              description='frog positions for the simulated calls (default: forest_survey.yaml)'),
        DeclareLaunchArgument('gui', default_value='false', description='start the Gazebo GUI (gzclient)'),
        DeclareLaunchArgument('follow', default_value='true',
                              description='with gui:=true, the Gazebo GUI camera follows the robot once both exist '
                                          '(press Esc in the Gazebo 3D view to stop following)'),
        DeclareLaunchArgument('show_truth', default_value='true',
                              description='draw the true frog positions and the calling frog on '
                                          '/sim/coqui_truth_markers: a viewer-only RViz layer (simulation answer '
                                          'key); the robot side never subscribes to it'),
        DeclareLaunchArgument('seed', default_value='0', description='random seed of the frogs calling schedule'),
        DeclareLaunchArgument('x', default_value='48.0'),
        DeclareLaunchArgument('y', default_value='2.1'),
        DeclareLaunchArgument('z', default_value='2.75'),
        DeclareLaunchArgument('yaw', default_value='-3.13'),
        OpaqueFunction(function=setup),
    ])
