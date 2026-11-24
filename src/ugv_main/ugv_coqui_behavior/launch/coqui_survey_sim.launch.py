"""SIMULATION side of the Coqui survey: Gazebo world with the frogs + robot + the frogs' calls.

    ros2 launch ugv_coqui_behavior coqui_survey_sim.launch.py [profile:=forest|hill]
         [gui:=true] [follow:=false] [show_truth:=false] [seed:=auto|<N>] [randomize:=true|false]
         [randomize_spawn:=false] [motor_noise_gain:=0.06] [overlap_prob:=0.25] [echo_gain:=0.12]
         [schedule:=chorus|turns] [call_variation:=true]
         [x:= y:= z:= yaw:=]

This is "the world". The robot stack (coqui_survey.launch.py) runs separately and never reads the
truth file used here, nor anything published under /sim/. By default (`randomize:=true`, `seed:=auto`),
every launch generates a fresh random seed and places frogs across real 3D bushes, tree crowns/trunks,
tall grass, and glades in a temporary world and truth file. Pass `seed:=<N>` to replay a specific layout,
or `randomize:=false` for the static canonical truth file.
"""
import math
import os
import re
import secrets
import tempfile

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, LogInfo, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ugv_audio.audio_core import randomize_truth_scenario

ROBOT_MODEL = 'ugv_pro_audio'   # UGV_MODEL = the robot's Gazebo model (entity) name, see spawn_ugv.launch.py
FOLLOW_TIMEOUT_SEC = 180        # give up waiting for the robot and the Gazebo GUI after this long

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


def _rewrite_world_frogs(world_path, truth_doc):
    """Create a temporary Gazebo world file with `coqui_frog` includes updated from `truth_doc`."""
    txt = open(world_path, encoding='utf-8').read()
    # Strip existing coqui_frog includes
    txt = re.sub(
        r'\s*<include>\s*<name>coqui_\d+</name>.*?</include>',
        '', txt, flags=re.S,
    )
    spawn = truth_doc.get('spawn', [0.0, 0.0, 0.0, 0.0])
    frogs_xml = []
    for fr in truth_doc.get('frogs', []):
        x, y, z = fr['pos']
        yaw = float(fr['yaw']) if 'yaw' in fr else math.atan2(spawn[1] - y, spawn[0] - x)
        frogs_xml.append(
            f'\n    <include>\n'
            f'      <name>{fr["id"]}</name>\n'
            f'      <uri>model://coqui_frog</uri>\n'
            f'      <static>true</static>\n'
            f'      <pose>{x} {y} {z} 0 0 {yaw:.3f}</pose>\n'
            f'    </include>'
        )
    idx = txt.rindex('</world>')
    txt = txt[:idx] + ''.join(frogs_xml) + '\n  ' + txt[idx:]
    fd, out_world = tempfile.mkstemp(prefix='coqui_rand_world_', suffix='.world')
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        f.write(txt)
    return out_world


def setup(context):
    lc = lambda n: LaunchConfiguration(n).perform(context)  # noqa: E731
    ugv = get_package_share_directory('ugv_gazebo')
    gz = get_package_share_directory('gazebo_ros')
    beh = get_package_share_directory('ugv_coqui_behavior')
    bring = os.path.join(ugv, 'launch', 'bringup')

    # model paths (same as bringup_forest.launch.py; avoid adding install/ugv_description/share
    # when ~/.gazebo/models/ugv_description exists so gzclient does not warn on ament_index/colcon-core)
    ws_root = os.path.abspath(os.path.join(ugv, '..', '..', '..', '..'))
    gz_models_home = os.path.expanduser('~/.gazebo/models')
    extra = [os.path.join(ws_root, 'gazebo_models'), '/root/ugv_ws/gazebo_models',
             gz_models_home, os.path.join(ugv, 'models')]
    if not os.path.isdir(os.path.join(gz_models_home, 'ugv_description')):
        extra.insert(0, os.path.dirname(get_package_share_directory('ugv_description')))
    parts = [p for p in os.environ.get('GAZEBO_MODEL_PATH', '').split(':') if p]
    for p in reversed(extra):
        if os.path.isdir(p) and p not in parts:
            parts.insert(0, p)
    os.environ['GAZEBO_MODEL_PATH'] = ':'.join(parts)
    os.environ['GAZEBO_MODEL_DATABASE_URI'] = ''

    profile = lc('profile') or 'forest'
    randomize = lc('randomize').strip().lower() in ('true', '1', 'yes')
    randomize_spawn = lc('randomize_spawn').strip().lower() in ('true', '1', 'yes')
    seed_raw = lc('seed').strip().lower()
    if randomize:
        if seed_raw in ('', 'auto', '0'):
            seed = secrets.randbelow(900000) + 100000
        else:
            seed = int(seed_raw)
    else:
        seed = 0 if seed_raw in ('', 'auto') else int(seed_raw)

    audio_share = get_package_share_directory('ugv_audio')
    truth_file = lc('truth_file') or os.path.join(audio_share, 'config', 'sim_truth', f'{profile}_survey.yaml')
    world = lc('world') or os.path.join(ugv, 'worlds', f'{profile}_coqui_survey.world')

    spawn = [48.0, 2.1, 2.75, -3.13]
    t_doc = yaml.safe_load(open(truth_file)) if os.path.isfile(truth_file) else {}
    domain = os.environ.get('ROS_DOMAIN_ID', '0').strip() or '0'
    latest_truth_domain = f'/tmp/coqui_latest_truth_{profile}_d{domain}.yaml'
    if randomize and t_doc:
        prof_path = os.path.join(beh, 'config', 'profiles', f'{profile}.yaml')
        survey_area = None
        if os.path.isfile(prof_path):
            pdoc = yaml.safe_load(open(prof_path)) or {}
            survey_area = pdoc.get('survey', {}).get('coqui_survey', {}).get('ros__parameters', {}).get('survey_area')
        # Keep trailhead spawn aligned with Terminal 2's static EKF map->odom unless randomize_spawn:=true
        t_doc = randomize_truth_scenario(
            t_doc, seed, survey_area=survey_area,
            spawn_jitter_m=(1.5 if randomize_spawn else 0.0),
        )
        truth_out = lc('truth_out').strip()
        if truth_out:
            os.makedirs(os.path.dirname(os.path.abspath(truth_out)), exist_ok=True)
            with open(truth_out, 'w', encoding='utf-8') as f:
                yaml.safe_dump(t_doc, f, sort_keys=False)
            truth_file = truth_out
        else:
            fd, truth_file = tempfile.mkstemp(prefix=f'coqui_truth_{profile}_s{seed}_', suffix='.yaml')
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                yaml.safe_dump(t_doc, f, sort_keys=False)
        world = _rewrite_world_frogs(world, t_doc)
    elif t_doc:
        truth_out = lc('truth_out').strip()
        if truth_out:
            os.makedirs(os.path.dirname(os.path.abspath(truth_out)), exist_ok=True)
            with open(truth_out, 'w', encoding='utf-8') as f:
                yaml.safe_dump(t_doc, f, sort_keys=False)

    if t_doc:
        try:
            with open(latest_truth_domain, 'w', encoding='utf-8') as f:
                yaml.safe_dump(t_doc, f, sort_keys=False)
        except OSError:
            pass

    if t_doc and 'spawn' in t_doc and len(t_doc['spawn']) >= 4:
        spawn = t_doc['spawn']
    sx = lc('x') or str(spawn[0])
    sy = lc('y') or str(spawn[1])
    sz = lc('z') or str(spawn[2])
    syaw = lc('yaw') or str(spawn[3])
    frog_summary = '\n'.join(
        f'  {fr["id"]}: pos={fr["pos"]}  [{fr.get("habitat", "")}]'
        for fr in (t_doc.get('frogs', []) if t_doc else [])
    )
    acts = [
        LogInfo(msg=f'[coqui_survey_sim] profile={profile} seed={seed} randomize={randomize} '
                    f'world={world} truth={truth_file} spawn=({sx}, {sy}, {sz}, {syaw})\n'
                    f'{frog_summary}'),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(gz, 'launch', 'gzserver.launch.py')),
                                 launch_arguments={'world': world, 'verbose': 'false',
                                                   'profile': '', 'seed': str(seed)}.items()),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(bring, 'robot_state_publisher.launch.py')),
                                 launch_arguments={'use_sim_time': 'true'}.items()),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(bring, 'spawn_ugv.launch.py')),
                                 launch_arguments={'x_pose': sx, 'y_pose': sy, 'z_pose': sz,
                                                   'yaw_pose': syaw, 'start_ekf': 'false'}.items()),
        Node(package='ugv_audio', executable='coqui_sound_emitter', name='coqui_sound_emitter', output='screen',
             parameters=[{
                 'use_sim_time': True,
                 'mode': 'survey',
                 'truth_file': truth_file,
                 'seed': seed,
                 'start_delay_sec': 8.0,
                 'schedule': lc('schedule'),
                 'call_period_sec': 0.25 if lc('schedule') == 'chorus' else 0.75,   # audio frames per second
                 'max_audible_range_m': 30.0,
                 'call_variation': lc('call_variation') == 'true',
                 'publish_truth_markers': lc('show_truth') == 'true',
                 'motor_noise_gain': float(lc('motor_noise_gain')),
                 'overlap_prob': float(lc('overlap_prob')),
                 'echo_gain': float(lc('echo_gain')),
                 'ambient_clutter_prob': float(lc('ambient_clutter_prob')),
             }]),
    ]
    if lc('gui') == 'true':
        acts.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(gz, 'launch', 'gzclient.launch.py')),
            launch_arguments={'gui_required': 'true'}.items()))
        if lc('follow') == 'true':
            acts.append(ExecuteProcess(
                cmd=['bash', '-c', FOLLOW_SCRIPT, 'gazebo_follow', ROBOT_MODEL, str(FOLLOW_TIMEOUT_SEC)],
                name='gazebo_follow', output='screen'))
    return acts


def generate_launch_description():
    os.environ['UGV_MODEL'] = ROBOT_MODEL   # read by spawn_ugv / robot_state_publisher launch files
    return LaunchDescription([
        DeclareLaunchArgument('profile', default_value='forest',
                              description='world profile (forest | hill); picks <profile>_coqui_survey.world and spawn'),
        DeclareLaunchArgument('world', default_value='', description='override world path'),
        DeclareLaunchArgument('truth_file', default_value='', description='override frog truth file'),
        DeclareLaunchArgument('truth_out', default_value='',
                              description='optional path to write randomized truth YAML for eval_survey'),
        DeclareLaunchArgument('gui', default_value='false', description='start the Gazebo GUI (gzclient)'),
        DeclareLaunchArgument('follow', default_value='true',
                              description='with gui:=true, the Gazebo GUI camera follows the robot'),
        DeclareLaunchArgument('show_truth', default_value='true',
                              description='draw true frog positions on /sim/coqui_truth_markers (viewer only)'),
        DeclareLaunchArgument('seed', default_value='auto',
                              description='random seed for schedule & layout ("auto" generates a new seed each run)'),
        DeclareLaunchArgument('randomize', default_value='true',
                              description='when true (default), place frogs at randomized 3D bush/tree/ground spots'),
        DeclareLaunchArgument('randomize_spawn', default_value='false',
                              description='also perturb robot trailhead spawn pose (keep false unless syncing init_x/y/yaw)'),
        DeclareLaunchArgument('motor_noise_gain', default_value='0.06',
                              description='motor self-noise amplitude proportional to wheel velocity'),
        DeclareLaunchArgument('overlap_prob', default_value='0.25',
                              description='probability of overlapping second frog call in the same frame'),
        DeclareLaunchArgument('echo_gain', default_value='0.12',
                              description='multipath foliage/ground echo gain'),
        DeclareLaunchArgument('ambient_clutter_prob', default_value='0.20',
                              description='probability of non-Coqui insect/bird clutter'),
        DeclareLaunchArgument('schedule', default_value='chorus',
                              description='chorus: every frog on its own rhythm | turns: one frog at a time'),
        DeclareLaunchArgument('call_variation', default_value='true',
                              description='call level varies per frog, call, path and direction'),
        DeclareLaunchArgument('x', default_value=''),
        DeclareLaunchArgument('y', default_value=''),
        DeclareLaunchArgument('z', default_value=''),
        DeclareLaunchArgument('yaw', default_value=''),
        OpaqueFunction(function=setup),
    ])
