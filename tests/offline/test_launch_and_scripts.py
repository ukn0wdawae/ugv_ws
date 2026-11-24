"""Launch file and Docker helper script hygiene regression tests.

Covers:
  - SLAM mode argument collision (`localization:=slam` vs rtabmap_3d_lidar boolean)
  - Two RViz windows (`rviz:=true` leaking into traversability.launch.py)
  - No hard-coded sim time in launch files
  - docker/latest-scripts stale path messages, profile mismatch checks, and stale node checks
"""
import ast
import os


def test_rtabmap_launch_handles_non_bool_parent_localization_arg(ws_root):
    """Regression: rtabmap_3d_lidar.launch.py must NOT pass LaunchConfiguration('localization')
    to IfCondition/UnlessCondition directly, which threw InvalidConditionExpressionError on 'slam'."""
    p = os.path.join(ws_root, 'src', 'ugv_main', 'ugv_gazebo', 'launch', 'slam', 'rtabmap_3d_lidar.launch.py')
    txt = open(p, encoding='utf-8').read()
    ast.parse(txt)
    assert 'IfCondition(localization)' not in txt
    assert 'UnlessCondition(localization)' not in txt
    assert "{'use_sim_time': True}" not in txt, 'robot_pose_publisher must not hard-code use_sim_time: True'


def test_coqui_survey_launch_isolates_rviz_and_slam_and_sim_time(ws_root):
    """Regression: coqui_survey.launch.py must pass rviz:='false' to traversability.launch.py,
    rtabmap_localization:='false' to rtabmap_3d_lidar.launch.py, default localization to 'ekf',
    and expose use_sim_time + nav2_params_file + bt_xml launch arguments."""
    p = os.path.join(ws_root, 'src', 'ugv_main', 'ugv_coqui_behavior', 'launch', 'coqui_survey.launch.py')
    txt = open(p, encoding='utf-8').read()
    ast.parse(txt)
    assert "'rviz': 'false'" in txt, 'must suppress duplicate RViz in traversability.launch.py'
    assert "'traversability_params': trav_yaml" in txt, (
        'must explicitly resolve & forward traversability.yaml so empty arg never shadows traversability.launch.py')
    assert "'rtabmap_localization': 'false'" in txt, 'must isolate RTAB-Map localization boolean'
    assert "DeclareLaunchArgument('use_sim_time'" in txt
    assert "DeclareLaunchArgument('nav2_params_file'" in txt
    assert "DeclareLaunchArgument('bt_xml'" in txt
    assert "DeclareLaunchArgument('localization', default_value='ekf'" in txt


def test_docker_latest_scripts_paths_and_safety_checks(ws_root):
    """Regression: scripts in docker/latest-scripts/ must reference ./docker/latest-scripts/
    (not old ./docker/), check for stale nodes, and verify matching Terminal 1 / Terminal 2 profile."""
    sim_sh = open(os.path.join(ws_root, 'docker', 'latest-scripts', 'run_coqui_survey_sim.sh')).read()
    bot_sh = open(os.path.join(ws_root, 'docker', 'latest-scripts', 'run_coqui_survey.sh')).read()
    assert './docker/run_coqui_survey.sh' not in sim_sh
    assert './docker/run_coqui_survey_sim.sh' not in bot_sh
    assert './docker/latest-scripts/run_coqui_survey.sh' in sim_sh
    assert './docker/latest-scripts/run_coqui_survey_sim.sh' in bot_sh
    assert 'SIM_PROFILE' in sim_sh and 'SIM_PROFILE' in bot_sh
    assert 'Profile mismatch' in bot_sh
    assert 'coqui_sound_emitter' in sim_sh
    assert 'ros2 node list' in sim_sh and 'ros2 node list' in bot_sh


def test_docker_cleanup_and_shm_isolation(ws_root):
    """Regression: latest-scripts and trial runner must use _cleanup_lib.sh tag reaping so closing
    a terminal (SIGHUP) or double Ctrl+C never strands gzserver/rviz2/rtabmap/ros2-daemon, and
    docker/shell.sh + launch_docker_sim.sh must use --init and --shm-size instead of --ipc host."""
    cleanup_lib = os.path.join(ws_root, 'docker', 'latest-scripts', '_cleanup_lib.sh')
    assert os.path.isfile(cleanup_lib), '_cleanup_lib.sh helper must exist'
    lib_txt = open(cleanup_lib, encoding='utf-8').read()
    assert 'UGV_LAUNCH_TAG' in lib_txt and 'ros2-daemon' in lib_txt and 'fastrtps_' in lib_txt

    for rel in (
        ('docker', 'latest-scripts', 'run_coqui_survey_sim.sh'),
        ('docker', 'latest-scripts', 'run_coqui_survey.sh'),
        ('docker', 'run_coqui_survey_trial.sh'),
    ):
        txt = open(os.path.join(ws_root, *rel), encoding='utf-8').read()
        assert '_cleanup_lib.sh' in txt and 'ugv_init_launch_cleanup' in txt, f'{rel} missing cleanup trap'

    for rel in (('docker', 'shell.sh'), ('launch_docker_sim.sh',)):
        txt = open(os.path.join(ws_root, *rel), encoding='utf-8').read()
        assert '--init' in txt, f'{rel} must pass --init to reap zombies and handle SIGTERM'
        assert '--shm-size=' in txt, f'{rel} must allocate private container /dev/shm'
        assert '--ipc host' not in txt, f'{rel} must not leak FastDDS shm segments onto host /dev/shm'
        assert 'ALSOFT_DRIVERS' in txt, f'{rel} must set ALSOFT_DRIVERS=null to suppress ALSA errors'


def test_gazebo_profile_arg_isolation_and_clean_python_shutdown(ws_root):
    """Regression:
    - coqui_survey_sim.launch.py must pass profile:='' to gzserver.launch.py (avoid --profile=forest)
      and gui_required:='true' to gzclient.launch.py.
    - coqui_survey.launch.py must set fast sigterm_timeout on rviz2 so Ctrl+C does not hang 15s.
    - Python ROS 2 nodes must guard rclpy.shutdown() with rclpy.ok() and catch ExternalShutdownException."""
    sim_launch = open(os.path.join(
        ws_root, 'src', 'ugv_main', 'ugv_coqui_behavior', 'launch', 'coqui_survey_sim.launch.py')).read()
    assert "'profile': ''" in sim_launch
    assert "'gui_required': 'true'" in sim_launch

    bot_launch = open(os.path.join(
        ws_root, 'src', 'ugv_main', 'ugv_coqui_behavior', 'launch', 'coqui_survey.launch.py')).read()
    assert 'sigterm_timeout' in bot_launch

    for rel in (
        ('src', 'ugv_main', 'ugv_audio', 'ugv_audio', 'coqui_sound_emitter.py'),
        ('src', 'ugv_main', 'ugv_audio', 'ugv_audio', 'mic_array_localizer.py'),
        ('src', 'ugv_main', 'ugv_coqui_behavior', 'ugv_coqui_behavior', 'pt_camera_controller.py'),
        ('src', 'ugv_main', 'ugv_coqui_behavior', 'ugv_coqui_behavior', 'coqui_hunter_node.py'),
    ):
        py_txt = open(os.path.join(ws_root, *rel), encoding='utf-8').read()
        ast.parse(py_txt)
        assert 'ExternalShutdownException' in py_txt, f'{rel} must catch ExternalShutdownException'
        assert 'if rclpy.ok():' in py_txt, f'{rel} must guard rclpy.shutdown() with rclpy.ok()'


def test_per_run_frog_randomization_enabled_by_default(ws_root):
    """Regression: run_coqui_survey_sim.sh and coqui_survey_sim.launch.py must randomize frog
    positions by default on every run with a fresh non-zero seed, keep robot spawn aligned with
    EKF unless randomize_spawn:=true, and forward truth_out through coqui_survey.launch.py."""
    import xml.etree.ElementTree as ET
    import yaml
    from ugv_audio import audio_core as ac

    sim_sh = open(os.path.join(ws_root, 'docker', 'latest-scripts', 'run_coqui_survey_sim.sh'),
                  encoding='utf-8').read()
    assert 'RANDOMIZE="true"' in sim_sh
    assert 'SIM_TRUTH_FILE=' in sim_sh
    assert 'secrets.randbelow' in sim_sh

    sim_launch_path = os.path.join(
        ws_root, 'src', 'ugv_main', 'ugv_coqui_behavior', 'launch', 'coqui_survey_sim.launch.py')
    sim_launch = open(sim_launch_path, encoding='utf-8').read()
    ast.parse(sim_launch)
    assert "DeclareLaunchArgument('randomize', default_value='true'" in sim_launch
    assert "DeclareLaunchArgument('seed', default_value='auto'" in sim_launch
    assert "DeclareLaunchArgument('randomize_spawn', default_value='false'" in sim_launch
    assert "'seed': str(seed)" in sim_launch, 'must resolve auto seed to integer before forwarding to gzserver'

    bot_launch = open(os.path.join(
        ws_root, 'src', 'ugv_main', 'ugv_coqui_behavior', 'launch', 'coqui_survey.launch.py'),
        encoding='utf-8').read()
    assert "DeclareLaunchArgument('truth_out'" in bot_launch
    assert "'truth_out': lc('truth_out')" in bot_launch

    # Verify _rewrite_world_frogs produces valid SDF XML with exact randomized frog poses
    ns = {}
    with open(sim_launch_path, encoding='utf-8') as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == '_rewrite_world_frogs':
            mod_ast = ast.fix_missing_locations(ast.Module(body=[
                ast.Import(names=[ast.alias(name='math'), ast.alias(name='os'),
                                  ast.alias(name='re'), ast.alias(name='tempfile')]),
                node,
            ], type_ignores=[]))
            code = compile(mod_ast, sim_launch_path, 'exec')
            exec(code, ns)
    base_truth = yaml.safe_load(open(os.path.join(
        ws_root, 'src', 'ugv_main', 'ugv_audio', 'config', 'sim_truth', 'forest_survey.yaml')))
    rand_doc = ac.randomize_truth_scenario(base_truth, seed=77, survey_area=[18.0, 50.0, -15.0, 15.0],
                                           spawn_jitter_m=0.0)
    world_in = os.path.join(ws_root, 'src', 'ugv_main', 'ugv_gazebo', 'worlds', 'forest_coqui_survey.world')
    out_world = ns['_rewrite_world_frogs'](world_in, rand_doc)
    try:
        wroot = ET.parse(out_world).getroot().find('world')
        inc_poses = {
            inc.findtext('name'): [float(v) for v in inc.findtext('pose').split()[:3]]
            for inc in wroot.findall('include')
            if (inc.findtext('name') or '').startswith('coqui_')
        }
        assert len(inc_poses) == 6
        for fr in rand_doc['frogs']:
            assert inc_poses[fr['id']] == fr['pos']
    finally:
        if os.path.exists(out_world):
            os.remove(out_world)
