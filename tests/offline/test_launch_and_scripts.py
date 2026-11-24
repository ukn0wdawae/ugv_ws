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
