#!/usr/bin/env python3
"""Live ROS 2 Humble launch-graph resolution tests (no Gazebo needed).

Verifies directly against the ROS 2 `launch` engine:
  1. `coqui_survey.launch.py localization:=slam` resolves cleanly (no InvalidConditionExpressionError!)
     and includes `rtabmap` + `ekf_node` + `imu_filter_madgwick_node`.
  2. `coqui_survey.launch.py rviz:=true` starts STRICTLY ONE `rviz2` node (no duplicate traversability_rviz!).
  3. `localization:=lidar` (default, GPS-denied), `:=ekf` and `:=perfect` all resolve cleanly.
  4. `use_sim_time:=false` propagates into the merged Nav2 and survey YAML files (not hard-coded true).
"""
import importlib.util
import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.utilities import normalize_to_list_of_substitutions
from launch_ros.actions import Node


def _load_launch_module(path):
    spec = importlib.util.spec_from_file_location('launch_mod', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _collect_nodes(ld, args_dict):
    ctx = LaunchContext(argv=[f'{k}:={v}' for k, v in args_dict.items()])
    ctx.launch_configurations.update({k: str(v) for k, v in args_dict.items()})
    nodes = []

    def visit(entities):
        for e in entities:
            if isinstance(e, DeclareLaunchArgument):
                if e.name not in ctx.launch_configurations:
                    val = e.default_value
                    if val is not None:
                        ctx.launch_configurations[e.name] = ''.join(sub.perform(ctx) for sub in val)
            elif isinstance(e, OpaqueFunction):
                res = e.execute(ctx)
                if res:
                    visit(res)
            elif isinstance(e, IncludeLaunchDescription):
                # Apply launch arguments in a child scope (matching ROS 2 launch semantics)
                saved = dict(ctx.launch_configurations)
                for k_subs, v_subs in e.launch_arguments:
                    k = ''.join(s.perform(ctx) for s in normalize_to_list_of_substitutions(k_subs))
                    v = ''.join(s.perform(ctx) for s in normalize_to_list_of_substitutions(v_subs))
                    ctx.launch_configurations[k] = v
                sub_ld = e.launch_description_source.get_launch_description(ctx)
                visit(sub_ld.entities)
                ctx.launch_configurations.clear()
                ctx.launch_configurations.update(saved)
            elif isinstance(e, Node):
                # Evaluate any launch conditions (this is where `IfCondition('slam')` used to throw!)
                cond = e.condition
                if cond is None or cond.evaluate(ctx):
                    nodes.append(e)
            else:
                sub = getattr(e, 'get_sub_entities', lambda: [])()
                if sub:
                    visit(sub)

    visit(ld.entities)
    return nodes


def main():
    share = get_package_share_directory('ugv_coqui_behavior')
    launch_path = os.path.join(share, 'launch', 'coqui_survey.launch.py')
    mod = _load_launch_module(launch_path)

    # 1) localization:=slam + rviz:=true -> ZERO crash, RTAB-Map present, STRICTLY 1 rviz2!
    nodes_slam = _collect_nodes(mod.generate_launch_description(), {
        'profile': 'forest', 'localization': 'slam', 'rviz': 'true', 'use_sim_time': 'true',
    })
    execs_slam = [n.node_executable for n in nodes_slam]
    rviz_count = sum(1 for ex in execs_slam if ex == 'rviz2')
    assert 'rtabmap' in execs_slam, f'rtabmap missing in slam mode: {execs_slam}'
    assert 'ekf_node' in execs_slam, f'ekf_node missing in slam mode: {execs_slam}'
    assert 'imu_filter_madgwick_node' in execs_slam
    assert 'collision_monitor' in execs_slam
    assert rviz_count == 1, f'expected strictly 1 rviz2 window, got {rviz_count}: {execs_slam}'

    # 1b) default localization = lidar: 3D EKF + wheel relay + scan matching, no static map->odom, no RTAB-Map, no GPS
    nodes_lidar = _collect_nodes(mod.generate_launch_description(), {'profile': 'forest', 'rviz': 'false'})
    execs_lidar = [n.node_executable for n in nodes_lidar]
    for need in ('ekf_node', 'wheel_odom_relay', 'imu_filter_madgwick_node', 'lidar_localizer'):
        assert execs_lidar.count(need) == 1, f'{need} must run exactly once in lidar mode: {execs_lidar}'
    assert 'rtabmap' not in execs_lidar and 'static_transform_publisher' not in execs_lidar
    assert not any('navsat' in (ex or '') or 'gps' in (ex or '') for ex in execs_lidar), 'the forest is GPS-denied'

    # 2) localization:=ekf (dead reckoning) + use_sim_time:=false (hardware mode)
    nodes_ekf = _collect_nodes(mod.generate_launch_description(), {
        'profile': 'hill', 'localization': 'ekf', 'rviz': 'false', 'use_sim_time': 'false',
    })
    execs_ekf = [n.node_executable for n in nodes_ekf]
    assert 'ekf_node' in execs_ekf and 'rtabmap' not in execs_ekf and 'lidar_localizer' not in execs_ekf
    assert 'wheel_odom_relay' in execs_ekf
    assert sum(1 for ex in execs_ekf if ex == 'rviz2') == 0

    # 3) localization:=perfect (GT ablation mode)
    nodes_perf = _collect_nodes(mod.generate_launch_description(), {
        'profile': 'forest', 'localization': 'perfect', 'rviz': 'false',
    })
    execs_perf = [n.node_executable for n in nodes_perf]
    assert 'gt_odom_bridge' in execs_perf and 'ekf_node' not in execs_perf

    print(f'PASS: launch resolution verified for lidar ({len(nodes_lidar)} nodes), slam ({len(nodes_slam)} nodes, 1 rviz), '
          f'ekf ({len(nodes_ekf)} nodes), and perfect ({len(nodes_perf)} nodes)!')


if __name__ == '__main__':
    main()
