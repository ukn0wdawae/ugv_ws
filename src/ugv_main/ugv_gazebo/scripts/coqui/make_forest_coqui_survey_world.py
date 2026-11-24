#!/usr/bin/env python3
"""Generate worlds/forest_coqui_survey.world = the full forest.world + the Coqui frogs from the
simulation truth file + a real GPS origin + a Gazebo GUI start view at the robot.

    python3 make_forest_coqui_survey_world.py [--truth PATH] [--lat 18.30 --lon -65.78]
                                              [--cam-back 6.0 --cam-up 3.5 --cam-pitch 0.5]

The frog positions live only in ugv_audio/config/sim_truth/forest_survey.yaml (the answer key);
this script copies them into the world so the frogs are visible to the cameras. Re-run it after
editing the truth file. The Gazebo GUI camera (<gui><camera><pose>) starts behind the robot spawn
of the truth file (spawn: [x, y, z, yaw]), looking along the robot's heading.
"""
import argparse
import math
import os
import re

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, '..', '..'))                       # ugv_gazebo
SRC = os.path.abspath(os.path.join(PKG, '..'))                               # src/ugv_main
DEFAULT_TRUTH = os.path.join(SRC, 'ugv_audio', 'config', 'sim_truth', 'forest_survey.yaml')

# Gazebo GUI start view, relative to the robot spawn of the truth file
GUI_CAM_BACK_M = 6.0    # behind the robot, along its heading
GUI_CAM_UP_M = 3.5      # above the spawn z
GUI_CAM_PITCH = 0.5     # rad; in Gazebo a positive pitch looks down


def gui_camera_pose(spawn, back, up, pitch):
    """Pose text 'x y z roll pitch yaw' of a camera `back` m behind and `up` m above the spawn [x, y, z, yaw],
    looking along the spawn heading."""
    x, y, z, yaw = (float(v) for v in spawn[:4])
    return f'{x - back * math.cos(yaw):.3f} {y - back * math.sin(yaw):.3f} {z + up:.3f} 0 {pitch:.3f} {yaw:.3f}'


def set_gui_camera_pose(txt, pose):
    """Replace the pose of the Gazebo GUI camera (<gui> <camera> <pose>) in the world text."""
    gui = re.search(r'<gui\b.*?</gui>', txt, flags=re.S)
    if gui is None:
        raise SystemExit('base world has no <gui> block (Gazebo GUI camera)')
    block, n = re.subn(r'(<camera\b.*?<pose\b[^>]*>)[^<]*(</pose>)', lambda m: m.group(1) + pose + m.group(2),
                       gui.group(0), count=1, flags=re.S)
    if n != 1:
        raise SystemExit('base world has no <gui><camera><pose> (Gazebo GUI camera start pose)')
    return txt[:gui.start()] + block + txt[gui.end():]


def set_physics_step(txt, step_size):
    """Set ODE physics step size and matching real_time_update_rate (keeps real_time_factor = 1)."""
    rate = int(round(1.0 / step_size))
    phys = re.search(r'<physics\b.*?</physics>', txt, flags=re.S)
    if phys is None:
        return txt
    block = re.sub(r'(<max_step_size>)[^<]*(</max_step_size>)', rf'\g<1>{step_size:g}\g<2>', phys.group(0), count=1)
    block = re.sub(r'(<real_time_update_rate>)[^<]*(</real_time_update_rate>)', rf'\g<1>{rate}\g<2>', block, count=1)
    return txt[:phys.start()] + block + txt[phys.end():]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--truth', default=DEFAULT_TRUTH)
    ap.add_argument('--base', default=os.path.join(PKG, 'worlds', 'forest.world'))
    ap.add_argument('--out', default=os.path.join(PKG, 'worlds', 'forest_coqui_survey.world'))
    # El Yunque National Forest, Puerto Rico (Coqui habitat); the forest origin is placed here
    ap.add_argument('--lat', type=float, default=18.3000)
    ap.add_argument('--lon', type=float, default=-65.7800)
    ap.add_argument('--elevation', type=float, default=600.0)
    ap.add_argument('--cam-back', type=float, default=GUI_CAM_BACK_M,
                    help='Gazebo GUI start view: metres behind the robot spawn, along its heading')
    ap.add_argument('--cam-up', type=float, default=GUI_CAM_UP_M,
                    help='Gazebo GUI start view: metres above the spawn z')
    ap.add_argument('--cam-pitch', type=float, default=GUI_CAM_PITCH,
                    help='Gazebo GUI start view: pitch in rad (positive looks down)')
    ap.add_argument('--physics-step', type=float, default=0.002,
                    help='ODE max_step_size in s (0.002 = 500 Hz; cuts physics CPU load in half vs 1000 Hz)')
    a = ap.parse_args()

    truth = yaml.safe_load(open(a.truth))
    txt = open(a.base).read()
    if a.physics_step > 0:
        txt = set_physics_step(txt, a.physics_step)

    sph = ('<spherical_coordinates>\n'
           '      <surface_model>EARTH_WGS84</surface_model>\n'
           f'      <latitude_deg>{a.lat}</latitude_deg>\n'
           f'      <longitude_deg>{a.lon}</longitude_deg>\n'
           f'      <elevation>{a.elevation}</elevation>\n'
           '      <heading_deg>0</heading_deg>\n'
           '    </spherical_coordinates>')
    txt, n = re.subn(r'<spherical_coordinates>.*?</spherical_coordinates>', sph, txt, count=1, flags=re.S)
    if n != 1:
        raise SystemExit('base world has no <spherical_coordinates> block')

    cam_pose = gui_camera_pose(truth['spawn'], a.cam_back, a.cam_up, a.cam_pitch)
    txt = set_gui_camera_pose(txt, cam_pose)

    frogs = []
    for fr in truth['frogs']:
        x, y, z = fr['pos']
        yaw = math.atan2(truth['spawn'][1] - y, truth['spawn'][0] - x)   # face roughly toward the start
        frogs.append(f"""
    <include>
      <name>{fr['id']}</name>
      <uri>model://coqui_frog</uri>
      <static>true</static>
      <pose>{x} {y} {z} 0 0 {yaw:.3f}</pose>
    </include>""")
    marker = '</world>'
    i = txt.rindex(marker)
    txt = txt[:i] + '    <!-- Coqui frogs (generated from ' + os.path.relpath(a.truth, SRC) + ') -->' + \
        ''.join(frogs) + '\n  ' + txt[i:]
    open(a.out, 'w').write(txt)
    print(f'wrote {a.out}: {len(frogs)} frogs, GPS origin {a.lat}, {a.lon}, GUI camera pose {cam_pose}')


if __name__ == '__main__':
    main()
