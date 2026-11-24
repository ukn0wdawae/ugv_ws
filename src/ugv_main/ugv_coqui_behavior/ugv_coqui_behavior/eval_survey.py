#!/usr/bin/env python3
"""Grade a finished survey against the simulation truth file (the answer key). Offline tool: the robot
never runs this; it is how a test run is scored afterwards.

    ros2 run ugv_coqui_behavior eval_survey <survey.yaml> [--truth forest_survey.yaml] [--match 2.0] [--json out.json]

A recorded frog matches the nearest unmatched true frog within --match metres (horizontal).
"""
import argparse
import json
import math
import os
import sys

import yaml


def default_truth(profile='forest'):
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory('ugv_audio'), 'config', 'sim_truth', f'{profile}_survey.yaml')
    except Exception:
        return ''


def resolve_truth(survey_path, profile='forest', explicit_truth=''):
    if explicit_truth:
        return explicit_truth
    # 1) Trial runner / saved run-local truth_used.yaml next to survey.yaml
    if survey_path:
        local_truth = os.path.join(os.path.dirname(os.path.abspath(survey_path)), 'truth_used.yaml')
        if os.path.isfile(local_truth):
            return local_truth
    # 2) Active/most-recent Terminal 1 randomized truth in /tmp if created before/during this survey run
    domain = os.environ.get('ROS_DOMAIN_ID', '0').strip() or '0'
    session_truth = f'/tmp/coqui_latest_truth_{profile}_d{domain}.yaml'
    if os.path.isfile(session_truth) and survey_path and os.path.isfile(survey_path):
        try:
            if os.path.getmtime(survey_path) + 2.0 >= os.path.getmtime(session_truth):
                return session_truth
        except OSError:
            pass
    return default_truth(profile)


def grade(survey, truth, match_m):
    found = survey.get('frogs', [])
    true = truth['frogs']
    pairs = sorted(((math.hypot(f['pos'][0] - t['pos'][0], f['pos'][1] - t['pos'][1]), i, j)
                    for i, f in enumerate(found) for j, t in enumerate(true)))
    used_f, used_t, matches = set(), set(), {}
    for d, i, j in pairs:
        if d > match_m or i in used_f or j in used_t:
            continue
        used_f.add(i)
        used_t.add(j)
        matches[j] = (i, d)
    rows = []
    for j, t in enumerate(true):
        tz = t['pos'][2] + 0.05
        if j in matches:
            i, d = matches[j]
            f = found[i]
            rows.append({'true_id': t['id'], 'habitat': t.get('habitat', ''), 'found_as': f.get('id'),
                         'err_xy_m': round(d, 2), 'err_z_m': round(f['pos'][2] - tz, 2),
                         'visual': f.get('visually_confirmed'), 'vantage': f.get('vantage'),
                         'height_class': f.get('height_class'), 'mission_time_s': f.get('mission_time_s'),
                         'photo': f.get('photo')})
        else:
            near = min((math.hypot(f['pos'][0] - t['pos'][0], f['pos'][1] - t['pos'][1]) for f in found),
                       default=None)
            rows.append({'true_id': t['id'], 'habitat': t.get('habitat', ''), 'found_as': None,
                         'nearest_record_m': None if near is None else round(near, 2)})
    false_pos = [{'id': f.get('id'), 'pos': f['pos'],
                  'nearest_true_m': round(min(math.hypot(f['pos'][0] - t['pos'][0], f['pos'][1] - t['pos'][1])
                                              for t in true), 2)}
                 for i, f in enumerate(found) if i not in used_f]
    n = len(matches)
    errs = [r['err_xy_m'] for r in rows if r.get('found_as')]
    return {
        'frogs_true': len(true), 'frogs_recorded': len(found), 'matched': n,
        'recall': round(n / len(true), 3) if true else None,
        'precision': round(n / len(found), 3) if found else None,
        'mean_err_xy_m': round(sum(errs) / len(errs), 2) if errs else None,
        'max_err_xy_m': max(errs) if errs else None,
        'visually_confirmed': sum(1 for r in rows if r.get('visual')),
        'unmatched_records': false_pos,
        'per_frog': rows,
        'mission': survey.get('mission', {}),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('survey')
    ap.add_argument('--profile', default='forest', help='world profile (forest | hill) for default truth file')
    ap.add_argument('--truth', default='')
    ap.add_argument('--match', type=float, default=2.0)
    ap.add_argument('--json', default='')
    a = ap.parse_args(argv)
    truth_path = resolve_truth(a.survey, a.profile, a.truth)
    if not truth_path:
        sys.exit('pass --truth')
    print(f'truth file: {truth_path}')
    res = grade(yaml.safe_load(open(a.survey)), yaml.safe_load(open(truth_path)), a.match)
    print(f"recorded {res['frogs_recorded']}, true {res['frogs_true']}, matched {res['matched']} "
          f"(recall {res['recall']}, precision {res['precision']}), mean xy error {res['mean_err_xy_m']} m, "
          f"photos with frog in view {res['visually_confirmed']}")
    for r in res['per_frog']:
        if r.get('found_as'):
            print(f"  {r['true_id']:9s} FOUND as {r['found_as']}: xy err {r['err_xy_m']:.2f} m, z err {r['err_z_m']:+.2f} m,"
                  f" in photo={r['visual']}, {r['height_class']}, vantage {r['vantage']}, t={r['mission_time_s']} s"
                  f"  [{r['habitat']}]")
        else:
            print(f"  {r['true_id']:9s} missed (nearest record {r['nearest_record_m']} m)  [{r['habitat']}]")
    for f in res['unmatched_records']:
        print(f"  extra record {f['id']} at {f['pos']} (nearest true frog {f['nearest_true_m']} m)")
    print('mission:', res['mission'])
    if a.json:
        json.dump(res, open(a.json, 'w'), indent=1)
    return res


if __name__ == '__main__':
    main()
