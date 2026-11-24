"""The survey record: where each Coqui was found, its photo, and how sure we are.

Without SLAM there is no map file, so the survey keeps its own record, written after every frog:

  <output_dir>/<run_id>/
      survey.yaml      frogs in the map frame, uncertainty, photo, robot pose; mission info; `unresolved`: frogs that
                       were heard but not located (rough place, why)
      frogs.geojson    the frogs as GeoJSON points, only when the operator gave a geo_ref (no GPS is used)
      photos/          annotated pan-tilt camera photos
      events.jsonl     timestamped mission log (one JSON object per line)

Positions are in the `map` frame, which starts at the trailhead pose of the profile (initial_pose). The forest is
GPS-denied: nothing here comes from a GPS. With geo_ref (one point whose lat/lon was surveyed) lat/lon is added.
`load_known()` reads a previous survey.yaml so a new run skips frogs that were already recorded.
"""
import json
import os
import time

import yaml


class SurveyStore:
    def __init__(self, output_dir, run_id, meta):
        self.dir = os.path.join(output_dir, run_id)
        self.photo_dir = os.path.join(self.dir, 'photos')
        os.makedirs(self.photo_dir, exist_ok=True)
        self.meta = dict(meta, run_id=run_id, started_wall=time.strftime('%Y-%m-%d %H:%M:%S'))
        self.frogs = []
        self.mission = {}
        self.unresolved = []
        self._write()

    # ----------------------------------------------------------------------------------- frogs
    def add_frog(self, rec):
        rec = dict(rec)
        rec.setdefault('id', f'coqui_{len(self.frogs) + 1:03d}')
        self.frogs.append(rec)
        self._write()
        return rec['id']

    def photo_path(self, frog_id, suffix=''):
        return os.path.join(self.photo_dir, f'{frog_id}{suffix}.png')

    def set_mission(self, **kw):
        self.mission.update(kw)
        self._write()

    def set_unresolved(self, leads):
        self.unresolved = list(leads)
        self._write()

    def log_event(self, event, **kw):
        kw.update(event=event, wall=round(time.time(), 2))
        with open(os.path.join(self.dir, 'events.jsonl'), 'a') as f:
            f.write(json.dumps(kw) + '\n')

    # ------------------------------------------------------------------------------------ files
    def _write(self):
        doc = {'survey': self.meta, 'mission': self.mission, 'frogs': self.frogs, 'unresolved': self.unresolved}
        _atomic_write(os.path.join(self.dir, 'survey.yaml'), yaml.safe_dump(doc, sort_keys=False))
        feats = []
        for f in self.frogs:
            if f.get('lat') is None:
                continue
            props = {k: v for k, v in f.items() if k not in ('lat', 'lon')}
            feats.append({'type': 'Feature',
                          'geometry': {'type': 'Point', 'coordinates': [f['lon'], f['lat']]},
                          'properties': props})
        _atomic_write(os.path.join(self.dir, 'frogs.geojson'),
                      json.dumps({'type': 'FeatureCollection', 'features': feats}, indent=1))


def load_known(path):
    """Frogs from an earlier survey.yaml (list of dicts with at least 'pos': [x, y, z])."""
    if not path or not os.path.exists(path):
        return []
    doc = yaml.safe_load(open(path)) or {}
    return [f for f in doc.get('frogs', []) if 'pos' in f]


def _atomic_write(path, text):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        f.write(text)
    os.replace(tmp, path)
