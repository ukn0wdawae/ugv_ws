#!/usr/bin/env python3
"""Make the forest worlds physically traversable for the small 6-wheel UGV.

Why this exists
---------------
`forest*.world` was ported from a Clearpath Husky setup (33 cm wheels, ~13 cm
clearance).  Our `ugv_pro_audio` / `ugv_rover` have 8 cm wheels (r = 0.04 m)
and 6 cm chassis clearance.  Two artefacts of the world make every off-path
manoeuvre fail:

1. **Path curb.**  `camino.dae` (the path) lies in a 1.5 m trench cut into
   `terreno_dataset.dae`.  The two meshes intersect at a sharp V and the
   terrain lip ends up 5-9 cm *above* the path surface -> a near-vertical
   curb taller than the wheel radius along both path edges.
2. **Rigid vegetation.**  268 `altaniv` / `altaniv_d` tall-grass clumps have
   full trimesh `<collision>` 20-60 cm tall.  ODE treats them as rock.

What it does
------------
``terrain``  Re-generates the terrain mesh with a *feathered* path edge:
             * terrain vertices hidden under the path ("deep") are lifted to
               just below the path surface (no ditch can open between the two
               meshes),
             * the first ring of terrain vertices around the path ("lip") is
               put just below the path surface (kills the curb),
             * further out the terrain is clamped to a ramp of at most
               ``--ramp-deg`` from the path level, blended out to zero
               influence at ``--blend-end`` metres.
             The result is written as a new Gazebo model
             (``models/forest_terrain_traversable``); the original model in
             gazebo_models is not touched.
``worlds``   Patches forest worlds: terrain mesh URI -> new model, and
             ``collide_without_contact=1`` on every altaniv* collision so the
             robot can push through grass while ray sensors still see it.
``check``    Prints wheel-scale geometry statistics (before/after).

All three operations are idempotent.

Usage (from the workspace root)::

    python3 src/ugv_main/ugv_gazebo/scripts/forest_traversability/make_forest_traversable.py all \
        --gazebo-models /path/to/ugv_ws/gazebo_models
"""

import argparse
import heapq
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, '..', '..'))
MODEL_NAME = 'forest_terrain_traversable'
MESH_NAME = 'terreno_traversable.dae'
NEW_URI = f'model://{MODEL_NAME}/meshes/{MESH_NAME}'
OLD_URI = 'model://terrain/terreno_dataset.dae'
DEFAULT_WORLDS = ['forest.world', 'legacy_code/forest_tagged.world', 'legacy_code/forest_coqui.world']

# Geometry of the meshes as instantiated by the forest worlds (state block):
#   terrain link pose z = 3.0,  mesh scale 50 25 3
#   camino  link pose z = 2.96, mesh scale 50 25 4
XY_SCALE = np.array([50.0, 25.0])


# --------------------------------------------------------------------------- DAE io
class Dae:
    """Minimal text-preserving COLLADA editor for single-mesh Blender exports."""

    def __init__(self, path):
        self.path = path
        with open(path, 'r', encoding='utf-8') as f:
            self.text = f.read()
        self.positions = self._array('positions').reshape(-1, 3)
        idx = re.search(r'<triangles[^>]*>(.*?)</triangles>', self.text, re.S)
        tri_block = idx.group(1)
        inputs = re.findall(r'<input semantic="(\w+)"[^>]*offset="(\d+)"', tri_block)
        stride = max(int(o) for _, o in inputs) + 1
        voff = int([o for s, o in inputs if s == 'VERTEX'][0])
        p = np.array(re.search(r'<p>(.*?)</p>', tri_block, re.S).group(1).split(), dtype=np.int64)
        self.tris = p.reshape(-1, stride)[:, voff].reshape(-1, 3)
        self.normal_offset = int([o for s, o in inputs if s == 'NORMAL'][0]) if any(
            s == 'NORMAL' for s, _ in inputs) else None
        self.normal_idx = (p.reshape(-1, stride)[:, self.normal_offset].reshape(-1, 3)
                           if self.normal_offset is not None else None)

    def _array_match(self, kind):
        m = re.search(r'(<float_array id="[^"]*-%s-array" count="(\d+)">)(.*?)(</float_array>)' % kind,
                      self.text, re.S)
        if not m:
            raise RuntimeError(f'{self.path}: no {kind} float_array')
        return m

    def _array(self, kind):
        return np.array(self._array_match(kind).group(3).split(), dtype=np.float64)

    def _set_array(self, kind, values, fmt):
        m = self._array_match(kind)
        body = ' '.join(fmt % v for v in values.ravel())
        self.text = self.text[:m.start(3)] + body + self.text[m.end(3):]

    def recompute_normals(self, P):
        """Area-weighted vertex normals (Blender exports one normal per vertex)."""
        if self.normal_idx is None:
            return
        n_count = self._array('normals').size // 3
        if n_count != len(P) or not np.array_equal(self.normal_idx, self.tris):
            print('  normals are not per-vertex; leaving them unchanged')
            return
        a, b, c = P[self.tris[:, 0]], P[self.tris[:, 1]], P[self.tris[:, 2]]
        fn = np.cross(b - a, c - a)
        vn = np.zeros_like(P)
        for k in range(3):
            np.add.at(vn, self.tris[:, k], fn)
        vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12
        self._set_array('normals', vn, '%.6g')

    def save(self, path, P):
        self._set_array('positions', P, '%.7g')
        self.recompute_normals(P)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.text)


def find_model_file(rel, roots):
    for r in roots:
        p = os.path.join(r, rel)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f'{rel} not found in {roots} (pass --gazebo-models)')


def model_roots(args):
    roots = []
    if args.gazebo_models:
        roots.append(args.gazebo_models)
    roots += [p for p in os.environ.get('GAZEBO_MODEL_PATH', '').split(':') if p]
    roots += [os.path.abspath(os.path.join(PKG, '..', '..', '..', 'gazebo_models')),
              os.path.expanduser('~/.gazebo/models')]
    return roots


# --------------------------------------------------------------------------- algorithm
def edges_from_tris(tris):
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    e = np.sort(e, axis=1)
    return np.unique(e, axis=0)


def feather(xy, zt, zc, tris, deep_margin=0.03, fill_depth=0.04, lip_drop=0.01,
            ramp_deg=10.0, blend_start=1.0, blend_end=1.6):
    """Return modified terrain heights (world z) - see module docstring."""
    n = len(zt)
    E = edges_from_tris(tris)
    w = np.linalg.norm(xy[E[:, 0]] - xy[E[:, 1]], axis=1)
    nbrs = [[] for _ in range(n)]
    for (a, b), d in zip(E, w):
        nbrs[a].append((b, d))
        nbrs[b].append((a, d))

    deep = zt < zc - deep_margin
    out = zt.copy()
    out[deep] = np.maximum(zt[deep], zc[deep] - fill_depth)

    # lip vertices: not deep, adjacent to deep; reference = mean path height of deep nbrs
    ref = np.full(n, np.nan)
    lip = np.zeros(n, bool)
    for v in np.where(~deep)[0]:
        hs = [zc[u] for u, _ in nbrs[v] if deep[u]]
        if hs:
            lip[v] = True
            ref[v] = float(np.mean(hs))

    # multi-source Dijkstra from the lip ring over non-deep vertices
    dist = np.full(n, np.inf)
    src_ref = np.full(n, np.nan)
    heap = []
    for v in np.where(lip)[0]:
        dist[v] = 0.0
        src_ref[v] = ref[v]
        heap.append((0.0, int(v)))
    heapq.heapify(heap)
    while heap:
        d, v = heapq.heappop(heap)
        if d > dist[v] or d > blend_end:
            continue
        for u, wd in nbrs[v]:
            if deep[u]:
                continue
            nd = d + wd
            if nd < dist[u]:
                dist[u] = nd
                src_ref[u] = src_ref[v]
                heapq.heappush(heap, (nd, u))

    tan = np.tan(np.radians(ramp_deg))
    m = np.isfinite(dist) & ~deep
    target = src_ref[m] - lip_drop
    band = tan * dist[m]
    clipped = np.clip(zt[m], target - band, target + band)
    wgt = np.clip((blend_end - dist[m]) / max(blend_end - blend_start, 1e-6), 0.0, 1.0)
    out[m] = zt[m] + wgt * (clipped - zt[m])
    info = dict(deep=int(deep.sum()), lip=int(lip.sum()), touched=int((np.abs(out - zt) > 1e-6).sum()),
                max_lower=float((zt - out).max()), max_raise=float((out - zt).max()))
    return out, info


# --------------------------------------------------------------------------- analysis
def rasterize(P, tris, res, lo, shape, samples=14):
    """Max-height raster of a triangle mesh via dense barycentric sampling."""
    H = np.full(shape, -np.inf)
    A, B, C = P[tris[:, 0]], P[tris[:, 1]], P[tris[:, 2]]
    for i in range(samples + 1):
        for j in range(samples + 1 - i):
            a, b = i / samples, j / samples
            p = A * (1 - a - b) + B * a + C * b
            ix = np.floor((p[:, 0] - lo[0]) / res).astype(int)
            iy = np.floor((p[:, 1] - lo[1]) / res).astype(int)
            ok = (ix >= 0) & (iy >= 0) & (ix < shape[0]) & (iy < shape[1])
            np.maximum.at(H, (ix[ok], iy[ok]), p[ok, 2])
    H[~np.isfinite(H)] = np.nan
    return H


def step_map(S, cells=2):
    """Max |dz| to any cell within `cells` (chebyshev) - i.e. step over cells*res."""
    st = np.zeros_like(S)
    for dx in range(-cells, cells + 1):
        for dy in range(-cells, cells + 1):
            if dx == dy == 0:
                continue
            st = np.fmax(st, np.abs(np.roll(S, (dx, dy), axis=(0, 1)) - S))
    return st


def surface_stats(zt, zc, xy, tris, res=0.05, label=''):
    PT = np.c_[xy, zt]
    PC = np.c_[xy, zc]
    lo = xy.min(0)
    shape = tuple(((xy.max(0) - lo) / res).astype(int) + 1)
    HT = rasterize(PT, tris, res, lo, shape)
    HC = rasterize(PC, tris, res, lo, shape)
    S = np.fmax(HT, HC)
    path = HC > HT + 0.002
    st = step_map(S, 2)  # step over 10 cm
    # band of +-0.6 m around the path boundary
    edge = path ^ np.roll(path, 1, 0) | path ^ np.roll(path, 1, 1)
    band = edge.copy()
    k = int(0.6 / res)
    for dx in range(-k, k + 1):
        band |= np.roll(edge, dx, 0)
    b2 = band.copy()
    for dy in range(-k, k + 1):
        b2 |= np.roll(band, dy, 1)
    band = b2
    inner = np.zeros_like(path)
    inner[3:-3, 3:-3] = True  # ignore the outer map rim
    sel_e = band & inner & np.isfinite(st)
    sel_o = ~band & ~path & inner & np.isfinite(st)
    r = dict(edge_p50=np.nanpercentile(st[sel_e], 50), edge_p90=np.nanpercentile(st[sel_e], 90),
             edge_p99=np.nanpercentile(st[sel_e], 99), edge_frac4=float((st[sel_e] > 0.04).mean()),
             edge_frac3=float((st[sel_e] > 0.03).mean()),
             open_p50=np.nanpercentile(st[sel_o], 50), open_p90=np.nanpercentile(st[sel_o], 90),
             open_frac4=float((st[sel_o] > 0.04).mean()))
    print(f'  [{label}] path-edge band (+-0.6 m): step/10cm p50 {r["edge_p50"]*100:.1f} cm, '
          f'p90 {r["edge_p90"]*100:.1f} cm, p99 {r["edge_p99"]*100:.1f} cm, '
          f'>3 cm {100*r["edge_frac3"]:.1f} %, >4 cm {100*r["edge_frac4"]:.1f} %')
    print(f'  [{label}] open ground:            step/10cm p50 {r["open_p50"]*100:.1f} cm, '
          f'p90 {r["open_p90"]*100:.1f} cm, >4 cm {100*r["open_frac4"]:.2f} %')
    return r, (S, st, path, lo, res)


# --------------------------------------------------------------------------- commands
def load_pair(args):
    roots = model_roots(args)
    tpath = args.terrain_dae or find_model_file('terrain/terreno_dataset.dae', roots)
    cpath = args.camino_dae or find_model_file('camino_dataset/camino.dae', roots)
    T, C = Dae(tpath), Dae(cpath)
    if T.positions.shape != C.positions.shape or not np.allclose(T.positions[:, :2], C.positions[:, :2], atol=1e-6):
        sys.exit('terrain and camino meshes do not share a vertex grid - algorithm assumes they do')
    xy = T.positions[:, :2] * XY_SCALE
    zt = T.positions[:, 2] * args.terrain_z_scale + args.terrain_z_offset
    zc = C.positions[:, 2] * args.camino_z_scale + args.camino_z_offset
    print(f'terrain: {tpath}\ncamino:  {cpath}\nvertices {len(xy)}, triangles {len(T.tris)}')
    return T, xy, zt, zc


def cmd_terrain(args):
    T, xy, zt, zc = load_pair(args)
    new, info = feather(xy, zt, zc, T.tris, ramp_deg=args.ramp_deg, blend_start=args.blend_start,
                        blend_end=args.blend_end)
    print('feather:', info)
    P = T.positions.copy()
    P[:, 2] = (new - args.terrain_z_offset) / args.terrain_z_scale
    out_dir = os.path.join(PKG, 'models', MODEL_NAME)
    T.save(os.path.join(out_dir, 'meshes', MESH_NAME), P)
    with open(os.path.join(out_dir, 'model.config'), 'w') as f:
        f.write(MODEL_CONFIG)
    with open(os.path.join(out_dir, 'model.sdf'), 'w') as f:
        f.write(MODEL_SDF)
    print('wrote', out_dir)
    return T, xy, zt, zc, new


def patch_world_text(txt):
    n_uri = txt.count(OLD_URI)
    txt = txt.replace(OLD_URI, NEW_URI)

    def fix_collision(m):
        block = m.group(0)
        if 'model://altaniv' not in block:
            return block
        if '<collide_without_contact>' in block:
            block = re.sub(r'<collide_without_contact>\s*\w+\s*</collide_without_contact>',
                           '<collide_without_contact>1</collide_without_contact>', block)
        elif re.search(r'<surface>\s*<contact>', block):
            block = re.sub(r'(<surface>\s*<contact>)',
                           r'\1<collide_without_contact>1</collide_without_contact>', block, count=1)
        elif '<surface>' in block:
            block = block.replace('<surface>', '<surface><contact><collide_without_contact>1'
                                  '</collide_without_contact></contact>', 1)
        else:
            block = block.replace('</collision>', '<surface><contact><collide_without_contact>1'
                                  '</collide_without_contact></contact></surface></collision>', 1)
        fix_collision.count += 1
        return block

    fix_collision.count = 0
    txt = re.sub(r'<collision\b[^>]*>.*?</collision>', fix_collision, txt, flags=re.S)
    return txt, n_uri, fix_collision.count


def cmd_worlds(args):
    for w in args.worlds or DEFAULT_WORLDS:
        path = w if os.path.isabs(w) else os.path.join(PKG, 'worlds', w)
        with open(path, 'r', encoding='utf-8', newline='') as f:
            txt = f.read()
        new, n_uri, n_col = patch_world_text(txt)
        if new != txt:
            with open(path, 'w', encoding='utf-8', newline='') as f:
                f.write(new)
        print(f'{os.path.basename(path)}: terrain uri replaced x{n_uri}, '
              f'vegetation collisions set non-contact x{n_col}{"" if new != txt else " (already patched)"}')


def cmd_check(args):
    T, xy, zt, zc = load_pair(args)
    surface_stats(zt, zc, xy, T.tris, label='original ')
    gen = os.path.join(PKG, 'models', MODEL_NAME, 'meshes', MESH_NAME)
    if os.path.isfile(gen):
        G = Dae(gen)
        zg = G.positions[:, 2] * args.terrain_z_scale + args.terrain_z_offset
        surface_stats(zg, zc, xy, T.tris, label='feathered')
        d = zg - zt
        print(f'  vertices changed: {(np.abs(d) > 1e-5).sum()}  (lowered max {-d.min()*100:.1f} cm, '
              f'raised max {d.max()*100:.1f} cm)')


MODEL_CONFIG = """<?xml version="1.0"?>
<model>
  <name>forest_terrain_traversable</name>
  <version>1.0</version>
  <sdf version="1.6">model.sdf</sdf>
  <description>
    terreno_dataset.dae with the path edge feathered (no curb between the
    camino path mesh and the terrain) so 8 cm-wheel UGVs can leave the path.
    Generated by ugv_gazebo/scripts/forest_traversability/make_forest_traversable.py.
    Materials are still taken from model://terrain.
  </description>
</model>
"""

MODEL_SDF = """<?xml version="1.0"?>
<sdf version="1.6">
  <model name="forest_terrain_traversable">
    <static>true</static>
    <link name="link">
      <pose>0 0 -2 0 0 0</pose>
      <collision name="collision">
        <geometry>
          <mesh>
            <uri>model://forest_terrain_traversable/meshes/terreno_traversable.dae</uri>
            <scale>50 25 3</scale>
          </mesh>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <mesh>
            <uri>model://forest_terrain_traversable/meshes/terreno_traversable.dae</uri>
            <scale>50 25 3</scale>
          </mesh>
        </geometry>
        <material>
          <script>
            <uri>model://terrain/materials/scripts</uri>
            <uri>model://terrain/materials/textures</uri>
            <name>vrc/terrain</name>
          </script>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=['terrain', 'worlds', 'check', 'all'])
    ap.add_argument('--gazebo-models', help='directory containing terrain/ and camino_dataset/')
    ap.add_argument('--terrain-dae')
    ap.add_argument('--camino-dae')
    ap.add_argument('--terrain-z-scale', type=float, default=3.0)
    ap.add_argument('--terrain-z-offset', type=float, default=3.0)
    ap.add_argument('--camino-z-scale', type=float, default=4.0)
    ap.add_argument('--camino-z-offset', type=float, default=2.96)
    ap.add_argument('--ramp-deg', type=float, default=10.0)
    ap.add_argument('--blend-start', type=float, default=1.0)
    ap.add_argument('--blend-end', type=float, default=1.6)
    ap.add_argument('--worlds', nargs='*')
    args = ap.parse_args()
    if args.command in ('terrain', 'all'):
        cmd_terrain(args)
    if args.command in ('worlds', 'all'):
        cmd_worlds(args)
    if args.command in ('check', 'all'):
        cmd_check(args)


if __name__ == '__main__':
    main()
