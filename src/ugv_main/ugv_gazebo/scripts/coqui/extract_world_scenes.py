#!/usr/bin/env python3
"""Extract compact 3D terrain elevation grids and real vegetation/tree/bush candidate catalogs
from `forest_coqui_survey.world` and `hill_coqui_survey.world` into `ugv_audio/config/sim_truth/world_scenes.npz`.

Why this exists:
  In `forest_coqui_survey.world`, the pine and bush `.obj` meshes (`pine_1..6`) have 10-37 m vertex
  offsets from their `<model><pose>` origins and +90 deg X rotations in `<link><pose>` / `<state>`.
  Transforming the actual COLLADA/OBJ/STL collision & visual meshes into Gazebo world coordinates
  gives millimeter-accurate ground elevation `ground_z(x, y)`, real interior bush coordinates,
  real tree-top / fallen-trunk-top surfaces, real standing-pine bark perches, tall-grass clumps,
  and obstacle keep-out footprints.

Run from workspace root whenever a base world changes:
    python3 src/ugv_main/ugv_gazebo/scripts/coqui/extract_world_scenes.py
"""
import argparse
import math
import os
import struct
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
UGV_GAZEBO = os.path.abspath(os.path.join(HERE, '..', '..'))
SRC_MAIN = os.path.abspath(os.path.join(UGV_GAZEBO, '..'))
WS_ROOT = os.path.abspath(os.path.join(SRC_MAIN, '..', '..'))
DEFAULT_OUT = os.path.join(SRC_MAIN, 'ugv_audio', 'config', 'sim_truth', 'world_scenes.npz')


def default_model_roots(extra_roots=()):
    roots = list(extra_roots) + [
        os.path.join(UGV_GAZEBO, 'models'),
        os.path.join(WS_ROOT, 'gazebo_models'),
        '/root/ugv_ws/gazebo_models',
        '/home/dark-tiger-ubun/projects/ugv_ws/gazebo_models',
        os.path.expanduser('~/.gazebo/models'),
    ]
    return [r for r in roots if r and os.path.isdir(r)]


def resolve_uri(uri, model_roots):
    if not uri or not uri.startswith('model://'):
        return None
    rel = uri[len('model://'):]
    for r in model_roots:
        p = os.path.join(r, rel)
        if os.path.isfile(p):
            return p
    return None


def pose_to_mat(pose_str):
    if not pose_str:
        return np.eye(4, dtype=np.float64)
    vals = [float(x) for x in pose_str.split()]
    while len(vals) < 6:
        vals.append(0.0)
    x, y, z, r, p, yw = vals[:6]
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(yw), math.sin(yw)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ]
    T[:3, 3] = [x, y, z]
    return T


def parse_obj(path):
    verts, faces = [], []
    with open(path, 'r', errors='ignore') as f:
        for ln in f:
            if ln.startswith('v '):
                p = ln.split()
                verts.append([float(p[1]), float(p[2]), float(p[3])])
            elif ln.startswith('f '):
                idx = [int(tok.split('/')[0]) - 1 for tok in ln.split()[1:]]
                for k in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[k], idx[k + 1]])
    v = np.asarray(verts, dtype=np.float64)
    fc = np.asarray(faces, dtype=np.int64) if faces else np.zeros((0, 3), dtype=np.int64)
    return v, fc


def parse_stl(path):
    with open(path, 'rb') as f:
        f.read(80)
        ntri = struct.unpack('<I', f.read(4))[0]
        data = np.frombuffer(f.read(ntri * 50), dtype=np.uint8).reshape(ntri, 50)
        verts = np.frombuffer(data[:, 12:48].tobytes(), dtype='<f4').reshape(ntri * 3, 3).astype(np.float64)
        faces = np.arange(ntri * 3, dtype=np.int64).reshape(ntri, 3)
        return verts, faces


def parse_dae(path):
    tree = ET.parse(path)
    root = tree.getroot()
    ns = {'c': 'http://www.collada.org/2005/11/COLLADASchema'}
    up = root.findtext('.//c:asset/c:up_axis', default='Z_UP', namespaces=ns)
    unit_el = root.find('.//c:asset/c:unit', namespaces=ns)
    meter = float(unit_el.attrib.get('meter', '1.0')) if unit_el is not None else 1.0

    geoms = {}
    for g in root.findall('.//c:geometry', namespaces=ns):
        gid = '#' + g.attrib.get('id', '')
        mesh = g.find('c:mesh', namespaces=ns)
        if mesh is None:
            continue
        sources = {}
        for s in mesh.findall('c:source', namespaces=ns):
            fa = s.find('c:float_array', namespaces=ns)
            if fa is not None and fa.text:
                sources['#' + s.attrib.get('id', '')] = np.fromstring(fa.text, sep=' ')
        v_el = mesh.find('c:vertices', namespaces=ns)
        pos_src = None
        if v_el is not None:
            for inp in v_el.findall('c:input', namespaces=ns):
                if inp.attrib.get('semantic') == 'POSITION':
                    pos_src = sources.get(inp.attrib.get('source'))
        if pos_src is None:
            continue
        verts = pos_src.reshape(-1, 3)
        tris_list = []
        for t_el in list(mesh.findall('c:triangles', namespaces=ns)) + list(mesh.findall('c:polylist', namespaces=ns)):
            inputs = t_el.findall('c:input', namespaces=ns)
            stride = max(int(inp.attrib.get('offset', '0')) for inp in inputs) + 1
            v_off = next((int(inp.attrib.get('offset', '0'))
                          for inp in inputs if inp.attrib.get('semantic') == 'VERTEX'), 0)
            p_el = t_el.find('c:p', namespaces=ns)
            if p_el is None or not p_el.text:
                continue
            idx = np.fromstring(p_el.text, sep=' ', dtype=np.int64)[v_off::stride]
            vc_el = t_el.find('c:vcount', namespaces=ns)
            if vc_el is not None and vc_el.text:
                pos = 0
                for c in np.fromstring(vc_el.text, sep=' ', dtype=np.int64):
                    poly = idx[pos:pos + c]
                    for k in range(1, c - 1):
                        tris_list.append([poly[0], poly[k], poly[k + 1]])
                    pos += c
            else:
                tris_list.append(idx.reshape(-1, 3))
        faces = np.vstack(tris_list) if tris_list else np.zeros((0, 3), dtype=np.int64)
        geoms[gid] = (verts, faces)

    out_v, out_f, off = [], [], 0

    def walk(node, T_parent):
        nonlocal off
        T = T_parent.copy()
        mat_el = node.find('c:matrix', namespaces=ns)
        if mat_el is not None and mat_el.text:
            T = T @ np.fromstring(mat_el.text, sep=' ').reshape(4, 4)
        for ig in node.findall('c:instance_geometry', namespaces=ns):
            url = ig.attrib.get('url', '')
            if url in geoms:
                v, f = geoms[url]
                vt = (T @ np.hstack([v, np.ones((len(v), 1))]).T).T[:, :3] * meter
                if up == 'Y_UP':
                    vt = np.stack([vt[:, 0], -vt[:, 2], vt[:, 1]], axis=1)
                out_v.append(vt)
                out_f.append(f + off)
                off += len(vt)
        for ch in node.findall('c:node', namespaces=ns):
            walk(ch, T)

    vs = root.find('.//c:visual_scene', namespaces=ns)
    if vs is not None:
        for n in vs.findall('c:node', namespaces=ns):
            walk(n, np.eye(4))
    if not out_v:
        for _, (v, f) in geoms.items():
            vt = v * meter
            if up == 'Y_UP':
                vt = np.stack([vt[:, 0], -vt[:, 2], vt[:, 1]], axis=1)
            out_v.append(vt)
            out_f.append(f + off)
            off += len(vt)
    return np.vstack(out_v), np.vstack(out_f)


def load_mesh(uri, model_roots, cache):
    if uri not in cache:
        p = resolve_uri(uri, model_roots)
        if p is None:
            raise FileNotFoundError(f'Cannot resolve mesh URI {uri} in {model_roots}')
        ext = os.path.splitext(p)[1].lower()
        if ext == '.obj':
            cache[uri] = parse_obj(p)
        elif ext == '.stl':
            cache[uri] = parse_stl(p)
        elif ext == '.dae':
            cache[uri] = parse_dae(p)
        else:
            raise ValueError(f'Unsupported mesh format: {p}')
    return cache[uri]


def rasterize_max_z(tris_xyz, xs, ys):
    """Rasterize highest surface Z of `tris_xyz` (N, 3, 3) onto regular grid `(xs, ys)` -> (len(ys), len(xs))."""
    dx = float(xs[1] - xs[0])
    dy = float(ys[1] - ys[0])
    x0_g, y0_g = float(xs[0]), float(ys[0])
    nx, ny = len(xs), len(ys)
    grid = np.full((ny, nx), -np.inf, dtype=np.float64)

    ax, ay, az = tris_xyz[:, 0, 0], tris_xyz[:, 0, 1], tris_xyz[:, 0, 2]
    bx, by, bz = tris_xyz[:, 1, 0], tris_xyz[:, 1, 1], tris_xyz[:, 1, 2]
    cx, cy, cz = tris_xyz[:, 2, 0], tris_xyz[:, 2, 1], tris_xyz[:, 2, 2]

    tmin_x = np.minimum(np.minimum(ax, bx), cx)
    tmax_x = np.maximum(np.maximum(ax, bx), cx)
    tmin_y = np.minimum(np.minimum(ay, by), cy)
    tmax_y = np.maximum(np.maximum(ay, by), cy)

    ix0 = np.maximum(0, np.floor((tmin_x - x0_g) / dx - 1e-6).astype(np.int64))
    ix1 = np.minimum(nx - 1, np.ceil((tmax_x - x0_g) / dx + 1e-6).astype(np.int64))
    iy0 = np.maximum(0, np.floor((tmin_y - y0_g) / dy - 1e-6).astype(np.int64))
    iy1 = np.minimum(ny - 1, np.ceil((tmax_y - y0_g) / dy + 1e-6).astype(np.int64))

    valid = np.where((ix0 <= ix1) & (iy0 <= iy1))[0]
    for i in valid:
        det = (by[i] - cy[i]) * (ax[i] - cx[i]) + (cx[i] - bx[i]) * (ay[i] - cy[i])
        if abs(det) < 1e-12:
            continue
        sub_x = xs[ix0[i]:ix1[i] + 1]
        sub_y = ys[iy0[i]:iy1[i] + 1]
        gx, gy = np.meshgrid(sub_x, sub_y)
        u = ((by[i] - cy[i]) * (gx - cx[i]) + (cx[i] - bx[i]) * (gy - cy[i])) / det
        v = ((cy[i] - ay[i]) * (gx - cx[i]) + (ax[i] - cx[i]) * (gy - cy[i])) / det
        w = 1.0 - u - v
        inside = (u >= -1e-5) & (v >= -1e-5) & (w >= -1e-5)
        if np.any(inside):
            tz = u * az[i] + v * bz[i] + w * cz[i]
            cur = grid[iy0[i]:iy1[i] + 1, ix0[i]:ix1[i] + 1]
            cur[inside] = np.maximum(cur[inside], tz[inside])
    # Fill any tiny boundary NaNs/-inf with nearest valid cell
    if np.any(~np.isfinite(grid)):
        vy, vx = np.where(np.isfinite(grid))
        by_i, bx_i = np.where(~np.isfinite(grid))
        for r, c in zip(by_i, bx_i):
            d2 = (vy - r) ** 2 + (vx - c) ** 2
            k = int(np.argmin(d2))
            grid[r, c] = grid[vy[k], vx[k]]
    return grid.astype(np.float32)


def interp_grid(grid, x0, y0, dx, dy, x, y):
    ny, nx = grid.shape
    fx = np.clip((x - x0) / dx, 0.0, nx - 1.0001)
    fy = np.clip((y - y0) / dy, 0.0, ny - 1.0001)
    ix = int(math.floor(fx))
    iy = int(math.floor(fy))
    tx = fx - ix
    ty = fy - iy
    z00 = float(grid[iy, ix])
    z10 = float(grid[iy, ix + 1])
    z01 = float(grid[iy + 1, ix])
    z11 = float(grid[iy + 1, ix + 1])
    return (1 - ty) * ((1 - tx) * z00 + tx * z10) + ty * ((1 - tx) * z01 + tx * z11)


def parse_world_state(world_path):
    tree = ET.parse(world_path)
    world = tree.getroot().find('world')
    state = world.find('state')
    state_m, state_l = {}, {}
    if state is not None:
        for sm in state.findall('model'):
            mn = sm.attrib.get('name', '')
            if sm.findtext('pose'):
                state_m[mn] = pose_to_mat(sm.findtext('pose'))
            for sl in sm.findall('link'):
                ln = sl.attrib.get('name', '')
                if sl.findtext('pose'):
                    state_l[(mn, ln)] = pose_to_mat(sl.findtext('pose'))
    return world, state_m, state_l


def transform_mesh(verts, T_link_world, elem):
    sc = [float(s) for s in (elem.findtext('geometry/mesh/scale') or '1 1 1').split()]
    Tw = T_link_world @ pose_to_mat(elem.findtext('pose')) @ np.diag([sc[0], sc[1], sc[2], 1.0])
    return (Tw @ np.hstack([verts, np.ones((len(verts), 1))]).T).T[:, :3]


def extract_forest(world_path, model_roots, cache):
    area = [18.0, 50.0, -15.0, 15.0]
    world, state_m, state_l = parse_world_state(world_path)

    # 1. Exact terrain height grid (`terrain` + `camino` collision meshes)
    t_tris = []
    for mn in ('terrain', 'camino'):
        m = world.find(f"model[@name='{mn}']")
        Tm = state_m.get(mn, pose_to_mat(m.findtext('pose')))
        l = m.find('link')
        Tl = state_l.get((mn, l.attrib['name']), Tm @ pose_to_mat(l.findtext('pose')))
        col = l.find('collision')
        uri = col.findtext('geometry/mesh/uri')
        v, f = load_mesh(uri, model_roots, cache)
        vw = transform_mesh(v, Tl, col)
        t_tris.append(vw[f])
    xs = np.arange(16.0, 51.0 + 1e-6, 0.25)
    ys = np.arange(-16.5, 16.5 + 1e-6, 0.25)
    elev = rasterize_max_z(np.vstack(t_tris), xs, ys)
    gz = lambda x, y: interp_grid(elev, xs[0], ys[0], 0.25, 0.25, x, y)  # noqa: E731

    # 2. Extract standing pine trunks, bushes (`bush_pine*`), fallen trunk (`trunk_clone_clone`), and grass
    trunks = []         # (cx, cy, r_bark, group_idx)
    bush_Circles = []   # (cx, cy, r_obs, group_idx)
    bush_cands = []     # [x, y, z, bush_group_id]
    tree_top_cands = [] # [x, y, z, yaw, tree_group_id, subtype_code (0=crown_top, 1=fallen_trunk_top)]
    tree_perch_cands = []  # [x, y, z, yaw, tree_group_id]
    grass_cands = []    # [x, y, z]

    xmin, xmax, ymin, ymax = area
    margin = 1.8

    # First pass: standing pine trunks + bushes + fallen trunk obstacles
    pine_models = []
    bush_models = []
    fallen_models = []
    grass_models = []

    for m in world.findall('model'):
        mn = m.attrib.get('name', '')
        if mn in ('terrain', 'camino'):
            continue
        Tm = state_m.get(mn, pose_to_mat(m.findtext('pose')))
        for l in m.findall('link'):
            ln = l.attrib.get('name', '')
            Tl = state_l.get((mn, ln), Tm @ pose_to_mat(l.findtext('pose')))
            for v_el in l.findall('visual'):
                uri = v_el.findtext('geometry/mesh/uri') or ''
                if not uri:
                    continue
                verts, faces = load_mesh(uri, model_roots, cache)
                vw = transform_mesh(verts, Tl, v_el)
                if uri.endswith('trunk.obj') and 'pine_6' not in uri:
                    pine_models.append((mn, uri, vw, faces))
                elif mn.startswith('bush_pine') and 'pine_6/crown.obj' in uri:
                    bush_models.append((mn, uri, vw, faces))
                elif mn.startswith('trunk') and 'pine_6/trunk.obj' in uri:
                    fallen_models.append((mn, uri, vw, faces))
                elif 'altaniv' in uri:
                    grass_models.append((mn, uri, vw))

    # Standing pine trunks near/in survey_area
    for tid, (mn, uri, vw, _faces) in enumerate(pine_models):
        # Measure bark slice at z in [3.3, 4.8] (~0.5..2.0 m above terrain)
        sl = vw[(vw[:, 2] >= 3.3) & (vw[:, 2] <= 4.8)]
        if len(sl) < 10:
            continue
        cx, cy = float(np.median(sl[:, 0])), float(np.median(sl[:, 1]))
        if not (xmin - 2.0 <= cx <= xmax + 2.0 and ymin - 2.0 <= cy <= ymax + 2.0):
            continue
        r_bark = float(np.percentile(np.hypot(sl[:, 0] - cx, sl[:, 1] - cy), 85))
        trunks.append((cx, cy, max(r_bark, 0.20), tid))

    # Bush footprints & bush interior + young-pine crown-top candidates
    for bid, (mn, uri, vw, faces) in enumerate(bush_models):
        above = vw[vw[:, 2] >= 2.7]
        if len(above) < 20:
            continue
        bcx, bcy = float(np.median(above[:, 0])), float(np.median(above[:, 1]))
        if not (xmin - 3.5 <= bcx <= xmax + 3.5 and ymin - 3.5 <= bcy <= ymax + 3.5):
            continue
        r_obs = float(np.percentile(np.hypot(above[:, 0] - bcx, above[:, 1] - bcy), 88))
        bush_Circles.append((bcx, bcy, r_obs, 100 + bid))

        # Rasterize bush upper canopy surface on 0.35 m local grid
        b_xs = np.arange(float(above[:, 0].min()), float(above[:, 0].max()) + 1e-6, 0.35)
        b_ys = np.arange(float(above[:, 1].min()), float(above[:, 1].max()) + 1e-6, 0.35)
        b_top = rasterize_max_z(vw[faces], b_xs, b_ys)
        # Keep true raycast hits only (re-mask cells outside triangle xy bounds)
        for iy, py in enumerate(b_ys):
            if not (ymin + margin <= py <= ymax - margin):
                continue
            for ix, px in enumerate(b_xs):
                if not (xmin + margin <= px <= xmax - margin):
                    continue
                g_z = gz(px, py)
                top_z = float(b_top[iy, ix])
                # Verify real nearby bush vertices at (px, py) so nearest-fill doesn't invent canopy
                near_v = above[np.hypot(above[:, 0] - px, above[:, 1] - py) <= 0.45]
                if len(near_v) == 0:
                    continue
                local_top = float(near_v[:, 2].max())
                dh = local_top - g_z
                dist_ctr = math.hypot(px - bcx, py - bcy)
                if any(math.hypot(px - tcx, py - tcy) < tr + 0.55 for tcx, tcy, tr, _ in trunks):
                    continue
                # 1) Inside bush: dense foliage overhead (dh >= 0.75 m), 0.7..2.1 m from center
                #    (~0.8..2.0 m inside the bush perimeter so frog is genuinely inside the bush!)
                if 0.75 <= dh <= 2.35 and 0.65 <= dist_ctr <= 2.10:
                    bush_cands.append([round(px, 3), round(py, 3), round(g_z, 3), float(100 + bid)])
                # 2) On top of young pine / bush-pine crown: upper cone/apex (dh >= 1.55 m, dist_ctr <= 1.35 m)
                if dh >= 1.55 and dist_ctr <= 1.35:
                    out_yaw = math.atan2(py - bcy, px - bcx) if dist_ctr > 0.15 else 0.0
                    tree_top_cands.append([round(px, 3), round(py, 3), round(local_top + 0.01, 3),
                                           round(out_yaw, 3), float(100 + bid), 0.0])

    # Fallen giant tree trunk (`trunk_clone_clone`) obstacle + top-of-trunk surface candidates
    fallen_obs = []
    for fid, (mn, uri, vw, faces) in enumerate(fallen_models):
        above = vw[vw[:, 2] >= 2.7]
        if len(above) < 20:
            continue
        fcx, fcy = float(np.median(above[:, 0])), float(np.median(above[:, 1]))
        if not (xmin - 3.0 <= fcx <= xmax + 3.0 and ymin - 3.0 <= fcy <= ymax + 3.0):
            continue
        f_xs = np.arange(float(above[:, 0].min()), float(above[:, 0].max()) + 1e-6, 0.35)
        f_ys = np.arange(float(above[:, 1].min()), float(above[:, 1].max()) + 1e-6, 0.35)
        f_top = rasterize_max_z(vw[faces], f_xs, f_ys)
        for iy, py in enumerate(f_ys):
            for ix, px in enumerate(f_xs):
                near_v = above[np.hypot(above[:, 0] - px, above[:, 1] - py) <= 0.40]
                if len(near_v) == 0:
                    continue
                g_z = gz(max(16.0, min(51.0, px)), max(-16.5, min(16.5, py)))
                top_z = float(f_top[iy, ix])
                if top_z >= g_z + 0.25:
                    fallen_obs.append((px, py, 0.65))
                if (xmin + margin <= px <= xmax - margin and ymin + margin <= py <= ymax - margin
                        and g_z + 1.05 <= top_z <= g_z + 2.55):
                    # Sitting right on top of the fallen pine tree trunk!
                    tree_top_cands.append([round(px, 3), round(py, 3), round(top_z + 0.01, 3),
                                           round(math.atan2(-py, 48.0 - px), 3), float(200 + fid), 1.0])

    def is_clear_ground(px, py, trunk_pad=0.55, bush_pad=0.25):
        if not (xmin + margin <= px <= xmax - margin and ymin + margin <= py <= ymax - margin):
            return False
        if any(math.hypot(px - tcx, py - tcy) < tr + trunk_pad for tcx, tcy, tr, _ in trunks):
            return False
        if any(math.hypot(px - bcx, py - bcy) < br + bush_pad for bcx, bcy, br, _ in bush_Circles):
            return False
        if any(math.hypot(px - fox, py - foy) < fr + 0.35 for fox, foy, fr in fallen_obs):
            return False
        return True

    # Standing pine bark/branch perches (flush on outer bark at 1.4..2.2 m above ground, facing clear vantage)
    for tcx, tcy, tr, tid in trunks:
        if not (xmin + margin <= tcx <= xmax - margin and ymin + margin <= tcy <= ymax - margin):
            continue
        # Get exact trunk vertices for this pine
        _, _, vw, _ = pine_models[tid]
        base_gz = gz(tcx, tcy)
        for dh in (1.45, 1.85, 2.20):
            target_z = base_gz + dh
            sl = vw[(vw[:, 2] >= target_z - 0.25) & (vw[:, 2] <= target_z + 0.25)]
            if len(sl) < 5:
                continue
            scx, scy = float(np.median(sl[:, 0])), float(np.median(sl[:, 1]))
            sr = float(np.percentile(np.hypot(sl[:, 0] - scx, sl[:, 1] - scy), 90))
            for k in range(8):
                ang = 2.0 * math.pi * k / 8.0
                # Require clear robot camera vantage ~2.2 m away in direction `ang`
                vx, vy = scx + 2.2 * math.cos(ang), scy + 2.2 * math.sin(ang)
                if not is_clear_ground(vx, vy, trunk_pad=0.45, bush_pad=0.20):
                    continue
                fx = scx + (sr + 0.035) * math.cos(ang)
                fy = scy + (sr + 0.035) * math.sin(ang)
                if xmin + 1.6 <= fx <= xmax - 1.6 and ymin + 1.6 <= fy <= ymax - 1.6:
                    tree_perch_cands.append([round(fx, 3), round(fy, 3), round(target_z, 3),
                                             round(ang, 3), float(tid)])

    # Tall-grass patch candidates (`altaniv` / `altaniv_d` clumps inside survey_area, clear of trunks/bushes)
    for _mn, _uri, vw in grass_models:
        gx, gy = float(np.median(vw[:, 0])), float(np.median(vw[:, 1]))
        if is_clear_ground(gx, gy, trunk_pad=0.60, bush_pad=0.30):
            grass_cands.append([round(gx, 3), round(gy, 3), round(gz(gx, gy), 3)])

    # Deduplicate dense obstacle list for compact storage
    obs_list = [[round(tcx, 3), round(tcy, 3), round(tr, 3), 0.0] for tcx, tcy, tr, _ in trunks]
    obs_list += [[round(bcx, 3), round(bcy, 3), round(br, 3), 1.0] for bcx, bcy, br, _ in bush_Circles]
    # Subsample fallen trunk obstacle discs at ~0.7 m spacing
    kept_fo = []
    for fox, foy, fr in fallen_obs:
        if all(math.hypot(fox - kx, foy - ky) >= 0.65 for kx, ky, _ in kept_fo):
            kept_fo.append((fox, foy, fr))
            obs_list.append([round(fox, 3), round(foy, 3), round(fr, 3), 2.0])

    return {
        'forest_grid_z': elev,
        'forest_grid_meta': np.array([xs[0], ys[0], 0.25, 0.25], dtype=np.float64),
        'forest_bushes': np.asarray(bush_cands, dtype=np.float64),
        'forest_tree_tops': np.asarray(tree_top_cands, dtype=np.float64),
        'forest_tree_perches': np.asarray(tree_perch_cands, dtype=np.float64),
        'forest_grass': np.asarray(grass_cands, dtype=np.float64),
        'forest_obstacles': np.asarray(obs_list, dtype=np.float64),
    }


def extract_hill(world_path, model_roots, cache):
    area = [-39.0, 48.0, -16.5, 16.5]
    world, state_m, state_l = parse_world_state(world_path)

    t_tris = []
    for hm in world.findall('model'):
        mn = hm.attrib.get('name', '')
        if mn not in ('hill', 'hill_ground'):
            continue
        Tm = state_m.get(mn, pose_to_mat(hm.findtext('pose')))
        for hl in hm.findall('link'):
            ln = hl.attrib.get('name', '')
            Tl = state_l.get((mn, ln), Tm @ pose_to_mat(hl.findtext('pose')))
            for col in hl.findall('collision'):
                uri = col.findtext('geometry/mesh/uri')
                if not uri:
                    continue
                v, f = load_mesh(uri, model_roots, cache)
                vw = transform_mesh(v, Tl, col)
                t_tris.append(vw[f])

    xs = np.arange(-40.0, 49.0 + 1e-6, 0.50)
    ys = np.arange(-17.5, 17.5 + 1e-6, 0.50)
    elev = rasterize_max_z(np.vstack(t_tris), xs, ys)
    gz = lambda x, y: interp_grid(elev, xs[0], ys[0], 0.50, 0.50, x, y)  # noqa: E731

    xmin, xmax, ymin, ymax = area
    margin = 2.0
    bush_cands, tree_perches, grass_cands, obs_list = [], [], [], []
    tid_seq, bid_seq = 0, 100

    for hm in world.findall('model'):
        mn = hm.attrib.get('name', '')
        if mn in ('hill', 'hill_ground'):
            continue
        Tm = state_m.get(mn, pose_to_mat(hm.findtext('pose')))
        for hl in hm.findall('link'):
            ln = hl.attrib.get('name', '')
            Tl = state_l.get((mn, ln), Tm @ pose_to_mat(hl.findtext('pose')))
            for vel in hl.findall('visual'):
                uri = vel.findtext('geometry/mesh/uri') or ''
                if not uri or 'path.dae' in uri:
                    continue
                verts, _ = load_mesh(uri, model_roots, cache)
                vw = transform_mesh(verts, Tl, vel)
                cx, cy = float(np.median(vw[:, 0])), float(np.median(vw[:, 1]))
                if not (xmin - 3.0 <= cx <= xmax + 3.0 and ymin - 3.0 <= cy <= ymax + 3.0):
                    continue
                if 'roca3' in uri:
                    g_z = gz(max(xs[0], min(xs[-1], cx)), max(ys[0], min(ys[-1], cy)))
                    near = vw[(vw[:, 2] >= g_z + 0.15) & (vw[:, 2] <= g_z + 2.5)]
                    if len(near) >= 5:
                        rcx, rcy = float(np.median(near[:, 0])), float(np.median(near[:, 1]))
                        rr = float(np.percentile(np.hypot(near[:, 0] - rcx, near[:, 1] - rcy), 85))
                        if rr <= 3.0:
                            obs_list.append([round(rcx, 3), round(rcy, 3), round(max(rr, 0.5), 3), 2.0])
                elif any(k in uri for k in ('tronco', 'troncot', 'bark8')):
                    g_z = gz(max(xs[0], min(xs[-1], cx)), max(ys[0], min(ys[-1], cy)))
                    sl = vw[(vw[:, 2] >= g_z + 0.3) & (vw[:, 2] <= g_z + 2.5)]
                    if len(sl) >= 5:
                        tcx, tcy = float(np.median(sl[:, 0])), float(np.median(sl[:, 1]))
                        tr = min(float(np.percentile(np.hypot(sl[:, 0] - tcx, sl[:, 1] - tcy), 85)), 0.45)
                        obs_list.append([round(tcx, 3), round(tcy, 3), round(max(tr, 0.25), 3), 0.0])
                        if xmin + margin <= tcx <= xmax - margin and ymin + margin <= tcy <= ymax - margin:
                            for dh in (1.45, 1.85):
                                for k in range(6):
                                    ang = 2.0 * math.pi * k / 6.0
                                    fx = tcx + (max(tr, 0.22) + 0.04) * math.cos(ang)
                                    fy = tcy + (max(tr, 0.22) + 0.04) * math.sin(ang)
                                    fgz = gz(max(xs[0], min(xs[-1], fx)), max(ys[0], min(ys[-1], fy)))
                                    tree_perches.append([round(fx, 3), round(fy, 3), round(fgz + dh, 3),
                                                         round(ang, 3), float(tid_seq)])
                            tid_seq += 1
                elif 'dry_bush' in uri or 'arbusto3_dry' in uri:
                    r_b = float(np.percentile(np.hypot(vw[:, 0] - cx, vw[:, 1] - cy), 80))
                    obs_list.append([round(cx, 3), round(cy, 3), round(max(r_b, 0.6), 3), 1.0])
                    if xmin + margin <= cx <= xmax - margin and ymin + margin <= cy <= ymax - margin:
                        bush_cands.append([round(cx, 3), round(cy, 3), round(gz(cx, cy), 3), float(bid_seq)])
                        bid_seq += 1
                elif 'altaniv' in uri:
                    if xmin + margin <= cx <= xmax - margin and ymin + margin <= cy <= ymax - margin:
                        grass_cands.append([round(cx, 3), round(cy, 3), round(gz(cx, cy), 3)])

    return {
        'hill_grid_z': elev,
        'hill_grid_meta': np.array([xs[0], ys[0], 0.50, 0.50], dtype=np.float64),
        'hill_bushes': np.asarray(bush_cands, dtype=np.float64),
        'hill_tree_perches': np.asarray(tree_perches, dtype=np.float64),
        'hill_grass': np.asarray(grass_cands, dtype=np.float64),
        'hill_obstacles': np.asarray(obs_list, dtype=np.float64),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--forest-world', default=os.path.join(UGV_GAZEBO, 'worlds', 'forest_coqui_survey.world'))
    ap.add_argument('--hill-world', default=os.path.join(UGV_GAZEBO, 'worlds', 'hill_coqui_survey.world'))
    ap.add_argument('--model-root', action='append', default=[], help='additional Gazebo model root directory')
    ap.add_argument('--out', default=DEFAULT_OUT)
    a = ap.parse_args()

    roots = default_model_roots(a.model_root)
    cache = {}
    data = {}
    data.update(extract_forest(a.forest_world, roots, cache))
    data.update(extract_hill(a.hill_world, roots, cache))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, **data)
    sz_kb = os.path.getsize(a.out) / 1024.0
    print(f'Wrote {a.out} ({sz_kb:.1f} KB):')
    for k, v in sorted(data.items()):
        print(f'  {k:22s}: shape={v.shape} dtype={v.dtype}')


if __name__ == '__main__':
    main()
