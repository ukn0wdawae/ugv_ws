"""PointCloud2 parsing tests (uses sensor_msgs if available, else a stand-in)."""
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ugv_traversability.cloud_utils import cloud_to_xyz, xyz_to_cloud_fields  # noqa: E402


def make_msg(points, extra_pad=4, height=1, nan_rows=()):
    pts = np.asarray(points, np.float32)
    n = len(pts)
    step = 12 + extra_pad
    raw = np.zeros((n, step), np.uint8)
    raw[:, :12] = pts.view(np.uint8).reshape(n, 12)
    width = n // height
    fields = [SimpleNamespace(name=k, offset=4 * i, datatype=7, count=1) for i, k in enumerate('xyz')]
    return SimpleNamespace(fields=fields, is_bigendian=False, point_step=step, row_step=step * width,
                           width=width, height=height, data=raw.tobytes())


def test_parse_with_padding_and_nans():
    pts = np.array([[1, 2, 3], [np.nan, 0, 0], [4, 5, 6], [7, 8, np.inf]], np.float32)
    out = cloud_to_xyz(make_msg(pts))
    assert np.allclose(out, [[1, 2, 3], [4, 5, 6]])


def test_parse_organised_stride():
    pts = np.arange(4 * 6 * 3, dtype=np.float32).reshape(-1, 3)
    out = cloud_to_xyz(make_msg(pts, height=4), stride=2)
    # rows 0,2 and cols 0,2,4 -> 6 points
    assert out.shape == (6, 3)
    assert np.allclose(out[0], pts[0]) and np.allclose(out[1], pts[2])


def test_pack_roundtrip():
    pts = np.random.default_rng(0).normal(size=(10, 3))
    data, step, fields = xyz_to_cloud_fields(pts, np.arange(10))
    assert step == 16 and [f[0] for f in fields] == ['x', 'y', 'z', 'intensity']
    msg = SimpleNamespace(fields=[SimpleNamespace(name=n, offset=o, datatype=d, count=c) for n, o, d, c in fields],
                          is_bigendian=False, point_step=step, row_step=step * 10, width=10, height=1, data=data)
    assert np.allclose(cloud_to_xyz(msg), pts, atol=1e-6)
