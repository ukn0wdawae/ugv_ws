"""VoxelSet tests (numpy only, no ROS; ~1 s).

    cd src/ugv_main/ugv_coqui_behavior && python3 -m pytest test/test_voxel_map.py -q
"""
import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ugv_coqui_behavior.voxel_map import OFFSET, VoxelSet, crop_range, pack_keys, quat_to_matrix, \
    transform_points, unpack_keys  # noqa: E402


def test_dedup_within_one_call():
    vs = VoxelSet(0.1)
    assert vs.add([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06], [0.09, 0.01, 0.0], [0.15, 0.0, 0.0]]) == 2
    assert len(vs) == 2


def test_dedup_across_calls():
    vs = VoxelSet(0.1)
    pts = np.array([[1.01, 2.02, 3.03], [5.05, 6.06, 7.07]])
    assert vs.add(pts) == 2
    assert vs.add(pts) == 0
    assert vs.add(pts + 0.02) == 0  # same voxels
    assert vs.add(np.vstack([pts, [[9.05, 9.05, 9.05]]])) == 1
    assert len(vs) == 3
    assert np.all(np.diff(vs.keys) > 0)  # sorted and unique


def test_negative_coordinates_use_floor():
    vs = VoxelSet(0.1)
    assert vs.add([[-0.05, 0.05, -0.15], [0.05, -0.05, 0.15]]) == 2
    # centres come out in (i, j, k) order
    np.testing.assert_allclose(vs.centers(), [[-0.05, 0.05, -0.15], [0.05, -0.05, 0.15]], atol=1e-6)
    # -0.05 and +0.05 are different voxels (floor, not truncation towards zero)
    assert vs.add([[0.05, 0.05, -0.15]]) == 1
    assert vs.add([[-0.01, 0.01, -0.11]]) == 0


def test_pack_unpack_round_trip():
    rng = np.random.default_rng(1)
    ijk = rng.integers(-OFFSET, OFFSET, size=(10_000, 3))
    ijk[:4] = [[-OFFSET] * 3, [OFFSET - 1] * 3, [0, 0, 0], [-1, 0, 1]]
    keys = pack_keys(ijk)
    assert keys.dtype == np.int64 and np.all(keys >= 0)
    np.testing.assert_array_equal(unpack_keys(keys), ijk)
    order = np.lexsort((ijk[:, 2], ijk[:, 1], ijk[:, 0]))
    assert np.all(np.diff(keys[order]) >= 0)  # key order == (i, j, k) order
    for bad in ([OFFSET, 0, 0], [0, -OFFSET - 1, 0], [0, 0, OFFSET]):
        with pytest.raises(ValueError):
            pack_keys([bad])


def test_exact_centres():
    vs = VoxelSet(0.25)  # centres exactly representable
    vs.add([[0.3, -0.3, 1.1], [10.1, 0.1, -7.6]])
    c = vs.centers()
    assert c.dtype == np.float32 and c.shape == (2, 3)
    np.testing.assert_array_equal(c, [[0.375, -0.375, 1.125], [10.125, 0.125, -7.625]])
    vs = VoxelSet(0.1)
    vs.add(np.array([[1.234, -5.678, 2.901]], dtype=np.float32))
    np.testing.assert_allclose(vs.centers(), [[1.25, -5.65, 2.95]], atol=1e-6)


def test_non_finite_rows_ignored():
    vs = VoxelSet(0.1)
    pts = np.array([[np.nan, 0, 0], [0, np.inf, 0], [0, 0, -np.inf], [0.05, 0.05, 0.05], [np.nan] * 3])
    assert vs.add(pts) == 1
    assert vs.add(pts.astype(np.float32)) == 0
    np.testing.assert_allclose(vs.centers(), [[0.05, 0.05, 0.05]], atol=1e-6)
    assert vs.out_of_range == 0


def test_points_beyond_the_key_range_are_dropped_and_counted():
    vs = VoxelSet(1.0)
    inside = [[OFFSET - 0.5, 0.5, 0.5], [-OFFSET + 0.5, 0.5, 0.5]]
    outside = [[OFFSET + 0.5, 0.5, 0.5], [0.5, -OFFSET - 0.5, 0.5], [1e30, 0.0, 0.0]]
    assert vs.add(inside + outside) == 2
    assert vs.out_of_range == 3
    np.testing.assert_array_equal(unpack_keys(vs.keys), [[-OFFSET, 0, 0], [OFFSET - 1, 0, 0]])


def test_cap_keeps_existing_and_adds_what_fits():
    vs = VoxelSet(1.0, max_voxels=10)
    first = np.array([[i + 0.5, 0.5, 0.5] for i in range(6)])
    assert vs.add(first) == 6 and not vs.full
    second = np.array([[i + 0.5, 1.5, 0.5] for i in range(8)])
    assert vs.add(second) == 4
    assert len(vs) == 10 and vs.full and vs.dropped == 4
    kept = {tuple(c) for c in vs.centers()}
    assert {tuple(c) for c in first.astype(np.float32)} <= kept
    assert vs.add(first) == 0  # known voxels: nothing new, nothing dropped
    assert vs.dropped == 4
    assert vs.add([[50.5, 50.5, 50.5]]) == 0
    assert len(vs) == 10 and vs.dropped == 5
    assert np.all(np.diff(vs.keys) > 0)


def test_cap_exactly_reached_is_not_full():
    vs = VoxelSet(1.0, max_voxels=3)
    assert vs.add([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [2.5, 0.5, 0.5]]) == 3
    assert not vs.full
    assert vs.add([[3.5, 0.5, 0.5]]) == 0 and vs.full


def test_empty_input():
    vs = VoxelSet()
    assert vs.add(np.empty((0, 3))) == 0
    assert vs.add([]) == 0
    assert vs.add(np.full((4, 3), np.nan)) == 0
    assert len(vs) == 0
    c = vs.centers()
    assert c.shape == (0, 3) and c.dtype == np.float32
    assert crop_range(np.empty((0, 3)), 0.3, 20.0).shape == (0, 3)
    assert crop_range([], 0.3, 20.0).shape == (0, 3)


def test_bad_input_rejected():
    with pytest.raises(ValueError):
        VoxelSet().add(np.zeros((5, 4)))
    with pytest.raises(ValueError):
        VoxelSet(0.0)


def test_clear():
    vs = VoxelSet(1.0, max_voxels=2)
    vs.add([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [2.5, 0.5, 0.5]])
    assert vs.full and vs.dropped == 1
    vs.clear()
    assert len(vs) == 0 and not vs.full and vs.dropped == 0
    assert vs.add([[0.5, 0.5, 0.5]]) == 1


def test_matches_a_reference_when_added_in_pieces():
    pts = np.random.default_rng(3).normal(scale=3.0, size=(20_000, 3))
    ref = np.unique(np.floor(pts / 0.2).astype(np.int64), axis=0)
    vs = VoxelSet(0.2)
    assert sum(vs.add(chunk) for chunk in np.array_split(pts, 7)) == len(ref) == len(vs)
    np.testing.assert_array_equal(unpack_keys(vs.keys), ref)


def test_crop_range():
    pts = np.array([[0.25, 0, 0], [0, 0.5, 0], [3, 4, 0], [0, 0, 20.0], [0, 0, 20.5], [np.nan, 1, 1],
                    [np.inf, 0, 0]], dtype=np.float32)
    out = crop_range(pts, 0.5, 20.0)  # inclusive at both ends
    np.testing.assert_array_equal(out, pts[[1, 2, 3]])
    assert out.dtype == np.float32


def test_transform_points():
    q = (0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4))  # +90 deg about z
    out = transform_points([[1.0, 0.0, 0.0], [0.0, 2.0, 1.0]], (48.0, 2.1, 2.75), q)
    np.testing.assert_allclose(out, [[48.0, 3.1, 2.75], [46.0, 2.1, 3.75]], atol=1e-12)
    np.testing.assert_allclose(quat_to_matrix(0.0, 0.0, 0.0, 0.0), np.eye(3))
    np.testing.assert_allclose(quat_to_matrix(0.0, 0.0, 2.0, 0.0), np.diag([-1.0, -1.0, 1.0]), atol=1e-12)
    R = quat_to_matrix(0.3, -0.2, 0.5, 0.7)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_speed_100k_points_into_500k_voxels():
    i = np.arange(500_000)
    grid = np.stack([i % 100, (i // 100) % 100, i // 10_000], axis=1) * 0.1 + 0.05
    vs = VoxelSet(0.1, max_voxels=1_500_000)
    assert vs.add(grid) == 500_000
    pts = np.random.default_rng(2).uniform(-50.0, 50.0, size=(100_000, 3)).astype(np.float32)
    t0 = time.perf_counter()
    added = vs.add(pts)
    dt = time.perf_counter() - t0
    assert 0 < added <= 100_000 and len(vs) == 500_000 + added
    assert dt < 0.5, f'adding 100k points took {dt:.3f} s'
