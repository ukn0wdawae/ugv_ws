"""Sparse 3D voxel set for the survey viewer (numpy only, no ROS).

survey_viz_aids accumulates the lidar returns in a VoxelSet to show, without SLAM, a 3D map of what the
robot has seen. The occupied voxels are one sorted int64 array of packed voxel indices, so adding a cloud is
a few vectorised numpy calls (unique + searchsorted + insert) and the memory is 8 bytes per voxel.

Key layout: 21 bits per axis, index + 2**20 in [0, 2**21), key = (i << 42) | (j << 21) | k.
With 0.10 m voxels that covers +-104 km around the origin.
"""

import numpy as np

BITS = 21
OFFSET = 1 << (BITS - 1)  # voxel index range per axis: [-OFFSET, OFFSET - 1]
MASK = (1 << BITS) - 1


def pack_keys(ijk):
    """(N, 3) integer voxel indices -> (N,) int64 keys. ValueError for an index outside [-2**20, 2**20 - 1]."""
    ijk = np.asarray(ijk, dtype=np.int64).reshape(-1, 3)
    if ijk.size and (ijk.min() < -OFFSET or ijk.max() >= OFFSET):
        raise ValueError(f'voxel index outside [{-OFFSET}, {OFFSET - 1}]')
    u = ijk + OFFSET
    return (u[:, 0] << (2 * BITS)) | (u[:, 1] << BITS) | u[:, 2]


def unpack_keys(keys):
    """(N,) int64 keys -> (N, 3) int64 voxel indices (inverse of pack_keys)."""
    k = np.asarray(keys, dtype=np.int64).reshape(-1)
    return np.stack([(k >> (2 * BITS)) & MASK, (k >> BITS) & MASK, k & MASK], axis=1) - OFFSET


def crop_range(points, min_r, max_r):
    """Rows of ``points`` (N x 3, sensor frame) whose distance from the sensor lies in [min_r, max_r].

    Rows containing NaN are dropped too (their distance compares false)."""
    p = np.asarray(points)
    if p.size == 0:
        return p.reshape(0, 3)
    r = np.linalg.norm(p, axis=1)
    return p[(r >= min_r) & (r <= max_r)]


def quat_to_matrix(x, y, z, w):
    """3x3 rotation matrix of the quaternion (x, y, z, w); normalised here, a zero quaternion gives identity."""
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1.0 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1.0 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1.0 - s * (x * x + y * y)],
    ])


def transform_points(points, translation, quaternion):
    """R(quaternion) @ p + translation for N x 3 points -> (N, 3) float64."""
    R = quat_to_matrix(*quaternion)
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return p @ R.T + np.asarray(translation, dtype=np.float64)


class VoxelSet:
    """Occupied cubic voxels of edge ``voxel`` metres, at most ``max_voxels`` of them.

    When the cap is reached the existing voxels are kept, new ones are dropped (counted in ``dropped``) and
    ``full`` is set. Finite points too far from the origin to pack are ignored (counted in ``out_of_range``).
    Not thread-safe: use one instance from one thread at a time.
    """

    def __init__(self, voxel=0.10, max_voxels=1_500_000):
        if not voxel > 0.0:
            raise ValueError(f'voxel must be > 0, got {voxel}')
        self.voxel = float(voxel)
        self.max_voxels = max(0, int(max_voxels))
        self.clear()

    def clear(self):
        self.keys = np.empty(0, dtype=np.int64)  # sorted, unique packed voxel indices
        self.full = False
        self.dropped = 0
        self.out_of_range = 0

    def __len__(self):
        return len(self.keys)

    def add(self, points):
        """Add N x 3 points (metres, any float dtype; NaN / inf rows are ignored).

        Returns the number of newly occupied voxels."""
        p = np.asarray(points, dtype=np.float64)
        if p.size == 0:
            return 0
        if p.ndim != 2 or p.shape[1] != 3:
            raise ValueError(f'points must be N x 3, got shape {p.shape}')
        q = np.floor(p[np.isfinite(p).all(axis=1)] / self.voxel)
        packable = ((q >= -OFFSET) & (q < OFFSET)).all(axis=1)
        if not packable.all():
            self.out_of_range += int(np.count_nonzero(~packable))
            q = q[packable]
        if len(q) == 0:
            return 0
        new = np.unique(pack_keys(q))
        pos = np.searchsorted(self.keys, new)
        known = np.zeros(len(new), dtype=bool)
        inside = pos < len(self.keys)
        known[inside] = self.keys[pos[inside]] == new[inside]
        new, pos = new[~known], pos[~known]
        room = max(self.max_voxels - len(self.keys), 0)
        if len(new) > room:
            self.full = True
            self.dropped += len(new) - room
            new, pos = new[:room], pos[:room]
        if len(new):
            self.keys = np.insert(self.keys, pos, new)  # positions refer to the old array: stays sorted
        return len(new)

    def centers(self):
        """(M, 3) float32 centres of the occupied voxels."""
        return ((unpack_keys(self.keys) + 0.5) * self.voxel).astype(np.float32)
