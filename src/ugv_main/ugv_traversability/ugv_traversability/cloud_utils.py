"""PointCloud2 / transform helpers without extra dependencies (numpy only)."""

import numpy as np

# sensor_msgs/PointField datatypes
_PF = {1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16, 5: np.int32, 6: np.uint32,
       7: np.float32, 8: np.float64}


def cloud_to_xyz(msg, stride=1):
    """Return an (N, 3) float64 array of finite xyz points from a PointCloud2.

    ``stride`` > 1 subsamples (every stride-th point; for organised clouds the
    subsampling is applied in both image directions).
    """
    fields = {f.name: f for f in msg.fields}
    if not all(k in fields for k in ('x', 'y', 'z')):
        raise ValueError('PointCloud2 has no x/y/z fields')
    endian = '>' if msg.is_bigendian else '<'
    dt = np.dtype({
        'names': ['x', 'y', 'z'],
        'formats': [np.dtype(_PF[fields[k].datatype]).newbyteorder(endian) for k in ('x', 'y', 'z')],
        'offsets': [fields[k].offset for k in ('x', 'y', 'z')],
        'itemsize': msg.point_step,
    })
    n = msg.width * msg.height
    if n == 0:
        return np.empty((0, 3))
    buf = np.frombuffer(bytes(msg.data) if not isinstance(msg.data, (bytes, bytearray, memoryview))
                        else msg.data, dtype=np.uint8)
    if msg.height > 1 and msg.row_step * msg.height == len(buf) and msg.row_step == msg.width * msg.point_step:
        arr = np.ndarray((msg.height, msg.width), dtype=dt, buffer=buf)
        if stride > 1:
            arr = arr[::stride, ::stride]
        arr = arr.ravel()
    else:
        arr = np.ndarray((n,), dtype=dt, buffer=buf[:n * msg.point_step])
        if stride > 1:
            arr = arr[::stride]
    xyz = np.stack([arr['x'], arr['y'], arr['z']], axis=1).astype(np.float64)
    return xyz[np.isfinite(xyz).all(axis=1)]


def xyz_to_cloud_fields(points, extra=None):
    """Pack (N,3) xyz (+ optional (N,) float 'intensity') to raw bytes / field specs.

    Returns (data_bytes, point_step, fields) where fields is a list of
    (name, offset, datatype=7(float32), count=1).
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    cols = [pts]
    names = ['x', 'y', 'z']
    if extra is not None:
        cols.append(np.asarray(extra, dtype=np.float32).reshape(-1, 1))
        names.append('intensity')
    data = np.ascontiguousarray(np.hstack(cols), dtype=np.float32)
    fields = [(n, 4 * i, 7, 1) for i, n in enumerate(names)]
    return data.tobytes(), 4 * len(names), fields


def quat_to_rot(x, y, z, w):
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ])


def rot_to_rpy(R):
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def rpy_to_rot(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def transform_to_matrix(tf):
    """geometry_msgs/Transform -> 4x4."""
    T = np.eye(4)
    q = tf.rotation
    T[:3, :3] = quat_to_rot(q.x, q.y, q.z, q.w)
    T[:3, 3] = (tf.translation.x, tf.translation.y, tf.translation.z)
    return T


def apply(T, pts):
    return pts @ T[:3, :3].T + T[:3, 3]
