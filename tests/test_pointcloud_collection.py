"""Metric camera projection must preserve axes, calibration, and frame origins."""

import numpy as np
import pytest

from kuafu.data.collection.pointcloud_recorder import fuse_depth_points


def test_depth_projection_uses_camera_z_and_local_extrinsics():
    depth = np.ones((1, 2, 8, 8), np.float32)
    intrinsics = np.tile(np.array([[10, 0, 3], [0, 10, 4], [0, 0, 1]], np.float32), (1, 2, 1, 1))
    rotation = np.tile(np.eye(3, dtype=np.float32), (1, 2, 1, 1))
    rotation[:, 1] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    position = np.array([[[0, 0, 0], [2, 0, 0]]], np.float32)
    points, counts = fuse_depth_points(depth, intrinsics, rotation, position, rng=np.random.default_rng(0),
        num_points=128, lower=(-10, -10, -10), upper=(10, 10, 10))
    assert counts.tolist() == [128]
    expected = []
    for camera in range(2):
        for v in range(8):
            for u in range(8):
                expected.append(rotation[0, camera] @ np.array([(u - 3) / 10, (v - 4) / 10, 1]) + position[0, camera])
    actual = {tuple(np.round(p, 5)) for p in points[0]}
    expected = {tuple(np.round(p.astype(np.float32), 5)) for p in expected}
    assert actual == expected


def test_invalid_depth_and_crop_are_excluded_without_advancing_global_rng():
    d = np.ones((1, 1, 8, 8), np.float32)
    d[0, 0, 0] = np.inf
    k = np.array([[[[10, 0, 3], [0, 10, 3], [0, 0, 1]]]], np.float32)
    r = np.eye(3, dtype=np.float32)[None, None]
    p = np.zeros((1, 1, 3), np.float32)
    result, count = fuse_depth_points(d, k, r, p, rng=np.random.default_rng(3), num_points=64)
    assert result.shape == (1, 64, 3) and count[0] < 64
    assert np.isfinite(result).all() and (result[..., 0] > -0.2).all()
    np.testing.assert_array_equal(result, fuse_depth_points(d, k, r, p, rng=np.random.default_rng(3), num_points=64)[0])
    with pytest.raises(ValueError, match="valid depth points"):
        fuse_depth_points(np.zeros_like(d), k, r, p, rng=np.random.default_rng(3))
