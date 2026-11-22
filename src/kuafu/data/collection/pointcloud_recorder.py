"""Fuse synchronized depth views into point clouds in environment-local meters."""

import json

import numpy as np

from kuafu.data.collection.rgb_recorder import CAMERAS, RGBTransitionRecorder, configure_rgb_cameras
from kuafu.data.collection.state_recorder import _array


WORKSPACE_MIN = (-0.2, -0.9, 0.01)
WORKSPACE_MAX = (1.5, 0.9, 1.5)


def configure_pointcloud_cameras(cfg):
    """Capture RGB and metric image-plane depth from the same camera exposure."""
    configure_rgb_cameras(cfg)
    for key in CAMERAS:
        getattr(cfg, key).data_types = ["rgb", "distance_to_image_plane"]


def fuse_depth_points(depths, intrinsics, rotations, positions, *, rng, num_points=1024,
                      lower=WORKSPACE_MIN, upper=WORKSPACE_MAX):
    """Unproject camera-z depth; rotations map ROS optical XYZ into the local frame.

    Arrays have leading [environment,camera] dimensions. Invalid depth and
    points outside the fixed workspace are excluded before uniform sampling.
    The function never reads object poses, segmentation, or robot state.
    """
    depths = np.asarray(depths, np.float32)
    if depths.ndim != 4 or num_points < 1:
        raise ValueError("Expected [environment,camera,height,width] depth and positive point count")
    b, c, height, width = depths.shape
    if (np.shape(intrinsics) != (b, c, 3, 3) or np.shape(rotations) != (b, c, 3, 3)
            or np.shape(positions) != (b, c, 3)):
        raise ValueError("Camera calibration dimensions do not match depth")
    if not all(np.isfinite(x).all() for x in (intrinsics, rotations, positions)):
        raise ValueError("Nonfinite camera calibration")
    if np.any(intrinsics[..., (0, 1), (0, 1)] <= 0):
        raise ValueError("Camera focal lengths must be positive")
    valid_depth = np.isfinite(depths) & (depths > 0.01) & (depths < 3.0)
    z = np.where(valid_depth, depths, 0)
    v, u = np.mgrid[:height, :width]
    x = (u - intrinsics[..., 0, 2, None, None]) * z / intrinsics[..., 0, 0, None, None]
    y = (v - intrinsics[..., 1, 2, None, None]) * z / intrinsics[..., 1, 1, None, None]
    xyz = np.stack((x, y, z), axis=-1)
    xyz = np.einsum("bcij,bchwj->bchwi", rotations, xyz) + positions[:, :, None, None]
    valid = valid_depth & np.all((xyz > np.asarray(lower)) & (xyz < np.asarray(upper)), axis=-1)
    result = np.empty((b, num_points, 3), np.float32)
    counts = np.zeros(b, np.int32)
    for env_id in range(b):
        points = xyz[env_id][valid[env_id]]
        counts[env_id] = len(points)
        if len(points) < 32:
            raise ValueError(f"Environment {env_id}: only {len(points)} valid depth points; inspect camera framing")
        result[env_id] = points[rng.choice(len(points), num_points, replace=len(points) < num_points)]
    return result, counts


class PointCloudTransitionRecorder(RGBTransitionRecorder):
    """Record paired RGB and XYZ observations before the applied physics action."""

    def __init__(self, env, path, target, metadata=None):
        metadata = {**(metadata or {}), "pointcloud_timing": "pre_action",
                    "pointcloud_frame": "environment-local world axes; meters",
                    "pointcloud_source": "unprojected image-plane depth from two cameras",
                    "pointcloud_num_points": 1024, "pointcloud_features": "XYZ",
                    "pointcloud_workspace_min": WORKSPACE_MIN,
                    "pointcloud_workspace_max": WORKSPACE_MAX}
        super().__init__(env, path, target, metadata)
        self.rng = np.random.default_rng(int(metadata.get("seed", 0)) + 105729)
        self.calibration_recorded = False

    def record_pre_step(self):
        from isaaclab.utils.math import matrix_from_quat

        super().record_pre_step()
        depths, intrinsics, rotations, positions = [], [], [], []
        for key in CAMERAS:
            camera = self.env.scene.sensors[key]
            # RGBTransitionRecorder just refreshed both cameras; use that exposure.
            depths.append(_array(camera.data.output["distance_to_image_plane"])[..., 0])
            intrinsics.append(_array(camera.data.intrinsic_matrices))
            rotations.append(_array(matrix_from_quat(camera.data.quat_w_ros)))
            positions.append(_array(camera.data.pos_w - self.env.scene.env_origins))
        calibration = [np.stack(values, axis=1) for values in (intrinsics, rotations, positions)]
        points, counts = fuse_depth_points(np.stack(depths, axis=1), *calibration, rng=self.rng)
        for i, row in enumerate(self.pending):
            row["obs/pointcloud"] = points[i]
            row["obs/pointcloud_count"] = counts[i]
        if not self.calibration_recorded:
            meta = json.loads(self.data.attrs["env_args"])
            meta["pointcloud_calibration_env0"] = dict(zip(
                ("intrinsics", "optical_to_local_rotation", "local_position_m"),
                [value[0].tolist() for value in calibration]))
            self.data.attrs["env_args"] = json.dumps(meta)
            self.calibration_recorded = True
