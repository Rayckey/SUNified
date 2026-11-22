"""Stationary object frames for Kuafu2Real's 10D EE pose and gripper contract."""

import numpy as np


OBJECT_RELATIVE = "eff-object-relative"
FRAME_TOLERANCE = 1e-5


def _finite(value, trailing_shape):
    value = np.asarray(value, dtype=np.float64)
    if value.shape[-len(trailing_shape):] != trailing_shape or not np.isfinite(value).all():
        raise ValueError(f"Expected finite values with trailing shape {trailing_shape}")
    return value


def _quaternion_matrix(quaternion):
    quaternion = _finite(quaternion, (4,))
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(np.abs(norm - 1) > 1e-3):
        raise ValueError("Reference rotations must be unit wxyz quaternions")
    w, x, y, z = np.moveaxis(quaternion / norm, -1, 0)
    return np.stack((1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y),
                     2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x),
                     2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)), axis=-1).reshape(
                         *quaternion.shape[:-1], 3, 3)


def rotation_from_6d(value):
    """Decode interleaved first-two-columns encoding, projecting onto SO(3)."""
    value = _finite(value, (6,))
    columns = value.reshape(*value.shape[:-1], 3, 2)
    first, second = columns[..., 0], columns[..., 1]
    norm = np.linalg.norm(first, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("Degenerate first rotation column")
    first = first / norm
    second = second - (first * second).sum(axis=-1, keepdims=True) * first
    norm = np.linalg.norm(second, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("Degenerate second rotation column")
    second = second / norm
    return np.stack((first, second, np.cross(first, second)), axis=-1)


class ObjectFrame:
    """Map environment-local meters to a fixed object's root frame and back.

    ``pose`` is [x, y, z, qw, qx, qy, qz]. Use the same reference throughout an
    observation history and action chunk. The gripper scalar is always absolute.
    """

    def __init__(self, pose):
        self.pose = _finite(pose, (7,)).copy()
        if self.pose.shape != (7,):
            raise ValueError("Choose one object reference pose")
        self.rotation = _quaternion_matrix(self.pose[3:])
        self.position = self.pose[:3]

    def check_stationary(self, poses):
        """Reject moving roots before mixing incompatible history/chunk frames."""
        poses = _finite(poses, (7,))
        position_error = np.max(np.abs(poses[..., :3] - self.position))
        rotation_error = np.max(np.abs(_quaternion_matrix(poses[..., 3:]) - self.rotation))
        if max(position_error, rotation_error) > FRAME_TOLERANCE:
            raise ValueError("Object-relative chunks require a stationary reference root; choose a fixed scene object")

    def _pose(self, value, inverse):
        value = _finite(value, (10,))
        rotation = rotation_from_6d(value[..., 3:9])
        if inverse:
            position = value[..., :3] @ self.rotation.T + self.position
            rotation = self.rotation @ rotation
        else:
            position = (value[..., :3] - self.position) @ self.rotation
            rotation = self.rotation.T @ rotation
        return np.concatenate((position, rotation[..., :, :2].reshape(*value.shape[:-1], 6),
                               value[..., 9:10]), axis=-1).astype(np.float32)

    def encode_pose(self, absolute):
        """Return inv(T_object) @ T_ee, preserving the gripper target/state."""
        return self._pose(absolute, inverse=False)

    def decode_pose(self, relative):
        """Return T_object @ T_relative for absolute EE execution."""
        return self._pose(relative, inverse=True)

    def _cloud(self, cloud, inverse):
        cloud = np.asarray(cloud, dtype=np.float32)
        if cloud.ndim < 2 or cloud.shape[-1] not in (3, 6) or not np.isfinite(cloud).all():
            raise ValueError("Expected finite XYZ or XYZRGB point clouds")
        result = cloud.copy()
        if inverse:
            result[..., :3] = cloud[..., :3] @ self.rotation.T + self.position
        else:
            result[..., :3] = (cloud[..., :3] - self.position) @ self.rotation
        return result

    def encode_cloud(self, cloud):
        """Express XYZ in the object frame; retain RGB and point order."""
        return self._cloud(cloud, inverse=False)

    def decode_cloud(self, cloud):
        """Restore environment-local XYZ for geometry audits."""
        return self._cloud(cloud, inverse=True)


def relative_contract(reference_object):
    if not isinstance(reference_object, str) or not reference_object or "/" in reference_object or reference_object == "robot":
        raise ValueError("Choose a named stationary task object")
    return {
        "schema_version": 1, "reference_object": reference_object, "reference_body": "root",
        "reference_motion": "stationary_per_episode", "pose_transform": "inverse(T_object) @ T_ee",
        "source_frame": "environment_local", "translation_units": "meters",
        "reference_pose_encoding": "xyz_wxyz", "rotation_encoding": "first_two_columns_row_major",
        "point_cloud": "object_frame_xyz_unchanged_rgb", "agent_pos": "object_frame_measured_pose",
        "action": "object_frame_absolute_pose_target", "gripper": "absolute_unchanged",
        "chunk_reference": "same_object_pose_for_all_observations_and_targets",
        "stationary_tolerance": FRAME_TOLERANCE,
    }


def validate_relative_contract(export):
    contract = export.get("object_relative")
    if contract != relative_contract((contract or {}).get("reference_object")):
        raise ValueError("Unsupported object-relative coordinate contract")
    if export.get("action_dim") != 10 or export.get("agent_pos_dim") != 10:
        raise ValueError("Object-relative data requires 10D EE poses and gripper")
    return contract
