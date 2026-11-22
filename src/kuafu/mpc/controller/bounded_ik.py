"""Joint-limit-aware pose IK for Cartesian MPC's Panda hand targets."""

from __future__ import annotations

import numpy as np
import pinocchio as pin
import torch
from scipy.optimize import least_squares


class BoundedPoseIK:
    """Honor each horizon seed and URDF joint limits when realizing MPC poses.

    Cartesian MPC already removes the fingertip offset before invoking IK, so
    this solves the hand frame. A fresh kinematics data object avoids mutating
    the controller's forward-kinematics cache. No simulator state is written.
    """

    def __init__(self, model, frame_name="panda_hand", max_evaluations=200, time_step=None):
        self.model = model
        self.data = model.createData()
        self.frame_id = model.getFrameId(frame_name)
        if self.frame_id >= len(model.frames):
            raise ValueError(f"Unknown IK frame: {frame_name}")
        self.lower = np.asarray(model.lowerPositionLimit) + 1e-5
        self.upper = np.asarray(model.upperPositionLimit) - 1e-5
        self.max_evaluations = max_evaluations
        self.max_joint_step = None if time_step is None else np.asarray(model.velocityLimit) * time_step
        self.diagnostics = []

    def __call__(self, p_des, quat_des, q_init=None, frame_name=None):
        if q_init is None:
            raise ValueError("Pose IK requires the current or previous horizon joint seed")
        seed = np.clip(np.asarray(q_init, dtype=float).reshape(-1), self.lower, self.upper)
        lower, upper = self.lower, self.upper
        if self.max_joint_step is not None:
            lower = np.maximum(lower, seed - self.max_joint_step)
            upper = np.minimum(upper, seed + self.max_joint_step)
        position = np.asarray(p_des, dtype=float)
        rotation = pin.Quaternion(*np.asarray(quat_des, dtype=float)).matrix()

        def residual(q):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            pose = self.data.oMf[self.frame_id]
            return np.concatenate((pose.translation - position, 0.3 * pin.log3(rotation.T @ pose.rotation)))

        result = least_squares(
            residual,
            seed,
            bounds=(lower, upper),
            max_nfev=self.max_evaluations,
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
        )
        error = residual(result.x)
        self.diagnostics.append(
            {
                "position_error_m": float(np.linalg.norm(error[:3])),
                "orientation_error_rad": float(np.linalg.norm(error[3:]) / 0.3),
                "evaluations": result.nfev,
            }
        )
        if not np.isfinite(result.x).all():
            raise RuntimeError("Inverse kinematics returned non-finite joint targets")
        return torch.as_tensor(result.x, dtype=torch.float32).unsqueeze(0)
