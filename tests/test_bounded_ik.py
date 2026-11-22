"""Real Panda kinematics checks for the Cartesian MPC-to-joint bridge."""

import ast
from types import SimpleNamespace

import numpy as np
import pytest

pin = pytest.importorskip("pinocchio")

from kuafu.mpc.controller.bounded_ik import BoundedPoseIK
from kuafu.paths import ROBOT_RESOURCE_ROOT, PROJECT_ROOT


@pytest.fixture
def ik():
    model = pin.buildModelFromUrdf(str(ROBOT_RESOURCE_ROOT / "franka_description/robots/panda_arm_hand_fixed.urdf"))
    return BoundedPoseIK(model)


def test_pose_round_trip_preserves_the_horizon_seed(ik):
    seed = np.array([0.0, -0.42, 0.0, -2.25, 0.0, 2.0, 0.8])
    pin.forwardKinematics(ik.model, ik.data, seed)
    pin.updateFramePlacements(ik.model, ik.data)
    pose = ik.data.oMf[ik.frame_id]
    quat = pin.Quaternion(pose.rotation).coeffs()[[3, 0, 1, 2]]
    actual = ik(pose.translation.copy(), quat, seed).numpy()[0]
    np.testing.assert_allclose(actual, seed, atol=1e-6)


def test_reachable_target_at_wrist_limit_uses_redundant_joints(ik):
    # Actual failed differential-IK configuration from the generated-cabinet scene.
    seed = np.array([-0.63495, 0.33119, 0.51409, -2.93493, 2.89730, 1.51089, -0.95689])
    actual = ik(np.array([0.3276, 0.0, 0.137]), np.array([0.5, -0.5, 0.5, -0.5]), seed).numpy()[0]
    assert np.all(actual >= ik.lower - 1e-6)
    assert np.all(actual <= ik.upper + 1e-6)
    assert ik.diagnostics[-1]["position_error_m"] < 2e-4
    assert ik.diagnostics[-1]["orientation_error_rad"] < 2e-3


def test_unreachable_target_remains_bounded_and_records_error(ik):
    seed = np.array([0.0, -0.42, 0.0, -2.25, 0.0, 2.0, 0.8])
    actual = ik([10.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], seed).numpy()[0]
    assert np.all(actual >= ik.lower - 1e-6)
    assert np.all(actual <= ik.upper + 1e-6)
    assert ik.diagnostics[-1]["position_error_m"] > 5.0


def test_horizon_step_respects_joint_velocity_limits(ik):
    bounded = BoundedPoseIK(ik.model, time_step=1 / 30)
    seed = np.array([0.0, -0.42, 0.0, -2.25, 0.0, 2.0, 0.8])
    actual = bounded([0.3, 0.0, 0.2], [0.5, -0.5, 0.5, -0.5], seed).numpy()[0]
    assert np.all(np.abs(actual - seed) <= ik.model.velocityLimit / 30 + 1e-6)


def test_cartesian_mpc_exports_absolute_joint_targets(ik):
    source = PROJECT_ROOT / "src/kuafu/mpc/controller/mpc.py"
    tree = ast.parse(source.read_text())
    controller = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                      and node.name == "ManipulatorIKController")
    method = next(node for node in controller.body if isinstance(node, ast.FunctionDef)
                  and node.name == "compute_control")
    def quaternion(rotation):
        return pin.Quaternion(rotation).coeffs()[[3, 0, 1, 2]]

    namespace = {"np": np, "pin": pin, "matrix_to_quat": quaternion}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    seed = np.array([0., -0.42, 0., -2.25, 0., 2., 0.8])
    pin.forwardKinematics(ik.model, ik.data, seed)
    pin.updateFramePlacements(ik.model, ik.data)
    pose = ik.data.oMf[ik.frame_id].copy()
    targets = np.tile(np.r_[pose.translation, quaternion(pose.rotation), [0.04, 0.04]], (3, 1))
    targets[:, 0] += [0., 0.001, 0.002]
    targets[:, -2:] = np.array([0.04, 0.03, 0.02])[:, None]
    stub = SimpleNamespace(
        _ik_fun=ik, _fk_fun=lambda *args: pose.homogeneous.copy(), ee_frame="panda_hand",
        solve=lambda *args, **kwargs: (targets, np.zeros((2, 8))),
        control_output_mode="absolute_joint", offset_trans=np.eye(4), dt=1 / 30,
    )
    joints = namespace["compute_control"](stub, np.r_[seed, [0.04, 0.04]], None)
    assert joints.shape == (3, 9)
    np.testing.assert_allclose(joints[:, -2:], targets[:, -2:])
    for q, target in zip(joints, targets):
        pin.forwardKinematics(ik.model, ik.data, q[:-2])
        pin.updateFramePlacements(ik.model, ik.data)
        np.testing.assert_allclose(ik.data.oMf[ik.frame_id].translation, target[:3], atol=1e-5)
