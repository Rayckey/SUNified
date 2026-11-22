"""Check MPC grasp feedback without importing the simulator."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("robot", ["kinova_robotiq", "ur5_robotiq"])
def test_single_driver_contact_requires_stall_effort_and_grasp_position(robot):
    path = Path(__file__).resolve().parents[1] / "src/kuafu/mpc/simulation/api_wrapper.py"
    tree = ast.parse(path.read_text())
    methods = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
               and node.name in {"_uses_joint_effort_finger_contact", "_joint_effort_finger_contact_force"}]
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace)
    positions = torch.zeros(5, 8)
    positions[:, -2:] = torch.tensor([[0.4], [0.4], [0.4], [0.8], [0.04]])
    velocities = torch.zeros_like(positions)
    velocities[2, -2] = 0.3
    efforts = torch.zeros_like(positions)
    # Only the master joint supplies effort; its passive follower stays at zero.
    efforts[:, -2] = torch.tensor([8., 5., 8., 20., 20.])
    wrapper = SimpleNamespace(
        reference_agent="robot", _robot_gripper_name=robot,
        joint_dict={"robot": {"joint_position": positions, "joint_velocity": velocities,
                              "joint_effort": efforts}},
        _gripper_opened=torch.tensor([0., 0.]), _gripper_closed=torch.tensor([0.8, 0.8]),
        _gripper_max_grasp_openness=0.8, _gripper_effort_contact_velocity_threshold=0.2,
        _kinova_gripper_effort_contact_openness_slope=12.,
        _kinova_gripper_effort_contact_openness_intercept=0.,
    )
    assert namespace["_uses_joint_effort_finger_contact"](wrapper)
    left, right = namespace["_joint_effort_finger_contact_force"](wrapper)
    expected = torch.tensor([8., 0., 0., 0., 0.])
    torch.testing.assert_close(left, expected)
    torch.testing.assert_close(right, expected)
