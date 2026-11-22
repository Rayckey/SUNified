#!/usr/bin/env python3
"""Visual Kinova Robotiq gripper open/close test.

By default this spawns the Kinova robot from ``base_env_learned.py`` and uses
the official contact-stable Robotiq USD mechanism. The logical 9D interface
still has two gripper slots, but the physical target is sent only to
``robotiq_85_left_knuckle_joint``; the remaining Robotiq joints follow the USD
mimic mechanism.

The script also displays the env-style panda_fingertip frame and, by default,
checks that the env FrameTransformer, SUNProgramEvaluator APIWrapper, SUNProgramExecutor APIWrapper, and
SUNProgramExecutor Pinocchio FK agree on that frame.
"""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from functools import partial
from pathlib import Path

from kuafu.paths import resolve_artifact_path

from isaaclab.app import AppLauncher

print = partial(print, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dt", type=float, default=1.0 / 120.0)
    parser.add_argument("--settle_steps", type=int, default=120)
    parser.add_argument("--close_steps", type=int, default=240)
    parser.add_argument("--hold_steps", type=int, default=240)
    parser.add_argument("--open_steps", type=int, default=120)
    parser.add_argument(
        "--cycles",
        type=int,
        default=1,
        help="Number of close cycles. Use 0 to repeat until the Isaac window is closed.",
    )
    parser.add_argument("--print_every", type=int, default=30)
    parser.add_argument(
        "--asset_mode",
        choices=("env", "packaged_isaaclab", "hybrid_mechanism"),
        default="env",
        help="Robot asset to spawn. hybrid_mechanism uses current meshes with the packaged six-joint Robotiq mechanism.",
    )
    parser.add_argument(
        "--usd_override",
        type=str,
        default="",
        help="Optional USD path override for packaged_isaaclab or hybrid_mechanism modes.",
    )
    parser.add_argument(
        "--frame_check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compare panda_fingertip from env FrameTransformer, SUNProgramEvaluator, SUNProgramExecutor API, and SUNProgramExecutor FK.",
    )
    parser.add_argument("--frame_check_every", type=int, default=30)
    parser.add_argument("--frame_pos_tolerance", type=float, default=5.0e-3)
    parser.add_argument("--frame_rot_tolerance", type=float, default=5.0e-3)
    parser.add_argument(
        "--gripper_joint_frame_check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compare Robotiq gripper joint-frame positions from the loaded USD against the URDF FK.",
    )
    parser.add_argument("--gripper_joint_frame_check_every", type=int, default=30)
    parser.add_argument("--gripper_joint_frame_pos_tolerance", type=float, default=1.0e-3)
    parser.add_argument(
        "--gripper_joint_frame_fail_on_mismatch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Raise if the USD/URDF gripper joint or link global position error exceeds the tolerance.",
    )
    parser.add_argument(
        "--arm_pose_joint_frame_check",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Check only Robotiq joint-frame world positions at several arm poses with the gripper open.",
    )
    parser.add_argument(
        "--arm_pose_joint_frame_csv",
        type=str,
        default="diagnostics/kinova_gripper_joint_world_positions.csv",
        help="CSV path for --arm_pose_joint_frame_check detailed joint-frame positions.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = False
    return args


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.markers.config import FRAME_MARKER_CFG  # noqa: E402
from isaaclab.sensors import FrameTransformerCfg  # noqa: E402
from isaaclab.sensors.frame_transformer.frame_transformer import FrameTransformer  # noqa: E402
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils.math import quat_from_matrix  # noqa: E402

from kuafu.mpc.simulation.sun_program_evaluator import SUNProgramEvaluator  # noqa: E402
from kuafu.mpc.simulation.sun_program_executor import SUNProgramExecutor  # noqa: E402
from kuafu.isaac.learned.kinova.kinova_table_env_learned import KinovaTableEnvLearnedCfg  # noqa: E402


GRIPPER_JOINT_NAMES = [
    "finger_joint",
    "left_inner_knuckle_joint",
    "left_inner_finger_joint",
    "right_outer_knuckle_joint",
    "right_inner_knuckle_joint",
    "right_inner_finger_joint",
]
DRIVER_JOINT_NAMES = list(GRIPPER_JOINT_NAMES)
OPEN_TARGET = torch.zeros(6, dtype=torch.float32)
ENV_CLOSE_SIGNS = torch.tensor([1.0, 1.0, -1.0, -1.0, -1.0, 1.0], dtype=torch.float32)
CLOSE_TARGET = ENV_CLOSE_SIGNS * 0.8
PACKAGED_GRIPPER_JOINT_NAMES = GRIPPER_JOINT_NAMES
PACKAGED_CLOSE_SIGNS = torch.tensor([1.0, 1.0, -1.0, 1.0, 1.0, -1.0], dtype=torch.float32)
PACKAGED_ASSET_MODES = {"packaged_isaaclab", "hybrid_mechanism"}
OFFICIAL_GRIPPER_JOINT_NAMES = [
    "robotiq_85_left_knuckle_joint",
    "robotiq_85_right_knuckle_joint",
    "robotiq_85_left_inner_knuckle_joint",
    "robotiq_85_right_inner_knuckle_joint",
    "robotiq_85_left_finger_tip_joint",
    "robotiq_85_right_finger_tip_joint",
]
OFFICIAL_LOGICAL_GRIPPER_JOINT_NAMES = [
    "robotiq_85_left_knuckle_joint",
    "robotiq_85_right_knuckle_joint",
]
OFFICIAL_DRIVER_JOINT_NAMES = ["robotiq_85_left_knuckle_joint"]
FRAME_NAME = "panda_fingertip"
GRIPPER_JOINT_FRAME_PAIRS = [
    ("robotiq_2f_85_fixed_joint", "gripper_base_joint"),
    ("finger_joint", "finger_joint"),
    ("left_outer_finger_joint", "left_outer_finger_joint"),
    ("left_inner_knuckle_joint", "left_inner_knuckle_joint"),
    ("left_inner_finger_joint", "left_inner_finger_joint"),
    ("right_outer_knuckle_joint", "right_outer_knuckle_joint"),
    ("right_outer_finger_joint", "right_outer_finger_joint"),
    ("right_inner_knuckle_joint", "right_inner_knuckle_joint"),
    ("right_inner_finger_joint", "right_inner_finger_joint"),
]
USD_ONLY_GRIPPER_JOINTS = []
URDF_ONLY_GRIPPER_JOINTS = [
    "left_inner_finger_pad_joint",
    "right_inner_finger_pad_joint",
]
GRIPPER_LINK_FRAME_PAIRS = [
    ("/Robotiq_2F_85/base_link", "robotiq_arg2f_base_link"),
    ("/Robotiq_2F_85/left_outer_knuckle", "left_outer_knuckle"),
    ("/Robotiq_2F_85/right_outer_knuckle", "right_outer_knuckle"),
    ("/Robotiq_2F_85/left_outer_finger", "left_outer_finger"),
    ("/Robotiq_2F_85/right_outer_finger", "right_outer_finger"),
    ("/Robotiq_2F_85/left_inner_knuckle", "left_inner_knuckle"),
    ("/Robotiq_2F_85/right_inner_knuckle", "right_inner_knuckle"),
    ("/Robotiq_2F_85/left_inner_finger", "left_inner_finger"),
    ("/Robotiq_2F_85/right_inner_finger", "right_inner_finger"),
]
URDF_ONLY_GRIPPER_LINKS = [
    "left_inner_finger_pad",
    "right_inner_finger_pad",
]


def _fmt_pair(values: torch.Tensor) -> str:
    vals = values.detach().cpu().tolist()
    return ", ".join(f"{v:+.6f}" for v in vals)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _packaged_isaaclab_usd_path() -> Path:
    return _repo_root() / "kinova_gen3_7dof_robotiq_2f85_vision_isaaclab.usd"


def _hybrid_mechanism_usd_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "assets"
        / "kinova"
        / "usd"
        / "gen3_7dof_vision_robotiq_2f_85_current_mesh_packaged_mechanism.usd"
    )


def _uses_six_joint_gripper_asset() -> bool:
    return args_cli.asset_mode in PACKAGED_ASSET_MODES


def _configure_packaged_isaaclab_asset(robot_cfg) -> None:
    if args_cli.usd_override:
        usd_path = Path(args_cli.usd_override).expanduser()
    elif args_cli.asset_mode == "hybrid_mechanism":
        usd_path = _hybrid_mechanism_usd_path()
    else:
        usd_path = _packaged_isaaclab_usd_path()
    robot_cfg.spawn.usd_path = str(usd_path)

    init_joint_pos = {
        "joint_1": 0.0,
        "joint_2": 0.25122,
        "joint_3": 0.0,
        "joint_4": 1.15040,
        "joint_5": 0.0,
        "joint_6": 0.46409,
        "joint_7": 0.0,
    }
    init_joint_pos.update({name: 0.0 for name in PACKAGED_GRIPPER_JOINT_NAMES})
    robot_cfg.init_state.joint_pos = init_joint_pos

    hand = robot_cfg.actuators.get("kinova_hand")
    if hand is None:
        raise RuntimeError("kinova_hand actuator is missing from the Kinova robot config")
    hand.joint_names_expr = list(PACKAGED_GRIPPER_JOINT_NAMES)
    hand.effort_limit_sim = 10.0
    hand.velocity_limit_sim = 2.0
    hand.stiffness = 11.25
    hand.damping = 0.10
    hand.friction = 0.0
    hand.armature = 0.0
    robot_cfg.actuators.pop("kinova_hand_distal", None)
    robot_cfg.actuators.pop("kinova_gripper_finger", None)
    robot_cfg.actuators.pop("kinova_gripper_passive", None)


def _active_gripper_joint_names() -> list[str]:
    if _uses_six_joint_gripper_asset():
        return list(PACKAGED_GRIPPER_JOINT_NAMES)
    return list(OFFICIAL_GRIPPER_JOINT_NAMES)


def _active_logical_gripper_joint_names() -> list[str]:
    if _uses_six_joint_gripper_asset():
        return list(PACKAGED_GRIPPER_JOINT_NAMES)
    return list(OFFICIAL_LOGICAL_GRIPPER_JOINT_NAMES)


def _active_driver_joint_names() -> list[str]:
    if _uses_six_joint_gripper_asset():
        return list(PACKAGED_GRIPPER_JOINT_NAMES)
    return list(OFFICIAL_DRIVER_JOINT_NAMES)


def _active_gripper_target(close_amount: float, reference: torch.Tensor) -> torch.Tensor:
    if args_cli.asset_mode in PACKAGED_ASSET_MODES:
        signs = PACKAGED_CLOSE_SIGNS.to(device=reference.device, dtype=reference.dtype)
        return signs * float(close_amount)
    return torch.full((1,), float(close_amount), device=reference.device, dtype=reference.dtype)


def _fmt_xyz(values: torch.Tensor) -> str:
    vals = values.detach().cpu().reshape(-1).tolist()
    return ", ".join(f"{v:+.6f}" for v in vals[:3])


def _stabilizer_targets(driver_targets: torch.Tensor) -> torch.Tensor:
    close_amount = 0.5 * (driver_targets[..., 0:1] - driver_targets[..., 1:2])
    close_amount = torch.clamp(close_amount, min=0.0, max=0.8)
    return torch.cat([-close_amount, -close_amount], dim=-1)


def _quat_angle_error(q_ref: torch.Tensor, q_other: torch.Tensor) -> torch.Tensor:
    q_ref = q_ref.reshape(q_other.shape)
    dot = torch.sum(q_ref * q_other, dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def _make_fingertip_frame(robot_cfg) -> FrameTransformer:
    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)
    return FrameTransformer(
        FrameTransformerCfg(
            prim_path=robot_cfg.frame_transformer_source_prim_path.replace("/World/envs/env_.*/Robot", "/World/Robot"),
            debug_vis=False,
            visualizer_cfg=marker_cfg.replace(prim_path="/Visuals/panda_fingertip_frame"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path=robot_cfg.frame_transformer_target_prim_path.replace("/World/envs/env_.*/Robot", "/World/Robot"),
                    name=FRAME_NAME,
                    offset=OffsetCfg(
                        pos=robot_cfg.frame_transformer_offset_pos,
                        rot=robot_cfg.frame_transformer_offset_rot,
                    ),
                )
            ],
        )
    )


def _logical_joint_state(robot: Articulation, controlled_ids: list[int]) -> dict:
    joint_pos = robot.data.joint_pos[:, controlled_ids].detach().clone()
    joint_vel = robot.data.joint_vel[:, controlled_ids].detach().clone()
    joint_effort = robot.data.applied_torque[:, controlled_ids].detach().clone()
    return {
        "agent": {
            "joint_position": joint_pos,
            "joint_velocity": joint_vel,
            "joint_effort": joint_effort,
        }
    }



def _fmt_np_xyz(values: np.ndarray) -> str:
    vals = np.asarray(values, dtype=float).reshape(-1)
    return ", ".join(f"{v:+.6f}" for v in vals[:3])


def _parse_float_token(token: str | None, default: float = 0.0) -> float:
    if token is None:
        return default
    token = token.strip()
    if not token:
        return default
    try:
        return float(token)
    except ValueError:
        return float(eval(token, {"__builtins__": {}}, {"pi": math.pi}))


def _parse_vec3(text: str | None, default: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    if text is None:
        return np.asarray(default, dtype=float)
    parts = text.split()
    if len(parts) != 3:
        raise ValueError(f"Expected xyz/rpy triple, got {text!r}")
    return np.asarray([_parse_float_token(part) for part in parts], dtype=float)


def _np_transform(pos: np.ndarray | None = None, rot: np.ndarray | None = None) -> np.ndarray:
    out = np.eye(4, dtype=float)
    if rot is not None:
        out[:3, :3] = rot
    if pos is not None:
        out[:3, 3] = np.asarray(pos, dtype=float)
    return out


def _np_rx(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


def _np_ry(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def _np_rz(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def _np_rpy_transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(v) for v in rpy]
    return _np_transform(xyz, _np_rz(yaw) @ _np_ry(pitch) @ _np_rx(roll))


def _np_axis_rotation(axis: np.ndarray | str, angle: float) -> np.ndarray:
    if isinstance(axis, str):
        token = axis.upper()
        if token == "X":
            axis = np.asarray([1.0, 0.0, 0.0], dtype=float)
        elif token == "Y":
            axis = np.asarray([0.0, 1.0, 0.0], dtype=float)
        elif token == "Z":
            axis = np.asarray([0.0, 0.0, 1.0], dtype=float)
        else:
            raise ValueError(f"Unsupported USD joint axis {axis!r}")
    axis = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm <= 0.0:
        return np.eye(3, dtype=float)
    x, y, z = axis / norm
    c = math.cos(float(angle))
    s = math.sin(float(angle))
    one_c = 1.0 - c
    return np.asarray(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=float,
    )


def _np_quat_wxyz_to_matrix(quat) -> np.ndarray:
    if quat is None:
        return np.eye(3, dtype=float)
    if hasattr(quat, "GetReal"):
        imag = quat.GetImaginary()
        wxyz = [quat.GetReal(), imag[0], imag[1], imag[2]]
    else:
        wxyz = list(quat)
    w, x, y, z = [float(v) for v in wxyz]
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 0.0:
        return np.eye(3, dtype=float)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _usd_vec3(value) -> np.ndarray:
    if value is None:
        return np.zeros(3, dtype=float)
    return np.asarray([float(value[0]), float(value[1]), float(value[2])], dtype=float)


def _robot_root_pose_w(robot: Articulation) -> np.ndarray:
    pos = robot.data.root_pos_w[0].detach().cpu().numpy()
    quat = robot.data.root_quat_w[0].detach().cpu().numpy()
    return _np_transform(pos, _np_quat_wxyz_to_matrix(quat))


def _resolve_unique_suffix(candidates: list[str], suffix: str) -> str:
    matches = [candidate for candidate in candidates if candidate.endswith(suffix)]
    if len(matches) == 1:
        return matches[0]
    if not matches and not suffix.startswith("/"):
        matches = [candidate for candidate in candidates if candidate.split("/")[-1] == suffix]
        if len(matches) == 1:
            return matches[0]
    raise RuntimeError(f"Could not resolve unique suffix {suffix!r}; matches={matches}")


def _resolve_body_id(robot: Articulation, usd_link_key: str, urdf_link_name: str) -> int | None:
    body_names = list(robot.body_names)
    exact_suffix_matches = [i for i, name in enumerate(body_names) if name == usd_link_key or name.endswith(usd_link_key)]
    if len(exact_suffix_matches) == 1:
        return exact_suffix_matches[0]
    usd_tail = usd_link_key.split("/Robotiq_2F_85/")[-1]
    robotiq_matches = [
        i
        for i, name in enumerate(body_names)
        if "Robotiq_2F_85" in name and name.split("/")[-1] == usd_tail
    ]
    if len(robotiq_matches) == 1:
        return robotiq_matches[0]
    if urdf_link_name == "robotiq_arg2f_base_link":
        return None
    simple_matches = [i for i, name in enumerate(body_names) if name.split("/")[-1] == usd_tail]
    if len(simple_matches) == 1:
        return simple_matches[0]
    urdf_matches = [i for i, name in enumerate(body_names) if name.split("/")[-1] == urdf_link_name]
    if len(urdf_matches) == 1:
        return urdf_matches[0]
    return None


class _UrdfJoint:
    def __init__(self, elem: ET.Element):
        self.name = elem.attrib["name"]
        self.type = elem.attrib.get("type", "fixed")
        parent = elem.find("parent")
        child = elem.find("child")
        if parent is None or child is None:
            raise ValueError(f"URDF joint {self.name!r} is missing parent or child")
        self.parent = parent.attrib["link"]
        self.child = child.attrib["link"]
        origin = elem.find("origin")
        xyz = _parse_vec3(origin.attrib.get("xyz") if origin is not None else None)
        rpy = _parse_vec3(origin.attrib.get("rpy") if origin is not None else None)
        self.origin = _np_rpy_transform(xyz, rpy)
        axis = elem.find("axis")
        self.axis = _parse_vec3(axis.attrib.get("xyz") if axis is not None else None, (1.0, 0.0, 0.0))
        mimic = elem.find("mimic")
        self.mimic_joint = mimic.attrib.get("joint") if mimic is not None else None
        self.mimic_multiplier = _parse_float_token(mimic.attrib.get("multiplier"), 1.0) if mimic is not None else 1.0
        self.mimic_offset = _parse_float_token(mimic.attrib.get("offset"), 0.0) if mimic is not None else 0.0

    def value(self, q_by_name: dict[str, float]) -> float:
        if self.type == "fixed":
            return 0.0
        if self.mimic_joint:
            return self.mimic_multiplier * float(q_by_name.get(self.mimic_joint, 0.0)) + self.mimic_offset
        return float(q_by_name.get(self.name, 0.0))


class _UrdfKinematics:
    def __init__(self, urdf_path: str | Path):
        self.urdf_path = Path(urdf_path)
        root = ET.parse(self.urdf_path).getroot()
        self.joints = {elem.attrib["name"]: _UrdfJoint(elem) for elem in root.findall("joint") if "name" in elem.attrib}
        self.children: dict[str, list[_UrdfJoint]] = {}
        child_links = set()
        for joint in self.joints.values():
            self.children.setdefault(joint.parent, []).append(joint)
            child_links.add(joint.child)
        parent_links = set(self.children)
        self.root_link = "base_link" if "base_link" in parent_links else next(iter(parent_links - child_links), "base_link")

    def forward(self, q_by_name: dict[str, float]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        link_poses = {self.root_link: np.eye(4, dtype=float)}
        joint_poses: dict[str, np.ndarray] = {}
        stack = [self.root_link]
        visited = set()
        while stack:
            parent = stack.pop()
            if parent in visited:
                continue
            visited.add(parent)
            parent_pose = link_poses[parent]
            for joint in self.children.get(parent, []):
                joint_pose = parent_pose @ joint.origin
                joint_poses[joint.name] = joint_pose
                value = joint.value(q_by_name)
                if joint.type in {"revolute", "continuous"}:
                    child_pose = joint_pose @ _np_transform(rot=_np_axis_rotation(joint.axis, value))
                elif joint.type == "prismatic":
                    child_pose = joint_pose @ _np_transform(pos=joint.axis * value)
                else:
                    child_pose = joint_pose
                link_poses[joint.child] = child_pose
                stack.append(joint.child)
        return link_poses, joint_poses


class _UsdJoint:
    def __init__(self, prim):
        self.name = prim.GetName()
        self.type = prim.GetTypeName()
        body0 = prim.GetRelationship("physics:body0").GetTargets()
        body1 = prim.GetRelationship("physics:body1").GetTargets()
        if not body0 or not body1:
            raise ValueError(f"USD joint {self.name!r} is missing body0/body1")
        self.parent = str(body0[0])
        self.child = str(body1[0])
        local_pos0 = _usd_vec3(prim.GetAttribute("physics:localPos0").Get())
        local_pos1 = _usd_vec3(prim.GetAttribute("physics:localPos1").Get())
        local_rot0 = _np_quat_wxyz_to_matrix(prim.GetAttribute("physics:localRot0").Get())
        local_rot1 = _np_quat_wxyz_to_matrix(prim.GetAttribute("physics:localRot1").Get())
        self.local0 = _np_transform(local_pos0, local_rot0)
        self.local1 = _np_transform(local_pos1, local_rot1)
        axis_attr = prim.GetAttribute("physics:axis")
        self.axis = str(axis_attr.Get()) if axis_attr and axis_attr.Get() is not None else "Z"

    @property
    def is_revolute(self) -> bool:
        return "RevoluteJoint" in self.type


class _UsdKinematics:
    def __init__(self, skip_joint_names: set[str] | None = None):
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("Could not get the current USD stage")
        skip_joint_names = skip_joint_names or set()
        self.joints: dict[str, _UsdJoint] = {}
        self.children: dict[str, list[_UsdJoint]] = {}
        child_links = set()
        for prim in stage.Traverse():
            type_name = prim.GetTypeName()
            if "PhysicsRevoluteJoint" not in type_name and "PhysicsFixedJoint" not in type_name:
                continue
            try:
                joint = _UsdJoint(prim)
            except ValueError:
                continue
            self.joints[joint.name] = joint
            if joint.name in skip_joint_names:
                continue
            self.children.setdefault(joint.parent, []).append(joint)
            child_links.add(joint.child)
        if "joint_1" in self.joints:
            self.root_link = self.joints["joint_1"].parent
        else:
            parent_links = set(self.children)
            self.root_link = next(iter(parent_links - child_links), None)
        if self.root_link is None:
            raise RuntimeError("Could not determine USD kinematic root link")

    def forward(self, q_by_name: dict[str, float]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        link_poses = {self.root_link: np.eye(4, dtype=float)}
        joint_poses: dict[str, np.ndarray] = {}
        stack = [self.root_link]
        visited = set()
        while stack:
            parent = stack.pop()
            if parent in visited:
                continue
            visited.add(parent)
            parent_pose = link_poses[parent]
            for joint in self.children.get(parent, []):
                joint_pose = parent_pose @ joint.local0
                joint_poses[joint.name] = joint_pose
                if joint.is_revolute:
                    motion = _np_transform(rot=_np_axis_rotation(joint.axis, float(q_by_name.get(joint.name, 0.0))))
                else:
                    motion = np.eye(4, dtype=float)
                child_pose = joint_pose @ motion @ np.linalg.inv(joint.local1)
                link_poses[joint.child] = child_pose
                stack.append(joint.child)
        return link_poses, joint_poses


class GripperJointFrameChecker:
    def __init__(
        self,
        env_cfg: KinovaTableEnvLearnedCfg,
        robot: Articulation,
        pos_tolerance: float,
        check_every: int,
        fail_on_mismatch: bool,
    ):
        self.pos_tolerance = float(pos_tolerance)
        self.check_every = max(1, int(check_every))
        self.fail_on_mismatch = bool(fail_on_mismatch)
        self.urdf = _UrdfKinematics(env_cfg.mpc_config["urdf_path"])
        self.usd = _UsdKinematics(skip_joint_names=set(USD_ONLY_GRIPPER_JOINTS))
        self.num_checks = 0
        self.max_joint_by_pair: dict[tuple[str, str], tuple[float, np.ndarray, np.ndarray]] = {}
        self.max_link_by_pair: dict[tuple[str, str], tuple[float, np.ndarray, np.ndarray]] = {}
        self.max_body_by_pair: dict[tuple[str, str], tuple[float, np.ndarray, np.ndarray, np.ndarray]] = {}

        missing_usd = [usd_name for usd_name, _ in GRIPPER_JOINT_FRAME_PAIRS if usd_name not in self.usd.joints]
        missing_urdf = [urdf_name for _, urdf_name in GRIPPER_JOINT_FRAME_PAIRS if urdf_name not in self.urdf.joints]
        if missing_usd or missing_urdf:
            raise RuntimeError(
                "Cannot compare gripper joint frames: "
                f"missing_usd={missing_usd}, missing_urdf={missing_urdf}"
            )

        usd_links = sorted(
            set(self.usd.children.keys())
            | {joint.child for children in self.usd.children.values() for joint in children}
        )
        self.link_pairs: list[tuple[str, str, int | None]] = []
        for usd_suffix, urdf_link in GRIPPER_LINK_FRAME_PAIRS:
            usd_link = _resolve_unique_suffix(usd_links, usd_suffix)
            if urdf_link not in self.urdf.children and urdf_link not in {joint.child for joint in self.urdf.joints.values()}:
                raise RuntimeError(f"URDF link {urdf_link!r} is missing from {self.urdf.urdf_path}")
            body_id = _resolve_body_id(robot, usd_link, urdf_link)
            self.link_pairs.append((usd_link, urdf_link, body_id))

        print(
            "[KinovaGripperJointFrameCheck] enabled: global USD joint/link positions vs URDF FK "
            f"(warn_tol={self.pos_tolerance:g} m, fail_on_mismatch={self.fail_on_mismatch})"
        )
        print(
            "[KinovaGripperJointFrameCheck] mapped joints: "
            + ", ".join(f"{usd}->{urdf}" for usd, urdf in GRIPPER_JOINT_FRAME_PAIRS)
        )
        print(
            "[KinovaGripperJointFrameCheck] mapped links: "
            + ", ".join(f"{usd}->{urdf}" for usd, urdf, _ in self.link_pairs)
        )
        body_mappings = [
            f"{urdf}->{robot.body_names[body_id]}" if body_id is not None else f"{urdf}-><unavailable>"
            for _, urdf, body_id in self.link_pairs
        ]
        print("[KinovaGripperJointFrameCheck] Isaac body_pos_w mappings: " + ", ".join(body_mappings))
        missing_body_links = [urdf for _, urdf, body_id in self.link_pairs if body_id is None]
        if missing_body_links:
            print(
                "[KinovaGripperJointFrameCheck] Isaac body_pos_w unavailable for links: "
                + ", ".join(missing_body_links)
            )
        print(
            "[KinovaGripperJointFrameCheck] USD-only loop/closure joints excluded from tree FK: "
            + ", ".join(USD_ONLY_GRIPPER_JOINTS)
        )
        print(
            "[KinovaGripperJointFrameCheck] URDF-only pad joints/links: "
            + ", ".join(URDF_ONLY_GRIPPER_JOINTS + URDF_ONLY_GRIPPER_LINKS)
        )

    def update(self, step_count: int, label: str, robot: Articulation, joint_name_to_id: dict[str, int]) -> None:
        if step_count % self.check_every != 0:
            return
        q_by_name = {
            name: float(robot.data.joint_pos[0, joint_id].detach().cpu())
            for name, joint_id in joint_name_to_id.items()
        }
        usd_link_poses, usd_joint_poses = self.usd.forward(q_by_name)
        urdf_link_poses, urdf_joint_poses = self.urdf.forward(q_by_name)
        root_pose_w = _robot_root_pose_w(robot)

        worst_joint_pair = None
        worst_joint_error = -1.0
        worst_joint_usd_pos = None
        worst_joint_urdf_pos = None
        for pair in GRIPPER_JOINT_FRAME_PAIRS:
            usd_name, urdf_name = pair
            usd_pos = (root_pose_w @ usd_joint_poses[usd_name])[:3, 3]
            urdf_pos = (root_pose_w @ urdf_joint_poses[urdf_name])[:3, 3]
            error = float(np.linalg.norm(usd_pos - urdf_pos))
            previous = self.max_joint_by_pair.get(pair)
            if previous is None or error > previous[0]:
                self.max_joint_by_pair[pair] = (error, usd_pos.copy(), urdf_pos.copy())
            if error > worst_joint_error:
                worst_joint_error = error
                worst_joint_pair = pair
                worst_joint_usd_pos = usd_pos
                worst_joint_urdf_pos = urdf_pos

        worst_link_pair = None
        worst_link_error = -1.0
        worst_link_usd_pos = None
        worst_link_urdf_pos = None
        worst_body_pair = None
        worst_body_error = -1.0
        worst_body_usd_pos = None
        worst_body_urdf_pos = None
        worst_body_pos = None
        for usd_link, urdf_link, body_id in self.link_pairs:
            pair = (usd_link, urdf_link)
            usd_pos = (root_pose_w @ usd_link_poses[usd_link])[:3, 3]
            urdf_pos = (root_pose_w @ urdf_link_poses[urdf_link])[:3, 3]
            error = float(np.linalg.norm(usd_pos - urdf_pos))
            previous = self.max_link_by_pair.get(pair)
            if previous is None or error > previous[0]:
                self.max_link_by_pair[pair] = (error, usd_pos.copy(), urdf_pos.copy())
            if error > worst_link_error:
                worst_link_error = error
                worst_link_pair = pair
                worst_link_usd_pos = usd_pos
                worst_link_urdf_pos = urdf_pos

            if body_id is not None:
                body_pos = robot.data.body_pos_w[0, body_id].detach().cpu().numpy()
                body_error = float(np.linalg.norm(body_pos - urdf_pos))
                previous_body = self.max_body_by_pair.get(pair)
                if previous_body is None or body_error > previous_body[0]:
                    self.max_body_by_pair[pair] = (body_error, body_pos.copy(), usd_pos.copy(), urdf_pos.copy())
                if body_error > worst_body_error:
                    worst_body_error = body_error
                    worst_body_pair = pair
                    worst_body_usd_pos = usd_pos
                    worst_body_urdf_pos = urdf_pos
                    worst_body_pos = body_pos

        self.num_checks += 1
        print(
            f"[KinovaGripperJointFrameCheck] step={step_count:04d} {label:>11s} "
            f"joint_max={worst_joint_error:.3e} worst_joint={worst_joint_pair[0]}->{worst_joint_pair[1]} "
            f"link_max={worst_link_error:.3e} worst_link={worst_link_pair[0]}->{worst_link_pair[1]}"
        )
        print(
            f"  joint usd=[{_fmt_np_xyz(worst_joint_usd_pos)}] urdf=[{_fmt_np_xyz(worst_joint_urdf_pos)}]"
        )
        print(
            f"  link  usd=[{_fmt_np_xyz(worst_link_usd_pos)}] urdf=[{_fmt_np_xyz(worst_link_urdf_pos)}]"
        )
        if worst_body_pair is not None:
            print(
                f"  body  body_pos_w_max={worst_body_error:.3e} worst_body={worst_body_pair[1]} "
                f"isaac=[{_fmt_np_xyz(worst_body_pos)}] usd=[{_fmt_np_xyz(worst_body_usd_pos)}] "
                f"urdf=[{_fmt_np_xyz(worst_body_urdf_pos)}]"
            )

        fail_error = max(worst_joint_error, worst_link_error, worst_body_error)
        if fail_error > self.pos_tolerance:
            print(
                "[KinovaGripperJointFrameCheck] WARNING: USD/URDF Robotiq global joint/link "
                f"position mismatch max_pos_error={fail_error:.6g} m exceeds tolerance={self.pos_tolerance:.6g} m"
            )
            if self.fail_on_mismatch:
                raise RuntimeError(
                    "USD/URDF Robotiq global joint/link position mismatch: "
                    f"max_pos_error={fail_error:.6g} m, tolerance={self.pos_tolerance:.6g} m"
                )

    def print_summary(self) -> None:
        if self.num_checks <= 0:
            print("[KinovaGripperJointFrameCheck] no checks were run")
            return
        print("[KinovaGripperJointFrameCheck] max per-joint global position errors:")
        for pair in GRIPPER_JOINT_FRAME_PAIRS:
            error, usd_pos, urdf_pos = self.max_joint_by_pair[pair]
            print(
                f"  {pair[0]} -> {pair[1]}: max={error:.3e} m "
                f"usd=[{_fmt_np_xyz(usd_pos)}] urdf=[{_fmt_np_xyz(urdf_pos)}]"
            )
        print("[KinovaGripperJointFrameCheck] max per-link global position errors:")
        for usd_link, urdf_link, body_id in self.link_pairs:
            pair = (usd_link, urdf_link)
            error, usd_pos, urdf_pos = self.max_link_by_pair[pair]
            print(
                f"  {usd_link} -> {urdf_link}: max={error:.3e} m "
                f"usd=[{_fmt_np_xyz(usd_pos)}] urdf=[{_fmt_np_xyz(urdf_pos)}]"
            )
            if body_id is not None and pair in self.max_body_by_pair:
                body_error, body_pos, body_usd_pos, body_urdf_pos = self.max_body_by_pair[pair]
                print(
                    f"    Isaac body_pos_w vs URDF: max={body_error:.3e} m "
                    f"isaac=[{_fmt_np_xyz(body_pos)}] usd=[{_fmt_np_xyz(body_usd_pos)}] "
                    f"urdf=[{_fmt_np_xyz(body_urdf_pos)}]"
                )
        print(
            "[KinovaGripperJointFrameCheck] topology note: compared shared joint/link positions only; "
            f"USD-only={USD_ONLY_GRIPPER_JOINTS}; URDF-only={URDF_ONLY_GRIPPER_JOINTS + URDF_ONLY_GRIPPER_LINKS}"
        )


class GripperJointOnlyWorldChecker:
    def __init__(self, env_cfg: KinovaTableEnvLearnedCfg, pos_tolerance: float):
        self.pos_tolerance = float(pos_tolerance)
        self.urdf = _UrdfKinematics(env_cfg.mpc_config["urdf_path"])
        self.usd = _UsdKinematics(skip_joint_names=set(USD_ONLY_GRIPPER_JOINTS))
        missing_usd = [usd_name for usd_name, _ in GRIPPER_JOINT_FRAME_PAIRS if usd_name not in self.usd.joints]
        missing_urdf = [urdf_name for _, urdf_name in GRIPPER_JOINT_FRAME_PAIRS if urdf_name not in self.urdf.joints]
        if missing_usd or missing_urdf:
            raise RuntimeError(
                "Cannot compare gripper joint frames: "
                f"missing_usd={missing_usd}, missing_urdf={missing_urdf}"
            )
        self.rows: list[dict[str, object]] = []
        self.max_error = 0.0
        self.max_row: dict[str, object] | None = None
        print(
            "[KinovaArmPoseJointFrameCheck] enabled: global Robotiq joint-frame positions only "
            f"(gripper open, warn_tol={self.pos_tolerance:g} m)"
        )
        print(
            "[KinovaArmPoseJointFrameCheck] mapped joints: "
            + ", ".join(f"{usd}->{urdf}" for usd, urdf in GRIPPER_JOINT_FRAME_PAIRS)
        )

    def check(self, pose_name: str, robot: Articulation, joint_name_to_id: dict[str, int]) -> None:
        q_by_name = {
            name: float(robot.data.joint_pos[0, joint_id].detach().cpu())
            for name, joint_id in joint_name_to_id.items()
        }
        _, usd_joint_poses = self.usd.forward(q_by_name)
        _, urdf_joint_poses = self.urdf.forward(q_by_name)
        root_pose_w = _robot_root_pose_w(robot)

        pose_worst = None
        for usd_name, urdf_name in GRIPPER_JOINT_FRAME_PAIRS:
            usd_pos = (root_pose_w @ usd_joint_poses[usd_name])[:3, 3]
            urdf_pos = (root_pose_w @ urdf_joint_poses[urdf_name])[:3, 3]
            error = float(np.linalg.norm(usd_pos - urdf_pos))
            row = {
                "pose": pose_name,
                "usd_joint": usd_name,
                "urdf_joint": urdf_name,
                "error_m": error,
                "usd_x": float(usd_pos[0]),
                "usd_y": float(usd_pos[1]),
                "usd_z": float(usd_pos[2]),
                "urdf_x": float(urdf_pos[0]),
                "urdf_y": float(urdf_pos[1]),
                "urdf_z": float(urdf_pos[2]),
            }
            self.rows.append(row)
            if pose_worst is None or error > float(pose_worst["error_m"]):
                pose_worst = row
            if error > self.max_error:
                self.max_error = error
                self.max_row = row

        assert pose_worst is not None
        print(
            f"[KinovaArmPoseJointFrameCheck] pose={pose_name} "
            f"max={float(pose_worst['error_m']):.3e} "
            f"worst={pose_worst['usd_joint']}->{pose_worst['urdf_joint']} "
            f"usd=[{pose_worst['usd_x']:+.6f}, {pose_worst['usd_y']:+.6f}, {pose_worst['usd_z']:+.6f}] "
            f"urdf=[{pose_worst['urdf_x']:+.6f}, {pose_worst['urdf_y']:+.6f}, {pose_worst['urdf_z']:+.6f}]"
        )
        if float(pose_worst["error_m"]) > self.pos_tolerance:
            print(
                "[KinovaArmPoseJointFrameCheck] WARNING: pose max joint-frame position error "
                f"{float(pose_worst['error_m']):.6g} m exceeds tolerance {self.pos_tolerance:.6g} m"
            )

    def print_summary(self, csv_path: str | Path) -> None:
        if not self.rows:
            print("[KinovaArmPoseJointFrameCheck] no checks were run")
            return
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "pose",
            "usd_joint",
            "urdf_joint",
            "error_m",
            "usd_x",
            "usd_y",
            "usd_z",
            "urdf_x",
            "urdf_y",
            "urdf_z",
        ]
        with csv_path.open("w", encoding="utf-8") as f:
            f.write(",".join(fields) + "\n")
            for row in self.rows:
                f.write(
                    ",".join(
                        str(row[field]) if isinstance(row[field], str) else f"{float(row[field]):.12g}"
                        for field in fields
                    )
                    + "\n"
                )
        assert self.max_row is not None
        print(
            "[KinovaArmPoseJointFrameCheck] overall max: "
            f"{self.max_error:.3e} m at pose={self.max_row['pose']} "
            f"joint={self.max_row['usd_joint']}->{self.max_row['urdf_joint']}"
        )
        print(f"[KinovaArmPoseJointFrameCheck] wrote CSV: {csv_path}")


def _open_gripper_state(joint_pos: torch.Tensor, joint_name_to_id: dict[str, int], gripper_joint_names: list[str]) -> None:
    for name in gripper_joint_names:
        joint_pos[:, joint_name_to_id[name]] = 0.0


def _arm_pose_samples(default_arm_pose: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
    device = default_arm_pose.device
    dtype = default_arm_pose.dtype
    offsets = [
        ("default", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ("yaw_left", [0.45, 0.10, -0.25, 0.20, 0.20, -0.15, 0.10]),
        ("yaw_right", [-0.45, 0.10, 0.25, -0.20, -0.20, 0.15, -0.10]),
        ("elbow_up", [0.15, 0.35, -0.10, -0.35, 0.25, 0.25, -0.20]),
        ("elbow_down", [-0.15, -0.35, 0.10, 0.35, -0.25, -0.25, 0.20]),
        ("wrist_sweep", [0.20, -0.10, 0.35, -0.15, 0.45, 0.35, -0.45]),
    ]
    return [
        (name, default_arm_pose + torch.tensor(delta, device=device, dtype=dtype))
        for name, delta in offsets
    ]


class FrameConsistencyChecker:
    def __init__(
        self,
        env_cfg: KinovaTableEnvLearnedCfg,
        device: str,
        sim_dt: float,
        pos_tolerance: float,
        rot_tolerance: float,
        check_every: int,
    ):
        self.device = device
        self.pos_tolerance = float(pos_tolerance)
        self.rot_tolerance = float(rot_tolerance)
        self.check_every = max(1, int(check_every))
        self.num_checks = 0
        self.max_sun_program_evaluator_pos_error = 0.0
        self.max_sun_program_executor_api_pos_error = 0.0
        self.max_sun_program_executor_fk_pos_error = 0.0
        self.max_sun_program_evaluator_rot_error = 0.0
        self.max_sun_program_executor_api_rot_error = 0.0
        self.max_sun_program_executor_fk_rot_error = 0.0

        cfg_sun_program_evaluator = dict(env_cfg.mpc_config)
        cfg_sun_program_evaluator["device"] = device
        cfg_sun_program_evaluator["num_envs"] = 1
        cfg_sun_program_evaluator.setdefault("use_per_env_weight", False)
        cfg_sun_program_evaluator["load_program"] = None
        self.sun_program_evaluator = SUNProgramEvaluator(OmegaConf.create(cfg_sun_program_evaluator), auto_load_program=False)

        repo_root = Path(__file__).resolve().parents[5]
        cfg_sun_program_executor = {
            "controller_name": "KinovaIKController",
            "device": device,
            "gripper_opened": env_cfg.mpc_config.get("gripper_opened", [0.0, 0.0]),
            "gripper_closed": env_cfg.mpc_config.get(
                "gripper_closed", [0.8, 0.8]
            ),
            "gripper_max_force": env_cfg.mpc_config.get("gripper_max_force", 1.0),
            "gripper_effort_contact_velocity_threshold": env_cfg.mpc_config.get(
                "gripper_effort_contact_velocity_threshold", 0.15
            ),
            "kinova_gripper_effort_contact_openness_slope": env_cfg.mpc_config.get(
                "kinova_gripper_effort_contact_openness_slope", 12.0
            ),
            "kinova_gripper_effort_contact_openness_intercept": env_cfg.mpc_config.get(
                "kinova_gripper_effort_contact_openness_intercept", 0.0
            ),
            "robot_gripper_name": env_cfg.mpc_config.get("robot_gripper_name", "kinova_robotiq"),
            "has_finger_contact_sensors": False,
            "controller": {
                "urdf_path": env_cfg.mpc_config["urdf_path"],
                "ee_frame": env_cfg.mpc_config.get("ee_frame", FRAME_NAME),
                "horizon_length": 4,
                "time_step": sim_dt,
            },
            "load_program": None,
            "control_output_mode": "absolute_ee",
            "agent": {
                "id": "agent",
                "device": device,
                "task": "Kinova gripper frame consistency",
                "prompt_dir": str(repo_root / "assets/prompts/Manipulation"),
                "initial_residual": env_cfg.mpc_config["initial_residual"],
                "initial_predicate": env_cfg.mpc_config["initial_predicate"],
                "model_version": "offline",
                "load_path": None,
                "reuse_program": True,
                "max_correction_steps": 1,
            },
        }
        self.sun_program_executor = SUNProgramExecutor(OmegaConf.create(cfg_sun_program_executor))
        self._sun_program_executor_frame_id = self.sun_program_executor.active_controller.model.getFrameId(FRAME_NAME)

        print(
            "[KinovaFrameCheck] enabled: env FrameTransformer vs SUNProgramEvaluator vs SUNProgramExecutor API vs SUNProgramExecutor FK "
            f"(pos_tol={self.pos_tolerance:g} m, rot_tol={self.rot_tolerance:g} rad)"
        )

    def _frame_sensor_msg(self, fingertip_frame: FrameTransformer) -> tuple[dict, dict, torch.Tensor, torch.Tensor]:
        data = fingertip_frame.data
        env_pos_b = data.target_pos_source[:, 0].detach().clone()
        env_quat_b = data.target_quat_source[:, 0].detach().clone()
        env_pos_w = data.target_pos_w[:, 0].detach().clone()
        env_quat_w = data.target_quat_w[:, 0].detach().clone()
        base_pos_w = data.source_pos_w.detach().clone()
        base_quat_w = data.source_quat_w.detach().clone()
        sensors = {
            FRAME_NAME: [
                env_pos_w,
                env_quat_w,
                FRAME_NAME,
                ("Kinova Robotiq virtual fingertip/tool frame", "/Visuals/panda_fingertip_frame"),
            ],
            "agent": [
                base_pos_w,
                base_quat_w,
                "robot",
                ("Kinova base_link frame", "/World/Robot/base_link"),
            ],
        }
        return {}, sensors, env_pos_b, env_quat_b

    def _sun_program_executor_fk_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        placement = self.sun_program_executor.active_controller.data.oMf[self._sun_program_executor_frame_id]
        pos = torch.as_tensor(
            placement.translation,
            device=self.sun_program_executor.controller_api_wrapper._gripper_opened.device,
            dtype=torch.float32,
        ).view(1, 3)
        rot = torch.as_tensor(
            placement.rotation,
            device=pos.device,
            dtype=torch.float32,
        ).view(1, 3, 3)
        quat = quat_from_matrix(rot)
        return pos, quat

    def update(
        self,
        step_count: int,
        label: str,
        robot: Articulation,
        fingertip_frame: FrameTransformer,
        controlled_ids: list[int],
    ) -> None:
        odom, sensors, env_pos_b, env_quat_b = self._frame_sensor_msg(fingertip_frame)
        joint_state = _logical_joint_state(robot, controlled_ids)

        self.sun_program_evaluator.callbackOdom(odom, sensors)
        self.sun_program_evaluator.callbackJointState(joint_state)
        self.sun_program_executor.callbackOdom(odom, sensors)
        self.sun_program_executor.callbackJointState(joint_state)

        sun_program_evaluator_api = self.sun_program_evaluator.controller_api_wrapper
        sun_program_executor_api = self.sun_program_executor.controller_api_wrapper
        sun_program_evaluator_pos = sun_program_evaluator_api.lookup_robot_frame_position(FRAME_NAME)
        sun_program_evaluator_quat = sun_program_evaluator_api.lookup_robot_frame_rotation(FRAME_NAME)
        sun_program_executor_pos = sun_program_executor_api.lookup_robot_frame_position(FRAME_NAME)
        sun_program_executor_quat = sun_program_executor_api.lookup_robot_frame_rotation(FRAME_NAME)
        sun_program_executor_fk_pos, sun_program_executor_fk_quat = self._sun_program_executor_fk_pose()

        sun_program_evaluator_pos_error = torch.linalg.norm(sun_program_evaluator_pos - env_pos_b, dim=-1).max()
        sun_program_executor_api_pos_error = torch.linalg.norm(sun_program_executor_pos - env_pos_b, dim=-1).max()
        sun_program_executor_fk_pos_error = torch.linalg.norm(sun_program_executor_fk_pos - env_pos_b, dim=-1).max()
        sun_program_evaluator_rot_error = _quat_angle_error(env_quat_b, sun_program_evaluator_quat).max()
        sun_program_executor_api_rot_error = _quat_angle_error(env_quat_b, sun_program_executor_quat).max()
        sun_program_executor_fk_rot_error = _quat_angle_error(env_quat_b, sun_program_executor_fk_quat).max()

        errors = {
            "sun_program_evaluator_pos": float(sun_program_evaluator_pos_error.detach().cpu()),
            "sun_program_executor_api_pos": float(sun_program_executor_api_pos_error.detach().cpu()),
            "sun_program_executor_fk_pos": float(sun_program_executor_fk_pos_error.detach().cpu()),
            "sun_program_evaluator_rot": float(sun_program_evaluator_rot_error.detach().cpu()),
            "sun_program_executor_api_rot": float(sun_program_executor_api_rot_error.detach().cpu()),
            "sun_program_executor_fk_rot": float(sun_program_executor_fk_rot_error.detach().cpu()),
        }
        self.max_sun_program_evaluator_pos_error = max(self.max_sun_program_evaluator_pos_error, errors["sun_program_evaluator_pos"])
        self.max_sun_program_executor_api_pos_error = max(self.max_sun_program_executor_api_pos_error, errors["sun_program_executor_api_pos"])
        self.max_sun_program_executor_fk_pos_error = max(self.max_sun_program_executor_fk_pos_error, errors["sun_program_executor_fk_pos"])
        self.max_sun_program_evaluator_rot_error = max(self.max_sun_program_evaluator_rot_error, errors["sun_program_evaluator_rot"])
        self.max_sun_program_executor_api_rot_error = max(self.max_sun_program_executor_api_rot_error, errors["sun_program_executor_api_rot"])
        self.max_sun_program_executor_fk_rot_error = max(self.max_sun_program_executor_fk_rot_error, errors["sun_program_executor_fk_rot"])
        self.num_checks += 1

        fail_pos = max(errors["sun_program_evaluator_pos"], errors["sun_program_executor_api_pos"], errors["sun_program_executor_fk_pos"])
        fail_rot = max(errors["sun_program_evaluator_rot"], errors["sun_program_executor_api_rot"], errors["sun_program_executor_fk_rot"])
        should_print = (step_count % self.check_every == 0) or fail_pos > self.pos_tolerance or fail_rot > self.rot_tolerance
        if should_print:
            print(
                f"[KinovaFrameCheck] step={step_count:04d} {label:>11s} "
                f"env_b=[{_fmt_xyz(env_pos_b[0])}] "
                f"sat={errors['sun_program_evaluator_pos']:.3e}/{errors['sun_program_evaluator_rot']:.3e} "
                f"sun_program_executor_api={errors['sun_program_executor_api_pos']:.3e}/{errors['sun_program_executor_api_rot']:.3e} "
                f"sun_program_executor_fk={errors['sun_program_executor_fk_pos']:.3e}/{errors['sun_program_executor_fk_rot']:.3e}"
            )

        if fail_pos > self.pos_tolerance or fail_rot > self.rot_tolerance:
            raise RuntimeError(
                "panda_fingertip frame mismatch: "
                f"max_pos_error={fail_pos:.6g} m, max_rot_error={fail_rot:.6g} rad, "
                f"tolerances=({self.pos_tolerance:.6g} m, {self.rot_tolerance:.6g} rad)"
            )

    def print_summary(self) -> None:
        if self.num_checks <= 0:
            print("[KinovaFrameCheck] no checks were run")
            return
        print(
            "[KinovaFrameCheck] max errors: "
            f"sun_program_evaluator={self.max_sun_program_evaluator_pos_error:.3e}m/{self.max_sun_program_evaluator_rot_error:.3e}rad, "
            f"sun_program_executor_api={self.max_sun_program_executor_api_pos_error:.3e}m/{self.max_sun_program_executor_api_rot_error:.3e}rad, "
            f"sun_program_executor_fk={self.max_sun_program_executor_fk_pos_error:.3e}m/{self.max_sun_program_executor_fk_rot_error:.3e}rad"
        )


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(dt=float(args_cli.dt), device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.0, -1.1, 0.75], target=[0.22, 0.0, 0.30])

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.78, 0.78, 0.78))
    light_cfg.func("/World/Light", light_cfg)

    env_cfg = KinovaTableEnvLearnedCfg()
    robot_cfg = env_cfg.robot.copy().replace(prim_path="/World/Robot")
    if args_cli.asset_mode in PACKAGED_ASSET_MODES:
        _configure_packaged_isaaclab_asset(robot_cfg)
    elif args_cli.usd_override:
        robot_cfg.spawn.usd_path = str(Path(args_cli.usd_override).expanduser())
    print(f"[KinovaGripperVisual] asset_mode={args_cli.asset_mode}")
    print(f"[KinovaGripperVisual] usd={robot_cfg.spawn.usd_path}")
    robot = Articulation(cfg=robot_cfg)
    fingertip_frame = _make_fingertip_frame(robot_cfg)

    sim.reset()
    fingertip_frame.reset()

    root_state = robot.data.default_root_state.clone()
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = torch.zeros_like(robot.data.default_joint_vel)
    joint_name_to_id = {name: i for i, name in enumerate(robot.joint_names)}
    active_gripper_joint_names = _active_gripper_joint_names()
    missing_gripper_joints = [name for name in active_gripper_joint_names if name not in joint_name_to_id]
    if missing_gripper_joints:
        raise RuntimeError(f"Missing expected gripper joints for {args_cli.asset_mode}: {missing_gripper_joints}")
    for name in active_gripper_joint_names:
        joint_pos[:, joint_name_to_id[name]] = 0.0
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()
    robot.update(sim.get_physics_dt())
    fingertip_frame.update(sim.get_physics_dt())

    arm_ids, arm_names = robot.find_joints(list(robot_cfg.arm_actuated_joints))
    logical_gripper_joint_names = _active_logical_gripper_joint_names()
    missing_logical_joints = [name for name in logical_gripper_joint_names if name not in joint_name_to_id]
    if missing_logical_joints:
        raise RuntimeError(f"Missing logical gripper joints for {args_cli.asset_mode}: {missing_logical_joints}")
    logical_gripper_ids = [joint_name_to_id[name] for name in logical_gripper_joint_names]

    driver_joint_names = _active_driver_joint_names()
    missing_driver_joints = [name for name in driver_joint_names if name not in joint_name_to_id]
    if missing_driver_joints:
        raise RuntimeError(f"Missing driver joints for {args_cli.asset_mode}: {missing_driver_joints}")
    driver_ids = [joint_name_to_id[name] for name in driver_joint_names]
    driver_names = list(driver_joint_names)
    controlled_ids = [int(i) for i in arm_ids] + [int(i) for i in driver_ids]
    logical_state_ids = [int(i) for i in arm_ids] + [int(i) for i in logical_gripper_ids]
    targets = robot.data.default_joint_pos[:, controlled_ids].clone()
    gripper_target_start = len(controlled_ids) - len(driver_ids)
    targets[:, gripper_target_start:] = _active_gripper_target(0.0, targets[:, gripper_target_start:]).view(1, -1)

    if bool(args_cli.arm_pose_joint_frame_check):
        pose_checker = GripperJointOnlyWorldChecker(env_cfg, args_cli.gripper_joint_frame_pos_tolerance)
        arm_ids_list = [int(i) for i in arm_ids]
        default_arm_pose = robot.data.default_joint_pos[0, arm_ids_list].detach().clone()
        for pose_name, arm_pose in _arm_pose_samples(default_arm_pose):
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = torch.zeros_like(robot.data.default_joint_vel)
            joint_pos[:, arm_ids_list] = arm_pose.view(1, -1)
            _open_gripper_state(joint_pos, joint_name_to_id, active_gripper_joint_names)
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
            robot.update(sim.get_physics_dt())
            fingertip_frame.update(sim.get_physics_dt())
            pose_checker.check(pose_name, robot, joint_name_to_id)
        pose_checker.print_summary(str(resolve_artifact_path(args_cli.arm_pose_joint_frame_csv, "Diagnostic CSV")))
        return

    frame_check_enabled = bool(args_cli.frame_check)
    gripper_joint_frame_check_enabled = bool(args_cli.gripper_joint_frame_check)
    if _uses_six_joint_gripper_asset():
        if frame_check_enabled:
            print("[KinovaGripperVisual] disabling SUNProgramEvaluator/SUNProgramExecutor frame consistency check for six-joint gripper asset")
        if gripper_joint_frame_check_enabled:
            print("[KinovaGripperVisual] disabling old USD/URDF gripper joint-frame check for six-joint gripper asset")
        frame_check_enabled = False
        gripper_joint_frame_check_enabled = False
    elif gripper_joint_frame_check_enabled:
        print("[KinovaGripperVisual] disabling old USD/URDF gripper joint-frame check for official single-master gripper asset")
        gripper_joint_frame_check_enabled = False

    frame_checker = None
    if frame_check_enabled:
        frame_checker = FrameConsistencyChecker(
            env_cfg,
            args_cli.device,
            sim.get_physics_dt(),
            args_cli.frame_pos_tolerance,
            args_cli.frame_rot_tolerance,
            args_cli.frame_check_every,
        )

    gripper_joint_checker = None
    if gripper_joint_frame_check_enabled:
        gripper_joint_checker = GripperJointFrameChecker(
            env_cfg,
            robot,
            args_cli.gripper_joint_frame_pos_tolerance,
            args_cli.gripper_joint_frame_check_every,
            args_cli.gripper_joint_frame_fail_on_mismatch,
        )

    close_target = _active_gripper_target(0.8, targets[:, gripper_target_start:])
    open_target = _active_gripper_target(0.0, targets[:, gripper_target_start:])
    close_target_text = ", ".join(
        f"{name}={float(value):+.6f}" for name, value in zip(driver_names, close_target.detach().cpu().tolist())
    )
    print(f"[KinovaGripperVisual] controlled gripper joints={driver_names}")
    print("[KinovaGripperVisual] fingertip frame marker=/Visuals/panda_fingertip_frame")
    print(f"[KinovaGripperVisual] close target: {close_target_text}")
    print("[KinovaGripperVisual] close the Isaac window to stop")

    sim_dt = sim.get_physics_dt()
    print_every = max(1, int(args_cli.print_every))
    step_count = 0
    cycle = 0

    def step_to(gripper_target: torch.Tensor, label: str) -> None:
        nonlocal step_count
        targets[:, gripper_target_start:] = gripper_target.to(device=targets.device, dtype=targets.dtype).view(1, -1)
        robot.set_joint_position_target(targets, joint_ids=controlled_ids)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)
        fingertip_frame.update(sim_dt)
        if frame_checker is not None:
            frame_checker.update(step_count, label, robot, fingertip_frame, logical_state_ids)
        if gripper_joint_checker is not None:
            gripper_joint_checker.update(step_count, label, robot, joint_name_to_id)
        if step_count % print_every == 0:
            measured = robot.data.joint_pos[:, driver_ids][0]
            print(f"[KinovaGripperVisual] step={step_count:04d} {label:>11s} measured=[{_fmt_pair(measured)}]")
        step_count += 1

    try:
        while simulation_app.is_running() and (int(args_cli.cycles) == 0 or cycle < int(args_cli.cycles)):
            for _ in range(int(args_cli.settle_steps)):
                if not simulation_app.is_running():
                    break
                step_to(open_target, "open-hold")

            for i in range(1, int(args_cli.close_steps) + 1):
                if not simulation_app.is_running():
                    break
                alpha = i / max(1, int(args_cli.close_steps))
                step_to((1.0 - alpha) * open_target + alpha * close_target, "closing")

            for _ in range(int(args_cli.hold_steps)):
                if not simulation_app.is_running():
                    break
                step_to(close_target, "closed-hold")

            cycle += 1
            if int(args_cli.cycles) == 0 and simulation_app.is_running():
                for i in range(1, int(args_cli.open_steps) + 1):
                    if not simulation_app.is_running():
                        break
                    alpha = i / max(1, int(args_cli.open_steps))
                    step_to((1.0 - alpha) * close_target + alpha * open_target, "opening")
    finally:
        if frame_checker is not None:
            frame_checker.print_summary()
        if gripper_joint_checker is not None:
            gripper_joint_checker.print_summary()

    final = robot.data.joint_pos[:, driver_ids][0]
    print(f"[KinovaGripperVisual] final controlled gripper joints=[{_fmt_pair(final)}]")
    print("[KinovaGripperVisual] final gripper joints:")
    for name in active_gripper_joint_names:
        joint_id = joint_name_to_id[name]
        pos = float(robot.data.joint_pos[0, joint_id].detach().cpu())
        target = float(robot.data.joint_pos_target[0, joint_id].detach().cpu())
        torque = float(robot.data.applied_torque[0, joint_id].detach().cpu())
        print(f"  {name}: pos={pos:.9f}, target={target:.9f}, applied_torque={torque:.9f}")


if __name__ == "__main__":
    main()
