#!/usr/bin/env python3
"""Create copied Robotiq USD variants for internal mimic-sign diagnostics."""

from __future__ import annotations

import shutil
from pathlib import Path

from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": True, "enable_cameras": False})
simulation_app = app_launcher.app

from pxr import PhysxSchema, Sdf, Usd, UsdPhysics  # noqa: E402


LEARNED_DIR = Path(__file__).resolve().parents[1]
ASSET_DIR = LEARNED_DIR / "assets/kinova/usd"
BASE_NAME = "gen3_7dof_vision_robotiq_2f_85_end_effector_attach"
BASE_ROOT = ASSET_DIR / f"{BASE_NAME}.usd"
BASE_PAYLOAD_DIR = ASSET_DIR / BASE_NAME
ROBOT_PRIM = "/GEN3_7DOF_VISION_ROBOTIQ_2F_85"
JOINT_PATH = f"{ROBOT_PRIM}/Robotiq_2F_85/right_outer_knuckle_joint"
FINGER_JOINT_PATH = f"{ROBOT_PRIM}/Robotiq_2F_85/finger_joint"


def copy_variant(name: str) -> Path:
    root_path = ASSET_DIR / f"{name}.usd"
    payload_dir = ASSET_DIR / name
    if not root_path.exists():
        shutil.copy2(BASE_ROOT, root_path)
    if not payload_dir.exists():
        shutil.copytree(BASE_PAYLOAD_DIR, payload_dir)

    stage = Usd.Stage.Open(str(root_path))
    robot_prim = stage.GetPrimAtPath(ROBOT_PRIM)
    refs = robot_prim.GetReferences()
    refs.ClearReferences()
    refs.AddReference(f"./{name}/gen3_7dof_vision_robotiq_2f_85.usd")
    stage.GetRootLayer().Save()
    return root_path


def set_right_outer_knuckle(root_path: Path, gearing: float, lower_deg: float, upper_deg: float) -> None:
    stage = Usd.Stage.Open(str(root_path))
    joint_prim = stage.GetPrimAtPath(JOINT_PATH)
    if not joint_prim:
        raise RuntimeError(f"Missing joint prim: {JOINT_PATH}")
    joint = UsdPhysics.RevoluteJoint(joint_prim)
    joint.CreateLowerLimitAttr().Set(float(lower_deg))
    joint.CreateUpperLimitAttr().Set(float(upper_deg))
    mimic = PhysxSchema.PhysxMimicJointAPI.Apply(joint_prim, "rotZ")
    mimic.CreateReferenceJointRel().SetTargets([Sdf.Path(FINGER_JOINT_PATH)])
    mimic.CreateGearingAttr().Set(float(gearing))
    mimic.CreateOffsetAttr().Set(0.0)
    stage.GetRootLayer().Save()


def main() -> None:
    neg = copy_variant("gen3_7dof_vision_robotiq_2f_85_end_effector_attach_mimic_neg_limits")
    set_right_outer_knuckle(neg, gearing=-1.0, lower_deg=-75.0, upper_deg=0.0)
    print(neg)

    pos = copy_variant("gen3_7dof_vision_robotiq_2f_85_end_effector_attach_mimic_pos")
    set_right_outer_knuckle(pos, gearing=1.0, lower_deg=0.0, upper_deg=75.0)
    print(pos)


if __name__ == "__main__":
    main()
    simulation_app.close()
