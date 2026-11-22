#!/usr/bin/env python3
"""Create a Kinova Robotiq USD copy where both outer knuckles are explicit drivers."""

from __future__ import annotations

import shutil
from pathlib import Path

from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": True, "enable_cameras": False})
simulation_app = app_launcher.app

from pxr import PhysxSchema, Sdf, Usd, UsdPhysics  # noqa: E402


LEARNED_DIR = Path(__file__).resolve().parents[1]
ASSET_DIR = LEARNED_DIR / "assets/kinova/usd"
BASE_NAME = "gen3_7dof_vision_robotiq_2f_85"
VARIANT_NAME = "gen3_7dof_vision_robotiq_2f_85_two_driver"
ROBOT_PRIM = "/GEN3_7DOF_VISION_ROBOTIQ_2F_85"
RIGHT_OUTER_JOINT = f"{ROBOT_PRIM}/Robotiq_2F_85/right_outer_knuckle_joint"


def remove_authored_mimic_properties(layer: Sdf.Layer) -> None:
    prim_spec = layer.GetPrimAtPath(RIGHT_OUTER_JOINT)
    if prim_spec is None:
        return
    for prop_spec in list(prim_spec.properties):
        if prop_spec.name.startswith("physxMimicJoint:rotZ:"):
            prim_spec.RemoveProperty(prop_spec)


def clear_right_outer_mimic(stage_path: Path) -> None:
    stage = Usd.Stage.Open(str(stage_path))
    joint_prim = stage.GetPrimAtPath(RIGHT_OUTER_JOINT)
    if not joint_prim:
        raise RuntimeError(f"Missing joint prim in {stage_path}: {RIGHT_OUTER_JOINT}")

    remove_authored_mimic_properties(stage.GetRootLayer())
    for prop in list(joint_prim.GetProperties()):
        prop_name = prop.GetName()
        if prop_name.startswith("physxMimicJoint:rotZ:"):
            joint_prim.RemoveProperty(prop_name)
    joint_prim.RemoveAPI(PhysxSchema.PhysxMimicJointAPI, "rotZ")

    drive = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
    drive.CreateTargetPositionAttr(0.0)
    drive.CreateStiffnessAttr(0.17000000178813934)
    drive.CreateDampingAttr(0.00019999999494757503)
    drive.CreateMaxForceAttr(16.5)
    stage.GetRootLayer().Save()


def main() -> None:
    base_root = ASSET_DIR / f"{BASE_NAME}.usd"
    base_payload_dir = ASSET_DIR / BASE_NAME
    variant_root = ASSET_DIR / f"{VARIANT_NAME}.usd"
    variant_payload_dir = ASSET_DIR / VARIANT_NAME

    if not variant_root.exists():
        shutil.copy2(base_root, variant_root)
    if not variant_payload_dir.exists():
        shutil.copytree(base_payload_dir, variant_payload_dir)

    stage = Usd.Stage.Open(str(variant_root))
    robot_prim = stage.GetPrimAtPath(ROBOT_PRIM)
    refs = robot_prim.GetReferences()
    refs.ClearReferences()
    refs.AddReference(f"./{VARIANT_NAME}/gen3_7dof_vision_robotiq_2f_85.usd")
    stage.GetRootLayer().Save()

    clear_right_outer_mimic(variant_root)
    payload = variant_payload_dir / "gen3_7dof_vision_robotiq_2f_85.usd"
    if payload.exists():
        clear_right_outer_mimic(payload)
    print(variant_root)


if __name__ == "__main__":
    main()
    simulation_app.close()
