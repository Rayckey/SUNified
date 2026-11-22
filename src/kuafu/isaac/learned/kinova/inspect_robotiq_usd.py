#!/usr/bin/env python3
"""Print composed Robotiq USD joint metadata for debugging."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("usd_path")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    args.enable_cameras = False
    return args


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import PhysxSchema, Usd, UsdPhysics  # noqa: E402


def attr_value(attr):
    if not attr:
        return None
    return attr.Get()


def main() -> None:
    stage = Usd.Stage.Open(args_cli.usd_path)
    print(f"default={stage.GetDefaultPrim()}")
    first_prims = []
    for i, p in zip(range(80), stage.Traverse()):
        first_prims.append((str(p.GetPath()), str(p.GetTypeName())))
    print(f"first_prims={first_prims}")
    print(f"layers={[l.identifier for l in stage.GetLayerStack()]}" )
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if "Robotiq_2F_85" not in path:
            continue
        type_name = prim.GetTypeName()
        if "Joint" not in str(type_name) and "joint" not in path:
            continue
        print(path)
        print(f"  type={type_name}")
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        joint = UsdPhysics.RevoluteJoint(prim)
        print(f"  body0={joint.GetBody0Rel().GetTargets()}")
        print(f"  body1={joint.GetBody1Rel().GetTargets()}")
        print(f"  axis={attr_value(joint.GetAxisAttr())}")
        print(f"  lower={attr_value(joint.GetLowerLimitAttr())}")
        print(f"  upper={attr_value(joint.GetUpperLimitAttr())}")
        for api in prim.GetAppliedSchemas():
            if not api.startswith("PhysxMimicJointAPI:"):
                continue
            axis_name = api.split(":", 1)[1]
            mimic = PhysxSchema.PhysxMimicJointAPI.Get(prim, axis_name)
            print(f"  mimic[{axis_name}].ref={mimic.GetReferenceJointRel().GetTargets()}")
            print(f"  mimic[{axis_name}].gear={attr_value(mimic.GetGearingAttr())}")
            print(f"  mimic[{axis_name}].offset={attr_value(mimic.GetOffsetAttr())}")
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if drive:
            print(f"  drive.target={attr_value(drive.GetTargetPositionAttr())}")
            print(f"  drive.stiffness={attr_value(drive.GetStiffnessAttr())}")
            print(f"  drive.damping={attr_value(drive.GetDampingAttr())}")
            print(f"  drive.maxForce={attr_value(drive.GetMaxForceAttr())}")


if __name__ == "__main__":
    main()
    simulation_app.close()
