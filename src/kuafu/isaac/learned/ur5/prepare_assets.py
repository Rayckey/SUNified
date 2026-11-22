# SPDX-License-Identifier: Apache-2.0
"""Rebuild the bundled UR5/Robotiq USD from its local ROS-Industrial URDF sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET

from kuafu.paths import ROBOT_ASSET_ROOT, external_artifact_root
from .assets import ARM_JOINTS, GRIPPER_ROOT, TCP_POSITION, TCP_QUATERNION


def _controller_urdf(stage, path: Path) -> None:
    """Export the six arm joints with both USD joint frames preserved."""
    import numpy as np
    from pxr import UsdPhysics
    from scipy.spatial.transform import Rotation

    def vector(values):
        return " ".join(f"{float(value):.12g}" for value in values)

    def transform(position, quaternion, inverse=False):
        rotation = Rotation.from_quat([*quaternion.GetImaginary(), quaternion.GetReal()])
        translation = np.asarray(position, dtype=float)
        if inverse:
            rotation = rotation.inv()
            translation = -rotation.apply(translation)
        return {"xyz": vector(translation), "rpy": vector(rotation.as_euler("xyz"))}

    robot = ET.Element("robot", name="kuafu_ur5_robotiq")
    ET.SubElement(robot, "link", name="base_link")
    joints = {prim.GetName(): UsdPhysics.RevoluteJoint(prim) for prim in stage.Traverse()
              if prim.IsA(UsdPhysics.RevoluteJoint)}
    axes = {"X": "1 0 0", "Y": "0 1 0", "Z": "0 0 1"}
    for index, name in enumerate(ARM_JOINTS):
        joint = joints[name]
        parent, child = joint.GetBody0Rel().GetTargets()[0].name, joint.GetBody1Rel().GetTargets()[0].name
        # An intermediate link retains nonzero child-side origins and arbitrary
        # USD joint axes; collapsing these frames can silently change IK.
        frame = name + "_frame"
        ET.SubElement(robot, "link", name=frame)
        ET.SubElement(robot, "link", name=child)
        element = ET.SubElement(robot, "joint", name=name, type="revolute")
        ET.SubElement(element, "parent", link=parent)
        ET.SubElement(element, "child", link=frame)
        ET.SubElement(element, "origin", **transform(joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get()))
        ET.SubElement(element, "axis", xyz=axes[joint.GetAxisAttr().Get()])
        ET.SubElement(element, "limit", lower=str(math.radians(joint.GetLowerLimitAttr().Get())),
                      upper=str(math.radians(joint.GetUpperLimitAttr().Get())),
                      effort="150" if index < 3 else "28", velocity=str(math.pi))
        fixed = ET.SubElement(robot, "joint", name=name + "_child_frame", type="fixed")
        ET.SubElement(fixed, "parent", link=frame)
        ET.SubElement(fixed, "child", link=child)
        ET.SubElement(fixed, "origin", **transform(joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get(), True))
    for alias, parent, xyz in (("panda_link4", "forearm_link", (0, 0, 0)),
                               ("panda_link7", "wrist_3_link", (0, 0, 0)),
                               ("panda_fingertip", "wrist_3_link", TCP_POSITION)):
        ET.SubElement(robot, "link", name=alias)
        fixed = ET.SubElement(robot, "joint", name=alias + "_fixed", type="fixed")
        ET.SubElement(fixed, "parent", link=parent)
        ET.SubElement(fixed, "child", link=alias)
        ET.SubElement(fixed, "origin", xyz=vector(xyz), rpy="0 0 0")
    ET.indent(robot)
    ET.ElementTree(robot).write(path, encoding="utf-8", xml_declaration=True)


def prepare(output: Path, source: Path | None = None) -> dict:
    """Convert local sources, attach the gripper, and export a portable USD."""
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

    source = (source or ROBOT_ASSET_ROOT / "ur5_robotiq/source").resolve()
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in ("arm", "gripper"):
        if not (source / f"{name}.urdf").is_file():
            raise FileNotFoundError(f"Missing bundled source: {source / (name + '.urdf')}")
    with tempfile.TemporaryDirectory(prefix="kuafu-ur5-build-") as temporary:
        work = Path(temporary)
        converted = {}
        for name in ("arm", "gripper"):
            converter = UrdfConverter(UrdfConverterCfg(
                asset_path=str(source / f"{name}.urdf"), usd_dir=str(work / name),
                usd_file_name=f"{name}.usd", fix_base=name == "arm", merge_fixed_joints=True,
                make_instanceable=False, force_usd_conversion=True,
                joint_drive=UrdfConverterCfg.JointDriveCfg(
                    gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)),
            ))
            converted[name] = converter.usd_path
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        root = UsdGeom.Xform.Define(stage, "/Robot").GetPrim()
        root.GetReferences().AddReference(converted["arm"])
        stage.SetDefaultPrim(root)
        wrist = stage.GetPrimAtPath("/Robot/wrist_3_link")
        if not wrist:
            raise ValueError("URDF conversion omitted wrist_3_link")
        mount = UsdGeom.Xform.Define(stage, "/Robot/Gripper")
        mount.AddTransformOp().Set(UsdGeom.XformCache().GetLocalToWorldTransform(wrist))
        gripper = UsdGeom.Xform.Define(stage, f"/Robot/{GRIPPER_ROOT}").GetPrim()
        gripper.GetReferences().AddReference(converted["gripper"])
        for prim in Usd.PrimRange(gripper):
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        # Scenes bind contact materials to fingertip collisions after spawning.
        # The importer can instance these meshes even with make_instanceable=False.
        for link_name in ("left_inner_finger", "right_inner_finger"):
            collisions = stage.GetPrimAtPath(f"/Robot/{GRIPPER_ROOT}/{link_name}/collisions")
            instances = [prim for prim in Usd.PrimRange(collisions) if prim.IsInstance()]
            while instances:
                for prim in instances:
                    prim.SetInstanceable(False)
                instances = [prim for prim in Usd.PrimRange(collisions) if prim.IsInstance()]
        gripper_joints = {prim.GetName(): prim for prim in Usd.PrimRange(gripper)
                          if prim.IsA(UsdPhysics.RevoluteJoint)}
        for definition in ET.parse(source / "gripper.urdf").findall("joint"):
            coupling = definition.find("mimic")
            if coupling is None:
                continue
            follower = gripper_joints[definition.get("name")]
            axis = UsdPhysics.RevoluteJoint(follower).GetAxisAttr().Get()
            mimic = PhysxSchema.PhysxMimicJointAPI.Apply(follower, "rot" + axis)
            mimic.CreateReferenceJointRel().SetTargets([gripper_joints[coupling.get("joint")].GetPath()])
            # PhysX uses follower + gearing * reference + offset = 0.
            mimic.CreateGearingAttr().Set(-float(coupling.get("multiplier", "1")))
            mimic.CreateOffsetAttr().Set(-math.degrees(float(coupling.get("offset", "0"))))
        joint = UsdPhysics.FixedJoint.Define(stage, "/Robot/robotiq_mount")
        joint.CreateBody0Rel().SetTargets([wrist.GetPath()])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(f"/Robot/{GRIPPER_ROOT}/base_link")])
        joint.CreateLocalPos0Attr(Gf.Vec3f(0))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0))
        joint.CreateLocalRot0Attr(Gf.Quatf(1))
        joint.CreateLocalRot1Attr(Gf.Quatf(1))
        destination = output / "ur5_robotiq.usd"
        stage.Flatten().Export(str(destination))
        urdf = output / "ur5_controller.urdf"
        _controller_urdf(stage, urdf)
    provenance = json.loads((source.parent / "provenance.json").read_text())
    manifest = {
        "robot_id": "ur5_robotiq", "units": "metres/radians", "quaternion_order": "wxyz",
        "arm_joint_names": ARM_JOINTS,
        "tcp_in_wrist_3_link": {"position": TCP_POSITION, "quaternion": TCP_QUATERNION},
        "sources": provenance["sources"],
        "source_urdf_sha256": {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                               for name in ("arm.urdf", "gripper.urdf")},
        "adaptations": provenance["adaptations"],
        "builder": "kuafu.isaac.learned.ur5.prepare_assets (Isaac Lab URDF importer)",
        "uncertainty": "Virtual TCP at 162 mm; simulated mount and nominal UR5 geometry require hardware calibration.",
        "outputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (destination, urdf)},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=external_artifact_root() / "robots/ur5_robotiq")
    parser.add_argument("--source-dir", type=Path, default=ROBOT_ASSET_ROOT / "ur5_robotiq/source")
    args = parser.parse_args()
    app = AppLauncher(headless=True).app
    try:
        prepare(args.output_dir, args.source_dir)
        print(f"Prepared UR5 assets: {args.output_dir}", flush=True)
    finally:
        app.close()


if __name__ == "__main__":
    main()
