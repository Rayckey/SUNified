"""Check the bundled robot is portable and its MPC model matches the USD joints."""

import hashlib
import json
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from kuafu.isaac.learned.ur5.assets import ARM_JOINTS, HOME_JOINTS, asset_directory, require_assets
from kuafu.paths import ROBOT_ASSET_ROOT


def test_default_robot_uses_checkout_and_override_is_explicit(monkeypatch, tmp_path):
    monkeypatch.delenv("KUAFU_UR5_ASSET_DIR", raising=False)
    monkeypatch.setenv("KUAFU_ARTIFACT_ROOT", str(tmp_path / "empty-artifacts"))
    assert asset_directory() == ROBOT_ASSET_ROOT / "ur5_robotiq"
    assert all(path.is_file() for path in require_assets())
    monkeypatch.setenv("KUAFU_UR5_ASSET_DIR", str(tmp_path / "custom"))
    assert asset_directory() == tmp_path / "custom"
    with pytest.raises(FileNotFoundError, match="KUAFU_UR5_ASSET_DIR"):
        require_assets()


def test_bundle_hashes_and_rebuild_meshes_are_complete():
    root = ROBOT_ASSET_ROOT / "ur5_robotiq"
    manifest = json.loads((root / "manifest.json").read_text())
    for name, digest in manifest["outputs"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    for name, digest in manifest["source_urdf_sha256"].items():
        assert hashlib.sha256((root / "source" / name).read_bytes()).hexdigest() == digest
        for mesh in ET.parse(root / "source" / name).findall(".//mesh"):
            path = Path(mesh.get("filename"))
            assert not path.is_absolute() and "://" not in str(path)
            assert (root / "source" / path).resolve().is_relative_to(root)
            assert (root / "source" / path).is_file()
    provenance = json.loads((root / "provenance.json").read_text())
    for name, source in provenance["source_files"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == source["sha256"]


def test_usd_loads_after_relocation_without_source_meshes(tmp_path):
    pytest.importorskip("pxr.Usd")
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    path = tmp_path / "robot.usd"
    shutil.copyfile(ROBOT_ASSET_ROOT / "ur5_robotiq/ur5_robotiq.usd", path)
    stage = Usd.Stage.Open(str(path))
    assert stage.GetDefaultPrim().GetPath() == Sdf.Path("/Robot")
    # Empty asset paths are references to other prims within the same USD.
    assert not any(stage.GetRootLayer().GetExternalReferences())
    prims = list(Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()))
    assert any(prim.IsA(UsdGeom.Mesh) for prim in prims)
    for name in ("left_inner_finger", "right_inner_finger"):
        finger = stage.GetPrimAtPath(f"/Robot/Gripper/Robotiq_2F_85/{name}")
        collisions = [prim for prim in Usd.PrimRange(finger) if prim.HasAPI(UsdPhysics.CollisionAPI)]
        assert len(collisions) == 2, "Fingertip collisions must be editable for contact material binding"
        assert all(not prim.IsInstanceProxy() for prim in collisions)
    joints = {prim.GetName() for prim in prims if prim.IsA(UsdPhysics.RevoluteJoint)}
    assert set(ARM_JOINTS) | {"finger_joint", "right_outer_knuckle_joint"} <= joints
    for prim in prims:
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            assert UsdPhysics.MassAPI(prim).GetMassAttr().Get() > 0, prim.GetPath()
        if prim.IsA(UsdPhysics.RevoluteJoint) and "Gripper" in str(prim.GetPath()):
            if prim.GetName() != "finger_joint":
                axis = UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get()
                prefix = f"physxMimicJoint:rot{axis}:"
                expected = 1.0 if "inner_finger" in prim.GetName() else -1.0
                assert prim.GetAttribute(prefix + "gearing").Get() == expected
                assert prim.GetRelationship(prefix + "referenceJoint").GetTargets()[0].name == "finger_joint"
        for attribute in prim.GetAttributes():
            value = attribute.Get()
            if isinstance(value, Sdf.AssetPath) and value.path:
                # Standard shader modules are supplied by the Isaac renderer.
                assert Path(value.path).name == value.path and value.path.endswith(".mdl"), value.path


def test_mpc_kinematics_match_usd_joint_frames():
    pin = pytest.importorskip("pinocchio")
    pytest.importorskip("pxr.Usd")
    import numpy as np
    from pxr import Usd, UsdPhysics
    from scipy.spatial.transform import Rotation

    root = ROBOT_ASSET_ROOT / "ur5_robotiq"
    model = pin.buildModelFromUrdf(str(root / "ur5_controller.urdf"))
    assert model.nq == 6
    data = model.createData()
    stage = Usd.Stage.Open(str(root / "ur5_robotiq.usd"))
    joints = {prim.GetName(): UsdPhysics.RevoluteJoint(prim) for prim in stage.Traverse()
              if prim.IsA(UsdPhysics.RevoluteJoint)}

    def frame(position, quaternion):
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_quat([*quaternion.GetImaginary(), quaternion.GetReal()]).as_matrix()
        matrix[:3, 3] = position
        return matrix

    for q in (np.zeros(6), np.array(HOME_JOINTS), np.random.default_rng(42).uniform(-1, 1, 6)):
        transforms = {"base_link": np.eye(4)}
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        for name, angle in zip(ARM_JOINTS, q):
            joint = joints[name]
            parent = joint.GetBody0Rel().GetTargets()[0].name
            child = joint.GetBody1Rel().GetTargets()[0].name
            axis = np.eye(3)["XYZ".index(joint.GetAxisAttr().Get())]
            motion = np.eye(4)
            motion[:3, :3] = Rotation.from_rotvec(angle * axis).as_matrix()
            transforms[child] = (transforms[parent]
                @ frame(joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get()) @ motion
                @ np.linalg.inv(frame(joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get())))
            np.testing.assert_allclose(data.oMf[model.getFrameId(child)].homogeneous, transforms[child], atol=1e-7)
    pin.forwardKinematics(model, data, np.array(HOME_JOINTS))
    pin.updateFramePlacements(model, data)
    np.testing.assert_allclose(data.oMf[model.getFrameId("panda_fingertip")].translation,
                               [0.4869, 0.10915, 0.269859], atol=1e-6)
