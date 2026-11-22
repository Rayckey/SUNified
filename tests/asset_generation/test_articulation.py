import copy
import json
import shutil

import numpy as np
import pytest
import trimesh
from PIL import Image
from pxr import Usd, UsdPhysics

from kuafu.asset_generation.articulation import posed_asset
from kuafu.asset_generation.export import export_asset
from kuafu.asset_generation.geometry import build_asset
from kuafu.asset_generation.io import digest
from kuafu.asset_generation.pipeline import reconstruct
from kuafu.asset_generation.reviewed import verify_physics_report
from kuafu.asset_generation.schema import PhysicsConfig, Review, RunConfig, Scale, SceneSpec
from kuafu.asset_generation.textured import build_textured_asset
from kuafu.asset_generation.video import prepare


@pytest.fixture
def drawer(mug):
    data = mug.model_dump()
    data["parts"] = []
    for name, size, position in (
        ("frame", (0.3, 0.2, 0.02), (0, 0, 0)),
        ("drawer", (0.1, 0.15, 0.01), (0, 0, 0.05)),
        ("handle", (0.05, 0.02, 0.02), (0, -0.1, 0.07)),
    ):
        part = copy.deepcopy(mug.model_dump()["parts"][0])
        part.update(
            name=name,
            link=name,
            geometry={"kind": "box", "size": dict(zip("xyz", size))},
            position=dict(zip("xyz", position)),
            rotation_deg=dict(x=0, y=0, z=0),
        )
        data["parts"].append(part)
    data["articulation"] = {
        "root": "frame",
        "fixed_base": True,
        "links": [{"name": name} for name in ("frame", "drawer", "handle")],
        "joints": [
            {
                "name": "drawer_joint",
                "parent": "frame",
                "child": "drawer",
                "kind": "prismatic",
                "axis": "-Y",
                "lower": 0,
                "upper": 0.12,
                "origin": dict(x=0, y=0, z=0.05),
            },
            {"name": "handle_joint", "parent": "drawer", "child": "handle", "kind": "fixed"},
        ],
    }
    data["semantics"] = {
        "name": "cabinet",
        "description": "Synthetic sliding drawer",
        "frames": [
            {
                "name": "drawer_handle_frame",
                "body": "handle",
                "role": "grasp",
                "description": "Drawer handle",
                "position": dict(x=0, y=-0.1, z=0.07),
                "joint": "drawer_joint",
                "opening_direction": "positive",
                "evidence": "Synthetic handle center.",
            }
        ],
    }
    return SceneSpec.model_validate(data)


def test_articulated_usd_has_separate_bodies_passive_joint_and_fixed_base(drawer, tmp_path):
    asset = build_asset(drawer, Scale(size_m=0.6, axis="x"))
    export_asset(asset, drawer, tmp_path, PhysicsConfig(mass_kg=3))
    stage = Usd.Stage.Open(str(tmp_path / "asset_dynamic.usda"))
    assert not stage.GetDefaultPrim().HasAPI(UsdPhysics.RigidBodyAPI)
    assert stage.GetPrimAtPath("/Asset/root_joint").HasAPI(UsdPhysics.ArticulationRootAPI)
    for name in ("frame", "drawer", "handle"):
        body = stage.GetPrimAtPath(f"/Asset/{name}")
        assert body.HasAPI(UsdPhysics.RigidBodyAPI)
        assert UsdPhysics.MassAPI(body).GetMassAttr().Get() == 1
        assert stage.GetPrimAtPath(f"/Asset/{name}/Collisions")
    joint = UsdPhysics.PrismaticJoint(stage.GetPrimAtPath("/Asset/drawer_joint"))
    assert joint.GetUpperLimitAttr().Get() == pytest.approx(0.24)
    assert joint.GetBody1Rel().GetTargets() == ["/Asset/drawer"]
    assert joint.GetAxisAttr().Get() == "Y"
    assert tuple(joint.GetLocalRot1Attr().Get().GetImaginary()) == (1, 0, 0)
    drive = UsdPhysics.DriveAPI(joint.GetPrim(), "linear")
    assert drive.GetStiffnessAttr().Get() == 0
    assert drive.GetDampingAttr().Get() == 3
    assert "ArticulationCfg" in (tmp_path / "isaaclab_snippet.py").read_text()
    static = Usd.Stage.Open(str(tmp_path / "asset_static.usda"))
    assert not any(prim.HasAPI(UsdPhysics.RigidBodyAPI) for prim in static.Traverse())
    assert (tmp_path / "articulation_open/preview.png").is_file()


def test_scaled_open_pose_moves_handle_and_colliders_with_drawer(drawer):
    asset = build_asset(drawer, Scale(size_m=0.6, axis="x"))
    posed = posed_asset(asset, 1)
    for original, moved in zip(asset.visuals + asset.collisions, posed.visuals + posed.collisions):
        delta = [0, 0, 0] if original.metadata["link"] == "frame" else [0, -0.24, 0]
        np.testing.assert_allclose(moved.vertices, original.vertices + delta, atol=1e-12)
    np.testing.assert_allclose(
        list(asset.metadata["articulation"]["joints"][0]["origin"].values()),
        np.array([0, 0, 0.05]) * 2 + asset.metadata["translation_m"],
    )


@pytest.mark.parametrize("change", ["unknown", "cycle", "duplicate", "limits", "unowned", "injection"])
def test_invalid_mechanism_fails_before_export(drawer, change):
    data = drawer.model_dump()
    if change == "unknown":
        data["articulation"]["joints"][0]["parent"] = "missing"
    elif change == "cycle":
        data["articulation"]["joints"][0]["parent"] = "handle"
    elif change == "duplicate":
        data["articulation"]["joints"][1]["child"] = "drawer"
    elif change == "limits":
        data["articulation"]["joints"][0]["upper"] = -0.1
    elif change == "unowned":
        data["parts"][0]["link"] = "unowned"
    else:
        data["articulation"]["links"][0]["name"] = 'frame"\n}'
    with pytest.raises(ValueError):
        SceneSpec.model_validate(data)


def test_two_video_sentence_reaches_agent_and_cannot_be_flattened(video, drawer, tmp_path):
    second = tmp_path / "second.avi"
    shutil.copy2(video, second)
    workspace = tmp_path / "job"
    description = "The removed white-handled drawer belongs in this cabinet."
    inputs = prepare(video, description, workspace, 4, reference_videos=[second], motion="articulated")
    assert [frame.source_index for frame in inputs.frames] == [0, 0, 1, 1]
    assert len({frame.path for frame in inputs.frames}) == 4
    assert len(inputs.reference_videos) == 1

    class Agent:
        proposals = 0

        def complete(self, schema, instructions, prompt, images):
            assert description in prompt and len(images) >= 4
            if schema is SceneSpec:
                self.proposals += 1
                if self.proposals == 1:
                    data = drawer.model_dump()
                    data["articulation"] = None
                    for part in data["parts"]:
                        part["link"] = "base"
                    data["semantics"] = None
                    return SceneSpec.model_validate(data)
                assert "Articulated motion was requested" in prompt
                return drawer
            assert images[-1].parent.name == "articulation_open"
            return Review(score=0.9, acceptable=True, issues=[], suggested_changes=[], uncertainty=[])

    result = reconstruct(workspace, RunConfig(iterations=2), PhysicsConfig(), Agent())
    assert result["iterations"][0]["status"] == "invalid_geometry"
    assert result["status"] == "complete"
    assert result["metadata"]["asset_type"] == "articulation"
    assert result["metadata"]["isaac_physics_tested"] is False


def test_textured_articulation_requires_and_retains_mesh_ownership(drawer, tmp_path):
    asset = build_asset(drawer)
    glb = trimesh.Scene()
    mapping = {}
    for i, mesh in enumerate(asset.visuals):
        mesh = mesh.copy()
        mesh.visual = trimesh.visual.TextureVisuals(
            uv=np.zeros((len(mesh.vertices), 2)),
            material=trimesh.visual.material.PBRMaterial(baseColorTexture=Image.new("RGB", (8, 8), "tan")),
        )
        mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
        node = f"Part_{i:03d}"
        glb.add_geometry(mesh, node_name=node, geom_name=node)
        mapping[node] = mesh.metadata["link"]
    source = tmp_path / "source.glb"
    glb.export(source)
    scene = tmp_path / "scene.json"
    scene.write_text(drawer.model_dump_json())
    with pytest.raises(ValueError, match="visual_links"):
        build_textured_asset(source, scene, tmp_path / "bad", PhysicsConfig())
    assert not (tmp_path / "bad").exists()
    out = tmp_path / "textured"
    build_textured_asset(source, scene, out, PhysicsConfig(), visual_links=mapping)
    stage = Usd.Stage.Open(str(out / "asset_dynamic.usda"))
    assert stage.GetPrimAtPath("/Asset/drawer_joint").IsA(UsdPhysics.PrismaticJoint)
    assert len([p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]) == 3
    assert json.loads((out / "asset.json").read_text())["visual_links"] == mapping
    semantics = json.loads((out / "semantics.json").read_text())
    assert semantics == asset.metadata["semantics"]
    assert stage.GetPrimAtPath(semantics["frames"]["drawer_handle_frame"]["usd_path"])
    assert (out / "isaaclab_semantics.py").is_file()


def test_rigid_physics_pass_cannot_publish_an_articulated_asset(drawer, tmp_path):
    asset = build_asset(drawer)
    export_asset(asset, drawer, tmp_path, PhysicsConfig())
    report = {
        "passed": True,
        "asset": str(tmp_path / "asset_dynamic.usda"),
        "steps": 240,
        "physics_dt": 1 / 120,
        "asset_layers": [
            {"path": str(tmp_path / name), "sha256": digest(tmp_path / name)}
            for name in ("asset_dynamic.usda", "asset_geometry.usda")
        ],
    }
    (tmp_path / "physics_smoke.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="every articulated joint"):
        verify_physics_report(tmp_path)


def test_revolute_preview_uses_scaled_hinge_without_scaling_angles(drawer):
    data = drawer.model_dump()
    joint = data["articulation"]["joints"][0]
    joint.update(kind="revolute", axis="Z", lower=0, upper=90)
    spec = SceneSpec.model_validate(data)
    asset = build_asset(spec, Scale(size_m=0.6, axis="x"))
    assert asset.metadata["articulation"]["joints"][0]["upper"] == 90
    anchor = np.array(list(asset.metadata["articulation"]["joints"][0]["origin"].values()))
    posed = posed_asset(asset, 1)
    rotation = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1])[:3, :3]
    for original, moved in zip(asset.visuals[1:], posed.visuals[1:]):
        np.testing.assert_allclose(
            moved.vertices, (original.vertices - anchor) @ rotation.T + anchor, atol=1e-12
        )


def test_drawer_must_clear_frame_through_travel_not_just_at_rest(drawer):
    data = drawer.model_dump()
    obstacle = copy.deepcopy(data["parts"][0])
    obstacle.update(
        name="obstruction",
        geometry={"kind": "box", "size": dict(x=0.03, y=0.02, z=0.03)},
        position=dict(x=0, y=-0.15, z=0.05),
    )
    data["parts"].append(obstacle)
    with pytest.raises(ValueError, match="penetrate during travel"):
        build_asset(SceneSpec.model_validate(data))
