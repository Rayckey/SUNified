import json
from pathlib import Path

import numpy as np
import pytest
import trimesh
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from scipy.spatial import ConvexHull

from kuafu.asset_generation.export import export_asset
from kuafu.asset_generation.geometry import build_asset
from kuafu.asset_generation.pipeline import reconstruct
from kuafu.asset_generation.schema import PhysicsConfig, Review, RunConfig, Scale, SceneSpec
from kuafu.asset_generation.semantic_runtime import describe
from kuafu.asset_generation.semantics import build_semantics, require_agent_semantics
from kuafu.asset_generation.video import prepare


@pytest.fixture
def cabinet():
    path = Path(__file__).resolve().parents[2] / "assets/recipes/0028_0029.articulated.scene.json"
    return SceneSpec.model_validate_json(path.read_text())


def test_exported_cabinet_has_language_and_six_body_local_usd_frames(cabinet, tmp_path):
    asset = build_asset(cabinet, Scale(size_m=0.8, axis="x"))
    export_asset(asset, cabinet, tmp_path, PhysicsConfig(mass_kg=2.5))
    semantics = json.loads((tmp_path / "semantics.json").read_text())
    assert len(semantics["frames"]) == 6
    assert semantics == json.loads((tmp_path / "asset.json").read_text())["semantics"]
    stage = Usd.Stage.Open(str(tmp_path / "asset_dynamic.usda"))
    for frame in cabinet.semantics.frames:
        entry = semantics["frames"][frame.name]
        expected = np.array(frame.position.array()) * 2 + asset.metadata["translation_m"]
        np.testing.assert_allclose(entry["position_m"], expected, atol=1e-10)
        prim = stage.GetPrimAtPath(entry["usd_path"])
        assert prim and not prim.HasAPI(UsdPhysics.RigidBodyAPI)
        matrix = UsdGeom.Xformable(prim).GetLocalTransformation()
        np.testing.assert_allclose(matrix.ExtractTranslation(), expected, atol=1e-10)
        np.testing.assert_allclose(prim.GetAttribute("semantics:axis:open").Get(), [0, -1, 0])
        assert "local frame -y is the opening direction" in entry["language"]
        assert "local frame +y is the closing direction" in entry["language"]
        body = stage.GetPrimAtPath("/Asset/" + entry["body_path"])
        assert body.HasAPI(UsdPhysics.RigidBodyAPI)
        # For this recipe, grasp points lie on handle material while center
        # points lie in empty cavities, not at a mesh origin or inside a floor.
        inside = []
        for collider in asset.collisions:
            if collider.metadata["link"] != frame.body:
                continue
            planes = ConvexHull(collider.vertices).equations
            if np.all(planes[:, :3] @ expected + planes[:, 3] < 1e-7):
                inside.append(collider.metadata["part_name"])
        if frame.role == "grasp":
            assert frame.part in inside
        elif frame.role == "center":
            assert not inside
    assert semantics["frames"]["top_drawer_handle_frame"]["motion"]["limits"] == [0, 0.29]
    assert "make_semantic_cfg" in (tmp_path / "isaaclab_snippet.py").read_text()


def test_rotated_frame_axes_match_joint_axis_and_negative_revolute_opening(cabinet):
    data = cabinet.model_dump()
    joint = data["articulation"]["joints"][0]
    joint.update(kind="revolute", axis="Z", lower=-90, upper=0)
    frames = [f for f in data["semantics"]["frames"] if f["joint"] == joint["name"]]
    for frame in frames:
        frame["rotation_deg"] = dict(x=90, y=0, z=45)
        frame["opening_direction"] = "negative"
    frames[0].update(role="hinge", position=joint["origin"], part=None)
    scene = SceneSpec.model_validate(data)
    semantics = build_semantics(scene, 2, np.array([1, 2, 3]), [0.8, 0.5, 0.52])
    for frame in frames:
        entry = semantics["frames"][frame["name"]]
        rotation = trimesh.transformations.quaternion_matrix(entry["quaternion_wxyz"])[:3, :3]
        np.testing.assert_allclose(rotation @ entry["axes"]["open"]["direction"], [0, 0, -1], atol=1e-12)
        assert entry["motion"]["limits"] == [-90, 0]  # Do not scale angles.
        assert "rotation around local frame" in entry["language"]


@pytest.mark.parametrize(
    "invalid",
    [
        "body",
        "part",
        "joint",
        "stationary",
        "direction",
        "zero_axis",
        "nan_axis",
        "reserved_axis",
        "duplicate",
        "hinge",
        "frame_suffix",
        "axis_duplicate",
        "missing_direction",
    ],
)
def test_inconsistent_semantics_rejected(cabinet, invalid):
    data = cabinet.model_dump()
    frame = data["semantics"]["frames"][0]
    if invalid == "body":
        frame["body"] = "unknown"
    elif invalid == "part":
        frame["part"] = data["parts"][0]["name"]
    elif invalid == "joint":
        frame["joint"] = "missing_joint"
    elif invalid == "stationary":
        frame.update(body="frame", part=None)
    elif invalid == "direction":
        frame["opening_direction"] = "negative"
    elif invalid == "zero_axis":
        frame["axes"][0]["direction"] = dict(x=0, y=0, z=0)
    elif invalid == "nan_axis":
        frame["axes"][0]["direction"]["x"] = float("nan")
    elif invalid == "reserved_axis":
        frame["axes"][0]["name"] = "open"
    elif invalid == "duplicate":
        data["semantics"]["frames"][1]["name"] = frame["name"]
    elif invalid == "hinge":
        frame["role"] = "hinge"
    elif invalid == "frame_suffix":
        frame["name"] = "no_suffix"
    elif invalid == "axis_duplicate":
        frame["axes"].append(frame["axes"][0])
    else:
        frame["opening_direction"] = None
    with pytest.raises(ValueError):
        SceneSpec.model_validate(data)


def test_legacy_recipe_still_builds_but_agent_must_supply_semantics(mug):
    data = mug.model_dump()
    data["semantics"] = None
    legacy = SceneSpec.model_validate(data)
    assert build_asset(legacy).metadata["semantics"]["status"] == "missing"
    with pytest.raises(ValueError, match="Supply semantics"):
        require_agent_semantics(legacy)


def test_agent_must_cover_every_moving_joint(cabinet):
    data = cabinet.model_dump()
    data["semantics"]["frames"] = data["semantics"]["frames"][:2]
    with pytest.raises(ValueError, match="lower_left_drawer_joint"):
        require_agent_semantics(SceneSpec.model_validate(data))


def test_pipeline_repairs_missing_semantics_instead_of_exporting_empty_frames(video, mug, tmp_path):
    workspace = tmp_path / "job"
    prepare(video, "Mug with an open interior and graspable side handle", workspace, 2)

    class Agent:
        calls = 0

        def complete(self, schema, instructions, prompt, images):
            if schema is SceneSpec:
                self.calls += 1
                assert "INTERACTION SEMANTICS ARE REQUIRED" in instructions
                if self.calls == 1:
                    data = mug.model_dump()
                    data["semantics"] = None
                    return SceneSpec.model_validate(data)
                assert "Supply semantics" in prompt
                return mug
            return Review(score=0.9, acceptable=True, issues=[], suggested_changes=[], uncertainty=[])

    state = reconstruct(workspace, RunConfig(iterations=2), PhysicsConfig(), Agent())
    assert state["status"] == "complete"
    assert state["iterations"][0]["status"] == "invalid_geometry"
    assert json.loads((workspace / "final/semantics.json").read_text())["status"] == "annotated"


def test_rigid_frame_usd_orientation_and_language_escaping(mug, tmp_path):
    data = mug.model_dump()
    frame = data["semantics"]["frames"][0]
    frame["description"] = 'The "cup" interior\nwith an opening'
    frame["rotation_deg"] = dict(x=90, y=0, z=45)
    scene = SceneSpec.model_validate(data)
    asset = build_asset(scene)
    export_asset(asset, scene, tmp_path, PhysicsConfig())
    entry = asset.metadata["semantics"]["frames"][frame["name"]]
    stage = Usd.Stage.Open(str(tmp_path / "asset_dynamic.usda"))
    prim = stage.GetPrimAtPath(entry["usd_path"])
    assert entry["body_path"] == ""
    assert stage.GetDefaultPrim().HasAPI(UsdPhysics.RigidBodyAPI)
    matrix = UsdGeom.Xformable(prim).GetLocalTransformation()
    expected = trimesh.transformations.quaternion_matrix(entry["quaternion_wxyz"])
    np.testing.assert_allclose(matrix.TransformDir(Gf.Vec3d(1, 0, 0)), expected[:3, 0], atol=1e-10)
    assert prim.GetAttribute("semantics:description").Get() == entry["language"]
    assert "(0.14, 0.14, 0.19) meters" in describe(entry, scale=2)
