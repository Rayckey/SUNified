import json

import numpy as np
import pytest
import trimesh

from kuafu.asset_generation.export import export_asset
from kuafu.asset_generation.geometry import build_asset
from kuafu.asset_generation.schema import PhysicsConfig, Scale


@pytest.fixture(params=["mug", "holder"])
def exported(tmp_path, request):
    spec = request.getfixturevalue(request.param)
    scale = Scale(size_m=0.1, axis="z")
    asset = build_asset(spec, scale)
    export_asset(asset, spec, tmp_path, PhysicsConfig(mass_kg=0.25), scale)
    return tmp_path, asset


def test_mesh_exports_roundtrip_with_correct_units_and_axes(exported):
    out, asset = exported
    obj = trimesh.load(out / "asset.obj", force="mesh", process=False)
    stl = trimesh.load(out / "asset.stl", force="mesh")
    glb = trimesh.load(out / "asset.glb", force="scene")
    assert np.allclose(obj.bounds, asset.mesh().bounds, atol=1e-6)
    assert np.allclose(stl.extents, asset.mesh().extents, atol=1e-6)
    assert np.allclose(glb.extents, asset.mesh().extents[[0, 2, 1]], atol=1e-6)
    assert (out / "viewer.html").read_text().count("__MESH_DATA__") == 0


def test_usd_composes_as_one_rigid_body_with_separate_convex_colliders(exported):
    pxr = pytest.importorskip("pxr", reason="Install usd-core for USD composition checks")
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    out, asset = exported
    dynamic = Usd.Stage.Open(str(out / "asset_dynamic.usda"))
    static = Usd.Stage.Open(str(out / "asset_static.usda"))
    assert dynamic and static and pxr
    root = dynamic.GetDefaultPrim()
    assert root.GetPath().pathString == "/Asset"
    assert root.HasAPI(UsdPhysics.RigidBodyAPI)
    assert UsdPhysics.MassAPI(root).GetMassAttr().Get() == pytest.approx(0.25)
    assert not static.GetDefaultPrim().HasAPI(UsdPhysics.RigidBodyAPI)
    assert UsdGeom.GetStageMetersPerUnit(dynamic) == 1
    assert UsdGeom.GetStageUpAxis(dynamic) == "Z"
    colliders = [p for p in dynamic.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
    assert len(colliders) == len(asset.collisions)
    for prim in colliders:
        assert UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == "convexHull"
        assert UsdGeom.Imageable(prim).GetVisibilityAttr().Get() == "invisible"
        assert UsdGeom.Mesh.ValidateTopology(
            UsdGeom.Mesh(prim).GetFaceVertexIndicesAttr().Get(),
            UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get(),
            len(UsdGeom.Mesh(prim).GetPointsAttr().Get()),
        )[0]
        bound, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
        assert bound and bound.GetPrim().HasAPI(UsdPhysics.MaterialAPI)
    for prim, mesh in zip(dynamic.GetPrimAtPath("/Asset/Visuals").GetChildren(), asset.visuals, strict=True):
        assert not prim.HasAPI(UsdPhysics.CollisionAPI)
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        assert material
        if mesh.metadata.get("smooth_shading"):
            usd_mesh = UsdGeom.Mesh(prim)
            assert usd_mesh.GetNormalsInterpolation() == "vertex"
            normals = np.array(usd_mesh.GetNormalsAttr().Get())
            assert normals.shape == mesh.vertices.shape
            assert np.allclose(np.linalg.norm(normals, axis=1), 1)
    assert Usd.Stage.Open(str(out / "asset.usda")).GetDefaultPrim().HasAPI(UsdPhysics.RigidBodyAPI)


@pytest.mark.parametrize(
    "mass,estimate,basis",
    [(None, False, "placeholder"), (0.35, True, "estimated"), (0.35, False, "user_supplied")],
)
def test_mass_provenance_is_explicit(tmp_path, holder, monkeypatch, mass, estimate, basis):
    # Exercise the real exporters but omit previews already covered above.
    monkeypatch.setattr("kuafu.asset_generation.export.render_views", lambda *args: None)
    monkeypatch.setattr("kuafu.asset_generation.export.write_viewer", lambda *args: None)
    export_asset(
        build_asset(holder), holder, tmp_path, PhysicsConfig(mass_kg=mass, mass_is_estimate=estimate)
    )
    metadata = json.loads((tmp_path / "asset.json").read_text())
    assert metadata["mass_basis"] == basis
    assert metadata["scale_basis"] == "estimated"
    assert metadata["effective_mass_kg"] == (0.2 if mass is None else mass)
