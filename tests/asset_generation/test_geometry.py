import numpy as np
import pytest
from pydantic import ValidationError
from scipy.spatial import ConvexHull

from kuafu.asset_generation.geometry import build_asset, triangulate_profile
from kuafu.asset_generation.schema import PhysicsConfig, RoundedBox, Scale, SceneSpec


def inside_any(point, pieces):
    for mesh in pieces:
        equations = ConvexHull(mesh.vertices).equations
        if np.all(equations[:, :3] @ point + equations[:, 3] <= 1e-9):
            return True
    return False


def test_cup_colliders_keep_interior_open_but_support_contents(mug):
    body = mug.model_copy(update={"parts": mug.parts[:1]})
    asset = build_asset(body)
    assert not inside_any(np.array([0, 0, 0.05]), asset.collisions)
    assert inside_any(np.array([0.038, 0, 0.05]), asset.collisions)
    assert inside_any(np.array([0, 0, 0.0025]), asset.collisions)
    assert all(m.is_volume and m.is_convex for m in asset.collisions)
    assert asset.mesh().volume == pytest.approx(np.pi * (0.04**2 * 0.1 - 0.035**2 * 0.095), rel=0.01)


def test_scaling_keeps_visual_and_collision_coordinates_aligned(mug):
    original = build_asset(mug)
    scaled = build_asset(mug, Scale(size_m=0.2, axis="z"))
    assert scaled.metadata["dimensions_m"][2] == pytest.approx(0.2)
    assert scaled.metadata["scale_basis"] == "user_measurement"
    assert np.allclose(scaled.mesh().bounds, original.mesh().bounds * 2)
    for a, b in zip(original.collisions, scaled.collisions, strict=True):
        assert np.allclose(b.vertices, a.vertices * 2)
    assert scaled.mesh().bounds[0, 2] == pytest.approx(0)
    assert np.allclose(scaled.mesh().bounds.mean(axis=0)[:2], 0)


@pytest.mark.parametrize(
    "geometry",
    [
        {"kind": "box", "size": {"x": 0.1, "y": 0.2, "z": 0.3}},
        {"kind": "rounded_box", "size": {"x": 0.1, "y": 0.2, "z": 0.3}, "radius": 0.012},
        {"kind": "ellipsoid", "radii": {"x": 0.1, "y": 0.2, "z": 0.3}},
        {"kind": "cylinder", "radius": 0.05, "height": 0.1},
        {"kind": "frustum", "radius_bottom": 0.05, "radius_top": 0.03, "height": 0.1},
        {"kind": "frustum", "radius_bottom": 0.05, "radius_top": 0, "height": 0.1},
        {"kind": "torus", "major_radius": 0.05, "minor_radius": 0.01},
        {
            "kind": "convex_hull",
            "points": [{"x": x, "y": y, "z": z} for x, y, z in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]],
        },
    ],
)
def test_supported_solids_generate_usable_convex_physics(geometry, mug):
    data = mug.model_dump()
    data["parts"] = data["parts"][:1]
    data["parts"][0]["geometry"] = geometry
    asset = build_asset(SceneSpec.model_validate(data))
    assert asset.mesh().is_volume
    assert all(m.is_volume and m.is_convex for m in asset.collisions)
    if geometry["kind"] == "torus":
        assert not inside_any(np.array([0, 0, 0.01]), asset.collisions)
    if geometry["kind"] == "rounded_box":
        assert asset.mesh().extents == pytest.approx([0.1, 0.2, 0.3])
        assert len(asset.collisions) == 1
        assert len(asset.collisions[0].vertices) <= 256
        assert asset.visuals[0].metadata["smooth_shading"]


def test_rejects_self_intersecting_and_flat_profiles():
    for p in ([[0, 0], [1, 1], [0, 1], [1, 0]], [[0, 0], [1, 0], [2, 0]]):
        with pytest.raises(ValueError):
            triangulate_profile(np.array(p))


def test_model_output_cannot_inject_code_or_nonfinite_coordinates(mug):
    data = mug.model_dump()
    data["parts"][0]["geometry"] = {"kind": "python", "code": "print('not executable')"}
    with pytest.raises(ValidationError):
        SceneSpec.model_validate(data)
    data = mug.model_dump()
    data["parts"][0]["position"]["x"] = float("nan")
    with pytest.raises(ValidationError):
        SceneSpec.model_validate(data)


def test_dimensions_are_not_inferred_from_scale_claim_alone(mug):
    assert mug.scale_basis == "user_measurement"
    assert build_asset(mug).metadata["scale_basis"] == "estimated"


def test_srgb_colors_survive_mesh_construction(mug):
    asset = build_asset(mug)
    assert asset.visuals[0].visual.face_colors[0, :3].tolist() == [31, 110, 184]


def test_rounded_holder_has_closed_shell_and_open_physics_cavity(holder):
    asset = build_asset(holder)
    mesh = asset.mesh()
    assert mesh.is_volume and mesh.euler_number == 2
    assert len(mesh.split()) == 1
    assert asset.metadata["dimensions_m"] == pytest.approx([0.08, 0.08, 0.112], abs=0.0001)
    assert len(asset.collisions) == 17
    assert all(m.is_volume and m.is_convex for m in asset.collisions)
    assert inside_any(np.array([0, 0, 0.002]), asset.collisions)
    for z in [0.012, 0.05, 0.105]:
        for x in [-0.02, 0, 0.02]:
            for y in [-0.02, 0, 0.02]:
                assert not inside_any(np.array([x, y, z]), asset.collisions)
        for x, y in [(0.038, 0), (-0.038, 0), (0, 0.038), (0, -0.038)]:
            assert inside_any(np.array([x, y, z]), asset.collisions)
    assert np.isfinite(mesh.vertex_normals).all()
    assert np.allclose(np.linalg.norm(mesh.vertex_normals, axis=1), 1)


@pytest.mark.parametrize(
    "overrides",
    [
        {"corner_radius": 0.041},
        {"wall_thickness": 0.01},
        {"edge_radius": 0.005},
        {"bottom_inset": 0.01},
        {"size": {"x": 0.08, "y": 0.08, "z": 0.009}},
    ],
)
def test_rounded_holder_rejects_collapsed_or_self_intersecting_profiles(holder, overrides):
    data = holder.model_dump()
    data["parts"][0]["geometry"].update(overrides)
    with pytest.raises(ValidationError):
        SceneSpec.model_validate(data)


def test_rounded_box_rejects_radius_that_collapses_the_core():
    with pytest.raises(ValidationError):
        RoundedBox(kind="rounded_box", size={"x": 0.1, "y": 0.2, "z": 0.03}, radius=0.02)


def test_estimated_mass_requires_an_explicit_value():
    with pytest.raises(ValidationError):
        PhysicsConfig(mass_is_estimate=True)
