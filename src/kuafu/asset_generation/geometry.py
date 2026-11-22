"""Visual meshes and compound convex collisions that preserve hollow shapes."""

from dataclasses import dataclass
from itertools import product

import numpy as np
import trimesh
from scipy.spatial import QhullError

from .rounded import rounded_container
from .schema import Scale, SceneSpec


@dataclass
class BuiltAsset:
    visuals: list[trimesh.Trimesh]
    collisions: list[trimesh.Trimesh]
    metadata: dict

    def mesh(self) -> trimesh.Trimesh:
        return trimesh.util.concatenate(self.visuals)


def _cross(a, b, c):
    ab, ac = b - a, c - a
    return ab[0] * ac[1] - ab[1] * ac[0]


def triangulate_profile(profile: np.ndarray) -> list[np.ndarray]:
    """Ear clipping of a simple radial polygon; reject ambiguous outlines."""
    p = np.asarray(profile, dtype=float)
    if np.allclose(p[0], p[-1], rtol=0, atol=1e-12):
        p = p[:-1]
    if len(p) < 3 or np.any(np.linalg.norm(p - np.roll(p, 1, axis=0), axis=1) < 1e-10):
        raise ValueError("lathe profile needs at least three distinct, non-repeated corners")
    epsilon = 1e-10 * max(float(np.ptp(p, axis=0).max()), 1e-9) ** 2
    for i in range(len(p)):
        a, b = p[i], p[(i + 1) % len(p)]
        for j in range(i + 1, len(p)):
            if j == i + 1 or (i == 0 and j == len(p) - 1):
                continue
            c, d = p[j], p[(j + 1) % len(p)]
            if _cross(a, b, c) * _cross(a, b, d) < -(epsilon**2) and _cross(c, d, a) * _cross(c, d, b) < -(
                epsilon**2
            ):
                raise ValueError("lathe profile must not self-intersect")
    area = np.sum(p[:, 0] * np.roll(p[:, 1], -1) - p[:, 1] * np.roll(p[:, 0], -1))
    if abs(area) <= epsilon:
        raise ValueError("lathe profile has zero cross-sectional area")
    if area < 0:
        p = p[::-1]
    remaining = list(range(len(p)))
    triangles = []
    while len(remaining) > 3:
        for i, current in enumerate(remaining):
            previous, following = remaining[i - 1], remaining[(i + 1) % len(remaining)]
            a, b, c = p[[previous, current, following]]
            if _cross(a, b, c) <= epsilon:
                continue
            others = [p[index] for index in remaining if index not in (previous, current, following)]
            if any(min(_cross(a, b, q), _cross(b, c, q), _cross(c, a, q)) >= -epsilon for q in others):
                continue
            triangles.append(np.array([a, b, c]))
            remaining.pop(i)
            break
        else:
            raise ValueError("lathe profile is degenerate; remove collinear or intersecting corners")
    triangles.append(p[remaining])
    return triangles


def _hull(points) -> trimesh.Trimesh:
    try:
        return trimesh.convex.convex_hull(np.asarray(points))
    except QhullError as error:
        raise ValueError("convex_hull points must span a non-zero 3D volume") from error


def _lathe_collisions(profile: np.ndarray) -> list[trimesh.Trimesh]:
    pieces = []
    angles = np.linspace(0, 2 * np.pi, 25)
    for triangle in triangulate_profile(profile):
        for a, b in zip(angles[:-1], angles[1:], strict=True):
            points = [[r * np.cos(t), r * np.sin(t), z] for t in (a, b) for r, z in triangle]
            pieces.append(_hull(points))
    return pieces


def _shape(g) -> tuple[trimesh.Trimesh, list[trimesh.Trimesh]]:
    if g.kind == "rounded_container":
        return rounded_container(g)
    if g.kind == "box":
        mesh = trimesh.creation.box(extents=g.size.array())
        return mesh, [mesh.copy()]
    if g.kind == "rounded_box":
        half_core = np.array(g.size.array()) / 2 - g.radius
        corners = np.array(list(product(*[[-v, v] for v in half_core])))
        meshes = []
        for subdivisions in (2, 1):
            sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=g.radius)
            meshes.append(_hull((corners[:, None, :] + sphere.vertices[None, :, :]).reshape(-1, 3)))
        return meshes[0], [meshes[1]]
    if g.kind == "ellipsoid":
        meshes = [trimesh.creation.icosphere(subdivisions=n) for n in (3, 1)]
        for mesh in meshes:
            mesh.apply_scale(g.radii.array())
        return meshes[0], [meshes[1]]
    if g.kind == "cylinder":
        return (
            trimesh.creation.cylinder(g.radius, g.height, sections=64),
            [trimesh.creation.cylinder(g.radius, g.height, sections=24)],
        )
    if g.kind == "frustum":
        h = g.height / 2
        profile = [[0, -h], [g.radius_bottom, -h], [g.radius_top, h], [0, h]]
        if g.radius_top == 0:
            profile.pop()
        profile.append(profile[0])
        return (
            trimesh.creation.revolve(profile, sections=64),
            [trimesh.creation.revolve(profile, sections=24)],
        )
    if g.kind == "torus":
        mesh = trimesh.creation.torus(g.major_radius, g.minor_radius, major_sections=64, minor_sections=16)
        pieces = []
        major = np.linspace(0, 2 * np.pi, 25)
        minor = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        for a, b in zip(major[:-1], major[1:], strict=True):
            points = [
                [
                    (g.major_radius + g.minor_radius * np.cos(v)) * np.cos(u),
                    (g.major_radius + g.minor_radius * np.cos(v)) * np.sin(u),
                    g.minor_radius * np.sin(v),
                ]
                for u in (a, b)
                for v in minor
            ]
            pieces.append(_hull(points))
        return mesh, pieces
    if g.kind == "lathe":
        profile = np.array([[p.radius, p.z] for p in g.profile])
        pieces = _lathe_collisions(profile)
        if not np.allclose(profile[0], profile[-1]):
            profile = np.vstack([profile, profile[0]])
        mesh = trimesh.creation.revolve(profile, sections=64)
        mesh.fix_normals()
        return mesh, pieces
    if g.kind == "sweep":
        points = np.array([point.array() for point in g.points])
        if np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) < 1e-8):
            raise ValueError("sweep consecutive points must be distinct")
        visual, colliders = [], []
        for a, b in zip(points[:-1], points[1:], strict=True):
            visual.append(trimesh.creation.cylinder(g.radius, segment=[a, b], sections=32))
            colliders.append(trimesh.creation.cylinder(g.radius, segment=[a, b], sections=16))
        for point in points:
            for collection, subdivisions in ((visual, 2), (colliders, 1)):
                ball = trimesh.creation.icosphere(subdivisions=subdivisions, radius=g.radius)
                ball.apply_translation(point)
                collection.append(ball)
        return trimesh.util.concatenate(visual), colliders
    if g.kind == "convex_hull":
        mesh = _hull([point.array() for point in g.points])
        return mesh, [mesh.copy()]
    raise ValueError(f"Unsupported geometry: {g.kind}")


def build_asset(spec: SceneSpec, scale: Scale | None = None) -> BuiltAsset:
    visuals, collisions = [], []
    for part in spec.parts:
        mesh, proxies = _shape(part.geometry)
        matrix = trimesh.transformations.euler_matrix(*np.deg2rad(part.rotation_deg.array()), axes="sxyz")
        matrix[:3, 3] = part.position.array()
        for geometry in [mesh, *proxies]:
            geometry.metadata.update(link=part.link, part_name=part.name)
            geometry.apply_transform(matrix)
            if not np.isfinite(geometry.vertices).all() or not geometry.is_volume:
                raise ValueError(f"Part {part.name!r} is not a closed, consistently oriented solid")
        mesh.visual.face_colors = np.round(np.array([*part.color.array(), 1.0]) * 255).astype(np.uint8)
        mesh.metadata.update(name=part.name, roughness=part.roughness, metallic=part.metallic)
        if part.geometry.kind in ("rounded_container", "rounded_box"):
            mesh.metadata["smooth_shading"] = True
        visuals.append(mesh)
        collisions.extend(proxies)
        if sum(len(m.faces) for m in visuals) > 80000 or len(collisions) > 512:
            raise ValueError("Asset exceeds 80,000 visual triangles or 512 collision pieces; simplify it")
    combined = trimesh.util.concatenate(visuals)
    dimensions = combined.extents
    if np.min(dimensions) <= 1e-7:
        raise ValueError("Object must have non-zero dimensions on all three axes")
    factor = 1.0
    if scale is not None:
        axis = int(np.argmax(dimensions)) if scale.axis == "longest" else "xyz".index(scale.axis)
        factor = scale.size_m / dimensions[axis]
    bounds = combined.bounds * factor
    translation = -np.array(
        [(bounds[0, 0] + bounds[1, 0]) / 2, (bounds[0, 1] + bounds[1, 1]) / 2, bounds[0, 2]]
    )
    for mesh in [*visuals, *collisions]:
        mesh.apply_scale(factor)
        mesh.apply_translation(translation)
    metadata = {
        "units": "meters",
        "up_axis": "Z",
        "front_axis": "-Y",
        "origin": "bounding-box center in XY, bottom contact plane at Z=0",
        "dimensions_m": (dimensions * factor).tolist(),
        "scale_factor": float(factor),
        "translation_m": translation.tolist(),
        "scale_basis": "user_measurement" if scale else "estimated",
        "visual_triangles": sum(len(m.faces) for m in visuals),
        "collision_pieces": len(collisions),
        "collisions": (
            "compound convex pieces; 24 segments for revolved shapes, "
            "16 wall pieces plus base for rounded containers"
        ),
        "warnings": ["Solids can overlap; the visual mesh is not a boolean-unioned CAD solid."],
    }
    if scale is None:
        metadata["warnings"].append("Absolute scale is estimated; provide --size-m and --size-axis.")
    metadata["asset_type"] = "articulation" if spec.articulation else "rigid"
    metadata["part_links"] = {part.name: part.link for part in spec.parts}
    metadata["part_frames"] = {
        part.name: {
            "body": part.link,
            "position_m": (np.array(part.position.array()) * factor + translation).tolist(),
            "quaternion_wxyz": trimesh.transformations.quaternion_from_euler(
                *np.deg2rad(part.rotation_deg.array()), axes="sxyz"
            ).tolist(),
        }
        for part in spec.parts
    }
    if spec.articulation:
        articulation = spec.articulation.model_dump()
        for joint in articulation["joints"]:
            origin = joint["origin"]
            point = np.array([origin[axis] for axis in "xyz"]) * factor + translation
            joint["origin"] = dict(zip("xyz", point.tolist(), strict=True))
            if joint["kind"] == "prismatic":
                joint["lower"] *= factor
                joint["upper"] *= factor
        metadata["articulation"] = articulation
        metadata["warnings"].append(
            "Joint travel, masses, clearances and unseen rails are inferred unless measured. "
            "Validate articulation motion in Isaac before use; GLB/OBJ/STL are rest-pose visuals only."
        )
    from .semantics import build_semantics

    metadata["semantics"] = build_semantics(spec, factor, translation, dimensions * factor)
    if spec.semantics is None:
        metadata["warnings"].append("Legacy recipe has no interaction semantics; annotate scene.semantics.")
    asset = BuiltAsset(visuals, collisions, metadata)
    if spec.articulation:
        from .articulation import check_drawer_clearance

        clearance = check_drawer_clearance(asset)
        asset.metadata["drawer_clearance"] = clearance
        if clearance.get("passed") is False:
            raise ValueError(f"Drawer colliders penetrate during travel: {clearance}")
    return asset
