#!/usr/bin/env python3
"""Generate an Isaac Sim-ready four-branch MyLifeUNIT-style mug tree asset.

The model is dimensioned in meters, Z-up, with the origin at the center of the
contact plane.  It uses a procedural hard-surface reconstruction because the
product consists of simple manufactured cylindrical parts; this yields cleaner
collision geometry and more reliable dimensions than single-view image-to-3D.

Outputs:
  - static and dynamic USDA assets with PBR textures and collision proxies
  - GLB and OBJ/MTL interchange files
  - preview images and a dimensions sheet
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import json
import zipfile
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import trimesh
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals


ROOT = Path(__file__).resolve().parent
TEX_DIR = ROOT / "textures"
TEX_DIR.mkdir(parents=True, exist_ok=True)

# Product dimensions (meters).  Overall dimensions reported by the listing:
# base diameter 6.3 in and height 13.8 in.
INCH = 0.0254
OVERALL_HEIGHT = 13.8 * INCH       # 0.35052 m
BASE_DIAMETER = 6.3 * INCH         # 0.16002 m
BASE_RADIUS = BASE_DIAMETER / 2.0
BASE_BOTTOM_Z = 0.0015
BASE_TOP_Z = 0.0205
POLE_RADIUS = 0.0127                # 1.0 in diameter
BRANCH_RADIUS = 0.0050              # ~10 mm diameter
BRANCH_EXPOSED = 0.105              # pole surface to tip
BRANCH_EMBED = 0.0075               # visual overlap into pole
BRANCH_ELEV_DEG = 32.0
BRANCH_LEVELS_ABOVE_BASE_TOP = (0.1155, 0.2055)  # measured branch centerlines
BRANCH_LEVELS = tuple(BASE_TOP_Z + h for h in BRANCH_LEVELS_ABOVE_BASE_TOP)
REMOVED_LEVEL = 0.306               # top threaded sockets, branches removed
MODEL_MASS_KG = 0.31                # product minus two removed hooks, approximate


@dataclass
class MeshData:
    name: str
    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray
    uv: np.ndarray
    material: str

    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.vertices.min(axis=0), self.vertices.max(axis=0)


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("Cannot normalize zero vector")
    return v / n


def basis_from_axis(axis: np.ndarray) -> np.ndarray:
    """Return a 3x3 matrix whose columns are local x, y, z in world space."""
    z = normalize(axis)
    helper = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(z, helper))) > 0.92:
        helper = np.array([1.0, 0.0, 0.0])
    x = normalize(np.cross(helper, z))
    y = normalize(np.cross(z, x))
    return np.column_stack((x, y, z))


def merge_meshes(name: str, meshes: List[MeshData], material: str | None = None) -> MeshData:
    if not meshes:
        raise ValueError("No meshes to merge")
    vertices, faces, normals, uv = [], [], [], []
    offset = 0
    for m in meshes:
        vertices.append(m.vertices)
        faces.append(m.faces + offset)
        normals.append(m.normals)
        uv.append(m.uv)
        offset += len(m.vertices)
    return MeshData(
        name=name,
        vertices=np.vstack(vertices),
        faces=np.vstack(faces),
        normals=np.vstack(normals),
        uv=np.vstack(uv),
        material=material or meshes[0].material,
    )


def make_beveled_cylinder(
    name: str,
    p0: np.ndarray,
    p1: np.ndarray,
    radius: float,
    bevel: float,
    segments: int,
    bevel_segments: int,
    material: str,
    texture_scale: float = 0.12,
) -> MeshData:
    """Create a capped, smoothly beveled cylinder between two points."""
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    axis = p1 - p0
    length = float(np.linalg.norm(axis))
    if length <= 2.0 * bevel:
        bevel = max(0.0, length * 0.15)
    basis = basis_from_axis(axis)

    # Rounded quarter-circle bevels in the radius-z profile.
    profile: List[Tuple[float, float]] = []
    for j in range(bevel_segments + 1):
        t = j / bevel_segments if bevel_segments else 1.0
        a = -0.5 * math.pi + 0.5 * math.pi * t
        profile.append((radius - bevel + bevel * math.cos(a), bevel + bevel * math.sin(a)))
    if length - bevel > profile[-1][1] + 1e-9:
        profile.append((radius, length - bevel))
    for j in range(1, bevel_segments + 1):
        t = j / bevel_segments
        a = 0.5 * math.pi * t
        profile.append((radius - bevel + bevel * math.cos(a), length - bevel + bevel * math.sin(a)))

    rings = len(profile)
    verts: List[np.ndarray] = []
    norms: List[np.ndarray] = []
    uvs: List[Tuple[float, float]] = []

    # Profile normals: outward normal in r-z plane is (dz, -dr).
    profile_normals: List[Tuple[float, float]] = []
    for i in range(rings):
        if i == 0:
            dr = profile[1][0] - profile[0][0]
            dz = profile[1][1] - profile[0][1]
        elif i == rings - 1:
            dr = profile[-1][0] - profile[-2][0]
            dz = profile[-1][1] - profile[-2][1]
        else:
            dr = profile[i + 1][0] - profile[i - 1][0]
            dz = profile[i + 1][1] - profile[i - 1][1]
        nr, nz = dz, -dr
        nn = math.hypot(nr, nz)
        profile_normals.append((nr / nn, nz / nn))

    for i, ((r, z), (nr, nz)) in enumerate(zip(profile, profile_normals)):
        for j in range(segments + 1):
            theta = 2.0 * math.pi * (j / segments)
            c, s = math.cos(theta), math.sin(theta)
            local = np.array([r * c, r * s, z])
            local_n = np.array([nr * c, nr * s, nz])
            verts.append(p0 + basis @ local)
            norms.append(normalize(basis @ local_n))
            uvs.append((j / segments, z / texture_scale))

    faces: List[Tuple[int, int, int]] = []
    stride = segments + 1
    for i in range(rings - 1):
        for j in range(segments):
            a = i * stride + j
            b = a + 1
            c = (i + 1) * stride + j + 1
            d = (i + 1) * stride + j
            faces.append((a, b, c))
            faces.append((a, c, d))

    # Caps use duplicated vertices so their normals remain flat.
    def add_cap(z: float, r: float, top: bool) -> None:
        nonlocal verts, norms, uvs, faces
        base_idx = len(verts)
        local_center = np.array([0.0, 0.0, z])
        cap_n_local = np.array([0.0, 0.0, 1.0 if top else -1.0])
        verts.append(p0 + basis @ local_center)
        norms.append(basis @ cap_n_local)
        uvs.append((0.5, 0.5))
        for j in range(segments + 1):
            theta = 2.0 * math.pi * (j / segments)
            c, s = math.cos(theta), math.sin(theta)
            local = np.array([r * c, r * s, z])
            verts.append(p0 + basis @ local)
            norms.append(basis @ cap_n_local)
            uvs.append((0.5 + 0.5 * c, 0.5 + 0.5 * s))
        for j in range(segments):
            center = base_idx
            a = base_idx + 1 + j
            b = base_idx + 1 + j + 1
            faces.append((center, a, b) if top else (center, b, a))

    add_cap(0.0, profile[0][0], top=False)
    add_cap(length, profile[-1][0], top=True)

    return MeshData(
        name=name,
        vertices=np.asarray(verts, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        normals=np.asarray(norms, dtype=np.float64),
        uv=np.asarray(uvs, dtype=np.float64),
        material=material,
    )


def make_lathed_base(name: str, segments: int = 96) -> MeshData:
    # Slightly rounded/chamfered disk profile matched to the photographed item.
    profile = [
        (0.0740, BASE_BOTTOM_Z),
        (0.0782, BASE_BOTTOM_Z + 0.0018),
        (BASE_RADIUS, BASE_BOTTOM_Z + 0.0048),
        (BASE_RADIUS, BASE_TOP_Z - 0.0050),
        (0.0785, BASE_TOP_Z - 0.0020),
        (0.0755, BASE_TOP_Z),
    ]
    verts: List[np.ndarray] = []
    norms: List[np.ndarray] = []
    uvs: List[Tuple[float, float]] = []
    faces: List[Tuple[int, int, int]] = []

    profile_normals: List[Tuple[float, float]] = []
    for i in range(len(profile)):
        if i == 0:
            dr = profile[1][0] - profile[0][0]
            dz = profile[1][1] - profile[0][1]
        elif i == len(profile) - 1:
            dr = profile[-1][0] - profile[-2][0]
            dz = profile[-1][1] - profile[-2][1]
        else:
            dr = profile[i + 1][0] - profile[i - 1][0]
            dz = profile[i + 1][1] - profile[i - 1][1]
        nr, nz = dz, -dr
        nn = math.hypot(nr, nz)
        profile_normals.append((nr / nn, nz / nn))

    stride = segments + 1
    for (r, z), (nr, nz) in zip(profile, profile_normals):
        for j in range(segments + 1):
            theta = 2.0 * math.pi * j / segments
            c, s = math.cos(theta), math.sin(theta)
            verts.append(np.array([r * c, r * s, z]))
            norms.append(np.array([nr * c, nr * s, nz]))
            uvs.append((j / segments * 2.0, z / 0.08))
    for i in range(len(profile) - 1):
        for j in range(segments):
            a = i * stride + j
            b = a + 1
            c = (i + 1) * stride + j + 1
            d = (i + 1) * stride + j
            faces.extend([(a, b, c), (a, c, d)])

    # Top and bottom planar caps with separate normals and UVs.
    def add_cap(z: float, r: float, top: bool) -> None:
        base_idx = len(verts)
        verts.append(np.array([0.0, 0.0, z]))
        norms.append(np.array([0.0, 0.0, 1.0 if top else -1.0]))
        uvs.append((0.5, 0.5))
        for j in range(segments + 1):
            theta = 2.0 * math.pi * j / segments
            c, s = math.cos(theta), math.sin(theta)
            verts.append(np.array([r * c, r * s, z]))
            norms.append(np.array([0.0, 0.0, 1.0 if top else -1.0]))
            uvs.append((0.5 + 0.5 * c, 0.5 + 0.5 * s))
        for j in range(segments):
            a, b = base_idx + 1 + j, base_idx + 2 + j
            faces.append((base_idx, a, b) if top else (base_idx, b, a))

    add_cap(BASE_TOP_Z, profile[-1][0], top=True)
    add_cap(BASE_BOTTOM_Z, profile[0][0], top=False)
    return MeshData(name, np.asarray(verts), np.asarray(faces), np.asarray(norms), np.asarray(uvs), "WoodBase")


def make_oriented_disc(
    name: str,
    center: np.ndarray,
    normal: np.ndarray,
    radius: float,
    material: str,
    segments: int = 48,
    inner_radius: float = 0.0,
) -> MeshData:
    center = np.asarray(center, dtype=np.float64)
    normal = normalize(normal)
    basis = basis_from_axis(normal)
    verts: List[np.ndarray] = []
    norms: List[np.ndarray] = []
    uvs: List[Tuple[float, float]] = []
    faces: List[Tuple[int, int, int]] = []
    if inner_radius <= 0.0:
        verts.append(center)
        norms.append(normal)
        uvs.append((0.5, 0.5))
        for j in range(segments + 1):
            a = 2.0 * math.pi * j / segments
            c, s = math.cos(a), math.sin(a)
            verts.append(center + basis[:, 0] * radius * c + basis[:, 1] * radius * s)
            norms.append(normal)
            uvs.append((0.5 + 0.5 * c, 0.5 + 0.5 * s))
        for j in range(segments):
            faces.append((0, 1 + j, 2 + j))
    else:
        for j in range(segments + 1):
            a = 2.0 * math.pi * j / segments
            c, s = math.cos(a), math.sin(a)
            for r in (inner_radius, radius):
                verts.append(center + basis[:, 0] * r * c + basis[:, 1] * r * s)
                norms.append(normal)
                uvs.append((0.5 + 0.5 * (r / radius) * c, 0.5 + 0.5 * (r / radius) * s))
        for j in range(segments):
            a = 2 * j
            b = a + 1
            c = a + 3
            d = a + 2
            faces.extend([(a, b, c), (a, c, d)])
    return MeshData(name, np.asarray(verts), np.asarray(faces), np.asarray(norms), np.asarray(uvs), material)


def make_hex_disc(name: str, center: np.ndarray, normal: np.ndarray, radius: float, material: str) -> MeshData:
    center = np.asarray(center, dtype=np.float64)
    normal = normalize(normal)
    basis = basis_from_axis(normal)
    verts = [center]
    norms = [normal]
    uvs = [(0.5, 0.5)]
    for j in range(7):
        a = 2.0 * math.pi * j / 6.0 + math.pi / 6.0
        c, s = math.cos(a), math.sin(a)
        verts.append(center + basis[:, 0] * radius * c + basis[:, 1] * radius * s)
        norms.append(normal)
        uvs.append((0.5 + 0.5 * c, 0.5 + 0.5 * s))
    faces = [(0, 1 + j, 2 + j) for j in range(6)]
    return MeshData(name, np.asarray(verts), np.asarray(faces), np.asarray(norms), np.asarray(uvs), material)


def branch_direction(azimuth_deg: float) -> np.ndarray:
    az = math.radians(azimuth_deg)
    el = math.radians(BRANCH_ELEV_DEG)
    return normalize(np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)]))


def branch_endpoints(z_surface: float, azimuth_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    az = math.radians(azimuth_deg)
    d = branch_direction(azimuth_deg)
    surface = np.array([POLE_RADIUS * math.cos(az), POLE_RADIUS * math.sin(az), z_surface])
    return surface - d * BRANCH_EMBED, surface + d * BRANCH_EXPOSED, d


def build_visual_meshes() -> List[MeshData]:
    meshes: List[MeshData] = []
    meshes.append(make_lathed_base("Base"))
    meshes.append(make_beveled_cylinder(
        "Pole", np.array([0.0, 0.0, BASE_TOP_Z - 0.004]), np.array([0.0, 0.0, OVERALL_HEIGHT]),
        POLE_RADIUS, bevel=0.0012, segments=72, bevel_segments=4, material="WoodLong", texture_scale=0.14))

    # Lower pair and middle pair are rotated by 90 degrees relative to each other.
    retained = [
        (BRANCH_LEVELS[0], 30.0),
        (BRANCH_LEVELS[0], 210.0),
        (BRANCH_LEVELS[1], 120.0),
        (BRANCH_LEVELS[1], 300.0),
    ]
    for idx, (z, az) in enumerate(retained, start=1):
        p0, p1, _ = branch_endpoints(z, az)
        meshes.append(make_beveled_cylinder(
            f"Branch_{idx:02d}", p0, p1, BRANCH_RADIUS, bevel=0.0008,
            segments=48, bevel_segments=4, material="WoodLong", texture_scale=0.09))

    # Four rubber pads underneath the base.
    for i, az in enumerate((45.0, 135.0, 225.0, 315.0), start=1):
        a = math.radians(az)
        c = np.array([0.054 * math.cos(a), 0.054 * math.sin(a), 0.0])
        meshes.append(make_beveled_cylinder(
            f"RubberPad_{i:02d}", c, c + np.array([0.0, 0.0, BASE_BOTTOM_Z]),
            radius=0.0075, bevel=0.00035, segments=32, bevel_segments=2,
            material="Rubber", texture_scale=0.03))

    # Visible threaded inserts where the top two branches were removed.
    socket_parts: List[MeshData] = []
    for i, az in enumerate((30.0, 210.0), start=1):
        d = branch_direction(az)
        a = math.radians(az)
        surface = np.array([POLE_RADIUS * math.cos(a), POLE_RADIUS * math.sin(a), REMOVED_LEVEL])
        # Tiny offset avoids z-fighting with the pole surface.
        center = surface + d * 0.00045
        socket_parts.append(make_oriented_disc(
            f"SocketRing_{i:02d}", center, d, radius=0.00425, inner_radius=0.00265,
            material="SocketMetal", segments=48))
        socket_parts.append(make_oriented_disc(
            f"SocketRecess_{i:02d}", center + d * 0.00008, d, radius=0.00260,
            material="SocketRecess", segments=48))
        socket_parts.append(make_hex_disc(
            f"SocketHex_{i:02d}", center + d * 0.00012, d, radius=0.00135,
            material="SocketMetal"))
    meshes.extend(socket_parts)
    return meshes


def make_simple_cylinder(name: str, p0: np.ndarray, p1: np.ndarray, radius: float, segments: int = 20) -> MeshData:
    return make_beveled_cylinder(name, p0, p1, radius, bevel=0.0, segments=segments,
                                 bevel_segments=0, material="Collision", texture_scale=1.0)


def build_collision_meshes() -> List[MeshData]:
    meshes: List[MeshData] = []
    meshes.append(make_simple_cylinder("BaseCollision", np.array([0.0, 0.0, 0.0]),
                                       np.array([0.0, 0.0, BASE_TOP_Z]), 0.0788, 32))
    meshes.append(make_simple_cylinder("PoleCollision", np.array([0.0, 0.0, BASE_TOP_Z - 0.005]),
                                       np.array([0.0, 0.0, OVERALL_HEIGHT]), POLE_RADIUS, 24))
    retained = [
        (BRANCH_LEVELS[0], 30.0),
        (BRANCH_LEVELS[0], 210.0),
        (BRANCH_LEVELS[1], 120.0),
        (BRANCH_LEVELS[1], 300.0),
    ]
    for idx, (z, az) in enumerate(retained, start=1):
        p0, p1, _ = branch_endpoints(z, az)
        meshes.append(make_simple_cylinder(f"BranchCollision_{idx:02d}", p0, p1, BRANCH_RADIUS, 16))
    return meshes


def periodic_noise(size: int, seed: int, anisotropy: Tuple[float, float]) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    x *= 2.0 * math.pi / size
    y *= 2.0 * math.pi / size
    field = np.zeros((size, size), dtype=np.float64)
    for octave in range(1, 7):
        for _ in range(5):
            kx = int(rng.integers(1, 5 * octave + 2))
            ky = int(rng.integers(1, 5 * octave + 2))
            amp = 1.0 / (octave ** 1.35)
            phase = float(rng.random() * 2.0 * math.pi)
            field += amp * np.sin(anisotropy[0] * kx * x + anisotropy[1] * ky * y + phase)
    field -= field.mean()
    field /= max(field.std(), 1e-9)
    return field


def normal_from_height(height: np.ndarray, strength: float) -> np.ndarray:
    dx = (np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)) * 0.5
    dy = (np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)) * 0.5
    nx = -dx * strength
    ny = -dy * strength
    nz = np.ones_like(height)
    n = np.stack([nx, ny, nz], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return np.clip((n * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)


def save_pbr_set(prefix: str, longitudinal: bool, size: int = 1024) -> Dict[str, Path]:
    rng_seed = 716 if longitudinal else 923
    if longitudinal:
        # Stronger variation across U than along V gives long, subtle wood fibers.
        n1 = periodic_noise(size, rng_seed, (1.0, 0.18))
        n2 = periodic_noise(size, rng_seed + 1, (2.4, 0.08))
        yy, xx = np.mgrid[0:size, 0:size]
        fibers = np.sin(2 * math.pi * (xx / size * 26.0 + 0.6 * n1))
        height = 0.62 * n1 + 0.24 * n2 + 0.14 * fibers
    else:
        n1 = periodic_noise(size, rng_seed, (1.0, 0.7))
        n2 = periodic_noise(size, rng_seed + 1, (2.0, 1.5))
        yy, xx = np.mgrid[0:size, 0:size]
        grain = np.sin(2 * math.pi * (xx / size * 17.0 + 0.38 * n1))
        height = 0.58 * n1 + 0.25 * n2 + 0.17 * grain
    height -= height.mean()
    height /= max(height.std(), 1e-9)

    base = np.array([25.0, 27.0, 30.0], dtype=np.float64)
    color_var = np.clip(height * 2.8 + n1 * 1.2, -6.0, 6.0)
    rgb = np.clip(base[None, None, :] + color_var[..., None] * np.array([0.84, 0.92, 1.0]), 5, 50).astype(np.uint8)
    rough = np.clip(108.0 + 11.0 * n1 - 4.0 * n2, 78, 142).astype(np.uint8)
    normal = normal_from_height(height, strength=0.48 if longitudinal else 0.38)
    mr = np.zeros((size, size, 4), dtype=np.uint8)
    mr[..., 0] = 255
    mr[..., 1] = rough
    mr[..., 2] = 0
    mr[..., 3] = 255

    paths = {
        "basecolor": TEX_DIR / f"{prefix}_basecolor.png",
        "roughness": TEX_DIR / f"{prefix}_roughness.png",
        "normal": TEX_DIR / f"{prefix}_normal.png",
        "metallicroughness": TEX_DIR / f"{prefix}_metallic_roughness.png",
    }
    if all(path.exists() for path in paths.values()):
        return paths
    Image.fromarray(rgb, "RGB").save(paths["basecolor"], optimize=True)
    Image.fromarray(rough, "L").save(paths["roughness"], optimize=True)
    Image.fromarray(normal, "RGB").save(paths["normal"], optimize=True)
    Image.fromarray(mr, "RGBA").save(paths["metallicroughness"], optimize=True)
    return paths


def generate_textures() -> Dict[str, Dict[str, Path]]:
    return {
        "WoodLong": save_pbr_set("black_painted_wood_long", longitudinal=True),
        "WoodBase": save_pbr_set("black_painted_wood_base", longitudinal=False),
    }


def mesh_to_trimesh(mesh: MeshData, materials: Dict[str, PBRMaterial]) -> trimesh.Trimesh:
    visual = TextureVisuals(uv=mesh.uv, material=materials[mesh.material])
    tm = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces,
                         vertex_normals=mesh.normals, visual=visual,
                         process=False, validate=False)
    tm.metadata["name"] = mesh.name
    return tm


def export_glb(visuals: List[MeshData], texture_paths: Dict[str, Dict[str, Path]]) -> Path:
    wood_long = texture_paths["WoodLong"]
    wood_base = texture_paths["WoodBase"]
    materials: Dict[str, PBRMaterial] = {
        "WoodLong": PBRMaterial(
            name="BlackPaintedWood_Longitudinal",
            baseColorTexture=Image.open(wood_long["basecolor"]),
            metallicRoughnessTexture=Image.open(wood_long["metallicroughness"]),
            normalTexture=Image.open(wood_long["normal"]),
            baseColorFactor=[1.0, 1.0, 1.0, 1.0], metallicFactor=0.0, roughnessFactor=1.0),
        "WoodBase": PBRMaterial(
            name="BlackPaintedWood_Base",
            baseColorTexture=Image.open(wood_base["basecolor"]),
            metallicRoughnessTexture=Image.open(wood_base["metallicroughness"]),
            normalTexture=Image.open(wood_base["normal"]),
            baseColorFactor=[1.0, 1.0, 1.0, 1.0], metallicFactor=0.0, roughnessFactor=1.0),
        "Rubber": PBRMaterial(name="Rubber", baseColorFactor=[0.012, 0.013, 0.015, 1.0],
                              metallicFactor=0.0, roughnessFactor=0.82),
        "SocketMetal": PBRMaterial(name="SocketMetal", baseColorFactor=[0.025, 0.027, 0.030, 1.0],
                                   metallicFactor=0.72, roughnessFactor=0.32),
        "SocketRecess": PBRMaterial(name="SocketRecess", baseColorFactor=[0.002, 0.002, 0.003, 1.0],
                                    metallicFactor=0.0, roughnessFactor=0.60),
    }
    scene = trimesh.Scene()
    for m in visuals:
        scene.add_geometry(mesh_to_trimesh(m, materials), node_name=m.name, geom_name=m.name)
    out = ROOT / "mylifeunit_mug_tree_4branch.glb"
    scene.export(out)
    return out


def export_obj(visuals: List[MeshData], texture_paths: Dict[str, Dict[str, Path]]) -> Tuple[Path, Path]:
    obj_path = ROOT / "mylifeunit_mug_tree_4branch.obj"
    mtl_path = ROOT / "mylifeunit_mug_tree_4branch.mtl"

    mtl = []
    def mat_block(name: str, kd: Tuple[float, float, float], rough: float, metallic: float,
                  tex: Dict[str, Path] | None = None) -> None:
        mtl.append(f"newmtl {name}\n")
        mtl.append(f"Kd {kd[0]:.6f} {kd[1]:.6f} {kd[2]:.6f}\n")
        mtl.append("Ka 0.000000 0.000000 0.000000\n")
        mtl.append("Ks 0.120000 0.120000 0.120000\n")
        ns = max(1.0, (1.0 - rough) * 500.0)
        mtl.append(f"Ns {ns:.3f}\n")
        mtl.append(f"Pm {metallic:.4f}\nPr {rough:.4f}\n")
        mtl.append("illum 2\n")
        if tex:
            rel = lambda p: Path("textures") / p.name
            mtl.append(f"map_Kd {rel(tex['basecolor']).as_posix()}\n")
            mtl.append(f"map_Pr {rel(tex['roughness']).as_posix()}\n")
            mtl.append(f"map_Bump -bm 0.35 {rel(tex['normal']).as_posix()}\n")
        mtl.append("\n")
    mat_block("WoodLong", (0.025, 0.027, 0.030), 0.42, 0.0, texture_paths["WoodLong"])
    mat_block("WoodBase", (0.025, 0.027, 0.030), 0.44, 0.0, texture_paths["WoodBase"])
    mat_block("Rubber", (0.008, 0.009, 0.010), 0.82, 0.0)
    mat_block("SocketMetal", (0.030, 0.032, 0.036), 0.32, 0.72)
    mat_block("SocketRecess", (0.002, 0.002, 0.003), 0.60, 0.0)
    mtl_path.write_text("".join(mtl), encoding="utf-8")

    lines = ["# MyLifeUNIT-style four-branch mug tree\n", f"mtllib {mtl_path.name}\n"]
    v_offset = 0
    vt_offset = 0
    vn_offset = 0
    for m in visuals:
        lines.append(f"\no {m.name}\nusemtl {m.material}\n")
        for v in m.vertices:
            lines.append(f"v {v[0]:.9f} {v[1]:.9f} {v[2]:.9f}\n")
        for uv in m.uv:
            lines.append(f"vt {uv[0]:.9f} {uv[1]:.9f}\n")
        for n in m.normals:
            lines.append(f"vn {n[0]:.9f} {n[1]:.9f} {n[2]:.9f}\n")
        for f in m.faces:
            ids = []
            for idx in f:
                vi = v_offset + int(idx) + 1
                ti = vt_offset + int(idx) + 1
                ni = vn_offset + int(idx) + 1
                ids.append(f"{vi}/{ti}/{ni}")
            lines.append("f " + " ".join(ids) + "\n")
        v_offset += len(m.vertices)
        vt_offset += len(m.uv)
        vn_offset += len(m.normals)
    obj_path.write_text("".join(lines), encoding="utf-8")
    return obj_path, mtl_path


def fmt_vec(v: np.ndarray, precision: int = 8) -> str:
    return "(" + ", ".join(f"{float(x):.{precision}g}" for x in v) + ")"


def chunked(items: List[str], n: int = 8, indent: str = "            ") -> str:
    return (",\n" + indent).join(", ".join(items[i:i+n]) for i in range(0, len(items), n))


def usd_mesh_block(mesh: MeshData, indent: str = "        ", collision: bool = False) -> str:
    mn, mx = mesh.bounds()
    api = ' (\n            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]\n        )' if collision else ""
    lines = [f'{indent}def Mesh "{mesh.name}"{api}\n{indent}{{\n']
    if collision:
        lines.append(f'{indent}    bool physics:collisionEnabled = true\n')
        lines.append(f'{indent}    uniform token physics:approximation = "convexHull"\n')
        lines.append(f'{indent}    token visibility = "invisible"\n')
        lines.append(f'{indent}    rel material:binding:physics = </MugTree/Looks/PhysicsMaterial>\n')
    else:
        lines.append(f'{indent}    rel material:binding = </MugTree/Looks/{mesh.material}>\n')
    lines.append(f'{indent}    float3[] extent = [{fmt_vec(mn)}, {fmt_vec(mx)}]\n')

    pts = [fmt_vec(v) for v in mesh.vertices]
    lines.append(f'{indent}    point3f[] points = [\n{indent}        {chunked(pts, 4, indent + "        ")}\n{indent}    ]\n')
    counts = ["3"] * len(mesh.faces)
    lines.append(f'{indent}    int[] faceVertexCounts = [\n{indent}        {chunked(counts, 24, indent + "        ")}\n{indent}    ]\n')
    idx = [str(int(i)) for i in mesh.faces.reshape(-1)]
    lines.append(f'{indent}    int[] faceVertexIndices = [\n{indent}        {chunked(idx, 24, indent + "        ")}\n{indent}    ]\n')
    normals = [fmt_vec(n) for n in mesh.normals]
    lines.append(f'{indent}    normal3f[] normals = [\n{indent}        {chunked(normals, 4, indent + "        ")}\n{indent}    ] (\n{indent}        interpolation = "vertex"\n{indent}    )\n')
    if not collision:
        uvs = [fmt_vec(t) for t in mesh.uv]
        lines.append(f'{indent}    texCoord2f[] primvars:st = [\n{indent}        {chunked(uvs, 6, indent + "        ")}\n{indent}    ] (\n{indent}        interpolation = "vertex"\n{indent}    )\n')
    lines.append(f'{indent}    uniform token subdivisionScheme = "none"\n')
    lines.append(f'{indent}}}\n')
    return "".join(lines)


def usd_textured_material(name: str, tex: Dict[str, Path]) -> str:
    base = Path("textures") / tex["basecolor"].name
    rough = Path("textures") / tex["roughness"].name
    normal = Path("textures") / tex["normal"].name
    p = f"/MugTree/Looks/{name}"
    return f'''        def Material "{name}"
        {{
            token outputs:surface.connect = <{p}/PreviewSurface.outputs:surface>

            def Shader "PreviewSurface"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor.connect = <{p}/BaseColor.outputs:rgb>
                float inputs:metallic = 0
                float inputs:roughness.connect = <{p}/Roughness.outputs:r>
                normal3f inputs:normal.connect = <{p}/Normal.outputs:rgb>
                float inputs:ior = 1.5
                token outputs:surface
            }}

            def Shader "PrimvarReader"
            {{
                uniform token info:id = "UsdPrimvarReader_float2"
                string inputs:varname = "st"
                float2 outputs:result
            }}

            def Shader "BaseColor"
            {{
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @{base.as_posix()}@
                token inputs:sourceColorSpace = "sRGB"
                float2 inputs:st.connect = <{p}/PrimvarReader.outputs:result>
                token inputs:wrapS = "repeat"
                token inputs:wrapT = "repeat"
                float3 outputs:rgb
            }}

            def Shader "Roughness"
            {{
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @{rough.as_posix()}@
                token inputs:sourceColorSpace = "raw"
                float2 inputs:st.connect = <{p}/PrimvarReader.outputs:result>
                token inputs:wrapS = "repeat"
                token inputs:wrapT = "repeat"
                float outputs:r
            }}

            def Shader "Normal"
            {{
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @{normal.as_posix()}@
                token inputs:sourceColorSpace = "raw"
                float4 inputs:scale = (2, 2, 2, 1)
                float4 inputs:bias = (-1, -1, -1, 0)
                float2 inputs:st.connect = <{p}/PrimvarReader.outputs:result>
                token inputs:wrapS = "repeat"
                token inputs:wrapT = "repeat"
                float3 outputs:rgb
            }}
        }}
'''


def usd_constant_material(name: str, color: Tuple[float, float, float], roughness: float, metallic: float) -> str:
    p = f"/MugTree/Looks/{name}"
    return f'''        def Material "{name}"
        {{
            token outputs:surface.connect = <{p}/PreviewSurface.outputs:surface>
            def Shader "PreviewSurface"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = ({color[0]}, {color[1]}, {color[2]})
                float inputs:metallic = {metallic}
                float inputs:roughness = {roughness}
                float inputs:ior = 1.5
                token outputs:surface
            }}
        }}
'''


def export_usda(visuals: List[MeshData], collisions: List[MeshData], texture_paths: Dict[str, Dict[str, Path]], dynamic: bool) -> Path:
    suffix = "dynamic" if dynamic else "static"
    out = ROOT / f"mylifeunit_mug_tree_4branch_{suffix}.usda"
    api = ' (\n        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]\n    )' if dynamic else ""
    lines = [
        '#usda 1.0\n',
        '(\n',
        '    defaultPrim = "MugTree"\n',
        '    metersPerUnit = 1\n',
        '    kilogramsPerUnit = 1\n',
        '    upAxis = "Z"\n',
        ')\n\n',
        f'def Xform "MugTree"{api}\n{{\n',
    ]
    if dynamic:
        lines += [
            '    bool physics:rigidBodyEnabled = true\n',
            f'    float physics:mass = {MODEL_MASS_KG}\n',
            '    point3f physics:centerOfMass = (0, 0, 0.105)\n',
        ]
    lines.append('    custom string asset:description = "MyLifeUNIT-style mug tree with top pair of branches removed"\n')
    lines.append(f'    custom double asset:overallHeightMeters = {OVERALL_HEIGHT}\n')
    lines.append(f'    custom double asset:baseDiameterMeters = {BASE_DIAMETER}\n')
    lines.append('    custom string asset:units = "meters"\n\n')

    lines.append('    def Scope "Looks"\n    {\n')
    lines.append(usd_textured_material("WoodLong", texture_paths["WoodLong"]))
    lines.append(usd_textured_material("WoodBase", texture_paths["WoodBase"]))
    lines.append(usd_constant_material("Rubber", (0.010, 0.011, 0.013), 0.82, 0.0))
    lines.append(usd_constant_material("SocketMetal", (0.025, 0.027, 0.031), 0.32, 0.72))
    lines.append(usd_constant_material("SocketRecess", (0.0015, 0.0015, 0.0020), 0.60, 0.0))
    lines.append('        def Material "PhysicsMaterial" (\n            prepend apiSchemas = ["PhysicsMaterialAPI"]\n        )\n        {\n')
    lines.append('            float physics:staticFriction = 0.56\n')
    lines.append('            float physics:dynamicFriction = 0.46\n')
    lines.append('            float physics:restitution = 0.03\n')
    lines.append('        }\n')
    lines.append('    }\n\n')

    lines.append('    def Xform "Visuals"\n    {\n')
    for m in visuals:
        lines.append(usd_mesh_block(m, indent="        ", collision=False))
    lines.append('    }\n\n')

    lines.append('    def Xform "Collisions"\n    {\n')
    for m in collisions:
        lines.append(usd_mesh_block(m, indent="        ", collision=True))
    lines.append('    }\n')
    lines.append('}\n')
    out.write_text("".join(lines), encoding="utf-8")
    return out


# ------------------------------- Preview renderer -------------------------------

def load_texture_array(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def look_at(camera: np.ndarray, target: np.ndarray, up: np.ndarray = np.array([0.0, 0.0, 1.0])) -> Tuple[np.ndarray, np.ndarray]:
    forward = normalize(target - camera)
    right = normalize(np.cross(forward, up))
    true_up = normalize(np.cross(right, forward))
    # Camera coordinates: x right, y up, positive z forward.
    R = np.vstack([right, true_up, forward])
    t = -R @ camera
    return R, t


def rasterize_scene(
    visuals: List[MeshData], texture_paths: Dict[str, Dict[str, Path]], out_path: Path,
    camera: np.ndarray, target: np.ndarray, width: int = 900, height: int = 900,
    fov_deg: float = 34.0, transparent: bool = False,
    show_collision: bool = False, collisions: List[MeshData] | None = None,
) -> None:
    # 2x supersampling for smooth silhouettes.
    scale = 1
    W, H = width * scale, height * scale
    if transparent:
        bg = np.zeros((H, W, 4), dtype=np.float64)
    else:
        yy = np.linspace(0.0, 1.0, H)[:, None]
        top = np.array([0.94, 0.95, 0.97])
        bottom = np.array([0.76, 0.79, 0.83])
        rgb = top[None, None, :] * (1 - yy[..., None]) + bottom[None, None, :] * yy[..., None]
        bg = np.ones((H, W, 4), dtype=np.float64)
        bg[..., :3] = np.broadcast_to(rgb, (H, W, 3))
    zbuf = np.full((H, W), np.inf, dtype=np.float64)
    img = bg.copy()

    tex_arrays = {
        "WoodLong": load_texture_array(texture_paths["WoodLong"]["basecolor"]),
        "WoodBase": load_texture_array(texture_paths["WoodBase"]["basecolor"]),
    }
    base_colors = {
        "Rubber": np.array([0.008, 0.009, 0.011]),
        "SocketMetal": np.array([0.045, 0.047, 0.052]),
        "SocketRecess": np.array([0.001, 0.001, 0.0015]),
        "Collision": np.array([0.1, 0.75, 0.9]),
    }
    roughness = {"WoodLong": 0.40, "WoodBase": 0.43, "Rubber": 0.82,
                 "SocketMetal": 0.30, "SocketRecess": 0.65, "Collision": 0.45}
    metallic = {"SocketMetal": 0.72}

    R, t = look_at(camera, target)
    focal = 0.5 * W / math.tan(math.radians(fov_deg) * 0.5)
    cx, cy = W * 0.5, H * 0.53
    light_dirs = [normalize(np.array([0.45, -0.65, 0.62])), normalize(np.array([-0.55, -0.18, 0.36]))]
    light_strength = [1.10, 0.42]
    view_pos = camera

    render_meshes = list(visuals)
    if show_collision and collisions:
        render_meshes += collisions

    def sample_material(mat: str, uv: np.ndarray) -> np.ndarray:
        if mat in tex_arrays:
            tex = tex_arrays[mat]
            h, w = tex.shape[:2]
            u = uv[..., 0] % 1.0
            v = uv[..., 1] % 1.0
            ix = np.clip((u * (w - 1)).astype(np.int64), 0, w - 1)
            iy = np.clip(((1.0 - v) * (h - 1)).astype(np.int64), 0, h - 1)
            # Lift the nearly black texture slightly for a readable preview.
            return np.clip(tex[iy, ix] * 1.18 + 0.004, 0, 1)
        return np.broadcast_to(base_colors.get(mat, np.array([0.15, 0.15, 0.15])), uv.shape[:-1] + (3,))

    for mesh in render_meshes:
        verts_cam = (R @ mesh.vertices.T).T + t
        z = verts_cam[:, 2]
        valid_v = z > 0.01
        proj = np.empty((len(mesh.vertices), 2), dtype=np.float64)
        proj[:, 0] = cx + focal * verts_cam[:, 0] / np.maximum(z, 1e-8)
        proj[:, 1] = cy - focal * verts_cam[:, 1] / np.maximum(z, 1e-8)

        for face in mesh.faces:
            if not np.all(valid_v[face]):
                continue
            p = proj[face]
            # Signed area; skip backfaces except for collision overlay.
            area = (p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1]) - (p[1, 1] - p[0, 1]) * (p[2, 0] - p[0, 0])
            if abs(area) < 1e-8:
                continue
            if area >= 0 and mesh.material != "Collision":
                continue
            xmin = max(0, int(math.floor(p[:, 0].min())))
            xmax = min(W - 1, int(math.ceil(p[:, 0].max())))
            ymin = max(0, int(math.floor(p[:, 1].min())))
            ymax = min(H - 1, int(math.ceil(p[:, 1].max())))
            if xmax < xmin or ymax < ymin:
                continue
            xs = np.arange(xmin, xmax + 1)
            ys = np.arange(ymin, ymax + 1)
            X, Y = np.meshgrid(xs, ys)
            x0, y0 = p[0]
            x1, y1 = p[1]
            x2, y2 = p[2]
            denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if abs(denom) < 1e-12:
                continue
            w0 = ((y1 - y2) * (X - x2) + (x2 - x1) * (Y - y2)) / denom
            w1 = ((y2 - y0) * (X - x2) + (x0 - x2) * (Y - y2)) / denom
            w2 = 1.0 - w0 - w1
            inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
            if not inside.any():
                continue
            zz = w0 * z[face[0]] + w1 * z[face[1]] + w2 * z[face[2]]
            zregion = zbuf[ymin:ymax + 1, xmin:xmax + 1]
            mask = inside & (zz < zregion)
            if not mask.any():
                continue

            uv = (w0[..., None] * mesh.uv[face[0]] +
                  w1[..., None] * mesh.uv[face[1]] +
                  w2[..., None] * mesh.uv[face[2]])
            n = (w0[..., None] * mesh.normals[face[0]] +
                 w1[..., None] * mesh.normals[face[1]] +
                 w2[..., None] * mesh.normals[face[2]])
            n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
            world_p = (w0[..., None] * mesh.vertices[face[0]] +
                       w1[..., None] * mesh.vertices[face[1]] +
                       w2[..., None] * mesh.vertices[face[2]])
            albedo = sample_material(mesh.material, uv)
            ambient = 0.15
            lit = np.full_like(albedo, ambient)
            for ld, strength_l in zip(light_dirs, light_strength):
                ndotl = np.maximum(0.0, np.sum(n * ld[None, None, :], axis=-1))[..., None]
                lit += albedo * ndotl * strength_l
            vdir = view_pos[None, None, :] - world_p
            vdir /= np.maximum(np.linalg.norm(vdir, axis=-1, keepdims=True), 1e-9)
            hdir = normalize(light_dirs[0] + normalize(camera - target))
            ndoth = np.maximum(0.0, np.sum(n * hdir[None, None, :], axis=-1))[..., None]
            rgh = roughness.get(mesh.material, 0.5)
            spec_power = max(4.0, 130.0 * (1.0 - rgh) ** 2)
            spec = ndoth ** spec_power * (0.13 + 0.55 * metallic.get(mesh.material, 0.0))
            shaded = np.clip(lit + spec, 0, 1)
            if mesh.material == "Collision":
                shaded = np.broadcast_to(np.array([0.10, 0.78, 0.92]), shaded.shape)
                alpha = 0.40
            else:
                alpha = 1.0
            reg = img[ymin:ymax + 1, xmin:xmax + 1]
            if alpha < 1.0:
                reg[..., :3][mask] = reg[..., :3][mask] * (1.0 - alpha) + shaded[mask] * alpha
            else:
                reg[..., :3][mask] = shaded[mask]
            reg[..., 3][mask] = 1.0
            zregion[mask] = zz[mask]

    # Simple contact shadow under the base.
    if not transparent and not show_collision:
        yy, xx = np.mgrid[0:H, 0:W]
        # Project origin and a nearby floor point to estimate the ground ellipse location.
        origin_cam = R @ np.array([0.0, 0.0, 0.0]) + t
        ox = cx + focal * origin_cam[0] / origin_cam[2]
        oy = cy - focal * origin_cam[1] / origin_cam[2]
        sx, sy = W * 0.12, H * 0.022
        shadow = np.exp(-(((xx - ox) / sx) ** 2 + ((yy - (oy + H * 0.008)) / sy) ** 2) * 2.0)
        shadow *= (zbuf == np.inf)
        img[..., :3] *= (1.0 - 0.16 * shadow[..., None])

    out = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    im = Image.fromarray(out, "RGBA")
    im = im.resize((width, height), Image.Resampling.LANCZOS)
    if not transparent:
        im = im.convert("RGB")
    im.save(out_path, quality=95)


def create_dimension_sheet(preview_path: Path, out_path: Path) -> None:
    canvas = Image.new("RGB", (1600, 1120), (247, 248, 250))
    draw = ImageDraw.Draw(canvas)
    try:
        font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 44)
        font = ImageFont.truetype("DejaVuSans.ttf", 27)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 21)
    except OSError:
        font_title = font = font_small = ImageFont.load_default()
    prev = Image.open(preview_path).convert("RGB")
    prev.thumbnail((820, 820), Image.Resampling.LANCZOS)
    canvas.paste(prev, (45, 105))
    draw.text((45, 30), "Four-branch mug tree — reconstruction dimensions", fill=(25, 28, 34), font=font_title)

    panel = (880, 92, 1555, 1075)
    draw.rounded_rectangle(panel, radius=22, outline=(200, 203, 210), width=3)
    x0 = 915
    y = 120
    rows = [
        ("Overall height", f"{OVERALL_HEIGHT*1000:.1f} mm  (13.8 in)"),
        ("Base maximum diameter", f"{BASE_DIAMETER*1000:.1f} mm  (6.3 in)"),
        ("Base thickness", f"{(BASE_TOP_Z-BASE_BOTTOM_Z)*1000:.1f} mm"),
        ("Main pole diameter", f"{POLE_RADIUS*2*1000:.1f} mm"),
        ("Branch diameter", f"{BRANCH_RADIUS*2*1000:.1f} mm"),
        ("Exposed branch length", f"{BRANCH_EXPOSED*1000:.1f} mm"),
        ("Branch elevation", f"{BRANCH_ELEV_DEG:.1f}° above horizontal"),
        ("Branch levels above base top", f"{BRANCH_LEVELS_ABOVE_BASE_TOP[0]*1000:.1f}, {BRANCH_LEVELS_ABOVE_BASE_TOP[1]*1000:.1f} mm"),
        ("Branch levels from bottom", f"{BRANCH_LEVELS[0]*1000:.1f}, {BRANCH_LEVELS[1]*1000:.1f} mm"),
        ("Removed top sockets", f"{REMOVED_LEVEL*1000:.0f} mm"),
        ("Coordinate system", "meters; Z-up; origin at contact center"),
    ]
    for label, value in rows:
        draw.text((x0, y), label, fill=(75, 80, 90), font=font_small)
        draw.text((x0, y + 27), value, fill=(18, 20, 25), font=font)
        y += 76
    draw.text((915, 1025), "Top pair removed; threaded sockets retained.", fill=(75, 80, 90), font=font_small)
    canvas.save(out_path, quality=95)

def write_readme(files: Dict[str, str]) -> Path:
    readme = ROOT / "README.md"
    text = f"""# MyLifeUNIT-style mug tree — four-branch reconstruction

This is a clean, dimensioned reconstruction of the photographed black wooden mug tree with the **top two branches removed** and the four middle/lower branches retained.  The two exposed threaded sockets are modeled visually.

## Primary Isaac Sim asset

- `mylifeunit_mug_tree_4branch.usda`: small wrapper pointing to the recommended static asset.
- `mylifeunit_mug_tree_4branch_static.usda`: static collider; use this when the stand is fixed on a table.
- `mylifeunit_mug_tree_4branch_dynamic.usda`: rigid body with an approximate mass of {MODEL_MASS_KG:.2f} kg.

Both assets use **meters**, **Z-up**, and place the origin at the center of the bottom contact plane.  The USDA files contain separate visual meshes and low-poly convex collision proxies.

## Dimensions used

| Parameter | Value |
|---|---:|
| Overall height | {OVERALL_HEIGHT*1000:.2f} mm |
| Base maximum diameter | {BASE_DIAMETER*1000:.2f} mm |
| Base thickness | {(BASE_TOP_Z-BASE_BOTTOM_Z)*1000:.2f} mm |
| Main pole diameter | {POLE_RADIUS*2*1000:.2f} mm |
| Branch diameter | {BRANCH_RADIUS*2*1000:.2f} mm |
| Exposed branch length | {BRANCH_EXPOSED*1000:.2f} mm |
| Branch elevation | {BRANCH_ELEV_DEG:.1f} degrees |
| Branch centerlines above base top | {BRANCH_LEVELS_ABOVE_BASE_TOP[0]*1000:.1f}, {BRANCH_LEVELS_ABOVE_BASE_TOP[1]*1000:.1f} mm |
| Branch centerlines from bottom contact plane | {BRANCH_LEVELS[0]*1000:.1f}, {BRANCH_LEVELS[1]*1000:.1f} mm |
| Removed top socket level | {REMOVED_LEVEL*1000:.1f} mm |

The overall height and base diameter follow the product listing. The retained branch centerlines use the user-corrected measurements of 115.5 mm and 205.5 mm above the base top (136.0 mm and 226.0 mm from the bottom contact plane). The remaining smaller component dimensions were reconstructed from the supplied photograph and product imagery and are exposed as constants at the top of `generate_mug_tree_asset.py`.

## Isaac Lab example

```python
from isaaclab.assets import AssetBaseCfg
from isaaclab.sim import UsdFileCfg

MUG_TREE_CFG = AssetBaseCfg(
    prim_path="/World/envs/env_.*/MugTree",
    spawn=UsdFileCfg(
        usd_path="/absolute/path/to/mylifeunit_mug_tree_4branch_static.usda",
        scale=(1.0, 1.0, 1.0),
    ),
    init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
)
```

Do not apply an additional `0.01` scale: the asset is already authored in meters.

## Files

- USDA: Isaac Sim visual materials and collision geometry
- GLB: embedded PBR interchange asset
- OBJ/MTL: editable interchange asset with external PNG maps
- `textures/`: base color, roughness, normal, and packed metallic/roughness maps
- `generate_mug_tree_asset.py`: fully parametric rebuild script
- previews: appearance, collision overlay, and dimension reference
"""
    readme.write_text(text, encoding="utf-8")
    return readme


def validate_outputs(visuals: List[MeshData], glb_path: Path, usda_paths: List[Path]) -> Dict[str, object]:
    report: Dict[str, object] = {}
    scene = trimesh.load(glb_path, force="scene", process=False)
    bounds = scene.bounds
    report["glb_geometry_count"] = len(scene.geometry)
    report["glb_bounds_m"] = bounds.tolist()
    report["glb_extents_m"] = (bounds[1] - bounds[0]).tolist()
    report["visual_meshes"] = len(visuals)
    report["visual_triangles"] = int(sum(len(m.faces) for m in visuals))
    report["usda_files"] = {p.name: p.stat().st_size for p in usda_paths}
    report["height_error_mm"] = float(abs(bounds[1, 2] - OVERALL_HEIGHT) * 1000.0)
    report["origin_floor_error_mm"] = float(abs(bounds[0, 2]) * 1000.0)
    report["branch_centerlines_above_base_top_mm"] = [round(float(h * 1000.0), 3) for h in BRANCH_LEVELS_ABOVE_BASE_TOP]
    report["branch_centerlines_from_bottom_mm"] = [round(float(z * 1000.0), 3) for z in BRANCH_LEVELS]
    report_path = ROOT / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def make_zip() -> Path:
    out = ROOT.parent / "mylifeunit_mug_tree_4branch_isaacsim.zip"
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(ROOT.rglob("*")):
            if path.is_file() and path.name != out.name and "__pycache__" not in path.parts:
                zf.write(path, arcname=Path(ROOT.name) / path.relative_to(ROOT))
    return out


def main() -> None:
    texture_paths = generate_textures()
    visuals = build_visual_meshes()
    collisions = build_collision_meshes()
    glb_path = export_glb(visuals, texture_paths)
    obj_path, mtl_path = export_obj(visuals, texture_paths)
    static_usda = export_usda(visuals, collisions, texture_paths, dynamic=False)
    dynamic_usda = export_usda(visuals, collisions, texture_paths, dynamic=True)
    primary_usda = ROOT / "mylifeunit_mug_tree_4branch.usda"
    primary_usda.write_text(
        '#usda 1.0\n(\n    defaultPrim = "MugTree"\n    metersPerUnit = 1\n    upAxis = "Z"\n)\n\n'
        'def Xform "MugTree" (\n    prepend references = @mylifeunit_mug_tree_4branch_static.usda@</MugTree>\n)\n{\n}\n',
        encoding="utf-8")

    preview = ROOT / "preview_perspective.png"
    rasterize_scene(
        visuals, texture_paths, preview,
        camera=np.array([0.69546659, -0.18634971, 0.46]), target=np.array([0.0, 0.0, 0.17]),
        width=900, height=900, fov_deg=30.0)
    front = ROOT / "preview_front.png"
    rasterize_scene(
        visuals, texture_paths, front,
        camera=np.array([0.71726018, -0.06275213, 0.44]), target=np.array([0.0, 0.0, 0.17]),
        width=760, height=900, fov_deg=29.0)
    collision_preview = ROOT / "preview_collision_overlay.png"
    rasterize_scene(
        visuals, texture_paths, collision_preview,
        camera=np.array([0.69546659, -0.18634971, 0.46]), target=np.array([0.0, 0.0, 0.17]),
        width=900, height=900, fov_deg=30.0, show_collision=True, collisions=collisions)
    create_dimension_sheet(preview, ROOT / "dimension_reference.png")

    readme = write_readme({})
    report = validate_outputs(visuals, glb_path, [static_usda, dynamic_usda])
    zip_path = make_zip()
    print(json.dumps({
        "root": str(ROOT),
        "zip": str(zip_path),
        "glb": str(glb_path),
        "static_usda": str(static_usda),
        "dynamic_usda": str(dynamic_usda),
        "primary_usda": str(primary_usda),
        "obj": str(obj_path),
        "mtl": str(mtl_path),
        "readme": str(readme),
        "validation": report,
    }, indent=2))


if __name__ == "__main__":
    main()
