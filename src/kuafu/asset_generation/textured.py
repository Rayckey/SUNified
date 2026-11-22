"""Import a SAM-style textured GLB visual while retaining editable convex colliders."""

import json
import shutil
from pathlib import Path

import numpy as np
import trimesh

from .articulation import isaaclab_snippet, posed_asset
from .export import ISAACLAB_SNIPPET, export_usda
from .geometry import BuiltAsset, build_asset
from .io import digest, fresh_directory, write_json
from .render import render_views
from .schema import PhysicsConfig, Scale, SceneSpec
from .semantics import SEMANTICS_SNIPPET, export_semantics


def load_textured_glb(path: Path) -> list[trimesh.Trimesh]:
    if path.suffix.lower() != ".glb":
        raise ValueError("Supply a self-contained textured GLB")
    scene = trimesh.load(path, force="scene", process=False)
    meshes = []
    for node in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node]
        mesh = scene.geometry[geometry_name].copy()
        if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
            raise ValueError("The GLB must contain triangle meshes")
        mesh.apply_transform(transform)
        visual = mesh.visual
        if (
            visual.kind != "texture"
            or visual.uv is None
            or np.shape(visual.uv) != (len(mesh.vertices), 2)
            or not np.isfinite(visual.uv).all()
            or not np.isfinite(mesh.vertices).all()
        ):
            raise ValueError("Every visual mesh needs finite vertices and per-vertex UV coordinates")
        material = visual.material
        if not isinstance(material, trimesh.visual.material.PBRMaterial) or material.baseColorTexture is None:
            raise ValueError("Every visual mesh needs an embedded PBR base-color texture")
        if (material.alphaMode or "OPAQUE") != "OPAQUE":
            raise ValueError("This SAM 3D importer supports opaque base-color materials only")
        for attribute in ("normalTexture", "metallicRoughnessTexture", "occlusionTexture", "emissiveTexture"):
            if getattr(material, attribute, None) is not None:
                raise ValueError(f"Unsupported {attribute}; refusing to silently discard a material map")
        if material.emissiveFactor is not None and np.any(material.emissiveFactor):
            raise ValueError("Emissive materials are not supported by this importer")
        mesh.metadata.update(
            source_node=node,
            roughness=1.0 if material.roughnessFactor is None else float(material.roughnessFactor),
            metallic=1.0 if material.metallicFactor is None else float(material.metallicFactor),
            smooth_shading=True,
            double_sided=bool(material.doubleSided),
        )
        meshes.append(mesh)
    if not meshes or sum(len(m.faces) for m in meshes) > 250000:
        raise ValueError("GLB needs 1..250,000 total visual triangles; simplify oversized meshes first")
    return meshes


def align_visuals(meshes, target_dimensions, rotation_deg=(0, 0, 0), scale_axis="longest"):
    """Uniform size estimate + bottom-center origin, NOT solved camera/shape alignment."""
    rotation_deg = np.asarray(rotation_deg, dtype=float)
    if rotation_deg.shape != (3,) or not np.isfinite(rotation_deg).all():
        raise ValueError("Visual rotation needs three finite Euler angles in degrees")
    if scale_axis not in ("x", "y", "z", "longest"):
        raise ValueError("Visual scale axis must be x, y, z, or longest")
    # Official SAM 3D GLB is Y-up. Other pipeline artifacts use meters/Z-up.
    to_z = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
    rotation = trimesh.transformations.euler_matrix(*np.deg2rad(rotation_deg), axes="sxyz")
    matrix = rotation @ to_z
    for mesh in meshes:
        mesh.apply_transform(matrix)
    bounds = np.array(
        [np.min([m.bounds[0] for m in meshes], axis=0), np.max([m.bounds[1] for m in meshes], axis=0)]
    )
    dimensions = bounds[1] - bounds[0]
    if dimensions.min() <= 1e-7:
        raise ValueError("Visual must have nonzero dimensions on all three axes")
    target = np.asarray(target_dimensions)
    if scale_axis == "longest":
        scale = target.max() / dimensions.max()
    else:
        axis = "xyz".index(scale_axis)
        scale = target[axis] / dimensions[axis]
    translation = -np.array([*bounds.mean(axis=0)[:2], bounds[0, 2]]) * scale
    size_transform = np.eye(4)
    size_transform[:3, :3] *= scale
    size_transform[:3, 3] = translation
    for mesh in meshes:
        mesh.apply_transform(size_transform)
    return size_transform @ matrix, dimensions * scale


def build_textured_asset(
    visual_path: Path,
    collision_scene: Path,
    out: Path,
    physics: PhysicsConfig,
    rotation_deg=(0, 0, 0),
    scale_axis="longest",
    scale: Scale | None = None,
    visual_links: dict[str, str] | None = None,
) -> dict:
    spec = SceneSpec.model_validate_json(collision_scene.read_text())
    collision_asset = build_asset(spec, scale)
    visuals = load_textured_glb(visual_path)
    if spec.articulation:
        links = {link.name for link in spec.articulation.links}
        nodes = {mesh.metadata["source_node"] for mesh in visuals}
        if visual_links is None or set(visual_links) != nodes or set(visual_links.values()) != links:
            raise ValueError(
                "Articulated GLB requires a visual_links mapping of every mesh node to a defined link; "
                "all links must have visuals. Segment a monolithic generated mesh before importing."
            )
        for mesh in visuals:
            mesh.metadata["link"] = visual_links[mesh.metadata["source_node"]]
    transform, dimensions = align_visuals(
        visuals, collision_asset.metadata["dimensions_m"], rotation_deg, scale_axis
    )
    # Validate everything before opening a new output directory. Never alter old final assets.
    provenance_path = visual_path.parent / "provenance.json"
    provenance = None
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text())
        if provenance.get("visual_sha256") != digest(visual_path):
            raise ValueError("Visual GLB no longer matches its inference provenance")
    out = fresh_directory(out)
    (out / "textures").mkdir()
    scene = trimesh.Scene()
    obj, mtl, offset = ["# meters, Z-up; textured visual only\nmtllib asset.mtl\n"], [], 1
    for i, mesh in enumerate(visuals):
        relative = f"textures/basecolor_{i:03d}.png"
        mesh.visual.material.baseColorTexture.save(out / relative)
        mesh.metadata["texture_file"] = relative
        name = f"Material_{i}"
        factor = mesh.visual.material.baseColorFactor
        factor = np.ones(4) if factor is None else np.asarray(factor) / 255
        mtl.append(
            f"newmtl {name}\nKd {factor[0]:.8g} {factor[1]:.8g} {factor[2]:.8g}\nmap_Kd {relative}\n\n"
        )
        obj.extend(
            [
                f"o Part_{i:03d}\nusemtl {name}\n",
                "".join(f"v {x:.9g} {y:.9g} {z:.9g}\n" for x, y, z in mesh.vertices),
                "".join(f"vt {u:.9g} {v:.9g}\n" for u, v in mesh.visual.uv),
                "".join(
                    "f " + " ".join(f"{v + offset}/{v + offset}" for v in face) + "\n" for face in mesh.faces
                ),
            ]
        )
        offset += len(mesh.vertices)
        y_up = mesh.copy()
        y_up.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
        scene.add_geometry(y_up, node_name=mesh.metadata["source_node"], geom_name=f"Part_{i:03d}")
    asset = BuiltAsset(visuals, collision_asset.collisions, collision_asset.metadata.copy())
    export_usda(asset, out, physics)
    scene.export(out / "asset.glb")
    (out / "asset.obj").write_text("".join(obj))
    (out / "asset.mtl").write_text("".join(mtl))
    render_views(asset, out)
    if spec.articulation:
        render_views(posed_asset(asset), out / "articulation_open")
    export_semantics(asset, out)
    (out / "isaaclab_snippet.py").write_text(
        (isaaclab_snippet(asset) if spec.articulation else ISAACLAB_SNIPPET) + SEMANTICS_SNIPPET
    )
    write_json(out / "collision_scene.json", spec)
    if provenance is not None:
        shutil.copy2(provenance_path, out / "inference_provenance.json")
    mismatch = (dimensions / np.asarray(collision_asset.metadata["dimensions_m"]) - 1).tolist()
    metadata = {
        **collision_asset.metadata,
        "dimensions_m": dimensions.tolist(),
        "collision_dimensions_m": collision_asset.metadata["dimensions_m"],
        "visual_triangles": sum(len(m.faces) for m in visuals),
        "source_visual_sha256": digest(visual_path),
        "collision_scene_sha256": digest(collision_scene),
        "visual_from_glb_matrix": transform.tolist(),
        "visual_scale_axis": scale_axis,
        "visual_links": visual_links,
        "visual_collision_size_relative_difference": mismatch,
        "texture_status": "UV base-color textures preserved; resemblance needs multi-view review",
        "texture_source": provenance.get("texture_source")
        if provenance
        else "imported GLB; unverified origin",
        "physics": physics.model_dump(),
        "effective_mass_kg": physics.mass_kg or 0.2,
        "mass_basis": "placeholder"
        if physics.mass_kg is None
        else ("estimated" if physics.mass_is_estimate else "user_supplied"),
        "scale": scale.model_dump() if scale else None,
        "isaac_physics_tested": False,
        "status": "needs_visual_alignment_and_physics_review",
        "warnings": [
            "Visual alignment is a uniform bounding-size estimate, not a solved correspondence.",
            "Review orientation, cavities and collider coverage before simulation use.",
            "The old asset's physics pass does not validate this new visual/collider pairing.",
            "Inferred metric scale, mass and unobserved surfaces are not measurements.",
        ],
        "files_sha256": {str(p.relative_to(out)): digest(p) for p in sorted(out.rglob("*")) if p.is_file()},
    }
    write_json(out / "asset.json", metadata)
    (out / "README.md").write_text(
        "# Textured asset\n\n"
        "1. Compare `preview.png` and `collisions.png` with the video. Check scale, "
        "orientation, collider coverage, and the frames in `semantics.json`.\n"
        "2. Run `kuafu assets validate --asset /path/to/asset_dynamic.usda`.\n"
        "3. Load `asset_dynamic.usda` in Isaac Sim, `asset_static.usda` for scenery, "
        "or `asset.glb` in a glTF viewer. Keep `textures/` with the USD/OBJ files.\n\n"
        "Edit colliders through `collision_scene.json` and visuals through the source GLB. "
        "Use `isaaclab_semantics.py` to register interaction frames and descriptions.\n"
    )
    return metadata
