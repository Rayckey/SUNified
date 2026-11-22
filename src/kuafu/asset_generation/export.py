"""Kuafu-style static/dynamic USDA, plus GLB and OBJ/MTL interchange."""

from pathlib import Path

import numpy as np
import trimesh

from .articulation import articulation_geometry, articulation_physics, isaaclab_snippet, posed_asset
from .geometry import BuiltAsset
from .io import resolve_artifact_path, write_json
from .render import render_views, write_viewer
from .schema import PhysicsConfig, Scale, SceneSpec
from .semantics import SEMANTICS_SNIPPET, export_semantics, semantic_frames_usda, semantic_properties_usda

HEADER = """#usda 1.0
(
    defaultPrim = "Asset"
    metersPerUnit = 1
    kilogramsPerUnit = 1
    upAxis = "Z"
)
"""


def _vectors(rows) -> str:
    return "[" + ", ".join("(" + ", ".join(f"{v:.9g}" for v in row) + ")" for row in rows) + "]"


def _mesh_usda(mesh, name: str, collision: bool, material: str) -> str:
    apis = (
        '["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "MaterialBindingAPI"]'
        if collision
        else '["MaterialBindingAPI"]'
    )
    extra = (
        (
            "            bool physics:collisionEnabled = true\n"
            '            uniform token physics:approximation = "convexHull"\n'
            '            token visibility = "invisible"\n'
            '            uniform token purpose = "guide"\n'
            "            rel material:binding:physics = </Asset/Looks/Physics>\n"
        )
        if collision
        else (f"            rel material:binding = </Asset/Looks/{material}>\n")
    )
    if not collision and mesh.metadata.get("smooth_shading"):
        extra += (
            f"            normal3f[] normals = {_vectors(mesh.vertex_normals)} (\n"
            '                interpolation = "vertex"\n'
            "            )\n"
        )
    if not collision and mesh.visual.kind == "texture" and mesh.visual.uv is not None:
        extra += (
            f"            texCoord2f[] primvars:st = {_vectors(mesh.visual.uv)} (\n"
            '                interpolation = "vertex"\n'
            "            )\n"
        )
    return (
        f'        def Mesh "{name}" (prepend apiSchemas = {apis})\n        {{\n'
        f"            point3f[] points = {_vectors(mesh.vertices)}\n"
        f"            int[] faceVertexCounts = [{', '.join(['3'] * len(mesh.faces))}]\n"
        f"            int[] faceVertexIndices = [{', '.join(map(str, mesh.faces.ravel()))}]\n"
        f"            float3[] extent = {_vectors(mesh.bounds)}\n"
        '            uniform token subdivisionScheme = "none"\n'
        '            uniform token orientation = "rightHanded"\n'
        f"            bool doubleSided = {str(bool(mesh.metadata.get('double_sided', False))).lower()}\n"
        + extra
        + "        }\n"
    )


def export_usda(asset: BuiltAsset, out: Path, physics: PhysicsConfig) -> None:
    out = resolve_artifact_path(out, "Generated USD directory")
    looks = ['    def Scope "Looks"\n    {\n']
    for i, mesh in enumerate(asset.visuals):
        if mesh.metadata.get("texture_file"):
            looks.append(_textured_material_usda(mesh, i))
            continue
        # Authored shader colors are linear; source colors and GLB factors use sRGB input here.
        srgb = mesh.visual.face_colors[0, :3] / 255
        rgb = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
        color = "(" + ", ".join(f"{c:.7g}" for c in rgb) + ")"
        looks.append(f"""        def Material "Material_{i}"
        {{
            token outputs:surface.connect = </Asset/Looks/Material_{i}/Surface.outputs:surface>
            def Shader "Surface"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = {color}
                float inputs:roughness = {mesh.metadata["roughness"]}
                float inputs:metallic = {mesh.metadata["metallic"]}
                token outputs:surface
            }}
        }}
""")
    looks.append(f"""        def Material "Physics" (prepend apiSchemas = ["PhysicsMaterialAPI"])
        {{
            float physics:staticFriction = {physics.friction}
            float physics:dynamicFriction = {physics.friction}
            float physics:restitution = {physics.restitution}
        }}
    }}
""")
    if asset.metadata.get("articulation"):
        (out / "asset_geometry.usda").write_text(HEADER + articulation_geometry(asset, looks, _mesh_usda))
        for dynamic in (False, True):
            properties = articulation_physics(asset, physics) if dynamic else ""
            wrapper = (
                HEADER
                + '\ndef Xform "Asset" (\n    prepend references = @asset_geometry.usda@\n)\n{\n'
                + properties
                + "}\n"
            )
            (out / f"asset_{'dynamic' if dynamic else 'static'}.usda").write_text(wrapper)
        (out / "asset.usda").write_text(
            HEADER + '\ndef Xform "Asset" (\n    prepend references = @asset_dynamic.usda@\n) {}\n'
        )
        return
    body = HEADER + '\ndef Xform "Asset"\n{\n' + "".join(looks)
    if "semantics" in asset.metadata:
        body += semantic_properties_usda(asset.metadata["semantics"]["object"])
        body += semantic_frames_usda(asset, "base")
    body += '    def Scope "Visuals"\n    {\n'
    body += (
        "".join(_mesh_usda(m, f"Part_{i:03d}", False, f"Material_{i}") for i, m in enumerate(asset.visuals))
        + "    }\n"
    )
    body += '    def Scope "Collisions"\n    {\n'
    body += (
        "".join(_mesh_usda(m, f"Collider_{i:03d}", True, "") for i, m in enumerate(asset.collisions))
        + "    }\n}\n"
    )
    (out / "asset_geometry.usda").write_text(body)
    for dynamic in (False, True):
        schemas = '\n    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]' if dynamic else ""
        properties = (
            (
                f"    bool physics:rigidBodyEnabled = true\n"
                f"    float physics:mass = {physics.mass_kg if physics.mass_kg is not None else 0.2}\n"
            )
            if dynamic
            else ""
        )
        wrapper = (
            HEADER
            + '\ndef Xform "Asset" (\n    prepend references = @asset_geometry.usda@'
            + schemas
            + "\n)\n{\n"
            + properties
            + "}\n"
        )
        (out / f"asset_{'dynamic' if dynamic else 'static'}.usda").write_text(wrapper)
    (out / "asset.usda").write_text(
        HEADER + '\ndef Xform "Asset" (\n    prepend references = @asset_dynamic.usda@\n) {}\n'
    )


def _textured_material_usda(mesh, index: int) -> str:
    """Base-color UV material. Textures are sRGB; glTF color factors are linear."""
    name = f"Material_{index}"
    path = f"/Asset/Looks/{name}"
    factor = mesh.visual.material.baseColorFactor
    factor = np.ones(4) if factor is None else np.asarray(factor, dtype=float) / 255
    scale = "(" + ", ".join(f"{v:.7g}" for v in factor) + ")"
    return f'''        def Material "{name}"
        {{
            token outputs:surface.connect = <{path}/Surface.outputs:surface>
            def Shader "Surface"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor.connect = <{path}/BaseColor.outputs:rgb>
                float inputs:roughness = {mesh.metadata["roughness"]}
                float inputs:metallic = {mesh.metadata["metallic"]}
                float inputs:opacity = 1
                token outputs:surface
            }}
            def Shader "UV"
            {{
                uniform token info:id = "UsdPrimvarReader_float2"
                string inputs:varname = "st"
                float2 outputs:result
            }}
            def Shader "BaseColor"
            {{
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @{mesh.metadata["texture_file"]}@
                token inputs:sourceColorSpace = "sRGB"
                float2 inputs:st.connect = <{path}/UV.outputs:result>
                token inputs:wrapS = "repeat"
                token inputs:wrapT = "repeat"
                float4 inputs:scale = {scale}
                float3 outputs:rgb
            }}
        }}
'''


def export_asset(
    asset: BuiltAsset, spec: SceneSpec, out: Path, physics: PhysicsConfig, scale: Scale | None = None
) -> dict:
    out = resolve_artifact_path(out, "Generated asset directory")
    out.mkdir(parents=True, exist_ok=True)
    export_usda(asset, out, physics)
    scene = trimesh.Scene()
    obj, mtl, offset = ["# Kuafu: meters, Z-up\nmtllib asset.mtl\n"], [], 1
    for i, mesh in enumerate(asset.visuals):
        rgb = mesh.visual.face_colors[0, :3] / 255
        obj.extend(
            [
                f"o Part_{i:03d}\nusemtl Material_{i}\n",
                "".join(f"v {x:.9g} {y:.9g} {z:.9g}\n" for x, y, z in mesh.vertices),
                "".join(f"f {a + offset} {b + offset} {c + offset}\n" for a, b, c in mesh.faces),
            ]
        )
        offset += len(mesh.vertices)
        mtl.append(f"newmtl Material_{i}\nKd {rgb[0]:.7g} {rgb[1]:.7g} {rgb[2]:.7g}\n\n")
        colored = mesh.copy()
        linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        colored.visual = trimesh.visual.TextureVisuals(
            material=trimesh.visual.material.PBRMaterial(
                name=f"Material_{i}",
                baseColorFactor=[*linear, 1.0],
                roughnessFactor=mesh.metadata["roughness"],
                metallicFactor=mesh.metadata["metallic"],
            )
        )
        # glTF uses Y-up. USDA, OBJ, STL and the scene specification stay Z-up.
        colored.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
        scene.add_geometry(colored, node_name=f"Part_{i:03d}", geom_name=f"Part_{i:03d}")
    scene.export(out / "asset.glb")
    (out / "asset.obj").write_text("".join(obj))
    (out / "asset.mtl").write_text("".join(mtl))
    asset.mesh().export(out / "asset.stl")
    render_views(asset, out)
    if spec.articulation:
        render_views(posed_asset(asset), out / "articulation_open")
    write_viewer(asset, out / "viewer.html")
    write_json(out / "scene.json", spec)
    metadata = {
        **asset.metadata,
        "physics": physics.model_dump(),
        "effective_mass_kg": physics.mass_kg if physics.mass_kg is not None else 0.2,
        "mass_basis": (
            ("estimated" if physics.mass_is_estimate else "user_supplied")
            if physics.mass_kg is not None
            else "placeholder"
        ),
        "scale": scale.model_dump() if scale else None,
        "part_names": [part.name for part in spec.parts],
        "glb_up_axis": "Y",
        "assumptions": spec.assumptions,
        "isaac_physics_tested": False,
        "texture_status": "flat PBR colors; no recovered image textures",
    }
    metadata["warnings"] = list(asset.metadata["warnings"])
    if physics.mass_kg is None:
        metadata["warnings"].append("Dynamic mass defaults to 0.2 kg; set --mass-kg for your object.")
    elif physics.mass_is_estimate:
        metadata["warnings"].append("Dynamic mass is an explicit estimate, not a measured physical property.")
    write_json(out / "asset.json", metadata)
    export_semantics(asset, out)
    (out / "isaaclab_snippet.py").write_text(
        (isaaclab_snippet(asset) if spec.articulation else ISAACLAB_SNIPPET) + SEMANTICS_SNIPPET
    )
    if spec.articulation:
        (out / "README.md").write_text(
            f"# {spec.name}\n\n{spec.summary}\n\n"
            "1. Inspect `preview.png`, `articulation_open/preview.png`, and `collisions.png`.\n"
            "2. Run `kuafu assets validate --asset /path/to/asset_dynamic.usda`. "
            "Revalidate after changing the asset.\n"
            "3. Load `ARTICULATED_OBJECT` from `isaaclab_snippet.py` and call "
            "`attach_semantics` to register interaction sensors. Keep the asset folder together.\n\n"
            "Joints are passive; handles and frames follow their drawer bodies. "
            "Read descriptions and axes in `semantics.json`, and part offsets in `PART_FRAMES`. "
            "Use `asset_static.usda` for frozen scenery and GLB/OBJ/STL for rest-pose visuals. "
            "Supply measured scale and mass to replace inferred values.\n"
        )
    return metadata


ISAACLAB_SNIPPET = '''"""Use in your Kuafu/Isaac Lab scene configuration after AppLauncher starts.

Move the entire asset directory together; USD layers use relative references.
"""
from pathlib import Path

from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sim import UsdFileCfg

ASSET_DIR = Path(__file__).resolve().parent

DYNAMIC_OBJECT = RigidObjectCfg(
    prim_path="/World/envs/env_.*/GeneratedObject",
    spawn=UsdFileCfg(usd_path=str(ASSET_DIR / "asset_dynamic.usda"), scale=(1.0, 1.0, 1.0)),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
)

STATIC_OBJECT = AssetBaseCfg(
    prim_path="/World/envs/env_.*/GeneratedFixture",
    spawn=UsdFileCfg(usd_path=str(ASSET_DIR / "asset_static.usda"), scale=(1.0, 1.0, 1.0)),
    init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
)
'''
