"""Link-preserving USD authoring and kinematic previews of inferred mechanisms.

Meshes and joint anchors share the assembled rest coordinate system. Link Xforms
start at identity; only the physics solver moves them. This avoids double offsets.
"""

import copy
import json

import numpy as np
import trimesh

from .geometry import BuiltAsset
from .schema import ArticulationSpec
from .semantics import semantic_frames_usda, semantic_properties_usda


def verify_articulation_report(directory, report):
    """A rigid-body smoke pass cannot certify an articulated export."""
    metadata_path = directory / "asset.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    spec = metadata.get("articulation")
    dynamic = directory / "asset_dynamic.usda"
    has_articulation = dynamic.is_file() and "PhysicsArticulationRootAPI" in dynamic.read_text()
    if not spec:
        if has_articulation:
            raise ValueError("Articulated USD is missing its articulation metadata")
        return
    expected = ArticulationSpec.model_validate(spec)
    expected_joints = {joint.name: joint for joint in expected.joints if joint.kind != "fixed"}
    checked = report.get("articulation", {})
    results = checked.get("joints", [])
    if (
        checked.get("passed") is not True
        or checked.get("joint_count") != len(expected_joints)
        or len(results) != len(expected_joints)
        or {j.get("name") for j in results} != set(expected_joints)
    ):
        raise ValueError("Physics report must exercise every articulated joint")
    if checked.get("fixed_base") != expected.fixed_base:
        raise ValueError("Physics report has a different base attachment")
    for result in results:
        joint = expected_joints[result["name"]]
        factor = np.pi / 180 if joint.kind == "revolute" else 1
        if (
            result.get("passed") is not True
            or result.get("kind") != joint.kind
            or result.get("independent_motion") is not True
            or result.get("fixed_children_follow") is not True
            or not np.allclose(
                [result.get("lower", np.nan), result.get("upper", np.nan)],
                [joint.lower * factor, joint.upper * factor],
                atol=1e-6,
            )
        ):
            raise ValueError(f"Physics report does not validate joint {joint.name}")
        if joint.kind == "prismatic" and abs(result.get("passive_pull_displacement_m", 0)) < 1e-6:
            raise ValueError(f"Physics report lacks a passive force test for {joint.name}")


def axis_vector(axis):
    vector = np.zeros(3)
    vector["XYZ".index(axis[-1])] = -1 if axis.startswith("-") else 1
    return vector


def posed_asset(asset: BuiltAsset, fraction: float = 0.8, positions: dict | None = None) -> BuiltAsset:
    """A review render, not a physics test. Move every descendant and its colliders."""
    spec = asset.metadata["articulation"]
    transforms = {spec["root"]: np.eye(4)}
    pending = list(spec["joints"])
    while pending:
        ready = [joint for joint in pending if joint["parent"] in transforms]
        if not ready:
            raise ValueError("Disconnected articulation")
        for joint in ready:
            transform = np.eye(4)
            value = fraction * (joint["upper"] if joint["upper"] else joint["lower"])
            if positions is not None:
                value = positions.get(joint["name"], 0.0)
            axis = axis_vector(joint["axis"])
            if joint["kind"] == "prismatic":
                transform[:3, 3] = axis * value
            elif joint["kind"] == "revolute":
                point = [joint["origin"][axis] for axis in "xyz"]
                transform = trimesh.transformations.rotation_matrix(np.deg2rad(value), axis, point)
            transforms[joint["child"]] = transforms[joint["parent"]] @ transform
            pending.remove(joint)
    collections = []
    for meshes in (asset.visuals, asset.collisions):
        moved = []
        for mesh in meshes:
            mesh = mesh.copy()
            mesh.apply_transform(transforms[mesh.metadata["link"]])
            moved.append(mesh)
        collections.append(moved)
    return BuiltAsset(*collections, copy.deepcopy(asset.metadata))


def check_drawer_clearance(asset: BuiltAsset, tolerance_m: float = 0.0005) -> dict:
    """Reject penetrating drawer colliders at rest and sampled independent travel poses.

    Convex half-space intersection measures penetration, avoiding false positives
    from bounding boxes around ring handles. This is sampled geometry validation,
    not continuous collision detection or a substitute for simulation.
    """
    from scipy.optimize import linprog
    from scipy.spatial import ConvexHull

    spec = asset.metadata.get("articulation")
    if not spec or any(joint["kind"] == "revolute" for joint in spec["joints"]):
        return {"checked": False, "reason": "Only prismatic/fixed assemblies use the drawer clearance check"}
    groups = {link["name"]: link["name"] for link in spec["links"]}
    for _ in spec["links"]:
        for joint in spec["joints"]:
            if joint["kind"] == "fixed":
                groups[joint["child"]] = groups[joint["parent"]]
    moving = [joint for joint in spec["joints"] if joint["kind"] == "prismatic"]
    poses = [("closed", {})]
    for joint in moving:
        for end in (joint["lower"], joint["upper"]):
            if end:
                for fraction in (0.25, 0.5, 0.75, 1):
                    poses.append((f"{joint['name']}:{fraction * end}", {joint["name"]: fraction * end}))
    poses.append(("all_open", {j["name"]: j["upper"] or j["lower"] for j in moving}))
    collisions = []
    for label, positions in poses:
        meshes = posed_asset(asset, positions=positions).collisions
        bounds = np.array([mesh.bounds for mesh in meshes])
        equations = {}
        for i, first in enumerate(meshes):
            for k in range(i + 1, len(meshes)):
                second = meshes[k]
                if groups[first.metadata["link"]] == groups[second.metadata["link"]]:
                    continue
                overlap = np.minimum(bounds[i, 1], bounds[k, 1]) - np.maximum(bounds[i, 0], bounds[k, 0])
                if np.any(overlap <= tolerance_m):
                    continue
                for index in (i, k):
                    if index not in equations:
                        equations[index] = ConvexHull(meshes[index].vertices).equations
                halfspaces = np.vstack([equations[i], equations[k]])
                solution = linprog(
                    [0, 0, 0, -1],
                    A_ub=np.column_stack([halfspaces[:, :3], np.ones(len(halfspaces))]),
                    b_ub=-halfspaces[:, 3],
                    bounds=[(None, None)] * 3 + [(0, None)],
                    method="highs",
                )
                if solution.status not in (0, 2):
                    raise ValueError("Could not establish drawer clearance: " + solution.message)
                if solution.success and solution.x[3] * 2 > tolerance_m:
                    collisions.append(
                        {
                            "pose": label,
                            "first": first.metadata["part_name"],
                            "second": second.metadata["part_name"],
                            "penetration_diameter_m": float(solution.x[3] * 2),
                        }
                    )
                    if len(collisions) >= 16:
                        break
            if len(collisions) >= 16:
                break
        if len(collisions) >= 16:
            break
    return {
        "checked": True,
        "passed": not collisions,
        "sampled_poses": len(poses),
        "tolerance_m": tolerance_m,
        "collisions": collisions,
    }


def articulation_geometry(asset, looks, mesh_usda):
    """Geometry layer has separate link scopes; wrappers add or omit physics."""
    body = '\ndef Xform "Asset"\n{\n' + "".join(looks)
    if "semantics" in asset.metadata:
        body += semantic_properties_usda(asset.metadata["semantics"]["object"])
    for link in asset.metadata["articulation"]["links"]:
        name = link["name"]
        body += f'    def Xform "{name}"\n    {{\n'
        body += semantic_frames_usda(asset, name)
        for collision, collection, scope in (
            (False, asset.visuals, "Visuals"),
            (True, asset.collisions, "Collisions"),
        ):
            body += f'        def Scope "{scope}"\n        {{\n'
            for i, mesh in enumerate(collection):
                if mesh.metadata.get("link") not in {
                    item["name"] for item in asset.metadata["articulation"]["links"]
                }:
                    raise ValueError("Articulated visuals and colliders require an explicit valid link")
                if mesh.metadata["link"] == name:
                    prim = f"Collider_{i:03d}" if collision else f"Part_{i:03d}"
                    body += mesh_usda(mesh, prim, collision, f"Material_{i}")
            body += "        }\n"
        body += "    }\n"
    return body + "}\n"


def articulation_physics(asset, physics):
    spec = asset.metadata["articulation"]
    total = sum(link["mass_fraction"] for link in spec["links"])
    mass = physics.mass_kg if physics.mass_kg is not None else 0.2
    body = ""
    for link in spec["links"]:
        root_api = (
            ', "PhysicsArticulationRootAPI", "PhysxArticulationAPI"'
            if link["name"] == spec["root"] and not spec["fixed_base"]
            else ""
        )
        body += f'''    over "{link["name"]}" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"{root_api}]
    ) {{
        bool physics:rigidBodyEnabled = true
        float physics:mass = {mass * link["mass_fraction"] / total:.9g}
'''
        if root_api:
            body += _solver_properties()
        body += "    }\n"
    if spec["fixed_base"]:
        body += f"""    def PhysicsFixedJoint "root_joint" (
        prepend apiSchemas = ["PhysicsArticulationRootAPI", "PhysxArticulationAPI"]
    ) {{
        rel physics:body1 = </Asset/{spec["root"]}>
        bool physics:jointEnabled = true
        bool physics:excludeFromArticulation = false
{_solver_properties()}    }}
"""
    for joint in spec["joints"]:
        kind = joint["kind"]
        usd_type = {"fixed": "Fixed", "prismatic": "Prismatic", "revolute": "Revolute"}[kind]
        drive = "linear" if kind == "prismatic" else "angular"
        schemas = f' (prepend apiSchemas = ["PhysicsDriveAPI:{drive}"])' if kind != "fixed" else ""
        point = "(" + ", ".join(f"{joint['origin'][axis]:.9g}" for axis in "xyz") + ")"
        # Rotate the positive USD axis 180 degrees around a perpendicular axis.
        axis = joint["axis"][-1]
        flip = {"X": "(0, 0, 0, 1)", "Y": "(0, 1, 0, 0)", "Z": "(0, 0, 1, 0)"}
        rotation = flip[axis] if joint["axis"].startswith("-") else "(1, 0, 0, 0)"
        body += f'''    def Physics{usd_type}Joint "{joint["name"]}"{schemas}
    {{
        rel physics:body0 = </Asset/{joint["parent"]}>
        rel physics:body1 = </Asset/{joint["child"]}>
        point3f physics:localPos0 = {point}
        point3f physics:localPos1 = {point}
        quatf physics:localRot0 = {rotation}
        quatf physics:localRot1 = {rotation}
        bool physics:jointEnabled = true
        bool physics:collisionEnabled = false
        bool physics:excludeFromArticulation = false
'''
        if kind != "fixed":
            body += f'''        uniform token physics:axis = "{axis}"
        float physics:lowerLimit = {joint["lower"]:.9g}
        float physics:upperLimit = {joint["upper"]:.9g}
        uniform token drive:{drive}:physics:type = "force"
        float drive:{drive}:physics:stiffness = 0
        float drive:{drive}:physics:damping = {joint["damping"]:.9g}
        float drive:{drive}:physics:maxForce = {joint["max_effort"]:.9g}
        float drive:{drive}:physics:targetPosition = 0
        float drive:{drive}:physics:targetVelocity = 0
'''
        body += "    }\n"
    return body


def _solver_properties():
    return """        bool physxArticulation:enabledSelfCollisions = false
        int physxArticulation:solverPositionIterationCount = 32
        int physxArticulation:solverVelocityIterationCount = 4
"""


def isaaclab_snippet(asset):
    joints = [joint for joint in asset.metadata["articulation"]["joints"] if joint["kind"] != "fixed"]
    actuators = "\n".join(
        f"        {j['name']!r}: ImplicitActuatorCfg(joint_names_expr=[{j['name']!r}], "
        f"effort_limit_sim={j['max_effort']!r}, stiffness=0.0, damping={j['damping']!r}),"
        for j in joints
    )
    return f'''"""Load the complete articulated asset after Isaac Lab AppLauncher starts.

Passive joints match the reference drawer: a robot can pull the handles.
GLB/OBJ/STL are rest-pose visuals; use the USD for simulation.
"""
from pathlib import Path
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim import UsdFileCfg

ASSET_DIR = Path(__file__).resolve().parent
# FrameTransformer targets can use the moving body plus these local part offsets.
# Example: select the white pull entry, target its body, and pass position_m and
# quaternion_wxyz to OffsetCfg. These frames move with joints.
PART_FRAMES = {asset.metadata.get("part_frames", {})!r}

ARTICULATED_OBJECT = ArticulationCfg(
    prim_path="/World/envs/env_.*/GeneratedCabinet",
    spawn=UsdFileCfg(usd_path=str(ASSET_DIR / "asset_dynamic.usda"), scale=(1.0, 1.0, 1.0)),
    init_state=ArticulationCfg.InitialStateCfg(joint_pos={{".*": 0.0}}),
    actuators={{
{actuators}
    }},
)
'''
