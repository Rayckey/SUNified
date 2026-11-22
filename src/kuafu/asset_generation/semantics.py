"""Language semantics and interaction poses in the same coordinates as the asset."""

import json
from pathlib import Path

import numpy as np
import trimesh

from .io import write_json
from .schema import SceneSpec
from .semantic_runtime import describe


def require_agent_semantics(spec: SceneSpec):
    """Old recipes can still build, but new agent results must support interaction."""
    if spec.semantics is None:
        raise ValueError("Supply semantics with language descriptions and important interaction frames")
    if spec.articulation:
        covered = {frame.joint for frame in spec.semantics.frames}
        missing = {joint.name for joint in spec.articulation.joints if joint.kind != "fixed"} - covered
        if missing:
            raise ValueError(
                f"Supply joint-linked interaction frames for every moving joint: {sorted(missing)}"
            )


def _axes(axes):
    return {
        axis.name: {"direction": axis.direction.array(), "description": axis.description} for axis in axes
    }


def build_semantics(spec, factor, translation, dimensions):
    semantics = spec.semantics
    result = {
        "version": 1,
        "status": "annotated" if semantics else "missing",
        "asset_type": "articulation" if spec.articulation else "rigid",
        "units": "meters",
        "pose_convention": "body-local offsets; quaternion wxyz; axes local to each interaction frame",
        "object": {
            "name": semantics.name if semantics else "generated_object",
            "description": semantics.description if semantics else spec.summary,
            "size_m": list(dimensions),
            "axes": _axes(semantics.axes) if semantics else {},
        },
        "frames": {},
    }
    result["object"]["language"] = describe(result["object"])
    joints = {j.name: j for j in spec.articulation.joints} if spec.articulation else {}
    for frame in semantics.frames if semantics else []:
        rotation = trimesh.transformations.euler_matrix(*np.deg2rad(frame.rotation_deg.array()), axes="sxyz")
        body_path = frame.body if spec.articulation else ""
        entry = {
            "body": frame.body,
            "body_path": body_path,
            "usd_path": "/Asset/"
            + (body_path + "/" if body_path else "")
            + "InteractionFrames/"
            + frame.name,
            "role": frame.role,
            "description": frame.description,
            "position_m": (np.array(frame.position.array()) * factor + translation).tolist(),
            "quaternion_wxyz": trimesh.transformations.quaternion_from_matrix(rotation).tolist(),
            "size_m": (np.array(frame.size.array()) * factor).tolist() if frame.size else None,
            "part": frame.part,
            "axes": _axes(frame.axes),
            "motion": None,
            "evidence": frame.evidence,
        }
        if frame.joint:
            joint = joints[frame.joint]
            direction = np.zeros(3)
            direction["XYZ".index(joint.axis[-1])] = -1 if joint.axis.startswith("-") else 1
            local = rotation[:3, :3].T @ direction
            opening = local * (1 if frame.opening_direction == "positive" else -1)
            motion = "translation" if joint.kind == "prismatic" else "rotation"
            for name, vector, meaning in (
                ("joint_positive", local, "the positive joint direction"),
                ("open", opening, "the opening direction"),
                ("close", -opening, "the closing direction"),
            ):
                entry["axes"][name] = {"direction": vector.tolist(), "description": meaning, "motion": motion}
            limit_scale = factor if joint.kind == "prismatic" else 1.0
            entry["motion"] = {
                "joint": joint.name,
                "kind": joint.kind,
                "opening_direction": frame.opening_direction,
                "limits": [joint.lower * limit_scale, joint.upper * limit_scale],
                "units": "meters" if joint.kind == "prismatic" else "degrees",
            }
        entry["language"] = describe(entry)
        result["frames"][frame.name] = entry
    return result


def semantic_properties_usda(entry):
    text = (
        f"        custom string semantics:description = {json.dumps(entry['language'], ensure_ascii=False)}\n"
    )
    for name, axis in entry["axes"].items():
        vector = "(" + ", ".join(f"{v:.12g}" for v in axis["direction"]) + ")"
        text += f"        custom vector3d semantics:axis:{name} = {vector}\n"
    return text


def semantic_frames_usda(asset, body):
    frames = [
        (name, entry)
        for name, entry in asset.metadata.get("semantics", {}).get("frames", {}).items()
        if entry["body"] == body
    ]
    if not frames:
        return ""
    result = '        def Xform "InteractionFrames"\n        {\n'
    for name, entry in frames:
        position = "(" + ", ".join(f"{v:.12g}" for v in entry["position_m"]) + ")"
        rotation = "(" + ", ".join(f"{v:.12g}" for v in entry["quaternion_wxyz"]) + ")"
        result += (
            f'            def Xform "{name}"\n            {{\n'
            f"                double3 xformOp:translate = {position}\n"
            f"                quatd xformOp:orient = {rotation}\n"
            '                uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]\n'
            f'                custom token semantics:role = "{entry["role"]}"\n'
            + semantic_properties_usda(entry)
            + "            }\n"
        )
    return result + "        }\n"


def export_semantics(asset, out):
    write_json(out / "semantics.json", asset.metadata["semantics"])
    (out / "isaaclab_semantics.py").write_text(Path(__file__).with_name("semantic_runtime.py").read_text())


SEMANTICS_SNIPPET = """
# Portable helpers use the adjacent semantics.json; no pipeline install is needed.
import runpy as _runpy
_semantic_api = _runpy.run_path(str(ASSET_DIR / "isaaclab_semantics.py"))
SEMANTICS = _semantic_api["load_semantics"]()
make_semantic_cfg = _semantic_api["make_semantic_cfg"]
attach_semantics = _semantic_api["attach_semantics"]
"""
