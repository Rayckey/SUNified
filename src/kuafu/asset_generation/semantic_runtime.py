"""Portable Isaac Lab integration, copied beside each exported semantics.json.

No reconstruction package is needed at simulation time. Import after AppLauncher.
Directions in semantics.json are local to each frame; rotate them by the sensor's
target_quat_w (or target_quat_source) to use them in that coordinate system.
"""

import json
import math
import re
from pathlib import Path


def _direction_text(direction):
    for index, value in enumerate(direction):
        if abs(abs(value) - 1) < 1e-6 and all(
            abs(other) < 1e-6 for i, other in enumerate(direction) if i != index
        ):
            return ("+" if value > 0 else "-") + "xyz"[index]
    return "direction (" + ", ".join(f"{value:.6g}" for value in direction) + ")"


def describe(entry, scale=1.0):
    """Use the same local-axis language consumed by Kuafu detectable_objects."""
    pieces = [entry["description"].rstrip(". ;")]
    for axis in entry["axes"].values():
        prefix = "rotation around local frame" if axis.get("motion") == "rotation" else "local frame"
        pieces.append(f"{prefix} {_direction_text(axis['direction'])} is {axis['description']}")
    if entry.get("size_m"):
        size = ", ".join(f"{value * scale:.6g}" for value in entry["size_m"])
        pieces.append(f"estimated size: ({size}) meters")
    return "; ".join(pieces)


def load_semantics(asset_dir=None):
    directory = Path(asset_dir) if asset_dir is not None else Path(__file__).resolve().parent
    data = json.loads((directory / "semantics.json").read_text())
    if data.get("version") != 1:
        raise ValueError("Unsupported semantic metadata version")
    return data


def make_semantic_cfg(
    asset_cfg, source_prim_path, *, object_name=None, frame_prefix="", debug_vis=False, asset_dir=None
):
    """Return (detectable_objects, one FrameTransformerCfg per interaction point).

    object_name must match the key used to register the asset in scene.articulations
    or scene.rigid_objects. Frame keys match scene.sensors exactly. The target is the
    owning rigid body plus an offset, never a non-rigid USD marker prim.
    """
    from isaaclab.assets import ArticulationCfg, RigidObjectCfg
    from isaaclab.markers.config import FRAME_MARKER_CFG
    from isaaclab.sensors import FrameTransformerCfg
    from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg

    data = load_semantics(asset_dir)
    if data["status"] != "annotated":
        raise ValueError("This legacy asset has no authored interaction semantics; annotate its scene first")
    if not isinstance(asset_cfg, (ArticulationCfg, RigidObjectCfg)):
        raise ValueError("Frame sensors require the dynamic rigid/articulated asset configuration")
    if (data["asset_type"] == "articulation") != isinstance(asset_cfg, ArticulationCfg):
        raise ValueError("Semantic body paths do not match the asset configuration type")
    name = object_name or data["object"]["name"]
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) or (
        frame_prefix and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", frame_prefix)
    ):
        raise ValueError("Object name and optional frame prefix must be identifiers")
    scale = asset_cfg.spawn.scale or (1.0, 1.0, 1.0)
    if (
        len(scale) != 3
        or any(not math.isfinite(v) or v <= 0 for v in scale)
        or any(abs(v - scale[0]) > 1e-8 for v in scale)
    ):
        raise ValueError("Semantic frames require a positive uniform spawn scale; rebuild for other scales")
    scale = float(scale[0])
    prim_path = asset_cfg.prim_path.rstrip("/")
    detectable = {name: (describe(data["object"], scale), prim_path.replace("env_.*", "env_*"))}
    frames = {}
    for frame_name, frame in data["frames"].items():
        key = frame_prefix + frame_name
        if key == name:
            raise ValueError("Object name conflicts with an interaction frame name")
        marker = FRAME_MARKER_CFG.copy()
        marker.prim_path = f"/Visuals/{key}"
        marker.markers["frame"].scale = (0.03, 0.03, 0.03)
        target = prim_path + ("/" + frame["body_path"] if frame["body_path"] else "")
        frames[key] = FrameTransformerCfg(
            prim_path=source_prim_path,
            debug_vis=debug_vis,
            visualizer_cfg=marker,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path=target,
                    name=key,
                    offset=OffsetCfg(
                        pos=tuple(v * scale for v in frame["position_m"]),
                        rot=tuple(frame["quaternion_wxyz"]),
                    ),
                )
            ],
        )
        detectable[key] = (describe(frame, scale), marker.prim_path)
    return detectable, frames


def attach_semantics(env, object_name, source_prim_path, **kwargs):
    """Register all sensors and merge descriptions during a DirectRLEnv._setup_scene.

    Call after registering the object, before clone_environments/reset. Existing
    robot entries are retained. Name collisions fail before changing the scene.
    InteractiveSceneCfg users can instead install make_semantic_cfg's returned
    frame configurations as scene config attributes.
    """
    from isaaclab.sensors import FrameTransformer

    obj = env.scene.articulations.get(object_name) or env.scene.rigid_objects.get(object_name)
    if obj is None:
        raise ValueError(f"Register {object_name!r} in the scene before attaching its semantics")
    detectable, configs = make_semantic_cfg(obj.cfg, source_prim_path, object_name=object_name, **kwargs)
    current = getattr(env.cfg, "detectable_objects", {})
    conflicts = (set(configs) & set(env.scene.sensors)) | ((set(detectable) - {object_name}) & set(current))
    if conflicts:
        raise ValueError(f"Semantic names already registered: {sorted(conflicts)}; choose a frame_prefix")
    sensors = {name: FrameTransformer(cfg) for name, cfg in configs.items()}
    env.scene.sensors.update(sensors)
    env.cfg.detectable_objects = {**current, **detectable}
    return sensors
