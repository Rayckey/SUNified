"""Visual-only USD background helpers for media capture scripts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import omni.usd
import isaaclab.sim as sim_utils
import torch
from pxr import Gf, Usd, UsdGeom, UsdPhysics


def parse_vec3(text: str) -> tuple[float, float, float]:
    values = [float(x.strip()) for x in text.split(",")]
    if len(values) != 3:
        raise ValueError(f"Expected comma-separated vec3, got: {text}")
    return tuple(values)


def disable_physics_under(stage: Usd.Stage, root_path: str) -> None:
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        raise ValueError(f"Invalid background root path: {root_path}")

    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)


def hide_matching_visual_prims(prim_pattern: str, label: str = "visuals", warn_if_empty: bool = True) -> int:
    hidden_count = 0
    for root_prim in sim_utils.find_matching_prims(prim_pattern):
        for prim in Usd.PrimRange(root_prim):
            if prim.IsA(UsdGeom.Imageable):
                UsdGeom.Imageable(prim).MakeInvisible()
                hidden_count += 1

    if warn_if_empty and hidden_count == 0:
        print(f"[paper {label}] WARNING: no imageable prims matched {prim_pattern}", flush=True)
    else:
        print(f"[paper {label}] Hid {hidden_count} imageable prims under {prim_pattern}", flush=True)
    return hidden_count


def current_stage() -> Usd.Stage:
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("USD stage does not exist yet.")
    return stage


def set_xform(
    prim,
    pos: tuple[float, float, float],
    rot_xyz_deg: tuple[float, float, float],
    scale: tuple[float, float, float],
) -> None:
    xform_api = UsdGeom.XformCommonAPI(prim)
    xform_api.SetTranslate(Gf.Vec3d(*pos))
    xform_api.SetRotate(
        Gf.Vec3f(*rot_xyz_deg),
        UsdGeom.XformCommonAPI.RotationOrderXYZ,
    )
    xform_api.SetScale(Gf.Vec3f(*scale))


def add_custom_usd_paper_background(
    usd_path: str,
    pos: tuple[float, float, float],
    rot_xyz_deg: tuple[float, float, float],
    scale: tuple[float, float, float],
    prim_path: str = "/World/PaperBackground",
) -> tuple[float, float, float]:
    if not usd_path:
        raise ValueError("--paper_background_usd is required when --paper_background custom_usd")
    usd_file = Path(usd_path).expanduser()
    if not usd_file.exists():
        raise FileNotFoundError(f"Custom background USD does not exist: {usd_file}")

    stage = current_stage()
    if stage.GetPrimAtPath(prim_path).IsValid():
        stage.RemovePrim(prim_path)

    xform = UsdGeom.Xform.Define(stage, prim_path)
    prim = xform.GetPrim()
    set_xform(prim, pos, rot_xyz_deg, scale)

    asset_prim = UsdGeom.Xform.Define(stage, f"{prim_path}/Asset").GetPrim()
    asset_prim.GetReferences().AddReference(str(usd_file.resolve()))

    disable_physics_under(stage, prim_path)
    print(
        f"[paper background] Added custom USD {usd_file} at {prim_path}, "
        f"pos={pos}, rot={rot_xyz_deg}, scale={scale}",
        flush=True,
    )
    return pos


def _env_origin(raw_env: Any, env_id: int) -> torch.Tensor:
    origins = getattr(getattr(raw_env, "scene", None), "env_origins", None)
    if origins is None:
        return torch.zeros(3, dtype=torch.float32)
    env_index = int(env_id)
    if env_index < 0 or env_index >= int(origins.shape[0]):
        raise ValueError(f"Background env id {env_index} is outside num_envs={int(origins.shape[0])}.")
    return origins[env_index].detach().cpu().to(dtype=torch.float32)


def paper_background_env_id(args: argparse.Namespace) -> int:
    record_env_id = int(getattr(args, "probe_record_env_id", -1))
    return record_env_id if record_env_id >= 0 else 0


def add_configured_paper_background(args: argparse.Namespace, raw_env: Any) -> tuple[float, float, float] | None:
    background_name = str(getattr(args, "paper_background", "none"))
    if background_name == "none":
        return None
    if background_name != "custom_usd":
        raise ValueError(f"Unknown paper background: {background_name}")

    env_id = paper_background_env_id(args)
    origin = _env_origin(raw_env, env_id)
    offset = torch.tensor(parse_vec3(str(args.paper_background_pos)), dtype=torch.float32)
    pos = tuple((origin + offset).tolist())
    return add_custom_usd_paper_background(
        str(args.paper_background_usd),
        pos=pos,
        rot_xyz_deg=parse_vec3(str(args.paper_background_rot)),
        scale=parse_vec3(str(args.paper_background_scale)),
    )
