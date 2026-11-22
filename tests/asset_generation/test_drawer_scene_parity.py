"""Keep the single-drawer object's geometry consistent across the two robots."""

import ast
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from kuafu.paths import OBJECT_ROOT

Usd = pytest.importorskip("pxr.Usd")
from pxr import Gf, UsdGeom, UsdPhysics


ROOT = Path(__file__).resolve().parents[2]
ENVIRONMENTS = ROOT / "src/kuafu/isaac/learned"


def scene_source(robot):
    path = ENVIRONMENTS / robot / f"{robot}_drawer_env_learned.py"
    return path, ast.parse(path.read_text())


def drawer_asset(robot):
    path, tree = scene_source(robot)
    assignment = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "_cabinet_path" for t in node.targets))
    return Path(eval(compile(ast.Expression(assignment.value), str(path), "eval"),
                     {"os": os, "__file__": str(path), "OBJECT_ROOT": OBJECT_ROOT}))


def frame_offset(robot, frame):
    _, tree = scene_source(robot)
    assignment = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == frame for t in node.targets))
    offset = next(node for node in ast.walk(assignment) if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Name) and node.func.id == "OffsetCfg")
    return {kw.arg: ast.literal_eval(kw.value) for kw in offset.keywords}


def test_drawer_asset_and_interaction_frames_match_panda():
    asset = drawer_asset("panda")
    assert drawer_asset("kinova") == asset
    assert asset.is_file()
    stage = Usd.Stage.Open(str(asset))
    root = stage.GetDefaultPrim().GetPath()
    assert stage.GetPrimAtPath(root.AppendChild("upper_drawer_handle")).IsValid()
    for name in ("top_drawer_handle_frame", "top_drawer_center_frame"):
        assert frame_offset("kinova", name) == frame_offset("panda", name)


def runtime_collisions(robot):
    path, tree = scene_source(robot)
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                  and node.name == "_replace_frame_collision_with_cube")
    stage = Usd.Stage.CreateInMemory()
    cabinet_path = "/World/envs/env_0/Cabinet"
    cabinet = UsdGeom.Xform.Define(stage, cabinet_path).GetPrim()
    cabinet.GetReferences().AddReference(str(drawer_asset(robot)))
    namespace = {
        "get_current_stage": lambda: stage, "Usd": Usd, "UsdGeom": UsdGeom,
        "sim_utils": SimpleNamespace(find_matching_prims=lambda pattern: [cabinet]),
        "schemas": SimpleNamespace(define_collision_properties=lambda path, props, stage:
                                   UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(path))),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    env = SimpleNamespace(cfg=SimpleNamespace(cabinet=SimpleNamespace(
        prim_path=cabinet_path, spawn=SimpleNamespace(collision_props=object()))))
    namespace[method.name](env)
    original = stage.GetPrimAtPath(cabinet_path + "/frame/collisions")
    assert not original.IsActive()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], False, True)
    result = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(cabinet_path + "/frame")):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
            result.append((str(prim.GetPath()), tuple(box.GetMin()), tuple(box.GetMax())))
    assert result
    return result


def test_runtime_drawer_collision_matches_panda():
    assert runtime_collisions("kinova") == runtime_collisions("panda")


@pytest.mark.parametrize("azimuth,yaw_delta", [(-1.2, -.8), (1.2, .8), (0., 0.)])
def test_kinova_reset_accounts_for_drawer_joint_axis(azimuth, yaw_delta):
    path, tree = scene_source("kinova")
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                  and node.name == "_reset_idx")
    yaw_node = next(node for node in ast.walk(method) if isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "yaw" for t in node.targets))
    yaw = eval(compile(ast.Expression(yaw_node.value), str(path), "eval"),
               {"azimuth": azimuth, "yaw_delta": yaw_delta, "torch": SimpleNamespace(pi=math.pi)})
    stage = Usd.Stage.Open(str(drawer_asset("kinova")))
    joint = next(UsdPhysics.PrismaticJoint(prim) for prim in stage.Traverse()
                 if prim.IsA(UsdPhysics.PrismaticJoint))
    body = stage.GetPrimAtPath(joint.GetBody0Rel().GetTargets()[0])
    body_rotation = Gf.Transform(UsdGeom.Xformable(body).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default())).GetRotation()
    axis = {"X": Gf.Vec3d(1, 0, 0), "Y": Gf.Vec3d(0, 1, 0), "Z": Gf.Vec3d(0, 0, 1)}[joint.GetAxisAttr().Get()]
    local_axis = body_rotation.TransformDir(Gf.Rotation(joint.GetLocalRot0Attr().Get()).TransformDir(axis))
    world_axis = Gf.Rotation(Gf.Vec3d(0, 0, 1), math.degrees(yaw)).TransformDir(local_axis)
    # Positive drawer travel points toward the robot, with the sampled yaw offset.
    expected = (-math.cos(azimuth + yaw_delta), -math.sin(azimuth + yaw_delta), 0.)
    assert tuple(world_axis) == pytest.approx(expected, abs=1e-6)
