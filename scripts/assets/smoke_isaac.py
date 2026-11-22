"""Drop an exported rigid object on a ground plane using Isaac Sim's core API.

./isaaclab.sh -p scripts/assets/smoke_isaac.py --headless --asset /path/asset_dynamic.usda
"""

import argparse
import hashlib
import json
import traceback
from pathlib import Path

from kuafu.paths import resolve_artifact_path

parser = argparse.ArgumentParser()
parser.add_argument("--asset", type=Path, required=True)
parser.add_argument("--steps", type=int, default=240)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--report", type=Path)
parser.add_argument(
    "--probe-radius", type=float, help="Optionally drop a sphere into a centered, upright cavity"
)
parser.add_argument(
    "--cavity-floor-z", type=float, help="Expected floor height in asset-local meters; required with probe"
)
args = parser.parse_args()
args.asset = resolve_artifact_path(args.asset, "Generated simulation asset")
if args.report is not None:
    args.report = resolve_artifact_path(args.report, "Asset physics report")
if not args.asset.is_file() or args.steps < 1:
    parser.error("Supply an existing dynamic USDA and a positive number of steps")
if (args.probe_radius is None) != (args.cavity_floor_z is None):
    parser.error("Supply --probe-radius and --cavity-floor-z together")
if args.probe_radius is not None and (args.probe_radius <= 0 or args.cavity_floor_z < 0):
    parser.error("Probe radius must be positive and cavity floor must be nonnegative")

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": args.headless, "multi_gpu": False, "fast_shutdown": True})
exit_code, report = 0, None

try:
    import numpy as np
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicSphere
    from isaacsim.core.prims import RigidPrim
    from isaacsim.core.utils.rotations import quat_to_rot_matrix
    from isaacsim.core.utils.stage import add_reference_to_stage
    from pxr import Usd, UsdGeom, UsdPhysics

    source_stage = Usd.Stage.Open(str(args.asset.resolve()))
    asset_layers = []
    for layer in source_stage.GetUsedLayers():
        if layer.realPath:
            path = Path(layer.realPath).resolve()
            asset_layers.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

    if any(prim.HasAPI(UsdPhysics.ArticulationRootAPI) for prim in source_stage.Traverse()):
        from articulation_check import check_articulation

        if args.probe_radius is not None:
            raise ValueError("A centered rigid cavity probe does not specify an articulated drawer cavity")
        report = check_articulation(args.asset, source_stage, args.steps)
        report["asset_layers"] = asset_layers
        print(
            f"PASS: {report['articulation']['joint_count']} moving joints; {report['steps']} steps",
            flush=True,
        )
    else:
        world = World(stage_units_in_meters=1.0, physics_dt=1 / 120, backend="numpy", device="cpu")
        world.scene.add_default_ground_plane()
        prim = add_reference_to_stage(str(args.asset.resolve()), "/World/TestObject")
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError("Use asset_dynamic.usda: the asset root must already be a rigid body")
        obj = world.scene.add(
            RigidPrim(
                prim_paths_expr="/World/TestObject", name="test_object", positions=np.array([[0, 0, 0.1]])
            )
        )
        probe = None
        if args.probe_radius is not None:
            bounds = (
                UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
                .ComputeLocalBound(prim)
                .ComputeAlignedRange()
            )
            top = float(bounds.GetMax()[2])
            probe = world.scene.add(
                DynamicSphere(
                    prim_path="/World/CavityProbe",
                    name="cavity_probe",
                    radius=args.probe_radius,
                    mass=0.005,
                    position=np.array([0, 0, 0.1 + top + 0.05]),
                    color=np.array([0.8, 0.2, 0.1]),
                )
            )
        world.reset()
        for _ in range(args.steps):
            world.step(render=False)
            positions, orientations = obj.get_world_poses()
            if not np.isfinite(positions).all() or not np.isfinite(orientations).all():
                raise RuntimeError("Physics produced a non-finite rigid-body state")
            if probe is not None and not np.isfinite(probe.get_world_pose()[0]).all():
                raise RuntimeError("Physics produced a non-finite probe state")
        position = positions[0].tolist()
        if position[2] < -0.02:
            raise RuntimeError(f"Object fell through the ground: {position}")
        report = {
            "passed": True,
            "asset": str(args.asset.resolve()),
            "steps": args.steps,
            "physics_dt": 1 / 120,
            "asset_layers": asset_layers,
            "final_root_position": position,
            "scope": "Finite-state and ground-contact smoke test only; not a task benchmark.",
        }
        if probe is not None:
            probe_position, _ = probe.get_world_pose()
            local_position = quat_to_rot_matrix(orientations[0]).T @ (probe_position - positions[0])
            expected_z = args.cavity_floor_z + args.probe_radius
            # Allow ordinary PhysX contact offsets, but reject support at the rim or through the floor.
            if not expected_z - 0.003 <= local_position[2] <= expected_z + 0.01:
                raise RuntimeError(
                    f"Probe did not land on the interior floor: local position {local_position}"
                )
            if np.linalg.norm(local_position[:2]) > args.probe_radius * 2:
                raise RuntimeError(f"Probe escaped the centered cavity: local position {local_position}")
            report["cavity_probe"] = {
                "passed": True,
                "radius_m": args.probe_radius,
                "mass_kg": 0.005,
                "expected_floor_z_m": args.cavity_floor_z,
                "expected_center_z_m": expected_z,
                "final_local_position_m": local_position.tolist(),
            }
            report["scope"] = (
                "Finite state, ground contact, and one centered sphere dropping onto the cavity floor; "
                "not a task benchmark."
            )
        print(
            f"PASS: finite rigid-body state after {args.steps} steps; root position = {position}", flush=True
        )
except Exception as error:
    exit_code = 1
    report = {
        "passed": False,
        "asset": str(args.asset.resolve()),
        "error": f"{type(error).__name__}: {error}",
    }
    traceback.print_exc()
finally:
    if args.report and report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    # Preserve failure status when Kit's fast shutdown terminates the process.
    app.app.post_quit(exit_code)
    app.close(wait_for_replicator=False)

raise SystemExit(exit_code)
