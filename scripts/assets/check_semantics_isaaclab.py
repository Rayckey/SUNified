"""Exercise exported language/frame integration with real Isaac Lab frame sensors.

Run with an Isaac Lab Python environment, not standalone usd-core. Each supplied
asset gets a translated/rotated instance and a uniformly scaled instance. Checks
cover sensor registration, body following, frame orientation, local motion axes,
source/world transforms, and driven independent joint motion. No model calls.
"""

import argparse
import hashlib
import importlib
import json
import math
import runpy
import traceback
from pathlib import Path
from types import SimpleNamespace

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--asset-dir", type=Path, action="append", required=True)
parser.add_argument("--report", type=Path, required=True)
parser.add_argument("--observation-module", help="Optional environment module exposing extract_frame_sensors")
parser.add_argument("--device", default="cuda:0")
args = parser.parse_args()
app = AppLauncher(headless=True, device=args.device).app
status = 0
report = {"passed": False}
try:
    import isaaclab.sim as sim_utils
    import torch
    from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
    from isaaclab.utils.math import combine_frame_transforms, quat_apply, subtract_frame_transforms

    simulation = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1 / 120, device=args.device, gravity=(0.0, 0.0, 0.0))
    )
    reference = RigidObject(
        RigidObjectCfg(
            prim_path="/World/Reference",
            spawn=sim_utils.CuboidCfg(
                size=(0.01, 0.01, 0.01),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=1),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, -0.2, 0.1), rot=(0.9238795, 0, 0, 0.3826834)),
        )
    )
    # The observation reader only needs a robot base pose. A kinematic reference
    # body supplies it here; this is not a robot grasp-success test.
    env = SimpleNamespace(
        scene=SimpleNamespace(articulations={"robot": reference}, rigid_objects={}, sensors={}),
        cfg=SimpleNamespace(
            detectable_objects={"existing_robot_frame": ("Robot frame", "/Visuals/existing")}
        ),
    )
    env.unwrapped = env
    assets = []
    evidence = []
    mechanisms = {}
    for directory in args.asset_dir:
        directory = directory.resolve()
        api = runpy.run_path(str(directory / "isaaclab_snippet.py"))
        semantics = api["SEMANTICS"]
        articulated = semantics["asset_type"] == "articulation"
        for scale in (1.0, 1.25):
            index = len(assets)
            name = f"object_{index}"
            metadata = json.loads((directory / "asset.json").read_text())
            mechanisms[name] = {j["name"]: j for j in metadata.get("articulation", {}).get("joints", [])}
            cfg = api["ARTICULATED_OBJECT" if articulated else "DYNAMIC_OBJECT"].copy()
            cfg.prim_path = f"/World/Object_{index}"
            cfg.spawn.scale = (scale,) * 3
            yaw = 0.7 + index * 0.2
            cfg.init_state.pos = (index * 1.2, 0.4, 0.3)
            cfg.init_state.rot = (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))
            obj = Articulation(cfg) if articulated else RigidObject(cfg)
            registry = env.scene.articulations if articulated else env.scene.rigid_objects
            registry[name] = obj
            sensors = api["attach_semantics"](env, name, "/World/Reference", frame_prefix=name + "_")
            assert name in env.cfg.detectable_objects
            assert "existing_robot_frame" in env.cfg.detectable_objects
            # Registration must reject a duplicate instead of replacing a live sensor.
            try:
                api["attach_semantics"](env, name, "/World/Reference", frame_prefix=name + "_")
            except ValueError:
                pass
            else:
                raise AssertionError("Duplicate sensor registration was accepted")
            entry = {
                "asset": str(directory),
                "object_name": name,
                "spawn_scale": scale,
                "rotation_wxyz": list(cfg.init_state.rot),
                "frames": list(sensors),
                "joint_motion": [],
                "max_position_error_m": 0,
                "max_axis_error": 0,
                "files_sha256": {
                    file.name: hashlib.sha256(file.read_bytes()).hexdigest()
                    for file in directory.iterdir()
                    if file.suffix == ".usda"
                    or file.name in {"semantics.json", "isaaclab_snippet.py", "isaaclab_semantics.py"}
                },
            }
            assets.append((obj, semantics, sensors, scale, entry))
            evidence.append(entry)
    simulation.reset()
    reference.reset()
    for obj, _, sensors, _, _ in assets:
        obj.reset()
        for sensor in sensors.values():
            sensor.reset()
    consumer = None
    if args.observation_module:
        consumer = importlib.import_module(args.observation_module).extract_frame_sensors
    steps = 0

    def verify_frames():
        for obj, semantics, sensors, scale, entry in assets:
            for frame_name, frame in semantics["frames"].items():
                key = entry["object_name"] + "_" + frame_name
                sensor = sensors[key]
                data = sensor.data
                assert data.target_frame_names == [key]
                body = obj.body_names.index(frame["body"]) if semantics["asset_type"] == "articulation" else 0
                pos = obj.data.body_pos_w[:, body]
                quat = obj.data.body_quat_w[:, body]
                offset = torch.tensor([frame["position_m"]], device=obj.device) * scale
                offset_quat = torch.tensor([frame["quaternion_wxyz"]], device=obj.device)
                expected_pos, expected_quat = combine_frame_transforms(pos, quat, offset, offset_quat)
                error = float(torch.max(torch.abs(data.target_pos_w[:, 0] - expected_pos)))
                entry["max_position_error_m"] = max(entry["max_position_error_m"], error)
                assert error < 3e-5, (key, error)
                # q and -q represent the same orientation.
                dot = torch.sum(data.target_quat_w[:, 0] * expected_quat, dim=-1).abs()
                assert torch.allclose(dot, torch.ones_like(dot), atol=2e-5), key
                source_pos, source_quat = subtract_frame_transforms(
                    reference.data.root_pos_w, reference.data.root_quat_w, expected_pos, expected_quat
                )
                assert torch.allclose(data.target_pos_source[:, 0], source_pos, atol=3e-5), key
                assert torch.allclose(
                    torch.sum(data.target_quat_source[:, 0] * source_quat, -1).abs(),
                    torch.ones_like(dot),
                    atol=2e-5,
                ), key
                if frame["motion"]:
                    joint = mechanisms[entry["object_name"]][frame["motion"]["joint"]]
                    parent = obj.body_names.index(joint["parent"])
                    axis = torch.zeros((1, 3), device=obj.device)
                    axis[0, "XYZ".index(joint["axis"][-1])] = -1 if joint["axis"].startswith("-") else 1
                    if frame["motion"]["opening_direction"] == "negative":
                        axis *= -1
                    expected_axis = quat_apply(obj.data.body_quat_w[:, parent], axis)
                    local_axis = torch.tensor([frame["axes"]["open"]["direction"]], device=obj.device)
                    actual_axis = quat_apply(data.target_quat_w[:, 0], local_axis)
                    error = float(torch.max(torch.abs(expected_axis - actual_axis)))
                    entry["max_axis_error"] = max(entry["max_axis_error"], error)
                    assert error < 3e-5, (key, error)
        if consumer:
            for env_id in (-1, 0):
                observed = consumer(env, {}, env.cfg.detectable_objects, env_id=env_id)
                for name, sensor in env.scene.sensors.items():
                    assert name in observed, name
                    expected = (
                        sensor.data.target_pos_w[:, 0] if env_id == -1 else sensor.data.target_pos_source[0]
                    )
                    assert torch.allclose(observed[name][0], expected, atol=3e-5), name
                    assert observed[name][3] == env.cfg.detectable_objects[name]

    def simulate(count):
        global steps
        for _ in range(count):
            for obj, *_ in assets:
                obj.write_data_to_sim()
            simulation.step(render=False)
            steps += 1
            reference.update(1 / 120)
            for obj, _, sensors, _, _ in assets:
                obj.update(1 / 120)
                for sensor in sensors.values():
                    sensor.update(1 / 120, force_recompute=True)
            if steps % 15 == 0:
                verify_frames()

    simulate(30)
    for obj, semantics, sensors, _, entry in assets:
        if semantics["asset_type"] != "articulation":
            continue
        # Test-only position drives; the exported asset remains passive.
        obj.write_joint_stiffness_to_sim(torch.full((1, obj.num_joints), 800.0, device=obj.device))
        obj.write_joint_damping_to_sim(torch.full((1, obj.num_joints), 80.0, device=obj.device))
        for index, name in enumerate(obj.joint_names):
            limits = obj.data.joint_pos_limits[0, index]
            target = float(limits[1] if limits[1] > 0 else limits[0]) * 0.7
            targets = torch.zeros((1, obj.num_joints), device=obj.device)
            targets[0, index] = target
            before = {key: sensor.data.target_pos_w.clone() for key, sensor in sensors.items()}
            obj.set_joint_position_target(targets)
            simulate(180)
            actual = float(obj.data.joint_pos[0, index])
            assert abs(actual - target) < max(0.003, abs(target) * 0.08), (name, target, actual)
            for frame_name, frame in semantics["frames"].items():
                key = entry["object_name"] + "_" + frame_name
                if (
                    frame["motion"]
                    and frame["motion"]["joint"] == name
                    and frame["motion"]["kind"] == "prismatic"
                ):
                    sensor = sensors[key]
                    local_axis = torch.tensor(
                        [frame["axes"]["joint_positive"]["direction"]], device=obj.device
                    )
                    axis = quat_apply(sensor.data.target_quat_w[:, 0], local_axis)
                    delta = sensor.data.target_pos_w[:, 0] - before[key][:, 0]
                    assert torch.allclose(delta, axis * actual, atol=0.003), key
            obj.set_joint_position_target(torch.zeros_like(targets))
            simulate(180)
            closed = float(obj.data.joint_pos[0, index])
            assert abs(closed) < 0.003, (name, closed)
            entry["joint_motion"].append(
                {"joint": name, "target": target, "opened": actual, "closed": closed}
            )
    report = {
        "passed": True,
        "device": args.device,
        "steps": steps,
        "physics_dt": 1 / 120,
        "observation_module": args.observation_module,
        "assets": evidence,
        "scope": "Actual FrameTransformer sensors, moving joints, local axes and environment lookup; "
        "no robot grasp test",
    }
except BaseException as error:
    status = 1
    report["error"] = f"{type(error).__name__}: {error}"
    traceback.print_exc()
finally:
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({"passed": report["passed"], "report": str(args.report), "error": report.get("error")}),
        flush=True,
    )
    import omni.kit.app

    if "env" in globals():
        env.scene.sensors.clear()
        env.scene.articulations.clear()
        env.scene.rigid_objects.clear()
    if "assets" in globals():
        assets.clear()
    if "simulation" in globals():
        if simulation.cfg.create_stage_in_memory:
            import omni.physx

            omni.physx.get_physx_simulation_interface().detach_stage()
            simulation.stop()
            simulation.clear()
        simulation.clear_all_callbacks()
        simulation.clear_instance()
    omni.kit.app.get_app().post_quit(status)
    app.close(wait_for_replicator=False)
