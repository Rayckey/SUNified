"""Exercise an exported articulation in the running Isaac Sim application.

Position targets and forces drive the solver; joint states are never teleported.
Controllers are temporary test instrumentation, not edits to the exported USD.
"""


def check_articulation(asset, source_stage, minimum_steps):
    import numpy as np
    from isaacsim.core.api import World
    from isaacsim.core.prims import Articulation
    from isaacsim.core.utils.stage import add_reference_to_stage
    from pxr import UsdPhysics
    from scipy.spatial.transform import Rotation

    joints = {}
    fixed = []
    fixed_base = False
    for prim in source_stage.Traverse():
        if prim.IsA(UsdPhysics.FixedJoint):
            joint = UsdPhysics.FixedJoint(prim)
            if not joint.GetBody0Rel().GetTargets() and joint.GetBody1Rel().GetTargets():
                fixed_base = True
            elif joint.GetBody0Rel().GetTargets() and joint.GetBody1Rel().GetTargets():
                fixed.append(
                    (
                        str(joint.GetBody0Rel().GetTargets()[0]).split("/")[-1],
                        str(joint.GetBody1Rel().GetTargets()[0]).split("/")[-1],
                    )
                )
        elif prim.IsA(UsdPhysics.PrismaticJoint) or prim.IsA(UsdPhysics.RevoluteJoint):
            linear = prim.IsA(UsdPhysics.PrismaticJoint)
            joint = UsdPhysics.PrismaticJoint(prim) if linear else UsdPhysics.RevoluteJoint(prim)
            rotation = joint.GetLocalRot0Attr().Get()
            quaternion = [*rotation.GetImaginary(), rotation.GetReal()]
            axis = Rotation.from_quat(quaternion).apply(np.eye(3)["XYZ".index(joint.GetAxisAttr().Get())])
            factor = 1 if linear else np.pi / 180
            joints[prim.GetName()] = {
                "name": prim.GetName(),
                "kind": "prismatic" if linear else "revolute",
                "parent": str(joint.GetBody0Rel().GetTargets()[0]).split("/")[-1],
                "child": str(joint.GetBody1Rel().GetTargets()[0]).split("/")[-1],
                "axis": axis,
                "lower": joint.GetLowerLimitAttr().Get() * factor,
                "upper": joint.GetUpperLimitAttr().Get() * factor,
            }
    if not joints:
        raise ValueError("Articulated asset has no moving joints")
    world = World(stage_units_in_meters=1.0, physics_dt=1 / 120, backend="numpy", device="cpu")
    world.scene.add_default_ground_plane()
    add_reference_to_stage(str(asset.resolve()), "/World/TestObject")
    obj = world.scene.add(Articulation(prim_paths_expr="/World/TestObject", name="test_articulation"))
    world.reset()
    if set(obj.dof_names) != set(joints):
        raise ValueError(f"Simulator joints differ from USD: {obj.dof_names} versus {list(joints)}")
    names = list(obj.dof_names)
    body_indices = {name: index for index, name in enumerate(obj.body_names)}
    limits = obj.get_dof_limits()[0]
    expected = np.array([[joints[name]["lower"], joints[name]["upper"]] for name in names])
    if not np.isfinite(limits).all() or not np.allclose(limits, expected, atol=1e-5):
        raise ValueError("Simulator did not load finite authored joint limits")
    ranges = limits[:, 1] - limits[:, 0]
    if np.any(ranges <= 0):
        raise ValueError("Articulation has a locked or reversed joint range")
    tolerance = np.maximum(0.002, ranges * 0.025)
    # Same passive mechanism, with temporary test drives to cycle it deterministically.
    obj.set_gains(kps=np.full((1, len(names)), 800.0), kds=np.full((1, len(names)), 80.0))
    obj.set_max_efforts(np.full((1, len(names)), 87.0))
    steps = 0
    phase_steps = max(120, int(np.ceil(minimum_steps / (4 * len(names)))))
    initial_root = obj.get_world_poses()[0].copy()
    max_drift = 0.0

    def poses():
        # PhysX transforms use XYZW quaternions, unlike the core pose API's WXYZ.
        values = np.asarray(obj._physics_view.get_link_transforms()[0])
        if not np.isfinite(values).all():
            raise RuntimeError("Non-finite link state")
        result = {}
        for name, index in body_indices.items():
            matrix = np.eye(4)
            matrix[:3, :3] = Rotation.from_quat(values[index, 3:]).as_matrix()
            matrix[:3, 3] = values[index, :3]
            result[name] = matrix
        return result

    def advance(target, count=phase_steps, effort=None):
        nonlocal steps, max_drift
        obj.set_joint_position_targets(np.asarray(target, dtype=np.float32)[None, :])
        for _ in range(count):
            if effort is not None:
                obj.set_joint_efforts(np.asarray(effort, dtype=np.float32)[None, :])
            world.step(render=False)
            steps += 1
            q = obj.get_joint_positions()[0]
            if not np.isfinite(q).all() or not np.isfinite(obj.get_joint_velocities()).all():
                raise RuntimeError("Non-finite articulation state")
            if np.any(q < limits[:, 0] - tolerance) or np.any(q > limits[:, 1] + tolerance):
                raise RuntimeError(f"Joint exceeded its limit: {q.tolist()}")
            drift = float(np.linalg.norm(obj.get_world_poses()[0] - initial_root))
            max_drift = max(max_drift, drift)
            if fixed_base and drift > 0.002:
                raise RuntimeError("The cabinet base moved despite its world-fixed joint")
        return q.copy(), poses()

    zero = np.zeros(len(names))
    _, resting = advance(zero)
    results = []
    for index, name in enumerate(names):
        joint = joints[name]
        endpoint = limits[index, 1] if abs(limits[index, 1]) >= abs(limits[index, 0]) else limits[index, 0]
        target = zero.copy()
        target[index] = 0.8 * endpoint
        opened, opened_poses = advance(target)
        if abs(opened[index] - target[index]) > tolerance[index]:
            raise RuntimeError(
                f"{name} did not open under solver control: {opened[index]} vs {target[index]}"
            )
        others = np.arange(len(names)) != index
        if np.any(np.abs(opened[others]) > tolerance[others]):
            raise RuntimeError(f"Opening {name} moved another independent joint")
        parent, child = joint["parent"], joint["child"]
        relative = np.linalg.inv(opened_poses[parent]) @ opened_poses[child]
        rest_relative = np.linalg.inv(resting[parent]) @ resting[child]
        direction_error = 0.0
        if joint["kind"] == "prismatic":
            displacement = relative[:3, 3] - rest_relative[:3, 3]
            direction_error = float(np.linalg.norm(displacement - joint["axis"] * opened[index]))
            if direction_error > tolerance[index]:
                raise RuntimeError(f"{name} moves in the wrong direction: {displacement}")
        for parent_name, child_name in fixed:
            relative_fixed = np.linalg.inv(opened_poses[parent_name]) @ opened_poses[child_name]
            rest_fixed = np.linalg.inv(resting[parent_name]) @ resting[child_name]
            if not np.allclose(relative_fixed, rest_fixed, atol=0.002):
                raise RuntimeError(f"Fixed child {child_name} failed to follow {parent_name}")
        closed, _ = advance(zero)
        if abs(closed[index]) > tolerance[index]:
            raise RuntimeError(f"{name} failed to close: {closed[index]}")
        extremes = []
        for side in (1, 0):
            target[index] = limits[index, side] + (0.25 * ranges[index] * (1 if side else -1))
            measured, _ = advance(target)
            if abs(measured[index] - limits[index, side]) > tolerance[index]:
                raise RuntimeError(f"{name} did not reach its {'upper' if side else 'lower'} travel stop")
            extremes.append(float(measured[index]))
        advance(zero)
        results.append(
            {
                "name": name,
                "kind": joint["kind"],
                "passed": True,
                "lower": float(limits[index, 0]),
                "upper": float(limits[index, 1]),
                "opened": float(opened[index]),
                "closed": float(closed[index]),
                "upper_stop": extremes[0],
                "lower_stop": extremes[1],
                "direction_error_m": direction_error,
                "independent_motion": True,
                "fixed_children_follow": True,
            }
        )
    # Restore passive stiffness and pull each prismatic joint with a small force.
    obj.set_gains(kps=np.zeros((1, len(names))), kds=np.full((1, len(names)), 3.0))
    for index, name in enumerate(names):
        if joints[name]["kind"] != "prismatic":
            continue
        before = obj.get_joint_positions()[0, index]
        force = zero.copy()
        force[index] = 1.0 if limits[index, 1] > 0 else -1.0
        pulled, _ = advance(zero, count=60, effort=force)
        if (pulled[index] - before) * force[index] < min(0.005, ranges[index] * 0.05):
            raise RuntimeError(f"Passive drawer {name} did not respond to a one-newton pull")
        results[index]["passive_pull_displacement_m"] = float(pulled[index] - before)
    obj.set_joint_efforts(np.zeros((1, len(names))))
    return {
        "passed": True,
        "asset": str(asset.resolve()),
        "steps": steps,
        "physics_dt": 1 / 120,
        "articulation": {
            "passed": True,
            "joint_count": len(names),
            "body_count": obj.num_bodies,
            "fixed_base": fixed_base,
            "max_base_drift_m": max_drift,
            "joints": results,
        },
        "scope": "Solver-driven independent open/close, travel stops, signed motion direction, fixed-child "
        "following, finite state and passive drawer force response; not a robot grasp benchmark.",
    }
