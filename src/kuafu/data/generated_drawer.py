"""Observation/action contract for learning on the generated drawer scene."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


OBSERVATION_LAYOUT = [
    "robot_joint_position", "robot_joint_velocity", "cabinet_joint_position",
    "handle_position_world_m", "handle_quaternion_wxyz", "sun_stage_index",
]


def policy_observation(snapshot, stage_index):
    """Use the same measured state before teacher and learned-policy actions."""
    handle = snapshot["frames"]["top_drawer_handle_frame"]
    values = [*snapshot["robot_joints"], *snapshot["robot_joint_velocities"],
              *snapshot["cabinet_joints"], *handle["position"],
              *handle["quaternion_wxyz"], float(stage_index)]
    observation = np.asarray(values, dtype=np.float32)
    if not np.isfinite(observation).all():
        raise ValueError("Generated drawer observation contains non-finite values")
    return observation


def write_demonstration(path, observations, actions, next_observations, metadata):
    """Store one physically verified episode with aligned pre-action observations."""
    import h5py

    obs, act, nxt = (np.asarray(value, dtype=np.float32)
                     for value in (observations, actions, next_observations))
    if obs.ndim != 2 or act.ndim != 2 or nxt.shape != obs.shape or len(obs) != len(act) or len(obs) == 0:
        raise ValueError("A demonstration needs aligned, nonempty rank-2 transitions")
    if not all(np.isfinite(value).all() for value in (obs, act, nxt)):
        raise ValueError("Demonstration contains non-finite values")
    with h5py.File(path, "w") as handle:
        handle.attrs["metadata"] = json.dumps(metadata, sort_keys=True)
        group = handle.create_group("data/demo_0")
        group.attrs["num_samples"] = len(obs)
        group.create_dataset("obs/state", data=obs)
        group.create_dataset("next_obs/state", data=nxt)
        group.create_dataset("actions/absolute_joint", data=act)
        done = np.zeros(len(obs), dtype=bool)
        done[-1] = True
        group.create_dataset("dones", data=done)


def combine_demonstrations(sources, output):
    """Merge episodes while rejecting a change of robot, asset or SUN Program."""
    import h5py

    if len(sources) < 2:
        raise ValueError("Collect at least two episodes for a whole-episode validation split")
    contract = None
    total = 0
    with h5py.File(output, "w") as destination:
        data = destination.create_group("data")
        for index, path in enumerate(sources):
            with h5py.File(path, "r") as source:
                metadata = json.loads(source.attrs["metadata"])
                current = {name: metadata[name] for name in
                           ("observation_layout", "joint_names", "cabinet_joint_names",
                            "asset_hashes", "program_hashes", "target_travel_m")}
                if contract is not None and current != contract:
                    raise ValueError("Demonstrations have different scene or program contracts")
                contract = current
                source.copy("data/demo_0", data, name=f"demo_{index}")
                data[f"demo_{index}"].attrs["provenance"] = json.dumps(metadata, sort_keys=True)
                total += int(data[f"demo_{index}"].attrs["num_samples"])
        destination.attrs["metadata"] = json.dumps(contract, sort_keys=True)
        destination.attrs["source_episodes"] = json.dumps([str(Path(p).resolve()) for p in sources])
    return {"episodes": len(sources), "transitions": total, "contract": contract}
