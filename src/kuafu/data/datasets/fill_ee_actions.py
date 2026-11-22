#!/usr/bin/env python3
"""Fill missing action keys in an HDF5 dataset recorded in joint action mode.

Reverse-computes EE actions from stored observations and adds:
  actions/absolute_joint (9D): next actuated joint positions [arm(7), grip(2)]
  actions/relative_joint (9D): copy of existing 'actions'   [arm(7), grip(2)]
  actions/absolute_ee   (11D): [next_ee_pos(3), next_ee_rot6d(6), grip(2)]
  actions/relative_ee    (8D): [pos_delta(3), aa_delta(3), grip(2)]

The obs layout (slices for EE pos/rot and joint positions) is auto-detected
from the rigid-object names stored in initial_state/rigid_object.  Use
--env-name to force a specific layout or --list-envs to see all known envs.

Known env layouts
-----------------
pen_cup (FR3FinRayPenCupEnvLearned)   — obs/state 31D, full_state 65D
  objects: marker, pen_holder
  [6:9]   fingertip_pos   (obs/state)
  [21:27] fingertip_rot6d (obs/state)
  [36:45] joint_pos       (obs/full_state)

hang_cup (FR3FinRayHangCupEnvLearned) — obs/state 31D, full_state 65D
  objects: mug, cup_hanger
  [6:9]   fingertip_pos   (obs/state)
  [21:27] fingertip_rot6d (obs/state)
  [36:45] joint_pos       (obs/full_state)

drawer (FR3FinRayDrawerEnvLearned)    — obs/state 22D, full_state 56D
  objects: handle
  [3:6]   fingertip_pos   (obs/state)
  [12:18] fingertip_rot6d (obs/state)
  [27:36] joint_pos       (obs/full_state)

stack2 (FR3FinRayStack2EnvLearned)    — obs/state 64D (joint_pos included)
  objects: cube_1, cube_2
  [6:9]   fingertip_pos   (obs/state)
  [24:30] fingertip_rot6d (obs/state)
  [36:45] joint_pos       (obs/state — no separate full_state)

Usage:
  python fill_ee_actions.py <path/to/dataset.hdf5> [--inplace] [--overwrite]
  python fill_ee_actions.py <path/to/dataset.hdf5> --env-name pen_cup
  python fill_ee_actions.py --list-envs

Without --inplace a new file is written with suffix _filled.
"""

import argparse
import shutil
from pathlib import Path

from kuafu.paths import resolve_artifact_path
from typing import NamedTuple

import h5py
import numpy as np
import torch

from kuafu.mpc.simulation.spatial_utils import rotation_6d_to_quat
from isaaclab.utils.math import quat_box_minus

# ---------------------------------------------------------------------------
# Obs layout registry
# ---------------------------------------------------------------------------

class ObsLayout(NamedTuple):
    ee_pos:    slice   # slice into the ee obs source
    ee_rot6d:  slice   # slice into the ee obs source
    joint_pos: slice   # slice into the joint obs source
    joint_in_full_state: bool = True   # False → read joint_pos from obs/state
    ee_in_full_state: bool = False     # True → read ee_pos/ee_rot6d from obs/full_state


# Maps canonical env name → layout
LAYOUTS: dict[str, ObsLayout] = {
    "pen_cup": ObsLayout(
        ee_pos=slice(6, 9),
        ee_rot6d=slice(24, 30),
        joint_pos=slice(36, 45),
        joint_in_full_state=True,
        ee_in_full_state=True,
    ),
    "hang_cup": ObsLayout(
        ee_pos=slice(6, 9),
        ee_rot6d=slice(24, 30),
        joint_pos=slice(36, 45),
        joint_in_full_state=True,
        ee_in_full_state=True,
    ),
    "drawer": ObsLayout(
        ee_pos=slice(3, 6),
        ee_rot6d=slice(12, 18),
        joint_pos=slice(27, 36),
        joint_in_full_state=True,
        ee_in_full_state=False,
    ),
    "stack2": ObsLayout(
        ee_pos=slice(6, 9),
        ee_rot6d=slice(24, 30),
        joint_pos=slice(36, 45),
        joint_in_full_state=False,
        ee_in_full_state=False,
    ),
}

# Rigid-object key sets → env name (subset match: all listed keys must be present)
_DETECTION_RULES: list[tuple[frozenset[str], str]] = [
    (frozenset({"marker", "pen_holder"}), "pen_cup"),
    (frozenset({"mug", "cup_hanger"}),    "hang_cup"),
    (frozenset({"cube_1"}),               "stack2"),
    (frozenset({"handle"}),               "drawer"),
]

GRIP_ACTION_SLICE = slice(7, 9)   # last 2 dims of 9D 'actions' tensor

# Env physics: sim_dt=1/120, decimation=2 → dt=1/60. EE pos action scale separate from joint scale.
_SIM_DT = 1.0 / 120.0
_DECIMATION = 2
_DT = _SIM_DT * _DECIMATION          # 1/60 s
_EE_ACTION_SCALE = 1.0               # matches BaseLearnedEnvCfg.ee_action_scale
_EE_POS_NORM = _DT * _EE_ACTION_SCALE  # divide raw pos_delta by this to get normalized action

KEYS_TO_FILL = frozenset({
    "actions/absolute_joint",
    "actions/relative_joint",
    "actions/absolute_ee",
    "actions/relative_ee",
})

# In old datasets "actions" is a flat 9D dataset (= relative_joint in new convention).
# We detect this and migrate before writing sub-keys.
def _migrate_flat_actions(grp: h5py.Group) -> np.ndarray | None:
    """If 'actions' is a flat dataset, read it, delete it, return the data.
    The caller is responsible for writing it back as actions/relative_joint."""
    node = grp["actions"]
    if isinstance(node, h5py.Dataset):
        data = node[:]
        del grp["actions"]
        return data
    return None  # already a group


# ---------------------------------------------------------------------------
# Layout detection
# ---------------------------------------------------------------------------

def detect_layout(f: h5py.File) -> tuple[str, ObsLayout]:
    """Infer env layout from rigid-object names in the first demo."""
    first_demo = sorted(f["data"].keys())[0]
    grp = f["data"][first_demo]

    obj_keys: set[str] = set()
    if "initial_state/rigid_object" in grp:
        obj_keys = set(grp["initial_state/rigid_object"].keys())

    for required_keys, env_name in _DETECTION_RULES:
        if required_keys <= obj_keys:
            return env_name, LAYOUTS[env_name]

    raise ValueError(
        f"Could not auto-detect env layout from rigid objects {obj_keys}. "
        f"Pass --env-name explicitly. Known envs: {list(LAYOUTS)}"
    )


# ---------------------------------------------------------------------------
# Per-demo processing
# ---------------------------------------------------------------------------

def process_demo(grp: h5py.Group, keys_to_fill: set, layout: ObsLayout) -> None:
    # Migrate old-convention datasets where "actions" is a flat dataset
    # (= relative_joint). After migration "actions" becomes a group.
    flat_actions = _migrate_flat_actions(grp)

    obs_state      = torch.from_numpy(grp["obs/state"][:]).float()        # (T, D)
    next_obs_state = torch.from_numpy(grp["next_obs/state"][:]).float()   # (T, D)

    if flat_actions is not None:
        actions = torch.from_numpy(flat_actions).float()                   # (T, 9) — old convention
    else:
        actions = torch.from_numpy(grp["actions/relative_joint"][()]).float()  # type: ignore[index]
    grip = actions[:, GRIP_ACTION_SLICE]                                   # (T, 2)

    # Choose EE obs source: full_state or state
    if layout.ee_in_full_state:
        ee_src      = torch.from_numpy(grp["obs/full_state"][:]).float()       # type: ignore[index]
        next_ee_src = torch.from_numpy(grp["next_obs/full_state"][:]).float()  # type: ignore[index]
    else:
        ee_src      = obs_state
        next_ee_src = next_obs_state

    # Detect broken recording: next_obs == obs (post-step recorder didn't update).
    # Fall back to consecutive-frame diff: next[t] = src[t+1], dropping the last step.
    if torch.allclose(next_ee_src, ee_src):
        next_ee_src = ee_src[1:].clone()
        ee_src      = ee_src[:-1]
        grip        = grip[:-1]
        actions     = actions[:-1]
        obs_state      = obs_state[:-1]
        next_obs_state = next_obs_state[:-1]

    # Drop step 0 from all arrays: EE teleports from reset position on the first step,
    # making all delta-action keys meaningless at t=0. Uniform trim keeps lengths consistent.
    ee_src         = ee_src[1:]
    next_ee_src    = next_ee_src[1:]
    grip           = grip[1:]
    actions        = actions[1:]
    obs_state      = obs_state[1:]
    next_obs_state = next_obs_state[1:]

    ee_pos      = ee_src[:, layout.ee_pos]          # (T-1, 3)
    ee_rot6d    = ee_src[:, layout.ee_rot6d]        # (T-1, 6)
    next_ee_pos = next_ee_src[:, layout.ee_pos]     # (T-1, 3)
    next_rot6d  = next_ee_src[:, layout.ee_rot6d]   # (T-1, 6)

    if "actions/absolute_ee" in keys_to_fill:
        _write(grp, "actions/absolute_ee",
               torch.cat([next_ee_pos, next_rot6d, grip], dim=-1).numpy())

    if "actions/relative_ee" in keys_to_fill:
        curr_quat = rotation_6d_to_quat(ee_rot6d)          # (T-1, 4) wxyz
        next_quat = rotation_6d_to_quat(next_rot6d)        # (T-1, 4) wxyz
        pos_delta = (next_ee_pos - ee_pos) / _EE_POS_NORM      # (T-1, 3) normalized
        aa_delta  = quat_box_minus(next_quat, curr_quat) / _EE_POS_NORM       # (T-1, 3) axis-angle in rad
        _write(grp, "actions/relative_ee",
               torch.cat([pos_delta, aa_delta, grip], dim=-1).numpy())

    if "actions/absolute_joint" in keys_to_fill:
        if layout.joint_in_full_state:
            src = torch.from_numpy(grp["next_obs/full_state"][1:]).float()  # type: ignore[index]
        else:
            src = next_obs_state
        _write(grp, "actions/absolute_joint", src[:, layout.joint_pos].numpy())

    if "actions/relative_joint" in keys_to_fill:
        _write(grp, "actions/relative_joint", actions.numpy())


def _write(grp: h5py.Group, key: str, data: np.ndarray) -> None:
    if key in grp:
        del grp[key]
    grp.create_dataset(key, data=data.astype(np.float32))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", default="Trajectories/fr3_finray_table_pen_cup.hdf5", help="Path to the HDF5 dataset file.")
    parser.add_argument("--inplace", action="store_true",
                        help="Modify in place (default: write <name>_filled.hdf5).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing keys (default: skip already-filled demos).")
    parser.add_argument("--env-name", default="pen_cup",
                        help=f"Force a specific obs layout. Choices: {list(LAYOUTS)}")
    parser.add_argument("--list-envs", action="store_true",
                        help="Print known env layouts and exit.")

    parser.add_argument("--enable_cameras", action="store_true")
    args = parser.parse_args()

    if args.list_envs:
        for name, layout in LAYOUTS.items():
            print(f"{name}:")
            print(f"  ee_pos    = obs/state[{layout.ee_pos.start}:{layout.ee_pos.stop}]")
            print(f"  ee_rot6d  = obs/state[{layout.ee_rot6d.start}:{layout.ee_rot6d.stop}]")
            src = "full_state" if layout.joint_in_full_state else "state"
            print(f"  joint_pos = obs/{src}[{layout.joint_pos.start}:{layout.joint_pos.stop}]")
        return

    if args.dataset is None:
        parser.error("dataset argument is required (unless --list-envs).")

    src = resolve_artifact_path(args.dataset, "Trajectory dataset")
    if not src.exists():
        raise FileNotFoundError(src)

    dst = src if args.inplace else src.with_name(src.stem + "_filled" + src.suffix)
    if not args.inplace and not dst.exists():
        shutil.copy2(src, dst)
        print(f"Copied {src} → {dst}")

    with h5py.File(dst, "r+") as f:
        if args.env_name:
            if args.env_name not in LAYOUTS:
                raise ValueError(f"Unknown env '{args.env_name}'. Known: {list(LAYOUTS)}")
            env_name = args.env_name
            layout   = LAYOUTS[env_name]
        else:
            env_name, layout = detect_layout(f)

        print(f"Env layout: {env_name}  "
              f"(ee_pos={layout.ee_pos}, ee_rot6d={layout.ee_rot6d}, "
              f"joint_pos={layout.joint_pos}, "
              f"from_full_state={layout.joint_in_full_state})")

        demos = sorted(f["data"].keys())
        print(f"Processing {len(demos)} demos in {dst} ...")
        for i, demo_key in enumerate(demos):
            grp = f["data"][demo_key]
            # Old convention: "actions" is a flat Dataset → always needs migration.
            # New convention: "actions" is a Group → check which sub-keys exist.
            if isinstance(grp.get("actions"), h5py.Dataset) or args.overwrite:  # type: ignore[union-attr]
                missing = KEYS_TO_FILL
            else:
                missing = {k for k in KEYS_TO_FILL if k not in grp}  # type: ignore[operator]
            if missing:
                process_demo(grp, missing, layout)
            if (i + 1) % 100 == 0 or (i + 1) == len(demos):
                print(f"  {i + 1}/{len(demos)} done")

    print("Done.")


if __name__ == "__main__":
    main()
