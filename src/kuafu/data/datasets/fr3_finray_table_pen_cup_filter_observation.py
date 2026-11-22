#!/usr/bin/env python3
"""
Filter specific, semantically chosen dimensions out of obs/state (and
next_obs/state) in the FR3FinRay pen-cup dataset and write a new HDF5 with
reduced observation dim.

Usage (from repo root):

    python src/kuafu/data/datasets/fr3_finray_table_pen_cup_filter_observation.py \
        --input src/kuafu/data/datasets/fr3_finray_table_pen_cup.hdf5 \
        --output src/kuafu/data/datasets/fr3_finray_table_pen_cup_filtered.hdf5

The observation layout for pen-cup (56 dims, see replay_utils.set_env_from_observation_pen_cup):

    [0:3]   marker xyz
    [3:6]   pen_holder xyz
    [6:9]   panda_fingertip pos
    [9:12]  panda_link4 pos
    [12:16] marker quat (wxyz)
    [16:20] pen_holder quat (wxyz)
    [20:24] panda_fingertip quat
    [24:28] panda_link4 quat
    [28:37] joint_position (9)
    [37:46] joint_velocity (9)
    [46:55] joint_effort (9)
    [55:56] stage (1)

We drop only:
  - pen_holder_z               (index 5)
  - pen_holder_quat x, y       (indices 17, 18)
  - joint_effort (9 dims)      (indices 46–54)

All remaining dims are kept. The script stores the selected indices as an HDF5
attribute so training / replay code can recover the mapping if needed.
"""

import argparse
import sys
from pathlib import Path

from kuafu.paths import resolve_artifact_path
from typing import List

import h5py
import numpy as np


def filter_obs_dimensions(
    input_path: str,
    output_path: str,
    obs_key: str = "obs/state",
    next_obs_key: str = "next_obs/state",
) -> None:
    """
    Create a new HDF5 where obs/state and next_obs/state have low-variance
    dimensions removed (based on std_threshold).

    Args:
        input_path: Source HDF5.
        output_path: Destination HDF5.
        obs_key: Name of obs dataset inside each demo.
        next_obs_key: Name of next_obs dataset inside each demo.
    """
    # For pen-cup, obs/state is 56-dim. We explicitly drop:
    #   5         : pen_holder_z
    #   17, 18    : pen_holder_quat x, y (wxyz)
    #   46..54    : joint_effort (9 dims)
    D_EXPECTED = 56
    DROP_INDICES = [5, 17, 18] + list(range(46, 55))

    print(f"[FILTER] Input:  {input_path}")
    print(f"[FILTER] Output: {output_path}")
    print(f"[FILTER] Expected obs/state dim: {D_EXPECTED}")
    print(f"[FILTER] Dropping indices: {DROP_INDICES}")

    with h5py.File(input_path, "r") as src, h5py.File(output_path, "w") as dst:
        if "data" not in src:
            raise ValueError(f"No 'data' group found in {input_path}")

        data_in = src["data"]
        data_out = dst.create_group("data")

        # Copy env_args attribute if present.
        if "env_args" in data_in.attrs:
            data_out.attrs["env_args"] = data_in.attrs["env_args"]

        # We need to infer the actual obs dimension from the first demo.
        first_demo = None
        for name in sorted(data_in.keys()):
            if obs_key in data_in[name]:
                first_demo = name
                break
        if first_demo is None:
            raise RuntimeError(f"No demo with {obs_key} found in {input_path}")

        sample = np.asarray(data_in[first_demo][obs_key])
        if sample.ndim != 2:
            raise ValueError(
                f"Sample {first_demo}/{obs_key} has shape {sample.shape}, expected (T, D)."
            )
        D = sample.shape[1]
        if D != D_EXPECTED:
            print(
                f"[FILTER][WARN] Expected obs dim {D_EXPECTED}, but dataset has {D}. "
                "Proceeding with provided DROP_INDICES; please verify indices."
            )

        keep_mask = np.ones(D, dtype=bool)
        for idx in DROP_INDICES:
            if 0 <= idx < D:
                keep_mask[idx] = False
        keep_indices: List[int] = np.nonzero(keep_mask)[0].tolist()
        dropped_indices: List[int] = np.nonzero(~keep_mask)[0].tolist()

        print(f"[FILTER] Keeping  {len(keep_indices)} dims")
        print(f"[FILTER] Dropping {len(dropped_indices)} dims (after bounds check): {dropped_indices}")

        if len(keep_indices) == 0:
            raise RuntimeError("All obs/state dims would be dropped; check DROP_INDICES.")

        # Also store mapping of kept indices for downstream use.
        data_out.attrs["state_keep_indices"] = np.asarray(keep_indices, dtype=np.int64)
        data_out.attrs["state_dropped_indices"] = np.asarray(
            dropped_indices, dtype=np.int64
        )

        for demo_name in sorted(data_in.keys()):
            grp_in = data_in[demo_name]
            grp_out = data_out.create_group(demo_name)

            # Copy attributes (e.g., num_samples).
            for k, v in grp_in.attrs.items():
                grp_out.attrs[k] = v

            # Recursively walk all datasets under this demo (so we see "obs/state").
            def _copy_item(name: str, obj: h5py.Dataset | h5py.Group):
                # name is relative to grp_in, e.g. "actions", "obs/state"
                if isinstance(obj, h5py.Group):
                    # Ensure the group exists in the output.
                    if name == "":
                        return
                    grp_out.require_group(name)
                    return

                # Dataset path handling: split into parent group + dataset name.
                if "/" in name:
                    parent_name, ds_name = name.rsplit("/", 1)
                    parent_out = grp_out.require_group(parent_name)
                else:
                    ds_name = name
                    parent_out = grp_out

                arr = np.asarray(obj)

                # Filter obs/state and next_obs/state only.
                if name == obs_key or name == next_obs_key:
                    if arr.ndim != 2 or arr.shape[1] != D:
                        raise ValueError(
                            f"Dataset '{demo_name}/{name}' has shape {arr.shape}, "
                            f"expected (T, {D})."
                        )
                    arr = arr[:, keep_indices]

                dset_out = parent_out.create_dataset(
                    ds_name,
                    data=arr,
                    compression=obj.compression,
                    shuffle=obj.shuffle,
                    fletcher32=obj.fletcher32,
                    dtype=arr.dtype,
                )
                for k, v in obj.attrs.items():
                    dset_out.attrs[k] = v

            grp_in.visititems(_copy_item)

    print(f"[FILTER] Wrote filtered dataset to {output_path}")
    print(
        "[FILTER] NOTE: New obs/state dim is "
        f"{len(keep_indices)} (was {D_EXPECTED})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter low-variance obs/state dims from FR3FinRay pen-cup dataset."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="Trajectories/fr3_finray_table_pen_cup.hdf5",
        help="Path to input HDF5 dataset.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="Trajectories/fr3_finray_table_pen_cup_filtered.hdf5",
        help="Path to output filtered HDF5 dataset.",
    )

    args = parser.parse_args()
    try:
        input_path = resolve_artifact_path(args.input, "Input trajectory")
        output_path = resolve_artifact_path(args.output, "Filtered trajectory")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        filter_obs_dimensions(
            input_path=str(input_path),
            output_path=str(output_path),
            obs_key="obs/state",
            next_obs_key="next_obs/state",
        )
    except Exception as e:
        print(f"[FILTER][ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

