"""Merge and repair the recorder HDF5 datasets consumed by native BC training.

Each file contains ``data/demo_*`` groups with actions, observations, rewards,
and termination flags. Collection launchers and the workflow share these helpers.
"""

import os
import sys

import h5py
import numpy as np


# -----------------------------------------------------------------------------
# Pure HDF5 format (single-file, data/demo_*)
# -----------------------------------------------------------------------------

_TIMESERIES_PREFIXES = ("actions", "obs/", "next_obs/", "dones", "rewards")


def merge_hdf5_files(
    input_files: list[str],
    output_file: str,
    *,
    strict: bool = True,
) -> int:
    """Merge multiple pure-HDF5 dataset files into one.

    Expects each input file to have a top-level "data" group with demo_0, demo_1, ...
    Writes combined data/demo_0, data/demo_1, ... to output_file (order preserved).

    Args:
        input_files: Paths to HDF5 files (order preserved when concatenating demos).
        output_file: Path for the combined output HDF5 file.
        strict: If True (default), raise on missing files or invalid structure.
                If False, skip missing files and files without "data" (not recommended).

    Returns:
        Number of episodes (demos) written to output_file.

    Raises:
        FileNotFoundError: If strict and any input file is missing.
        ValueError: If strict and any input file does not contain a "data" group.
    """
    if strict:
        for fpath in input_files:
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"Input file does not exist: {fpath}")

    with h5py.File(output_file, "w") as out_f:
        episode_idx = 0
        attrs_copied = False
        for fpath in sorted(input_files):
            if not os.path.exists(fpath):
                if strict:
                    raise FileNotFoundError(f"Input file does not exist: {fpath}")
                continue
            try:
                with h5py.File(fpath, "r") as in_f:
                    if "data" not in in_f:
                        if strict:
                            raise ValueError(
                                f"File has no top-level 'data' group (invalid pure HDF5 format): {fpath}"
                            )
                        continue
                    data_grp = in_f["data"]
                    demo_keys = sorted(
                        (k for k in data_grp.keys() if k.startswith("demo_")),
                        key=lambda k: int(k.split("_")[-1]) if k.split("_")[-1].isdigit() else 0,
                    )
                    if not demo_keys and strict:
                        raise ValueError(f"File has 'data' group but no demo_* keys: {fpath}")
                    for demo_key in demo_keys:
                        in_f.copy(f"data/{demo_key}", out_f, f"data/demo_{episode_idx}")
                        episode_idx += 1
                    if not attrs_copied and "env_args" in data_grp.attrs:
                        out_f.require_group("data").attrs["env_args"] = data_grp.attrs["env_args"]
                        attrs_copied = True
            except OSError as e:
                if strict:
                    raise ValueError(
                        f"Truncated or corrupt HDF5 file (cannot open): {fpath}. "
                        "Remove it from the input list or re-collect."
                    ) from e
                print(f"[WARN] Skipping truncated/corrupt file: {fpath}", file=sys.stderr)

    return episode_idx


def fix_demo_length_mismatches(
    hdf5_path: str,
    *,
    strict: bool = False,
) -> list[str]:
    """Fix demos where per-timestep keys have inconsistent lengths (pure HDF5 format).

    The recorder can produce an off-by-one between obs/actions and dones/rewards.
    This trims mismatched keys to the shortest length and syncs the ``num_samples``
    attribute. Only considers timeseries keys (actions, obs/, next_obs/, dones,
    rewards). Modifies the file in-place.

    Args:
        hdf5_path: Path to the HDF5 file.
        strict: If True, raise ValueError when any demo required fixing (caller can
                then inspect or reject the dataset).

    Returns:
        List of demo names that were modified.

    Raises:
        ValueError: If strict and at least one demo was fixed (length mismatches found).
    """
    fixed: list[str] = []
    with h5py.File(hdf5_path, "r+") as f:
        if "data" not in f:
            return fixed
        for demo_name in sorted(f["data"].keys()):
            grp = f[f"data/{demo_name}"]

            lengths: dict[str, int] = {}

            def _collect(name: str, obj: object) -> None:
                if isinstance(obj, h5py.Dataset):
                    if any(name.startswith(p) for p in _TIMESERIES_PREFIXES):
                        lengths[name] = obj.shape[0]

            grp.visititems(_collect)

            if not lengths:
                continue
            min_len = min(lengths.values())
            if all(v == min_len for v in lengths.values()):
                if grp.attrs.get("num_samples", min_len) == min_len:
                    continue

            # For debugging / dataset inspection: report which keys are "dangling".
            longer_keys: list[str] = []
            shorter_keys: list[str] = []
            for name, cur_len in lengths.items():
                if cur_len > min_len:
                    longer_keys.append(name)
                elif cur_len < min_len:
                    shorter_keys.append(name)
            if longer_keys or shorter_keys:
                print(
                    f"[LENGTH_MISMATCH] File '{hdf5_path}', demo '{demo_name}': "
                    f"min_len={min_len}, "
                    f"longer={sorted(longer_keys) if longer_keys else []}, "
                    f"shorter={sorted(shorter_keys) if shorter_keys else []}"
                )

            for name in longer_keys:
                path = f"data/{demo_name}/{name}"
                data = np.array(f[path])[:min_len]
                del f[path]
                f[path] = data
            grp.attrs["num_samples"] = min_len
            fixed.append(demo_name)

    if fixed and strict:
        raise ValueError(
            f"Length mismatches were fixed in {len(fixed)} demo(s) in {hdf5_path}: {fixed}. "
            "Inspect the dataset or re-record to avoid inconsistent lengths."
        )
    return fixed
