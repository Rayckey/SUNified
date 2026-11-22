#!/usr/bin/env python3
"""Combine multiple **pure HDF5** dataset files into one.

Use this script when you have multiple single-file HDF5 datasets with the
structure: top-level group "data" containing demo_0, demo_1, ... (each with
actions, obs/, next_obs/, dones, rewards, etc.). Typical use: merging
per-worker outputs from MPC or other recorders that write one .hdf5 per run.

The pipeline consumes this recorder format directly for native MLP BC training.
Missing/corrupt files and inconsistent trajectory lengths are errors unless
explicitly allowed by ``--skip-bad-files`` or ``--no-strict-length``.
"""

import argparse
import sys

from kuafu.paths import resolve_artifact_path

import h5py
import numpy as np

from kuafu.data.dataset_utils import fix_demo_length_mismatches, merge_hdf5_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine multiple pure HDF5 dataset files into one (data/demo_* format).",
    )
    parser.add_argument(
        "input_files",
        nargs="+",
        help="Paths to HDF5 files to combine (order preserved).",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Path for the combined output HDF5 file.",
    )
    parser.add_argument(
        "--no-strict-length",
        action="store_true",
        help="If set, allow fixing demo length mismatches instead of failing.",
    )
    parser.add_argument(
        "--skip-bad-files",
        action="store_true",
        help="If set, skip truncated/corrupt or missing input files instead of failing.",
    )
    parser.add_argument(
        "--robot-name",
        choices=["fr3_finray", "kinova", "panda"],
        default=None,
        help="Optional robot label to store in output HDF5 metadata.",
    )
    args = parser.parse_args()
    args.input_files = [str(resolve_artifact_path(path, "Input trajectory")) for path in args.input_files]
    args.output = str(resolve_artifact_path(args.output, "Combined trajectory dataset"))

    try:
        merge_hdf5_files(args.input_files, args.output, strict=not args.skip_bad_files)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    try:
        fixed = fix_demo_length_mismatches(args.output, strict=not args.no_strict_length)
        if fixed:
            print(f"[FIX] Adjusted length in {len(fixed)} demo(s): {fixed}")
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        print(
            "Re-run with --no-strict-length to allow auto-fix, or fix the source data.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.robot_name:
        with h5py.File(args.output, "a") as f:
            f.attrs["robot_name"] = args.robot_name
            f.require_group("data").attrs["robot_name"] = args.robot_name
        print(f"[META] robot_name={args.robot_name}")

    # ------------------------------------------------------------------
    # Compute global standard deviations for actions, observations, rewards
    # across all demos in the combined file.
    # ------------------------------------------------------------------
    def _update_accumulator(acc, data: np.ndarray):
        """Update (count, sum, sum_sq) accumulator along time axis (axis 0)."""
        if acc is None:
            acc = {
                "count": data.shape[0],
                "sum": data.sum(axis=0, dtype=np.float64),
                "sum_sq": np.square(data, dtype=np.float64).sum(axis=0),
            }
        else:
            acc["count"] += data.shape[0]
            acc["sum"] += data.sum(axis=0, dtype=np.float64)
            acc["sum_sq"] += np.square(data, dtype=np.float64).sum(axis=0)
        return acc

    def _is_numeric_array(data: np.ndarray) -> bool:
        return bool(np.issubdtype(data.dtype, np.number) and not np.issubdtype(data.dtype, np.bool_))

    actions_acc: dict[str, dict] = {}
    rewards_acc: dict | None = None
    obs_acc: dict[str, dict] = {}

    with h5py.File(args.output, "r") as f:
        if "data" not in f:
            print("[STATS] No 'data' group found in output file; skipping stats.")
            return

        for demo_name in sorted(f["data"].keys()):
            grp = f[f"data/{demo_name}"]

            # Actions: flat (T, ...) dataset or recorder-style actions/* group.
            if "actions" in grp:
                actions_obj = grp["actions"]
                if isinstance(actions_obj, h5py.Dataset):
                    actions = np.asarray(actions_obj)
                    if (
                        actions.ndim >= 1
                        and actions.shape[0] > 0
                        and _is_numeric_array(actions)
                    ):
                        actions_acc["actions"] = _update_accumulator(
                            actions_acc.get("actions"), actions
                        )
                elif isinstance(actions_obj, h5py.Group):

                    def _collect_actions(name: str, obj: object) -> None:
                        if not isinstance(obj, h5py.Dataset):
                            return
                        data = np.asarray(obj)
                        if data.ndim < 1 or data.shape[0] == 0:
                            return
                        if not _is_numeric_array(data):
                            return
                        key = f"actions/{name}"
                        actions_acc[key] = _update_accumulator(actions_acc.get(key), data)

                    actions_obj.visititems(_collect_actions)

            # Rewards: timeseries dataset with shape (T,) or (T, 1, ...)
            if "rewards" in grp:
                rewards = np.asarray(grp["rewards"])
                if (
                    rewards.ndim >= 1
                    and rewards.shape[0] > 0
                    and _is_numeric_array(rewards)
                ):
                    rewards_acc = _update_accumulator(rewards_acc, rewards)

            # Observations: any dataset under obs/ prefix within the demo group.
            def _collect_obs(name: str, obj: object) -> None:
                if not isinstance(obj, h5py.Dataset):
                    return
                if not name.startswith("obs/"):
                    return
                data = np.asarray(obj)
                if data.ndim < 1 or data.shape[0] == 0:
                    return
                if not _is_numeric_array(data):
                    return
                key = name  # e.g., "obs/qpos"
                obs_acc[key] = _update_accumulator(obs_acc.get(key), data)

            grp.visititems(_collect_obs)

    def _compute_mean_std(acc: dict | None):
        if acc is None:
            return None, None
        count = acc["count"]
        if count == 0:
            return None, None
        mean = acc["sum"] / count
        var = acc["sum_sq"] / count - np.square(mean)
        var = np.maximum(var, 0.0)
        std = np.sqrt(var)
        return mean, std

    if actions_acc:
        print("[STATS] action component means/stds:")
        for key in sorted(actions_acc.keys()):
            mean, std = _compute_mean_std(actions_acc[key])
            if std is not None:
                print(f"  - {key} mean shape {mean.shape}:\n{mean}")
                print(f"    {key} std  shape {std.shape}:\n{std}")
    else:
        print("[STATS] No numeric actions data found; cannot compute mean/std.")

    rewards_mean, rewards_std = _compute_mean_std(rewards_acc)
    if rewards_std is not None:
        print(f"[STATS] rewards mean shape {rewards_mean.shape}:\n{rewards_mean}")
        print(f"[STATS] rewards std  shape {rewards_std.shape}:\n{rewards_std}")
    else:
        print("[STATS] No rewards data found; cannot compute mean/std.")

    if obs_acc:
        print("[STATS] observation component means/stds:")
        for key in sorted(obs_acc.keys()):
            mean, std = _compute_mean_std(obs_acc[key])
            if std is not None:
                print(f"  - {key} mean shape {mean.shape}:\n{mean}")
                print(f"    {key} std  shape {std.shape}:\n{std}")
    else:
        print("[STATS] No observation datasets (obs/*) found; cannot compute stds.")


if __name__ == "__main__":
    main()
