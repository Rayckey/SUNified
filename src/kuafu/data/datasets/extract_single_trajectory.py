import argparse
import random
from pathlib import Path

from kuafu.paths import resolve_artifact_path

import h5py


def copy_group(src, dst):
    """Recursively copy a group/dataset from src to dst."""
    for key in src.keys():
        item = src[key]
        if isinstance(item, h5py.Group):
            grp = dst.require_group(key)
            for attr_key, attr_val in item.attrs.items():
                grp.attrs[attr_key] = attr_val
            copy_group(item, grp)
        else:
            dst.create_dataset(key, data=item[()])
            for attr_key, attr_val in item.attrs.items():
                dst[key].attrs[attr_key] = attr_val


def main():
    parser = argparse.ArgumentParser(description="Extract a single trajectory from an HDF5 dataset.")
    parser.add_argument("--input", type=str, default="Trajectories/fr3_finray_table_pen_cup.hdf5", help="Path to the input HDF5 dataset.")
    parser.add_argument("--output", type=str, default="Trajectories/fr3_finray_table_pen_cup_single_trajectory.hdf5", help="Path to the output HDF5 file.")
    parser.add_argument("--demo", type=int, default=None, help="Index of the demo to extract (default: random).")
    args = parser.parse_args()
    input_path = resolve_artifact_path(args.input, "Input trajectory")
    output_path = resolve_artifact_path(args.output, "Extracted trajectory")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    args.input = str(input_path)
    args.output = str(output_path)

    with h5py.File(args.input, "r") as src_f:
        n_demos = len(src_f["data"].keys())
        demo_idx = args.demo if args.demo is not None else random.randrange(n_demos)
        demo_key = f"demo_{demo_idx}"

        if "data" not in src_f or demo_key not in src_f["data"]:
            available = list(src_f["data"].keys()) if "data" in src_f else []
            raise ValueError(f"'{demo_key}' not found. Available demos: {available[:10]}")

        with h5py.File(args.output, "w") as dst_f:
            # Copy top-level 'data' group attributes
            data_grp = dst_f.create_group("data")
            for attr_key, attr_val in src_f["data"].attrs.items():
                data_grp.attrs[attr_key] = attr_val

            # Copy the single demo under 'data/demo_0'
            demo_dst = data_grp.create_group("demo_0")
            src_demo = src_f["data"][demo_key]
            for attr_key, attr_val in src_demo.attrs.items():
                demo_dst.attrs[attr_key] = attr_val
            copy_group(src_demo, demo_dst)

    print(f"Extracted '{demo_key}' from '{args.input}' -> '{args.output}' as 'demo_0'.")


if __name__ == "__main__":
    main()
