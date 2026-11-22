#!/usr/bin/env python
"""
Inspect an IsaacLab/robomimic-style HDF5 trajectory file.

This prints:
- the dataset tree (groups + datasets)
- candidate observation/action dataset paths
- basic sanity stats (shape, dtype, NaN/Inf, min/max, percentiles)

Example:
  python datasets/BC/inspect_hdf5_keys.py \
    --dataset datasets/fr3_finray_table_pen_cup_single_trajectory.hdf5
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import h5py
import numpy as np

from kuafu.paths import resolve_artifact_path


@dataclass(frozen=True)
class Stats:
    shape: tuple[int, ...]
    dtype: str
    nan: int
    inf: int
    finite_frac: float
    min: float | None
    max: float | None
    mean: float | None
    std: float | None
    quantiles: dict[float, float]


def compute_stats(x: np.ndarray) -> Stats:
    x = np.asarray(x)
    finite = np.isfinite(x)
    nan = int(np.isnan(x).sum())
    inf = int(np.isinf(x).sum())
    finite_frac = float(finite.mean()) if x.size > 0 else 0.0
    if finite.any():
        xf = x[finite]
        qs = [0.0, 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 1.0]
        qv = np.quantile(xf, qs)
        quantiles = {float(k): float(v) for k, v in zip(qs, qv)}
        return Stats(
            shape=tuple(x.shape),
            dtype=str(x.dtype),
            nan=nan,
            inf=inf,
            finite_frac=finite_frac,
            min=float(xf.min()),
            max=float(xf.max()),
            mean=float(xf.mean()),
            std=float(xf.std()),
            quantiles=quantiles,
        )
    return Stats(
        shape=tuple(x.shape),
        dtype=str(x.dtype),
        nan=nan,
        inf=inf,
        finite_frac=finite_frac,
        min=None,
        max=None,
        mean=None,
        std=None,
        quantiles={},
    )


def print_stats(name: str, ds: h5py.Dataset, sample_rows: int) -> None:
    n = ds.shape[0] if len(ds.shape) > 0 else 1
    n_read = min(n, sample_rows)
    x = np.array(ds[:n_read])
    st = compute_stats(x)
    print(f"\n--- {name} ---")
    print(f"shape={ds.shape} dtype={ds.dtype} sampled={x.shape}")
    print(f"nan={st.nan} inf={st.inf} finite_frac={st.finite_frac:.6f}")
    if st.min is None:
        print("no finite values in sample")
        return
    print(f"min={st.min:.6g} max={st.max:.6g} mean={st.mean:.6g} std={st.std:.6g}")
    print("quantiles:", " ".join(f"{k:g}:{v:.6g}" for k, v in st.quantiles.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max_tree_lines", type=int, default=500)
    parser.add_argument("--sample_rows", type=int, default=4096)
    args = parser.parse_args()

    args.dataset = str(resolve_artifact_path(args.dataset, "HDF5 trajectory"))
    if not os.path.exists(args.dataset):
        raise FileNotFoundError(args.dataset)

    print(f"Dataset: {args.dataset}  size_bytes={os.path.getsize(args.dataset)}")

    tree_lines: list[str] = []
    action_candidates: list[str] = []
    obs_candidates: list[str] = []

    with h5py.File(args.dataset, "r") as f:
        print("Top-level keys:", list(f.keys()))

        def visit(name: str, obj):
            if isinstance(obj, h5py.Group):
                tree_lines.append(f"GRP  {name} (n={len(obj)})")
            elif isinstance(obj, h5py.Dataset):
                tree_lines.append(f"DSET {name} shape={obj.shape} dtype={obj.dtype}")
                low = name.lower()
                if low.endswith("actions") or "/actions" in low:
                    action_candidates.append(name)
                if "/obs/" in low or low.endswith("/obs") or "observations" in low:
                    obs_candidates.append(name)

        f.visititems(visit)

        print("\n==== HDF5 TREE (truncated) ====")
        for line in tree_lines[: args.max_tree_lines]:
            print(line)
        if len(tree_lines) > args.max_tree_lines:
            print(f"... ({len(tree_lines) - args.max_tree_lines} more lines)")

        print("\n==== Candidate dataset paths (heuristic) ====")
        print("actions:", action_candidates[:50], f"... total={len(action_candidates)}")
        print("obs    :", obs_candidates[:50], f"... total={len(obs_candidates)}")

        # IsaacLab demonstrations are typically under /data/demo_*/...
        if "data" in f:
            demos = sorted(list(f["data"].keys()))
            print("\n/data demos:", demos[:20], f"... total={len(demos)}")
            if demos:
                dk = demos[0]
                g = f["data"][dk]
                print(f"demo={dk} keys:", list(g.keys()))
                if "obs" in g and isinstance(g["obs"], h5py.Group):
                    print(f"demo={dk} obs keys:", list(g["obs"].keys()))

                for path in [
                    f"data/{dk}/actions",
                    f"data/{dk}/obs/state",
                    f"data/{dk}/obs/full_state",
                    f"data/{dk}/states",
                    f"data/{dk}/next_obs",
                ]:
                    if path in f and isinstance(f[path], h5py.Dataset):
                        print_stats(path, f[path], sample_rows=args.sample_rows)

                # length mismatch checks for common pairs
                if f"data/{dk}/actions" in f and f"data/{dk}/obs" in f:
                    aT = f[f"data/{dk}/actions"].shape[0]
                    for ok in ["state", "full_state"]:
                        op = f"data/{dk}/obs/{ok}"
                        if op in f:
                            oT = f[op].shape[0]
                            print(f"\nlen_check {op}: T_obs={oT} vs actions: T_act={aT} (diff={oT - aT})")


if __name__ == "__main__":
    main()

