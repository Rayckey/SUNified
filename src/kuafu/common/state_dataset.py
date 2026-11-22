"""
sim_io.py – minimal I/O layer for streaming PyTorch tensors to disk (HDF5)
and reading them back for training.

Dependencies:
  torch, h5py, numpy
"""

import threading, queue, atexit
from pathlib import Path
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader

from kuafu.paths import external_artifact_root, resolve_artifact_path

# ---------------------------------------------------------------------------
# 1.  Dataset: lazy, thread-safe read access
# ---------------------------------------------------------------------------


class SimDataset(Dataset):
    """
    Random-access reader for a file written by `H5AsyncWriter`.

    Each __getitem__ returns (state, control) as float32 torch tensors.
    """

    _DATASETS = ("pose", "control")  # change here if you store more

    def __init__(self, path: str | Path):
        self.path = str(resolve_artifact_path(path, "State trajectory dataset"))
        with h5py.File(self.path, "r") as f:
            self.length = f[self._DATASETS[0]].shape[0]
        self._h5 = None  # one handle per DataLoader worker

    # Lazily open one HDF5 handle per worker process ------------------------
    def _file(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r")
        return self._h5

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        f = self._file()
        tensors = [torch.from_numpy(f[k][idx]) for k in self._DATASETS]
        return tuple(tensors)  # (pose, control)


# ---------------------------------------------------------------------------
# 2.  Writer: non-blocking enqueue + background thread
# ---------------------------------------------------------------------------


class SimAsyncWriter:
    """
    Create once at sim start, call .append(pose, control) in the main loop.

    • Automatically creates/extends datasets with gzip compression.
    • Uses a background thread so the simulation never waits for disk.
    • Handles CUDA→CPU, contiguous, and dtype conversion internally.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        chunk_rows: int = 1024,
        compression: str = "gzip",
        compression_lvl: int = 4,
        queue_size: int = 4096,
    ):
        self.path = str(resolve_artifact_path(path, "State trajectory output"))
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)

        # HDF5 file & extendable datasets -----------------------------------
        # STATE IS 13 NUMBER
        # XYZ POSITION (3)
        # ORIENTATION (4)
        # XYZ VELOCITY (3)
        # ORIENTATION VELOCITY (3)
        self._h5 = h5py.File(self.path, "w")
        self._pose_ds = self._h5.create_dataset(
            "pose",
            shape=(0, 13),
            maxshape=(None, 13),
            dtype="float32",
            chunks=(chunk_rows, 13),
            compression=compression,
            compression_opts=compression_lvl,
        )

        # CONTROL IS 2 NUMBER
        # LEFT WHEEL VELOCITY (1)
        # RIGHT WHEEL VELOCITY (1)
        self._control_ds = self._h5.create_dataset(
            "control",
            shape=(0, 2),
            maxshape=(None, 2),
            dtype="float32",
            chunks=(chunk_rows, 2),
            compression=compression,
            compression_opts=compression_lvl,
        )

        # RESIDUAL is 9 number, LARGER IS WORSE
        # RESIDUAL TO GOAL (1)
        # RESIDUAL TO OBSTACLE (8)
        self._residual_ds = self._h5.create_dataset(
            "residual",
            shape=(0, 9),
            maxshape=(None, 9),
            dtype="float32",
            chunks=(chunk_rows, 9),
            compression=compression,
            compression_opts=compression_lvl,
        )

        # CURRENT DISTANCE is 9 number
        # DISTANCE TO GOAL (1)
        # DISTANCE TO OBSTACLE (8)
        self._distance_ds = self._h5.create_dataset(
            "distance",
            shape=(0, 9),
            maxshape=(None, 9),
            dtype="float32",
            chunks=(chunk_rows, 9),
            compression=compression,
            compression_opts=compression_lvl,
        )

        # TARGET POSE IS 6 NUMBER
        # XYZ POSITION (3)
        # VECTOR TO TARGET (3)
        self._target_ds = self._h5.create_dataset(
            "target",
            shape=(0, 6),
            maxshape=(None, 6),
            dtype="float32",
            chunks=(chunk_rows, 6),
            compression=compression,
            compression_opts=compression_lvl,
        )

        # EOF REASON IS 1 NUMBER
        # 0 = NORMAL/TIMEOUT, 1 = SUCCESS, 2 = COLLISION
        self._eof_ds = self._h5.create_dataset(
            "eof",
            shape=(0, 1),
            maxshape=(None, 1),
            dtype="int8",
            chunks=(chunk_rows, 1),
            compression=compression,
            compression_opts=compression_lvl,
        )


        # Queue + writer thread ---------------------------------------------
        self._q = queue.Queue(maxsize=queue_size)
        self._th = threading.Thread(target=self._writer, daemon=True)
        self._th.start()
        atexit.register(self.close)  # ensure cleanup on Ctrl-C

    # Public API ------------------------------------------------------------
    def append(
        self,
        pose_t: torch.Tensor,
        control_t: torch.Tensor,
        residual: np.ndarray,
        distance: np.ndarray,
        target: np.ndarray,
        eof: np.ndarray,
        #    , rgb_cpu: torch.Tensor, depth_cpu: torch.Tensor):
    ):
        """
        Enqueue a (B,13) pose tensor and (B,2) control tensor.

        • Accepts GPU or CPU tensors, any dtype.
        • Non-blocking; back-pressure only if internal queue is full.
        """
        pose_cpu = (
            pose_t.detach()
            .to(dtype=torch.float32, device="cpu", non_blocking=True)
            .contiguous()
        )
        control_cpu = (
            control_t.detach()
            .to(dtype=torch.float32, device="cpu", non_blocking=True)
            .contiguous()
        )
        self._q.put(
            (pose_cpu.numpy(), control_cpu.numpy(), residual, distance, target, eof), block=True
        )

    def close(self):
        """Flush queue, stop thread, close file (safe to call multiple times)."""
        if getattr(self, "_closed", False):
            return
        self._q.put(None)  # sentinel → stop
        self._th.join()
        self._h5.close()
        self._closed = True

    # ----------------------------------------------------------------------
    # Internal background writer
    # ----------------------------------------------------------------------
    def _writer(self):
        while True:
            item = self._q.get()
            if item is None:  # shutdown
                break
            pose_np, control_np, residual_np, distance_np, target_np, eof_np = item
            n_old = self._pose_ds.shape[0]
            n_new = n_old + len(pose_np)
            self._pose_ds.resize(n_new, axis=0)
            self._control_ds.resize(n_new, axis=0)
            self._residual_ds.resize(n_new, axis=0)
            self._distance_ds.resize(n_new, axis=0)
            self._target_ds.resize(n_new, axis=0)
            self._eof_ds.resize(n_new, axis=0)

            self._pose_ds[n_old:n_new] = pose_np
            self._control_ds[n_old:n_new] = control_np
            self._residual_ds[n_old:n_new] = residual_np
            self._distance_ds[n_old:n_new] = distance_np
            self._target_ds[n_old:n_new] = target_np
            self._eof_ds[n_old:n_new] = eof_np

            self._q.task_done()


# ---------------------------------------------------------------------------
# 3.  Convenience: build a DataLoader in one call
# ---------------------------------------------------------------------------


def make_dataloader(
    path: str | Path,
    *,
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    ds = SimDataset(path)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


if __name__ == "__main__":
    dl = make_dataloader(external_artifact_root() / "debug" / "state_dataset" / "state.h5", batch_size=512)

    for epoch in range(3):
        for pose, control in dl:
            print(pose.shape, control.shape)
