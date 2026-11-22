"""Task-balanced Kinova sequences with a shared, explicitly masked state layout."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from kuafu.dynamics.sequences import SequenceDataset, symlog


# Object order matches each task's _get_observations(), before fingertip/link4.
TASK_OBJECTS = {
    "drawer": ("top_drawer_handle",),
    "drawer_stack": ("top_drawer_handle", "strawberry_center"),
    "cabinet_door": ("door_knob",),
    "stack2": ("cube1", "cube2"),
    "stack3": ("cube1", "cube2", "cube3"),
    "hang_cup": ("mug", "hanger"),
    "can": ("can",),
    "estop_open": ("estop_cap_tip", "estop_base"),
    "estop_close": ("estop_cap_tip", "estop_cap_center"),
}
STATE_DIM = 73
ROBOT_DIM = 46


def state_indices(task):
    """Map native state columns (without SUN stage) into the shared 73-vector.

    Shared order: fingertip pose (xyz, rotation-6D), link4 pose, joint position,
    velocity, normalized effort/contact, gripper openness, then three object
    poses. Coordinates and units are preserved; no frame conversion is applied.
    """
    if task not in TASK_OBJECTS:
        raise ValueError(f"No audited Kinova state layout for {task!r}")
    objects = len(TASK_OBJECTS[task])
    pose_starts = [ROBOT_DIM + 9 * i for i in range(objects)] + [0, 9]
    columns = [start + i for start in pose_starts for i in range(3)]
    columns += [start + i for start in pose_starts for i in range(3, 9)]
    columns += list(range(18, ROBOT_DIM))
    return np.asarray(columns, np.int32)


def canonical_state(values, indices):
    """Align native observations, zero-filling only absent object slots."""
    values = np.asarray(values, np.float32)
    if values.shape[-1] != len(indices):
        raise ValueError(f"Expected {len(indices)} state columns, got {values.shape[-1]}")
    result = np.zeros((*values.shape[:-1], STATE_DIM), np.float32)
    result[..., indices] = values
    return result


class MultiTaskSequenceDataset:
    """Split demonstrations within each task and sample tasks with equal weight."""

    def __init__(self, index_path, *, seed=42, skip_first=1):
        path = Path(index_path).expanduser().resolve()
        index = json.loads(path.read_text())
        if index.get("phase") != "complete" or not index.get("tasks"):
            raise ValueError("Use a completed dataset index containing task datasets")
        self.tasks = sorted(index["tasks"])
        self.datasets, self.indices, self.masks = {}, {}, {}
        self.state_mean, self.state_scale = {}, {}
        self.split = {key: {} for key in ("train", "validation", "test")}
        for task in self.tasks:
            row = index["tasks"][task]
            dataset_path = Path(row["dataset"])
            if not dataset_path.is_absolute():
                dataset_path = path.parent / dataset_path
            data = SequenceDataset(dataset_path, seed=seed, skip_first=skip_first)
            if data.metadata["sha256"] != row["sha256"]:
                raise ValueError(f"Dataset hash changed for {task}")
            columns = state_indices(task)
            if data.metadata["state_dim"] != len(columns) or data.metadata["action_dim"] != 9:
                raise ValueError(f"Unexpected state/action layout for {task}")
            if data.metadata["env_args"].get("env_name") != row["task"]:
                raise ValueError(f"Task identity mismatch for {task}")
            self.datasets[task], self.indices[task] = data, columns
            training_states = np.concatenate([episode.state for episode in data.for_split("train")])
            self.state_mean[task] = training_states.mean(0, dtype=np.float64).astype(np.float32)
            self.state_scale[task] = data.state_std.copy()
            mask = np.zeros(STATE_DIM, np.float32)
            mask[columns] = 1
            self.masks[task] = mask
            for split in self.split:
                self.split[split][task] = data.split[split]
        # All tasks use identical joint-command units. Fit one scale on training
        # episodes only so the same command has the same representation everywhere.
        self.action_scale = np.max([data.action_scale for data in self.datasets.values()], axis=0)
        self.metadata = {
            "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "layout": "kinova_robot46_objects3x9_v1", "state_dim": STATE_DIM,
            "action_dim": 9, "task_dim": len(self.tasks), "tasks": self.tasks,
            "sampling": "equal task counts per batch; uniform windows within each task",
            "padding": "short episodes padded at the end; all padded timestep losses masked",
            "state_transform": "per-task training mean/std (minimum std 1e-3), then upstream signed log",
            "action_scale": self.action_scale.tolist(),
            "source_transitions": sum(d.metadata["source_transitions"] for d in self.datasets.values()),
            "usable_transitions": sum(d.metadata["usable_transitions"] for d in self.datasets.values()),
            "demonstrations": sum(d.metadata["demonstrations"] for d in self.datasets.values()),
            "per_task": {
                task: {**data.metadata, "objects": list(TASK_OBJECTS[task]),
                       "shared_indices": self.indices[task].tolist(),
                       "state_mask": self.masks[task].tolist(),
                       "state_mean": self.state_mean[task].tolist(),
                       "state_scale": self.state_scale[task].tolist()}
                for task, data in self.datasets.items()
            },
        }
        self._windows = {}

    def _sample_task(self, task, split, count, length, rng, *, pad):
        key = (task, split, length, pad)
        if key not in self._windows:
            episodes = self.datasets[task].for_split(split)
            counts = np.asarray([max(1 if pad else 0, len(e.state) - length + 1) for e in episodes])
            self._windows[key] = episodes, counts.cumsum()
        episodes, cumulative = self._windows[key]
        if not len(cumulative) or not cumulative[-1]:
            raise ValueError(f"No {task}/{split} sequences long enough for length {length}")
        draws = rng.integers(cumulative[-1], size=count)
        selected = np.searchsorted(cumulative, draws, side="right")
        starts = draws - np.r_[0, cumulative[:-1]][selected]
        rows = []
        task_id = self.tasks.index(task)
        for episode_id, start in zip(selected, starts):
            episode = episodes[episode_id]
            valid = min(length, len(episode.state) - start)
            row = {}
            for field in ("state", "prev_action", "reward", "reward_mask"):
                value = getattr(episode, field)[start:start + valid]
                if field == "state":
                    value = canonical_state((value - self.state_mean[task]) / self.state_scale[task],
                                            self.indices[task])
                padded = np.zeros((length, *value.shape[1:]), value.dtype)
                padded[:valid] = value
                row[field] = padded
            row["prev_action"] = np.clip(symlog(row["prev_action"]) / self.action_scale, -1, 1)
            row["prev_action"][0] = 0
            row["reward_mask"][0] = 0
            row["time_mask"] = (np.arange(length) < valid).astype(np.float32)
            row["state_mask"] = np.broadcast_to(self.masks[task], (length, STATE_DIM)).copy()
            row["task"] = np.broadcast_to(np.eye(len(self.tasks), dtype=np.float32)[task_id],
                                           (length, len(self.tasks))).copy()
            row["is_first"] = np.arange(length) == 0
            rows.append(row)
        return {key: np.stack([row[key] for row in rows]) for key in rows[0]}

    def sample_task(self, task, split, batch_size, length, rng):
        """Sample unpadded evaluation windows for one task."""
        if task not in self.datasets or split not in self.split or batch_size < 1 or length < 2:
            raise ValueError("Invalid task, split, batch size, or sequence length")
        return self._sample_task(task, split, batch_size, length, rng, pad=False)

    def sample(self, split, batch_size, length, rng):
        """Sample balanced tasks without crossing episode or task boundaries."""
        if split not in self.split or length < 2 or batch_size < len(self.tasks):
            raise ValueError("Sequence length must be >=2 and batch size >= task count")
        order = rng.permutation(len(self.tasks))
        counts = np.full(len(self.tasks), batch_size // len(self.tasks))
        counts[order[:batch_size % len(self.tasks)]] += 1
        batches = [self._sample_task(task, split, int(counts[i]), length, rng, pad=True)
                   for i, task in enumerate(self.tasks)]
        permutation = rng.permutation(batch_size)
        return {key: np.concatenate([batch[key] for batch in batches])[permutation]
                for key in batches[0]}
