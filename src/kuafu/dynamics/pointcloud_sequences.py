"""Task-balanced point-cloud sequences with train-only action preprocessing."""

import json
from pathlib import Path

import h5py
import numpy as np

from kuafu.dynamics.rgb_sequences import RGBSequenceDataset, file_hash
from kuafu.dynamics.sequences import symlog


def prepare_pointcloud_cache(index_path, cache, *, num_points=512):
    """Cache actual depth-derived XYZ; use no simulator state or RGB features."""
    index_path, cache = Path(index_path).resolve(), Path(cache).resolve()
    index = json.loads(index_path.read_text())
    if index.get("format") != "kuafu-pointcloud-datasets-v1" or index.get("phase") != "complete":
        raise ValueError("Expected a completed kuafu-pointcloud-datasets-v1 index")
    if num_points < 32:
        raise ValueError("num_points must be at least 32")
    identity = {"index_sha256": file_hash(index_path), "num_points": num_points}
    if (cache / "manifest.json").exists():
        metadata = json.loads((cache / "manifest.json").read_text())
        if metadata["identity"] != identity:
            raise ValueError("Point-cloud cache identity mismatch")
        return metadata
    cache.mkdir(parents=True, exist_ok=True)
    metadata = {"identity": identity, "source_index": str(index_path), "tasks": {},
                "num_points": num_points, "action_dim": 9, "inputs": ["points", "previous_action", "task"],
                "coordinate_frame": "environment-local world axes; meters",
                "point_center": [0.65, 0.0, 0.755], "point_scale": [0.85, 0.9, 0.745],
                "point_transform": "fixed workspace center/half-extent, no per-frame centering",
                "terminal_transitions": "omitted; no reset observations"}
    center, scale = np.array(metadata["point_center"]), np.array(metadata["point_scale"])
    for task, spec in sorted(index["tasks"].items()):
        if not task.replace("_", "").isalnum():
            raise ValueError("Invalid task identifier")
        source_path = Path(spec["path"]).resolve()
        if file_hash(source_path) != spec["sha256"]:
            raise ValueError(f"{task}: source checksum mismatch")
        directory = cache / task
        directory.mkdir(exist_ok=True)
        episodes = []
        with h5py.File(source_path) as source:
            demos = source["data"]
            env_args = json.loads(demos.attrs["env_args"])
            if env_args.get("pointcloud_timing") != "pre_action":
                raise ValueError(f"{task}: point-cloud timing is not audited")
            if len(demos) != spec["demonstrations"] or len(demos) < 5:
                raise ValueError("Need at least five correctly indexed demonstrations per task")
            for number, (name, demo) in enumerate(sorted(demos.items())):
                points = np.asarray(demo["obs/pointcloud"], np.float32)
                actions = np.asarray(demo["actions/relative_joint"], np.float32)
                rewards = np.asarray(demo["rewards"], np.float32).reshape(-1)
                if (not demo.attrs.get("success", False) or not demo.attrs.get("terminal_transition_omitted", False)
                        or points.ndim != 3 or points.shape[-1] != 3 or points.shape[1] < num_points
                        or actions.shape != (len(points), 9) or len(rewards) != len(points) or len(points) < 4
                        or not all(np.isfinite(x).all() for x in (points, actions, rewards))
                        or np.asarray(demo["dones"]).any()):
                    raise ValueError(f"{task}/{name}: invalid or misaligned point-cloud data")
                # Recorder order is uniformly randomized; first N is a uniform subset.
                points = ((points[1:, :num_points] - center) / scale).astype(np.float32)
                if np.max(np.abs(points)) > 1.001:
                    raise ValueError(f"{task}/{name}: point cloud outside the recorded workspace")
                point_path = directory / f"{number:04d}.npy"
                np.save(point_path, points)
                values_path = directory / f"{number:04d}.npz"
                np.savez(values_path, previous_action=actions[:-1], reward=rewards[:-1])
                episodes.append({"name": name, "points": str(point_path.relative_to(cache)),
                                 "values": str(values_path.relative_to(cache)), "frames": len(points)})
            metadata["tasks"][task] = {"source": spec, "env_args": env_args, "episodes": episodes}
        print(json.dumps({"cached_pointcloud_task": task, "episodes": len(episodes),
                          "frames": sum(e["frames"] for e in episodes)}), flush=True)
    (cache / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


class PointCloudSequenceDataset:
    """Sample whole-episode splits and equally many windows from every task."""

    sample = RGBSequenceDataset.sample

    def __init__(self, cache, seed=42):
        self.cache = Path(cache).resolve()
        self.metadata = json.loads((self.cache / "manifest.json").read_text())
        self.tasks = sorted(self.metadata["tasks"])
        self.episodes = {}
        self.split = {key: {} for key in ("train", "validation", "test")}
        self.action_scale = np.ones(9, np.float32)
        for task_id, task in enumerate(self.tasks):
            episodes = []
            for spec in self.metadata["tasks"][task]["episodes"]:
                with np.load(self.cache / spec["values"], allow_pickle=False) as values:
                    episodes.append({"name": spec["name"], "points": np.load(self.cache / spec["points"], mmap_mode="r"),
                                     "previous_action": values["previous_action"], "reward": values["reward"]})
            self.episodes[task] = episodes
            order = np.random.default_rng(seed + task_id).permutation(len(episodes))
            heldout = max(1, int(round(0.1 * len(episodes))))
            for split, ids in (("test", order[:heldout]), ("validation", order[heldout:2 * heldout]),
                               ("train", order[2 * heldout:])):
                self.split[split][task] = [episodes[int(i)]["name"] for i in ids]
            train_names = set(self.split["train"][task])
            for episode in episodes:
                if episode["name"] in train_names:
                    self.action_scale = np.maximum(self.action_scale, np.abs(symlog(episode["previous_action"])).max(0))
        self.metadata = {**self.metadata, "action_scale": self.action_scale.tolist()}

    def sample_task(self, task, split, batch_size, length, rng, *, pad=True):
        names = set(self.split[split][task])
        choices = [e for e in self.episodes[task] if e["name"] in names and (pad or len(e["points"]) >= length)]
        if not choices:
            raise ValueError(f"{task}/{split}: no episodes long enough for {length} frames")
        counts = np.array([max(1, len(e["points"]) - length + 1) for e in choices])
        batch = {"points": np.zeros((batch_size, length, self.metadata["num_points"], 3), np.float32),
                 "prev_action": np.zeros((batch_size, length, 9), np.float32),
                 "reward": np.zeros((batch_size, length), np.float32),
                 "reward_mask": np.zeros((batch_size, length), np.float32),
                 "time_mask": np.zeros((batch_size, length), np.float32),
                 "is_first": np.zeros((batch_size, length), bool),
                 "task": np.zeros((batch_size, length, len(self.tasks)), np.float32)}
        for i in range(batch_size):
            selected = int(rng.choice(len(choices), p=counts / counts.sum()))
            episode = choices[selected]
            start = int(rng.integers(counts[selected]))
            stop = min(start + length, len(episode["points"]))
            n = stop - start
            batch["points"][i, :n] = episode["points"][start:stop]
            batch["prev_action"][i, :n] = np.clip(symlog(episode["previous_action"][start:stop]) / self.action_scale, -1, 1)
            batch["prev_action"][i, 0] = 0
            batch["reward"][i, :n] = episode["reward"][start:stop]
            batch["reward_mask"][i, 1:n] = 1
            batch["time_mask"][i, :n] = 1
        batch["is_first"][:, 0] = True
        batch["task"][..., self.tasks.index(task)] = 1
        return batch
