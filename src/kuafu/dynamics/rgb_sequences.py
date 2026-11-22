"""Episode-safe, task-balanced RGB sequences with disk-backed image storage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from kuafu.dynamics.sequences import symlog


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resize_rgb(frames, height=64, width=112):
    """Resize each camera separately, then concatenate their RGB channels."""
    from PIL import Image

    frames = np.asarray(frames)
    if frames.dtype != np.uint8 or frames.ndim != 5 or frames.shape[-1] != 3:
        raise ValueError("Expected uint8 [time,camera,height,width,3] RGB")
    result = np.empty((len(frames), height, width, 3 * frames.shape[1]), np.uint8)
    for step in range(len(frames)):
        for camera in range(frames.shape[1]):
            result[step, ..., 3 * camera:3 * camera + 3] = np.asarray(
                Image.fromarray(frames[step, camera]).resize((width, height), Image.Resampling.BILINEAR))
    return result


def prepare_rgb_cache(index_path, cache, *, height=64, width=112):
    """Verify immutable successful demonstrations and cache only RGB/actions/rewards."""
    index_path, cache = Path(index_path).resolve(), Path(cache).resolve()
    index = json.loads(index_path.read_text())
    if index.get("format") != "kuafu-rgb-datasets-v1" or index.get("phase") != "complete":
        raise ValueError("Expected a completed kuafu-rgb-datasets-v1 index")
    if min(height, width) < 48 or height % 16 or width % 16:
        raise ValueError("Dreamer image dimensions must be multiples of 16 and at least 48")
    identity = {"index_sha256": file_hash(index_path), "height": height, "width": width}
    if (cache / "manifest.json").exists():
        metadata = json.loads((cache / "manifest.json").read_text())
        if metadata["identity"] != identity:
            raise ValueError("RGB cache identity mismatch")
        return metadata
    cache.mkdir(parents=True, exist_ok=True)
    metadata = {"identity": identity, "source_index": str(index_path), "tasks": {},
                "image_shape": [height, width, 6], "action_dim": 9,
                "inputs": ["rgb", "previous_action", "task"],
                "action_alignment": "previous_action[t] = applied action preceding RGB[t]",
                "terminal_transitions": "omitted; no reset images or invented success targets"}
    for task, spec in sorted(index["tasks"].items()):
        if not task.replace("_", "").isalnum():
            raise ValueError(f"Invalid task identifier: {task}")
        source_path = Path(spec["path"]).resolve()
        if file_hash(source_path) != spec["sha256"]:
            raise ValueError(f"{task}: source checksum mismatch")
        directory = cache / task
        directory.mkdir(exist_ok=True)
        episodes = []
        with h5py.File(source_path) as source:
            demos = source["data"]
            env_args = json.loads(demos.attrs["env_args"])
            if env_args.get("rgb_timing") != "pre_action":
                raise ValueError(f"{task}: RGB timing is not audited")
            if len(demos) != spec["demonstrations"] or len(demos) < 5:
                raise ValueError(f"{task}: need at least five correctly indexed demonstrations")
            for number, (name, demo) in enumerate(sorted(demos.items())):
                if not demo.attrs.get("success", False) or not demo.attrs.get("terminal_transition_omitted", False):
                    raise ValueError(f"{task}/{name}: expected a successful episode without autoreset rows")
                rgb = demo["obs/rgb"]
                actions = np.asarray(demo["actions/relative_joint"], np.float32)
                rewards = np.asarray(demo["rewards"], np.float32).reshape(-1)
                if (rgb.dtype != np.uint8 or rgb.ndim != 5 or rgb.shape[1] != 2 or rgb.shape[-1] != 3
                        or actions.shape != (len(rgb), 9) or len(rewards) != len(rgb) or len(rgb) < 4
                        or not np.isfinite(actions).all() or not np.isfinite(rewards).all()
                        or np.asarray(demo["dones"]).any()):
                    raise ValueError(f"{task}/{name}: invalid or misaligned RGB/action sequence")
                # Skip the first post-reset image; omit the final unpaired action.
                count = len(rgb) - 1
                image_path = directory / f"{number:04d}.npy"
                images = np.lib.format.open_memmap(image_path, mode="w+", dtype=np.uint8,
                                                   shape=(count, height, width, 6))
                for start in range(0, count, 128):
                    images[start:start + 128] = resize_rgb(rgb[1 + start:1 + min(start + 128, count)], height, width)
                images.flush()
                del images
                values_path = directory / f"{number:04d}.npz"
                np.savez(values_path, previous_action=actions[:-1], reward=rewards[:-1])
                episodes.append({"name": name, "rgb": str(image_path.relative_to(cache)),
                                 "values": str(values_path.relative_to(cache)), "frames": count})
            metadata["tasks"][task] = {"source": spec, "env_args": env_args, "episodes": episodes}
        print(json.dumps({"cached_rgb_task": task, "episodes": len(episodes),
                          "frames": sum(e["frames"] for e in episodes)}), flush=True)
    (cache / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


class RGBSequenceDataset:
    """Only camera images enter the encoder; task identity conditions dynamics."""

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
                    episodes.append({"name": spec["name"], "rgb": np.load(self.cache / spec["rgb"], mmap_mode="r"),
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
        choices = [e for e in self.episodes[task] if e["name"] in names and (pad or len(e["rgb"]) >= length)]
        if not choices:
            raise ValueError(f"{task}/{split}: no episodes long enough for {length} frames")
        counts = np.array([max(1, len(e["rgb"]) - length + 1) for e in choices])
        batch = {"rgb": np.zeros((batch_size, length, *self.metadata["image_shape"]), np.uint8),
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
            stop = min(start + length, len(episode["rgb"]))
            n = stop - start
            batch["rgb"][i, :n] = episode["rgb"][start:stop]
            batch["prev_action"][i, :n] = np.clip(symlog(episode["previous_action"][start:stop]) / self.action_scale, -1, 1)
            batch["prev_action"][i, 0] = 0
            batch["reward"][i, :n] = episode["reward"][start:stop]
            batch["reward_mask"][i, 1:n] = 1
            batch["time_mask"][i, :n] = 1
        batch["is_first"][:, 0] = True
        batch["task"][..., self.tasks.index(task)] = 1
        return batch

    def sample(self, split, batch_size, length, rng):
        if batch_size < len(self.tasks) or batch_size % len(self.tasks):
            raise ValueError("batch_size must be a positive multiple of the task count")
        batches = [self.sample_task(task, split, batch_size // len(self.tasks), length, rng) for task in self.tasks]
        return {key: np.concatenate([batch[key] for batch in batches]) for key in batches[0]}
