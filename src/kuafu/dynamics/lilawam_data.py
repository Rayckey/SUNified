"""Episode-safe Kinova RGB/action inputs for the official LiLa-WAM model."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from kuafu.paths import resolve_artifact_path


FORMAT = "kuafu-lilawam-data-v1"
STATE_DIMS = {"drawer": 56, "can": 56, "cabinet_door": 56, "stack3": 74,
              "stack2": 65, "drawer_stack": 65, "hang_cup": 65, "estop_open": 65, "estop_close": 65}
MEAN = np.asarray([.485, .456, .406], np.float32)[:, None, None]
STD = np.asarray([.229, .224, .225], np.float32)[:, None, None]


def file_hash(path):
    """Hash a file without loading a recording into memory."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def encode_actions(values):
    """Encode logged incremental controls, including binary gripper commands."""
    values = np.asarray(values, np.float32)
    return np.sign(values) * np.log1p(np.abs(values))


def decode_actions(values):
    """Invert symlog into the original relative-joint controller command units."""
    values = np.asarray(values, np.float32)
    return np.sign(values) * np.expm1(np.abs(values))


def joint_positions(full_state, task):
    """Extract nine logical joint positions; omit object poses and program stage."""
    values = np.asarray(full_state, np.float32)
    if task not in STATE_DIMS or values.shape[-1] != STATE_DIMS[task]:
        raise ValueError(f"Unaudited full-state layout for {task}: {values.shape}")
    # All nine audited Kinova environments end in q[9], dq[9], effort[9], openness, stage.
    return values[..., -29:-20].copy()


def camera_image(frame, image_size=(512, 144)):
    """Pack both RGB cameras left-to-right, retaining their native aspect ratio."""
    frame = np.asarray(frame)
    if frame.shape != (2, 72, 128, 3) or frame.dtype != np.uint8:
        raise ValueError(f"Expected uint8 two-camera RGB, got {frame.shape}/{frame.dtype}")
    width, height = image_size
    if width % 32 or height % 16 or width / 2 / height != 128 / 72:
        raise ValueError("Packed size must preserve each camera aspect ratio and patch boundaries")
    panels = [np.asarray(Image.fromarray(view).resize((width // 2, height), Image.Resampling.BILINEAR))
              for view in frame]
    image = np.concatenate(panels, axis=1).astype(np.float32).transpose(2, 0, 1) / 255
    return np.ascontiguousarray((image - MEAN) / STD)


def prepare_data(index_path, split_run, output, *, horizon=16, image_size=(512, 144)):
    """Validate recordings and fit normalization exclusively on training episodes."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    camera_image(np.zeros((2, 72, 128, 3), np.uint8), image_size)
    output = resolve_artifact_path(output, "LiLa-WAM prepared data")
    if (output / "manifest.json").exists():
        raise ValueError(f"Prepared manifest already exists: {output}")
    index_path, split_run = Path(index_path).resolve(), Path(split_run).resolve()
    index, previous = [json.loads(path.read_text()) for path in (index_path, split_run)]
    if index.get("format") != "kuafu-rgb-datasets-v1" or index.get("phase") != "complete":
        raise ValueError("Expected a completed synchronized RGB dataset index")
    if previous["dataset"]["identity"]["index_sha256"] != file_hash(index_path):
        raise ValueError("Split provenance does not match the input dataset index")
    if set(index["tasks"]) != set(STATE_DIMS):
        raise ValueError("Expected the nine audited Kinova tasks")
    result = {"format": FORMAT, "source_index": str(index_path), "index_sha256": file_hash(index_path),
              "split_run": str(split_run), "split_run_sha256": file_hash(split_run),
              "horizon": horizon, "image_size": list(image_size), "camera_order": ["table_cam_0", "table_cam_1"],
              "action_key": "actions/relative_joint", "action_encoding": "symlog",
              "action_units": "scaled incremental arm controls and binary gripper commands; not radian displacements",
              "proprioception": "full_state[-29:-20]: seven arm and two logical gripper joint positions",
              "object_pose_input": False, "program_stage_input": False,
              "timing": "RGB[t], joint_position[t], actions[t:t+H], RGB[t+H]; anchors start at t=1",
              "padding": "none; require a real future frame and a full action chunk within the same episode",
              "splits": previous["split"], "tasks": {}}
    bounds = {key: [np.full(9, np.inf), np.full(9, -np.inf)] for key in ("action", "state")}
    raw_action_ranges = {}
    for task, spec in sorted(index["tasks"].items()):
        if file_hash(spec["path"]) != spec["sha256"]:
            raise ValueError(f"{task}: source checksum mismatch")
        groups = {split: set(previous["split"][split][task]) for split in ("train", "validation", "test")}
        if any(groups[a] & groups[b] for a, b in (("train", "test"), ("train", "validation"), ("validation", "test"))):
            raise ValueError(f"{task}: overlapping episode splits")
        episodes = []
        raw_min, raw_max = np.full(9, np.inf), np.full(9, -np.inf)
        with h5py.File(spec["path"], "r") as handle:
            data = handle["data"]
            if set(data) != set.union(*groups.values()) or len(data) != spec["demonstrations"]:
                raise ValueError(f"{task}: split coverage does not match the recording")
            env = json.loads(data.attrs["env_args"])
            if env["rgb_timing"] != "pre_action" or env["camera_keys"] != result["camera_order"]:
                raise ValueError(f"{task}: incompatible image timing or camera order")
            for name in sorted(data):
                demo = data[name]
                frames = demo["obs/rgb"]
                if not demo.attrs.get("success") or not demo.attrs.get("terminal_transition_omitted"):
                    raise ValueError(f"{task}/{name}: expected a successful episode without autoreset rows")
                actions = np.asarray(demo[result["action_key"]], np.float32)
                state = joint_positions(demo["obs/full_state"][:], task)
                if (frames.shape != (len(actions), 2, 72, 128, 3) or frames.dtype != np.uint8
                        or actions.shape != state.shape or actions.shape[1] != 9
                        or not np.isfinite(actions).all() or not np.isfinite(state).all()
                        or np.asarray(demo["dones"]).any()):
                    raise ValueError(f"{task}/{name}: invalid aligned recording")
                anchors = len(actions) - horizon - 1
                if anchors < 1:
                    raise ValueError(f"{task}/{name}: too short for a real {horizon}-step future")
                split = next(key for key, values in groups.items() if name in values)
                episodes.append({"name": name, "frames": len(actions), "anchors": anchors, "split": split})
                if split == "train":
                    selected = actions[1:-1]
                    raw_min = np.minimum(raw_min, selected.min(0)); raw_max = np.maximum(raw_max, selected.max(0))
                    for key, values in (("action", encode_actions(selected)), ("state", state[1:len(actions) - horizon])):
                        bounds[key][0] = np.minimum(bounds[key][0], values.min(0))
                        bounds[key][1] = np.maximum(bounds[key][1], values.max(0))
            result["tasks"][task] = {"source": spec, "control_dt": env["control_dt"], "episodes": episodes,
                "anchors_by_split": {split: sum(e["anchors"] for e in episodes if e["split"] == split) for split in groups}}
        raw_action_ranges[task] = {"min": raw_min.tolist(), "max": raw_max.tolist()}
        print(json.dumps({"validated_task": task, "episodes": len(episodes),
                          "anchors_by_split": result["tasks"][task]["anchors_by_split"]}), flush=True)
    stats = {"robotwin2": {key: {"min": lo.tolist(), "max": hi.tolist()} for key, (lo, hi) in bounds.items()},
             "provenance": {"fit_split": "train", "dataset": "Kinova; robotwin2 is the upstream JSON schema key",
                            "action_encoding": "symlog", "index_sha256": result["index_sha256"],
                            "raw_action_ranges_by_task": raw_action_ranges}}
    write_json(output / "normalization.json", stats)
    result["normalization"] = str(output / "normalization.json")
    result["normalization_sha256"] = file_hash(output / "normalization.json")
    write_json(output / "manifest.json", result)
    return result


class KinovaLiLaDataset:
    """Lazy, read-only HDF5 samples; never pad actions or cross episode boundaries."""

    def __init__(self, manifest, split="train", task_cond_dir=None):
        self.manifest_path = Path(manifest).resolve()
        self.metadata = json.loads(self.manifest_path.read_text())
        if self.metadata["format"] != FORMAT or split not in ("train", "validation", "test"):
            raise ValueError("Invalid LiLa-WAM manifest or split")
        self.split = split
        self.tasks = sorted(self.metadata["tasks"])
        self.episodes, self.task_ranges, self.ends = [], {}, []
        self.conditions, self._handles, self._pid = {}, {}, os.getpid()
        offset = 0
        for task in self.tasks:
            start = offset
            for episode in self.metadata["tasks"][task]["episodes"]:
                if episode["split"] != split:
                    continue
                self.episodes.append({**episode, "task": task, "start": offset})
                offset += episode["anchors"]
                self.ends.append(offset)
            self.task_ranges[task] = (start, offset)
            if offset == start:
                raise ValueError(f"No samples for {task}/{split}")
            if task_cond_dir is not None:
                vector = np.load(Path(task_cond_dir) / task / "task_cond.npy", allow_pickle=False).astype(np.float32)
                if vector.ndim != 1 or not np.isfinite(vector).all():
                    raise ValueError(f"Invalid task condition for {task}")
                self.conditions[task] = vector
        self.ends = np.asarray(self.ends, np.int64)

    def __len__(self):
        return int(self.ends[-1])

    def __getstate__(self):
        return {**self.__dict__, "_handles": {}, "_pid": None}

    def close(self):
        for handle in self._handles.values():
            handle.close()
        self._handles = {}

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode = self.episodes[int(np.searchsorted(self.ends, index, side="right"))]
        task, anchor = episode["task"], 1 + int(index) - episode["start"]
        if self._pid != os.getpid():
            self.close(); self._pid = os.getpid()
        if task not in self._handles:
            self._handles[task] = h5py.File(self.metadata["tasks"][task]["source"]["path"], "r")
        demo = self._handles[task]["data"][episode["name"]]
        horizon = self.metadata["horizon"]
        result = {"state": joint_positions(demo["obs/full_state"][anchor], task)[None],
                  "action_sequence": encode_actions(demo[self.metadata["action_key"]][anchor:anchor + horizon]),
                  "previous_action": encode_actions(demo[self.metadata["action_key"]][anchor - 1]),
                  "pixel_values": camera_image(demo["obs/rgb"][anchor], self.metadata["image_size"]),
                  "future_pixel_values": camera_image(demo["obs/rgb"][anchor + horizon], self.metadata["image_size"]),
                  "task_id": self.tasks.index(task), "episode": episode["name"], "anchor": anchor}
        if self.conditions:
            result["task_cond"] = self.conditions[task]
        return result
