"""Trajectory partitions and current-decision labels for Kuafu2Real windows."""

import copy
import json
from pathlib import Path

import numpy as np
import torch
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from kuafu2real.action_stride_dataset import ActionStrideSequenceSampler

from .artifacts import json_digest, zarr_digest
from .object_frame import OBJECT_RELATIVE, ObjectFrame, validate_relative_contract


class StageDataset(BaseDataset):
    def __init__(self, zarr_path, *, horizon, n_obs_steps, action_stride=1, pad_before=None,
                 pad_after=None, partition="train", verify_content=False):
        self.path = Path(zarr_path).expanduser().resolve()
        self.manifest = json.loads((self.path / "stage_dataset.json").read_text())
        content = dict(self.manifest)
        identity = content.pop("dataset_id")
        if identity != json_digest(content):
            raise ValueError("Dataset manifest identity has changed")
        if verify_content and content["content_sha256"] != zarr_digest(self.path):
            raise ValueError("Dataset arrays differ from the recorded content hash")
        keys = ["state", "action", "point_cloud", "stage_id", "decision_valid", "source_row"]
        relative = content["export"]["action_format"] == OBJECT_RELATIVE
        if relative:
            validate_relative_contract(content["export"])
            keys.append("reference_pose")
        self.replay_buffer = ReplayBuffer.copy_from_path(str(self.path), keys=keys)
        self.ends = self.replay_buffer.episode_ends[:]
        if len(self.ends) < 3 or np.any(np.diff(np.r_[0, self.ends]) <= 0):
            raise ValueError("Expected three or more nonempty trajectories")
        if relative:
            references = self.replay_buffer["reference_pose"]
            if references.shape != (int(self.ends[-1]), 7):
                raise ValueError("Expected one xyz_wxyz reference pose per decision")
            for start, end in zip(np.r_[0, self.ends[:-1]], self.ends):
                ObjectFrame(references[start]).check_stationary(references[start:end])
        split = content["splits"]
        ids = [index for part in ("train", "validation", "test") for index in split[part]]
        if sorted(ids) != list(range(len(self.ends))) or any(not split[part] for part in split):
            raise ValueError("Trajectory partitions must be nonempty, disjoint, and exhaustive")
        self.horizon, self.n_obs_steps, self.action_stride = horizon, n_obs_steps, action_stride
        self.pad_before = n_obs_steps - 1 if pad_before is None else pad_before
        # Keep late decisions even when their remaining action horizon is short.
        # __getitem__ masks padded targets, including with strided actions.
        self.pad_after = horizon - n_obs_steps if pad_after is None else pad_after
        self.num_stages = len(content["stage_names"])
        valid = self.replay_buffer["decision_valid"]
        stages = self.replay_buffer["stage_id"]
        if valid.dtype != bool or stages.dtype.kind not in "iu":
            raise ValueError("Stage arrays must have boolean/integer types")
        if np.any((stages[valid] < 0) | (stages[valid] >= self.num_stages)) or np.any(stages[~valid] != -1):
            raise ValueError("Stage arrays violate the declared vocabulary")
        self._set_partition(partition)

    def _episode_mask(self, partition):
        mask = np.zeros(len(self.ends), dtype=bool)
        mask[self.manifest["splits"][partition]] = True
        return mask

    def _set_partition(self, partition):
        if partition not in {"train", "validation", "test"}:
            raise ValueError("partition must be train, validation, or test")
        self.partition = partition
        self.sampler = ActionStrideSequenceSampler(
            self.replay_buffer, horizon=self.horizon, n_obs_steps=self.n_obs_steps,
            action_stride=self.action_stride, pad_before=self.pad_before, pad_after=self.pad_after,
            episode_mask=self._episode_mask(partition),
        )
        indices = self.sampler.indices
        self.sampler.indices = indices[self.replay_buffer["decision_valid"][indices[:, 2]]]
        if len(self.sampler) == 0:
            raise ValueError(f"No usable decision windows in {partition}; check horizons and padding")

    def for_partition(self, partition):
        result = copy.copy(self)
        result._set_partition(partition)
        return result

    def get_validation_dataset(self):
        return self.for_partition("validation")

    def get_normalizer(self, **kwargs):
        # Fit once on valid training rows; held-out extrema must not influence scaling.
        row_mask = np.repeat(self._episode_mask("train"), np.diff(np.r_[0, self.ends]))
        row_mask &= self.replay_buffer["decision_valid"]
        if not row_mask.any():
            raise ValueError("No valid training rows for normalization")
        normalizer = LinearNormalizer()
        normalizer.fit(data={
            "action": self.replay_buffer["action"][row_mask],
            "agent_pos": self.replay_buffer["state"][row_mask],
            "point_cloud": self.replay_buffer["point_cloud"][row_mask, ..., :3],
        }, last_n_dims=1, mode="limits", **kwargs)
        return normalizer

    def class_counts(self):
        labels = self.replay_buffer["stage_id"][self.sampler.indices[:, 2]]
        return np.bincount(labels, minlength=self.num_stages).tolist()

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, index):
        start, end, current = self.sampler.indices[index]
        sample = self.sampler.sample_sequence(index)
        offsets = np.arange(self.horizon) - (self.n_obs_steps - 1)
        offsets[offsets > 0] *= self.action_stride
        rows = current + offsets
        valid = (rows >= start) & (rows < end)
        valid &= self.replay_buffer["decision_valid"][rows.clip(start, end - 1)]
        return {
            "obs": {"point_cloud": torch.from_numpy(sample["point_cloud"][..., :3].astype(np.float32)),
                    "agent_pos": torch.from_numpy(sample["state"].astype(np.float32))},
            "action": torch.from_numpy(sample["action"].astype(np.float32)),
            "action_valid": torch.from_numpy(valid),
            "stage_id": torch.tensor(int(self.replay_buffer["stage_id"][current]), dtype=torch.long),
        }
