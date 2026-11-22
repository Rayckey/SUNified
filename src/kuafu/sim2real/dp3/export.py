"""Add aligned SUN supervision to the Kuafu2Real HDF5-to-DP3 exporter."""

from pathlib import Path
import json
import shutil
import tempfile

import h5py
import numpy as np
import zarr

from kuafu.data.stage_labels import STAGE_TIMING
from kuafu.paths import resolve_artifact_path

from .artifacts import file_digest, json_digest, write_json, zarr_digest


def trajectory_splits(count, *, seed=42, validation_fraction=0.15, test_fraction=0.15):
    if count < 3 or not 0 < validation_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("Use at least three trajectories and positive validation/test fractions")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions leave no training partition")
    n_val, n_test = max(1, int(count * validation_fraction)), max(1, int(count * test_fraction))
    if n_val + n_test >= count:
        raise ValueError("Fractions leave no training trajectories")
    order = np.random.default_rng(seed).permutation(count).tolist()
    return {"train": sorted(order[n_val + n_test:]), "validation": sorted(order[:n_val]),
            "test": sorted(order[n_val:n_val + n_test])}


def export_dataset(inputs, output, *, task, stage_names, num_points=4096, action_format="eff-simple",
                   action_shift=None, seed=42, validation_fraction=0.15, test_fraction=0.15,
                   clip_zone=None, provenance=None):
    from kuafu2real.dataset import export_kuafu_hdf5_to_zarr, find_episode_h5, _valid_pair_indices

    if not task or len(stage_names) < 2 or len(set(stage_names)) != len(stage_names):
        raise ValueError("Specify one task and at least two distinct ordered stage names")
    if any(not isinstance(name, str) or not name.strip() for name in stage_names):
        raise ValueError("Stage names must be nonempty strings in teacher stage-ID order")
    if action_format not in {"eff-simple", "bc"}:
        raise ValueError("This experiment supports Kuafu2Real eff-simple and bc action contracts")
    shift = (0 if action_format == "bc" else 1) if action_shift is None else int(action_shift)
    if shift < 0 or (action_format == "bc" and shift != 0):
        raise ValueError("Use a nonnegative action shift; bc requires zero")
    output = resolve_artifact_path(output, "Stage dataset output")
    if output.exists():
        raise FileExistsError(f"Choose a new dataset output: {output}")
    episodes = find_episode_h5(inputs)
    splits = trajectory_splits(len(episodes), seed=seed, validation_fraction=validation_fraction,
                               test_fraction=test_fraction)
    labels, validity, rows, sources, transitions = [], [], [], [], []
    recordings, observed_grippers = [], []
    for path in episodes:
        with h5py.File(path, "r") as handle:
            required = {"decision_stage", "decision_valid", "observed_gripper", "action"}
            if not required.issubset(handle) or handle.attrs.get("decision_stage_timing") != STAGE_TIMING:
                raise ValueError(f"{path}: collect decision_stage, decision_valid, and observed_gripper before env.step()")
            recordings.append(json.loads(handle.attrs.get("point_cloud_recording", "null")))
            stages = np.asarray(handle["decision_stage"])
            valid = np.asarray(handle["decision_valid"])
            length = len(handle["action"])
            gripper = np.asarray(handle["observed_gripper"])
            if gripper.shape != (length,) or not np.isfinite(gripper).all() or np.any((gripper < 0) | (gripper > 1)):
                raise ValueError(f"{path}: expected one measured gripper openness in [0, 1] per observation")
            pc_key = "point_cloud" if "point_cloud" in handle else "point_cloud_count"
            if pc_key not in handle or len(handle[pc_key]) != length:
                raise ValueError(f"{path}: observation/action lengths differ")
            if stages.shape != (length,) or valid.shape != (length,) or stages.dtype.kind not in "iu" or valid.dtype != bool:
                raise ValueError(f"{path}: stage arrays must have one integer/bool per observation")
            if not valid.any() or ((stages[valid] < 0) | (stages[valid] >= len(stage_names))).any():
                raise ValueError(f"{path}: valid decision stages are outside the declared vocabulary")
            if (stages[~valid] != -1).any():
                raise ValueError(f"{path}: unavailable/terminal decisions must have stage -1")
            if int(handle.attrs.get("decision_stage_count", len(stage_names))) != len(stage_names):
                raise ValueError(f"{path}: stage vocabulary differs from the recorded SUN Program")
            if action_format == "bc" and any(key not in handle or len(handle[key]) != length
                                             for key in ("joint_pos", "joint_targets")):
                raise ValueError(f"{path}: bc joint observations/commands are misaligned")
            obs_idx, _ = _valid_pair_indices(length, length, shift)
            if len(obs_idx) == 0:
                raise ValueError(f"{path}: no obs/action pairs")
            observed_grippers.append(gripper[obs_idx].astype(np.float32))
            labels.append(stages[obs_idx].astype(np.int64))
            validity.append(valid[obs_idx])
            rows.append(obs_idx)
            changes = np.flatnonzero((stages[1:] != stages[:-1]) & valid[1:] & valid[:-1]) + 1
            transitions.append({"source": str(path), "transitions": [
                {"observation_row": int(row), "stage_before": int(stages[row - 1]),
                 "stage_at_decision": int(stages[row]), "target_action_row": int(row + shift)}
                for row in changes if row + shift < length]})
        sources.append({"path": str(path), "sha256": file_digest(path)})
    if len({source["sha256"] for source in sources}) != len(sources):
        raise ValueError("Duplicate trajectories would contaminate the train/validation/test split")
    if len({json_digest(value) for value in recordings}) != 1:
        raise ValueError("Trajectories use different camera/preprocessing settings")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + ".", dir=output.parent))
    try:
        metadata = export_kuafu_hdf5_to_zarr(
            episodes, temporary, num_points=num_points, action_format=action_format,
            action_shift=shift, seed=seed, clip_zone=clip_zone,
        )
        if any(file_digest(source["path"]) != source["sha256"] for source in sources):
            raise ValueError("A source trajectory changed during export")
        root = zarr.open_group(str(temporary), mode="a")
        measured = np.concatenate(observed_grippers)
        for key in ("state", "agent_pos"):
            root["data"][key][:, -1] = measured
        metadata["agent_gripper_source"] = "measured_pre_action_openness"
        write_json(temporary / "kuafu_export_metadata.json", metadata)
        expected_ends = np.cumsum([len(value) for value in labels])
        if not np.array_equal(root["meta/episode_ends"][:], expected_ends):
            raise ValueError("Kuafu2Real exporter changed observation/action alignment")
        for name, values in (("stage_id", labels), ("decision_valid", validity), ("source_row", rows)):
            root["data"].create_dataset(name, data=np.concatenate(values), chunks=(min(1024, metadata["num_steps"]),))
        content = {"schema_version": 1, "task": task, "stage_names": stage_names,
                   "stage_timing": STAGE_TIMING, "seed": seed, "splits": splits,
                   "recording": recordings[0],
                   "sources": sources, "export": metadata, "dependencies": provenance or {},
                   "content_sha256": zarr_digest(temporary)}
        content["dataset_id"] = json_digest(content)
        write_json(temporary / "stage_dataset.json", content)
        write_json(temporary / "stage_transitions.json", transitions)
        temporary.rename(output)
        return content
    except BaseException:
        shutil.rmtree(temporary)
        raise
