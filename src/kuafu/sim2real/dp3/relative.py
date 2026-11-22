"""Convert a verified stage dataset into a separately identified object frame."""

import copy
import json
from pathlib import Path
import shutil
import tempfile

import h5py
import numpy as np
import zarr

from kuafu.paths import resolve_artifact_path

from .artifacts import file_digest, json_digest, write_json, zarr_digest
from .object_frame import OBJECT_RELATIVE, ObjectFrame, relative_contract


def _reference(handle, name):
    pairs = [(f"articulated_root_pos/{name}", f"articulated_root_rot/{name}"),
             (f"obj_positions/{name}", f"obj_rotations/{name}")]
    pairs = [(p, q) for p, q in pairs if p in handle and q in handle]
    if len(pairs) != 1:
        raise ValueError(f"{handle.filename}: expected one recorded root pose for {name}")
    position, quaternion = (handle[key][:] for key in pairs[0])
    length = len(handle["action"])
    if position.shape != (length, 3) or quaternion.shape != (length, 4):
        raise ValueError(f"{handle.filename}: reference root and observations are misaligned")
    poses = np.concatenate((position, quaternion), axis=-1)
    frame = ObjectFrame(poses[0])
    frame.check_stationary(poses)
    return frame


def convert_object_relative(source, output, *, reference_object, chunk_rows=128, provenance=None):
    """Preserve source rows/splits and transform all geometry with a recorded fixed root."""
    contract = relative_contract(reference_object)
    if type(chunk_rows) is not int or chunk_rows < 1:
        raise ValueError("chunk_rows must be a positive integer")
    source = Path(source).expanduser().resolve()
    output = resolve_artifact_path(output, "Object-relative dataset output")
    if output.exists():
        raise FileExistsError(f"Choose a new dataset output: {output}")
    if source in output.parents:
        raise ValueError("Place the converted dataset outside its source")
    manifest = json.loads((source / "stage_dataset.json").read_text())
    content = dict(manifest)
    identity = content.pop("dataset_id")
    if identity != json_digest(content) or content["content_sha256"] != zarr_digest(source):
        raise ValueError("Source dataset identity/content hash has changed")
    if manifest["export"]["action_format"] != "eff-simple":
        raise ValueError("Convert an absolute eff-simple stage dataset")
    src = zarr.open_group(str(source), mode="r")
    ends = src["meta/episode_ends"][:]
    if len(ends) != len(manifest["sources"]) or np.any(np.diff(np.r_[0, ends]) <= 0):
        raise ValueError("Source trajectories and episode boundaries differ")
    length = int(ends[-1])
    for name in ("state", "agent_pos", "action"):
        if src[f"data/{name}"].shape != (length, 10):
            raise ValueError("Expected matching 10D eff-simple state, agent_pos, and action")
    frames = []
    start = 0
    for end, recorded in zip(ends, manifest["sources"]):
        if file_digest(recorded["path"]) != recorded["sha256"]:
            raise ValueError(f"Source trajectory changed: {recorded['path']}")
        with h5py.File(recorded["path"], "r") as handle:
            rows = src["data/source_row"][start:end]
            if rows.dtype.kind not in "iu" or np.any(rows < 0) or np.any(rows >= len(handle["action"])):
                raise ValueError("Source row indices are outside the recorded trajectory")
            if (not np.array_equal(handle["decision_stage"][:][rows], src["data/stage_id"][start:end])
                    or not np.array_equal(handle["decision_valid"][:][rows], src["data/decision_valid"][start:end])):
                raise ValueError("Source decision labels are misaligned")
            frames.append(_reference(handle, reference_object))
        if file_digest(recorded["path"]) != recorded["sha256"]:
            raise ValueError(f"Source trajectory changed while reading: {recorded['path']}")
        start = end
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + ".", dir=output.parent))
    try:
        dst = zarr.open_group(str(temporary), mode="w")
        dst.attrs.update(src.attrs.asdict())
        for group in ("data", "meta"):
            target_group = dst.create_group(group)
            target_group.attrs.update(src[group].attrs.asdict())
            for name, array in src[group].arrays():
                target = target_group.create_dataset(name, shape=array.shape, chunks=array.chunks,
                    dtype=array.dtype, compressor=array.compressor, filters=array.filters,
                    fill_value=array.fill_value, order=array.order)
                target.attrs.update(array.attrs.asdict())
                if group == "meta" or name not in {"state", "agent_pos", "action", "point_cloud"}:
                    for row in range(0, len(array), chunk_rows):
                        target[row:row + chunk_rows] = array[row:row + chunk_rows]
        reference = dst["data"].create_dataset("reference_pose", shape=(length, 7), dtype="f4", chunks=(1024, 7))
        reference.attrs.update(encoding="xyz_wxyz", frame="environment_local", object=reference_object)
        errors = {name: 0.0 for name in ("state", "agent_pos", "action", "point_cloud")}
        start = 0
        for episode, (end, frame) in enumerate(zip(ends, frames)):
            for row in range(start, end, chunk_rows):
                stop = min(row + chunk_rows, end)
                reference[row:stop] = np.broadcast_to(frame.pose.astype(np.float32), (stop - row, 7))
                for name in errors:
                    original = src[f"data/{name}"][row:stop]
                    encoded = frame.encode_cloud(original) if name == "point_cloud" else frame.encode_pose(original)
                    restored = frame.decode_cloud(encoded) if name == "point_cloud" else frame.decode_pose(encoded)
                    error = float(np.max(np.abs(original - restored)))
                    errors[name] = max(errors[name], error)
                    if error > 2e-6:
                        raise ValueError(f"{name}: object-frame round trip exceeds 2e-6 at row {row}")
                    unchanged = slice(3, None) if name == "point_cloud" else slice(9, None)
                    if not np.array_equal(original[..., unchanged], encoded[..., unchanged]):
                        raise ValueError(f"{name}: conversion changed RGB/gripper values")
                    dst[f"data/{name}"][row:stop] = encoded
                if not np.array_equal(src["data/state"][row:stop], src["data/agent_pos"][row:stop]):
                    raise ValueError("state and agent_pos differ")
            start = end
            if (episode + 1) % 100 == 0:
                print(f"Converted {episode + 1}/{len(ends)} trajectories", flush=True)
        for name in errors:
            dst[f"data/{name}"].attrs["coordinate_frame"] = f"{reference_object}/root"
        converted = copy.deepcopy(manifest)
        converted.pop("dataset_id")
        converted["export"].update(action_format=OBJECT_RELATIVE, object_relative=contract)
        converted["conversion"] = {"source_dataset": str(source), "source_dataset_id": identity,
                                   "dependencies": provenance or {}, "reference_source": "recorded_hdf5_root_xyz_wxyz"}
        converted["content_sha256"] = zarr_digest(temporary)
        converted["dataset_id"] = json_digest(converted)
        write_json(temporary / "stage_dataset.json", converted)
        write_json(temporary / "kuafu_export_metadata.json", converted["export"])
        shutil.copyfile(source / "stage_transitions.json", temporary / "stage_transitions.json")
        write_json(temporary / "conversion_validation.json", {
            "source_dataset_id": identity, "dataset_id": converted["dataset_id"],
            "trajectories": len(ends), "decisions": length, "max_roundtrip_absolute_error": errors,
            "reference": contract, "gripper_and_rgb_unchanged": True,
            "source_labels_verified": True, "splits_preserved": True,
        })
        if zarr_digest(source) != manifest["content_sha256"]:
            raise ValueError("Source dataset changed during conversion")
        temporary.rename(output)
        return converted
    except BaseException:
        shutil.rmtree(temporary)
        raise
