#!/usr/bin/env python3
"""Project recorder HDF5 files into the flat layout expected by robomimic.

This utility is intentionally robomimic-only. It never modifies the source
trajectory. Each output keeps one observation branch and flattens one action
branch from ``actions/<mode>`` to the standard ``actions`` dataset.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py


REPO_ROOT = Path(__file__).resolve().parents[4]
_CONFIGURED_ARTIFACT_ROOT = os.environ.get("KUAFU_ARTIFACT_ROOT", "").strip()
ARTIFACT_ROOT = (
    Path(_CONFIGURED_ARTIFACT_ROOT).expanduser().resolve()
    if _CONFIGURED_ARTIFACT_ROOT
    else REPO_ROOT.parent / "KuafuArtifacts"
)
DEFAULT_MANIFEST = Path(__file__).with_name("manifest.json")


def _require_external_artifact(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if resolved == REPO_ROOT or REPO_ROOT in resolved.parents:
        raise ValueError(f"{label} must be outside the source project: {resolved}")
    return resolved


def _artifact_reference(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return _require_external_artifact(path, "Artifact path")
    return _require_external_artifact(ARTIFACT_ROOT / path, "Artifact path")


def _copy_attrs(source: h5py.AttributeManager, destination: h5py.AttributeManager) -> None:
    for key, value in source.items():
        destination[key] = value


def _prepared_name(source: Path, obs_key: str, action_mode: str) -> str:
    return f"{source.stem}__obs-{obs_key}__action-{action_mode}.hdf5"


def prepare_dataset(source: Path, destination: Path, obs_key: str, action_mode: str, overwrite: bool) -> dict:
    source = source.resolve()
    destination = destination.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists() and not overwrite:
        return validate_dataset(destination, obs_key, action_mode, expected_source=source)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()

    total_samples = 0
    try:
        with h5py.File(source, "r") as src, h5py.File(temporary, "w") as dst:
            if "data" not in src:
                raise KeyError(f"{source}: missing /data")
            src_data = src["data"]
            dst_data = dst.create_group("data")
            _copy_attrs(src.attrs, dst.attrs)
            _copy_attrs(src_data.attrs, dst_data.attrs)

            demos = sorted(src_data.keys())
            if not demos:
                raise ValueError(f"{source}: /data contains no demonstrations")
            for demo_name in demos:
                src_demo = src_data[demo_name]
                obs_path = f"obs/{obs_key}"
                next_obs_path = f"next_obs/{obs_key}"
                action_path = f"actions/{action_mode}"
                for required in (obs_path, action_path):
                    if required not in src_demo:
                        raise KeyError(f"{source}: {demo_name} missing {required}")

                length = int(src_demo[obs_path].shape[0])
                if int(src_demo[action_path].shape[0]) != length:
                    raise ValueError(f"{source}: {demo_name} observation/action length mismatch")

                dst_demo = dst_data.create_group(demo_name)
                _copy_attrs(src_demo.attrs, dst_demo.attrs)
                dst_demo.attrs["num_samples"] = length
                dst_obs = dst_demo.create_group("obs")
                src.copy(src_demo[obs_path], dst_obs, name=obs_key)
                if next_obs_path in src_demo:
                    dst_next_obs = dst_demo.create_group("next_obs")
                    src.copy(src_demo[next_obs_path], dst_next_obs, name=obs_key)
                src.copy(src_demo[action_path], dst_demo, name="actions")
                for optional in ("rewards", "dones"):
                    if optional in src_demo:
                        src.copy(src_demo[optional], dst_demo, name=optional)
                total_samples += length

            dst_data.attrs["total"] = total_samples
            dst.attrs["robomimic_only"] = True
            dst.attrs["source_dataset"] = str(source)
            dst.attrs["obs_key"] = obs_key
            dst.attrs["action_mode"] = action_mode
            dst.flush()
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return validate_dataset(destination, obs_key, action_mode, expected_source=source)


def validate_dataset(path: Path, obs_key: str, action_mode: str, expected_source: Path | None = None) -> dict:
    with h5py.File(path, "r") as dataset:
        if not bool(dataset.attrs.get("robomimic_only", False)):
            raise ValueError(f"{path}: existing output is not marked robomimic_only")
        if str(dataset.attrs.get("obs_key", "")) != obs_key:
            raise ValueError(f"{path}: obs_key metadata does not match {obs_key}")
        if str(dataset.attrs.get("action_mode", "")) != action_mode:
            raise ValueError(f"{path}: action_mode metadata does not match {action_mode}")
        if expected_source is not None:
            recorded = Path(str(dataset.attrs.get("source_dataset", ""))).resolve()
            if recorded != expected_source.resolve():
                raise ValueError(f"{path}: source metadata points to {recorded}, expected {expected_source}")
        demos = sorted(dataset["data"].keys())
        first = dataset["data"][demos[0]]
        if not isinstance(first["actions"], h5py.Dataset):
            raise TypeError(f"{path}: actions must be a flat dataset for robomimic")
        return {
            "path": str(path),
            "demos": len(demos),
            "samples": int(dataset["data"].attrs["total"]),
            "obs_dim": int(first[f"obs/{obs_key}"].shape[-1]),
            "action_dim": int(first["actions"].shape[-1]),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    obs_key = str(manifest["obs_key"])
    action_mode = str(manifest["action_mode"])
    prepared_dir = _artifact_reference(str(manifest["prepared_dir"]))
    results = []
    for job in manifest["jobs"]:
        source = _artifact_reference(str(job["dataset"]))
        destination = prepared_dir / _prepared_name(source, obs_key, action_mode)
        result = prepare_dataset(source, destination, obs_key, action_mode, args.overwrite)
        job["prepared_dataset"] = str(destination)
        results.append({"name": job["name"], **result})
        print(json.dumps(results[-1], sort_keys=True), flush=True)

    resolved_path = prepared_dir / "resolved_manifest.json"
    resolved_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {resolved_path}")


if __name__ == "__main__":
    main()
