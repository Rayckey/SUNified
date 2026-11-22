#!/usr/bin/env python3
"""Launch the unified robomimic-only BC jobs across four GPUs."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py


REPO_ROOT = Path(__file__).resolve().parents[4]
_CONFIGURED_ARTIFACT_ROOT = os.environ.get("KUAFU_ARTIFACT_ROOT", "").strip()
ARTIFACT_ROOT = (
    Path(_CONFIGURED_ARTIFACT_ROOT).expanduser().resolve()
    if _CONFIGURED_ARTIFACT_ROOT
    else REPO_ROOT.parent / "KuafuArtifacts"
)
DEFAULT_MANIFEST = ARTIFACT_ROOT / "Trajectories" / "robomimic" / "resolved_manifest.json"


def _require_external_artifact(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if resolved == REPO_ROOT or REPO_ROOT in resolved.parents:
        raise ValueError(f"{label} must be outside the source project: {resolved}")
    return resolved


def _validate_flat_dataset(path: Path, obs_key: str, action_mode: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Prepared robomimic dataset missing: {path}")
    with h5py.File(path, "r") as dataset:
        if not bool(dataset.attrs.get("robomimic_only", False)):
            raise ValueError(f"{path} is not a robomimic-only prepared dataset")
        if str(dataset.attrs.get("obs_key", "")) != obs_key:
            raise ValueError(f"{path} observation metadata mismatch")
        if str(dataset.attrs.get("action_mode", "")) != action_mode:
            raise ValueError(f"{path} action metadata mismatch")
        first_demo = dataset["data"][sorted(dataset["data"].keys())[0]]
        if not isinstance(first_demo["actions"], h5py.Dataset):
            raise TypeError(f"{path}: /data/<demo>/actions is not flat")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--run-id", default=dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest_path = _require_external_artifact(args.manifest, "Prepared-data manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_config_path = REPO_ROOT / manifest["base_config"]
    base_config = json.loads(base_config_path.read_text(encoding="utf-8"))
    run_root = _require_external_artifact(
        ARTIFACT_ROOT / "standalone" / "robomimic" / "training" / args.run_id,
        "Training run",
    )
    config_dir = run_root / "configs"
    process_log_dir = run_root / "process_logs"
    config_dir.mkdir(parents=True, exist_ok=False)
    process_log_dir.mkdir(parents=True, exist_ok=False)

    launch_records = []
    for job in manifest["jobs"]:
        dataset_path = _require_external_artifact(
            Path(job["prepared_dataset"]), "Prepared dataset"
        )
        _validate_flat_dataset(dataset_path, manifest["obs_key"], manifest["action_mode"])
        job_output = run_root / "models" / job["name"]
        config = copy.deepcopy(base_config)
        config["experiment"]["name"] = f"robomimic_bc_{job['name']}"
        config["train"]["data"] = str(dataset_path)
        config["train"]["output_dir"] = str(job_output)
        if args.epochs is not None:
            config["train"]["num_epochs"] = int(args.epochs)
        config_path = config_dir / f"{job['name']}.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

        command = [sys.executable, "-m", "robomimic.scripts.train", "--config", str(config_path)]
        record = {
            "name": job["name"],
            "task": job["task"],
            "gpu": int(job["gpu"]),
            "source_dataset": job["dataset"],
            "prepared_dataset": job["prepared_dataset"],
            "obs_key": manifest["obs_key"],
            "action_mode": manifest["action_mode"],
            "config": str(config_path),
            "process_log": str(process_log_dir / f"{job['name']}.log"),
            "command": command,
        }
        if not args.dry_run:
            environment = os.environ.copy()
            environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            environment["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
            environment["PYTHONUNBUFFERED"] = "1"
            log_handle = (process_log_dir / f"{job['name']}.log").open("w", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            log_handle.close()
            record["pid"] = process.pid
        launch_records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    state = {
        "robomimic_only": True,
        "run_id": args.run_id,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "base_config": str(base_config_path.relative_to(REPO_ROOT)),
        "jobs": launch_records,
    }
    (run_root / "launch_state.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {run_root / 'launch_state.json'}")


if __name__ == "__main__":
    main()
