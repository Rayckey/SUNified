"""Collect, fit and evaluate a BC policy for a verified generated cabinet."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from kuafu.agents.jobs import asset_hashes, digest, read_json, tree_hashes, write_json


def check_inputs(directory, request):
    if tree_hashes(directory / "input") != request["input_hashes"]:
        raise ValueError("Learning inputs changed after this attempt was created")
    if asset_hashes(request["context"]["asset"]) != request["context"]["asset_hashes"]:
        raise ValueError("Generated asset differs from the verified asset")


def replay_command(directory, request, output, seed, *, policy=None):
    context = request["context"]
    command = [request["python"], "-m", "kuafu", "verify-drawer", "--headless", "--enable_cameras",
               "--device", request["device"], "--asset", context["asset"], "--task", context["task"],
               "--target-travel", str(context["target_travel_m"]), "--position", *map(str, context["position"]),
               "--yaw", str(context["yaw_deg"]), "--max-requests", "0", "--mode", "replay",
               "--program", str(directory / "input/program"), "--output", str(output),
               "--max-steps", "1800", "--stage-steps", "500", "--seed", str(seed)]
    return command + (["--policy", str(policy)] if policy else ["--record-demo"])


def collect(directory, request, output):
    from kuafu.data.generated_drawer import combine_demonstrations

    episodes = []
    for index in range(request["episodes"]):
        destination = output / f"episode_{index:03d}"
        write_json(output / "progress.json", {"phase": "collect", "episode": index + 1,
                                             "episodes": request["episodes"]})
        # Inherit the job's process group so cancellation also stops Isaac.
        subprocess.run(replay_command(directory, request, destination, request["seed"] + index),
                       check=True, timeout=request["timeout_s"])
        report = read_json(destination / "verification.json")
        if not (report["passed"] and report["solver_passed"] and report["model_requests"] == 0):
            raise ValueError(f"Episode {index} did not pass MPC and physical verification")
        episodes.append(destination / "demonstrations.hdf5")
    dataset = output / "demonstrations.hdf5"
    result = combine_demonstrations(episodes, dataset)
    return result, {"dataset": dataset}


def train(directory, request, output):
    import h5py
    import json
    import numpy as np
    import torch
    import yaml

    dataset = directory / "input/demonstrations.hdf5"
    with h5py.File(dataset, "r") as handle:
        contract = json.loads(handle.attrs["metadata"])
    if request.get("method", "mlp") == "nearest_neighbor":
        from kuafu.bc.generated_drawer import fit_nearest_policy

        checkpoint = output / "policy.pth"
        result = fit_nearest_policy(dataset, checkpoint, request["seed"])
        return result, {"checkpoint": checkpoint}
    stages, metrics = [], []
    count = len(list((directory / "input/program").glob("subtask_*")))
    for index in range(count):
        stage = output / f"stage_{index:02d}"
        stage.mkdir()
        stage_dataset = stage / "demonstrations.hdf5"
        with h5py.File(dataset, "r") as source, h5py.File(stage_dataset, "w") as destination:
            for name, demo in source["data"].items():
                observations = demo["obs/state"][:]
                mask = observations[:, -1] == index
                if not mask.any():
                    raise ValueError(f"Episode {name} has no decisions for SUN stage {index}")
                target = destination.create_group(f"data/{name}")
                target.create_dataset("obs/state", data=observations[mask])
                actions = demo["actions/absolute_joint"][:][mask]
                target.create_dataset("actions/relative_joint", data=actions - observations[mask, :actions.shape[1]])
                target.attrs["num_samples"] = int(np.count_nonzero(mask))
        config = {
            "dataset": {"dataset": str(stage_dataset), "obs_key": "state", "action_mode": "relative_joint",
                        "time_shift": 0, "val_fraction": 0.25, "split_mode": "random", "seed": request["seed"]},
            "model": {"hidden_dims": [64, 64], "dropout": 0.0},
            "optimization": {"lr": 0.001, "batch_size": 64, "num_workers": 0,
                             "num_epochs": request["epochs"], "save_every_n_epochs": request["epochs"]},
            "logging": {"log_dir": str(stage / "training")},
        }
        config_path = stage / "train.yaml"
        config_path.write_text(yaml.safe_dump(config))
        write_json(output / "progress.json", {"phase": "train", "stage": index, "stages": count})
        subprocess.run([request["python"], "-m", "kuafu.bc.training.train_bc_prior_mlp", "--config", str(config_path)],
                       check=True, timeout=request["timeout_s"], env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        checkpoints = list((stage / "training").glob("*/checkpoints/final.pth"))
        if len(checkpoints) != 1:
            raise ValueError("Each SUN stage must produce exactly one final checkpoint")
        saved = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
        stages.append(saved)
        metrics.append({"stage": index, "epochs": saved["epoch"], "train_loss": saved["train_loss"],
                        "val_loss": saved["val_loss"]})
    checkpoint = output / "policy.pth"
    torch.save({"generated_drawer_contract": contract, "stages": stages}, checkpoint)
    return {"epochs_per_stage": request["epochs"], "stages": metrics,
            "contract": contract, "checkpoint_selection": "final"}, {"checkpoint": checkpoint}


def evaluate(directory, request, output):
    rollout = output / "rollout"
    result = subprocess.run(replay_command(directory, request, rollout, request["seed"],
                                          policy=directory / "input/policy.pth"),
                            check=False, timeout=request["timeout_s"])
    report_path = rollout / "verification.json"
    report = read_json(report_path)
    if result.returncode not in (0, 1) or report.get("error") or report.get("controller") != "bc_policy":
        raise ValueError(f"Policy evaluation did not finish: {report.get('error', result.returncode)}")
    if report["model_requests"] != 0 or report["mpc_solves"] != 0:
        raise ValueError("Policy evaluation invoked a model or MPC action solver")
    return {"policy_passed": report["passed"], "physical": report["physical"],
            "program_passed": report["program_passed"], "control_steps": report["control_steps"],
            "mpc_solves": report["mpc_solves"]}, {"evaluation": report_path,
                                                   "rollout_image": rollout / "final_table_cam_0.png"}


def finish(directory, request):
    """Validate outputs before a job advertises a completed learning handoff."""
    check_inputs(directory, request)
    report = read_json(directory / "output/report.json")
    if report.get("kind") != request["kind"] or report.get("completed") is not True:
        raise ValueError("Learning job did not record a completed result")
    required = {"drawer-collect": {"dataset"}, "drawer-train": {"checkpoint"},
                "drawer-evaluate": {"evaluation", "rollout_image"}}[request["kind"]]
    if set(report["artifacts"]) != required:
        raise ValueError("Learning job is missing required output artifacts")
    for path in report["artifacts"].values():
        if not Path(path).resolve().is_relative_to((directory / "output").resolve()):
            raise ValueError("Learning artifact escaped its job output directory")
        if digest(path) != report["files"].get(path):
            raise ValueError("Learning artifact changed after completion")
    return report


def main():
    directory = Path(sys.argv[1]).resolve()
    request = read_json(directory / "request.json")
    check_inputs(directory, request)
    output = directory / "output"
    output.mkdir()
    operation = {"drawer-collect": collect, "drawer-train": train, "drawer-evaluate": evaluate}[request["kind"]]
    result, artifacts = operation(directory, request, output)
    check_inputs(directory, request)
    write_json(output / "report.json", {
        "kind": request["kind"], "completed": True, "result": result,
        "artifacts": {key: str(path) for key, path in artifacts.items()},
        "files": {str(path): digest(path) for path in artifacts.values()},
    })
    write_json(output / "progress.json", {"phase": "complete"})


if __name__ == "__main__":
    main()
