"""Load completed BC evaluations without importing the simulator."""

import json
from pathlib import Path

from kuafu.artifact_contracts import file_sha256, validate_program_metadata
from kuafu.paths import resolve_artifact_path


def load_bc_eval_stats(path, *, prior_checkpoint, task, expected_program=None):
    path = resolve_artifact_path(path, "BC evaluation statistics")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"BC eval stats must deserialize to a mapping: {path}")

    checkpoint = Path(prior_checkpoint).resolve()
    recorded = [payload[key] for key in ("checkpoint", "prior_checkpoint") if payload.get(key)]
    if not recorded or any(resolve_artifact_path(value, "BC prior checkpoint") != checkpoint for value in recorded):
        raise ValueError("BC eval stats checkpoint does not match --prior_checkpoint or is missing.")
    for key in ("checkpoint_sha256", "prior_checkpoint_sha256"):
        if payload.get(key) and payload[key] != file_sha256(checkpoint):
            raise ValueError("BC eval stats checkpoint content has changed; rerun the evaluation.")
    if payload.get("task") and payload["task"] != task:
        raise ValueError("BC eval stats task does not match --task.")
    validate_program_metadata(payload, expected_program)
    completed = int(payload.get("completed_episodes", payload.get("total_episodes", 0)))
    if completed <= 0:
        raise ValueError("BC eval stats must contain completed episodes.")

    constraints = payload.get("constraint_met_rate_stats")
    transitions = payload.get("stage_transition_stats")
    if not isinstance(constraints, dict) or not isinstance(transitions, dict):
        raise TypeError("BC eval stats must contain constraint and stage-transition mappings.")
    payload["constraint_met_rate_stats"] = {
        int(stage): {int(component): dict(stats) for component, stats in components.items()}
        for stage, components in constraints.items()
    }
    payload["stage_transition_stats"] = {int(stage): dict(stats) for stage, stats in transitions.items()}
    payload["completed_episodes"] = completed
    payload["prior_checkpoint"] = str(checkpoint)
    return payload
