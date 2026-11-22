"""Checkpoint collection options that can be validated without Isaac Sim."""

import json
from pathlib import Path

import torch

from kuafu.paths import resolve_artifact_path


PANDA_CAMERA_KEYS = ("table_cam_0", "table_cam_1", "table_cam_2")
KINOVA_CAMERA_KEYS = ("table_cam_0", "table_cam_1")


def collection_camera_keys(requested=None, *, is_panda=False, is_kinova=False):
    keys = tuple(requested) if requested else None
    if keys is not None and len(set(keys)) != len(keys):
        raise ValueError(f"--camera_keys must not contain duplicates: {keys}")
    if is_panda:
        if keys is not None and keys != PANDA_CAMERA_KEYS:
            raise ValueError(
                "Panda collection requires the three calibrated simulated cameras in order: "
                f"{PANDA_CAMERA_KEYS}; got {keys}."
            )
        return PANDA_CAMERA_KEYS
    return keys if keys is not None else KINOVA_CAMERA_KEYS if is_kinova else None


def configure_object_sides(env_cfg, *, disable_randomization=False, force_swap=False):
    if disable_randomization and force_swap:
        raise ValueError("--disable_left_right_randomization and --force_left_right_swap are mutually exclusive.")
    if not (disable_randomization or force_swap):
        return
    for field in ("randomize_left_right", "fixed_left_right_swap"):
        if not hasattr(env_cfg, field):
            raise ValueError(f"Task does not expose a {field} environment setting.")
    env_cfg.randomize_left_right = False
    env_cfg.fixed_left_right_swap = bool(force_swap)


def checkpoint_params_directory(path: str | Path) -> Path:
    directory = resolve_artifact_path(path, "Checkpoint playback parameters")
    missing = [name for name in ("agent.yaml", "env.yaml") if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Explicit checkpoint params directory {directory} is missing: {', '.join(missing)}")
    return directory


def find_checkpoint_params(checkpoint_path: str | Path, project_root: str | Path) -> Path | None:
    """Find the selected worker's settings beside a checkpoint or inside its moved archive."""
    checkpoint = Path(checkpoint_path).resolve()
    archive = checkpoint.parent.parent

    def params_for(path):
        candidates = (path / "params", path.parent / "params", path.parent.parent / "params")
        return next((candidate for candidate in candidates
                     if (candidate / "agent.yaml").is_file() and (candidate / "env.yaml").is_file()), None)

    direct = params_for(checkpoint)
    if direct is not None:
        return direct
    for name in ("best_checkpoint.json", "residuals_archive.json"):
        try:
            metadata = json.loads((archive / name).read_text())
        except FileNotFoundError:
            continue
        if not isinstance(metadata, dict):
            continue
        references = []
        for key, fields in (("best_checkpoint", ("selected_worker_checkpoint_archive_path", "selected_worker_checkpoint_source_path")),
                            ("selected_worker", ("archive_log_dir", "source_log_dir"))):
            section = metadata.get(key, {})
            if isinstance(section, dict):
                references.extend(section[field] for field in fields if section.get(field))
        for reference in references:
            path = Path(reference)
            candidates = []
            # The task/run identity survives moving an archive to another checkout or disk.
            anchor = (archive.parent.name, archive.name)
            for index in range(len(path.parts) - 1):
                if path.parts[index:index + 2] == anchor:
                    candidates.append(archive.joinpath(*path.parts[index + 2:]))
            if not path.is_absolute():
                candidates.extend((archive / path, Path(project_root) / path))
            else:
                candidates.append(path)
            for candidate in candidates:
                params = params_for(candidate)
                if params is not None:
                    return params
    return None


def limit_completed_successes(dones, successes, remaining: int):
    """Return a fresh saver mask and completion count, capped to the remaining budget."""
    device = dones.device if isinstance(dones, torch.Tensor) else None
    done_mask = torch.as_tensor(dones, dtype=torch.bool, device=device)
    success_mask = torch.as_tensor(successes, dtype=torch.bool, device=done_mask.device)
    if done_mask.ndim != 1 or done_mask.shape != success_mask.shape:
        raise ValueError("Completion and success masks must be one-dimensional with matching lengths.")
    accepted = done_mask & success_mask
    completed_ids = accepted.nonzero(as_tuple=False).flatten()
    accepted[completed_ids[max(0, int(remaining)):]] = False
    return accepted, int(completed_ids.numel())
