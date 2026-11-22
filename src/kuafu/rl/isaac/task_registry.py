"""Resolve robot IDs and register their Isaac task families for residual RL."""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass


DEFAULT_ARTIFACT_ROBOT_ID = "fr3_finray"
DEFAULT_EXECUTION_ROBOT_ID = "fr3_finray"
BC_TASK_VARIANT_SUFFIXES = ("-fff3", "-ff3", "-f3", "-fff", "-ff", "-f")


@dataclass(frozen=True)
class RobotTaskRegistration:
    robot_id: str
    task_prefixes: tuple[str, ...]
    import_modules: tuple[str, ...]
    aliases: tuple[str, ...] = ()


ROBOT_TASK_REGISTRY: dict[str, RobotTaskRegistration] = {
    "ur5_robotiq": RobotTaskRegistration(
        robot_id="ur5_robotiq",
        task_prefixes=("UR5-Table",),
        import_modules=("kuafu.isaac.learned.ur5",),
        aliases=("ur5", "ur5-robotiq"),
    ),
    "fr3_finray": RobotTaskRegistration(
        robot_id="fr3_finray",
        task_prefixes=("FR3-FinRay-Table",),
        import_modules=("kuafu.isaac.learned",),
        aliases=(
            "fr3_finray",
            "franka",
            "franka_fr3_finray",
            "fr3-finray-table-franka",
            "fr3_finray_panda",
        ),
    ),
    "franka_panda": RobotTaskRegistration(
        robot_id="franka_panda",
        task_prefixes=("Panda-Table",),
        import_modules=("kuafu.isaac.learned.panda",),
        aliases=(
            "panda",
            "panda-franka",
            "panda_table_franka",
            "panda-table-franka",
            "franka_panda",
        ),
    ),
    "kinova_gen3": RobotTaskRegistration(
        robot_id="kinova_gen3",
        task_prefixes=("Kinova-Table-Gen3",),
        import_modules=("kuafu.isaac.learned.kinova",),
        aliases=(
            "kinova",
            "kinova-gen3",
            "gen3",
        ),
    ),
}

_ALIAS_TO_ROBOT_ID: dict[str, str] = {}
for _robot_id, _registration in ROBOT_TASK_REGISTRY.items():
    _ALIAS_TO_ROBOT_ID[_robot_id] = _robot_id
    _ALIAS_TO_ROBOT_ID[_robot_id.replace("_", "-")] = _robot_id
    for _alias in _registration.aliases:
        _ALIAS_TO_ROBOT_ID[_alias] = _robot_id
        _ALIAS_TO_ROBOT_ID[_alias.replace("_", "-")] = _robot_id


def normalize_robot_id(robot_id: str | None) -> str | None:
    """Return the canonical robot id, or None when the caller gave no preference."""

    if robot_id is None:
        return None
    text = str(robot_id).strip()
    if not text:
        return None
    key = text.lower().replace(" ", "_")
    key = key.replace("-", "_")
    normalized = _ALIAS_TO_ROBOT_ID.get(key) or _ALIAS_TO_ROBOT_ID.get(key.replace("_", "-"))
    if normalized is not None:
        return normalized
    valid = ", ".join(sorted(ROBOT_TASK_REGISTRY))
    raise ValueError(f"Unknown residual-RL robot id {robot_id!r}. Expected one of: {valid}.")


def infer_robot_id_from_task(task: str | None) -> str:
    """Infer the execution robot from a robot-aware manipulation task name."""

    text = str(task or "").strip()
    if text:
        for robot_id, registration in ROBOT_TASK_REGISTRY.items():
            if any(text.startswith(prefix) for prefix in registration.task_prefixes):
                return robot_id
        valid_prefixes = ", ".join(
            f"{prefix}*"
            for registration in ROBOT_TASK_REGISTRY.values()
            for prefix in registration.task_prefixes
        )
        raise ValueError(
            f"Residual-RL task {task!r} is not a robot-aware manipulation task id. "
            f"Expected one of these prefixes: {valid_prefixes}."
        )
    return DEFAULT_EXECUTION_ROBOT_ID


def resolve_execution_robot_id(task: str | None, execution_robot_id: str | None = None) -> str:
    return normalize_robot_id(execution_robot_id) or infer_robot_id_from_task(task)


def resolve_artifact_robot_id(artifact_robot_id: str | None = None) -> str:
    return normalize_robot_id(artifact_robot_id) or DEFAULT_ARTIFACT_ROBOT_ID


def registered_robot_ids() -> tuple[str, ...]:
    return tuple(sorted(ROBOT_TASK_REGISTRY))


def canonical_task_name_for_metadata(task: str | None) -> str:
    """Normalize task ids for metadata checks while ignoring BC actor variants."""

    text = str(task or "").strip()
    lower = text.lower()
    for suffix in BC_TASK_VARIANT_SUFFIXES:
        if lower.endswith(suffix):
            return text[: -len(suffix)].strip()
    return text


def task_names_match_for_metadata(left: str | None, right: str | None) -> bool:
    """Return true when two task ids differ only by a known BC actor suffix."""

    return canonical_task_name_for_metadata(left).lower() == canonical_task_name_for_metadata(right).lower()


def task_log_root(
    base_dir: str,
    task: str,
    execution_robot_id: str | None,
    *,
    legacy_fr3_finray_layout: bool = True,
) -> str:
    """Return the task-level log root, preserving FR3FinRay's historical layout."""

    robot_id = resolve_execution_robot_id(task, execution_robot_id)
    if legacy_fr3_finray_layout and robot_id == DEFAULT_EXECUTION_ROBOT_ID:
        return os.path.join(base_dir, task)
    return os.path.join(base_dir, robot_id, task)


def import_module(module_name: str, *, strict: bool = True) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        if strict:
            raise
        return False


def register_task_modules(
    *,
    task: str | None = None,
    execution_robot_id: str | None = None,
    strict: bool = True,
) -> str:
    """Import only robot-aware manipulation task modules for kuafu.rl."""

    robot_id = resolve_execution_robot_id(task, execution_robot_id)
    import_module("isaaclab_tasks", strict=False)
    registration = ROBOT_TASK_REGISTRY[robot_id]
    for module_name in registration.import_modules:
        import_module(module_name, strict=strict)
    return robot_id
