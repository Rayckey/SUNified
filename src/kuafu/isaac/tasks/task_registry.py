"""Task registration helpers for short and robot-aware task ids."""

from __future__ import annotations

import importlib


SHORT_TASK_IDS = {
    "fr3-finray-drawer-stack",
    "hang-cup",
    "open-drawer",
    "open-fr3-finray-drawer",
    "pen-cup",
    "stack-3",
    "stack-2",
}

SHORT_TASK_MODULES = (
    "kuafu.isaac.tasks.open_drawer",
    "kuafu.isaac.tasks.pen_cup",
    "kuafu.isaac.tasks.hang_cup",
    "kuafu.isaac.tasks.stack_two",
    "kuafu.isaac.tasks.drawer_stack",
    "kuafu.isaac.tasks.stack_three",
)

ROBOT_TASK_MODULES = (
    ("ur5_robotiq", ("UR5-Table",), ("kuafu.isaac.learned.ur5",)),
    ("fr3_finray", ("FR3-FinRay-Table",), ("kuafu.isaac.learned",)),
    ("franka_panda", ("Panda-Table",), ("kuafu.isaac.learned.panda",)),
    ("kinova_gen3", ("Kinova-Table-Gen3",), ("kuafu.isaac.learned.kinova",)),
)


def _robot_registration_for_task(task: str | None):
    text = str(task or "").strip()
    for robot_id, prefixes, modules in ROBOT_TASK_MODULES:
        if any(text.startswith(prefix) for prefix in prefixes):
            return robot_id, modules
    return None, ()


def resolve_execution_robot_id(task: str | None, execution_robot_id: str | None = None) -> str:
    if execution_robot_id:
        return str(execution_robot_id)
    text = str(task or "").strip()
    if text in SHORT_TASK_IDS:
        return "task_library"
    robot_id, _ = _robot_registration_for_task(text)
    return robot_id or "isaaclab_tasks"


def register_task_modules(*, task: str | None = None, execution_robot_id: str | None = None, strict: bool = True) -> str:
    """Register short ids, robot-aware manipulation ids, or Isaac Lab stock ids."""

    try:
        importlib.import_module("isaaclab_tasks")
    except Exception:
        if strict:
            raise

    text = str(task or "").strip()
    if text in SHORT_TASK_IDS:
        importlib.import_module("kuafu.isaac.tasks")
        for module_name in SHORT_TASK_MODULES:
            try:
                importlib.import_module(module_name)
            except Exception:
                if strict:
                    raise
        return "task_library"

    robot_id, modules = _robot_registration_for_task(text)
    for module_name in modules:
        try:
            importlib.import_module(module_name)
        except Exception:
            if strict:
                raise
    return str(execution_robot_id or robot_id or "isaaclab_tasks")
