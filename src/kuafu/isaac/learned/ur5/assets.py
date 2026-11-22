# SPDX-License-Identifier: Apache-2.0
"""UR5 asset locations and the ordered robot-state contract; no simulator imports."""

import os
from pathlib import Path

from kuafu.paths import ROBOT_ASSET_ROOT

ARM_JOINTS = (
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
)
GRIPPER_JOINTS = ("finger_joint", "right_outer_knuckle_joint")
JOINT_NAMES = ARM_JOINTS + GRIPPER_JOINTS
GRIPPER_ROOT = "Gripper/Robotiq_2F_85"
# Fixed virtual grasp center in wrist_3_link; metres and wxyz.
TCP_POSITION = (0.0, 0.0, 0.162)
TCP_QUATERNION = (1.0, 0.0, 0.0, 0.0)
HOME_JOINTS = (0.0, -1.57079632679, 1.57079632679, -1.57079632679, -1.57079632679, 0.0)


def asset_directory() -> Path:
    """Use the bundled robot, or an explicitly selected replacement."""
    override = os.environ.get("KUAFU_UR5_ASSET_DIR", "").strip()
    return Path(override).expanduser().resolve() if override else ROBOT_ASSET_ROOT / "ur5_robotiq"


def require_assets() -> tuple[Path, Path]:
    """Fail before scene creation with an actionable setup command."""
    directory = asset_directory()
    paths = directory / "ur5_robotiq.usd", directory / "ur5_controller.urdf"
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError(
            f"UR5 assets are missing in {directory}. Restore assets/robots/ur5_robotiq "
            "from the checkout, or select a complete robot with KUAFU_UR5_ASSET_DIR."
        )
    return paths
