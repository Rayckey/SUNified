# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run learn-programs for opening both doors on the Sektion cabinet."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Learn-programs task for opening both Sektion cabinet doors."
)
parser.add_argument(
    "--disable_fabric",
    type=bool,
    default=True,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to simulate."
)
parser.add_argument(
    "--max_atomic_corrections",
    type=int,
    default=5,
    help="Maximum automatic atomic-action correction attempts per failed stage.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from kuafu.programs.cabinet_door_learn_programs_common import (
    run_cabinet_door_learn_programs,
)


ENV_NAME = "Formulate-Panda-Cabinet-Doors-Both-v0"
OUTPUT_ENV_NAME = "manipulation-cabinet-doors-both"

TASK_DESCRIPTION = "open both the left cabinet door and the right cabinet door fully"

detectable_objects = {
    "cabinet": (
        "mesh of the Sektion cabinet, local frame +z is up direction and local frame +x is the forward direction",
        "/World/envs/env_*/Cabinet",
    ),
    "door_left_link": (
        "mesh of the left cabinet door, hinged on its outer vertical edge and opened by rotating away from the cabinet up to approximately 90 degrees; estimated size (approx.): (0.04, 0.34, 0.45)",
        "/World/envs/env_*/Cabinet/door_left_link",
    ),
    "door_right_link": (
        "mesh of the right cabinet door, hinged on its outer vertical edge and opened by rotating away from the cabinet up to approximately 90 degrees; estimated size (approx.): (0.04, 0.34, 0.45)",
        "/World/envs/env_*/Cabinet/door_right_link",
    ),
    "left_door_hinge_frame": (
        "pivot frame on the outer vertical hinge edge of the left cabinet door; local frame +z is the hinge rotation axis/up direction; left_door_nob_frame should move in an arc around this +z axis while opening",
        "/Visuals/left_door_hinge_frame",
    ),
    "right_door_hinge_frame": (
        "pivot frame on the outer vertical hinge edge of the right cabinet door; local frame +z is the hinge rotation axis/up direction; right_door_nob_frame should move in an arc around this +z axis while opening",
        "/Visuals/right_door_hinge_frame",
    ),
    "left_door_nob_frame": (
        "frame of the left door knob, a small round pull near the cabinet center seam; relative to left_door_hinge_frame the knob is across the door along local +y; local frame +z is up direction, local frame +x is the initial pulling/open direction away from the cabinet face; use this as the grasp/contact frame and pull it along an arc about left_door_hinge_frame",
        "/Visuals/door_left_nob_frame",
    ),
    "right_door_nob_frame": (
        "frame of the right door knob, a small round pull near the cabinet center seam; relative to right_door_hinge_frame the knob is across the door along local -y; local frame +z is up direction, local frame +x is the initial pulling/open direction away from the cabinet face; use this as the grasp/contact frame and pull it along an arc about right_door_hinge_frame",
        "/Visuals/door_right_nob_frame",
    ),
    "panda_link4": (
        "frame of robot link 4",
        "/Visuals/panda_link4_frame",
    ),
    "panda_fingertip": (
        "frame of robot tool frame",
        "/Visuals/panda_hand_frame",
    ),
}


def main():
    run_cabinet_door_learn_programs(
        ENV_NAME,
        OUTPUT_ENV_NAME,
        TASK_DESCRIPTION,
        detectable_objects,
        args_cli,
        simulation_app,
        max_counter=None,
        max_atomic_corrections=args_cli.max_atomic_corrections,
        device=args_cli.device,
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
