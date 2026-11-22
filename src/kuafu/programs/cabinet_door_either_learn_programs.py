# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run learn-programs for opening either door on the Sektion cabinet."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Learn-programs task for opening either Sektion cabinet door."
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
    "--manipulator_high_effort_threshold",
    "--manipulator-high-effort-threshold",
    dest="manipulator_high_effort_threshold",
    type=float,
    default=100.0,
    help=(
        "Total arm joint effort threshold for the manipulator_high_effort "
        "predicate. Increase this to delay effort-based contact stops."
    ),
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


ENV_NAME = "Formulate-Panda-Cabinet-Door-Either-v0"
OUTPUT_ENV_NAME = "manipulation-door-either"

TASK_DESCRIPTION = "open either the left cabinet door or the right cabinet door fully"

detectable_objects = {
    "cabinet": (
        "Sektion cabinet, local frame +z is up direction and local frame +x is the forward direction",
        "/World/envs/env_*/Cabinet",
    ),
    "door_left_link": (
        "left cabinet door, hinged on its outer vertical edge and opened by rotating away from the cabinet up to approximately 90 degrees; estimated size (approx.): (0.04, 0.34, 0.45)",
        "/World/envs/env_*/Cabinet/door_left_link",
    ),
    "door_right_link": (
        "right cabinet door, hinged on its outer vertical edge and opened by rotating away from the cabinet up to approximately 90 degrees; estimated size (approx.): (0.04, 0.34, 0.45)",
        "/World/envs/env_*/Cabinet/door_right_link",
    ),
    "left_door_hinge_frame": (
        "hinge frame of the left cabinet door; local frame +z is the hinge rotation axis; rotation around local frame -z to open left door, rotation around local frame +z to close left door",
        "/Visuals/left_door_hinge_frame",
    ),
    "right_door_hinge_frame": (
        "hinge frame of the right cabinet door; local frame +z is the hinge rotation axis; rotation around local frame +z to open right door, rotation around local frame -z to close right door",
        "/Visuals/right_door_hinge_frame",
    ),
    "left_door_nob_frame": (
        "frame of a circular left door knob, size (0.02, 0.02, 0.02); grasp against local frame +x",
        "/Visuals/door_left_nob_frame",
    ),
    "right_door_nob_frame": (
        "frame of a circular right door knob, size (0.02, 0.02, 0.02); grasp against local frame +x",
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
        manipulator_high_effort_threshold=args_cli.manipulator_high_effort_threshold,
        has_finger_contact_sensors=False,
        keep_running=True,
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
