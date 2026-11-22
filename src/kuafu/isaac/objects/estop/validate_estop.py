from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the articulated e-stop asset.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--steps-per-phase", type=int, default=120)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


args = _parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.sim import build_simulation_context

from kuafu.isaac.objects.estop import EStopArticulation, make_estop_cfg


def main() -> None:
    with build_simulation_context(
        device=args.device,
        add_ground_plane=True,
        auto_add_lighting=True,
    ) as sim:
        sim_utils.create_prim("/World/Origin", "Xform")
        estop = EStopArticulation(
            make_estop_cfg(
                prim_path="/World/EStop",
                spawn_pose=(0.0, 0.0, 0.05),
            )
        )

        sim.reset()
        estop.reset()

        phase_steps = int(args.steps_per_phase)
        dt = sim.get_physics_dt()

        for step in range(phase_steps * 4):
            if step == phase_steps:
                estop.press()
            elif step == phase_steps * 2:
                estop.unlock()
            elif step == phase_steps * 3:
                unlocked_mask = estop.release()
                print(f"unlock mask: {unlocked_mask.cpu().tolist()}")

            estop.write_data_to_sim()
            sim.step()
            estop.update(dt)

        print("joint_names:", estop.joint_names)
        print("final_joint_pos:", estop.data.joint_pos[0].detach().cpu().tolist())
        print("final_joint_vel:", estop.data.joint_vel[0].detach().cpu().tolist())

        sim.pause()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
