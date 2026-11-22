from __future__ import annotations

import torch


DEFAULT_ESTOP_TRIGGER_FINGERTIP_MAX_DISTANCE = 0.02


def estop_fingertip_interaction_allowed(
    env,
    target_object_name: str,
    max_distance: float = DEFAULT_ESTOP_TRIGGER_FINGERTIP_MAX_DISTANCE,
) -> torch.Tensor:
    api = env.program_runtime.controller_api_wrapper
    target_pos = api.get_object_xyz_position(target_object_name).to(env.device)
    fingertip_pos = api.lookup_robot_frame_position("panda_fingertip").to(env.device)
    distance = torch.linalg.norm(fingertip_pos - target_pos, dim=-1)
    return distance <= float(max_distance)


def update_estop_state_with_fingertip_gate(
    env,
    target_object_name: str,
    max_distance: float = DEFAULT_ESTOP_TRIGGER_FINGERTIP_MAX_DISTANCE,
) -> torch.Tensor:
    interaction_allowed = estop_fingertip_interaction_allowed(
        env, target_object_name, max_distance
    )
    env.estop.update_state(interaction_allowed=interaction_allowed)
    return interaction_allowed
