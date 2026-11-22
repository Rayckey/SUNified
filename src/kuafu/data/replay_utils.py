from __future__ import annotations
import math
import random
from pathlib import Path

import torch
from typing import Any, cast

from kuafu.mpc.simulation.spatial_utils import rotation_6d_to_quat


def load_adjacent_run_config(ckpt_path: str) -> dict | None:
    """Try to load meta/config.yaml (resolved nested config) saved alongside the checkpoint."""
    try:
        import yaml
        p = Path(ckpt_path)
        config_path = p.parent.parent / "meta" / "config.yaml"
        if not config_path.exists():
            return None
        with open(config_path) as f:
            return yaml.safe_load(f)
    except Exception:
        return None


class EpisodeActionStream:
    """Manages (episode, action buffer, index) per env for dataset-driven stepping."""

    def __init__(
        self,
        *,
        episodes: list[dict],
        num_envs: int,
        raw_env: Any,
        device: torch.device,
        action_mode: str,
        timeshift: int,
        past_begin_start: bool,
        full_state_setter,
        init_env_from_episode_fn,
        get_episode_actions_fn,
    ):
        self._episodes = episodes
        self._raw_env = raw_env
        self._device = device
        self._action_mode = action_mode
        self._timeshift = int(timeshift)
        self._past_begin_start = past_begin_start
        self._full_state_setter = full_state_setter
        self._init_env_from_episode = init_env_from_episode_fn
        self._get_episode_actions = get_episode_actions_fn

        self.env_actions_buf: list[torch.Tensor | None] = [None] * num_envs
        self.env_action_idx: list[int] = [0] * num_envs
        self.episode_per_env: list[dict | None] = [None] * num_envs

    def load_env(self, env_id: int) -> None:
        ep = random.choice(self._episodes)
        t = self._init_env_from_episode(
            self._raw_env, ep, env_id, self._past_begin_start, self._full_state_setter
        )
        actions = self._get_episode_actions(ep["episode_data"], self._action_mode)
        if actions is not None and actions.dim() == 3:
            actions = actions[:, 0, :]  # (T, N, act_dim) -> (T, act_dim)
        self.env_actions_buf[env_id] = actions
        self.env_action_idx[env_id] = t if self._timeshift == 0 else max(0, t - self._timeshift)
        self.episode_per_env[env_id] = ep

    def get_action(self, env_id: int) -> torch.Tensor | None:
        buf = self.env_actions_buf[env_id]
        idx = self.env_action_idx[env_id]
        if buf is None or idx >= buf.shape[0]:
            return None
        return buf[idx].to(device=self._device, dtype=torch.float32)

    def advance(self, env_id: int) -> None:
        self.env_action_idx[env_id] += 1

    def current_step(self, env_id: int) -> int:
        return self.env_action_idx[env_id]

    def current_episode(self, env_id: int) -> dict | None:
        return self.episode_per_env[env_id]


def _quat_conj_wxyz(q: torch.Tensor) -> torch.Tensor:
    """Quaternion conjugate for (w, x, y, z)."""
    q = q.view(-1)
    return torch.stack([q[0], -q[1], -q[2], -q[3]])


def _quat_mul_wxyz(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Quaternion multiply for (w, x, y, z). Returns q = q1 * q2."""
    q1 = q1.view(-1)
    q2 = q2.view(-1)
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def _quat_rotate_wxyz(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by quaternion q (wxyz)."""
    q = q.view(-1)
    v = v.view(-1)
    # Using: v' = v + 2*cross(q_vec, cross(q_vec, v) + w*v)
    w = q[0]
    qv = q[1:4]
    t = 2.0 * torch.cross(qv, v, dim=0)
    return v + w * t + torch.cross(qv, t, dim=0)


def _yaw_from_quat_wxyz(q: torch.Tensor) -> torch.Tensor:
    """Return Z-axis yaw from a wxyz quaternion."""
    q = q.view(-1)
    q = q / torch.clamp(torch.linalg.norm(q), min=1.0e-8)
    w, x, y, z = q
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _pose_inv(pos: torch.Tensor, quat_wxyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Invert a pose (pos, quat) with quat in wxyz."""
    q_inv = _quat_conj_wxyz(quat_wxyz)
    p_inv = _quat_rotate_wxyz(q_inv, -pos)
    return p_inv, q_inv


def _pose_mul(
    pos_a: torch.Tensor, quat_a: torch.Tensor, pos_b: torch.Tensor, quat_b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose poses: T = Ta * Tb, both (pos, quat) in world convention, quats wxyz."""
    p = pos_a + _quat_rotate_wxyz(quat_a, pos_b)
    q = _quat_mul_wxyz(quat_a, quat_b)
    return p, q


def _merge_recorded_state_into_runtime_state(
    runtime_state: dict[str, Any], recorded_state: dict[str, Any], env_id: int
) -> dict[str, Any]:
    """Merge a recorded per-episode state dict into a runtime state dict for one env.

    The recorder typically stores per-episode tensors with leading dimension 1
    (since each episode is collected from one env). The runtime scene state tensors
    have leading dimension `env.num_envs`. This helper copies recorded_state[0] into
    runtime_state[env_id] for matching tensor leaves.
    """

    for k, rv in recorded_state.items():
        if k not in runtime_state:
            continue
        tv = runtime_state[k]
        if isinstance(rv, dict) and isinstance(tv, dict):
            _merge_recorded_state_into_runtime_state(tv, rv, env_id)
        elif isinstance(rv, torch.Tensor) and isinstance(tv, torch.Tensor):
            rv_t = cast(torch.Tensor, rv)
            tv_t = cast(torch.Tensor, tv)
            # Only merge tensors that are batched across envs in runtime.
            if tv_t.dim() > 0 and tv_t.shape[0] > env_id and rv_t.dim() > 0:
                # recorded state is usually batched with leading dim 1
                if rv_t.shape[0] == 1:
                    tv_t[env_id].copy_(rv_t[0].to(device=tv_t.device, dtype=tv_t.dtype))
                # sometimes recorded state may already match runtime batching
                elif rv_t.shape[0] == tv_t.shape[0]:
                    tv_t[env_id].copy_(rv_t[env_id].to(device=tv_t.device, dtype=tv_t.dtype))
                else:
                    # shape mismatch: skip best-effort
                    continue
        else:
            continue
    return runtime_state


def _slice_scene_state_for_env_ids(scene_state: dict[str, Any], env_ids: torch.Tensor) -> dict[str, Any]:
    """Return a scene-state tree whose tensor leaves are narrowed to ``env_ids``.

    ``InteractiveScene.get_state()`` returns full-batch tensors for every env, while
    ``InteractiveScene.reset_to(..., env_ids=...)`` expects each tensor leaf to already
    match the selected env subset. Replay helpers usually mutate one env inside the
    full runtime state, so we narrow it right before applying the reset.
    """

    sliced_state: dict[str, Any] = {}
    for key, value in scene_state.items():
        if isinstance(value, dict):
            sliced_state[key] = _slice_scene_state_for_env_ids(value, env_ids)
        elif isinstance(value, torch.Tensor):
            value_t = cast(torch.Tensor, value)
            if value_t.dim() > 0 and value_t.shape[0] > int(env_ids.max().item()):
                sliced_state[key] = value_t.index_select(0, env_ids).clone()
            else:
                sliced_state[key] = value_t.clone()
        else:
            sliced_state[key] = value
    return sliced_state


def _set_robot_actuated_joint_state_from_9d(
    env, runtime_state: dict, joint_pos_9: torch.Tensor, joint_vel_9: torch.Tensor, env_id: int
) -> None:
    """Write the first 9 articulation joints from an observation into runtime_state.

    The observation records ``scene.state["articulation"]["robot"]["joint_position"][:, :9]``
    (see ``combine_joint_states(..., robot="fr3")``), i.e. the first 9 joints in the
    articulation's natural ordering.  We must write them back to the same ``[:9]`` slice —
    NOT at ``_actuated_joint_ids_tensor`` positions, which can differ when ``find_joints``
    reorders the names into articulation order.
    """

    jp = joint_pos_9.to(device=env.device, dtype=torch.float32).view(-1)
    jv = joint_vel_9.to(device=env.device, dtype=torch.float32).view(-1)
    if jp.numel() != 9 or jv.numel() != 9:
        raise ValueError(f"Expected 9-d joint_pos/joint_vel, got {jp.numel()} / {jv.numel()}")

    art = runtime_state["articulation"]["robot"]
    art["joint_position"][env_id, :9] = jp
    art["joint_velocity"][env_id, :9] = jv


def reset_then_set_env_from_initial_state(env, recorded_initial_state: dict, env_id: int = 0) -> None:
    """Reset env buffers first, then overwrite sim state from a recorded initial_state dict."""
    device = env.device
    env_ids = torch.tensor([env_id], dtype=torch.int64, device=device)

    # 1) Reset internal env buffers for this env_id.
    if hasattr(env, "_reset_idx"):
        env._reset_idx(env_ids)
        env.scene.write_data_to_sim()
        env.sim.forward()

    # 2) Merge recorded state into current runtime state dict and apply it for env_id.
    runtime_state = env.scene.get_state(is_relative=True)
    _merge_recorded_state_into_runtime_state(runtime_state, recorded_initial_state, env_id=env_id)
    env.scene.reset_to(_slice_scene_state_for_env_ids(runtime_state, env_ids), env_ids, is_relative=True)
    env.scene.write_data_to_sim()
    env.sim.forward()
    # Sync scene's .data from sim so _get_observations() sees the state we just set.
    dt = getattr(env, "physics_dt", 1.0 / 60.0)
    env.scene.update(dt=dt)

    # 3) Align joint targets (avoids a big first-step jump).
    try:
        rec_art = recorded_initial_state.get("articulation", {}).get("robot", {})
        rec_jp = rec_art.get("joint_position", None)
        if isinstance(rec_jp, torch.Tensor):
            # If env exposes actuated ids/targets, set them consistently.
            if hasattr(env, "_actuated_joint_ids_tensor") and getattr(env, "_actuated_joint_ids_tensor") is not None:
                ids = env._actuated_joint_ids_tensor.to(device=device, dtype=torch.long).view(-1)
                if ids.numel() == getattr(env, "_robot_dof_targets", torch.zeros(9, device=device)).shape[-1]:
                    q = rec_jp[0].to(device=device, dtype=torch.float32).view(-1)
                    q_act = q[ids]
                    if hasattr(env, "_robot_dof_targets"):
                        env._robot_dof_targets[env_id] = q_act
            else:
                # Generic fallback: first 9 are the actuated joints in obs order.
                if hasattr(env, "_robot_dof_targets"):
                    env._robot_dof_targets[env_id, :9] = rec_jp[0, :9].to(device=device, dtype=torch.float32)
    except Exception:
        pass

    # 4) Sync controller-side joint state if available (important for correct action interpretation).
    try:
        if hasattr(env, "update_data_in_controller"):
            env.update_data_in_controller()
    except Exception:
        pass


def set_env_from_observation_pen_cup(env, obs: torch.Tensor, env_id: int = 0) -> None:
    """
    pen-cup-specific replay helpers: set scene from an observation.

    Observation layout (56 dims, matches pen_cup_env_learned._get_observations):
    [0:3]   marker_pos       [3:6]   holder_pos      [6:9]   fingertip_pos  [9:12]  link4_pos
    [12:18] marker_rot6d     [18:24] holder_rot6d    [24:30] fingertip_rot6d [30:36] link4_rot6d
    [36:45] joint_pos (9)    [45:54] joint_vel (9)   [54:63] joint_effort (9)
    [63:64] gripper_open     [64:65] stage
    """

    # Indices into the flat 56-d observation (pen-cup only).
    MARKER_POS = slice(0, 3)
    PEN_HOLDER_POS = slice(3, 6)
    MARKER_QUAT = slice(12, 18)
    PEN_HOLDER_QUAT = slice(18, 24)
    JOINT_POS = slice(36, 45)
    JOINT_VEL = slice(45, 54)
    STAGE = slice(64, 65)

    if obs.dim() == 2:
        obs = obs[0]
    assert obs.shape[-1] == 65, f"Expected pen-cup obs dim 65, got {obs.shape[-1]}"

    device = env.device
    obs = obs.to(device=device, dtype=torch.float32)

    state = env.scene.get_state(is_relative=True)

    # Marker: root_pose = pos (3) + quat (4) = 7
    marker_pos = obs[MARKER_POS].clone()
    marker_quat = _rot6d_to_quat_wxyz(obs[MARKER_QUAT].clone())
    marker_root_pose = torch.cat([marker_pos, marker_quat], dim=-1).unsqueeze(0)
    state["rigid_object"]["marker"]["root_pose"][env_id] = marker_root_pose.squeeze(0)
    state["rigid_object"]["marker"]["root_velocity"][env_id] = torch.zeros(6, device=device, dtype=torch.float32)

    # Pen holder: root_pose = pos (3) + quat (4) = 7
    pen_holder_pos = obs[PEN_HOLDER_POS].clone()
    pen_holder_quat = _rot6d_to_quat_wxyz(obs[PEN_HOLDER_QUAT].clone())
    pen_holder_root_pose = torch.cat([pen_holder_pos, pen_holder_quat], dim=-1).unsqueeze(0)
    state["rigid_object"]["pen_holder"]["root_pose"][env_id] = pen_holder_root_pose.squeeze(0)
    state["rigid_object"]["pen_holder"]["root_velocity"][env_id] = torch.zeros(6, device=device, dtype=torch.float32)

    # Robot: only joint position and velocity from obs; root unchanged
    joint_pos = obs[JOINT_POS].clone().to(device=device, dtype=torch.float32).unsqueeze(0)
    joint_vel = obs[JOINT_VEL].clone().unsqueeze(0)
    # Scene may have more than 9 joints (e.g. gripper); overwrite the 9 actuated joints.
    _set_robot_actuated_joint_state_from_9d(env, state, joint_pos.squeeze(0), joint_vel.squeeze(0), env_id=env_id)

    env_ids = torch.tensor([env_id], dtype=torch.int64, device=device)
    env.scene.reset_to(_slice_scene_state_for_env_ids(state, env_ids), env_ids, is_relative=True)

    stage_value = obs[STAGE].clone().to(device=device, dtype=torch.float32)
    if env.current_stage.dim() == 2:
        env.current_stage[env_id, 0] = stage_value.squeeze()
    else:
        env.current_stage[env_id] = stage_value.squeeze()


def reset_then_set_env_from_observation_pen_cup(env, obs: torch.Tensor, env_id: int = 0) -> None:
    """Reset env buffers first, then overwrite sim state from obs.

    Use this when "jumping" into a trajectory (e.g., random start timestep) so that
    internal buffers (joint targets, episode counters, program_runtime stage tracking, etc.)
    are reinitialized before we teleport the scene to the recorded observation.
    """
    device = env.device
    env_ids = torch.tensor([env_id], dtype=torch.int64, device=device)

    # 1) Reset internal env buffers for this env_id (mirrors DirectRLEnv.step reset path).
    if hasattr(env, "_reset_idx"):
        env._reset_idx(env_ids)
        # Update articulation kinematics similar to DirectRLEnv.reset()/step().
        env.scene.write_data_to_sim()
        env.sim.forward()

    # 2) Teleport scene to recorded observation.
    set_env_from_observation_pen_cup(env, obs, env_id=env_id)

    # 3) Make sure joint targets match the teleported joint state (avoids a big first-step jump).
    if obs.dim() == 2:
        obs_1d = obs[0]
    else:
        obs_1d = obs
    joint_pos_9 = obs_1d[36:45].to(device=device, dtype=torch.float32)
    if hasattr(env, "_robot") and hasattr(env._robot, "set_joint_position_target"):
        ids = env._actuated_joint_ids_tensor.to(device=device, dtype=torch.long).view(-1)
        target = joint_pos_9[ids] if ids.max() < 9 else joint_pos_9
        if hasattr(env, "_robot_dof_targets"):
            env._robot_dof_targets[env_id] = target
        env._robot.set_joint_position_target(target.unsqueeze(0), joint_ids=ids, env_ids=env_ids)

    env.scene.write_data_to_sim()
    env.sim.forward()
    # Sync controller-side caches to teleported state (important for correct first-step actions).
    env.update_data_in_controller()

    # 4) Best-effort: align program_runtime stage index to recorded stage (if present).
    stage_int = int(torch.round(obs_1d[64:65].view(-1)[0]).item())
    la = env.program_runtime.compiled_program
    la._batch_stage_indices[env_id] = stage_int

    env._prev_stage_indices[env_id] = stage_int

    env._highest_stage_per_env[env_id] = torch.maximum(
        env._highest_stage_per_env[env_id], torch.tensor(stage_int, device=device, dtype=env._highest_stage_per_env.dtype)
    )


def reset_then_set_env_from_observation_stack2(env, obs: torch.Tensor, env_id: int = 0) -> None:
    """Reset env buffers first, then overwrite sim state from a Stack2 observation."""
    device = env.device
    env_ids = torch.tensor([env_id], dtype=torch.int64, device=device)

    # 1) Reset internal env buffers for this env_id (mirrors DirectRLEnv.step reset path).
    if hasattr(env, "_reset_idx"):
        env._reset_idx(env_ids)
        env.scene.write_data_to_sim()
        env.sim.forward()

    # 2) Teleport scene to recorded observation.
    set_env_from_first_observation_stack2(env, obs, env_id=env_id)
    env.scene.write_data_to_sim()
    env.sim.forward()
    # Sync scene's .data / scene.state from sim BEFORE update_data_in_controller reads them.
    dt = getattr(env, "physics_dt", 1.0 / 60.0)
    env.scene.update(dt=dt)
    # Sync controller-side caches to teleported state (important for correct first-step actions).
    try:
        if hasattr(env, "update_data_in_controller"):
            env.update_data_in_controller()
    except Exception:
        pass

    # 3) Make sure joint targets match the teleported joint state (avoids a big first-step jump).
    try:
        if obs.dim() == 2:
            obs_1d = obs[0]
        else:
            obs_1d = obs
        joint_pos = obs_1d[slice(28, 37)].to(device=device, dtype=torch.float32)
        if hasattr(env, "_robot_dof_targets"):
            env._robot_dof_targets[env_id, :9] = joint_pos
        if hasattr(env, "_robot") and hasattr(env._robot, "set_joint_position_target"):
            target_row = None
            if hasattr(env, "_robot_dof_targets"):
                target_row = env._robot_dof_targets[env_id].clone()
            else:
                target_row = torch.zeros((env._robot.num_joints,), device=device, dtype=torch.float32)
                target_row[:9] = joint_pos
            env._robot.set_joint_position_target(target_row.unsqueeze(0), env_ids=env_ids)
    except Exception:
        pass

    # 4) Best-effort: align program_runtime stage index to recorded stage (if present).
    try:
        if obs.dim() == 2:
            obs_1d = obs[0]
        else:
            obs_1d = obs
        stage_int = int(torch.round(obs_1d[slice(55, 56)].view(-1)[0]).item())
        if hasattr(env, "program_runtime") and hasattr(env.program_runtime, "compiled_program"):
            la = env.program_runtime.compiled_program
            if hasattr(la, "_batch_stage_indices") and la._batch_stage_indices is not None:
                la._batch_stage_indices[env_id] = stage_int
        if hasattr(env, "_prev_stage_indices"):
            env._prev_stage_indices[env_id] = stage_int
        if hasattr(env, "_highest_stage_per_env"):
            env._highest_stage_per_env[env_id] = torch.maximum(
                env._highest_stage_per_env[env_id],
                torch.tensor(stage_int, device=device, dtype=env._highest_stage_per_env.dtype),
            )
    except Exception:
        pass


def reset_then_set_env_from_observation_fr3_finray_drawer(env, obs: torch.Tensor, env_id: int = 0) -> None:
    """Reset env buffers first, then overwrite sim state from a FR3FinRay drawer observation."""
    device = env.device
    env_ids = torch.tensor([env_id], dtype=torch.int64, device=device)

    # 1) Reset internal env buffers for this env_id (mirrors DirectRLEnv.step reset path).
    if hasattr(env, "_reset_idx"):
        env._reset_idx(env_ids)
        env.scene.write_data_to_sim()
        env.sim.forward()

    # 2) Teleport scene to recorded observation.
    set_env_from_first_observation_fr3_finray_drawer(env, obs, env_id=env_id)
    env.scene.write_data_to_sim()
    env.sim.forward()
    # Sync scene's .data / scene.state from sim BEFORE update_data_in_controller reads them.
    dt = getattr(env, "physics_dt", 1.0 / 60.0)
    env.scene.update(dt=dt)
    # Sync controller-side caches to teleported state (important for correct first-step actions).
    try:
        if hasattr(env, "update_data_in_controller"):
            env.update_data_in_controller()
    except Exception:
        pass

    # 3) Make sure joint targets match the teleported joint state (avoids a big first-step jump).
    try:
        if obs.dim() == 2:
            obs_1d = obs[0]
        else:
            obs_1d = obs
        joint_pos = obs_1d[slice(21, 30)].to(device=device, dtype=torch.float32)
        if hasattr(env, "_robot_dof_targets"):
            env._robot_dof_targets[env_id, :9] = joint_pos
        if hasattr(env, "_robot") and hasattr(env._robot, "set_joint_position_target"):
            target_row = None
            if hasattr(env, "_robot_dof_targets"):
                target_row = env._robot_dof_targets[env_id].clone()
            else:
                target_row = torch.zeros((env._robot.num_joints,), device=device, dtype=torch.float32)
                target_row[:9] = joint_pos
            env._robot.set_joint_position_target(target_row.unsqueeze(0), env_ids=env_ids)
    except Exception:
        pass

    # 4) Best-effort: align program_runtime stage index to recorded stage (if present).
    try:
        if obs.dim() == 2:
            obs_1d = obs[0]
        else:
            obs_1d = obs
        stage_int = int(torch.round(obs_1d[slice(48, 49)].view(-1)[0]).item())
        if hasattr(env, "program_runtime") and hasattr(env.program_runtime, "compiled_program"):
            la = env.program_runtime.compiled_program
            if hasattr(la, "_batch_stage_indices") and la._batch_stage_indices is not None:
                la._batch_stage_indices[env_id] = stage_int
        if hasattr(env, "_prev_stage_indices"):
            env._prev_stage_indices[env_id] = stage_int
        if hasattr(env, "_highest_stage_per_env"):
            env._highest_stage_per_env[env_id] = torch.maximum(
                env._highest_stage_per_env[env_id],
                torch.tensor(stage_int, device=device, dtype=env._highest_stage_per_env.dtype),
            )
    except Exception:
        pass


def reset_then_set_env_from_observation_fr3_finray_can(env, obs: torch.Tensor, env_id: int = 0) -> None:
    """Reset env buffers first, then overwrite sim state from a FR3FinRay can observation.

    Observation layout for full_state (56 dims, matches ``FR3FinRayCanEnvLearned._get_observations``):
        [0:3]   can_pos            [3:6]   fingertip_pos      [6:9]   link4_pos
        [9:15]  can_rot6d          [15:21] fingertip_rot6d    [21:27] link4_rot6d
        [27:36] joint_pos (9)      [36:45] joint_vel (9)      [45:54] joint_effort (9)
        [54:55] gripper_open       [55:56] stage
    """
    device = env.device
    env_ids = torch.tensor([env_id], dtype=torch.int64, device=device)

    # 1) Reset internal env buffers for this env_id (mirrors DirectRLEnv.step reset path).
    if hasattr(env, "_reset_idx"):
        env._reset_idx(env_ids)
        env.scene.write_data_to_sim()
        env.sim.forward()

    # Accept either (56,) or (1, 56) tensors.
    if obs.dim() == 2:
        obs_1d = obs[0]
    else:
        obs_1d = obs
    obs_1d = obs_1d.to(device=device, dtype=torch.float32)
    if obs_1d.shape[-1] < 56:
        raise ValueError(f"Expected FR3FinRay can full_state obs dim >= 56, got {obs_1d.shape[-1]}")

    CAN_POS = slice(0, 3)
    CAN_ROT6D = slice(9, 15)
    JOINT_POS = slice(27, 36)
    JOINT_VEL = slice(36, 45)
    STAGE = slice(55, 56)

    # 2) Teleport scene to recorded observation.
    state = env.scene.get_state(is_relative=True)
    try:
        can_pos = obs_1d[CAN_POS].clone()
        can_quat = _rot6d_to_quat_wxyz(obs_1d[CAN_ROT6D].clone())
        can_root = torch.cat([can_pos, can_quat], dim=-1)
        state["rigid_object"]["can"]["root_pose"][env_id] = can_root
        state["rigid_object"]["can"]["root_velocity"][env_id] = torch.zeros(
            6, device=device, dtype=torch.float32
        )
    except Exception:
        pass

    try:
        _set_robot_actuated_joint_state_from_9d(
            env, state, obs_1d[JOINT_POS].clone(), obs_1d[JOINT_VEL].clone(), env_id=env_id
        )
    except Exception:
        pass

    env.scene.reset_to(_slice_scene_state_for_env_ids(state, env_ids), env_ids, is_relative=True)
    env.scene.write_data_to_sim()
    env.sim.forward()

    # Sync scene caches before controller-side reads.
    dt = getattr(env, "physics_dt", 1.0 / 60.0)
    env.scene.update(dt=dt)
    try:
        if hasattr(env, "update_data_in_controller"):
            env.update_data_in_controller()
    except Exception:
        pass

    # 3) Make sure joint targets match the teleported joint state.
    try:
        joint_pos = obs_1d[JOINT_POS].to(device=device, dtype=torch.float32)
        if hasattr(env, "_robot_dof_targets"):
            env._robot_dof_targets[env_id, :9] = joint_pos
        if hasattr(env, "_robot") and hasattr(env._robot, "set_joint_position_target"):
            if hasattr(env, "_robot_dof_targets"):
                target_row = env._robot_dof_targets[env_id].clone()
            else:
                target_row = torch.zeros((env._robot.num_joints,), device=device, dtype=torch.float32)
                target_row[:9] = joint_pos
            env._robot.set_joint_position_target(target_row.unsqueeze(0), env_ids=env_ids)
    except Exception:
        pass

    # 4) Best-effort: align program_runtime stage index to recorded stage (if present).
    try:
        stage_int = int(torch.round(obs_1d[STAGE].view(-1)[0]).item())
        if hasattr(env, "program_runtime") and hasattr(env.program_runtime, "compiled_program"):
            la = env.program_runtime.compiled_program
            if hasattr(la, "_batch_stage_indices") and la._batch_stage_indices is not None:
                la._batch_stage_indices[env_id] = stage_int
        if hasattr(env, "_prev_stage_indices"):
            env._prev_stage_indices[env_id] = stage_int
        if hasattr(env, "_highest_stage_per_env"):
            env._highest_stage_per_env[env_id] = torch.maximum(
                env._highest_stage_per_env[env_id],
                torch.tensor(stage_int, device=device, dtype=env._highest_stage_per_env.dtype),
            )
    except Exception:
        pass


def reset_then_set_env_from_observation_stack3(env, obs: torch.Tensor, env_id: int = 0) -> None:
    """Reset env buffers first, then overwrite sim state from a FR3FinRay Stack-3 observation.

    Observation layout for full_state (74 dims, matches FR3FinRayStack3EnvLearned._get_observations):
        [0:3]   cube1_pos          [3:6]   cube2_pos         [6:9]   cube3_pos
        [9:12]  fingertip_pos      [12:15] link4_pos
        [15:21] cube1_rot6d        [21:27] cube2_rot6d       [27:33] cube3_rot6d
        [33:39] fingertip_rot6d    [39:45] link4_rot6d
        [45:54] joint_pos (9)      [54:63] joint_vel (9)     [63:72] joint_effort (9)
        [72:73] gripper_open       [73:74] stage
    """
    device = env.device
    env_ids = torch.tensor([env_id], dtype=torch.int64, device=device)

    # 1) Reset internal env buffers for this env_id (mirrors DirectRLEnv.step reset path).
    if hasattr(env, "_reset_idx"):
        env._reset_idx(env_ids)
        env.scene.write_data_to_sim()
        env.sim.forward()

    # Accept either (74,) or (1, 74) tensors.
    if obs.dim() == 2:
        obs_1d = obs[0]
    else:
        obs_1d = obs

    # 2) Teleport scene to recorded observation.
    # Indices for full_state observation as defined in FR3FinRayStack3EnvLearned._get_observations.
    CUBE1_POS = slice(0, 3)
    CUBE2_POS = slice(3, 6)
    CUBE3_POS = slice(6, 9)
    FINGERTIP_POS = slice(9, 12)   # unused for teleport (kinematically determined)
    LINK4_POS = slice(12, 15)      # unused for teleport (kinematically determined)

    CUBE1_ROT6D = slice(15, 21)
    CUBE2_ROT6D = slice(21, 27)
    CUBE3_ROT6D = slice(27, 33)
    FINGERTIP_ROT6D = slice(33, 39)  # unused for teleport
    LINK4_ROT6D = slice(39, 45)      # unused for teleport

    JOINT_POS = slice(45, 54)
    JOINT_VEL = slice(54, 63)
    STAGE = slice(73, 74)

    obs_1d = obs_1d.to(device=device, dtype=torch.float32)

    state = env.scene.get_state(is_relative=True)

    # Cubes: root_pose = pos (3) + quat (4) = 7, velocities set to zero.
    def _set_cube(name: str, pos_slice: slice, rot_slice: slice) -> None:
        pos = obs_1d[pos_slice].clone()
        quat = _rot6d_to_quat_wxyz(obs_1d[rot_slice].clone())
        root = torch.cat([pos, quat], dim=-1)
        state["rigid_object"][name]["root_pose"][env_id] = root
        state["rigid_object"][name]["root_velocity"][env_id] = torch.zeros(
            6, device=device, dtype=torch.float32
        )

    try:
        _set_cube("cube_1", CUBE1_POS, CUBE1_ROT6D)
        _set_cube("cube_2", CUBE2_POS, CUBE2_ROT6D)
        _set_cube("cube_3", CUBE3_POS, CUBE3_ROT6D)
    except Exception:
        # Best-effort: if scene layout changes, don't crash replay.
        pass

    # Robot: overwrite first 9 articulation joints (see _set_robot_actuated_joint_state_from_9d).
    joint_pos_9 = obs_1d[JOINT_POS].clone()
    joint_vel_9 = obs_1d[JOINT_VEL].clone()
    try:
        _set_robot_actuated_joint_state_from_9d(
            env, state, joint_pos_9, joint_vel_9, env_id=env_id
        )
    except Exception:
        pass

    env.scene.reset_to(_slice_scene_state_for_env_ids(state, env_ids), env_ids, is_relative=True)
    env.scene.write_data_to_sim()
    env.sim.forward()

    # Sync scene's .data / scene.state from sim BEFORE update_data_in_controller reads them.
    dt = getattr(env, "physics_dt", 1.0 / 60.0)
    env.scene.update(dt=dt)

    # Sync controller-side caches to teleported state (important for correct first-step actions).
    try:
        if hasattr(env, "update_data_in_controller"):
            env.update_data_in_controller()
    except Exception:
        pass

    # 3) Make sure joint targets match the teleported joint state (avoids a big first-step jump).
    try:
        joint_pos = obs_1d[JOINT_POS].to(device=device, dtype=torch.float32)
        if hasattr(env, "_robot_dof_targets"):
            env._robot_dof_targets[env_id, :9] = joint_pos
        if hasattr(env, "_robot") and hasattr(env._robot, "set_joint_position_target"):
            if hasattr(env, "_robot_dof_targets"):
                target_row = env._robot_dof_targets[env_id].clone()
            else:
                target_row = torch.zeros(
                    (env._robot.num_joints,), device=device, dtype=torch.float32
                )
                target_row[:9] = joint_pos
            env._robot.set_joint_position_target(target_row.unsqueeze(0), env_ids=env_ids)
    except Exception:
        pass

    # 4) Best-effort: align program_runtime stage index to recorded stage (if present).
    try:
        stage_int = int(torch.round(obs_1d[STAGE].view(-1)[0]).item())
        if hasattr(env, "program_runtime") and hasattr(env.program_runtime, "compiled_program"):
            la = env.program_runtime.compiled_program
            if hasattr(la, "_batch_stage_indices") and la._batch_stage_indices is not None:
                la._batch_stage_indices[env_id] = stage_int
        if hasattr(env, "_prev_stage_indices"):
            env._prev_stage_indices[env_id] = stage_int
        if hasattr(env, "_highest_stage_per_env"):
            env._highest_stage_per_env[env_id] = torch.maximum(
                env._highest_stage_per_env[env_id],
                torch.tensor(stage_int, device=device, dtype=env._highest_stage_per_env.dtype),
            )
    except Exception:
        pass


def _rot6d_to_quat_wxyz(rot6d: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation to wxyz quaternion. Uses kuafu.mpc.rotation_6d_to_quat."""
    return rotation_6d_to_quat(rot6d.to(dtype=torch.float32)).view(-1)


def reset_then_set_env_from_observation_hang_cup(env, obs: torch.Tensor, env_id: int = 0) -> None:
    """Reset env buffers first, then overwrite sim state from a FR3FinRay hang-cup observation.

    Observation layout for full_state (65 dims, matches ``FR3FinRayHangCupEnvLearned._get_observations``):
        [0:3]   mug_pos (mug_center_frame: root + R@[0,0,0.04])
        [3:6]   hanger_pos (branch_1_tip_frame: root + R@[0.1,0,0.28])
        [6:9]   fingertip_pos       [9:12]  link4_pos
        [12:18] mug_rot6d           [18:24] hanger_rot6d
        [24:30] fingertip_rot6d     [30:36] link4_rot6d
        [36:45] joint_pos (9)       [45:54] joint_vel (9)
        [54:63] joint_effort (9)    [63:64] gripper_open
        [64:65] stage
    """
    device = env.device
    env_ids = torch.tensor([env_id], dtype=torch.int64, device=device)

    # 1) Reset internal env buffers for this env_id.
    if hasattr(env, "_reset_idx"):
        env._reset_idx(env_ids)
        env.scene.write_data_to_sim()
        env.sim.forward()

    if obs.dim() == 2:
        obs_1d = obs[0]
    else:
        obs_1d = obs
    obs_1d = obs_1d.to(device=device, dtype=torch.float32)
    if obs_1d.shape[-1] < 65:
        raise ValueError(f"Expected hang_cup full_state obs dim >= 65, got {obs_1d.shape[-1]}")

    MUG_POS      = slice(0, 3)
    HANGER_POS   = slice(3, 6)
    MUG_ROT6D    = slice(12, 18)
    HANGER_ROT6D = slice(18, 24)
    JOINT_POS    = slice(36, 45)
    JOINT_VEL    = slice(45, 54)
    STAGE        = slice(64, 65)

    # 2) Teleport scene to recorded observation.
    state = env.scene.get_state(is_relative=True)

    try:
        # mug_pos is mug_center_frame = root + R@[0,0,0.04]; undo offset to get root pos.
        mug_frame_pos = obs_1d[MUG_POS].clone()
        mug_rot6d = obs_1d[MUG_ROT6D].clone()
        mug_quat = _rot6d_to_quat_wxyz(mug_rot6d)
        offset_mug = torch.tensor([0.0, 0.0, 0.04], device=device, dtype=torch.float32)
        mug_root_pos = mug_frame_pos - _quat_rotate_wxyz(mug_quat, offset_mug)
        state["rigid_object"]["mug"]["root_pose"][env_id] = torch.cat([mug_root_pos, mug_quat])
        state["rigid_object"]["mug"]["root_velocity"][env_id] = torch.zeros(6, device=device, dtype=torch.float32)
    except Exception:
        pass

    try:
        # hanger_pos is branch_1_tip_frame = root + R@[0.1,0,0.28]; undo offset and frame rotation.
        hanger_frame_pos = obs_1d[HANGER_POS].clone()
        hanger_rot6d = obs_1d[HANGER_ROT6D].clone()
        hanger_frame_quat = _rot6d_to_quat_wxyz(hanger_rot6d)
        offset_hanger = torch.tensor([0.1, 0.0, 0.28], device=device, dtype=torch.float32)
        offset_hanger_quat = torch.tensor(
            [0.696364240, 0.122787804, -0.122787804, -0.696364240],
            device=device,
            dtype=torch.float32,
        )
        hanger_root_quat = _quat_mul_wxyz(hanger_frame_quat, _quat_conj_wxyz(offset_hanger_quat))
        hanger_root_quat = hanger_root_quat / torch.clamp(torch.linalg.norm(hanger_root_quat), min=1.0e-8)
        hanger_root_pos = hanger_frame_pos - _quat_rotate_wxyz(hanger_root_quat, offset_hanger)
        state["rigid_object"]["cup_hanger"]["root_pose"][env_id] = torch.cat([hanger_root_pos, hanger_root_quat])
        state["rigid_object"]["cup_hanger"]["root_velocity"][env_id] = torch.zeros(6, device=device, dtype=torch.float32)
    except Exception:
        pass

    try:
        _set_robot_actuated_joint_state_from_9d(
            env, state, obs_1d[JOINT_POS].clone(), obs_1d[JOINT_VEL].clone(), env_id=env_id
        )
    except Exception:
        pass

    env.scene.reset_to(_slice_scene_state_for_env_ids(state, env_ids), env_ids, is_relative=True)
    env.scene.write_data_to_sim()
    env.sim.forward()

    dt = getattr(env, "physics_dt", 1.0 / 60.0)
    env.scene.update(dt=dt)
    try:
        if hasattr(env, "update_data_in_controller"):
            env.update_data_in_controller()
    except Exception:
        pass

    # 3) Sync joint targets to teleported state.
    try:
        joint_pos = obs_1d[JOINT_POS].to(device=device, dtype=torch.float32)
        if hasattr(env, "_robot_dof_targets"):
            env._robot_dof_targets[env_id, :9] = joint_pos
        if hasattr(env, "_robot") and hasattr(env._robot, "set_joint_position_target"):
            if hasattr(env, "_robot_dof_targets"):
                target_row = env._robot_dof_targets[env_id].clone()
            else:
                target_row = torch.zeros((env._robot.num_joints,), device=device, dtype=torch.float32)
                target_row[:9] = joint_pos
            env._robot.set_joint_position_target(target_row.unsqueeze(0), env_ids=env_ids)
    except Exception:
        pass

    # 4) Best-effort: align program_runtime stage index to recorded stage.
    try:
        stage_int = int(torch.round(obs_1d[STAGE].view(-1)[0]).item())
        if hasattr(env, "program_runtime") and hasattr(env.program_runtime, "compiled_program"):
            la = env.program_runtime.compiled_program
            if hasattr(la, "_batch_stage_indices") and la._batch_stage_indices is not None:
                la._batch_stage_indices[env_id] = stage_int
        if hasattr(env, "_prev_stage_indices"):
            env._prev_stage_indices[env_id] = stage_int
        if hasattr(env, "_highest_stage_per_env"):
            env._highest_stage_per_env[env_id] = torch.maximum(
                env._highest_stage_per_env[env_id],
                torch.tensor(stage_int, device=device, dtype=env._highest_stage_per_env.dtype),
            )
    except Exception:
        pass


def reset_then_set_env_from_observation_fr3_finray_estop_open(env, obs: torch.Tensor, env_id: int = 0) -> None:
    """Reset env buffers first, then overwrite sim state from a FR3FinRay EStop Open observation.

    Observation layout for full_state (65 dims, matches
    ``FR3FinRayEStopClosedEnvLearned._get_observations``):
        [0:3]   cap_tip_pos       [3:6]   base_pos
        [6:9]   fingertip_pos     [9:12]  link4_pos
        [12:18] cap_tip_rot6d     [18:24] base_rot6d
        [24:30] fingertip_rot6d   [30:36] link4_rot6d
        [36:45] joint_pos (9)     [45:54] joint_vel (9)     [54:63] joint_effort (9)
        [63:64] gripper_open      [64:65] stage
    """
    device = env.device
    env_ids = torch.tensor([env_id], dtype=torch.int64, device=device)

    if hasattr(env, "_reset_idx"):
        env._reset_idx(env_ids)
        env.scene.write_data_to_sim()
        env.sim.forward()

    if obs.dim() == 2:
        obs_1d = obs[0]
    else:
        obs_1d = obs
    obs_1d = obs_1d.to(device=device, dtype=torch.float32)
    if obs_1d.shape[-1] < 65:
        raise ValueError(f"Expected FR3FinRay EStop Open full_state obs dim >= 65, got {obs_1d.shape[-1]}")

    CAP_TIP_POS = slice(0, 3)
    BASE_POS = slice(3, 6)
    CAP_TIP_ROT6D = slice(12, 18)
    BASE_ROT6D = slice(18, 24)
    JOINT_POS = slice(36, 45)
    JOINT_VEL = slice(45, 54)
    STAGE = slice(64, 65)

    state = env.scene.get_state(is_relative=True)

    press_pos = torch.tensor(0.0, device=device, dtype=torch.float32)
    twist_pos = torch.tensor(0.0, device=device, dtype=torch.float32)

    try:
        base_pos = obs_1d[BASE_POS].clone()
        base_quat = _rot6d_to_quat_wxyz(obs_1d[BASE_ROT6D].clone())
        cap_tip_pos = obs_1d[CAP_TIP_POS].clone()
        cap_tip_quat = _rot6d_to_quat_wxyz(obs_1d[CAP_TIP_ROT6D].clone())

        base_frame_offset = torch.tensor([0.0, 0.0, 0.06], device=device, dtype=torch.float32)
        estop_root_pos = base_pos - _quat_rotate_wxyz(base_quat, base_frame_offset)
        estop_root = torch.cat([estop_root_pos, base_quat], dim=-1)

        estop_state = state.get("articulation", {}).get("estop")
        if estop_state is not None:
            estop_state["root_pose"][env_id] = estop_root
            if "root_velocity" in estop_state:
                estop_state["root_velocity"][env_id] = torch.zeros(6, device=device, dtype=torch.float32)

            press_joint_id = getattr(env, "press_joint_id", None)
            twist_joint_id = getattr(env, "twist_joint_id", None)
            if press_joint_id is None and hasattr(env, "estop"):
                press_joint_id = env.estop.find_joints(env.cfg.semantic_cfg.press_joint_name)[0][0]
            if twist_joint_id is None and hasattr(env, "estop"):
                twist_joint_id = env.estop.find_joints(env.cfg.semantic_cfg.twist_joint_name)[0][0]

            base_z = _quat_rotate_wxyz(
                base_quat,
                torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32),
            )
            # cap_tip z offset relative to base_frame:
            # (base height 0.073 + cap z offset -0.005 + cap-tip offset 0.045) - 0.06 = 0.053
            press_pos = torch.dot(cap_tip_pos - base_pos, base_z) - 0.053

            rel_cap_quat = _quat_mul_wxyz(_quat_conj_wxyz(base_quat), cap_tip_quat)
            twist_pos = _yaw_from_quat_wxyz(rel_cap_quat)

            press_lower, press_upper = -0.01, 0.0
            twist_lower, twist_upper = -math.radians(100.0), 0.0
            try:
                limits = env.estop.data.joint_pos_limits[0]
                press_lower = float(limits[press_joint_id, 0].item())
                press_upper = float(limits[press_joint_id, 1].item())
                twist_lower = float(limits[twist_joint_id, 0].item())
                twist_upper = float(limits[twist_joint_id, 1].item())
            except Exception:
                pass

            press_pos = torch.clamp(press_pos, min=press_lower, max=press_upper)
            twist_pos = torch.clamp(twist_pos, min=twist_lower, max=twist_upper)

            if press_joint_id is not None and twist_joint_id is not None:
                estop_state["joint_position"][env_id, press_joint_id] = press_pos
                estop_state["joint_position"][env_id, twist_joint_id] = twist_pos
                estop_state["joint_velocity"][env_id, press_joint_id] = 0.0
                estop_state["joint_velocity"][env_id, twist_joint_id] = 0.0
    except Exception:
        pass

    try:
        _set_robot_actuated_joint_state_from_9d(
            env, state, obs_1d[JOINT_POS].clone(), obs_1d[JOINT_VEL].clone(), env_id=env_id
        )
    except Exception:
        pass

    env.scene.reset_to(_slice_scene_state_for_env_ids(state, env_ids), env_ids, is_relative=True)
    env.scene.write_data_to_sim()
    env.sim.forward()

    dt = getattr(env, "physics_dt", 1.0 / 60.0)
    env.scene.update(dt=dt)
    try:
        if hasattr(env, "update_data_in_controller"):
            env.update_data_in_controller()
    except Exception:
        pass

    try:
        joint_pos = obs_1d[JOINT_POS].to(device=device, dtype=torch.float32)
        if hasattr(env, "_robot_dof_targets"):
            env._robot_dof_targets[env_id, :9] = joint_pos
        if hasattr(env, "_robot") and hasattr(env._robot, "set_joint_position_target"):
            if hasattr(env, "_robot_dof_targets"):
                target_row = env._robot_dof_targets[env_id].clone()
            else:
                target_row = torch.zeros((env._robot.num_joints,), device=device, dtype=torch.float32)
                target_row[:9] = joint_pos
            env._robot.set_joint_position_target(target_row.unsqueeze(0), env_ids=env_ids)
    except Exception:
        pass

    try:
        press_joint_id = getattr(env, "press_joint_id", None)
        twist_joint_id = getattr(env, "twist_joint_id", None)
        if press_joint_id is None and hasattr(env, "estop"):
            press_joint_id = env.estop.find_joints(env.cfg.semantic_cfg.press_joint_name)[0][0]
        if twist_joint_id is None and hasattr(env, "estop"):
            twist_joint_id = env.estop.find_joints(env.cfg.semantic_cfg.twist_joint_name)[0][0]
        if press_joint_id is not None and twist_joint_id is not None and hasattr(env, "estop"):
            estop_targets = torch.stack([press_pos, twist_pos]).view(1, 2)
            env.estop.set_joint_position_target(
                estop_targets,
                joint_ids=[press_joint_id, twist_joint_id],
                env_ids=env_ids,
            )

            if hasattr(env.estop, "_ensure_state_initialized"):
                env.estop._ensure_state_initialized()
            is_closed_tensor = getattr(env.estop, "_is_closed", None)
            if is_closed_tensor is not None:
                sem = getattr(getattr(env, "cfg", None), "semantic_cfg", None)
                latched = float(getattr(sem, "latched_position", 0.008))
                unlock_angle = float(getattr(sem, "unlock_twist_angle_rad", math.radians(60.0)))
                unlock_tol = float(getattr(sem, "unlock_tolerance_rad", 0.05))
                is_latched = press_pos <= -(0.75 * latched)
                is_unlocked = twist_pos <= -(unlock_angle - unlock_tol)
                is_closed = is_latched & ~is_unlocked
                is_closed_tensor[env_id] = bool(is_closed.item())
                if hasattr(env, "_unlocked_once"):
                    env._unlocked_once[env_id] = not bool(is_closed.item())
    except Exception:
        pass

    try:
        stage_int = int(torch.round(obs_1d[STAGE].view(-1)[0]).item())
        if hasattr(env, "program_runtime") and hasattr(env.program_runtime, "compiled_program"):
            la = env.program_runtime.compiled_program
            if hasattr(la, "_batch_stage_indices") and la._batch_stage_indices is not None:
                la._batch_stage_indices[env_id] = stage_int
        if hasattr(env, "_prev_stage_indices"):
            env._prev_stage_indices[env_id] = stage_int
        if hasattr(env, "_highest_stage_per_env"):
            env._highest_stage_per_env[env_id] = torch.maximum(
                env._highest_stage_per_env[env_id],
                torch.tensor(stage_int, device=device, dtype=env._highest_stage_per_env.dtype),
            )
    except Exception:
        pass


def get_state_setter(env_name: str):
    """Return a callable (env, obs, env_id) -> None for initializing env from obs, or None."""
    env_name_lower = str(env_name).lower()
    if "pen-cup" in env_name_lower or "pen_cup" in env_name_lower:
        return reset_then_set_env_from_observation_pen_cup
    if "stack-2" in env_name_lower or "stack2" in env_name_lower:
        return reset_then_set_env_from_observation_stack2
    if "stack-3" in env_name_lower or "stack3" in env_name_lower:
        return reset_then_set_env_from_observation_stack3
    if "right_can" in env_name_lower or "-can" in env_name_lower:
        return reset_then_set_env_from_observation_fr3_finray_can
    if "table-franka-drawer" in env_name_lower or "table-gen3-drawer" in env_name_lower:
        return reset_then_set_env_from_observation_fr3_finray_drawer
    if "hang-cup" in env_name_lower or "hang_cup" in env_name_lower:
        return reset_then_set_env_from_observation_hang_cup
    if (
        "estop-open" in env_name_lower
        or "estop_open" in env_name_lower
        or "estop-close" in env_name_lower
        or "estop_close" in env_name_lower
    ):
        return reset_then_set_env_from_observation_fr3_finray_estop_open
    raise ValueError(f"Unknown state_setter mode: {env_name}")
