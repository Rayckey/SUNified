import os
from collections.abc import Sequence
from typing import Optional

from isaaclab.sensors.frame_transformer.frame_transformer import FrameTransformer
import torch
import time
from pxr import Usd, UsdGeom, UsdPhysics

from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObjectCfg, RigidObject
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.managers.recorder_manager import (
    RecorderManagerBaseCfg,
    RecorderTerm,
    DatasetExportMode,
)
from isaaclab.managers.manager_term_cfg import RecorderTermCfg


# -----------------------------------------------------------------------------
# Recorder terms for BaseLearnedEnv (DirectRLEnv): use scene, obs_buf, actions.
# -----------------------------------------------------------------------------


class LearnedEnvInitialStateRecorder(RecorderTerm):
    """Records scene state at the end of reset (initial state of each episode)."""

    def record_post_reset(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = list(range(self._env.num_envs))
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.tolist()
        state = self._env.scene.get_state(is_relative=True)
        # Index state dict by env_ids for per-env episode data
        def _index(v):
            if isinstance(v, dict):
                return {k: _index(x) for k, x in v.items()}
            if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] == self._env.num_envs:
                return v[env_ids]
            return v
        return "initial_state", _index(state)


class LearnedEnvPostStepStateRecorder(RecorderTerm):
    """Records scene state at the end of each step (for replay / debugging)."""

    def record_post_step(self):
        return "states", self._env.scene.get_state(is_relative=True)


class LearnedEnvPreStepActionRecorder(RecorderTerm):
    """Records the action applied this step (set by env as _current_step_action before record_pre_step)."""

    def record_pre_step(self):
        return None, None


class LearnedEnvPreStepJointTargetsRecorder(RecorderTerm):
    """Records absolute joint position targets [arm(7), grip(2)] for every mode."""

    def record_pre_step(self):
        return "actions/absolute_joint", self._env._logged_joint_targets


class LearnedEnvPreStepRelativeJointRecorder(RecorderTerm):
    """Records normalized incremental joint actions [arm(7), grip(2)] for every mode."""

    def record_pre_step(self):
        return "actions/relative_joint", self._env._logged_relative_joint


class LearnedEnvPreStepEERecorder(RecorderTerm):
    """Records absolute EE target [pos(3), rot6d(6), grip(2)]. None (skipped) in joint modes."""

    def record_pre_step(self):
        if self._env._logged_absolute_ee is None:
            return None, None
        return "actions/absolute_ee", self._env._logged_absolute_ee


class LearnedEnvPreStepRelativeEERecorder(RecorderTerm):
    """Records relative EE action [pos_delta(3), aa_delta(3), grip(2)]. None (skipped) in joint modes."""

    def record_pre_step(self):
        if self._env._logged_relative_ee is None:
            return None, None
        return "actions/relative_ee", self._env._logged_relative_ee


class LearnedEnvPreStepPolicyObsRecorder(RecorderTerm):
    """Records policy observation at the start of the step (obs before applying action)."""

    def record_pre_step(self):
        return "obs/state", self._env.obs_buf["policy"]


class LearnedEnvPostStepPolicyObsRecorder(RecorderTerm):
    """Records policy observation at the end of the step (next_obs for BC)."""

    def record_post_step(self):
        return "next_obs/state", self._env.obs_buf["policy"]


class LearnedEnvPreStepFullStateRecorder(RecorderTerm):
    """Records full env state at the start of the step (obs/full_state for replay)."""

    def record_pre_step(self):
        if "full_state" not in self._env.obs_buf:
            return None, None
        return "obs/full_state", self._env.obs_buf["full_state"]


class LearnedEnvPostStepFullStateRecorder(RecorderTerm):
    """Records full env state at the end of the step (next_obs/full_state for replay)."""

    def record_post_step(self):
        if "full_state" not in self._env.obs_buf:
            return None, None
        return "next_obs/full_state", self._env.obs_buf["full_state"]


class LearnedEnvPostStepRewardsRecorder(RecorderTerm):
    """Records reward for the step that just finished."""

    def record_post_step(self):
        return "rewards", self._env.reward_buf


class LearnedEnvPostStepDonesRecorder(RecorderTerm):
    """Records done flags (terminated or truncated) for the step that just finished."""

    def record_post_step(self):
        done = self._env.reset_terminated | self._env.reset_time_outs
        return "dones", done


@configclass
class LearnedEnvInitialStateRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = LearnedEnvInitialStateRecorder


@configclass
class LearnedEnvPostStepStateRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = LearnedEnvPostStepStateRecorder


@configclass
class LearnedEnvPreStepActionRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = LearnedEnvPreStepActionRecorder


@configclass
class LearnedEnvPreStepJointTargetsRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = LearnedEnvPreStepJointTargetsRecorder


@configclass
class LearnedEnvPreStepRelativeJointRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = LearnedEnvPreStepRelativeJointRecorder


@configclass
class LearnedEnvPreStepEERecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = LearnedEnvPreStepEERecorder


@configclass
class LearnedEnvPreStepRelativeEERecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = LearnedEnvPreStepRelativeEERecorder


@configclass
class LearnedEnvPreStepPolicyObsRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = LearnedEnvPreStepPolicyObsRecorder


@configclass
class LearnedEnvPostStepPolicyObsRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = LearnedEnvPostStepPolicyObsRecorder


@configclass
class LearnedEnvPostStepRewardsRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = LearnedEnvPostStepRewardsRecorder


@configclass
class LearnedEnvPostStepDonesRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = LearnedEnvPostStepDonesRecorder


@configclass
class LearnedEnvPreStepFullStateRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = LearnedEnvPreStepFullStateRecorder


@configclass
class LearnedEnvPostStepFullStateRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = LearnedEnvPostStepFullStateRecorder


@configclass
class LearnedEnvRecorderManagerCfg(RecorderManagerBaseCfg):
    """Recorder manager config for learned envs: obs, actions, scene state, rewards, dones.
    Uses Isaac Lab HDF5 exporter; compatible with robomimic-style datasets (obs/state, next_obs/state, actions, etc.).
    """

    dataset_export_mode: DatasetExportMode = DatasetExportMode.EXPORT_ALL
    export_in_record_pre_reset: bool = True

    record_initial_state = LearnedEnvInitialStateRecorderCfg()
    record_post_step_states = LearnedEnvPostStepStateRecorderCfg()
    record_pre_step_actions = None
    record_pre_step_joint_targets = LearnedEnvPreStepJointTargetsRecorderCfg()
    record_pre_step_relative_joint = LearnedEnvPreStepRelativeJointRecorderCfg()
    record_pre_step_ee = LearnedEnvPreStepEERecorderCfg()
    record_pre_step_relative_ee = LearnedEnvPreStepRelativeEERecorderCfg()
    record_pre_step_policy_obs = LearnedEnvPreStepPolicyObsRecorderCfg()
    record_post_step_policy_obs = LearnedEnvPostStepPolicyObsRecorderCfg()
    record_pre_step_full_state = LearnedEnvPreStepFullStateRecorderCfg()
    record_post_step_full_state = LearnedEnvPostStepFullStateRecorderCfg()
    record_post_step_rewards = LearnedEnvPostStepRewardsRecorderCfg()
    record_post_step_dones = LearnedEnvPostStepDonesRecorderCfg()

