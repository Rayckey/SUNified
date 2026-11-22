from typing import *
import logging
import torch
import numpy as np
from torch import Tensor
import copy
from contextlib import contextmanager
from kuafu.mpc.simulation.spatial_utils import (
    Transform,
    x_axis_from_quat,
    y_axis_from_quat,
    z_axis_from_quat,
    quat_to_matrix,
    quat_to_rotation_6d,
    angle_between_vectors,
    codirectional_angle_between_vectors,
    antiparallel_angle_between_vectors,
    bipolar_angle_between_vector_and_line,
    orthogonality_angle_between_vector_and_line,
    quat_multiply_pos,
    quat_from_euler_xyz,
    quat_from_angle_axis,
    inverse_quat,
)

from isaaclab.utils.math import (
    quat_from_matrix,
    matrix_from_euler,
    matrix_from_quat,
    euler_xyz_from_quat,
    make_pose,
    pose_inv,
    unmake_pose,
    pose_in_A_to_pose_in_B,
)

from kuafu.mpc.simulation.struct import VisibleEntity, SensedEntity
from kuafu.mpc.torch.torch_predictive_control import (
    Controller,
    ManipulatorPositionController,
    ManipulatorIKController,
)
import isaacsim.core.utils.bounds as bounds_utils

logger = logging.getLogger(__name__)


class APIWrapper:
    def __init__(self, controller: Controller = Controller(), device="cpu",
                 gripper_opened=None, gripper_closed=None,
                 gripper_max_force: float = 30.0,
                 gripper_max_grasp_openness: float = 0.8,
                 gripper_effort_contact_velocity_threshold: float = 0.2,
                 robot_gripper_name: str = "contact_sensor",
                 kinova_gripper_effort_contact_openness_slope: float = 0.0,
                 kinova_gripper_effort_contact_openness_intercept: float = 0.0,
                 has_finger_contact_sensors: bool = True,
                 manipulator_high_effort_threshold: float = 8.0):

        self.controller = controller
        self.reference_agent = None
        self.has_finger_contact_sensors = has_finger_contact_sensors

        self.device = device
        self.batch = 1
        self.dtype = torch.float32

        self.twist_dict = {}
        self.joint_dict = {}

        # Gripper joint limits as tensors. Shape (num_gripper_joints,), typically (2,).
        # Default to Panda-style values [0.04, 0.04] / [0.0, 0.0] for backward compat.
        if gripper_opened is None:
            gripper_opened = [0.04, 0.04]
        if gripper_closed is None:
            gripper_closed = [0.0, 0.0]
        self._gripper_opened = torch.as_tensor(gripper_opened, dtype=torch.float32, device=device)
        self._gripper_closed = torch.as_tensor(gripper_closed, dtype=torch.float32, device=device)

        # Maximum gripper force (robot-dependent, injected into predicate scope).
        self._gripper_max_force = float(gripper_max_force)
        self._gripper_max_grasp_openness = float(gripper_max_grasp_openness)
        self._gripper_effort_contact_velocity_threshold = float(
            gripper_effort_contact_velocity_threshold
        )
        self._robot_gripper_name = str(robot_gripper_name or "contact_sensor")
        self._kinova_gripper_effort_contact_openness_slope = float(
            kinova_gripper_effort_contact_openness_slope
        )
        self._kinova_gripper_effort_contact_openness_intercept = float(
            kinova_gripper_effort_contact_openness_intercept
        )
        self._manipulator_high_effort_threshold = float(
            manipulator_high_effort_threshold
        )

        self.cache = bounds_utils.create_bbox_cache()
        self.prim_paths_dict = {}

        # Contact force data keyed by sensor name, e.g. "contact_sensor_left" -> (N, 3) tensor
        self.contact_forces: dict[str, torch.Tensor] = {}
        self._contact_force_lpf_alpha = 0.1

        # Batch selection for efficient per-stage operations - None means all environments.
        # _batch_indices is computed once when entering an active stage and reused
        # by _apply_mask(); this avoids repeating boolean indexing in every API call.
        self._batch_mask = None
        self._batch_indices: Tensor | None = None
        self._batch_mask_sum: int | None = None
        self._batch_mask_len: int | None = None
        # Cache signature for the active batch selection (used to key eval-cache memoization safely).
        # When no mask is active, this is ("all",).
        self._batch_mask_sig: tuple = ("all",)

        # Snapshots for stage transitions / time-stamped queries
        self.state_current = torch.Tensor([])
        self._joint_current = {}
        self._joint_stall = {}

        # Per-environment counters for detecting stalled motion / effort over multiple steps.
        # Lazily initialized to match the active (masked) batch size on first use.
        self.stall_joint_counter: Tensor | None = None
        self.stall_cost_counter: Tensor | None = None

        # PERFORMANCE: per-evaluation cache for derived API outputs.
        # This deduplicates repeated API calls inside compiled predicate/residual scopes.
        self._eval_cache: dict[tuple, Any] = {}

    def _cache_get(self, key: tuple) -> Any | None:
        return self._eval_cache.get(key)

    def _cache_set(self, key: tuple, value: Any) -> Any:
        self._eval_cache[key] = value

    def memo(self, key, val=None):
        cache_key = ("memo",) + tuple(self._batch_mask_sig) + key
        if val is None:
            return self._cache_get(cache_key)
        return self._cache_set(cache_key, val)

    @contextmanager
    def eval_cache(self):
        """Scope for one evaluation: memo() is valid inside the block; cache is cleared on exit."""
        try:
            yield
        finally:
            self._eval_cache.clear()

    def update_contact_forces(self, contact_forces: dict[str, torch.Tensor]) -> None:
        """Update contact forces with an exponential low-pass filter."""
        alpha = self._contact_force_lpf_alpha
        filtered_contact_forces: dict[str, torch.Tensor] = {}

        for key, raw_force in contact_forces.items():
            prev_force = self.contact_forces.get(key)
            if (
                prev_force is None
                or prev_force.shape != raw_force.shape
                or prev_force.device != raw_force.device
                or prev_force.dtype != raw_force.dtype
            ):
                filtered_contact_forces[key] = raw_force.clone()
            else:
                filtered_contact_forces[key] = alpha * raw_force + (1.0 - alpha) * prev_force

        self.contact_forces = filtered_contact_forces

    def set_batch_mask(self, mask: Optional[Tensor]):
        """Set the active batch mask for efficient per-stage cost computation."""
        if mask is None:
            self.clear_batch_mask()
            return

        # Normalize to a flat boolean vector once, then precompute integer indices.
        # _apply_mask() can then use index_select in every API call instead of
        # repeatedly applying the same boolean mask.
        if mask.dtype is not torch.bool:
            mask = mask.to(dtype=torch.bool)
        if mask.ndim != 1:
            mask = mask.view(-1)
        indices = torch.nonzero(mask, as_tuple=False).flatten()
        self.set_batch_indices(indices, full_batch_size=int(mask.shape[0]), mask=mask)

    def set_batch_indices(
        self,
        indices: Optional[Tensor],
        *,
        full_batch_size: int | None = None,
        mask: Optional[Tensor] = None,
    ):
        """Set the active batch by integer env ids.

        Equivalent to set_batch_mask(mask), but avoids recomputing nonzero(mask)
        and allows the SUNProgramEvaluator stage loop to pass its grouped env ids directly.
        """
        if indices is None:
            self.clear_batch_mask()
            return
        if indices.dtype is not torch.long:
            indices = indices.to(dtype=torch.long)
        if indices.ndim != 1:
            indices = indices.view(-1)
        if full_batch_size is None:
            full_batch_size = int(self.batch) if self.batch is not None else int(indices.numel())

        self._batch_mask = mask
        self._batch_indices = indices
        self._batch_mask_len = int(full_batch_size)
        self._batch_mask_sum = int(indices.shape[0])
        # Make memoization safe across different active stage subsets within the
        # same eval_cache() context. The env-id tensor data_ptr is enough to avoid
        # accidental cross-stage reuse without hashing large tensors.
        self._batch_mask_sig = (
            "indices",
            str(indices.device),
            int(self._batch_mask_len),
            int(self._batch_mask_sum),
            int(indices.data_ptr()),
        )

    def clear_batch_mask(self):
        """Clear the batch mask to operate on all environments."""
        self._batch_mask = None
        self._batch_indices = None
        self._batch_mask_sum = None
        self._batch_mask_len = None
        self._batch_mask_sig = ("all",)

    def _apply_mask(self, tensor: Tensor) -> Tensor:
        """Apply the current batch mask to a tensor if mask is set.
        
        If the tensor is already masked (i.e. has the masked batch size), return it as-is.
        This handles cases where internal data is already masked but external
        arguments need to be masked to match.
        """
        if self._batch_indices is None:
            return tensor

        # Scalars or tensors without a leading batch dimension should not be masked
        if tensor.ndim == 0:
            return tensor

        batch_dim = tensor.shape[0]
        full_batch = self._batch_mask_len
        masked_batch = self._batch_mask_sum

        # Should never happen because set_batch_mask() normalizes + populates these.
        if full_batch is None or masked_batch is None:
            return tensor

        # If the tensor already has the masked batch size (and not the full batch),
        # we assume it has already had this mask applied and leave it unchanged.
        if batch_dim == masked_batch and batch_dim != full_batch:
            return tensor

        # COMMON CASE: user passes a constant (e.g. (3,), (1,3), (1,4), ...) while a batch
        # mask is active. Broadcast it to the full batch so it aligns with internal data,
        # then apply the mask. This preserves prior behavior but makes it explicit.
        if batch_dim == 1 and full_batch > 1:
            # Use expand when possible (no copy); fall back to repeat if needed.
            try:
                tensor = tensor.expand((full_batch,) + tuple(tensor.shape[1:]))
            except RuntimeError:
                reps = (full_batch,) + (1,) * (tensor.ndim - 1)
                tensor = tensor.repeat(reps)
            batch_dim = tensor.shape[0]

        # If a 1D feature vector is passed (e.g. (3,), (4,), ...), treat it as a constant
        # and broadcast to (full_batch, features). This matches how these APIs are used.
        if tensor.ndim == 1 and batch_dim != full_batch and full_batch > 1:
            tensor = tensor.unsqueeze(0).expand((full_batch, tensor.shape[0]))
            batch_dim = tensor.shape[0]

        # FAIL-FAST: if a mask is active, tensors must align with either the full batch
        # or already-masked batch. Returning unchanged silently hides bugs.
        if batch_dim != full_batch:
            raise ValueError(
                f"Batch mask length mismatch: tensor batch_dim={batch_dim}, "
                f"mask_len={full_batch}, mask_sum={masked_batch}."
            )

        indices = self._batch_indices
        if indices is None:
            return tensor
        # Equivalent to tensor[self._batch_mask] for a full-batch leading dim,
        # but the indices were computed once for the current active stage.
        return torch.index_select(tensor, 0, indices)

    def updatemOdom(self, msg, msg_s):
        # Clear derived frame/object cache every time odometry changes.
        self._eval_cache.clear()

        # CRITICAL: this function stores tensors every step; it must never record autograd graphs
        # or VRAM usage can grow without bound (retained history). Always run inference-only.
        with torch.inference_mode():
            # Always store transforms in agent_frame: compute reference pose once here.
            if self.reference_agent is None:
                raise RuntimeError("reference_agent not set. Call setPOVAgent before updatemOdom.")
            if self.reference_agent not in msg_s:
                raise KeyError(
                    f"reference_agent '{self.reference_agent}' is missing from odom msg; "
                    "cannot convert transforms into agent_frame."
                )
            p_ref = msg_s[self.reference_agent][0]
            q_ref = msg_s[self.reference_agent][1]
            if self.reference_agent not in self.controller.entity_dict:
                tf = Transform(make_pose(p_ref, matrix_from_quat(q_ref)))
                self.controller.entity_dict[self.reference_agent] = SensedEntity(
                    self.reference_agent,
                    msg_s[self.reference_agent][3][0],
                    msg_s[self.reference_agent][3][1],
                    tf,
                )
                # learn batch and device info from reference agent pose.
                self.batch = p_ref.shape[0]
                self.dtype = p_ref.dtype
                self.device = p_ref.device
            else:
                tf = Transform(make_pose(p_ref, matrix_from_quat(q_ref)))
                self.controller.entity_dict[self.reference_agent].updateTransform(tf)

            inv_ref_tf = self.controller.entity_dict[self.reference_agent].transform_real_time.inverse

            for key in msg:
                if key not in self.controller.entity_dict:
                    tf = Transform(make_pose(msg[key][0], matrix_from_quat(msg[key][1])))
                    self.controller.entity_dict[key] = VisibleEntity(
                        key,
                        msg[key][3][0],
                        msg[key][3][1],
                        inv_ref_tf @ tf,
                    )
                    self.controller.entity_dict[key].size = self.get_object_xyz_size(key)
                else:
                    tf = Transform(make_pose(msg[key][0], matrix_from_quat(msg[key][1])))
                    self.controller.entity_dict[key].updateTransform(inv_ref_tf @ tf)

            for key in msg_s:
                if key == self.reference_agent:
                    continue
                tf = Transform(make_pose(msg_s[key][0], matrix_from_quat(msg_s[key][1])))
                if key not in self.controller.entity_dict:
                    self.controller.entity_dict[key] = SensedEntity(
                        key,
                        msg_s[key][3][0],
                        msg_s[key][3][1],
                        inv_ref_tf @ tf,
                    )
                else:
                    self.controller.entity_dict[key].updateTransform(inv_ref_tf @ tf)

    def getMinimumState(self):
        return self._apply_mask(self.joint_dict[self.reference_agent]["joint_position"][:, -2:])

    def getControllerState(self):
        """
        Get the state of the controller
        """
        states = []
        if issubclass(type(self.controller), ManipulatorPositionController):
            raise NotImplementedError(
                "ManipulatorPositionController does not support state retrieval."
            )
        elif issubclass(type(self.controller), ManipulatorIKController):
            temp_ = torch.cat(
                [self.lookup_robot_frame_position(self.controller.ee_frame), 
                 self.lookup_robot_frame_rotation(self.controller.ee_frame), 
                 self._apply_mask(self.joint_dict[self.reference_agent]["joint_position"][:, -2:])], dim=-1
            )

            states.append(temp_)

        else:
            raise NotImplementedError(
                "ManipulatorPositionController does not support state retrieval."
            )

        return tuple(states)

    def setPOVAgent(self, agent_name):
        self.reference_agent = agent_name

    def updatePOVState(self, env_mask=None):
        """
        Save current transforms as checkpoint for stage transitions.
        
        Args:
            env_mask: Optional boolean tensor [batch_size] indicating which environments
                     are transitioning stages. If None, saves all (single-env mode).
        """
        # Snapshotting stores tensors; ensure no autograd history is recorded.
        with torch.inference_mode():
            for key in self.controller.entity_dict:
                self.controller.entity_dict[key].saveTransform(env_mask)

        # Snapshot changed; clear any derived cache as well.
        self._eval_cache.clear()

        # Save joint snapshots for time-stamped API (e.g., manipulator_effort with current timestamp)
        # FAIL-FAST: if controller state retrieval is unsupported/broken, surface it.
        state_tuple = self.getControllerState()
        self.state_current = (
            torch.concat(state_tuple) if state_tuple else torch.Tensor([])
        )
        # Reset stall counters for environments that are transitioning (masked).
        if env_mask is not None:
            if self.stall_cost_counter is not None:
                self.stall_cost_counter = torch.where(
                    env_mask, torch.zeros_like(self.stall_cost_counter), self.stall_cost_counter
                )
            if self.stall_joint_counter is not None:
                self.stall_joint_counter = torch.where(
                    env_mask, torch.zeros_like(self.stall_joint_counter), self.stall_joint_counter
                )
        logger.debug("[Sat] Saved POV state for timestamped queries.")

    def countStall(self, threshold: float = 1e-2) -> Tensor:
        """
        Count how many consecutive steps the manipulator joint positions have remained
        approximately unchanged.

        Returns a per-environment counter tensor for the full batch (num_envs,).
        This function does not apply any batch mask; callers can mask the returned
        tensor if needed.
        """
        if self.reference_agent is None:
            raise RuntimeError("reference_agent not set. Call setPOVAgent before countStall.")

        rt_full = self.joint_dict[self.reference_agent]["joint_position"][:, :-self._gripper_opened.numel()]
        st_full = self._joint_stall[self.reference_agent]["joint_position"][:, :-self._gripper_opened.numel()]

        if self.stall_joint_counter is None:
            N = int(rt_full.shape[0])
            self.stall_joint_counter = torch.zeros(
                N, dtype=torch.int64, device=rt_full.device
            )

        diff = torch.sum((rt_full - st_full) ** 2, dim=-1)  # (N,)
        stable = diff < threshold  # (N,)

        c = self.stall_joint_counter
        c_new = torch.where(stable, c + 1, torch.where(c >= 2, c - 2, c))
        # Clamp between 0 and 100 so counters neither grow unbounded nor go negative.
        self.stall_joint_counter = torch.clamp(c_new, 0, 30)

        # Refresh stall snapshot per-environment when motion resumes.
        if torch.any(~stable):
            bad = ~stable
            self._joint_stall[self.reference_agent]["joint_position"][bad] = (
                self.joint_dict[self.reference_agent]["joint_position"][bad]
            )

        return self.stall_joint_counter

    def countEffortStall(
        self,
        effort_threshold: float = 40.0,
        min_effort_stall_steps: int = 10,
    ) -> Tensor:
        """
        Count consecutive updates whose total arm effort exceeds the threshold.

        Return full-batch counters capped at min_effort_stall_steps. An update
        at or below the threshold resets that environment's counter.
        """
        if self.reference_agent is None:
            raise RuntimeError(
                "reference_agent not set. Call setPOVAgent before countEffortStall."
            )

        rt_full = self.joint_dict[self.reference_agent]["joint_effort"][:, :-self._gripper_opened.numel()]

        if self.stall_cost_counter is None:
            N = int(rt_full.shape[0])
            self.stall_cost_counter = torch.zeros(
                N, dtype=torch.int64, device=rt_full.device
            )

        total_eff = torch.sum(torch.abs(rt_full), dim=-1)  # (N,)
        stable = total_eff > effort_threshold

        c = self.stall_cost_counter
        c_new = torch.where(stable, c + 1, torch.zeros_like(c))
        self.stall_cost_counter = torch.clamp(c_new, 0, min_effort_stall_steps)


        return self.stall_cost_counter

    def get_all_residual_apis(self):
        apis = {
            "get_object_xyz_position": self.get_object_xyz_position,
            "get_object_wxyz_quaternion": self.get_object_wxyz_quaternion,
            "get_object_xyz_size": self.get_object_xyz_size,
            "memo": self.memo,
            "quat_to_matrix": quat_to_matrix,
            "quat_to_rotation_6d": quat_to_rotation_6d,
            "inverse_quat": inverse_quat,
            "euler_xyz_from_quat": self.euler_xyz_from_quat_batched,
            "x_axis_from_quat": x_axis_from_quat,
            "y_axis_from_quat": y_axis_from_quat,
            "z_axis_from_quat": z_axis_from_quat,
            "quat_from_euler_xyz": quat_from_euler_xyz,
            "quat_from_matrix": quat_from_matrix,
            "quat_multiply_pos": quat_multiply_pos,
            "matrix_from_euler": matrix_from_euler,
            "matrix_from_quat": matrix_from_quat,
            "make_pose": make_pose,
            "pose_inv": pose_inv,
            "unmake_pose": unmake_pose,
            "object_to_world_xyz_position": self.object_to_world_xyz_position,
            "world_to_object_xyz_position":self.world_to_object_xyz_position,
            "angle_between_vectors": angle_between_vectors,
            "pose_in_A_to_pose_in_B": pose_in_A_to_pose_in_B,
            "lookup_robot_frame_position": self.lookup_robot_frame_position,
            "lookup_robot_frame_rotation": self.lookup_robot_frame_rotation,
            "calculate_robot_frame_position": self.lookup_robot_frame_position,  # OVER WRITE
            "calculate_robot_frame_rotation": self.lookup_robot_frame_rotation,  # OVER WRITE
        }

        return apis

    def euler_xyz_from_quat_batched(self, q: torch.Tensor) -> torch.Tensor:
        return torch.column_stack(euler_xyz_from_quat(q))

    def get_all_predicate_apis(self):
        apis = {
            "get_object_xyz_position": self.get_object_xyz_position,
            "get_object_wxyz_quaternion": self.get_object_wxyz_quaternion,
            "get_object_xyz_size": self.get_object_xyz_size,
            # Shared memoization hook for compiled predicate/residual code.
            "memo": self.memo,
            "quat_to_matrix": quat_to_matrix,
            "quat_to_rotation_6d": quat_to_rotation_6d,
            "quat_conj": inverse_quat,
            "inverse_quat": inverse_quat,
            "euler_xyz_from_quat": self.euler_xyz_from_quat_batched,
            "x_axis_from_quat": x_axis_from_quat,
            "y_axis_from_quat": y_axis_from_quat,
            "z_axis_from_quat": z_axis_from_quat,
            "quat_multiply_pos": quat_multiply_pos,
            "quat_from_euler_xyz": quat_from_euler_xyz,
            "quat_from_angle_axis": quat_from_angle_axis,
            "codirectional_angle_between_vectors": codirectional_angle_between_vectors,
            "antiparallel_angle_between_vectors": antiparallel_angle_between_vectors,
            "bipolar_angle_between_vector_and_line": bipolar_angle_between_vector_and_line,
            "angle_between_vectors": angle_between_vectors,
            "orthogonality_angle_between_vector_and_line": orthogonality_angle_between_vector_and_line,
            "object_to_world_xyz_position": self.object_to_world_xyz_position,
            "world_to_object_xyz_position":self.world_to_object_xyz_position,
            "world_to_object_wxyz_quaternion": self.world_to_object_wxyz_quaternion,
            "object_to_world_wxyz_quaternion": self.object_to_world_wxyz_quaternion,
            "lookup_robot_frame_position": self.lookup_robot_frame_position,
            "lookup_robot_frame_rotation": self.lookup_robot_frame_rotation,
            "gripper_openness": self.gripper_openness,
            "gripper_force": self.gripper_force,
            "finger_contact_force": self.finger_contact_force,
            "manipulator_effort": self.manipulator_effort,
            "high_effort_contact": self.high_effort_contact,
        }
        # Robot-dependent gripper max force injected into predicate scope.
        apis["GRIPPER_MAX_FORCE"] = self._gripper_max_force
        apis["GRIPPER_MAX_GRASP_OPENNESS"] = self._gripper_max_grasp_openness
        apis["ROBOT_GRIPPER_NAME"] = self._robot_gripper_name
        return apis

    def update_object_bounding_boxes(self):
        self.cache = bounds_utils.create_bbox_cache()

    def gripper_openness(self) -> Tensor:
        """
        Check the state of the gripper.
        :return: (batch,) tensor with 1.0 if the gripper is fully open, 0.0 if fully closed
        """
        if not self.controller.has_gripper:  # type: ignore[attr-defined]
            raise AttributeError("Controller does not have gripper state information.")
        key = ("gripper_openness",)
        cached = self._cache_get(key)
        if cached is None:
            # Cache the full-batch result and apply the active stage selection at return time.
            n_gripper = self._gripper_opened.shape[0]
            finger_pos = self.joint_dict[self.reference_agent]["joint_position"][:, -n_gripper:]
            gripper_range = self._gripper_opened - self._gripper_closed  # (n_gripper,)
            cached = ((finger_pos - self._gripper_closed) / gripper_range).mean(dim=-1)
            self._cache_set(key, cached)
        return self._apply_mask(cached)

    def gripper_force(self) -> float:
        """Return the larger finger joint effort for each selected environment."""

        # gripper_force should return a masked tensor when a stage batch is active.
        key = ("gripper_force",)
        cached = self._cache_get(key)
        if cached is None:
            effort = self.joint_dict[self.reference_agent]["joint_effort"][:, -2:]
            cached = torch.maximum(effort[:, 0], effort[:, 1])
            self._cache_set(key, cached)
        return self._apply_mask(cached)


    def _uses_joint_effort_finger_contact(self) -> bool:
        return self._robot_gripper_name in (
            "kinova_robotiq",
            "ur5_robotiq",
            "franka_panda",
        )

    def _joint_effort_finger_contact_force(self) -> tuple[torch.Tensor, torch.Tensor]:
        cache_key = ("finger_contact_force_joint_effort", self._robot_gripper_name)
        cached = self._cache_get(cache_key)
        if cached is None:
            effort = torch.abs(self.joint_dict[self.reference_agent]["joint_effort"][:, -2:])
            if self._robot_gripper_name in ("kinova_robotiq", "ur5_robotiq"):
                drive_effort = effort[:, :1]
                effort = drive_effort.expand(-1, effort.shape[-1])
            n_gripper = self._gripper_opened.shape[0]
            finger_pos = self.joint_dict[self.reference_agent]["joint_position"][:, -n_gripper:]
            gripper_range = self._gripper_opened - self._gripper_closed
            openness = ((finger_pos - self._gripper_closed) / gripper_range).mean(dim=-1)

            if self._robot_gripper_name in ("kinova_robotiq", "ur5_robotiq"):
                within_grasp_position = (openness > 0.1) & (openness <= self._gripper_max_grasp_openness)
            elif self._robot_gripper_name == "franka_panda":
                within_grasp_position = openness > 0.1
            else:
                raise RuntimeError(
                    f"No joint-effort finger contact model for robot gripper {self._robot_gripper_name!r}"
                )

            finger_vel = self.joint_dict[self.reference_agent].get("joint_velocity")
            if finger_vel is None:
                gripper_slow_or_stalled = torch.ones_like(within_grasp_position, dtype=torch.bool)
            else:
                finger_vel = finger_vel[:, -n_gripper:]
                if self._robot_gripper_name in ("kinova_robotiq", "ur5_robotiq"):
                    drive_vel = torch.abs(finger_vel[:, 0])
                    gripper_slow_or_stalled = drive_vel < self._gripper_effort_contact_velocity_threshold
                else:
                    gripper_slow_or_stalled = (
                        torch.max(torch.abs(finger_vel), dim=-1).values
                        < self._gripper_effort_contact_velocity_threshold
                    )

            contact_like_effort = within_grasp_position & gripper_slow_or_stalled
            if self._robot_gripper_name in ("kinova_robotiq", "ur5_robotiq"):
                required_effort = (
                    self._kinova_gripper_effort_contact_openness_slope * openness
                    + self._kinova_gripper_effort_contact_openness_intercept
                )
                required_effort = torch.clamp(required_effort, min=0.0)
                effort_meets_threshold = torch.min(effort, dim=-1).values >= required_effort
                contact_like_effort = contact_like_effort & effort_meets_threshold

            effort = torch.where(contact_like_effort.unsqueeze(-1), effort, torch.zeros_like(effort))
            cached = (effort[:, 0], effort[:, 1])
            self._cache_set(cache_key, cached)
        left_effort, right_effort = cached
        return self._apply_mask(left_effort), self._apply_mask(right_effort)

    def finger_contact_force(self, left_key: str = "contact_sensor_left", right_key: str = "contact_sensor_right") -> tuple[torch.Tensor, torch.Tensor]:
        """Return contact force magnitudes for left and right finger tips.

        Returns:
            (left_mag, right_mag): each (N,) tensor with force magnitude per environment.
            Returns zeros if contact data has not been fed via callbackContactForces yet.
        """
        if self._uses_joint_effort_finger_contact():
            return self._joint_effort_finger_contact_force()

        cache_key = ("finger_contact_force", str(left_key), str(right_key))
        cached = self._cache_get(cache_key)
        if cached is None:
            batch = self.joint_dict[self.reference_agent]["joint_position"].shape[0]
            device = self.joint_dict[self.reference_agent]["joint_position"].device

            def _mag(key):
                if key in self.contact_forces:
                    f = self.contact_forces[key]  # (N, 3)
                    return f.norm(dim=-1)          # (N,)
                return torch.zeros(batch, device=device)

            cached = (_mag(left_key), _mag(right_key))
            self._cache_set(cache_key, cached)
        left_mag, right_mag = cached
        return self._apply_mask(left_mag), self._apply_mask(right_mag)

    def manipulator_effort(self, time_stamp: str = "real time") -> torch.Tensor:
        """Total absolute joint effort (excluding gripper) per environment.

        Args:
            time_stamp: "real time" for current efforts, "current" for snapshot saved via updatePOVState.
        Returns:
            (batch, 1) tensor after masking representing summed absolute effort over the primary arm joints.
        Notes:
            Threshold heuristics (from simulation version): values above ~25 may imply contact; tune per robot.
        """
        if self.reference_agent is None:
            raise RuntimeError("reference_agent not set. Call setPOVAgent before querying manipulator_effort.")
        if time_stamp == "real time":
            eff_source = self.joint_dict
        elif time_stamp == "current":
            eff_source = self._joint_current
        else:
            raise KeyError(f"Unexpected timestamp '{time_stamp}' for manipulator_effort (use 'real time' or 'current').")

        if self.reference_agent not in eff_source:
            raise KeyError(f"Agent '{self.reference_agent}' joint efforts not available.")

        cache_key = ("manipulator_effort", str(time_stamp))
        cached = self._cache_get(cache_key)
        if cached is None:
            efforts = eff_source[self.reference_agent].get("joint_effort")
            if efforts is None:
                raise KeyError("joint_effort key missing in joint_dict for reference agent.")
            # Gripper slots follow the arm for every supported manipulator.
            cached = torch.sum(torch.abs(efforts[:, :-self._gripper_opened.numel()]), dim=-1).unsqueeze(-1)
            self._cache_set(cache_key, cached)
        return self._apply_mask(cached)

    def high_effort_contact(
        self,
        min_position_stall_steps: int = 20,
        min_effort_stall_steps: int = 5,
        position_similarity_threshold: float = 0.01,
        effort_magnitude_threshold: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Return a sustained-high-effort flag for each selected environment.

        The effort must exceed the threshold for min_effort_stall_steps updates.
        Position-stall arguments are retained for caller compatibility; this
        check does not test position stability.
        """
        if effort_magnitude_threshold is None:
            effort_magnitude_threshold = self._manipulator_high_effort_threshold

        # Full-batch per-env counters; mask is applied only at return.
        effort_stall_steps_full = self.countEffortStall(
            effort_threshold=effort_magnitude_threshold,
            min_effort_stall_steps=min_effort_stall_steps,
        )  # (N,)

        eff_ok_full = effort_stall_steps_full >= min_effort_stall_steps

        return self._apply_mask(eff_ok_full)

    def get_oriented_bounding_box_corners(self, a: str) -> np.ndarray:
        """
        Get the oriented bounding box of an object.
        :param a: Name of the object.
        :return: NumPy array of shape (8, 3)
        """
        if not self.check_known_object(a):
            raise KeyError(f"Object {a} doesn't exist in the scene!")
        centroid, axes, half_extent = bounds_utils.compute_obb(
            self.cache, prim_path=self.prim_paths_dict[a].replace("*", "0")
        )
        return bounds_utils.get_obb_corners(centroid, axes, half_extent)

    def get_object_xyz_size(self, a: str, time_stamp="NOT IMPORTANT") -> torch.Tensor:
        """
        Get the XYZ size of an object.
        :param a: Name of the object.
        :return: (3,) torch Tensor
        """
        if not self.check_known_object(a):
            raise KeyError(f"Object {a} doesn't exist in the scene!")

        if self.controller.entity_dict[a].has_size is False:
            raise KeyError(
                f"This reference {a} does not have a size property, it may be a reference frame for another object!"
            )
        centroid, axes, half_extent = bounds_utils.compute_obb(
            self.cache,
            prim_path=self.controller.entity_dict[a].prim_path.replace("*", "0"),
        )
        return torch.from_numpy(half_extent * 2).to(self.device)

    def lookup_robot_frame_position(
        self, frame_name: str, timestamp="real time"
    ) -> torch.Tensor:
        """
        Calculate the position of a frame on the robot using kinematics.
        :param frame_name: Name of the frame.
        :param joint_state: Joint state of the robot.
        :return: Position of the frame in the robot's base frame.
        """


        return self.get_object_xyz_position(frame_name, timestamp)

    def lookup_robot_frame_rotation(
        self, frame_name: str, timestamp="real time"
    ) -> torch.Tensor:
        """
        Calculate the position of a frame on the robot using kinematics.
        :param frame_name: Name of the frame.
        :param joint_state: Joint state of the robot.
        :return: Position of the frame in the robot's base frame.
        """
        return self.get_object_wxyz_quaternion(frame_name, timestamp)

    def inverse_transformation(self, trans: Transform) -> Transform:
        return trans.inverse

    def get_object_transformation(self, a, timestamp="real time") -> Transform:
        if not self.check_known_object(a):
            raise KeyError(f"Object {a} doesn't exist in the scene!")

        if self.reference_agent is None:
            raise RuntimeError("reference_agent not set. Call setPOVAgent first.")

        # Transforms are stored in agent_frame, so raw is already relative.
        return self.get_raw_transformation(a, timestamp)

    def _quat_mul_wxyz(self, q1: Tensor, q2: Tensor) -> Tensor:
        """Quaternion multiply for wxyz quaternions. Shapes (...,4)."""
        w1, x1, y1, z1 = q1.unbind(-1)
        w2, x2, y2, z2 = q2.unbind(-1)
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return torch.stack((w, x, y, z), dim=-1)

    def _get_object_rel_pose(self, a: str, timestamp: str) -> tuple[Tensor, Tensor]:
        """Return (pos, quat) of object `a` in the reference_agent frame.

        This avoids 4x4 pose inversion/matmul by using quaternion algebra:
          q_rel = q_ref_conj ⊗ q_obj
          p_rel = R(q_ref_conj) * (p_obj - p_ref)
        """
        if self.reference_agent is None:
            raise RuntimeError("reference_agent not set. Call setPOVAgent first.")
        if not self.check_known_object(a):
            raise KeyError(f"Object {a} doesn't exist in the scene!")
        if timestamp not in ("real time", "current"):
            raise KeyError(f"Unexpected look up for Object {a} ")

        cache_key = ("object_rel_pose", str(a), str(timestamp))
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        obj = self.get_raw_transformation(a, timestamp)

        # Transforms are stored in agent_frame, so rel pose is a direct read.
        p_rel = obj.translation
        q_rel = obj.quaternion
        cached = (p_rel, q_rel)
        self._cache_set(cache_key, cached)

        return cached

    def get_raw_transformation(self, a, timestamp="real time"):
        if a not in self.controller.entity_dict:
            raise KeyError(f"Object {a} doesn't exist in the scene!")
        if timestamp == "real time":
            return self.controller.entity_dict[a].transform_real_time
        elif timestamp == "current":
            return self.controller.entity_dict[a].transform_current
        else:
            raise KeyError(f"Unexpected look up for Object {a} ")

    def check_known_object(self, a) -> bool:
        return a in self.controller.entity_dict

    def get_object_xyz_position(self, a, timestamp="real time") -> torch.Tensor:
        pos, _ = self._get_object_rel_pose(str(a), timestamp)
        return self._apply_mask(pos)

    def get_object_xyz_euler(self, a, timestamp="real time") -> torch.Tensor:
        key = ("get_object_xyz_euler", str(a), str(timestamp))
        cached = self._cache_get(key)
        if cached is not None:
            return self._apply_mask(cached)

        trans = self.get_object_transformation(a, timestamp)
        rot = trans.rotation
        eul = self.euler_xyz_from_quat_batched(quat_from_matrix(rot))
        # Cache full-batch; apply mask at return-time.
        self._cache_set(key, eul)
        return self._apply_mask(eul)

    def get_object_wxyz_quaternion(self, a, timestamp="real time") -> torch.Tensor:
        _, quat = self._get_object_rel_pose(str(a), timestamp)
        return self._apply_mask(quat)

    def object_to_world_xyz_position(
        self, a, local_pos, timestamp="real time"
    ) -> torch.Tensor:
        """
        Transform a (batch,3) position in object frame to a (batch,3) position in world frame.
        """
        pos, q = self._get_object_rel_pose(str(a), timestamp)
        # IMPORTANT: if a stage batch-mask is active, mask all batch-aligned tensors
        # consistently (pos/q and inputs) to avoid (B,*) vs (M,*) mismatches.
        pos = self._apply_mask(pos)
        q = self._apply_mask(q)
        local_pos_masked = self._apply_mask(local_pos)
        return quat_multiply_pos(q, local_pos_masked) + pos

    def world_to_object_xyz_position(self, a, xyz_pos, timestamp="real time") -> torch.Tensor:
        """
        Transform a (batch,3) position in world frame to a (batch,3) position in object frame.
        """
        pos, q = self._get_object_rel_pose(str(a), timestamp)
        pos = self._apply_mask(pos)
        q = self._apply_mask(q)
        q_inv = quat_conj(q)
        # Inverse translation: rotate(-pos) by q_inv
        inv_pos = quat_multiply_pos(q_inv, -pos)
        xyz_pos_masked = self._apply_mask(xyz_pos)
        return quat_multiply_pos(q_inv, xyz_pos_masked) + inv_pos

    def world_to_object_wxyz_quaternion(self, a, world_quat, timestamp="real time"):
        """
        Transform a (batch,4) quaternion in world frame to a (batch,4) quaternion in object frame.
        """
        _, q = self._get_object_rel_pose(str(a), timestamp)
        q = self._apply_mask(q)
        q_inv = quat_conj(q)
        world_quat_masked = self._apply_mask(world_quat)
        return self._quat_mul_wxyz(q_inv, world_quat_masked)

    def object_to_world_wxyz_quaternion(self, a, local_quat, timestamp="real time"):
        """
        Transform a (batch,4) quaternion in object frame to a (batch,4) quaternion in world frame.
        """
        _, q = self._get_object_rel_pose(str(a), timestamp)
        q = self._apply_mask(q)
        local_quat_masked = self._apply_mask(local_quat)
        return self._quat_mul_wxyz(q, local_quat_masked)

    def get_previously_seen_objects(self) -> Set[VisibleEntity]:
        seen_objects = set(self.controller.entity_dict.keys())
        return seen_objects

    def get_previously_seen_entities(self) -> Set[VisibleEntity]:
        seen_objects = copy.deepcopy(self.controller.entity_dict)
        for f in self.controller.getFrameDescription():  # type: ignore[attr-defined]
            seen_objects.pop(f.name, None)
        if "agent" in seen_objects:
            seen_objects.pop("agent", None)
        return set(seen_objects.values())
