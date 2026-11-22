from typing import *
import numpy as np
import torch
import casadi as ca
import copy
from kuafu.mpc.simulation.spatial_utils import (
    Transform,
    x_axis_from_quat,
    y_axis_from_quat,
    z_axis_from_quat,
    quat_to_matrix,
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

from kuafu.mpc.simulation.struct import VisibleEntity, SensedEntity, StateEntity
from kuafu.mpc.controller.mpc import (
    Controller,
    ManipulatorPositionController,
    ManipulatorIKController,
    FR3FinRayIKController,
    DifferentialDriveController,
)
from kuafu.mpc.controller.controller_utils import (
    quat_to_matrix as quat_to_matrix_casadi,
    matrix_to_quat as matrix_to_quat_casadi,
    x_axis_from_quat as x_axis_from_quat_casadi,
    y_axis_from_quat as y_axis_from_quat_casadi,
    z_axis_from_quat as z_axis_from_quat_casadi,
    quat_mul as quat_mul_casadi,
    inverse_quat as inverse_quat_casadi,
    quat_from_euler_xyz as quat_from_euler_xyz_casadi,
)
import isaacsim.core.utils.bounds as bounds_utils


class APIWrapper:
    def __init__(self, controller: Controller = Controller(), device="cpu",
                 gripper_opened=None, gripper_closed=None,
                 gripper_max_force: float = 40.0,
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

        # Keep these available before the first odometry update.  Loaded
        # program may pre-populate every entity, in which case updatemOdom()
        # never enters its new-entity branch that refreshes these values.
        self.device = device
        self.batch = 1
        self.dtype = torch.float32

        self.twist_dict = {}
        self.joint_dict = {}

        self.state_current = torch.Tensor([])
        self.stall_joint_counter: Tensor | None = None
        self.gripper_stall_counter: Tensor | None = None
        self.stall_cost_counter: Tensor | None = None

        self._joint_current = {}
        self._joint_stall = {}
        self._gripper_stall_state: Tensor | None = None

        # Gripper joint limits as tensors. Shape (num_gripper_joints,), typically (2,).
        # Default to Panda-style values [0.04, 0.04] / [0.0, 0.0] for backward compat.
        if gripper_opened is None:
            gripper_opened = [0.04, 0.04]
        if gripper_closed is None:
            gripper_closed = [0.0, 0.0]
        self._gripper_opened = torch.as_tensor(gripper_opened, dtype=torch.float32, device=device)
        self._gripper_closed = torch.as_tensor(gripper_closed, dtype=torch.float32, device=device)
        # Per-finger sign of the closing direction along the contact axis (assumed x).
        # sign(closed - opened): negative joint range → finger closes in -x; positive → +x.
        self._finger_contact_signs = torch.sign(self._gripper_closed - self._gripper_opened)

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

        # Contact force data keyed by sensor name, e.g. "contact_sensor_left" -> (N, 3) tensor
        self.contact_forces: dict = {}
        self._contact_force_lpf_alpha = 0.1

        self.cache = bounds_utils.create_bbox_cache()

    def getControllerState(self):
        """
        Get the state of the controller
        """
        states = []
        if issubclass(type(self.controller), ManipulatorPositionController):
            joint_pos = self.joint_dict[self.reference_agent]["joint_position"]
            joint_vel = self.joint_dict[self.reference_agent]["joint_velocity"]
            if not isinstance(joint_pos, torch.Tensor):
                joint_pos = torch.as_tensor(joint_pos, device=self.device)
            if not isinstance(joint_vel, torch.Tensor):
                joint_vel = torch.as_tensor(joint_vel, device=self.device)
            states.append(joint_pos)
            states.append(joint_vel)
            if self.controller.has_gripper:
                gripper_state = self.joint_dict[self.reference_agent].get(
                    "gripper_state"
                )
                if gripper_state is not None:
                    if not isinstance(gripper_state, torch.Tensor):
                        gripper_state = torch.as_tensor(
                            gripper_state, device=joint_pos.device
                        )
                    gripper_state = gripper_state.to(
                        device=joint_pos.device, dtype=joint_pos.dtype
                    )
                    zero_pad = torch.zeros_like(gripper_state)
                    states[0] = torch.cat((states[0], gripper_state), dim=-1)
                    states[1] = torch.cat((states[1], zero_pad), dim=-1)
        elif issubclass(type(self.controller), ManipulatorIKController):
            temp_ = (
                self.joint_dict[self.reference_agent]["joint_position"].clone().detach()
            )
            n_gripper = self._gripper_opened.shape[0]
            gripper_range = self._gripper_opened - self._gripper_closed  # (n_gripper,)
            temp_[:, -n_gripper:] = (temp_[:, -n_gripper:] - self._gripper_closed) / gripper_range
            states.append(temp_)
            # Provide joint velocity too, so IK controllers can seed with q + dq*dt.
            joint_vel = (
                self.joint_dict[self.reference_agent]["joint_velocity"].clone().detach()
            )
            states.append(joint_vel)

        elif issubclass(type(self.controller), DifferentialDriveController):
            agent_pos = self.lookup_robot_frame_position(self.reference_agent)
            agent_quat = self.lookup_robot_frame_rotation(self.reference_agent)

            temp_ = (
                self.joint_dict[self.reference_agent]["joint_velocity"].clone().detach()
            )

            states.append(agent_pos)
            states.append(agent_quat)
            states.append(temp_)

        else:
            raise NotImplementedError(
                "Controller type not supported in APIWrapper getControllerState."
            )

        return tuple(states)

    def updatemOdom(self, msg, msg_s):
        for key in msg:
            if key not in self.controller.entity_dict:
                self.controller.entity_dict[key] = VisibleEntity(
                    key,
                    msg[key][3][0],
                    msg[key][3][1],
                    Transform(make_pose(msg[key][0], matrix_from_quat(msg[key][1]))),
                )
                # learn batch and device info from reference agent pose.
                self.batch = msg[key][0].shape[0]
                self.dtype = msg[key][0].dtype
                self.device = msg[key][0].device
                self.controller.entity_dict[key].size = self.get_object_xyz_size(key)


            else:
                self.controller.entity_dict[key].updateTransform(
                    Transform(make_pose(msg[key][0], matrix_from_quat(msg[key][1])))
                )

        for key in msg_s:
            if key not in self.controller.entity_dict:
                # If the key is not in the sensor dictionary, add it
                self.controller.entity_dict[key] = SensedEntity(
                    key,
                    msg_s[key][3][0],
                    msg_s[key][3][1],
                    Transform(
                        make_pose(msg_s[key][0], matrix_from_quat(msg_s[key][1]))
                    ),
                )
            else:
                self.controller.entity_dict[key].updateTransform(
                    Transform(make_pose(msg_s[key][0], matrix_from_quat(msg_s[key][1])))
                )

        return

    def setPOVAgent(self, agent_name):
        self.reference_agent = agent_name

    def updatePOVState(self):
        for key in self.controller.entity_dict:
            self.controller.entity_dict[key].saveTransform()

        self.controller.save_odometry_checkpoint()
        self.state_current = torch.concat(self.getControllerState())
        self._joint_current = copy.deepcopy(self.joint_dict)
        self._joint_stall = copy.deepcopy(self.joint_dict)
        self.gripper_stall_counter = None
        self._gripper_stall_state = None

    def countStall(self, threshold: float = 1e-2) -> torch.Tensor:
        """
        Count how many consecutive steps the manipulator joint positions have remained
        approximately unchanged.

        Returns a per-environment counter tensor for the full batch (num_envs,).
        This function does not apply any batch mask; callers can mask the returned
        tensor if needed.
        """
        if self.reference_agent is None:
            raise RuntimeError("reference_agent not set. Call setPOVAgent before countStall.")

        rt_full = self.joint_dict[self.reference_agent]["joint_position"][:, :7]  # (N,7)
        st_full = self._joint_stall[self.reference_agent]["joint_position"][:, :7]  # (N,7)

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

    def _normalized_gripper_position(self) -> torch.Tensor:
        if not self.controller.has_gripper:
            raise AttributeError("Controller does not have gripper state information.")
        n_gripper = self._gripper_opened.shape[0]
        finger_pos = self.joint_dict[self.reference_agent]["joint_position"][
            :, -n_gripper:
        ]
        gripper_range = self._gripper_opened - self._gripper_closed
        return (finger_pos - self._gripper_closed) / gripper_range

    def countGripperStall(self, threshold: float = 1e-4) -> torch.Tensor:
        """
        Count how many consecutive steps the gripper joints have remained
        approximately unchanged in normalized open/closed coordinates.
        """
        if self.reference_agent is None:
            raise RuntimeError(
                "reference_agent not set. Call setPOVAgent before countGripperStall."
            )
        if not self.controller.has_gripper:
            joint_pos = self.joint_dict[self.reference_agent]["joint_position"]
            return torch.zeros(
                joint_pos.shape[0], dtype=torch.int64, device=joint_pos.device
            )

        rt_full = self._normalized_gripper_position()
        if (
            self.gripper_stall_counter is None
            or self._gripper_stall_state is None
            or self._gripper_stall_state.shape != rt_full.shape
            or self._gripper_stall_state.device != rt_full.device
        ):
            self.gripper_stall_counter = torch.zeros(
                rt_full.shape[0], dtype=torch.int64, device=rt_full.device
            )
            self._gripper_stall_state = rt_full.clone().detach()

        diff = torch.sum((rt_full - self._gripper_stall_state) ** 2, dim=-1)
        stable = diff < threshold

        c = self.gripper_stall_counter
        c_new = torch.where(stable, c + 1, torch.where(c >= 2, c - 2, c))
        self.gripper_stall_counter = torch.clamp(c_new, 0, 30)

        if torch.any(~stable):
            moving = ~stable
            self._gripper_stall_state[moving] = rt_full[moving]

        return self.gripper_stall_counter

    def countEffortStall(
        self,
        effort_threshold: float = 40.0,
        min_effort_stall_steps: int = 10,
    ) -> torch.Tensor:
        """
        Count consecutive updates whose total arm effort exceeds the threshold.

        Return one counter per environment, capped at min_effort_stall_steps.
        An update at or below the threshold resets that environment's counter.
        """
        if self.reference_agent is None:
            raise RuntimeError(
                "reference_agent not set. Call setPOVAgent before countEffortStall."
            )

        rt_full = self.joint_dict[self.reference_agent]["joint_effort"][:, :7]  # (N,7)

        if self.stall_cost_counter is None:
            N = int(rt_full.shape[0])
            self.stall_cost_counter = torch.zeros(
                N, dtype=torch.int64, device=rt_full.device
            )

        total_eff = self.manipulator_effort()
        stable = total_eff > effort_threshold


        c = self.stall_cost_counter
        c_new = torch.where(stable, c + 1, torch.zeros_like(c))
        self.stall_cost_counter = torch.clamp(c_new, 0, min_effort_stall_steps)


        return self.stall_cost_counter

    def get_all_residual_apis(self):
        apis = {
            "get_object_xyz_position": self.get_object_xyz_position_casadi,
            "get_object_wxyz_quaternion": self.get_object_wxyz_quaternion_casadi,
            "get_object_xyz_size": self.get_object_xyz_size_casadi,
            "quat_to_matrix": quat_to_matrix_casadi,
            "quat_mul": quat_mul_casadi,
            "quat_from_euler_xyz": quat_from_euler_xyz_casadi,
            "inverse_quat": inverse_quat_casadi,
            "quat_conj": inverse_quat_casadi,
            "x_axis_from_quat": x_axis_from_quat_casadi,
            "y_axis_from_quat": y_axis_from_quat_casadi,
            "z_axis_from_quat": z_axis_from_quat_casadi,
            "object_to_world_xyz_position": self.object_to_world_xyz_position_casadi,
            "world_to_object_xyz_position": self.world_to_object_xyz_position_casadi,
            "world_to_object_wxyz_quaternion": self.world_to_object_wxyz_quaternion_casadi,
            "object_to_world_wxyz_quaternion": self.object_to_world_wxyz_quaternion_casadi,
        }

        if issubclass(type(self.controller), ManipulatorPositionController):
            apis.update(
                {
                    "calculate_robot_frame_position": self.calculate_frame_position_casadi,
                    "calculate_robot_frame_rotation": self.calculate_frame_rotation_casadi,
                }
            )

        if issubclass(type(self.controller), ManipulatorIKController):
            apis.update(
                {
                    "lookup_robot_frame_position": self.lookup_robot_frame_position_casadi,
                    "lookup_robot_frame_rotation": self.lookup_robot_frame_rotation_casadi,
                }
            )


        return apis

    def euler_xyz_from_quat_batched(self, q: torch.Tensor) -> torch.Tensor:
        return torch.column_stack(euler_xyz_from_quat(q))

    def get_all_predicate_apis(self):
        apis = {
            "get_object_xyz_position": self.get_object_xyz_position,
            "get_object_wxyz_quaternion": self.get_object_wxyz_quaternion,
            "get_object_xyz_size": self.get_object_xyz_size,
            "quat_to_matrix": quat_to_matrix,
            "euler_xyz_from_quat": self.euler_xyz_from_quat_batched,
            "quat_from_euler_xyz": quat_from_euler_xyz,
            "quat_from_angle_axis": quat_from_angle_axis,
            "x_axis_from_quat": x_axis_from_quat,
            "y_axis_from_quat": y_axis_from_quat,
            "z_axis_from_quat": z_axis_from_quat,
            "quat_conj": inverse_quat,
            "codirectional_angle_between_vectors": codirectional_angle_between_vectors,
            "antiparallel_angle_between_vectors": antiparallel_angle_between_vectors,
            "bipolar_angle_between_vector_and_line": bipolar_angle_between_vector_and_line,
            "angle_between_vectors": angle_between_vectors,
            "orthogonality_angle_between_vector_and_line": orthogonality_angle_between_vector_and_line,
            "object_to_world_xyz_position": self.object_to_world_xyz_position,
            "world_to_object_xyz_position": self.world_to_object_xyz_position,
            "world_to_object_wxyz_quaternion": self.world_to_object_wxyz_quaternion,
            "object_to_world_wxyz_quaternion": self.object_to_world_wxyz_quaternion,
            "quat_multiply_pos": quat_multiply_pos
        }
        if hasattr(self.controller, "forward_kinematics"):
            apis.update(
                {
                    "lookup_robot_frame_position": self.lookup_robot_frame_position,
                    "lookup_robot_frame_rotation": self.lookup_robot_frame_rotation,
                    "manipulator_effort": self.manipulator_effort,
                    "high_effort_contact": self.high_effort_contact,
                }
            )
        if self.controller.has_gripper:
            finger_contact_force = self.finger_contact_force
            apis.update(
                {
                    "gripper_openness": self.gripper_openness,
                    "gripper_force": self.gripper_force,
                    "finger_contact_force": finger_contact_force,
                }
            )
        # Robot-dependent gripper max force injected into predicate scope.
        apis["GRIPPER_MAX_FORCE"] = self._gripper_max_force
        apis["GRIPPER_MAX_GRASP_OPENNESS"] = self._gripper_max_grasp_openness
        apis["ROBOT_GRIPPER_NAME"] = self._robot_gripper_name
        return apis

    def update_object_bounding_boxes(self):
        self.cache = bounds_utils.create_bbox_cache()

    def gripper_openness(self) -> torch.Tensor:
        """
        Check the state of the gripper.
        :return: 1.0 if the gripper is fully open, 0.0 if the gripper is fully closed
        """
        if not self.controller.has_gripper:
            raise AttributeError("Controller does not have gripper state information.")
        # Per-finger normalization using tensor gripper limits, then average.
        n_gripper = self._gripper_opened.shape[0]
        finger_pos = self.joint_dict[self.reference_agent]["joint_position"][:, -n_gripper:]
        gripper_range = self._gripper_opened - self._gripper_closed  # (n_gripper,)
        result = ((finger_pos - self._gripper_closed) / gripper_range).mean(dim=-1)
        return result

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

    def gripper_force(self) -> torch.Tensor:
        """Return the larger of the two finger joint efforts per environment."""

        effort = self.joint_dict[self.reference_agent]["joint_effort"][:, -2:]
        return torch.maximum(effort[:,0], effort[:,1])


    def _uses_joint_effort_finger_contact(self) -> bool:
        return self._robot_gripper_name in (
            "kinova_robotiq",
            "ur5_robotiq",
            "franka_panda",
        )

    def _joint_effort_finger_contact_force(self) -> tuple[torch.Tensor, torch.Tensor]:
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
        return effort[:, 0], effort[:, 1]

    def finger_contact_force(self, left_key: str = "contact_sensor_left", right_key: str = "contact_sensor_right") -> tuple:
        """Return contact force magnitudes for left and right finger tips.

        Returns:
            (left_mag, right_mag): each (N,) tensor with force magnitude per environment.
            Returns zeros if contact data has not been fed via callbackContactForces yet.
        """
        if self._uses_joint_effort_finger_contact():
            return self._joint_effort_finger_contact_force()

        batch = self.joint_dict[self.reference_agent]["joint_position"].shape[0]
        device = self.joint_dict[self.reference_agent]["joint_position"].device

        # Get the fingertip y-axis in world frame — this is the closing axis for both fingers.
        ee_quat = self.lookup_robot_frame_rotation(self.controller.ee_frame)  # (N, 4) wxyz
        closing_axis = y_axis_from_quat(ee_quat)  # (N, 3)

        def _closing_force(key, sign):
            # Project world-frame contact force onto the closing axis, then gate by direction.
            if key in self.contact_forces:
                projection = (self.contact_forces[key] * closing_axis).sum(dim=-1)  # (N,)
                return torch.clamp(sign * projection, min=0.0)
            return torch.zeros(batch, device=device)

        left_sign = self._finger_contact_signs[0].item()
        right_sign = self._finger_contact_signs[1].item()
        return _closing_force(left_key, left_sign), _closing_force(right_key, right_sign)

    def manipulator_effort(self, time_stamp="real time") -> torch.Tensor:
        """
        check the total force applied by the manipulator joints not including the gripper
        :return above 40 if the manipulator is pushing against an object
        """

        if time_stamp == "real time":
            return torch.sum(
                torch.abs(self.joint_dict[self.reference_agent]["joint_effort"][:, :7]),
                dim=-1,
            )
        else:
            return torch.sum(
                torch.abs(
                    self._joint_current[self.reference_agent]["joint_effort"][:, :7]
                ),
                dim=-1,
            )

    def high_effort_contact(
        self,
        min_position_stall_steps: int = 20,
        min_effort_stall_steps: int = 5,
        position_similarity_threshold: float = 0.01,
        effort_magnitude_threshold: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Return whether environment zero has sustained high arm effort.

        The effort must exceed the threshold for min_effort_stall_steps updates.
        Position-stall arguments are retained for caller compatibility; this
        check does not test position stability.
        """
        if effort_magnitude_threshold is None:
            effort_magnitude_threshold = self._manipulator_high_effort_threshold

        effort_stall_steps_full = self.countEffortStall(
            effort_threshold=effort_magnitude_threshold,
            min_effort_stall_steps=min_effort_stall_steps,
        )  # (N,)

        eff_ok_full = effort_stall_steps_full >= min_effort_stall_steps

        return eff_ok_full[0]

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

    def get_object_xyz_size(self, a: str, time_stamp="NOT IMPORTANT") -> np.ndarray:
        """
        Get the XYZ size of an object.
        :param a: Name of the object.
        :return: NumPy array of shape (3, 1)
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

    def calculate_robot_frame_position(
        self, frame_name: str, joint_state: np.ndarray
    ) -> np.ndarray:
        """
        Calculate the position of a frame on the robot using kinematics.
        :param frame_name: Name of the frame.
        :param joint_state: Joint state of the robot.
        :return: Position of the frame in the robot's base frame.
        """
        self.controller.forward_kinematics(joint_state)
        return self.lookup_robot_frame_position(frame_name)

    def calculate_robot_frame_rotation(
        self, frame_name: str, joint_state: np.ndarray
    ) -> np.ndarray:
        """
        Calculate the position of a frame on the robot using kinematics.
        :param frame_name: Name of the frame.
        :param joint_state: Joint state of the robot.
        :return: Position of the frame in the robot's base frame.
        """
        self.controller.forward_kinematics(joint_state)
        return self.lookup_robot_frame_rotation(frame_name)

    def calculate_frame_position_casadi(
        self, frame_name: str, joint_state: np.ndarray
    ) -> np.ndarray:
        """
        Calculate the position of a frame on the robot using kinematics.
        :param frame_name: Name of the frame.
        :param joint_state: Joint state of the robot.
        :return: Position of the frame in the robot's base frame.
        """

        return self.controller.calculate_frame_position_casadi(frame_name, joint_state)

    def calculate_frame_rotation_casadi(
        self, frame_name: str, joint_state: np.ndarray
    ) -> ca.DM:
        """
        Calculate the position of a frame on the robot using kinematics.
        :param frame_name: Name of the frame.
        :param joint_state: Joint state of the robot.
        :return: Position of the frame in the robot's base frame.
        """
        return self.controller.calculate_frame_rotation_casadi(frame_name, joint_state)

    def lookup_robot_frame_position(
        self, frame_name: str, timestamp="real time"
    ) -> torch.Tensor:
        """
        Calculate the position of a frame on the robot using kinematics.
        :param frame_name: Name of the frame.
        :param joint_state: Joint state of the robot.
        :return: Position of the frame in the robot's base frame.
        """
        return self.controller.lookup_frame_position(frame_name, timestamp)

    def lookup_robot_frame_rotation(
        self, frame_name: str, timestamp="real time"
    ) -> torch.Tensor:
        """
        Calculate the position of a frame on the robot using kinematics.
        :param frame_name: Name of the frame.
        :param joint_state: Joint state of the robot.
        :return: Position of the frame in the robot's base frame.
        """
        return self.controller.lookup_frame_rotation(frame_name, timestamp)

    def lookup_robot_frame_position_casadi(
        self, frame_name: str, timestamp="real time"
    ) -> ca.DM:

        return self.controller.lookup_frame_position_casadi(frame_name, timestamp)

    def lookup_robot_frame_rotation_casadi(
        self, frame_name: str, timestamp="real time"
    ) -> ca.DM:
        return self.controller.lookup_frame_rotation_casadi(frame_name, timestamp)

    def get_all_residual_apis_description(self):

        apis = {
            "get_object_xyz_position(object_str, time_str)": "look up the (3) position of a visible or previously seen non-robot object, unit: m",
            "get_object_wxyz_quaternion(object_str, time_str)": "look up the (4) quaternion of a visible or previously seen non-robot object, unit: none",
            "quat_to_matrix(q)": "convert wxyz quaternion ( 4) to (3,3) rotation matrix",
            "quat_mul(q0,q1)": "multiply two wxyz quaternions ( 4) and ( 4) to get ( 4) quaternion",
            "quat_from_euler_xyz(xyz)": "convert euler angles ( 3) to wxyz quaternion ( 4)",
            "inverse_quat(q)": "calculate the inverse of wxyz quaternion ( 4)",
            "x_axis_from_quat(q)": "extract x-axis ( 3) vector from (4) quaternion (w, x, y, z)",
            "y_axis_from_quat(q)": "extract y-axis ( 3) vector from (4) quaternion (w, x, y, z)",
            "z_axis_from_quat(q)": "extract z-axis ( 3) vector from (4) quaternion (w, x, y, z)",
            "get_object_xyz_size(object_str)": "look up the (3) size of a visible or previously seen non-robot object, unit: m",
            "object_to_world_xyz_position(object_str, local_xyz_pos, time_str)": "transform a (3) position in object frame in the specified timestamp to a (3) position in world frame",
            "world_to_object_xyz_position(object_str, local_xyz_pos, time_str)": "transform a (3) position in world frame to a (3) position in object frame in the specified timestamp",
            "world_to_object_wxyz_quaternion(object_str, local_wxyz_quat, time_str)": "transform a (4) quaternion in world frame to a (4) quaternion in object frame in the specified timestamp",
            "object_to_world_wxyz_quaternion(object_str, local_wxyz_quat, time_str)": "transform a (4) quaternion in object frame in the specified timestamp to a (4) quaternion in world frame",
        }

        if issubclass(type(self.controller), ManipulatorPositionController):
            apis.update(
                {
                    "calculate_robot_frame_position(object_str, joint_state)": "calculate the (3,1) position of a frame on robot using kinematics, unit: m",
                    "calculate_robot_frame_rotation(object_str, joint_state)": "calculate the (4) quaternion of a frame on robot using kinematics",
                }
            )


        return apis

    def get_all_predicate_apis_description(self):

        apis = {
            "get_object_xyz_position(object_str,time_str)": "look up the real time (batch,3) position of a visible or previously seen non-robot object, unit: m",
            "get_object_wxyz_quaternion(object_str,time_str)": "look up the quaternion of a visible or previously seen object, unit: rad",
            "quat_to_matrix(q)": "convert wxyz quaternion (batch, 4) to (batch,3,3) rotation matrix",
            "euler_xyz_from_quat(q)": "convert wxyz quaternion (batch, 4) to euler angles (batch,3) in radians",
            "x_axis_from_quat(q)": "extract x-axis (batch,3) vector from (batch,4) quaternion (w, x, y, z)",
            "y_axis_from_quat(q)": "extract y-axis (batch,3) vector from (batch,4) quaternion (w, x, y, z)",
            "z_axis_from_quat(q)": "extract z-axis (batch,3) vector from (batch,4) quaternion (w, x, y, z)",
            "get_object_xyz_size(object_str)": "look up the (3) size of a visible or previously seen non-robot object, NOT BATCHED, unit: m",
            "codirectional_angle_between_vectors(a, b)": "calculate the angle (batch,1) between vector a (batch,3) and vector b (batch,3) that are meant to be codirectional (radians)",
            "antiparallel_angle_between_vectors(a, b)": "calculate the angle (batch,1) between vector a (batch,3) and vector b (batch,3) that are meant to be antiparallel (radians)",
            "bipolar_angle_between_vector_and_line(a, b)": "calculate the angle (batch,1) between vector a (batch,3) and line represented by b (batch,3) that can be either codirectional or antiparallel (radians)",
            "orthogonality_angle_between_vector_and_line(a, b)": "Smallest angular deviation from 90° (batch,1) between vector a (batch,3) and the line defined by ±b (batch,3) (radians)",
            "object_to_world_xyz_position(object_str, local_xyz_pos, time_str)": "transform a (batch,3) position in object frame to a (batch,3) position in world frame",
            "world_to_object_xyz_position(object_str, world_xyz_pos, time_str)": "transform a (batch,3) position in world frame to a (batch,3) position in object frame",
            "world_to_object_wxyz_quaternion(object_str, world_wxyz_quat, time_str)": "transform a (batch,4) quaternion in world frame to a (batch,4) quaternion in object frame",
            "object_to_world_wxyz_quaternion(object_str, local_wxyz_quat, time_str)": "transform a (batch,4) quaternion in object frame to a (batch,4) quaternion in world frame",
        }

        if issubclass(type(self.controller), ManipulatorPositionController):
            apis.update(
                {
                    "calculate_robot_frame_position(object_str, joint_state)": "calculate the (batch,3) position of a frame on robot using kinematics, unit: m",
                    "calculate_robot_frame_rotation(object_str, joint_state)": "calculate the (batch,4) quaternion of a frame on robot using kinematics",
                    "manipulator_effort(time_stamp)": "look up the time stamped total effort (batch, 1) applied by the manipulator joints not including the gripper, above 25 if the manipulator is actively in contact with another object not in gripper",
                }
            )
        if issubclass(type(self.controller), ManipulatorIKController) or issubclass(
            type(self.controller), DifferentialDriveController
        ):
            apis.update(
                {
                    "lookup_robot_frame_position(object_str, time_str)": "look up the (batch,3) position of a frame on robot using kinematics, unit: m",
                    "lookup_robot_frame_rotation(object_str)": "look up the (batch,4) quaternion of a frame on robot using kinematics",
                    "manipulator_effort(time_stamp)": "Use to check whether the manipulator is in active contact with another object not in gripper. Time stamped total effort (batch, 1) applied by the manipulator joints not including the gripper, above 10 if the manipulator is actively in contact",
                }
            )

        if self.controller.has_gripper:
            apis.update(
                {
                    "gripper_openness()": "Measured gripper joint (batch, 1), fully open is > 0.9, cannot fully close when gripping an object",
                    "gripper_force()": "Measured amount of force (batch, 1) applied by the gripper, above 20 if the grippers are closed or grasping an object. No effort (open) is close to 0.0",
                }
            )

        return apis

    def inverse_transformation(self, trans: Transform) -> Transform:
        return trans.inverse

    def get_object_transformation(self, a, timestamp="real time") -> Transform:
        if not self.check_known_object(a):
            raise KeyError(f"Object {a} doesn't exist in the scene and cannot be used as a root.")

        trans = self.get_raw_transformation(a, timestamp)

        return (
            self.get_raw_transformation(self.reference_agent, timestamp).inverse @ trans
        )

    def get_raw_transformation(self, a, timestamp="real time"):
        if a not in self.controller.entity_dict:
            raise KeyError(f"Object {a} doesn't exist in the scene!")
        if timestamp == "real time":
            if self.controller.entity_dict[a].transform_real_time is not None:
                return self.controller.entity_dict[a].transform_real_time
            else:
                raise KeyError(
                    f"Object {a} is not an object. If it is a robot frame, please use lookup_robot_frame APIs."
                )
        elif timestamp == "current":
            return self.controller.entity_dict[a].transform_current
        else:
            raise KeyError(f"Unexpected look up for Object {a} ")

    def check_known_object(self, a) -> bool:
        return a in self.controller.entity_dict

    def is_robot_frame(self, a) -> bool:
        frame_names = [f.name for f in self.controller.getFrameDescription()]
        return a in frame_names

    def get_object_xyz_position(self, a, timestamp="real time") -> torch.Tensor:
        trans = self.get_object_transformation(a, timestamp)
        return trans.translation  # add an extra dim for (3,1)

    def get_object_xyz_euler(self, a, timestamp="real time") -> torch.Tensor:
        trans = self.get_object_transformation(a, timestamp)
        return euler_xyz_from_quat(quat_from_matrix(trans.rotation))

    def get_object_wxyz_quaternion(self, a, timestamp="real time") -> torch.Tensor:
        trans = self.get_object_transformation(a, timestamp)
        return trans.quaternion

    def get_object_xyz_position_casadi(self, a, timestamp="real time") -> ca.DM:
        return ca.DM(self.get_object_xyz_position(a, timestamp)[0].cpu().numpy())

    def get_object_xyz_euler_casadi(self, a, timestamp="real time") -> ca.DM:
        return ca.DM(self.get_object_xyz_euler(a, timestamp)[0].cpu().numpy())

    def get_object_wxyz_quaternion_casadi(self, a, timestamp="real time") -> ca.DM:
        return ca.DM(self.get_object_wxyz_quaternion(a, timestamp)[0].cpu().numpy())

    def get_object_xyz_size_casadi(self, a, timestamp="NOT IMPORTANT") -> ca.DM:
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
        return ca.vertcat(half_extent * 2)

    def object_to_world_xyz_position_casadi(
        self, a, local_pos, timestamp="real time"
    ) -> ca.DM:
        rot = quat_to_matrix_casadi(
            self.get_object_wxyz_quaternion_casadi(a, timestamp)
        )
        pos = self.get_object_xyz_position_casadi(a, timestamp)
        xyz_pos_world = rot @ local_pos + pos
        return xyz_pos_world

    def object_to_world_xyz_position(
        self, a, local_pos, timestamp="real time"
    ) -> torch.Tensor:
        """
        Transform a (batch,3) position in object frame to a (batch,3) position in world frame.
        """
        q = self.get_object_wxyz_quaternion(a, timestamp)
        pos = self.get_object_xyz_position(a, timestamp)
        xyz_pos_world = quat_multiply_pos(q, local_pos) + pos
        return xyz_pos_world

    def world_to_object_xyz_position_casadi(
        self, a, world_pos, timestamp="real time"
    ) -> ca.DM:
        """
        Transform a (batch,3) position in world frame to a (batch,3) position in object frame.
        """
        trans = self.get_object_transformation(a, timestamp).inverse
        q = ca.vertcat(trans.quaternion[0].cpu().numpy())
        p = ca.vertcat(trans.translation[0].cpu().numpy())
        R = quat_to_matrix_casadi(q)
        xyz_pos_object = R @ world_pos + p
        return xyz_pos_object

    def world_to_object_xyz_position(
        self, a, xyz_pos, timestamp="real time"
    ) -> torch.Tensor:
        """
        Transform a (batch,3) position in world frame to a (batch,3) position in object frame.
        """
        trans = self.get_object_transformation(a, timestamp).inverse
        xyz_pos_object = (
            quat_multiply_pos(trans.quaternion, xyz_pos) + trans.translation
        )
        return xyz_pos_object

    def world_to_object_wxyz_quaternion_casadi(
        self, a, world_quat, timestamp="real time"
    ) -> ca.DM:
        """
        Transform a (batch,4) quaternion in world frame to a (batch,4) quaternion in object frame.
        """
        trans = self.get_object_transformation(a, timestamp).inverse
        rot = ca.vertcat(trans.quaternion[0].cpu().numpy())
        local_rotation = quat_mul_casadi(rot, world_quat)
        return local_rotation

    def world_to_object_wxyz_quaternion(self, a, world_quat, timestamp="real time"):
        """
        Transform a (batch,4) quaternion in world frame to a (batch,4) quaternion in object frame.
        """
        trans = self.get_object_transformation(a, timestamp).inverse
        rot = quat_to_matrix(trans.quaternion)
        word_rot = quat_to_matrix(world_quat)
        local_rotation = rot @ word_rot
        return quat_from_matrix(local_rotation)

    def object_to_world_wxyz_quaternion_casadi(
        self, a, local_quat, timestamp="real time"
    ) -> ca.DM:
        """
        Transform a (batch,4) quaternion in object frame to a (batch,4) quaternion in world frame.
        """
        world_quat = (
            self.object_to_world_wxyz_quaternion(a, local_quat, timestamp)[0]
            .cpu()
            .numpy()
        )

        quat = self.get_object_wxyz_quaternion_casadi(a, timestamp)
        local_rotation = quat_mul_casadi(quat, local_quat)

        return local_rotation

    def object_to_world_wxyz_quaternion(self, a, local_quat, timestamp="real time"):
        """
        Transform a (batch,4) quaternion in object frame to a (batch,4) quaternion in world frame.
        """
        trans = self.get_object_transformation(a, timestamp)
        rot = quat_to_matrix(trans.quaternion)
        local_rot = quat_to_matrix(local_quat)
        world_rot = rot @ local_rot
        return quat_from_matrix(world_rot)

    def get_previously_seen_objects(self) -> Set[VisibleEntity]:
        seen_objects = set(self.controller.entity_dict.keys())
        return seen_objects

    def get_previously_seen_entities(self) -> Set[VisibleEntity]:
        seen_objects = copy.deepcopy(self.controller.entity_dict)
        for f in self.controller.getFrameDescription():
            seen_objects.pop(f.name, None)
        if "agent" in seen_objects:
            seen_objects.pop("agent", None)
        return set(seen_objects.values())

    def process_control(self, u: np.ndarray):
        """
        Process the control input and apply it to the controller.
        :param u: Control input.
        """
        if self.controller.has_gripper:
            n_gripper = self._gripper_opened.shape[0]
            if np.mean(u[-n_gripper:]) > 0.5:
                u[-n_gripper:] = self._gripper_opened.cpu().numpy()
            else:
                u[-n_gripper:] = self._gripper_closed.cpu().numpy()
        return u
