import ast
import json
import os
import logging
from collections import deque
from pathlib import Path
from copy import deepcopy
import torch
from typing import Optional, Dict, Any
from omegaconf import OmegaConf, DictConfig
from scipy.spatial.transform import Rotation as R
from kuafu.mpc.simulation.spatial_utils import Transform, Twist, make_pose, matrix_from_quat
from kuafu.mpc.simulation.api_wrapper import APIWrapper
from kuafu.mpc.controller import mpc
from kuafu.programs.sun_program_learner import SUNProgramLearner
from kuafu.programs.llm.utils import compile_observation
from kuafu.programs.llm.io_util import subfolders_with_prefix

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class SUNProgramExecutor(object):
    """Ground and execute SUN Program stages with MPC and recovery support."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.picture_idx = 0

        self.device = cfg.device

        try:
            cls = getattr(mpc, cfg.controller_name)
        except AttributeError:
            raise ValueError(f"models.{cfg.controller_name} not found")
        mpc_controller = cls(**cfg.controller, device=self.device)
        control_output_mode = getattr(cfg, "control_output_mode", "relative_joint")
        mpc_controller.control_output_mode = control_output_mode

        self.active_controller = mpc_controller
        self.test_controller = deepcopy(mpc_controller)

        # Read gripper joint limits from config (tensor-valued).
        # Defaults to Panda-style [0.04, 0.04] / [0.0, 0.0] when not specified.
        gripper_opened = getattr(cfg, "gripper_opened", None)
        gripper_closed = getattr(cfg, "gripper_closed", None)
        gripper_max_force = getattr(cfg, "gripper_max_force", 40.0)
        gripper_max_grasp_openness = getattr(cfg, "gripper_max_grasp_openness", 0.8)
        gripper_effort_contact_velocity_threshold = getattr(
            cfg, "gripper_effort_contact_velocity_threshold", 0.2
        )
        robot_gripper_name = getattr(cfg, "robot_gripper_name", "contact_sensor")
        kinova_gripper_effort_contact_openness_slope = getattr(
            cfg, "kinova_gripper_effort_contact_openness_slope", 0.0
        )
        kinova_gripper_effort_contact_openness_intercept = getattr(
            cfg, "kinova_gripper_effort_contact_openness_intercept", 0.0
        )
        has_finger_contact_sensors = getattr(cfg, "has_finger_contact_sensors", True)
        manipulator_high_effort_threshold = getattr(
            cfg, "manipulator_high_effort_threshold", 8.0
        )
        self._manipulator_high_effort_threshold = float(
            manipulator_high_effort_threshold
        )
        self.controller_api_wrapper = APIWrapper(
            controller=self.active_controller, device=self.device,
            gripper_opened=gripper_opened,
            gripper_closed=gripper_closed,
            gripper_max_force=gripper_max_force,
            gripper_max_grasp_openness=gripper_max_grasp_openness,
            gripper_effort_contact_velocity_threshold=gripper_effort_contact_velocity_threshold,
            robot_gripper_name=robot_gripper_name,
            kinova_gripper_effort_contact_openness_slope=kinova_gripper_effort_contact_openness_slope,
            kinova_gripper_effort_contact_openness_intercept=kinova_gripper_effort_contact_openness_intercept,
            has_finger_contact_sensors=has_finger_contact_sensors,
            manipulator_high_effort_threshold=manipulator_high_effort_threshold,
        )
        self.controller_api_wrapper.setPOVAgent(cfg.agent.id)

        self.joint_state = None
        self.joint_target = None
        self.image = deque(maxlen=10)
        self.seg = deque(maxlen=10)
        # Low-pass filter state for joint_effort (exponential moving average).
        # alpha=1.0 means no filtering; lower values smooth more aggressively.
        self._effort_lpf_alpha: float = getattr(cfg, "effort_lpf_alpha", 0.2)
        self._filtered_joint_effort: Dict[str, torch.Tensor] = {}

        # Simple, single-stage residual weights: default 1.0 per constraint.
        # This mirrors the SUNProgramEvaluator API enough for tooling that inspects or
        # edits constraint weights, without changing the controller’s cost
        # function semantics.
        self._residual_weights: Dict[int, Any] = {}
        self._object_grasp_hold_steps: int = int(
            getattr(cfg, "object_grasp_hold_steps", 5)
        )
        self._object_grasp_hold_counter: Optional[torch.Tensor] = None

        self.program_learner = SUNProgramLearner(self, cfg.agent)
        self.program_stage_dirs = []

        if cfg.load_program:
            self.load_program(cfg.load_program)

    def hasStalled(self, threshold=20):
        """
        Check if the controller has stalled.

        For active gripper stages, arm stall alone is not enough: frozen-pose
        grasp stages intentionally stop the manipulator while the gripper closes.
        """
        manipulator_stalled = self.controller_api_wrapper.countStall() >= threshold
        if (
            not self._active_constraints_use_gripper()
            or not self.active_controller.has_gripper
        ):
            return manipulator_stalled

        gripper_stalled = self.controller_api_wrapper.countGripperStall() >= threshold
        return manipulator_stalled & gripper_stalled

    def stallFailureReason(self) -> str:
        if (
            self._active_constraints_use_gripper()
            and self.active_controller.has_gripper
        ):
            return "controller stalled (manipulator and gripper stalled)"
        return "controller stalled (manipulator stalled)"

    def hasCollision(self):
        """
        Check if the controller has collided.
        """
        return torch.any(
            self.controller_api_wrapper.manipulator_effort()
            > self._manipulator_high_effort_threshold
        )

    def terminateCurrentSubtask(self):
        """
        Terminate the current subtask, this is used to indicate the end of a subtask.
        """
        self.controller_api_wrapper.updatePOVState()
        self._reset_object_grasp_hold()
        if len(self.program_stage_dirs) > 0:
            next_program = self.program_stage_dirs.pop(0)
            logger.info(f"Loading program from {next_program}")
            self.program_learner.load_program(
                os.path.join(next_program, "residual.py"),
                os.path.join(next_program, "predicate.py"),
            )
            self.program_learner.load_agent_literal(
                os.path.join(next_program, "literal.py"),
            )
        else:
            if self.program_learner.terminate_current_task():
                logger.info("Moved on to the next subtask")
                logger.info(self.program_learner._constraint_generator._tasks.has_tasks)
            else:
                logger.info("No more subtasks, terminating the agent")

    def hasMoreTasks(self):
        return self.program_learner.has_tasks()

    def record_program(
        self,
        save_dir_task: str | Path,
        save_dir_atomic: str | Path,
        save_dir_constraint: str | Path,
    ):
        self.program_learner.save_program(
            save_dir_task, save_dir_atomic, save_dir_constraint
        )

    def quantifyPredicate(self, text: bool = False, max_envs: Optional[int] = None):
        result = self.program_learner.detectPredicates(text=text, max_envs=max_envs)
        if text:
            predicate_value, predicate_text = result
            return self._apply_object_grasp_hold(predicate_value), predicate_text
        return self._apply_object_grasp_hold(result)

    def _active_constraints_use_object_grasped(self) -> bool:
        active = getattr(
            self.program_learner._constraint_generator, "_active_constraints", {}
        )
        constraints = active.get("Constraints", [])
        return any(term.get("op") == "object_grasped" for term in constraints)

    def _active_constraints_use_gripper(self) -> bool:
        active = getattr(
            self.program_learner._constraint_generator, "_active_constraints", {}
        )
        constraints = active.get("Constraints", [])
        gripper_ops = {"open_gripper", "close_gripper", "object_grasped"}
        return any(
            str(term.get("op", "")).strip().lower() in gripper_ops
            for term in constraints
        )

    def _reset_object_grasp_hold(self) -> None:
        if self._object_grasp_hold_counter is not None:
            self._object_grasp_hold_counter.zero_()

    def _ensure_object_grasp_hold_counter(
        self, predicate_value: torch.Tensor
    ) -> torch.Tensor:
        flat_predicate = predicate_value.view(-1)
        if (
            self._object_grasp_hold_counter is None
            or self._object_grasp_hold_counter.shape[0] != flat_predicate.numel()
            or self._object_grasp_hold_counter.device != flat_predicate.device
        ):
            self._object_grasp_hold_counter = torch.zeros(
                flat_predicate.numel(),
                dtype=torch.int64,
                device=flat_predicate.device,
            )
        return self._object_grasp_hold_counter

    def _apply_object_grasp_hold(self, predicate_value: torch.Tensor) -> torch.Tensor:
        flat_predicate = predicate_value.view(-1).to(dtype=torch.bool)
        if not self._active_constraints_use_object_grasped():
            self._reset_object_grasp_hold()
            return flat_predicate.reshape(predicate_value.shape)

        counter = self._ensure_object_grasp_hold_counter(predicate_value)
        counter[:] = torch.where(
            flat_predicate,
            counter + 1,
            torch.zeros_like(counter),
        )
        return (counter >= self._object_grasp_hold_steps).reshape(predicate_value.shape)

    def _active_gripper_override(self) -> str | None:
        """
        Check whether the current active constraints include a gripper command.

        Returns
        -------
        ``"close_gripper"`` or ``"open_gripper"`` if either constraint is
        present. Defaults to ``"open_gripper"`` when no gripper constraint is active.
        If *both* appear (unusual), close wins.
        ``"object_grasped"`` is treated identically to ``"close_gripper"``.
        """
        active = getattr(
            self.program_learner._constraint_generator, "_active_constraints", {}
        )
        constraints = active.get("Constraints", [])
        found = "open_gripper"
        for term in constraints:
            op = term.get("op", "")
            if op in ("close_gripper", "object_grasped"):
                return "close_gripper"
            if op == "open_gripper":
                found = "open_gripper"
        return found

    def computeControl(self):
        state = self.controller_api_wrapper.getControllerState()
        x = self.active_controller.getState(*state)
        u = self.active_controller.getControl(*state)
        u_out = self.program_learner.computeControl(x, u)

        # Bypass MPC gripper output when a gripper constraint is active.
        # Gripper commands are open / close only, so we directly overwrite
        # the gripper portion of the trajectory with that command.
        gripper_override = self._active_gripper_override()
        if gripper_override is not None and self.active_controller.has_gripper:
            n_gripper = self.controller_api_wrapper._gripper_opened.shape[0]
            if gripper_override == "close_gripper":
                target = -1.0
            else:
                target = 1.0
            for idx in range(len(u_out)):
                u_out[idx, -n_gripper:] = target

        for idx in range(len(u_out)):
            u_out[idx,:] = self.processControl(
                u_out[idx,:]
            )

        return u_out

    def callbackImage(self, image):
        self.image.append(image)
        if len(self.image) > 10:
            self.image.popleft()

    def callbackOdom(self, msg: dict, msg_s: dict):
        self.controller_api_wrapper.updatemOdom(msg, msg_s)

    def hasKinematics(self):
        """
        Check if the controller has kinematics capabilities.
        """
        return hasattr(self.active_controller, "forward_kinematics")

    def callbackContactForces(self, contact_forces: dict) -> None:
        """Feed finger-tip contact force data into the api_wrapper.

        Args:
            contact_forces: dict mapping sensor name -> (N, 3) tensor of net forces in world frame.
                            Expected keys: "contact_sensor_left", "contact_sensor_right".
        """
        self.controller_api_wrapper.update_contact_forces(contact_forces)

    def callbackJointState(self, msg: dict):
        reference_agent = self.controller_api_wrapper.reference_agent
        alpha = self._effort_lpf_alpha
        for key in msg:
            self.controller_api_wrapper.joint_dict[key] = {}
            for k in msg[key]:
                if k == "joint_effort":
                    raw_effort = msg[key][k]
                    if key in self._filtered_joint_effort:
                        self._filtered_joint_effort[key] = (
                            alpha * raw_effort
                            + (1.0 - alpha) * self._filtered_joint_effort[key]
                        )
                    else:
                        self._filtered_joint_effort[key] = raw_effort.clone()
                    self.controller_api_wrapper.joint_dict[key][k] = (
                        self._filtered_joint_effort[key]
                    )
                else:
                    self.controller_api_wrapper.joint_dict[key][k] = msg[key][k]
        if hasattr(self.active_controller, "forward_kinematics"):
            self.active_controller.forward_kinematics(
                self.controller_api_wrapper.joint_dict[
                    self.controller_api_wrapper.reference_agent
                ]["joint_position"][0]
                .cpu()
                .numpy()
            )
        if (
            self.active_controller.has_gripper
            and reference_agent in self.controller_api_wrapper.joint_dict
        ):
            joint_pos = self.controller_api_wrapper.joint_dict[reference_agent][
                "joint_position"
            ]
            gripper_state = joint_pos[..., -2:]
            if hasattr(gripper_state, "detach"):
                gripper_state = gripper_state.detach().clone()
            else:
                gripper_state = torch.as_tensor(gripper_state).clone()
            self.controller_api_wrapper.joint_dict[reference_agent][
                "gripper_state"
            ] = gripper_state

    def callbackTwist(self, msg: dict):
        for key in msg:
            self.controller_api_wrapper.twist_dict[key] = Twist(msg[key])

    def callbackSegmentation(self, id_mask, id_lookup):
        self.seg.append({"image": id_mask, "id": id_lookup})

    def callbackTask(self):
        self.controller_api_wrapper.updatePOVState()

        if len(self.image) > 1:
            images = [self.image[-1]]
        else:
            images = []

        previous_objects = self.recall_entities()

        num_llm_calls, success = self.program_learner.query_task(
            compile_observation(
                previous_objects,
                include_size=False,
                include_pose=False,
            ),
            images,
            max_iteration=self.cfg.agent.max_correction_steps,
        )
        if not success:
            logger.warning("SUNProgramExecutor task query failed; resuming the previous task")
        return success

    def callbackAtomicActions(self, error: Optional[str] = None):
        """
        Stage 2: Post-Task - Ground the current high-level subtask into atomic actions.
        This should be called after callbackTask() and before callbackPredicate().
        """
        self.controller_api_wrapper.updatePOVState()

        if len(self.image) > 1:
            images = [self.image[-1]]
        else:
            images = []

        previous_objects = self.recall_entities()

        # Call VLM to ground current subtask into atomic actions
        num_llm_calls, success = self.program_learner.query_atomic_actions(
            compile_observation(
                previous_objects,
                include_size=False,
                include_pose=False,
            ),
            images,
            max_iteration=self.cfg.agent.max_correction_steps,
            error=error,
        )

        if not success:
            logger.warning(
                "SUNProgramExecutor atomic-action query failed"
            )

        return success

    def callbackConstraint(self, error: Optional[str] = None):
        images = []
        previous_objects = self.recall_entities()

        num_llm_calls, success = self.program_learner.query_constraint(
            compile_observation(
                previous_objects,
                include_size=False,
                include_pose=False,
            ),
            images,
            max_iteration=self.cfg.agent.max_correction_steps,
            error=error,
        )

        if not success:
            logger.warning(
                "SUNProgramExecutor constraint query failed; resuming the previous predicate"
            )

        # Reset stall counters so they initialize from the new stage state.
        self.controller_api_wrapper.stall_joint_counter = None
        self.controller_api_wrapper.gripper_stall_counter = None
        self.controller_api_wrapper._gripper_stall_state = None
        return success

    def applyConstraint(self, print_function=False):
        self.program_learner.apply_constraints(print_function=print_function)
        self._reset_object_grasp_hold()

    def processControl(self, u):
        return self.controller_api_wrapper.process_control(u)

    def observe(self):
        return self.seg[-1]["id"].keys()

    def recall(self, observed=None):
        if observed is None:
            observed = ()
        known = (
            self.controller_api_wrapper.get_previously_seen_objects()
            - set(observed)
            - set([self.controller_api_wrapper.reference_agent])
        )

        return known

    def recall_entities(self):
        return self.controller_api_wrapper.get_previously_seen_entities()

    @staticmethod
    def _parse_program_atomic_actions(raw: str) -> list[str]:
        raw = raw.strip()
        if not raw:
            return []
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, str):
            return [parsed]
        if isinstance(parsed, (list, tuple)):
            return [str(action) for action in parsed]
        return [str(parsed)]

    @staticmethod
    def read_program_stage(stage_dir: str | Path) -> Dict[str, Any]:
        stage_dir = Path(stage_dir)
        with open(stage_dir / "task.txt", "r", encoding="utf-8") as task_file:
            task = task_file.read().strip()
        with open(stage_dir / "atomic.txt", "r", encoding="utf-8") as atomic_file:
            atomic_actions = SUNProgramExecutor._parse_program_atomic_actions(
                atomic_file.read()
            )
        with open(stage_dir / "constraints.json", "r", encoding="utf-8") as constraints_file:
            constraints = json.load(constraints_file)
        return {
            "task": task,
            "atomic_actions": atomic_actions,
            "constraints": constraints,
        }

    def snapshot_program_stage(self) -> Dict[str, Any]:
        learner = self.program_learner._constraint_generator
        tasks = learner._tasks
        return {
            "task": tasks.current_task,
            "atomic_actions": list(tasks.current_atomic_actions),
            "constraints": deepcopy(learner._constraints),
            "pending_tasks": list(tasks.pending_tasks),
            "completed_tasks": list(tasks.completed_tasks),
            "completed_task_atomic_actions": deepcopy(
                tasks.completed_task_atomic_actions
            ),
        }

    def set_program_stage(
        self,
        task: str,
        atomic_actions: list[str],
        constraints: Dict[str, Any],
        *,
        pending_tasks: Optional[list[str]] = None,
        completed_tasks: Optional[list[str]] = None,
        completed_task_atomic_actions: Optional[list[Dict[str, Any]]] = None,
    ) -> None:
        learner = self.program_learner._constraint_generator
        tasks = learner._tasks
        tasks._tasks = ([task] if task else []) + list(pending_tasks or [])
        tasks._completed_tasks = list(completed_tasks or [])
        tasks._completed_task_atomic_actions = deepcopy(
            completed_task_atomic_actions or []
        )
        if tasks._completed_task_atomic_actions:
            tasks._last_completed_atomic_actions = list(
                tasks._completed_task_atomic_actions[-1].get("atomic_actions", [])
            )
        else:
            tasks._last_completed_atomic_actions = []
        tasks._current_atomic_actions = list(atomic_actions or [])
        learner._constraints = deepcopy(
            constraints if constraints is not None else {"Constraints": []}
        )
        learner._active_constraints = {}
        learner._predicate_scope = {}
        learner._residual_scope = {}
        self.controller_api_wrapper.stall_joint_counter = None
        self.controller_api_wrapper.gripper_stall_counter = None
        self.controller_api_wrapper._gripper_stall_state = None
        self._reset_object_grasp_hold()

    def load_program_stage(
        self,
        stage_dir: str | Path,
        *,
        pending_tasks: Optional[list[str]] = None,
        completed_tasks: Optional[list[str]] = None,
        completed_task_atomic_actions: Optional[list[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        stage = self.read_program_stage(stage_dir)
        self.set_program_stage(
            stage["task"],
            stage["atomic_actions"],
            stage["constraints"],
            pending_tasks=pending_tasks,
            completed_tasks=completed_tasks,
            completed_task_atomic_actions=completed_task_atomic_actions,
        )
        return stage

    def load_program(self, load_dir_base):
        self.program_stage_dirs = subfolders_with_prefix(
            os.path.join(load_dir_base, "program"), "subtask_"
        )
        self.program_stage_dirs.sort()
        if len(self.program_stage_dirs) == 0:
            logger.warning(
                f"No program base found in {load_dir_base}, starting from scratch"
            )
            return

        self.program_learner.load_program(
            self.program_stage_dirs[0] / "residuals.py",
            self.program_stage_dirs[0] / "predicates.py",
        )
        self.program_learner.load_agent_constraint(
            self.program_stage_dirs[0] / "constraints.json",
        )
        self._reset_object_grasp_hold()
