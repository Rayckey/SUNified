from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import torch
import numpy as np
from copy import deepcopy
from omegaconf import OmegaConf, DictConfig
from kuafu.mpc.torch.compiled_sun_program import CompiledSUNProgram
from kuafu.mpc.torch.api_wrapper import APIWrapper
from kuafu.mpc.torch import torch_predictive_control
from kuafu.programs.llm.io_util import subfolders_with_prefix
from kuafu.programs.llm.constraint_describe_utils import explain_constraints
class SUNProgramEvaluator(object):
    """Evaluate SUN Program predicates and rewards across batched environments."""

    def __init__(self, cfg: DictConfig, auto_load_program: bool = True):

        self.cfg = cfg
        self.device = cfg.device
        self.picture_idx = 0
        # Construct the lightweight torch-side controller requested by config.
        # These controllers expose frame metadata for learned constraints; live
        # poses still come from Isaac frame sensors.
        ee_frame = getattr(cfg, "ee_frame", "panda_fingertip")
        cfg_num_envs = getattr(cfg, "num_envs", None)
        if cfg_num_envs is None:
            raise AttributeError(
                "SUNProgramEvaluator requires cfg.num_envs to construct the MPC controller. "
                "Please set cfg.num_envs (or include it in the OmegaConf dict)."
            )
        controller_name = getattr(cfg, "controller_name", "FrankaIKController")
        controller_cls = getattr(
            torch_predictive_control,
            controller_name,
            torch_predictive_control.FrankaIKController,
        )
        mpc_controller = controller_cls(
            ee_frame=ee_frame,
            device=self.device,
            num_envs=cfg_num_envs,
        )

        self.mpc_controller = mpc_controller

        # Read gripper joint limits from config (tensor-valued).
        # Defaults to Panda-style [0.04, 0.04] / [0.0, 0.0] when not specified.
        gripper_opened = getattr(cfg, "gripper_opened", None)
        gripper_closed = getattr(cfg, "gripper_closed", None)
        gripper_max_force = getattr(cfg, "gripper_max_force", 30.0)
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
            mpc_controller, self.device,
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
        agent_id = "agent"
        self.controller_api_wrapper.setPOVAgent(agent_id)

        self.joint_state = None
        self.joint_target = None
        self.image = []
        self.seg = []
        # Low-pass filter state for joint_effort (exponential moving average).
        # alpha=1.0 means no filtering; lower values smooth more aggressively.
        self._effort_lpf_alpha: float = getattr(cfg, "effort_lpf_alpha", 0.2)
        self._filtered_joint_effort: Dict[str, torch.Tensor] = {}
        self._current_stage_index = -1
        self.batch_size = None
        self._needs_pov_update = None  # Flag to track which envs need POV state snapshot
        self._needs_pov_update_dirty = False
        self._object_grasp_hold_steps: int = int(
            getattr(cfg, "object_grasp_hold_steps", 5)
        )
        self._object_grasp_hold_counter: Optional[torch.Tensor] = None

        self.compiled_program = CompiledSUNProgram(self.controller_api_wrapper, self)

        # Load predicate utilities/functions into the shared registry.
        predicate_path = cfg.initial_predicate
        residual_path = cfg.initial_residual

        self.compiled_program.load_initial_predicate(predicate_path)
        self.compiled_program.load_initial_residual(residual_path)

        self.residual_values_buffer = None
        # Residual weights can be stored in two modes (selected by config):
        # - Global weights (default):     {stage_id: {constraint_id: float}}
        # - Per-env weights (optional):   {stage_id: {constraint_id: Tensor[num_envs]}}
        #
        # Configure with:
        #   cfg.use_per_env_weight: bool
        #       - False (default): residual weights are scalars shared across all envs.
        #       - True: residual weights are tensors of shape (cfg.num_envs,) per constraint.
        use_per_env_weight = getattr(cfg, "use_per_env_weight")
        self._residual_weights_per_env: bool = use_per_env_weight
        self._residual_weights_num_env: Optional[int] = (
            int(cfg_num_envs) if self._residual_weights_per_env else None
        )
        # Buffer to store residual weights per stage: {stage_id: {constraint_id: weight_value}}
        self._residual_weights_buffer: Dict[int, Dict[int, Any]] = {}
        self._print_scope_functions_on_compile: bool = getattr(
            cfg, "print_scope_functions_on_compile", False
        )

    def initialize_batch(self, batch_size: int):
        """
        Initialize tracking for a batch of parallel environments.
        """
        if self.compiled_program.stage_count() <= 0:
            raise RuntimeError(
                "SUNProgramEvaluator has no SUN Program stages loaded. "
                "Call load_program(...) before initialize_batch(...)."
            )
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.batch_size = batch_size
        self.compiled_program.initialize_batch_progress(batch_size)
        # Entering stage zero requires an initial POV snapshot for each environment.
        self._needs_pov_update = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        self._needs_pov_update_dirty = True
        self._object_grasp_hold_counter = torch.zeros(
            batch_size, dtype=torch.int64, device=self.device
        )

    def _ensure_object_grasp_hold_counter(self) -> torch.Tensor:
        if self.batch_size is None:
            raise RuntimeError("Batch tracking not initialized; call initialize_batch first.")
        expected = int(self.batch_size)
        if (
            self._object_grasp_hold_counter is None
            or self._object_grasp_hold_counter.shape[0] != expected
        ):
            self._object_grasp_hold_counter = torch.zeros(
                expected, dtype=torch.int64, device=self.device
            )
        return self._object_grasp_hold_counter

    def _reset_object_grasp_hold(
        self, env_mask: Optional[torch.Tensor] = None
    ) -> None:
        counter = self._object_grasp_hold_counter
        if counter is None:
            return
        if env_mask is None:
            counter.zero_()
            return
        if env_mask.dtype != torch.bool:
            full_mask = torch.zeros(
                counter.shape[0], dtype=torch.bool, device=counter.device
            )
            full_mask[env_mask] = True
            env_mask = full_mask
        if env_mask.shape[0] != counter.shape[0]:
            raise ValueError("env_mask must have the same length as the batch size.")
        counter[env_mask] = 0

    def _stage_uses_object_grasped(self, stage_id: int) -> bool:
        stage = self.compiled_program._program_stages[stage_id]
        constraints = stage.constraints.get("Constraints", [])
        return any(term.get("op") == "object_grasped" for term in constraints)

    def _apply_object_grasp_hold(
        self, stage_pred: torch.Tensor, env_mask: torch.Tensor, *, use_hold: bool
    ) -> torch.Tensor:
        flat_pred = stage_pred.view(-1).to(dtype=torch.bool, device=self.device)
        counter = self._ensure_object_grasp_hold_counter()
        if not use_hold:
            counter[env_mask] = 0
            return flat_pred.reshape(stage_pred.shape)

        held_counts = torch.where(
            flat_pred,
            counter[env_mask] + 1,
            torch.zeros_like(counter[env_mask]),
        )
        counter[env_mask] = held_counts
        return (held_counts >= self._object_grasp_hold_steps).reshape(stage_pred.shape)

    def _normalize_residual_weight_value(self, weight: Any) -> Any:
        """Normalize a weight value into the internal representation for this SUNProgramEvaluator instance."""
        if self._residual_weights_per_env:
            n_env = self._residual_weights_num_env
            assert n_env is not None
            # Accept float/int -> broadcast to all envs; tensor -> validate shape.
            if isinstance(weight, (float, int)):
                return torch.full(
                    (int(n_env),),
                    float(weight),
                    device=self.device,
                    dtype=torch.float32,
                )
            if torch.is_tensor(weight):
                w = weight.to(device=self.device, dtype=torch.float32).view(-1)
                if w.numel() != int(n_env):
                    raise ValueError(
                        f"Per-env residual weight tensor must have shape ({int(n_env)},) "
                        f"but got numel={int(w.numel())}."
                    )
                return w
            raise TypeError(
                f"Unsupported weight type {type(weight).__name__} for per-env residual weights. "
                "Use float/int or a torch.Tensor of shape (cfg.num_envs,)."
            )
        # Global mode: store scalar float.
        if isinstance(weight, (float, int)):
            return float(weight)
        if torch.is_tensor(weight):
            w = weight.to(dtype=torch.float32)
            if w.numel() != 1:
                raise ValueError(
                    "Global residual weights require a scalar. "
                    f"Got tensor with numel={int(w.numel())}."
                )
            return float(w.item())
        raise TypeError(
            f"Unsupported weight type {type(weight).__name__} for global residual weights. "
            "Use float/int or a scalar torch.Tensor."
        )

    def callbackImage(self, image):
        self.image.append(image)
        if len(self.image) > 10:
            self.image.pop(0)

    def callbackOdom(self, msg: dict, msg_s: dict):

        self.controller_api_wrapper.updatemOdom(msg, msg_s)

    def hasKinematics(self):
        """
        Check if the controller has kinematics capabilities.
        """
        return hasattr(self.mpc_controller, "forward_kinematics")

    def callbackJointState(self, msg: dict):

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

        if self.mpc_controller.has_gripper:
            self.controller_api_wrapper.joint_dict[key]["gripper_state"] = (
                self.controller_api_wrapper.joint_dict[
                    self.controller_api_wrapper.reference_agent
                ]["joint_position"][:, -2:]
            )

    def callbackContactForces(self, contact_forces: dict) -> None:
        """Feed finger-tip contact force data into the api_wrapper.

        Args:
            contact_forces: dict mapping sensor name -> (N, 3) tensor of net forces in world frame.
                            Expected keys: "contact_sensor_left", "contact_sensor_right".
        """
        self.controller_api_wrapper.update_contact_forces(contact_forces)

    def callbackTwist(self, msg: dict):
        for key in msg:
            self.controller_api_wrapper.twist_dict[key] = msg[key]

    def callbackSegmentation(self, id_mask, id_lookup):
        self.seg.append({"image": id_mask, "id": id_lookup})
        if len(self.seg) > 10:
            self.seg.pop(0)

    def load_program(self, load_dir_base):
        program_root = Path(load_dir_base) / "program"
        self.program_stage_dirs = subfolders_with_prefix(program_root, "subtask_")
        self.program_stage_dirs.sort(key=lambda p: int(p.name.removeprefix("subtask_")))
        if len(self.program_stage_dirs) == 0:
            print(f"No SUN Program stages found in {program_root}")
            return

        self.compiled_program.load_program_stages(
            [str(path) for path in self.program_stage_dirs],
            print_functions=self._print_scope_functions_on_compile,
        )
        self._initialize_residual_weights_for_all_stages()
        self._current_stage_index = -1
        self._reset_object_grasp_hold()

    def _initialize_residual_weights_for_stage(self, stage_id: int) -> None:
        """
        Initialize residual weights to 1.0 for all constraints in a stage.
        
        Args:
            stage_id: The stage index (0-based)
        """
        if stage_id < 0 or stage_id >= self.compiled_program.stage_count():
            return

        stage = self.compiled_program._program_stages[stage_id]
        constraints = stage.constraints.get("Constraints", [])
        num_constraints = len(constraints)

        if num_constraints == 0:
            return

        if stage_id not in self._residual_weights_buffer:
            self._residual_weights_buffer[stage_id] = {}

        for constraint_idx in range(num_constraints):
            if self._residual_weights_per_env:
                n_env = self._residual_weights_num_env
                assert n_env is not None
                self._residual_weights_buffer[stage_id][constraint_idx] = torch.ones(
                    int(n_env),
                    device=self.device,
                    dtype=torch.float32,
                )
            else:
                self._residual_weights_buffer[stage_id][constraint_idx] = 1.0

    def _stage_weights_for_mask(
        self, stage_id: int, mask: Optional[torch.Tensor]
    ) -> Optional[Dict[int, Any]]:
        """Return residual weights for a stage, sliced to the active env mask or ids."""
        weights = self._residual_weights_buffer.get(stage_id)
        if weights is None:
            return None
        if mask is None:
            return weights
        stage_weights: Dict[int, Any] = {}
        for idx, w in weights.items():
            if isinstance(w, torch.Tensor):
                stage_weights[idx] = w[mask]
            else:
                stage_weights[idx] = w
        return stage_weights

    def _initialize_residual_weights_for_all_stages(self) -> None:
        """
        Initialize residual weights to 1.0 for all constraints in all loaded stages.
        """
        num_stages = self.compiled_program.stage_count()
        for stage_id in range(num_stages):
            self._initialize_residual_weights_for_stage(stage_id)

    def reload_constraints(
        self,
        constraints_list: Dict[int, List[Dict[str, Any]]],
        stage_indices: List[int],
        stage_names: Optional[List[str]] = None,
    ) -> None:
        """
        Reload constraints by replacing specific parameters within existing stages (for live adjustment).
        
        STRICT RULES (exceptions raised if violated):
        - Cannot change the number of stages
        - Cannot change the number of constraints per stage
        - Cannot change constraint operators ("op")
        - Cannot change constraint logic ("logic")
        - Only constraint parameters ("args") can be updated
        - Each stage MUST have its own separate constraint dictionary (stages cannot share constraints)
        
        Args:
            constraints_list: List of constraint dictionaries (same format as constraints.json).
                             Each dictionary corresponds to ONE specific stage. The i-th dictionary
                             in this list will update the stage specified by stage_indices[i].
                             IMPORTANT: Each stage must have its own constraint dictionary - stages
                             cannot share the same constraint list even if they have similar constraints.
            stage_indices: REQUIRED list of stage indices to update. Must match length of constraints_list.
                          Each element specifies which stage the corresponding constraint dictionary updates.
                          Can specify any subset of stages (e.g., [3, 4] to update only stages 3 and 4).
            stage_names: Optional list of stage names. If provided, must match length of constraints_list.
        
        Raises:
            ValueError: If stage_indices is not provided, lengths don't match, or structure changes
            IndexError: If any stage_index is out of bounds
        
        Examples:

            >>> # Update multiple stages (e.g., stages 3 and 4, leaving others unchanged)
            >>> # Each stage MUST have its own separate constraint dictionary
            >>> constraints_stage3 = {"Constraints": [...]}  # Stage 3's constraints
            >>> constraints_stage4 = {"Constraints": [...]}  # Stage 4's constraints (separate from stage 3)
            >>> sun_program_evaluator.reload_constraints(
            ...     constraints_list=[constraints_stage3, constraints_stage4],
            ...     stage_indices=[3, 4]
            ... )
        """

        if len(stage_indices) != len(constraints_list):
            raise ValueError(
                f"stage_indices must have the same length as constraints_list. "
                f"Got {len(stage_indices)} stage_indices for {len(constraints_list)} constraint dictionaries."
            )

        if stage_names is not None and len(stage_names) != len(constraints_list):
            raise ValueError(
                f"stage_names must have the same length as constraints_list if provided. "
                f"Got {len(stage_names)} stage_names for {len(constraints_list)} constraint dictionaries."
            )

        current_stage_count = self.compiled_program.stage_count()
        unique_stage_indices = set(stage_indices)
        if len(unique_stage_indices) != len(stage_indices):
            raise ValueError(
                "Duplicate stage indices found. Each stage can only be updated once per call."
            )

        # Process each stage: constraints_list[i] updates stage_indices[i]
        # Each stage must have its own separate constraint dictionary
        for idx, constraints in enumerate(constraints_list):
            stage_idx = stage_indices[idx]
            stage_name = stage_names[idx] if stage_names is not None else None
            # Each constraint dictionary is specific to one stage - they cannot be shared
            self.compiled_program.load_stage_from_constraints(
                constraints=constraints,
                print_functions=self._print_scope_functions_on_compile,
                stage_index=stage_idx,
                stage_name=stage_name,
            )
            self._initialize_residual_weights_for_stage(stage_idx)

        self._reset_object_grasp_hold()

    def quantifyPredicate(self, text: bool = False, max_envs: int | None = None):
        """
        Evaluate predicates for all environments based on their current stage.
        Returns a boolean tensor indicating which environments satisfied their predicates.

        If *text* is True, also returns a human-readable string showing per-term
        predicate results (similar to SUNProgramExecutor.quantifyPredicate).
        """
        if self.batch_size is None:
            raise RuntimeError("Batch tracking not initialized; call initialize_batch first.")
        batch_size = int(self.batch_size)

        predicate_values = torch.zeros((batch_size,), dtype=torch.bool, device=self.device)
        # Enable per-evaluation API memoization so repeated calls across terms are shared.
        with self.controller_api_wrapper.eval_cache():
            for stage_id, env_ids, stage in self.compiled_program.batch_stage_assignments_with_indices():
                self.controller_api_wrapper.set_batch_indices(env_ids, full_batch_size=batch_size)
                try:
                    # Ensure scope is compiled/cached (predicate_function uses updated generator).
                    stage_predicates = self.compiled_program.evaluate_stage_predicates(stage_id)

                    # Normalise to a 1D boolean vector for assignment into predicate_values.
                    if isinstance(stage_predicates, torch.Tensor):
                        stage_predicates = stage_predicates.view(-1).to(
                            dtype=torch.bool, device=self.device
                        )

                    stage_batch_size = int(env_ids.numel())
                    if stage_predicates.numel() == batch_size:
                        predicate_values[env_ids] = torch.index_select(stage_predicates, 0, env_ids)
                    elif stage_predicates.numel() == stage_batch_size:
                        predicate_values[env_ids] = stage_predicates
                    else:
                        raise ValueError(
                            f"Predicate for stage {stage_id} returned shape {tuple(stage_predicates.shape)} "
                            f"which cannot be assigned to stage batch size {stage_batch_size}."
                        )

                    # Collect per-term predicate text for debugging.
                    if text:
                        text_lines: list[str] = []
                        n_envs_in_stage = stage_batch_size
                        text_lines.append(f"  Stage {stage_id} ({n_envs_in_stage} envs):")
                        pred_scope = stage.predicate_scope
                        if pred_scope is None:
                            text_lines.append("    <predicate scope not initialized>")
                            continue
                        term_names = pred_scope.get("predicate_term_functions", [])
                        if not isinstance(term_names, list) or len(term_names) == 0:
                            text_lines.append("    <no per-term predicate functions>")
                            continue
                        for idx, fn_name in enumerate(term_names):
                            fn = pred_scope.get(fn_name)
                            if fn is None or not callable(fn):
                                text_lines.append(f"    term {idx}: {fn_name} -> <missing>")
                                continue
                            try:
                                term_res = fn()
                            except Exception as e:
                                text_lines.append(f"    term {idx}: {fn_name} -> ERROR: {e}")
                                continue
                            if isinstance(term_res, torch.Tensor) and max_envs is not None:
                                flat = term_res.view(-1)
                                shown = flat[:max(0, int(max_envs))]
                                text_lines.append(f"    term {idx}: {fn_name} -> {shown}")
                            else:
                                text_lines.append(f"    term {idx}: {fn_name} -> {term_res}")
                finally:
                    # Always clear the active batch selection after computation.
                    self.controller_api_wrapper.clear_batch_mask()

        if text:
            return predicate_values, "\n".join(text_lines)
        return predicate_values

    def get_stage_indices(self):
        indices = self.compiled_program.get_stage_indices()
        if indices is None:
            raise RuntimeError("Batch tracking not initialized; call initialize_batch first.")
        return indices

    def get_per_constraint_satisfaction_and_rewards(
        self, use_unit_weights: bool = True
    ) -> Tuple[
        Dict[int, Dict[int, Tuple[torch.Tensor, torch.Tensor]]],
        Dict[int, Dict[int, Tuple[torch.Tensor, torch.Tensor]]],
    ]:
        """
        Evaluate per-constraint satisfaction and per-constraint rewards in one pass.

        Predicates and residuals are computed together per stage (one set_batch_mask
        per stage) to reuse the buffer and controller state, same pattern as
        check_and_advance_stages_and_compute_rewards.

        Returns:
            satisfaction: Dict[stage_id, Dict[constraint_id, (mask, satisfied)]]
                - mask: (batch_size,) bool, True for envs in this stage
                - satisfied: (mask.sum(),) bool, True where that constraint is met
            rewards: Dict[stage_id, Dict[constraint_id, (mask, reward_tensor)]]
                - mask: same as above
                - reward_tensor: (mask.sum(),) float, that constraint's residual (weighted).
                  Uses residual_function(..., return_per_term=True) when supported;
                  otherwise fallback: stage residual / num_constraints.
        """
        if self.batch_size is None:
            raise RuntimeError(
                "Batch tracking not initialized; call initialize_batch first."
            )
        batch_size = int(self.batch_size)
        satisfaction: Dict[int, Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = {}
        rewards: Dict[int, Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = {}

        self.controller_api_wrapper.clear_batch_mask()
        x_full = self.controller_api_wrapper.getMinimumState()

        with self.controller_api_wrapper.eval_cache():
            for stage_id, env_ids, stage in self.compiled_program.batch_stage_assignments_with_indices():
                self.controller_api_wrapper.set_batch_indices(env_ids, full_batch_size=batch_size)
                try:
                    p_scope = stage.predicate_scope
                    r_scope = stage.residual_scope
                    p_fn = p_scope["predicate_function"]
                    r_fn = r_scope["residual_function"]
                    stage_batch_size = int(env_ids.numel())

                    # Predicate returns tuple of per-term results; residual returns tuple of per-term costs.
                    term_satisfied = p_fn(return_per_term=True)
                    weights = self._stage_weights_for_mask(stage_id, env_ids)
                    x_stage = torch.index_select(x_full, 0, env_ids)
                    term_rewards = r_fn(x_stage, residual_weights=weights, return_per_term=True)
                    if len(term_satisfied) != len(term_rewards):
                        raise ValueError(
                            f"Stage {stage_id}: predicate terms ({len(term_satisfied)}) "
                            f"!= residual terms ({len(term_rewards)})"
                        )

                    mask_clone = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
                    mask_clone[env_ids] = True
                    satisfaction[stage_id] = {}
                    for cid, t in enumerate(term_satisfied):
                        t_b = t.view(-1).to(dtype=torch.bool, device=self.device)
                        if t_b.numel() == batch_size:
                            satisfied = torch.index_select(t_b, 0, env_ids).clone()
                        elif t_b.numel() == stage_batch_size:
                            satisfied = t_b.clone()
                        else:
                            raise ValueError(
                                f"Stage {stage_id} constraint {cid}: predicate term has invalid shape "
                                f"{tuple(t.shape)} (flattened={t_b.numel()}). Expected either full batch "
                                f"size {batch_size} or stage batch size {stage_batch_size}."
                            )
                        satisfaction[stage_id][cid] = (mask_clone, satisfied)
                    rewards[stage_id] = {}
                    for cid, t in enumerate(term_rewards):
                        reward_t = t.view(-1).to(device=self.device)
                        if reward_t.numel() != stage_batch_size:
                            raise ValueError(
                                f"Stage {stage_id} constraint {cid}: reward term has invalid shape "
                                f"{tuple(t.shape)} (flattened={reward_t.numel()}). Expected stage batch size "
                                f"{stage_batch_size}. This usually means the residual return_per_term path "
                                "reduced across the batch and returned a scalar or wrong-sized tensor."
                            )
                        rewards[stage_id][cid] = (mask_clone, reward_t.clone())
                finally:
                    self.controller_api_wrapper.clear_batch_mask()
        return satisfaction, rewards

    def get_per_constraint_satisfaction(
        self,
    ) -> Dict[int, Dict[int, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Evaluate per-constraint satisfaction for all stages and envs.

        For each stage, evaluates each predicate term (constraint) with the batch mask
        set to envs in that stage. Used by the env to aggregate constraint met rate stats.
        Delegates to get_per_constraint_satisfaction_and_rewards and returns only
        the satisfaction part (predicates and residuals are still computed together).

        Returns:
            Dict[stage_id, Dict[constraint_id, (mask, satisfied)] where:
            - mask: (batch_size,) bool, True for envs in this stage
            - satisfied: (mask.sum(),) bool, True where that constraint is met for those envs
        """
        satisfaction, _ = self.get_per_constraint_satisfaction_and_rewards()
        return satisfaction

    def update_pov_if_needed(self) -> None:
        """
        Update POV state for environments that need it (e.g., just entered or reset to stage 0).
        This is a lightweight check that only performs expensive updates when the flag is set.
        Safe to call every step - returns immediately if no updates needed.
        """
        if self._needs_pov_update is None or self.batch_size is None:
            raise RuntimeError(
                "Batch tracking not initialized; call initialize_batch first."
            )

        # Avoid a per-step CUDA sync from torch.any(...).item(). The CPU-side
        # dirty flag is set only by code paths that write True into _needs_pov_update.
        if not self._needs_pov_update_dirty:
            return

        self.controller_api_wrapper.updatePOVState(self._needs_pov_update)

        self._needs_pov_update[:] = False
        self._needs_pov_update_dirty = False


    def terminateCurrentSubtask(self, success_mask: torch.Tensor):
        """
        Terminate the current subtask, this is used to indicate the end of a subtask.
        """
        if success_mask.shape[0] != self.batch_size:
            raise ValueError("success_mask must have the same length as the batch size.")
        if self._needs_pov_update is None:
            raise RuntimeError("Batch tracking not initialized; call initialize_batch first.")

        # Backup transformations for environments transitioning to the next stage.
        # This must happen RIGHT at the moment of stage transition.
        self._needs_pov_update[success_mask] = True
        self._needs_pov_update_dirty = True
        self._reset_object_grasp_hold(success_mask)

        self.compiled_program.advance_batch_stage(success_mask)

        return

    def check_and_advance_stages_and_compute_rewards(
        self,
        compute_rewards: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute (predicates, past_final_stage, residual) in one stage-loop.

        past_final_stage is True for envs that are in the final stage and already
        satisfy the predicate (i.e. would move past the final stage).
        When compute_rewards is False, only predicate/stage evaluation runs and
        the returned residual tensor is all zeros.
        This avoids doing two separate passes (one for predicates in `_get_dones`
        and one for residuals in `_get_rewards`) which otherwise duplicates:
        - stage grouping / masking
        - API calls inside predicate/residual primitives
        - controller state extraction
        """
        if self.batch_size is None:
            raise RuntimeError("Batch tracking not initialized; call initialize_batch first.")
        batch_size = int(self.batch_size)

        # Compute controller state once (full batch) only when residual rewards
        # are needed. Predicate evaluation does not consume x_full.
        self.controller_api_wrapper.clear_batch_mask()
        x_full = self.controller_api_wrapper.getMinimumState() if compute_rewards else None

        satisfied = torch.zeros((batch_size,), dtype=torch.bool, device=self.device)
        residual_values = torch.zeros((batch_size,), device=self.device)


        with self.controller_api_wrapper.eval_cache():
            try:
                for stage_id, env_ids, stage in self.compiled_program.batch_stage_assignments_with_indices():
                    self.controller_api_wrapper.set_batch_indices(env_ids, full_batch_size=batch_size)

                    p_scope = stage.predicate_scope
                    p_fn = p_scope["predicate_function"]

                    stage_pred = p_fn()
                    held_stage_pred = self._apply_object_grasp_hold(
                        stage_pred,
                        env_ids,
                        use_hold=self._stage_uses_object_grasped(stage_id),
                    )
                    satisfied[env_ids] = held_stage_pred.view(-1)

                    if compute_rewards:
                        r_scope = stage.residual_scope
                        r_fn = r_scope["residual_function"]
                        # Get weights for this stage (None if not set, which defaults to all 1.0).
                        weights = self._stage_weights_for_mask(stage_id, env_ids)
                        x_stage = torch.index_select(x_full, 0, env_ids)
                        stage_reward = r_fn(x_stage, residual_weights=weights)

                        # Normalize by the sum of weights.
                        total_weight = None
                        for w in weights.values():
                            if total_weight is None:
                                total_weight = w
                            else:
                                total_weight = total_weight + w
                        denom = total_weight

                        residual_values[env_ids] = stage_reward / denom
            finally:
                self.controller_api_wrapper.clear_batch_mask()

        stage_indices = self.compiled_program.get_stage_indices()
        if stage_indices is None:
            raise RuntimeError("Batch tracking not initialized; call initialize_batch first.")
        n_stages = self.compiled_program.stage_count()
        if n_stages <= 0:
            raise RuntimeError(
                "SUNProgramEvaluator has no SUN Program stages loaded. "
                "Call load_program(...) before using stage logic."
            )
        final_stage = n_stages - 1
        in_final_stage = stage_indices >= final_stage
        # Past final = in final stage and satisfied *that* stage's predicate.
        # Uses pre-advance stage_indices, so envs that advance into the final stage
        # this step are not included (they had in_final_stage=False here).
        past_final_stage = in_final_stage & satisfied

        if torch.any(satisfied):
            self.terminateCurrentSubtask(satisfied)

        return satisfied, past_final_stage, residual_values

    def reset_stage_indices(self, env_ids: torch.Tensor | None = None) -> None:
        """
        Reset stage indices to 0 for the specified environments.
        
        Args:
            env_ids: Either a boolean mask of shape (batch_size,) or a tensor of environment IDs.
                     If None, resets all environments.
        """

        if env_ids is None:
            raise ValueError("env_ids must be provided to reset specific environments.")
        if self.batch_size is None or self._needs_pov_update is None:
            raise RuntimeError("Batch tracking not initialized; call initialize_batch first.")

        if env_ids.dtype == torch.bool:
            env_mask = env_ids
            if not torch.any(env_mask):
                return
        else:
            if len(env_ids) == 0:
                return
            env_mask = torch.zeros(int(self.batch_size), dtype=torch.bool, device=self.device)
            env_mask[env_ids] = True

        if self.compiled_program.stage_indices is None:
            raise RuntimeError("Batch tracking not initialized; call initialize_batch first.")
        assert self.compiled_program._batch_stage_indices is not None
        self.compiled_program._batch_stage_indices[env_mask] = 0


        # Mark reset environments as needing POV update when re-entering stage 0.
        self._needs_pov_update[env_mask] = True
        self._needs_pov_update_dirty = True
        self._reset_object_grasp_hold(env_mask)

    def compute_residual(self):
        """
        Compute the cost function (residual) for the given state and control.
        """

        if self.batch_size is None:
            raise RuntimeError(
                "Batch tracking not initialized; call initialize_batch first."
            )
        batch_size = int(self.batch_size)

        # Batched multi-stage case: compute costs per stage with masked batches.
        #
        # Avoid recomputing the minimum controller state for each stage.
        # We compute the full batch state once, then slice x by stage mask. We still
        # set the batch mask before calling residual_fn so any API calls inside the
        # residuals return stage-masked tensors.
        residual_values = torch.zeros((batch_size,), device=self.device)

        # Ensure we compute the full batch state (no mask applied).
        self.controller_api_wrapper.clear_batch_mask()
        x_full = self.controller_api_wrapper.getMinimumState()
        for stage_id, env_ids, stage in self.compiled_program.batch_stage_assignments_with_indices():
            # Set active env ids so all API calls inside residual_fn align with x_full[env_ids].
            self.controller_api_wrapper.set_batch_indices(env_ids, full_batch_size=batch_size)
            try:
                scope = self.compiled_program.get_residual_scope_for_stage(stage_id)
                residual_fn = scope["residual_function"]
                # Get weights for this stage (None if not set, which defaults to all 1.0).
                weights = self._stage_weights_for_mask(stage_id, env_ids)
                x_stage = torch.index_select(x_full, 0, env_ids)
                residual_values[env_ids] = residual_fn(x_stage, residual_weights=weights)
            except Exception as e:
                stage_name = (
                    self.compiled_program._program_stages[stage_id].name
                    if stage_id < len(self.compiled_program._program_stages)
                    else f"stage_{stage_id}"
                )
                print(f"Error computing cost for stage {stage_id} ({stage_name}):")
                print(
                    f"  Stage batch size: {int(env_ids.numel())}/{batch_size} environments"
                )
                print(f"  State shape: {tuple(x_full.shape)}")
                print(f"  Error: {type(e).__name__}: {e}")
                raise
            finally:
                self.controller_api_wrapper.clear_batch_mask()


        self.residual_values_buffer = residual_values.detach().clone()
        return residual_values

    def set_residual_weights(
        self, stage_id: int, weights: Dict[int, Any]
    ) -> None:
        """
        Set residual weights for constraints in a stage.
        
        Each constraint in the stage can have its own weight. If a stage has N constraints,
        you can set weights for any subset of them (indices 0 to N-1).
        
        Args:
            stage_id: The stage index (0-based)
            weights: Dictionary mapping constraint index -> weight value.
                     - If cfg.use_per_env_weight is True, each weight can be:
                       - float/int: broadcast to all environments
                       - torch.Tensor: per-env weights of shape (cfg.num_envs,)
                     - If cfg.use_per_env_weight is False, each weight is a scalar
                       shared across all environments (float/int or scalar tensor).
                     Constraint indices correspond to the order in the Constraints list.
                     Example: {0: 2.0, 1: 1.5, 2: 0.5} for a stage with at least 3 constraints
        
        Example:
            >>> # For a stage with 9 constraints, set weights for all of them:
            >>> program_runtime.set_residual_weights(0, {0: 2.0, 1: 1.5, 2: 1.0, 3: 0.5, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0})
            >>> # Or set weights for specific constraints only:
            >>> program_runtime.set_residual_weights(0, {0: 2.0, 1: 1.5})
        """
        if stage_id < 0 or stage_id >= self.compiled_program.stage_count():
            raise IndexError(
                f"Stage index {stage_id} is out of bounds (0-{self.compiled_program.stage_count() - 1})."
            )

        stage = self.compiled_program._program_stages[stage_id]
        constraints = stage.constraints.get("Constraints", [])
        num_constraints = len(constraints)

        for idx in weights.keys():
            if idx < 0 or idx >= num_constraints:
                raise IndexError(
                    f"Constraint index {idx} is out of bounds for stage {stage_id} "
                    f"(0-{num_constraints - 1})."
                )

        # Ensure stage exists in buffer (e.g. if called before load_program)
        if stage_id not in self._residual_weights_buffer:
            self._initialize_residual_weights_for_stage(stage_id)
            if stage_id not in self._residual_weights_buffer:
                self._residual_weights_buffer[stage_id] = {}

        normalized: Dict[int, Any] = {}
        for k, v in weights.items():
            normalized[int(k)] = self._normalize_residual_weight_value(v)
        self._residual_weights_buffer[stage_id].update(normalized)

    def get_residual_weights(self, stage_id: int) -> Dict[int, Any]:
        """
        Get current residual weights for a stage.
        
        Args:
            stage_id: The stage index (0-based)
        
        Returns:
            Dictionary mapping constraint index -> weight value.
            - If cfg.use_per_env_weight is True: values are torch.Tensor of shape (cfg.num_envs,)
            - If cfg.use_per_env_weight is False: values are floats
        """
        if stage_id < 0 or stage_id >= self.compiled_program.stage_count():
            raise IndexError(
                f"Stage index {stage_id} is out of bounds (0-{self.compiled_program.stage_count() - 1})."
            )

        return dict(self._residual_weights_buffer.get(stage_id, {}))

    def set_residual_weight(
        self, stage_id: int, constraint_index: int, weight: Any
    ) -> None:
        """
        Set residual weight for a single constraint in a stage.
        
        Args:
            stage_id: The stage index (0-based)
            constraint_index: The constraint index (0-based, corresponds to position in Constraints list)
            weight: The weight value to set
        
        Example:
            >>> program_runtime.set_residual_weight(0, 0, 2.0)
            # Sets weight 2.0 for constraint 0 in stage 0
        """
        self.set_residual_weights(stage_id, {constraint_index: weight})

    def list_constraints(self, stage_id: int) -> List[Dict[str, Any]]:
        """
        List all constraints in a stage with their indices and details.
        
        This is a helper method to identify which constraint index corresponds to which
        constraint when setting weights.
        
        Args:
            stage_id: The stage index (0-based)
        
        Returns:
            List of dictionaries, each containing:
            - "index": constraint index (0-based)
            - "op": the operator name (e.g., "reach_distance", "freeze_orientation")
            - "logic": the logic operator ("and" or "or")
            - "condition": list of condition tags
            - "args": the constraint arguments
        
        Example:
            >>> constraints = program_runtime.list_constraints(0)
            >>> for c in constraints:
            ...     print(f"Index {c['index']}: {c['op']} (logic: {c['logic']})")
            Index 0: reach_distance (logic: or)
            Index 1: freeze_orientation (logic: and)
            Index 2: open_gripper (logic: and)
        """
        if stage_id < 0 or stage_id >= self.compiled_program.stage_count():
            raise IndexError(
                f"Stage index {stage_id} is out of bounds (0-{self.compiled_program.stage_count() - 1})."
            )

        stage = self.compiled_program._program_stages[stage_id]
        constraints = stage.constraints.get("Constraints", [])

        result = []
        for idx, constraint in enumerate(constraints):
            result.append({
                "index": idx,
                "op": constraint.get("op", "unknown"),
                "logic": constraint.get("logic", "unknown"),
                "condition": constraint.get("condition", []),
                "args": constraint.get("args", {})
            })

        return result

    def get_constraints_by_stage(self, stage_id: int) -> Dict[str, Any]:
        """
        Get a specific constraint by its ID (index).
        
        Args:
            stage_id: The stage index (0-based)
            constraint_id: The constraint index (0-based)
        
        Returns:
            Dictionary containing the constraint details:
            - "op": the operator name
            - "logic": the logic operator ("and" or "or")
            - "condition": list of condition tags
            - "args": the constraint arguments
        
        Raises:
            IndexError: If stage_id or constraint_id is out of bounds
        """
        if stage_id < 0 or stage_id >= self.compiled_program.stage_count():
            raise IndexError(
                f"Stage index {stage_id} is out of bounds (0-{self.compiled_program.stage_count() - 1})."
            )

        stage = self.compiled_program._program_stages[stage_id]
        constraints = stage.constraints
        return constraints


    def get_atomic_action_by_id(self, stage_id: int) -> Dict[str, Any]:
        if stage_id < 0 or stage_id >= self.compiled_program.stage_count():
            raise IndexError(
                f"Stage index {stage_id} is out of bounds (0-{self.compiled_program.stage_count() - 1})."
            )

        stage = self.compiled_program._program_stages[stage_id]
        atomic_action = stage.atomic_action
        return atomic_action


    def explain_constraints(self, stage_id: int) -> List[str]:
        """
        Return brief, concise explanations of all constraints in a stage.
        
        Uses templates based on the constraint prompt to generate human-readable
        descriptions of what each constraint does.
        
        The constraint IDs in the output match exactly with get_constraint_id_mapping(),
        ensuring consistent indexing across both methods.
        
        Args:
            stage_id: The stage index (0-based)
        
        Returns:
            List of strings, where each string describes one constraint.
            The index of each string matches the constraint ID (same as get_constraint_id_mapping).
        
        Example:
            >>> descriptions = program_runtime.explain_constraints(0)
            >>> print(descriptions[0])
            "Constraint 0 [AND] [geometry]: [robot frame panda_fingertip] Reach target point..."
            >>> print(descriptions[1])
            "Constraint 1 [AND] [geometry]: [robot frame panda_fingertip] Align +Z-axis to be..."
        """
        if stage_id < 0 or stage_id >= self.compiled_program.stage_count():
            raise IndexError(
                f"Stage index {stage_id} is out of bounds (0-{self.compiled_program.stage_count() - 1})."
            )

        stage = self.compiled_program._program_stages[stage_id]

        # Extract constraints list in the same way as get_constraint_id_mapping()
        # This ensures the constraint IDs match exactly
        constraints = stage.constraints.get("Constraints", [])

        if not constraints:
            return []

        # Call the utility function with the constraints list
        # This uses enumerate() with the same list order as get_constraint_id_mapping()
        return explain_constraints(constraints)
