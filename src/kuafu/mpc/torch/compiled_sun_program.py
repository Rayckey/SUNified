from pathlib import Path
from dataclasses import dataclass
from typing import Any, List, Dict, Tuple, Union, Set, FrozenSet, Optional, Iterable
import numpy as np
import torch
import json
from copy import deepcopy
from kuafu.programs.llm.program_residuals import SUNProgramResiduals
from kuafu.programs.llm.program_predicates import SUNProgramPredicates
from kuafu.programs.compiler.residual_generator import ResidualModuleGenerator
from kuafu.programs.compiler.predicate_generator import PredicateModuleGenerator
from kuafu.programs.compiler.predicate_residual_generator import PredicateResidualModuleGenerator

from kuafu.programs.llm.utils import (
    create_literal_from_raw,
    parse_util_functions_from_text,
    parse_llm_functions_from_text,
    load_text,
)

from kuafu.programs.llm.structs import Literal, Residual


@dataclass
class SUNProgramStage:
    """A SUN Program stage with its action, constraints, and compiled functions."""

    name: str
    atomic_action: Optional[str] = None
    constraints: Optional[Dict[str, Dict]] = None
    residual_scope: Optional[Dict[str, Any]] = None
    predicate_scope: Optional[Dict[str, Any]] = None


class CompiledSUNProgram:
    """Compile SUN Program stages and track each environment's stage progress.

    Stages share compiled residuals and predicates across the batch, while each
    environment advances independently. Parameter updates preserve the stage's
    constraint structure and recompile both scopes from the updated arguments.
    """

    def __init__(
        self,
        controller_api_wrapper,
        program_runtime,
    ):

        # controller api wrapper
        self.controller_api_wrapper = controller_api_wrapper
        self.program_runtime = program_runtime

        self._predicate_generator = PredicateModuleGenerator()
        self._residual_generator = ResidualModuleGenerator()
        self._predicate_residual_generator = PredicateResidualModuleGenerator()


        self._residuals = SUNProgramResiduals()
        self._predicates = SUNProgramPredicates()

        self._program_stages: List[SUNProgramStage] = []
        self._batch_stage_indices: Optional[torch.Tensor] = None

        self.exceptions = {"state"}
        self.function_exceptions = set(
            self.controller_api_wrapper.get_all_residual_apis().keys()
        )

    def load_initial_residual(self, load_dir: Path):
        # read existing python functions
        loaded_python_functions = load_text(load_dir)
        util_functions = parse_util_functions_from_text(loaded_python_functions)
        context = loaded_python_functions.split("# Utility functions:")[0]

        # load example residuals
        parsed_residuals, new_functions, dependency = parse_llm_functions_from_text(
            loaded_python_functions, known_residual_names=[]
        )

        self._residuals.add_context_code(context)
        self._residuals.add_util_functions(util_functions)
        self._residuals.add_residuals(parsed_residuals, new_functions)

    def load_initial_predicate(self, load_dir: Path):
        # read existing python functions
        loaded_python_functions = load_text(load_dir)
        util_functions = parse_util_functions_from_text(loaded_python_functions)
        context = loaded_python_functions.split("# Utility functions:")[0]

        # load example predicates
        parsed_predicates, new_functions, dependency = parse_llm_functions_from_text(
            loaded_python_functions, known_residual_names=[], keyword="predicate"
        )
        self._predicates.add_context_code(context)
        self._predicates.add_util_functions(util_functions)
        self._predicates.add_predicates(parsed_predicates, new_functions)

    def _load_stage_directory(
        self, stage_dir: Union[str, Path], *, print_functions: bool = False
    ) -> SUNProgramStage:
        """
        Parse predicates and literals from a stage directory and return a ready-to-use container.
        """
        stage_dir = Path(stage_dir)
        constraint_path = stage_dir / "constraints.json"
        atomic_action_path = stage_dir / "atomic.txt"

        if not constraint_path.exists():
            raise FileNotFoundError(f"Missing constraint file at {constraint_path}")

        with open(constraint_path, "r") as constraint_file:
            constraints = json.load(constraint_file)

        with open(atomic_action_path, "r", encoding="utf-8") as atomic_action_file:
            atomic_action = atomic_action_file.read().strip()

        # Use properly ordered variables for scope conversion.
        # Only the predicate function itself is printed when print_functions is True; not residual, not helpers, not per-term components.
        residual_scope = self.convert_residuals_to_scope(
            constraints=constraints,
            should_print=False,
        )
        predicate_scope = self.convert_predicates_to_scope(
            constraints=constraints, print_function=print_functions
        )

        return SUNProgramStage(
            name=stage_dir.name,
            atomic_action=atomic_action,
            constraints=constraints,
            residual_scope=residual_scope,
            predicate_scope=predicate_scope
        )

    def load_stage_from_constraints(
        self,
        constraints: Dict[str, Any],
        stage_index: int,
        stage_name: Optional[str] = None,
        *,
        print_functions: bool = False,
    ) -> SUNProgramStage:
        """
        Replace an existing stage's constraints and recompile its residual and predicate scopes.
        
        STRICT RULES (when only parameters are changing):
        - Cannot change the number of constraints
        - Cannot change constraint operators ("op")
        - Cannot change constraint logic ("logic")
        - Only "args" (parameters) can be updated
        
        Scopes are fully recompiled from the new constraints (not updated in-place).
        
        Args:
            constraints: Dictionary containing constraint definitions (same format as constraints.json)
            stage_index: Index of the existing stage to update (required, cannot create new stages)
            stage_name: Optional name for the stage. If not provided, keeps current name.
        
        Returns:
            The updated SUNProgramStage object.
        
        Raises:
            IndexError: If stage_index is out of bounds
            ValueError: If the number of constraints or constraint structure doesn't match
        """
        if stage_index < 0 or stage_index >= len(self._program_stages):
            raise IndexError(
                f"Stage index {stage_index} is out of bounds (0-{len(self._program_stages) - 1})."
            )

        # Get existing stage for validation
        existing_stage = self._program_stages[stage_index]
        existing_constraints = existing_stage.constraints.get("Constraints", [])
        new_constraints = constraints.get("Constraints", [])

        # Validate: number of constraints must match
        if len(new_constraints) != len(existing_constraints):
            raise ValueError(
                f"Cannot change the number of constraints. "
                f"Stage {stage_index} has {len(existing_constraints)} constraints, "
                f"but new constraints has {len(new_constraints)} constraints."
            )

        # Validate: each constraint must have matching "op" and "logic"
        for idx, (existing_c, new_c) in enumerate(zip(existing_constraints, new_constraints)):
            existing_op = existing_c.get("op")
            new_op = new_c.get("op")
            existing_logic = existing_c.get("logic")
            new_logic = new_c.get("logic")

            if existing_op != new_op:
                raise ValueError(
                    f"Cannot change constraint operator. "
                    f"Constraint {idx} in stage {stage_index} has op '{existing_op}', "
                    f"but new constraint has op '{new_op}'. Only 'args' can be updated."
                )

            if existing_logic != new_logic:
                raise ValueError(
                    f"Cannot change constraint logic. "
                    f"Constraint {idx} in stage {stage_index} has logic '{existing_logic}', "
                    f"but new constraint has logic '{new_logic}'. Only 'args' can be updated."
                )

        # Validation passed - recompile scopes from the new constraints (full compile, no in-place adjust).
        # Only the predicate function itself is printed when print_functions is True; not residual, not helpers, not per-term components.
        residual_scope = self.convert_residuals_to_scope(
            constraints=constraints,
            should_print=False,
        )
        predicate_scope = self.convert_predicates_to_scope(
            constraints=constraints,
            print_function=print_functions,
        )

        self._program_stages[stage_index] = SUNProgramStage(
            name=stage_name if stage_name is not None else existing_stage.name,
            atomic_action=existing_stage.atomic_action,
            constraints=constraints,
            residual_scope=residual_scope,
            predicate_scope=predicate_scope
        )
        return self._program_stages[stage_index]

    def load_program_stages(
        self,
        stage_dirs: List[Union[str, Path]],
        *,
        reset_existing: bool = True,
        print_functions: bool = False,
    ) -> List[SUNProgramStage]:
        """
        Load residual functions and every stage directory so batch simulations can share them.
        """
        if len(stage_dirs) == 0:
            raise ValueError("stage_dirs must contain at least one directory.")

        self._program_stages = []
        self.reset_batch_tracking()

        for idx, stage_dir in enumerate(stage_dirs):
            stage_info = self._load_stage_directory(
                stage_dir, print_functions=print_functions
            )
            self._program_stages.append(stage_info)

        return self._program_stages

    def select_stage(self, stage_index: int) -> SUNProgramStage:
        """
        Switch the compiled program's active literals to the requested stage.
        """
        if not self._program_stages:
            raise RuntimeError("No stage program is loaded.")
        if stage_index < 0 or stage_index >= len(self._program_stages):
            raise IndexError(
                f"Stage index {stage_index} is out of bounds (0-{len(self._program_stages) - 1})."
            )

        stage = self._program_stages[stage_index]
        self._set_current_stage(stage)
        return stage

    def reset_batch_tracking(self) -> None:
        """
        Clear any cached batch-progress tensors.
        """
        self._batch_stage_indices = None

    @property
    def stage_indices(self):
        return self._batch_stage_indices

    def initialize_batch_progress(
        self, batch_size: int, initial_stage_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Prepare tensors that keep track of which batch element is assigned to which stage.
        """
        if not self._program_stages:
            raise RuntimeError(
                "Load stage program before initializing batch progress."
            )
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        device = self.controller_api_wrapper.device
        if initial_stage_indices is None:
            stage_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
        else:
            stage_indices = initial_stage_indices.to(device=device, dtype=torch.long)
            if stage_indices.shape[0] != batch_size:
                raise ValueError(
                    "initial_stage_indices must have the same length as batch_size."
                )

        if torch.any(stage_indices >= len(self._program_stages)) or torch.any(
            stage_indices < 0
        ):
            raise ValueError("Provided stage indices fall outside the known stages.")

        self._batch_stage_indices = stage_indices
        return self._batch_stage_indices.clone()

    def advance_batch_stage(self, success_mask: torch.Tensor) -> torch.Tensor:
        """
        Increment the stage index for every batch element that satisfied its success predicate.
        """
        if self._batch_stage_indices is None:
            raise RuntimeError("Batch tracking is not initialized.")
        if success_mask.shape[0] != self._batch_stage_indices.shape[0]:
            raise ValueError(
                "success_mask must have the same batch dimension as the tracked indices."
            )

        success_mask = success_mask.to(device=self.controller_api_wrapper.device, dtype=torch.bool)
        final_stage = len(self._program_stages) - 1

        active_mask = torch.logical_and(
            success_mask, self._batch_stage_indices < final_stage
        )
        self._batch_stage_indices = self._batch_stage_indices + active_mask.long()

        return self._batch_stage_indices.clone()

    def batch_stage_assignments(
        self,
    ) -> Iterable[Tuple[int, torch.Tensor, SUNProgramStage]]:
        """
        Yield (stage_index, batch_mask, stage) tuples for the currently tracked batch.
        """
        if self._batch_stage_indices is None:
            raise RuntimeError("Batch tracking is not initialized.")

        for stage_id, env_ids, stage in self.batch_stage_assignments_with_indices():
            mask = torch.zeros_like(self._batch_stage_indices, dtype=torch.bool)
            mask[env_ids] = True
            yield stage_id, mask, stage

    def batch_stage_assignments_with_indices(
        self,
    ) -> Iterable[Tuple[int, torch.Tensor, SUNProgramStage]]:
        """Yield active stages with integer env ids instead of boolean masks.

        The reward path repeatedly indexes controller/API tensors for each active
        stage. Returning env ids lets SUNProgramEvaluator compute that grouping once per
        reward pass and pass the same indices through APIWrapper.index_select().
        The legacy batch_stage_assignments() wrapper is kept for callers that
        still need full-batch boolean masks.
        """
        if self._batch_stage_indices is None:
            raise RuntimeError("Batch tracking is not initialized.")

        stage_indices = self._batch_stage_indices
        order = torch.argsort(stage_indices, stable=True)
        sorted_stage_indices = torch.index_select(stage_indices, 0, order)
        unique_stage_ids, counts = torch.unique_consecutive(
            sorted_stage_indices, return_counts=True
        )
        # Move only small active-stage metadata to CPU once; env-id tensors remain on device.
        unique_stage_ids_list = [int(x) for x in unique_stage_ids.detach().cpu().tolist()]
        counts_list = [int(x) for x in counts.detach().cpu().tolist()]
        offset = 0
        for stage_id, count in zip(unique_stage_ids_list, counts_list):
            env_ids = order.narrow(0, offset, count)
            offset += count
            yield stage_id, env_ids, self._program_stages[stage_id]

    def stage_count(self) -> int:
        return len(self._program_stages)

    def get_program_stages(self) -> List[SUNProgramStage]:
        return list(self._program_stages)

    def get_stage_indices(self) -> Optional[torch.Tensor]:
        """
        Return a snapshot of the current batch stage indices if batch tracking is active.
        """
        if self._batch_stage_indices is None:
            return None
        return self._batch_stage_indices.clone()

    def get_residual_scope_for_stage(self, stage_index: int) -> Dict[str, Any]:
        """
        Return (and cache) the compiled residual scope for the requested stage.
        """
        if stage_index < 0 or stage_index >= len(self._program_stages):
            raise IndexError(
                f"Stage index {stage_index} is out of bounds (0-{len(self._program_stages) - 1})."
            )
        stage = self._program_stages[stage_index]
        if stage.residual_scope is None:
            raise RuntimeError(
                f"Residual scope for stage {stage_index} is not initialized."
            )
        return stage.residual_scope

    def evaluate_stage_predicates(self, stage_index: int) -> torch.Tensor:
        """
        Compute predicate truth values for the specified stage using cached literals/scope.
        """
        if stage_index < 0 or stage_index >= len(self._program_stages):
            raise IndexError(
                f"Stage index {stage_index} is out of bounds (0-{len(self._program_stages) - 1})."
            )
        stage = self._program_stages[stage_index]
        return self.detect_predicates(stage.predicate_scope)

    def pop_current_task(self):
        """
        Pop the current task.
        """
        self._tasks.pop_task()
        if len(self._tasks.tasks) == 0:
            return False
        return True

    def load_task(self, load_dir: Path):
        return False

    def load_predicate(self, load_dir_function: Path, load_dir_literals: Path):

        predicate_function = load_text(load_dir_function)

        with open(load_dir_literals, "r") as f:
            previous_program = json.load(f)

        parsed_predicate, new_functions, _ = parse_llm_functions_from_text(
            predicate_function,
            known_residual_names=[],
            keyword="predicate",
        )

        # save to predicates
        self._predicates.add_predicates(parsed_predicate, new_functions)

        return True

    def preprocess_residual_scope(self) -> dict:
        """
        Preprocess residuals to python script.
        """
        residuals = self._residuals

        # prepare global scope
        global_scope = {"np": np, "torch": torch}
        global_scope.update({"DEVICE": self.controller_api_wrapper.device, 'BATCH': self.controller_api_wrapper.batch, 'DTYPE': self.controller_api_wrapper.dtype})
        global_scope.update(
            {name: eval(name) for name in ["Union", "List", "Dict", "Set", "FrozenSet"]}
        )
        if hasattr(self.controller_api_wrapper.controller, "ee_frame"):
            global_scope.update(
                {"state_frame": self.controller_api_wrapper.controller.ee_frame}
            )
        exec("Sym = torch.Tensor \n", global_scope)
        global_scope.update(self.controller_api_wrapper.get_all_residual_apis())

        # gather grounding functions & execute
        util_funcs = residuals.get_all_util_func_text()

        # execute util functions and residual functions (local)
        exec(util_funcs, global_scope)
        residual_funcs = residuals.get_all_residual_func_text()
        exec(residual_funcs, global_scope)

        return global_scope

    def convert_residuals_to_scope(
        self,
        constraints,
        *,
        should_print: bool = True,
        function_name: str = "residual_function",
    ) -> dict:
        """
        Convert residuals to python script, optionally overriding the literals and variables.
        """

        # prepare global scope
        global_scope = self.preprocess_residual_scope()

        self._residual_generator.compile_scope(
            global_scope,
            scope_name="residual_function",
            ir=constraints,
            print_function=should_print,
        )
        return global_scope

    def convert_predicates_to_scope(self, constraints, print_function=False):
        """
        Convert predicates to python scope.
        """
        predicates = self._predicates

        global_scope: Dict[str, Any] = {"np": np, "torch": torch}
        global_scope.update({"DEVICE": self.controller_api_wrapper.device, 'BATCH': self.controller_api_wrapper.batch, 'DTYPE': self.controller_api_wrapper.dtype})

        global_scope.update(
            {name: eval(name) for name in ["Union", "List", "Dict", "Set", "FrozenSet"]}
        )

        global_scope.update(
            {"state_frame": self.controller_api_wrapper.controller.ee_frame}
        )
        global_scope.update(self.controller_api_wrapper.get_all_predicate_apis())

        # gather grounding functions & execute
        util_funcs = predicates.get_all_util_func_text()

        # execute util functions and residual functions (local)
        exec(util_funcs, global_scope)

        predicate_funcs = predicates.get_all_predicate_func_text()
        exec(predicate_funcs, global_scope)
        # Compile combined reducer (`predicate_function`) + per-term functions.
        term_funcs = self._predicate_generator.compile_scope(
            global_scope,
            scope_name="predicate_function",
            ir=constraints,
            print_function=print_function,
        )
        global_scope["predicate_term_functions"] = term_funcs
        return global_scope

    def adjust_residuals_in_scope(
        self,
        existing_scope: Dict[str, Any],
        constraints: Dict[str, Any],
        should_print: bool = False,
    ) -> Dict[str, Any]:
        """
        Adjust existing residual scope by updating function bodies with new argument values.
        
        This is an optimized version that reuses the existing scope (utilities, APIs, etc.)
        and only updates the residual function body. Much faster than full recompilation.
        
        Args:
            existing_scope: The existing residual scope to update
            constraints: New constraint definitions (only args should differ)
            should_print: Whether to print the generated function source
        
        Returns:
            The updated scope (same dict, functions updated in-place)
        """
        # Reuse existing scope - it already has all utilities, APIs, etc.
        # Only update the residual function body with new argument values
        self._residual_generator.adjust_scope(
            existing_scope,
            scope_name="residual_function",
            ir=constraints,
        )
        return existing_scope

    def adjust_predicates_in_scope(
        self,
        existing_scope: Dict[str, Any],
        constraints: Dict[str, Any],
        existing_term_func_names: List[str],
        print_function: bool = False,
    ) -> Dict[str, Any]:
        """
        Adjust existing predicate scope by updating function bodies with new argument values.
        
        This is an optimized version that reuses the existing scope (utilities, APIs, etc.)
        and only updates the predicate function bodies. Much faster than full recompilation.
        
        Args:
            existing_scope: The existing predicate scope to update
            constraints: New constraint definitions (only args should differ)
            existing_term_func_names: List of existing term function names
            print_function: Whether to print the generated function source
        
        Returns:
            The updated scope (same dict, functions updated in-place)
        """
        # Reuse existing scope - it already has all utilities, APIs, etc.
        # Only update the predicate function bodies with new argument values
        term_func_names = self._predicate_generator.adjust_scope(
            existing_scope,
            scope_name="predicate_function",
            ir=constraints,
            existing_term_func_names=existing_term_func_names,
            print_function=print_function,
        )
        # Update the term function names list (should be the same, but update for consistency)
        existing_scope["predicate_term_functions"] = term_func_names
        return existing_scope

    def detect_predicates(
        self,
        predicate_scope: Optional[Dict[str, Any]] = None,
        text: bool = False,
    ):
        if predicate_scope is None:
            raise ValueError("predicate_scope must be provided")
        pred_fn = predicate_scope.get("predicate_function")
        if pred_fn is None or not callable(pred_fn):
            raise KeyError("predicate_scope must contain a callable 'predicate_function'")
        res = pred_fn()
        if not isinstance(res, torch.Tensor):
            raise TypeError(
                f"predicate_function() must return a torch.Tensor, got {type(res)}"
            )

        if not text:
            return res

        # Debug: evaluate each compiled per-term predicate and print output.
        # Term functions are created by PredicateModuleGenerator.compile_scope() and the
        # list is stored in the scope as `predicate_term_functions`.
        lines: List[str] = []
        term_names = predicate_scope.get("predicate_term_functions", [])
        if isinstance(term_names, list) and len(term_names) > 0:
            for i, fn_name in enumerate(term_names):
                fn = predicate_scope.get(fn_name)
                if fn is None or not callable(fn):
                    lines.append(f"[term {i}] {fn_name}: <missing>")
                    continue
                try:
                    term_res = fn()
                except Exception as e:
                    lines.append(f"[term {i}] {fn_name}: ERROR: {e}")
                    continue
                lines.append(f"[term {i}] {fn_name}: {term_res}")
        else:
            lines.append("No per-term predicate functions found in scope (predicate_term_functions missing).")

        print("\n".join(lines))
        return res
