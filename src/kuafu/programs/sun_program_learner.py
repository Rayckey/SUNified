from typing import Any, Dict, Optional
import json
from pathlib import Path
import numpy as np
import casadi as ca
from typing import Any, List, Dict, Tuple, Union, Set, FrozenSet
import copy
import logging
from kuafu.programs.llm.base_agent import BaseAgent
from kuafu.programs.llm.constraint_generator import SUNConstraintGenerator
from kuafu.programs.llm.io_util import write_txt
from kuafu.mpc.controller.mpc import ManipulatorIKController, DifferentialDriveController

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SUNProgramLearner(BaseAgent):
    """Build and revise SUN Program stages through a model and controller.

    Task planning, atomic-action grounding, and constraint generation share the
    same plan history so execution feedback can repair the current stage.
    """

    def __init__(self, program_runtime, agent_cfg, *args, **kwargs):
        super().__init__(program_runtime, agent_cfg)
        self.device = agent_cfg.device
        self.had_introduced_task = False
        self.had_introduced_code = False

        if issubclass(type(self._controller_api_wrapper.controller), ManipulatorIKController):
            self._constraint_generator = SUNConstraintGenerator(
                prompt_dir=agent_cfg.prompt_dir,
                controller_api_wrapper=self._controller_api_wrapper,
                program_runtime=self._program_runtime,
                model_version=agent_cfg.model_version,
                goal_text=agent_cfg.task,
                device=agent_cfg.device,
            )

        else:
            raise NotImplementedError("Only ManipulatorIKController is supported for SUNProgramLearner")

        self._constraint_generator.load_program(
            agent_cfg.initial_residual, agent_cfg.initial_predicate
        )

    def terminate_current_task(self):
        """
        Move on to the next subtask, this is used to indicate the end of a subtask.
        """
        self.clear_current_program()
        return self._constraint_generator.pop_current_task()

    def terminate_all_tasks(self):
        """
        Remove all sub tasks, this is either used to indicate the end of all subtasks or after triggering replan.
        """
        self.clear_current_program()
        return self._constraint_generator.clear_task()

    def clear_current_program(self):
        self._constraint_generator._constraints.clear()

    def has_tasks(self):
        return self._constraint_generator._tasks.has_tasks

    def query_task(self, memory: str, images: list, max_iteration: int):
        self.clear_current_program()
        num_llm_calls, success = self._constraint_generator.learn_tasks_from_description(
            memory, images, max_iteration
        )

        return num_llm_calls, success

    def query_atomic_actions(
        self, memory: str, images: list, max_iteration: int, error: Optional[str] = None
    ):
        """
        Stage 2: Post-Task - Ground the current high-level subtask into atomic actions.
        This should be called after query_task() and before query_predicate().
        """
        self.clear_current_program()
        num_llm_calls, success = self._constraint_generator.learn_atomic_actions_from_description(
            memory, images, max_iteration, error=error
        )

        return num_llm_calls, success

    def query_constraint(
        self, memory: str, images: list, max_iteration: int, error: Optional[str] = None
    ):

        num_llm_calls, success = (
            self._constraint_generator.learn_constraints_from_description(
                 memory, images, max_iteration, error=error
            )
        )

        return num_llm_calls, success

    def computeControl(self, state, control):
        return self._program_runtime.active_controller.compute_control(state, control)

    def detectPredicates(self, text: bool, max_envs: Optional[int] = None):
        return self._constraint_generator.detect_predicates(
            provide_text=text, max_envs=max_envs
        )

    def detectReplans(self):
        return self._constraint_generator.detect_replans()

    def apply_constraints(self, print_function=False):
        """
        Apply constraints to the controller.
        
        Parameters
        ----------
        print_function : bool, default False
            If True, print the generated residual function source.
        """
        self._constraint_generator.apply_constraints(print_function=print_function)


    def save_program(
        self, save_dir_task, save_dir_atomic, save_dir_constraint
    ):
        """
        Save current agent program.
        """

        self._constraint_generator.save_program(
            save_dir_task, save_dir_atomic, save_dir_constraint
        )

    def load_program(self, load_dir_residual: str | Path, load_dir_predicate: str|Path):
        """
        Load current agent program.
        """
        self._constraint_generator.load_program(
            load_dir_residual, load_dir_predicate
        )

    def load_agent_constraint(self, load_dir_constraint: Path):
        with open(load_dir_constraint, "r") as f:
            llm_constraints = json.load(f)
        self._constraint_generator._constraints = llm_constraints
