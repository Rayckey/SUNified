import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from copy import deepcopy
import traceback
import json
import numpy as np
import torch
from kuafu.programs.llm.program_generator import LLMBase
from kuafu.programs.llm.program_residuals import SUNProgramResiduals
from kuafu.programs.llm.program_predicates import SUNProgramPredicates
from kuafu.programs.compiler.residual_generator import ResidualModuleGenerator
from kuafu.programs.compiler.predicate_generator import PredicateModuleGenerator
from kuafu.skills.constraints.tolerance_policy import (
    geometry_for_prompt,
    receive_constraint_geometry,
)
from kuafu.programs.llm.sun_program_plan import SUNProgramPlan
import casadi as ca
from typing import Any, List, Dict, Tuple, Union, Set, FrozenSet
from kuafu.programs.llm.utils import (
    create_literal_from_raw,
    parse_util_functions_from_text,
    parse_llm_functions_from_text,
    load_text,
    write_txt,
    compile_observation,
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

_TYPING_EXPORTS = {
    "Union": Union,
    "List": List,
    "Dict": Dict,
    "Set": Set,
    "FrozenSet": FrozenSet,
}


class SUNConstraintGenerator(LLMBase):
    """Generate SUN Program tasks, actions, and constraints from scene context."""

    def __init__(
        self,
        prompt_dir: str,
        controller_api_wrapper,
        program_runtime,
        device: str = 'cpu',
        model_version: str = "gpt-5.5",
        goal_text: str = "",
    ):
        super().__init__(model_version=model_version)
        self.device = device
        self.prompt_dir = prompt_dir

        self.controller_api_wrapper = controller_api_wrapper
        self.program_runtime = program_runtime

        self._tasks = SUNProgramPlan()
        self._goal_text = goal_text

        self._residuals = SUNProgramResiduals()
        self._predicates = SUNProgramPredicates()

        self._predicate_generator = PredicateModuleGenerator()
        self._residual_generator = ResidualModuleGenerator()

        self._constraints = {"Constraints": []}
        self._active_constraints = {}
        self._predicate_scope: Dict[str, Any] = {}
        self._residual_scope: Dict[str, Any] = {}


        prompt_template_folder = Path(self.prompt_dir)

        # Pre-task stage (high-level task decomposition)
        self._pre_task_context = load_text(
            os.path.join(prompt_template_folder, "context", "pre_task_invent_context" + ".txt")
        )
        self._pre_task_context_continue = load_text(
            os.path.join(prompt_template_folder, "context", "pre_task_correct_context" + ".txt")
        )
        self._pre_task_init = load_text(
            os.path.join(prompt_template_folder, "pre_task_invent_prompt" + ".txt")
        )
        self._pre_task_continue = load_text(
            os.path.join(prompt_template_folder, "pre_task_correct_prompt" + ".txt")
        )

        # Post-task stage (ground tasks into detailed actions)
        self._post_task_context = load_text(
            os.path.join(prompt_template_folder, "context", "post_task_invent_context" + ".txt")
        )
        self._post_task_init = load_text(
            os.path.join(prompt_template_folder, "post_task_invent_prompt" + ".txt")
        )

        # Predicate stage
        self._constraint_context = load_text(
            os.path.join(prompt_template_folder, "context", "constraint_context" + ".txt")
        )
        self._constraint_init = load_text(
            os.path.join(prompt_template_folder, "constraint_prompt" + ".txt")
        )

        self._constraint_context_continue = load_text(
            os.path.join(prompt_template_folder, "context", "constraint_correct_context" + ".txt")
        )
        self._constraint_continue = load_text(
            os.path.join(prompt_template_folder, "constraint_correct_prompt" + ".txt")
        )

    def load_initial_residual(self, load_dir: str | Path):
        loaded_python_functions = load_text(load_dir)
        util_functions = parse_util_functions_from_text(loaded_python_functions)
        context = loaded_python_functions.split("# Utility functions:")[0]

        parsed_residuals, new_functions, dependency = parse_llm_functions_from_text(
            loaded_python_functions, known_residual_names=[]
        )

        self._residuals.add_context_code(context)
        self._residuals.add_util_functions(util_functions)

        self._residuals.add_residuals(parsed_residuals, new_functions)

    def load_initial_predicate(self, load_dir: str | Path):
        loaded_python_functions = load_text(load_dir)
        util_functions = parse_util_functions_from_text(loaded_python_functions)
        context = loaded_python_functions.split("# Utility functions:")[0]

        parsed_predicates, new_functions, dependency = parse_llm_functions_from_text(
            loaded_python_functions, known_residual_names=[], keyword="predicate"
        )
        self._predicates.add_context_code(context)
        self._predicates.add_util_functions(util_functions)
        self._predicates.add_predicates(parsed_predicates, new_functions)

    def load_program(self, load_dir_residual: str | Path, load_dir_predicate: str | Path):
        """
        Load agent program.
        """

        self.load_initial_residual(load_dir_residual)
        self.load_initial_predicate(load_dir_predicate)

    def save_program(
        self,
        save_dir_task: Path,
        save_dir_atomic_task: Path,
        save_dir_constraint: Path,
    ):
        """
        Save current agent program.
        """
        # save residuals in seperate python files (function names overlap)
        write_txt(save_dir_task, self._tasks.current_task)
        write_txt(save_dir_atomic_task, str(tuple(self._tasks.current_atomic_actions)))
        json.dump(self._constraints, open(save_dir_constraint, "w"), indent=4)

    def pop_current_task(self):
        """
        Pop the current task.
        """
        self._tasks.pop_task()
        if len(self._tasks.tasks) == 0:
            return False
        return True

    def clear_task(self):
        """
        remove the task queue
        """
        while len(self._tasks.tasks) > 0:
            self._tasks.pop_task()

    def learn_tasks_from_description(
        self,
        memory: str,
        images: list[np.array],
        max_iteration: int = 3,
    ):
        # Stage 1: Pre-task (high-level task decomposition)
        _iter = 0

        while _iter < max_iteration:
            if len(self._tasks.tasks) == 0:
                self._invent_pre_tasks_from_description(memory, images)
            else:
                self._continue_pre_tasks_from_description(memory, images)

            return self.num_llm_calls, True

        return self.num_llm_calls, False

    def learn_atomic_actions_from_description(
        self,
        memory: str,
        images: list[np.array],
        max_iteration: int = 3,
        error: Optional[str] = None,
    ):
        # Stage 2: Post-task (ground current subtask into atomic actions)
        _iter = 0

        while _iter < max_iteration:
            try:
                self._invent_post_tasks_from_description(memory, images, error=error)
                return self.num_llm_calls, True
            except Exception as e:
                message = f"{e}\nFailed to ground subtask into atomic actions, retrying..."
                logger.info(message)
                print(message, flush=True)
                error_message = str(traceback.format_exc())
                logger.warning(error_message)
                print(error_message, flush=True)
                _iter += 1

        return self.num_llm_calls, False

    def learn_constraints_from_description(
        self,
        memory: str,
        images: list[np.array],
        max_iteration: int = 3,
        error: Optional[str] = None,
    ):
        _iter = 1
        try:
            if not self._constraints.get("Constraints"):
                self._invent_constraints_from_description( memory, images)
            else: 
                self._continue_constraints_from_description(
                        memory, [], error=error or "failed to meet constraints within time limit"
                )                
            self.parse_predicates(
                self._constraints
            )
            self.parse_residuals(self._constraints)

            return self.num_llm_calls, True

        except Exception as e:
            message = f"{e}\nFailed to invent predicates, correcting..."
            logger.info(message)
            print(message, flush=True)
            error_message = str(traceback.format_exc())
            logger.warning(error_message)
            print(error_message, flush=True)
            error_message = str(e)
            while _iter < max_iteration:
                try:
                    self._continue_constraints_from_description(
                         memory, [], error=error_message
                    )
                    self.parse_predicates(
                        self._constraints
                    )
                    self.parse_residuals(self._constraints)
                    return self.num_llm_calls, True
                except Exception as e:
                    message = f"{e}\nFailed to correct predicates, retrying..."
                    logger.info(message)
                    print(message, flush=True)
                    error_message = str(traceback.format_exc())
                    logger.warning(error_message)
                    print(error_message, flush=True)
                    error_message = str(e)
                    _iter += 1
                    continue

        return self.num_llm_calls, False


    def _invent_pre_tasks_from_description(
        self,
        memory: str,
        images: list[np.array],
    ) -> bool:
        logger.info(f"Inventing high-level tasks from memory")

        invent_tasks_context = self._pre_task_context

        invent_tasks_prompt = deepcopy(self._pre_task_init)

        invent_tasks_prompt = invent_tasks_prompt.replace(
            "{frames}",
            compile_observation(
                self.program_runtime.active_controller.getFrameDescription(),
                include_size=False,
                include_pose=False,
            )
        )

        invent_tasks_context = invent_tasks_context.replace(
            "{state}",
            str(tuple(self.program_runtime.active_controller.getStateDescription())),
        )

        invent_tasks_context = invent_tasks_context.replace(
            "{control}",
            str(tuple(self.program_runtime.active_controller.getControlDescription())),
        )

        invent_tasks_context = invent_tasks_context.replace(
            "{apis}",
            str(self.controller_api_wrapper.get_all_predicate_apis_description()),
        )

        invent_tasks_context = invent_tasks_context.replace(
            "{residual}", self._residuals.get_available_residuals_text()
        )

        invent_tasks_context = invent_tasks_context.replace(
            "{robot}", self.controller_api_wrapper.controller.name
        )

        if len(memory) == 0:
            invent_tasks_prompt = invent_tasks_prompt.replace("{history}", "None")
        else:
            invent_tasks_prompt = invent_tasks_prompt.replace(
                "{history}", str((memory))
            )

        invent_tasks_prompt = invent_tasks_prompt.replace(
            "{task_description}", self._goal_text
        )

        image_content = []

        llm_output = self.prompt_llm(
            invent_tasks_context,
            invent_tasks_prompt,
            image_content,
            force_format="json_object",
        )
        llm_output = json.loads(llm_output)

        self.num_llm_calls += 1

        self._tasks.update_tasks(llm_output["Subtasks"])
        self._tasks.reasoning = llm_output["Reasoning"]

        return True

    def _continue_pre_tasks_from_description(
        self,
        memory: str,
        images: list[np.array],
        error="Failed to complete this subtask within allowed time limit",
    ) -> bool:
        logger.info(f"correcting high-level tasks from description")

        invent_tasks_context = self._pre_task_context_continue
        invent_tasks_prompt = deepcopy(self._pre_task_continue)

        if self.program_runtime.hasKinematics():
            invent_tasks_prompt = invent_tasks_prompt.replace(
                "{frames}", str(self.program_runtime.active_controller.getFrameDescription())
            )
        else:
            invent_tasks_prompt = invent_tasks_prompt.replace("{frames}", "None")

        if len(memory) == 0:
            invent_tasks_prompt = invent_tasks_prompt.replace("{history}", "None")
        else:
            invent_tasks_prompt = invent_tasks_prompt.replace("{history}", str(memory))

        invent_tasks_prompt = invent_tasks_prompt.replace(
            "{task_description}", self._goal_text
        )

        invent_tasks_prompt = invent_tasks_prompt.replace(
            "{state}",
            str(tuple(self.program_runtime.active_controller.getStateDescription())),
        )

        invent_tasks_prompt = invent_tasks_prompt.replace(
            "{control}",
            str(tuple(self.program_runtime.active_controller.getControlDescription())),
        )

        invent_tasks_prompt = invent_tasks_prompt.replace(
            "{apis}",
            str(self.controller_api_wrapper.get_all_predicate_apis_description()),
        )

        invent_tasks_prompt = invent_tasks_prompt.replace(
            "{robot}",
            self.controller_api_wrapper.controller.name
        )

        invent_tasks_prompt = invent_tasks_prompt.replace(
            "{residual}", self._residuals.get_available_residuals_text()
        )

        invent_tasks_prompt = invent_tasks_prompt.replace(
            "{current}", self._tasks.current_task
        )

        joined_string_list = "][".join(list(self._tasks.completed_tasks))

        invent_tasks_prompt = invent_tasks_prompt.replace(
            "{completed}", f"[{joined_string_list}]"
        )

        joined_string_list = "][".join(list(self._tasks.pending_tasks))

        invent_tasks_prompt = invent_tasks_prompt.replace(
            "{pending}", f"[{joined_string_list}]"
        )

        invent_tasks_prompt = invent_tasks_prompt.replace("{error}", error)

        invent_tasks_prompt = invent_tasks_prompt.replace(
            "{planner_reasoning}", self._tasks.reasoning if self._tasks.reasoning else "None"
        )

        joined_string_list = "][".join(self._tasks.tasks)
        invent_tasks_prompt = invent_tasks_prompt.replace(
            "{planner_subtasks}", f"[{joined_string_list}]"
        )

        joined_string_list = "][".join(list(self._tasks.completed_tasks))
        invent_tasks_prompt = invent_tasks_prompt.replace(
            "{completed_plan}", f"[{joined_string_list}]"
        )

        if len(images) > 0:
            invent_tasks_prompt = invent_tasks_prompt.replace(
                "{images}", f"{len(images)} image(s) embedded below"
            )
        else:
            invent_tasks_prompt = invent_tasks_prompt.replace("{images}", "None")

        image_content = []
        for img in images:
            invent_tasks_prompt, embedded_img = self.get_embedded_image(
                invent_tasks_prompt, img
            )
            image_content.append(embedded_img)

        llm_output = self.prompt_llm(
            invent_tasks_context,
            invent_tasks_prompt,
            image_content,
            force_format="json_object",
        )
        llm_output = json.loads(llm_output)

        self.num_llm_calls += 1

        self._tasks.update_tasks(llm_output["Subtasks"])
        self._tasks.reasoning = llm_output["Reasoning"]

        return True

    def _invent_post_tasks_from_description(
        self,
        memory: str,
        images: list[np.array],
        error: Optional[str] = None,
    ) -> bool:
        logger.info(f"Grounding current high-level subtask into atomic actions")

        invent_actions_context = self._post_task_context
        invent_actions_prompt = deepcopy(self._post_task_init)

        invent_actions_prompt = invent_actions_prompt.replace(
            "{frames}",
            compile_observation(
                self.program_runtime.active_controller.getFrameDescription(),
                include_size=False,
                include_pose=False,
            )
        )

        invent_actions_context = invent_actions_context.replace(
            "{state}",
            str(tuple(self.program_runtime.active_controller.getStateDescription())),
        )

        invent_actions_context = invent_actions_context.replace(
            "{control}",
            str(tuple(self.program_runtime.active_controller.getControlDescription())),
        )


        invent_actions_context = invent_actions_context.replace(
            "{residual}", self._residuals.get_available_residuals_text()
        )

        invent_actions_context = invent_actions_context.replace(
            "{robot}", self.controller_api_wrapper.controller.name
        )

        if len(memory) == 0:
            invent_actions_prompt = invent_actions_prompt.replace("{history}", "None")
        else:
            invent_actions_prompt = invent_actions_prompt.replace(
                "{history}", str((memory))
            )

        invent_actions_prompt = invent_actions_prompt.replace(
            "{task_description}", self._goal_text
        )

        invent_actions_prompt = invent_actions_prompt.replace(
            "{current_subtask}", self._tasks.current_task
        )

        # Include only previously completed atomic actions.
        if len(self._tasks.completed_task_atomic_actions) == 0:
            completed_text = "[]"
        else:
            completed_actions = []
            for completed in self._tasks.completed_task_atomic_actions[-5:]:
                atomic_actions = completed["atomic_actions"]
                completed_actions.extend(atomic_actions)
            if len(completed_actions) == 0:
                completed_text = "[]"
            else:
                completed_text = "\n".join(
                    f"- {action}" for action in completed_actions
                )

        invent_actions_prompt = invent_actions_prompt.replace(
            "{completed}", completed_text
        )

        joined_string_list = "][".join(list(self._tasks.pending_tasks))
        invent_actions_prompt = invent_actions_prompt.replace(
            "{pending}", f"[{joined_string_list}]"
        )

        invent_actions_prompt = invent_actions_prompt.replace(
            "{robot}", self.controller_api_wrapper.controller.name
        )

        invent_actions_prompt = invent_actions_prompt.replace(
            "{state}",
            str(tuple(self.program_runtime.active_controller.getStateDescription())),
        )

        invent_actions_prompt = invent_actions_prompt.replace(
            "{control}",
            str(tuple(self.program_runtime.active_controller.getControlDescription())),
        )

        invent_actions_prompt = invent_actions_prompt.replace(
            "{apis}",
            str(self.controller_api_wrapper.get_all_predicate_apis_description()),
        )


        if error is not None:
            previous_atomic = str(tuple(self._tasks.current_atomic_actions))
            invent_actions_prompt += (
                "\n\nAtomic-action correction request:\n"
                "The previous grounded atomic action failed during execution. "
                "Use the error message as the primary diagnostic signal. "
                "Keep the same high-level subtask and correct the AtomicActions "
                "natural-language grounding only; do not emit TermAPI JSON or constraints.\n"
                f"Previous AtomicActions: {previous_atomic}\n"
                f"Runtime error message:\n{error}\n"
                "Return the same JSON schema with exactly one corrected AtomicActions string."
            )

        image_content = []

        llm_output = self.prompt_llm(
            invent_actions_context,
            invent_actions_prompt,
            image_content,
            force_format="json_object",
        )
        llm_output = json.loads(llm_output)

        self.num_llm_calls += 1

        self._tasks.update_atomic_actions(llm_output["AtomicActions"])
        self._tasks.atomic_actions_reasoning = llm_output["Reasoning"]

        return True

    def _receive_constraint_geometry(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Assign thresholds from trusted scene metadata before compilation.

        This is used only for fresh LLM responses (including corrections), not
        for replaying existing saved programs with their recorded MPC values.
        """
        controller = self.program_runtime.active_controller
        objects: dict[str, Any] = {}
        entities = dict(controller.entity_dict)
        for frame in controller.getFrameDescription():
            entities.setdefault(frame.name, frame)
        for name, entity in entities.items():
            info: dict[str, Any] = {"description": str(entity.description)}
            if getattr(entity, "has_size", False):
                dimensions = entity.size
                if isinstance(dimensions, torch.Tensor):
                    dimensions = dimensions.detach().cpu().numpy()
                info["size_m"] = np.asarray(dimensions, dtype=float).reshape(-1).tolist()
            objects[str(name)] = info
        return receive_constraint_geometry(
            payload,
            atomic_action="\n".join(self._tasks.current_atomic_actions),
            detectable_objects=objects,
            profile="mpc",
        )

    def _invent_constraints_from_description(
        self,
        memory: str,
        images: list[np.array],
    ) -> bool:
        logger.info(f"Inventing predicates from memory: {memory}")

        invent_constraint_context = self._constraint_context

        invent_constraint_prompt = deepcopy(self._constraint_init)

        if self.program_runtime.hasKinematics():
            invent_constraint_prompt = invent_constraint_prompt.replace(
                "{frames}",
                compile_observation(
                    self.program_runtime.active_controller.getFrameDescription(),
                    include_size=False,
                    include_pose=False,
                ),
            )
        else:
            invent_constraint_prompt = invent_constraint_prompt.replace(
                "{frames}", "None"
            )

        if len(memory) == 0:
            invent_constraint_prompt = invent_constraint_prompt.replace(
                "{history}", "None"
            )
        else:
            invent_constraint_prompt = invent_constraint_prompt.replace(
                "{history}", str(memory)
            )

        # Use atomic actions instead of high-level task
        invent_constraint_prompt = invent_constraint_prompt.replace(
            "{task_description}", 
            str(tuple(self._tasks.current_atomic_actions)) 
        )

        invent_constraint_prompt = invent_constraint_prompt.replace(
            "{completed}", str(tuple(self._tasks.completed_tasks))
        )

        invent_constraint_prompt = invent_constraint_prompt.replace(
            "{pending}", str(tuple(self._tasks.pending_tasks))
        )

        llm_output = self.prompt_llm(
            invent_constraint_context,
            invent_constraint_prompt,
            force_format="json_object",
        )
        llm_output = json.loads(llm_output)

        self.num_llm_calls += 1


        self._constraints = self._receive_constraint_geometry(llm_output)

        return True

    def _continue_constraints_from_description(
        self,
        memory: str,
        images: list[np.array],
        error: str = "failed to meet constraints within time limit",
    ) -> bool:
        logger.info(f"Continuing predicates update from memory: {memory}")

        invent_constraint_context = self._constraint_context_continue

        invent_constraint_prompt = deepcopy(self._constraint_continue)

        if self.program_runtime.hasKinematics():
            invent_constraint_prompt = invent_constraint_prompt.replace(
                "{frames}",
                compile_observation(
                    self.program_runtime.active_controller.getFrameDescription(),
                    include_size=False,
                    include_pose=False,
                ),
            )
        else:
            invent_constraint_prompt = invent_constraint_prompt.replace(
                "{frames}", "None"
            )

        if len(memory) == 0:
            invent_constraint_prompt = invent_constraint_prompt.replace(
                "{history}", "None"
            )
        else:
            invent_constraint_prompt = invent_constraint_prompt.replace(
                "{history}", str(memory)
            )

        # Use atomic actions instead of high-level task
        invent_constraint_prompt = invent_constraint_prompt.replace(
            "{task_description}", 
            str(tuple(self._tasks.current_atomic_actions))
        )

        invent_constraint_prompt = invent_constraint_prompt.replace(
            "{completed}", str(tuple(self._tasks.completed_tasks))
        )

        invent_constraint_prompt = invent_constraint_prompt.replace(
            "{pending}", str(tuple(self._tasks.pending_tasks))
        )


        invent_constraint_prompt = invent_constraint_prompt.replace(
            "{previous_grounding}",
            json.dumps(geometry_for_prompt(self._constraints)),
        )

        if error == "failed to meet constraints within time limit":
            if self.program_runtime.hasCollision().item():
                error = "The robot has stopped moving for extended period of time and its effort is significant. The robot is likely in contact and actively pushing or pulling something. Please review your constraints, if high manipulator effort is expected, it must be added as a condition to a constraint.  Current results: \n "

            elif self.program_runtime.hasStalled() and not self.program_runtime.hasCollision().item():
                error = "The robot has stopped moving for an extended period with low effort. Check conflicting geometry, frame/axis choices, targets, and missing contact conditions. Current results:\n"

            else:
                error = "The robot has been moving for extended period of time but its effort is not significant. The most likely causes is that the constriants are conflicting and cannot be met. Current results: \n "

            results = self.detect_predicates(provide_text=True)
            error += results[1]

        invent_constraint_prompt = invent_constraint_prompt.replace(
            "{error_message}",
            error,
        )

        llm_output = self.prompt_llm(
            invent_constraint_context,
            invent_constraint_prompt,
            [],
            force_format="json_object",
        )
        llm_output = json.loads(llm_output)

        self.num_llm_calls += 1

        self._constraints = self._receive_constraint_geometry(llm_output)

        return True

    def convert_residuals_to_scope(
        self, constraints,print_function=False
    ) -> dict:
        """
        Convert residuals to python script.
        """

        global_scope = {"np": np, "ca": ca}
        global_scope.update(_TYPING_EXPORTS)
        if hasattr(self.controller_api_wrapper.controller, "ee_frame"):
            global_scope.update(
                {"state_frame": self.controller_api_wrapper.controller.ee_frame}
            )
        global_scope["Sym"] = Union[ca.MX, ca.SX, ca.DM]
        global_scope.update(self.controller_api_wrapper.get_all_residual_apis())

        # gather grounding functions & execute
        util_funcs = self._residuals.get_all_util_func_text()

        # execute util functions and residual functions (local)
        exec(util_funcs, global_scope)
        residual_funcs = self._residuals.get_all_residual_func_text()
        exec(residual_funcs, global_scope)

        self._residual_generator.compile_scope(global_scope, scope_name="residual_function", ir=constraints, print_function=print_function)

        return global_scope

    def convert_predicates_to_scope(
        self, constraints, print_function=False
    ):
        """
        Convert predicates to python scope.
        """

        global_scope: Dict[str, Any] = {"np": np, "torch": torch}
        global_scope.update({"DEVICE": self.controller_api_wrapper.device, 'BATCH': self.controller_api_wrapper.batch, 'DTYPE': self.controller_api_wrapper.dtype})

        global_scope.update(_TYPING_EXPORTS)
        if hasattr(self.controller_api_wrapper.controller, "ee_frame"):
            global_scope.update(
                {"state_frame": self.controller_api_wrapper.controller.ee_frame}
            )
        global_scope.update(self.controller_api_wrapper.get_all_predicate_apis())

        # gather grounding functions & execute
        util_funcs = self._predicates.get_all_util_func_text()

        # execute util functions and residual functions (local)
        exec(util_funcs, global_scope)
        predicate_funcs = self._predicates.get_all_predicate_func_text()
        exec(predicate_funcs, global_scope)

        def memo(keyu, fn):
            return

        global_scope.update({"memo": memo})

        # Compile combined reducer (`predicate_function`) + per-term functions.
        term_funcs = self._predicate_generator.compile_scope(
            global_scope,
            scope_name="predicate_function",
            ir=constraints,
            print_function=print_function,
        )
        # Let callers/debuggers know which functions to call (in order).
        global_scope["predicate_term_functions"] = term_funcs

        return global_scope

    def parse_residuals(self, constraints: dict):
        """
        Parse literals with residuals.
        Args:
            observation: str
            controller_api_wrapper: controllerAPIWrapper
            residual_name: if not None, only parse this residual
            positive_only: only consider residuals that mark essential
            include_negative_residuals: include invented negative residuals
        """

        _residual_scope = self.convert_residuals_to_scope(constraints, print_function=False)

        self.program_runtime.test_controller.update_cost_function(_residual_scope)

        self.program_runtime.test_controller.compute_control(
            np.random.random((self.program_runtime.test_controller.num_states)),
            np.random.random((self.program_runtime.test_controller.num_controls)),
        )

        return

    def parse_predicates(
        self, constraints
    ):
        """
        Parse literals with residuals.
        Args:
            observation: str
            controller_api_wrapper: controllerAPIWrapper
            residual_name: if not None, only parse this residual
            positive_only: only consider residuals that mark essential
            include_negative_residuals: include invented negative residuals
        """

        try:
            _predicate_scope = self.convert_predicates_to_scope(constraints, print_function=False)
            self.detect_predicates(_predicate_scope)

        except Exception as e:
            error_message = str(e)
            error_message = error_message.replace("[", "").replace("]", "")
            error_message = (
                f"Error occurred while parsing predicates:  "
                + error_message
            )
            raise AssertionError(error_message)

        return

    def detect_predicates(
        self, predicate_scope = None , provide_text: bool = False, max_envs: Optional[int] = None
    ):
        # PERFORMANCE: compiling predicate scope is expensive (exec + codegen).
        # We compile once in apply_constraints() and reuse the cached scope here.
        text = []

        if predicate_scope is None:
            predicate_scope = self._predicate_scope

            if not self._predicate_scope or "predicate_function" not in self._predicate_scope:
                # FAIL-FAST: predicate evaluation requires apply_constraints() to have been called.
                raise RuntimeError(
                    "Predicate scope is not initialized. "
                    "Call SUNConstraintGenerator.apply_constraints() before detect_predicates()."
                )
        res = predicate_scope["predicate_function"]()

        if provide_text:
            term_names = predicate_scope.get("predicate_term_functions", [])
            if isinstance(term_names, list) and len(term_names) > 0:
                text.append("Per-term predicate evaluation:")
                for idx, fn_name in enumerate(term_names):
                    fn = predicate_scope.get(fn_name)
                    if fn is None:
                        text.append(f"  - term {idx}: missing function {fn_name!r}")
                        continue
                    try:
                        term_res = fn()
                    except Exception as e:
                        text.append(f"  - term {idx} ({fn_name}): ERROR: {e}")
                        continue
                    if isinstance(term_res, torch.Tensor) and max_envs is not None:
                        flat = term_res.view(-1)
                        shown = flat[: max(0, int(max_envs))]
                        text.append(f"  - term {idx}: fn={fn_name} -> {shown}")
                    else:
                        text.append(f"  - term {idx}: fn={fn_name} -> {term_res}")

        if provide_text:
            return res, "\n".join(text)
        return res

    def detect_replans(self):
        res = None
        global_scope = self.convert_predicates_to_scope(self._active_predicate_lits)
        for lit in self._active_replan_lits:
            if hasattr(lit, "variables"):
                objectargs = []
                for s in lit.variables():
                    if s.isdigit():
                        objectargs.append(int(s))
                    elif (
                        s.replace(".", "").replace("-", "").isdigit()
                        or "e-" in s
                        or "e+" in s
                    ):
                        objectargs.append(float(s))
                    else:
                        objectargs.append(s)

                current_res = global_scope[lit.name()](*objectargs)
            else:
                current_res = eval(lit, global_scope)

            if res is None:
                res = current_res
                while len(res.shape) > 1:
                    res = res.squeeze(-1)
            else:
                _res = current_res
                while len(_res.shape) > 1:
                    _res = _res.squeeze(-1)
                res = torch.logical_and(res, _res)
        return res


    def apply_constraints(self, print_function=False):
        """
        Apply both residuals and predicates to the controller.
        """
        self._active_constraints = deepcopy(self._constraints)
        self._predicate_scope = self.convert_predicates_to_scope(self._active_constraints, print_function=print_function)
        self._residual_scope = self.convert_residuals_to_scope(self._active_constraints, print_function=print_function)
        self.program_runtime.active_controller.update_cost_function(self._residual_scope)

    @property
    def residuals_functions(self) -> str:
        return self._residuals.get_all_residual_func_text()

    @property
    def predicate_functions(self) -> str:
        return self._predicates.get_all_predicate_func_text()

    @property
    def all_functions(self) -> str:
        return self._residuals.get_all_func_text()

    @property
    def program_residuals(self) -> SUNProgramResiduals:
        return self._residuals

    @program_residuals.setter
    def program_residuals(self, program_residuals: SUNProgramResiduals) -> None:
        self._residuals = program_residuals

    @property
    def context_notes(self) -> List[str]:
        return self._context_notes

    @context_notes.setter
    def context_notes(self, context_notes: List[str]) -> None:
        self._context_notes = context_notes
