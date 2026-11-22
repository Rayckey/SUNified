import logging
from pathlib import Path
from typing import Any, Dict, List
from copy import deepcopy
import json
import numpy as np
import torch
import abc
import base64
import networkx as nx
from PIL import Image
from io import BytesIO
from kuafu.programs.llm.prompt_manager import PromptManager
from kuafu.common.prompt_context import content_for_model, text_for_model
from kuafu.programs.llm.program_residuals import SUNProgramResiduals
from kuafu.programs.llm.program_predicates import SUNProgramPredicates
from kuafu.programs.llm.sun_program_plan import SUNProgramPlan
from kuafu.programs.llm.program_variables import SUNProgramVariables
from kuafu.programs.llm.structs import Literal, Residual
import casadi as ca
from typing import Any, List, Dict, Tuple, Union, Set, FrozenSet
from kuafu.programs.llm.utils import (
    create_literal_from_raw,
    parse_util_functions_from_text,
    parse_llm_functions_from_text,
    load_text,
    write_txt,
    literals_to_text,
    compile_query_vector,
    make_unpacker_source,
    freeze_query_variables,
    literal_to_text,
    compile_observation,
    literals_to_dict,
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


class LLMBase(abc.ABC):
    def __init__(self, model_version: str, *args, **kwargs):

        self.prompt_manager = PromptManager(
            model_version=model_version, use_buffer=None, temperature=0.1
        )

        self.num_llm_calls = 0

        self.viewpoints = []

    def get_embedded_image(self, content: str, image: np.array = None):

        if isinstance(image, np.ndarray):
            img = Image.fromarray(np.uint8(image[:, :, ::-1]))
        elif isinstance(image, Image.Image):
            img = image
        else:
            return content, None

        im_file = BytesIO()
        img.save(im_file, format="JPEG")
        im_bytes = im_file.getvalue()  # im_bytes: image in binary format.
        coded_image = base64.b64encode(im_bytes).decode("utf-8")
        image_content = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{coded_image}",
            },
        }

        return content, image_content

    def prompt_llm(
        self, context: str, prompt: str, additional: list = [], force_format: str = None
    ):
        context = text_for_model(context)
        prompt = text_for_model(prompt)
        additional = content_for_model(additional)

        # clear all backlog
        self.prompt_manager.dropAllConversation()

        # reset system message
        self.prompt_manager.setSystemMessage(context)

        # feed prompt to llm
        logger.debug("\n" + "#" * 50)
        logger.debug(f"Prompt:\n{prompt}")
        messages = [{"type": "text", "text": prompt}] + additional

        try:
            llm_output = self.prompt_manager.callGPT(messages, force_format)
        except Exception as e:
            print(f"LLM call failed: {e}")
            raise Exception("Fail to connect to the LLM")

        logger.debug("\n" + "#" * 50)
        logger.debug(f"LLM output:\n{llm_output}")

        return llm_output

    def prompt_llm_multi_shot(
        self, context: str, prompts: list[str], force_format: str = None
    ):
        context = text_for_model(context)
        prompts = content_for_model(prompts)

        # clear all backlog
        self.prompt_manager.dropAllConversation()

        # reset system message
        self.prompt_manager.setSystemMessage(context)

        # feed prompt to llm
        messages = prompts

        try:
            llm_output = self.prompt_manager.callGPTMultiShot(messages, force_format)
        except Exception as e:
            print(f"LLM call failed: {e}")
            raise Exception("Fail to connect to the LLM")

        logger.debug("\n" + "#" * 50)
        logger.debug(f"LLM output:\n{llm_output}")

        return llm_output


class SUNResidualGenerator(LLMBase):
    """
    Invent, ground and correct residuals from human inputs & demonstrations.
    """

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

        # controller api wrapper
        self.controller_api_wrapper = controller_api_wrapper
        self.program_runtime = program_runtime

        self._tasks = SUNProgramPlan()
        self._residuals = SUNProgramResiduals()
        self._predicates = SUNProgramPredicates()

        self._active_residuals = SUNProgramResiduals()
        self._active_predicates = SUNProgramPredicates()


        self._goal_text = goal_text

        self.exceptions = {"state"}
        self.function_exceptions = set(
            self.controller_api_wrapper.get_all_residual_apis_description().keys()
        )

        # primary memory: learned residuals
        self._residuals = SUNProgramResiduals()
        self._predicates = SUNProgramPredicates()

        # current goal residuals
        self._current_residual_lits = {}
        self._active_residual_lits = {}
        self._current_predicate_lits = []
        self._active_predicate_lits = []
        self._current_raw_lits = []


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
        self._active_residuals.add_context_code(context)
        self._active_residuals.add_util_functions(util_functions)

        self._residuals.add_residuals(parsed_residuals, new_functions)
        self._active_residuals.add_residuals(parsed_residuals, new_functions)

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
        self._active_predicates.add_context_code(context)
        self._active_predicates.add_util_functions(util_functions)
        self._active_predicates.add_predicates(parsed_predicates, new_functions)

    def load_program(self, load_dir_residual: Path, load_dir_predicate: Path):
        """
        Load agent program.
        """

        self.load_initial_residual(load_dir_residual)
        self.load_initial_predicate(load_dir_predicate)

    def save_program(
        self,
        save_dir_task: Path,
        save_dir_residual: Path,
        save_dir_predicate: Path,
        save_dir_literal: Path,
    ):
        """
        Save current agent program.
        """
        # save residuals in seperate python files (function names overlap)
        all_residual_text = self._residuals.get_all_func_text()
        all_predicate_text = self._predicates.get_all_func_text()
        write_txt(save_dir_residual, all_residual_text)
        write_txt(save_dir_predicate, all_predicate_text)
        write_txt(save_dir_task, self._tasks.current_task)
        literals = {}
        literals["predicate"] = self._active_raw_predicate_lits
        literals["residual"] = self._active_raw_lits
        literals["variables"] = self._residuals._variables.get_variables_text().split(
            "\n"
        )
        json.dump(literals, open(save_dir_literal, "w"), indent=4)

    def get_residuals_raw(self) -> List[str]:
        return self._residuals.get_available_residuals_raw()

    def get_residuals_text(self) -> str:
        return self._residuals.get_available_residuals_text()

    def get_active_residuals_raw(self) -> List[str]:
        return self._active_residuals.get_available_residuals_raw()

    def get_active_residuals_text(self) -> str:
        return self._active_residuals.get_available_residuals_text()

    def learn_residuals_from_description(self):
        raise NotImplementedError("Subclasses should implement this method.")

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

    def learn_tasks_from_description(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def learn_predicates_from_description(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def convert_residuals_to_scope(
        self, residuals: SUNProgramResiduals, residual_lits: Dict
    ) -> str:
        """
        Convert residuals to python script.
        """

        # prepare global scope
        global_scope = {"np": np, "ca": ca}
        global_scope.update(_TYPING_EXPORTS)
        if hasattr(self.controller_api_wrapper.controller, "ee_frame"):
            global_scope.update(
                {"state_frame": self.controller_api_wrapper.controller.ee_frame}
            )
        global_scope["Sym"] = Union[ca.MX, ca.SX, ca.DM]
        global_scope.update(self.controller_api_wrapper.get_all_residual_apis())

        # gather grounding functions & execute
        util_funcs = residuals.get_all_util_func_text()

        # execute util functions and residual functions (local)
        exec(util_funcs, global_scope)
        residual_funcs = residuals.get_all_residual_func_text()
        exec(residual_funcs, global_scope)

        # create residual function in string
        lines = ["def residual_function(state, control):"]

        # Add variables directly
        var_text = residuals._variables.get_variables_text()
        if var_text:
            for line in var_text.split('\n'):
                if line.strip():
                    lines.append(f"    {line}")

        # residual to consider
        residual_terms = ["ca.DM([0])"]
        for lit, val in residual_lits.items():
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

                residual_terms.append(f"{lit.name()}({', '.join(map(str, objectargs))}) * {val}")
            else:
                residual_terms.append(f"{lit} * {val}")

        lines.append(f"    residual = {' + '.join(residual_terms)}")
        lines.append("    return residual")
        residual_function_str = "\n".join(lines)
        logger.debug("Compiled residual_function:\n%s", residual_function_str)
        exec(residual_function_str, global_scope)
        return global_scope

    def convert_predicates_to_scope(
        self, predicates: SUNProgramPredicates, print_function=False
    ):
        """
        Convert predicates to python scope.
        """

        # prepare global scope
        global_scope = {"np": np, "torch": torch, "DEVICE": self.device}
        global_scope.update(_TYPING_EXPORTS)

        global_scope.update(self.controller_api_wrapper.get_all_predicate_apis())

        # gather grounding functions & execute
        util_funcs = predicates.get_all_util_func_text()

        # execute util functions and residual functions (local)
        exec(util_funcs, global_scope)
        predicate_funcs = predicates.get_all_predicate_func_text()
        exec(predicate_funcs, global_scope)
        if print_function:
            logger.debug("Compiled predicate functions:\n%s", predicate_funcs)
        return global_scope

    def parse_residuals(self, program_residuals: SUNProgramResiduals, literals: Dict):
        """
        Parse literals with residuals.
        Args:
            observation: str
            controller_api_wrapper: controllerAPIWrapper
            residual_name: if not None, only parse this residual
            positive_only: only consider residuals that mark essential
            include_negative_residuals: include invented negative residuals
        """

        global_scope = self.convert_residuals_to_scope(program_residuals, literals)

        self.program_runtime.test_controller.update_cost_function(global_scope)

        self.program_runtime.test_controller.compute_control(
            np.random.random((self.program_runtime.test_controller.num_states)),
            np.random.random((self.program_runtime.test_controller.num_controls)),
        )

        return

    def parse_predicates(
        self, program_predicates: SUNProgramPredicates, literals: Dict[str, Literal]
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

        global_scope = self.convert_predicates_to_scope(
            program_predicates, print_function=True
        )

        try:
            for lit in literals:
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
                    logger.debug("Evaluating predicate %s%r", lit.name(), tuple(objectargs))
                    _ = global_scope[lit.name()](*objectargs)
                else:
                    logger.debug("Evaluating predicate %s", lit)
                    _ = eval(lit, global_scope)

                if len(_.shape) == 1:
                    _ = _.unsqueeze(0)
                _ = _.squeeze(-1) if _.shape[-1] == 1 and len(_.shape) > 2 else _
                if (len(_.shape) > 1 and _.shape[1] > 1) or len(_.shape) > 2:
                    name = lit.name() if hasattr(lit, "name") else lit
                    raise AssertionError(
                        f"Predicate {name} returned a {_.shape} tensor when it should have returned a (batch,1) or (batch,) tensor, please double check and input and output of the APIs, and squeeze as needed. "
                    )

        except Exception as e:
            error_message = str(e)
            error_message = error_message.replace("[", "").replace("]", "")
            name = lit.name() if hasattr(lit, "name") else lit
            error_message = (
                f"Error occurred while parsing predicates: {name}; "
                + error_message
            )
            raise AssertionError(error_message)

        return

    def detect_predicates(self, provide_text=False):
        res = None
        text = []
        global_scope = self.convert_predicates_to_scope(self._active_predicates)
        for lit in self._active_predicate_lits:
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

                # Execute literal object
                current_res = global_scope[lit.name()](*objectargs)
                lit_desc = f"Predicate {lit.name()} with arguments {objectargs}"
            else:
                # Execute string literal
                current_res = eval(lit, global_scope)
                lit_desc = f"Predicate {lit}"

            current_res = torch.reshape(current_res, (-1,))

            if res is None:
                res = current_res
                if provide_text:
                    text.append(f"{lit_desc} evaluated to {res}.")
            else:
                if provide_text:
                    text.append(f"{lit_desc} evaluated to {current_res}.")
                res = torch.logical_and(res, current_res)

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

    def apply_residuals(self):
        """
        Apply residuals to the controller.
        """
        # apply residuals to the controller
        self._active_residuals = deepcopy(self._residuals)
        self._active_residual_lits = deepcopy(self._current_residual_lits)
        self._active_raw_lits = deepcopy(self._current_raw_lits)

    def apply_predicates(self):
        """
        Apply predicates to the controller.
        """
        # apply predicates to the controller
        self._active_predicates = deepcopy(self._predicates)
        self._active_predicate_lits = deepcopy(self._current_predicate_lits)
        if hasattr(self, "_current_replan_lits"):
            self._active_replan_lits = deepcopy(self._current_replan_lits)
        self._active_raw_predicate_lits = deepcopy(self._current_raw_predicate_lits)

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

    def _invent_residuals_from_description(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def _continue_residuals_from_description(self):
        raise NotImplementedError("Subclasses should implement this method.")

    # Stage 1: Pre-task (high-level task decomposition)
    def _invent_pre_tasks_from_description(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def _continue_pre_tasks_from_description(self):
        raise NotImplementedError("Subclasses should implement this method.")

    # Stage 2: Post-task (ground tasks into atomic actions)
    # Only needs invention - if it fails, just retry
    def _invent_post_tasks_from_description(self):
        raise NotImplementedError("Subclasses should implement this method.")

    # Stage 3: Predicate (define success conditions)
    def _invent_predicate_from_description(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def _continue_predicate_from_description(self):
        raise NotImplementedError("Subclasses should implement this method.")

    # Legacy methods for backward compatibility (deprecated)
    def _invent_tasks_from_description(self):
        """Deprecated: Use _invent_pre_tasks_from_description instead"""
        raise NotImplementedError("Use _invent_pre_tasks_from_description instead.")

    def _continue_tasks_from_description(self):
        """Deprecated: Use _continue_pre_tasks_from_description instead"""
        raise NotImplementedError("Use _continue_pre_tasks_from_description instead.")

    def recall_literals(self, residual_literals, predicate_literals, variables) -> None:
        """
        Recall predicates from given literals.
        """

        _current_predicate_lits = []
        self._current_raw_predicate_lits = predicate_literals
        # realtion goal lits (positive only)
        for raw_lit in predicate_literals:
            # store raw lit verbatim
            _current_predicate_lits.append(raw_lit)

        # save to current goal lit
        self._current_predicate_lits = _current_predicate_lits

        self._residuals.add_variables(
            variables, self.exceptions, self.function_exceptions
        )

        # save to current goal lit
        self._current_raw_lits = residual_literals

        self._residuals.reasoning = residual_literals

        _current_residual_lits = {}

        # realtion goal lits (positive only)
        for raw_lit, lit_val in self._current_raw_lits.items():
            # store raw lit verbatim
            _current_residual_lits[raw_lit] = lit_val

        self._current_residual_lits = _current_residual_lits

        return

    def _continue_predicate_from_description(self):
        raise NotImplementedError("Subclasses should implement this method.")
