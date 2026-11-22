import re
import copy
from typing import Any, List, Dict, Tuple, Union, Set, FrozenSet
from kuafu.programs.llm.structs import Literal, Residual
from kuafu.programs.llm.program_variables import SUNProgramVariables
from kuafu.programs.llm.utils import (
    create_residual_from_raw,
)


class SUNProgramResiduals:
    def __init__(self):
        # Dict[residual_raw, explanation]
        self._raw_explain_mapping = {}
        # Dict[residual_name, SUNProgramResidual]
        self._residuals = {}
        # Dict[residual_raw, residual_name]
        self._raw_name_mapping = {}
        # Dict[residual_name, List[residual_name]]
        self._dependency = {}
        # Set[residual_name]
        self._essential_residuals = set()

        self._variables = SUNProgramVariables()

        self._reasoning = ""

        # Dict[util_name, util_func]
        self._util_functions = {}
        self._util_descriptions = {}
        self._context_code = ""

    def add_variables(
        self, variables_list: List[str], predefined_syms=None, query_funcs=None
    ) -> None:
        self._variables.add_variables(variables_list, predefined_syms, query_funcs)

    def get_variables_text(self) -> str:
        return self._variables.get_variables_text()

    def add_residuals(
        self,
        residuals_dict: Dict[str, str] = {},
        residuals_func_dict: Dict[str, str] = {},
    ) -> None:
        for raw, explanation in residuals_dict.items():
            residual = SUNProgramResidual(raw, explanation)

            # if new residuals, create & add SUNProgramResidual
            if raw not in self._raw_name_mapping:
                self._residuals[residual.name] = residual
                self._raw_name_mapping[raw] = residual.name
                self._raw_explain_mapping[raw] = explanation

        # set grounding functions (allow overwrite)
        for raw, func in residuals_func_dict.items():
            assert raw in self._raw_name_mapping, f"residual {raw} not in residuals!"
            name = self._raw_name_mapping[raw]
            self._residuals[name].func = func

    def remove_residuals(self, residuals: List[str]) -> None:
        for residual in residuals:
            if residual in self._residuals:
                self._raw_name_mapping.pop(self._residuals[residual].raw)
                self._raw_explain_mapping.pop(self._residuals[residual].raw)
                self._residuals.pop(residual)

    def add_util_functions(self, util_functions: Dict[str, str]) -> None:
        # allow overwrite
        for name, func in util_functions.items():
            self._util_functions[name] = func
            self._util_descriptions[name] = re.search(r'<<(.*?)>>', func, flags=re.DOTALL).group(1)

    def add_context_code(self, context: str) -> None:
        # allow overwrite
        self._context_code = context

    def check_residuals_grounding_func(self) -> bool:
        for residual in self._residuals.values():
            if not residual.has_grounding:
                return False
        return True

    def check_residuals_cost_func(self) -> bool:
        for residual in self._residuals.values():
            if not residual.has_cost and residual.name != "a_is_on_b":
                return False
        return True

    def get_dependency(self, residual_name: str) -> List[str]:
        return self._dependency.get(residual_name, [])

    def get_available_residuals_text(self) -> str:
        residuals_text_dict = {}
        for raw, pred_name in self._raw_name_mapping.items():
            # add positive residual
            residuals_text_dict[raw] = self._raw_explain_mapping[raw]
        return str(residuals_text_dict)

    def get_available_residuals_raw(
        self
    ) -> List[str]:
        residuals_raw = []
        for raw, pred_name in self._raw_name_mapping.items():

            # add positive residual
            residuals_raw.append(raw)


        return residuals_raw

    def get_residuals_to_ground_text(self) -> str:
        residuals_to_ground = {}
        for raw, explain in self._raw_explain_mapping.items():
            name = self._raw_name_mapping[raw]
            if not self._residuals[name].has_grounding:
                residuals_to_ground[raw] = explain
        return str(residuals_to_ground)

    def get_residual_func_text(
        self, residual_list: List[str], include_relevant_func: bool = False
    ) -> str:
        func_list = [
            self._residuals[residual_name].func for residual_name in residual_list
        ]
        if include_relevant_func:
            # iteratively add dependent functions
            pred_list = copy.deepcopy(residual_list)
            checked_pred = set()
            while len(pred_list) > 0:
                pred_to_check = pred_list.pop(0)
                checked_pred.add(pred_to_check)
                if pred_to_check in self._dependency:
                    for dep_name in self._dependency[pred_to_check]:
                        if dep_name not in checked_pred:
                            pred_list.append(dep_name)
                            func_list.append(self._residuals[dep_name].func)

        return "# residuals:\n" + "\n\n".join(func_list)

    def get_all_residual_func_text(self) -> str:
        func_list = []
        for residual in self._residuals.values():
            if residual.has_grounding:
                func_list.append(residual.func)
        return "# residuals:\n" + "\n\n".join(func_list)

    def get_all_util_func_text(self) -> str:
        return "# Utility functions:\n" + "\n\n".join(self._util_functions.values())

    def get_all_func_text(self) -> str:
        return (
            self._context_code
            + "\n\n"
            + self.get_all_util_func_text()
            + "\n\n"
            + self.get_all_residual_func_text()
        )

    def get_all_func_text_without_context(self) -> str:
        return (
            self.get_all_util_func_text()
            + "\n\n"
            + self.get_all_residual_func_text()
        )

    def get_all_residual_util_func_text(self) -> str:
        return (
            self._context_code
            + "\n\n"
            + self.get_all_util_func_text()
            + "\n\n"
            + self.get_all_residual_func_text()
        )


    def get_all_relevant_func_text(self, residual_list: List[str]) -> str:
        return (
            self._context_code
            + "\n\n"
            + self.get_all_util_func_text()
            + "\n\n"
            + self.get_residual_func_text(residual_list, include_relevant_func=True)
        )

    def get_all_grounded_residual_names(self) -> List[str]:
        return [
            residual.name
            for residual in self._residuals.values()
            if residual.has_grounding
        ]

    def get_residual_types(self) -> Dict[str, Residual]:
        all_residuals = {}
        for residual in self._residuals.values():
            all_residuals[residual.name] = residual.residual
        return all_residuals

    def find_matched_grounded_literal(self, pddl_literal: Literal) -> Tuple[str, bool]:
        for program_residuals in self._residuals.values():
            if pddl_literal.residual.name == program_residuals.residual.name:
                return (residual_literal_to_text(pddl_literal), True)
        return (str(pddl_literal), False)

    @property
    def reasoning(self) -> str:
        return self._reasoning
    
    @reasoning.setter
    def reasoning(self, reasoning: str) -> None:
        self._reasoning = reasoning

class SUNProgramResidual:
    def __init__(self, raw: str, explanation: str, func: str = None, cost: str = None):
        # on(x, y)
        self._raw = raw
        # x is on y
        self._explanation = explanation
        self._func = func
        self._cost = cost

        self._residual = create_residual_from_raw(raw)
        # on
        self._name = self._residual.name

    @property
    def has_grounding(self) -> bool:
        return self._func is not None

    @property
    def has_cost(self) -> bool:
        return self._cost is not None

    @property
    def func(self) -> str:
        assert self._func is not None, "The grounding function hasn't been set"
        return self._func

    @func.setter
    def func(self, func: str) -> None:
        assert func is not None or len(func) == 0, f"Invalid grounding function!"
        self._func = func

    @property
    def raw(self) -> str:
        return self._raw

    @property
    def explanation(self) -> str:
        return self._explanation

    @property
    def residual(self) -> Residual:
        return self._residual

    @property
    def name(self) -> str:
        return self._name
