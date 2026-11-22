import re
import copy
from typing import Any, List, Dict, Tuple, Union, Set, FrozenSet
from kuafu.programs.llm.structs import Literal, Predicate
from kuafu.programs.llm.program_variables import SUNProgramVariables
from kuafu.programs.llm.utils import (
    create_literal_from_raw,
    parse_util_functions_from_text,
    parse_llm_functions_from_text,
    create_predicate_from_raw,
    load_query,
    load_text,
    literals_to_text,
    literal_to_text,
    get_function_name_from_raw,
)


class SUNProgramPredicates:
    def __init__(self):
        # Dict[predicate_raw, explanation]
        self._raw_explain_mapping = {}
        # Dict[predicate_name, SUNProgramPredicate]
        self._predicates = {}
        # Dict[predicate_raw, predicate_name]
        self._raw_name_mapping = {}
        # Dict[predicate_name, List[predicate_name]]
        self._dependency = {}
        # Set[predicate_name]
        self._essential_predicates = set()

        self._variables = SUNProgramVariables()

        self._reasoning = ""

        # Dict[util_name, util_func]
        self._util_functions = {}
        self._util_descriptions = {}
        self._context_code = ""

    def add_variables(
        self, variables_list: List[str], predefined_syms, query_funcs
    ) -> None:
        self._variables.add_variables(variables_list, predefined_syms, query_funcs)

    def get_variables_text(self) -> str:
        return self._variables.get_variables_text()

    def add_predicates(
        self,
        predicates_dict: Dict[str, str] = {},
        predicates_func_dict: Dict[str, str] = {},
    ) -> None:
        for raw, explanation in predicates_dict.items():
            predicate = SUNProgramPredicate(raw, explanation)

            # if new predicates, create & add SUNProgramPredicate
            if raw not in self._raw_name_mapping:
                self._predicates[predicate.name] = predicate
                self._raw_name_mapping[raw] = predicate.name
                self._raw_explain_mapping[raw] = explanation

        # set grounding functions (allow overwrite)
        for raw, func in predicates_func_dict.items():
            assert raw in self._raw_name_mapping, f"predicate {raw} not in predicates!"
            name = self._raw_name_mapping[raw]
            self._predicates[name].func = func

    def remove_predicates(self, predicates: List[str]) -> None:
        for predicate in predicates:
            if predicate in self._predicates:
                self._raw_name_mapping.pop(self._predicates[predicate].raw)
                self._raw_explain_mapping.pop(self._predicates[predicate].raw)
                self._predicates.pop(predicate)

    def clear(self) -> None:
        """Clear all predicates."""
        self.remove_predicates([str(s) for s in self._predicates.keys()])

    def add_util_functions(self, util_functions: Dict[str, str]) -> None:
        # allow overwrite
        for name, func in util_functions.items():
            self._util_functions[name] = func
            self._util_descriptions[name] = re.search(r'<<(.*?)>>', func, flags=re.DOTALL).group(1)

    def add_context_code(self, context: str) -> None:
        # allow overwrite
        self._context_code = context

    def check_predicates_grounding_func(self) -> bool:
        for predicate in self._predicates.values():
            if not predicate.has_grounding:
                return False
        return True

    def check_predicates_cost_func(self) -> bool:
        for predicate in self._predicates.values():
            if not predicate.has_cost and predicate.name != "a_is_on_b":
                return False
        return True

    def get_dependency(self, predicate_name: str) -> List[str]:
        return self._dependency[predicate_name]

    def get_available_predicates_text(self) -> str:
        predicates_text_dict = {}
        for raw, pred_name in self._raw_name_mapping.items():
            # add positive predicate
            predicates_text_dict[raw] = self._raw_explain_mapping[raw]
        return str(predicates_text_dict)

    def get_available_predicates_raw(self) -> List[str]:
        predicates_raw = []
        for raw, pred_name in self._raw_name_mapping.items():

            # add positive predicate
            predicates_raw.append(raw)

        return predicates_raw

    def get_predicates_to_ground_text(self) -> str:
        predicates_to_ground = {}
        for raw, explain in self._raw_explain_mapping.items():
            name = self._raw_name_mapping[raw]
            if not self._predicates[name].has_grounding:
                predicates_to_ground[raw] = explain
        return str(predicates_to_ground)

    def get_predicate_func_text(
        self, predicate_list: List[str], include_relevant_func: bool = False
    ) -> str:
        func_list = [
            self._predicates[predicate_name].func for predicate_name in predicate_list
        ]
        if include_relevant_func:
            # iteratively add dependent functions
            pred_list = copy.deepcopy(predicate_list)
            checked_pred = set()
            while len(pred_list) > 0:
                pred_to_check = pred_list.pop(0)
                checked_pred.add(pred_to_check)
                for dep_name in self._dependency[pred_to_check]:
                    if dep_name not in checked_pred:
                        pred_list.append(dep_name)
                        func_list.append(self._predicates[dep_name].func)

        return "# predicates:\n" + "\n\n".join(func_list)

    def get_all_predicate_func_text(self) -> str:
        func_list = []
        for predicate in self._predicates.values():
            if predicate.has_grounding:
                func_list.append(predicate.func)
        return "# predicates:\n" + "\n\n".join(func_list)

    def get_all_util_func_text(self) -> str:
        return "# Utility functions:\n" + "\n\n".join(self._util_functions.values())

    def get_all_func_text(self) -> str:
        return (
            self._context_code
            + "\n\n"
            + self.get_all_util_func_text()
            + "\n\n"
            + self.get_all_predicate_func_text()
        )

    def get_all_func_text_without_context(self) -> str:
        return (
            self.get_all_util_func_text() + "\n\n" + self.get_all_predicate_func_text()
        )

    def get_all_predicate_util_func_text(self) -> str:
        return (
            self._context_code
            + "\n\n"
            + self.get_all_util_func_text()
            + "\n\n"
            + self.get_all_predicate_func_text()
        )

    def get_all_relevant_func_text(self, predicate_list: List[str]) -> str:
        return (
            self._context_code
            + "\n\n"
            + self.get_all_util_func_text()
            + "\n\n"
            + self.get_predicate_func_text(predicate_list, include_relevant_func=True)
        )

    def get_all_grounded_predicate_names(self) -> List[str]:
        return [
            predicate.name
            for predicate in self._predicates.values()
            if predicate.has_grounding
        ]

    def get_predicate_types(self) -> Dict[str, Predicate]:
        all_predicates = {}
        for predicate in self._predicates.values():
            all_predicates[predicate.name] = predicate.predicate
        return all_predicates

    @property
    def reasoning(self) -> str:
        return self._reasoning

    @reasoning.setter
    def reasoning(self, reasoning: str) -> None:
        self._reasoning = reasoning

class SUNProgramPredicate:
    def __init__(self, raw: str, explanation: str, func: str = None):
        # on(x, y)
        self._raw = raw
        # x is on y
        self._explanation = explanation
        self._func = func

        self._predicate = create_predicate_from_raw(raw)
        # on
        self._name = self._predicate.name

    @property
    def has_grounding(self) -> bool:
        return self._func is not None

    @property
    def func(self) -> str:
        assert self._func is not None, "The grounding function hasn't been set"
        return self._func

    @func.setter
    def func(self, func: str) -> None:
        assert func is not None or len(func) == 0, f"Invalid grounding function!"
        # check arity
        in_parenth = re.split(r"\(|\)", func)[1]
        if len(in_parenth) == 0:
            parsed_arity = 0
        elif in_parenth.count(",") == 0:
            parsed_arity = 1
        else:
            parsed_arity = in_parenth.count(",") + 1
        assert (
            parsed_arity == self._predicate.arity
        ), f"Wrong arity in grounding function! {self._predicate} with grounding function arity {parsed_arity}"
        self._func = func

    @property
    def raw(self) -> str:
        return self._raw

    @property
    def explanation(self) -> str:
        return self._explanation

    @property
    def predicate(self) -> Predicate:
        return self._predicate

    @property
    def name(self) -> str:
        return self._name
