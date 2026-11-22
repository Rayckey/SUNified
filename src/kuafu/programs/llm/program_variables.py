from typing import Any, List, Dict, Tuple, Union, Set, FrozenSet
from networkx import DiGraph
from kuafu.programs.llm.utils import (
    extract_function_call_variables,
    extract_ordered_variables,
    parse_code_lines,
    classify_variables,
)


class SUNProgramVariables:
    def __init__(self):
        # List[str] - stores variable definitions verbatim
        self._variables = []
        
        # Kept for compatibility but not used for logic anymore
        self._raw_explain_mapping = {}
        self._dependency_dict = {}
        self._graph = DiGraph()
        self._util_functions = {}
        self._context_code = ""

        self.constant_vars = set()
        self.predefined_vars = set()
        self.query_vars = set()
        self.ordered_rest = []

    def add_variables(
        self,
        variable_list: List[str],
        predefined_syms=None,
        query_funcs=None,
    ) -> None:
        # Just store the variables verbatim as received
        # We append new variables to the list
        for var in variable_list:
            if var not in self._variables:
                self._variables.append(var)

    def extract_function_call_variables(self, function_exceptions=None) -> List[str]:
        """
        Get the ordered code. Now just returns the variables as stored.
        """
        return self._variables

    def extract_ordered_variables(self, available_vars=None) -> List[str]:
        """
        Get the ordered code. Now just returns the variables as stored.
        """
        return self._variables

    def get_variables_text(self) -> str:
        return "\n".join(self._variables)
