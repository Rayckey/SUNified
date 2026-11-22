import numpy as np
from scipy.spatial.transform import Rotation as R
import openai
import os
import sys
import itertools
import csv
import copy
import base64
import re
import logging
from typing import *
from kuafu.programs.llm.structs import (
    Potential,
    Residual,
    Literal,
    Predicate,
    DEFAULT_TYPE,
    VALUE_TYPE,
)
from kuafu.mpc.simulation.struct import SimulationEntity, VisibleEntity, FrameEntity, SensedEntity
import networkx as nx
import ast
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import casadi as ca
from collections import OrderedDict


def load_text(file_address):
    with open(file_address, "r", encoding="utf-8") as f:
        content = f.read()
    return content


def write_txt(file, content):
    with open(file, "wb") as f:
        f.write(content.encode("utf-8"))


def is_constant_expr(node):
    if isinstance(node, (ast.Constant, ast.Num)):
        return True
    elif isinstance(node, ast.UnaryOp):
        return is_constant_expr(node.operand)
    elif isinstance(node, ast.BinOp):
        return is_constant_expr(node.left) and is_constant_expr(node.right)
    elif isinstance(node, ast.Tuple):
        return all(is_constant_expr(elt) for elt in node.elts)
    return False


# ----------------------------------------
# Check if RHS uses one of the given functions
# ----------------------------------------
def rhs_uses_function(value_node, function_exceptions):
    class FuncCallFinder(ast.NodeVisitor):
        def __init__(self):
            self.hit = False

        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id in function_exceptions:
                self.hit = True
            elif (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in function_exceptions
            ):
                self.hit = True
            self.generic_visit(node)

    visitor = FuncCallFinder()
    visitor.visit(value_node)
    return visitor.hit


# ----------------------------
# AST Visitor for Dependencies
# ----------------------------
class VariableOnlyVisitor(ast.NodeVisitor):
    def __init__(self, defined_vars, exceptions=None):
        self.defined_vars = set(defined_vars)
        self.exceptions = set(exceptions or [])
        self.deps = set()

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            if node.id in self.exceptions:
                return
            if node.id in self.defined_vars:
                self.deps.add(node.id)

    def visit_Attribute(self, node):
        self.visit(node.value)

    def visit_Call(self, node):
        for arg in node.args:
            self.visit(arg)
        for kw in node.keywords:
            self.visit(kw.value)


# ----------------------------
# Parse Code Line
# ----------------------------
def get_variable_and_dependencies(
    code_line, defined_vars, exceptions=None, function_exceptions=None
):
    tree = ast.parse(code_line)
    stmt = tree.body[0]
    if not isinstance(stmt, ast.Assign):
        return None, None

    target = stmt.targets[0]
    if not isinstance(target, ast.Name):
        return None, None

    var_name = target.id
    value_node = stmt.value

    if is_constant_expr(value_node):
        return var_name, set()

    if rhs_uses_function(value_node, set(function_exceptions or [])):
        return var_name, set()

    visitor = VariableOnlyVisitor(defined_vars, exceptions)
    visitor.visit(value_node)
    return var_name, visitor.deps


# ----------------------------------------
# Find variables that are constants or function_exceptions
# ----------------------------------------
def extract_function_call_variables(var_to_code, function_exceptions):
    """
    Given var_to_code and a set of function names,
    return variables whose RHS calls any of those functions.
    """

    result = set()
    function_exceptions = set(function_exceptions)

    for var, code in var_to_code.items():
        try:
            rhs = ast.parse(code).body[0].value  # only parse the RHS once

            class FuncFinder(ast.NodeVisitor):
                def __init__(self):
                    self.hit = False

                def visit_Call(self, node):
                    if (
                        isinstance(node.func, ast.Name)
                        and node.func.id in function_exceptions
                    ):
                        self.hit = True
                    elif (
                        isinstance(node.func, ast.Attribute)
                        and node.func.attr in function_exceptions
                    ):
                        self.hit = True
                    self.generic_visit(node)

            finder = FuncFinder()
            finder.visit(rhs)
            if finder.hit:
                result.add(var)

        except Exception as e:
            print(f"Skipping (parse error): {code}\n{e}")

    return result


def extract_ordered_variables(var_to_code, var_to_deps, exclude_vars):
    """
    Return variables not in `exclude_vars`, sorted in topological order.
    """
    remaining_vars = set(var_to_code) - set(exclude_vars)

    # Compute in-degree
    in_degree = defaultdict(int)
    for var in remaining_vars:
        for dep in var_to_deps.get(var, set()):
            if dep in remaining_vars:
                in_degree[var] += 1

    queue = deque([v for v in remaining_vars if in_degree[v] == 0])
    ordered = []

    while queue:
        v = queue.popleft()
        ordered.append(v)
        for u in remaining_vars:
            if v in var_to_deps.get(u, set()):
                in_degree[u] -= 1
                if in_degree[u] == 0:
                    queue.append(u)

    if len(ordered) != len(remaining_vars):
        raise ValueError("Cycle detected in dependency graph")

    return ordered


# ----------------------------------------
# Topological sorting of dependency graph
# ----------------------------------------
def topological_order(var_to_deps):
    in_degrees = defaultdict(int)
    dep_graph = defaultdict(set)
    for var, deps in var_to_deps.items():
        for dep in deps:
            dep_graph[dep].add(var)
            in_degrees[var] += 1
    queue = deque([v for v in var_to_deps if in_degrees[v] == 0])
    order = []
    while queue:
        var = queue.popleft()
        order.append(var)
        for neighbor in dep_graph[var]:
            in_degrees[neighbor] -= 1
            if in_degrees[neighbor] == 0:
                queue.append(neighbor)
    if len(order) != len(var_to_deps):
        raise ValueError("Cycle or unresolved variable detected")
    return order


def get_ordered_code(
    var_to_code,
    var_to_deps,
    exceptions=None,
    function_exceptions=None,
    exclude_vars=None,
):
    exclude_vars = set(exclude_vars or [])
    for v in exclude_vars:
        var_to_code.pop(v, None)
        var_to_deps.pop(v, None)
    order = topological_order(var_to_deps)
    return [var_to_code[v] for v in order]


# ----------------------------------------
# Draw dependency graph
# ----------------------------------------
def build_dependency_graph(
    code_lines, exceptions=None, function_exceptions=None, exclude_vars=None
):
    G = nx.DiGraph()
    defined_vars = set()
    for line in code_lines:
        var, deps = get_variable_and_dependencies(
            line, defined_vars, exceptions, function_exceptions
        )
        if var is None or (exclude_vars and var in exclude_vars):
            continue
        G.add_node(var)
        for dep in deps:
            G.add_edge(dep, var)
        defined_vars.add(var)
    return G


def draw_dependency_graph(G):
    pos = nx.spring_layout(G, seed=42)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="lightblue",
        node_size=2000,
        font_size=10,
        arrows=True,
    )
    plt.title("Variable Dependency Graph")
    plt.show()


def load_query(query_stage: str, file_name: str):
    content = load_text(os.path.join("templates", query_stage, file_name + ".txt"))

    if "###///" in content:
        token_position = content.find("###///")
        before_the_first_token = content[:token_position]
        cut_off_by_first_token = content[token_position + 6 :]
        image_path = cut_off_by_first_token[: cut_off_by_first_token.find("###///")]
        image_content = embed_image(image_path)

        content = [{"type": "text", "text": before_the_first_token}, image_content]
    else:
        content = [{"type": "text", "text": content}]
    return content


def embed_image(image_path):
    with open(image_path, "rb") as image_file:
        coded_image = base64.b64encode(image_file.read()).decode("utf-8")

        image_content = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{coded_image}",
            },
        }

        return image_content


def end_sentence(sentence=""):
    if len(sentence) == 0:
        return sentence
    return sentence[:-2] + ". "


def size2text(size):
    return f"that is {size[2]:.3f} m long and {size[0]:.3f} m wide; "


def create_residual_from_raw(raw_residual: str):
    # e.g. "on_table(x, y)"
    split = re.split(r"\(|\)|,", raw_residual)
    split = [x.strip(" \"'") for x in split if x != ""]

    residual_name = split[0]
    arity = len(split) - 1

    # assert arity > 1, "Residual arity must be at least 2."

    return Residual(residual_name, arity, var_types=[DEFAULT_TYPE] * arity)


def create_predicate_from_raw(raw_string: str):
    # e.g. "on_table(x, y)"
    split = re.split(r"\(|\)|,", raw_string)
    split = [x.strip(" \"'") for x in split if x != ""]

    predicate_name = split[0]
    arity = len(split) - 1

    # assert arity > 1, "Residual arity must be at least 2."

    return Predicate(predicate_name, arity, var_types=[DEFAULT_TYPE] * arity)


def split_call(call: str):
    """
    Turn a call string like
    "is_end_effector_above_object(state, control, 'cube_3')"
    into ["is_end_effector_above_object", "state", "control", "'cube_3'"].
    """
    # 1. Capture the part before the first '(' as the name,
    #    and everything inside the outermost parentheses as one group.
    match = re.fullmatch(r"\s*([A-Za-z_]\w*)\s*\((.*)\)\s*", call)
    if not match:
        raise ValueError("Input is not a single function call")

    name, arg_block = match.groups()

    # 2. Split arguments on commas that are not inside quotes or parentheses.
    #    For simple calls like yours, a basic split on ',' works after stripping.
    args = [arg.strip() for arg in re.split(r"\s*,\s*", arg_block) if arg.strip()]

    return [name] + args


def create_literal_from_raw(
    raw_lit: str,
    potentials: Dict[str, Potential],
    skip_unknown: bool = True,
):
    split = re.split(r"\(|\)|,", raw_lit)
    split = [x.strip(" \"'") for x in split if x != ""]
    residual_name = split[0]

    if skip_unknown and residual_name not in potentials:
        logging.warning(f"Unknown residual in literal {raw_lit}: {residual_name}")
        raise ValueError(f"Unknown residual in literal {raw_lit}: {residual_name}")

    assert (
        residual_name in potentials
    ), f"Unknown residual in literal {raw_lit}: {residual_name}"
    assert (
        len(split) - 1 == potentials[residual_name].arity
    ), f"Wrong number of arguments in literal {raw_lit}: {len(split) - 1} instead of {potentials[residual_name].arity}"

    residual = potentials[residual_name]
    typed_entities = []
    for var_name in split[1:]:
        if "=" in var_name:
            # I don't know why it keeps doing this
            var_stripped = re.split("=", var_name)[-1].strip()
        else:
            var_stripped = var_name.strip()

        typed_entities.append(DEFAULT_TYPE(var_stripped))

    positive_lit = residual(*typed_entities)

    return positive_lit


def get_function_name_from_raw(raw: str):
    # a raw literal or residual
    split = re.split(r"\(", raw)
    return split[0].strip()


def parse_util_functions_from_text(text: str):
    function_starts = [m.start() for m in re.finditer("# <utility>", text)]
    function_ends = [m.end() for m in re.finditer("# <end-of-utility>", text)]
    function_starts = sorted(function_starts)
    function_ends = sorted(function_ends)
    if len(function_starts) != len(function_ends):
        assert len(function_starts) == len(
            function_ends
        ), f"Number of function starts and ends don't match."

    # parse functions
    functions = {}  # Dict[function_name, function]
    for start, end in zip(function_starts, function_ends):
        func = text[start:end]
        # ['', '', '{function_name}', ...]
        split = re.split(r"# <utility>|def |\(", func)
        function_name = split[2].strip()
        # record
        functions[function_name] = func

    return functions


def parse_llm_functions_from_text(
    text: str, known_residual_names: List[str], keyword="residual"
):
    function_starts = [m.start() for m in re.finditer(f"# <{keyword}>", text)]
    function_ends = [m.end() for m in re.finditer(f"# <end-of-{keyword}>", text)]
    function_starts = sorted(function_starts)
    function_ends = sorted(function_ends)
    if len(function_starts) != len(function_ends):
        assert len(function_starts) == len(
            function_ends
        ), f"Number of function starts and ends don't match."

    # parse functions
    functions = {}  # Dict[residual_raw, function]
    parsed_residuals = {}  # Dict[residual_raw, explanation]
    raw_to_name_mapping = {}  # Dict[residual_raw, residual_name]
    for start, end in zip(function_starts, function_ends):
        func = text[start:end]
        # ['', '', '{residual}', '{args}', ...]
        split = re.split(rf"# <{keyword}>|def |\(|\)", func)
        residual_name = split[2].strip()

        # split args and remove ": str"
        arg_split = re.split(",", split[3].strip())
        arg_split = [x.split(":")[0].strip() for x in arg_split if x != ""]
        residual_raw = f"{residual_name}({', '.join(arg_split)})"

        # ['', '{explanation}', '', ...]
        split_2 = re.split("<<|>>", func)
        explanation = split_2[1].strip()

        # record
        functions[residual_raw] = func
        parsed_residuals[residual_raw] = explanation
        raw_to_name_mapping[residual_raw] = residual_name

    # calculate dependency
    dependency = {}  # Dict[residual_raw, List[residual_raw]]
    for residual_raw, func in functions.items():
        residual_name = raw_to_name_mapping[residual_raw]
        dependency[residual_name] = []
        # check dependency on known residuals
        for known_residual_name in known_residual_names:
            if known_residual_name == residual_name:
                continue
            if f"{known_residual_name}(" in func:
                dependency[residual_name].append(known_residual_name)
        # check dependency on other residuals
        for other_residual_raw, other_residual_name in raw_to_name_mapping.items():
            if other_residual_raw == residual_raw:
                continue
            if f"{other_residual_name}(" in func:
                dependency[residual_name].append(other_residual_name)

    return parsed_residuals, functions, dependency


def literals_to_text(literals: List[Literal]):
    lit_text = ""
    for lit in literals:
        lit_text += literal_to_text(lit)
        lit_text += ", "
    return lit_text[:-2]


def literal_to_text(literal: Literal):
    lit_text = (
        f"{literal.term.name}({', '.join([var.name for var in literal.variables])})"
    )
    return lit_text


def _symbols_used(expr_src: str) -> Set[str]:
    """Return the set of *variable* names appearing in an expression.

    • Attribute chains such as ``ca.vertcat`` are reduced to the
      right-most identifier (``vertcat``) so we can match function names.
    """

    class SymVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.syms: Set[str] = set()

        # plain identifiers
        def visit_Name(self, node: ast.Name) -> None:
            self.syms.add(node.id)

        # dotted identifiers (a.b.c) — keep the final part (c)
        def visit_Attribute(self, node: ast.Attribute) -> None:
            self.syms.add(node.attr)
            self.visit(node.value)  # continue walking down the chain

        # subscript like state[0] — we only care about the base symbol
        def visit_Subscript(self, node: ast.Subscript) -> None:
            self.visit(node.value)  # recurse into the value (e.g., state)

    tree = ast.parse(expr_src, mode="eval")
    v = SymVisitor()
    v.visit(tree)
    return v.syms


def parse_code_lines(
    lines: List[str],
) -> Tuple[Dict[str, str], Dict[str, Set[str]], nx.DiGraph]:
    """Return dictionaries and a DiGraph capturing inter-variable dependencies.

    Parameters
    ----------
    lines : list[str]
        Each element looks like ``"x = ca.f(y, z)"``.

    Returns
    -------
    var_to_code : dict[var, rhs_source]
    var_to_deps : dict[var, set[str]]
    dep_graph   : nx.DiGraph    # edges: dep → var
    """
    var_to_code: Dict[str, str] = {}
    var_to_deps: Dict[str, Set[str]] = {}
    g = nx.DiGraph()

    assign_re = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*(.+)$")

    for ln in lines:
        if len(ln) == 0:
            continue
        if ln[0] == "#" or ln[1] == "#":
            continue
        m = assign_re.match(ln)
        if not m:
            raise ValueError(f"Line not recognised as assignment: {ln}")
        lhs, rhs = m.group(1), m.group(2)
        var_to_code[lhs] = rhs
        deps = _symbols_used(rhs) - {lhs}  # remove self reference
        var_to_deps[lhs] = deps
        for d in deps:
            g.add_edge(d, lhs)

    # the graph should at least contain all variables as nodes
    g.add_nodes_from(var_to_code)
    return var_to_code, var_to_deps, g


def extract_constant_variables(
    var_to_code: Dict[str, str], exception_funcs: Set[str] | None = None
) -> Set[str]:
    """Variables whose RHS consists solely of numeric literals or arrays.

    A RHS that is *only* a literal/tuple/list of numbers, or the same wrapped
    by a function in `exception_funcs` (e.g. ca.DM, ca.vertcat), counts as
    constant.
    """
    if exception_funcs is None:
        exception_funcs = {"ca.DM", "ca.vertcat"}

    const_vars: Set[str] = set()
    numeric_literal_re = re.compile(r"^[\s\[\]\(\)\d\.,eE+\-]*$")

    def _is_const(rhs: str) -> bool:
        rhs = rhs.strip()
        # 1. Pure numeric array / scalar
        if numeric_literal_re.fullmatch(rhs):
            return True
        # 2. Wrapped by an allowed “exception” function
        for f in exception_funcs:
            if rhs.startswith(f + "(") and rhs.endswith(")"):
                inner = rhs[len(f) + 1 : -1]
                return numeric_literal_re.fullmatch(inner) is not None
        return False

    for v, rhs in var_to_code.items():
        if _is_const(rhs):
            const_vars.add(v)
    return const_vars


def extract_predefined_variables(
    var_to_deps: Dict[str, Set[str]], predefined_syms: Set[str]
) -> Set[str]:
    """Variables whose RHS uses at least one symbol in `predefined_syms`."""
    return {v for v, deps in var_to_deps.items() if deps & predefined_syms}


def extract_query_variables(
    var_to_code: Dict[str, str], query_funcs: Set[str]
) -> Set[str]:
    """Variables whose RHS **calls** (not merely mentions) a function in `query_funcs`."""
    call_re = re.compile(r"\b({})\s*\(".format("|".join(map(re.escape, query_funcs))))
    return {v for v, rhs in var_to_code.items() if call_re.search(rhs)}


def extract_ordered_dependent_variables(
    dep_graph: nx.DiGraph, excluded: Set[str]
) -> List[str]:
    """All remaining variables, returned in dependency order (topological)."""
    remaining = [v for v in nx.topological_sort(dep_graph) if v not in excluded]
    return remaining


def classify_variables(
    var_to_code,
    var_to_deps,
    g,
    predefined_syms: Set[str],
    query_funcs: Set[str],
    exception_funcs: Set[str] | None = None,
) -> Tuple[Set[str], Set[str], Set[str], List[str], nx.DiGraph]:
    """High-level helper: parse, classify, order.

    Returns
    -------
    constant_vars, predefined_vars, query_vars, ordered_rest, dep_graph
    """

    constant_vars = extract_constant_variables(var_to_code, exception_funcs)
    predefined_vars = extract_predefined_variables(var_to_deps, predefined_syms)
    query_vars = extract_query_variables(var_to_code, query_funcs)

    excluded = constant_vars | predefined_vars | query_vars
    ordered_rest = extract_ordered_dependent_variables(g, excluded)
    return constant_vars, predefined_vars, query_vars, ordered_rest


def compile_query_vector(
    query_vars: List[str],
    var_to_code: Dict[str, str],
    exec_env: dict | None = None,
) -> Tuple[ca.MX, OrderedDict[str, Tuple[int, int]]]:
    """
    Evaluate each query variable, pack them into a single CasADi vector,
    and return (vector,  mapping).

    Parameters
    ----------
    query_vars : list[str]
        Variable names in the *desired* packing order (usually topo-order).
    var_to_code : dict[str, str]
        Original RHS source for every variable (from parse_code_lines).
    exec_env : dict | None
        Namespace in which to `exec` every assignment.  If None, a fresh
        namespace initialised with ``{"ca": ca}`` is created.

    Returns
    -------
    q_vec : ca.MX
        `vertcat`-ed vector of all query variables.
    index_map : OrderedDict[str, (int, int)]
        For each variable, the half-open slice *(start, stop)* inside q_vec.
        Example:  index_map["w"] == (7, 10)  ⇒  q_vec[7:10] == w.
    """
    if exec_env is None:
        exec_env = {"ca": ca}

    # 1. Execute code for every variable that will eventually be used.
    for v in query_vars:
        code = f"{v} = {var_to_code[v]}"
        exec(code, exec_env)

    # 2. Pack & record indices
    pieces, idx_map = [], OrderedDict()
    cursor = 0
    for v in query_vars:
        val = exec_env[v]
        # CasADi .numel() is fast & robust for scalars / matrices / DM
        length = int(val.numel())
        pieces.append(val.reshape((-1, 1)))
        idx_map[v] = (cursor, cursor + length)
        cursor += length

    q_vec = ca.vertcat(*pieces)
    return q_vec, idx_map


def make_unpacker_source(
    index_map: "OrderedDict[str, Tuple[int, int]]",
    vec_name: str = "q_vec",
    initial_offset: int = 0,
) -> str:
    """
    Create a block of Python code (string) that unpacks *vec_name*
    into the original query variables.

    >>> print(make_unpacker_source(idx_map))
    a = q_vec[0:3]
    b = q_vec[3]
    c = q_vec[4:7]

    Parameters
    ----------
    index_map : OrderedDict[str, (int, int)]
        Mapping produced by `compile_query_vector`.
    vec_name  : str
        Name of the packed vector in the generated code.

    Returns
    -------
    str
        Multiline string of assignment statements.
    """
    lines = []
    for var, (lo, hi) in index_map.items():
        slice_expr = (
            f"{vec_name}[{lo+initial_offset}:{hi+initial_offset}]"
            if hi - lo > 1
            else f"{vec_name}[{lo+initial_offset}]"
        )
        lines.append(f"{var} = {slice_expr}")
    return lines


def freeze_query_variables(
    query_vars: List[str],
    var_to_code: Dict[str, str],
    exec_env: dict | None = None,
) -> List[str]:
    """
    Execute every query variable, capture its numeric value, and return a
    block of Python assignments such that running the block recreates the
    same variables _without_ recomputation.

    Parameters
    ----------
    query_vars  : list[str]
        Variable names in the order you want them to appear.
    var_to_code : dict[str, str]
        RHS source for each variable.
    exec_env    : dict | None
        Namespace for execution.  If None, a fresh {'ca': ca} namespace
        is used.

    Returns
    -------
    str
        Multiline string like ::
            w = ca.DM(3.14)
            x = ca.DM([1.0, 2.0, 3.0])
            y = ca.DM([[4.0, 5.0],
                        [6.0, 7.0]])
    """
    if exec_env is None:
        exec_env = {"ca": ca}

    # 1. Evaluate each query variable exactly once
    for v in query_vars:
        exec(f"{v} = {var_to_code[v]}", exec_env)

    # 2. Build literal-value assignments
    lines: List[str] = []
    for v in query_vars:
        val: ca.MX = exec_env[v]
        data = val.full().tolist()  # nested list (works for scalars too)

        # pretty formatting: scalar → DM(<num>), vector/matrix → DM([...])
        if val.numel() == 1:  # 1×1
            lit = repr(float(val))
            line = f"{v} = ca.DM({lit})"
        else:
            line = f"{v} = ca.DM({repr(data)})"
        lines.append(line)

    return lines


def compile_observation(object_list: list, include_pose: bool = True, include_size: bool = True) -> str:
    res = ""
    for obj in object_list:
        if issubclass(type(obj), SimulationEntity) :
            res += f"{obj.name}: {obj.description}"
        else:
            res += f"{obj}"
        if include_pose:
            numericals = []
            x_str = np.array2string(  obj.transform_real_time.translation.cpu().numpy()  , formatter={'float_kind':lambda x: "%.2f" % x}   )
            numericals.append(f"translation = {x_str},")
            q_str = np.array2string(  obj.transform_real_time.quaternion.cpu().numpy()  , formatter={'float_kind':lambda x: "%.2f" % x}   )
            numericals.append(
                f"orientation = {q_str}"
            )
            res += ", "
            res += ", ".join(numericals)
        if include_size:
            numericals = []
            if obj.has_size:
                s_str = np.array2string(  obj.size.cpu().numpy().flatten()  , formatter={'float_kind':lambda x: "%.2f" % x}   )
                numericals.append(f"size = {s_str}")
            else:
                numericals.append(f"no size property, frame only")
            res += ", "
            res += ", ".join(numericals)
        else:
            if obj.has_size:
                res += f", has size property"
            else:
                res += f", no size property"
        res += "\n"
    return res


def literals_to_dict(literals: List[Literal]) -> Dict[str, List]:
    lit_dict = {}
    lit_list = []
    for lit in literals:
        objectargs = []
        for s in lit.variables():
            if s.isdigit():
                objectargs.append(int(s))
            elif (
                s.replace(".", "").replace("-", "").isdigit() or "e-" in s or "e+" in s
            ):
                objectargs.append(float(s))
            else:
                objectargs.append(s)
        lit_list.append((lit.name(), objectargs))
    lit_dict[lit.name()] = lit_list
    return lit_dict
