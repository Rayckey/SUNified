from __future__ import annotations
from typing import Any, Dict, List, Mapping, Optional

from kuafu.programs.compiler.constraint_validation import (
    find_directional_rotate_axis_align_conflicts,
    find_manipulator_effort_condition_conflicts,
)


class ResidualModuleGenerator:
    """
    Compile a flat Constraint IR into a CasADi residual function that calls
    primitive residuals already defined in a provided scope.

    IR structure (per scope / subtask):

        {
          "Reasoning": "free-form explanation string",
          "Constraints": [
            {
              "logic": "and" | "or",   # IGNORED for residuals
              "condition": "geometry | gripper_state | ...",  # IGNORED
              "op": "free_move | directional_move | axis_align | "
                    "directional_rotate | freeze_position | freeze_orientation | "
                    "open_gripper | close_gripper | effort_reached",
              "args": { ... }  # kwargs for the primitive residual
            },
            ...
          ]
        }

    Each term `op` is mapped to a primitive residual function name, e.g.:

        "free_move"          -> "free_move_residual(state, ...)"
        "directional_move"   -> "directional_move_residual(state, ...)"
        "axis_align"         -> "axis_align_residual(state, ...)"
        "directional_rotate" -> "directional_rotate_residual(state, ...)"
        "effort_reached"     -> "effort_reached_residual(state, ...)"

    Generated function:

        def constraint_<scope_suffix>_residual(state: Sym) -> Sym:
            # Sum of all primitive residuals over IR["Constraints"]
            ...

    Assumptions
    -----------
    * You pass a `scope` dict to `compile_scope` that already contains:
        - Sym  (CasADi union alias)  [optional but needed for type hints]
        - primitive residuals: free_move_residual, axis_align_residual, ...
        - any other utilities used by those residuals.

    * This class does NOT define primitive residuals; it only builds the
      scope-level constraint residual that calls them.
    """

    def __init__(
        self,
        op_to_residual_func: Optional[Mapping[str, str]] = None,
        op_to_category: Optional[Mapping[str, str]] = None,
    ) -> None:
        # Map op name → primitive residual function name
        self.op_to_residual_func: Dict[str, str] = (
            dict(op_to_residual_func)
            if op_to_residual_func is not None
            else {
                # reach_point is the new name; keep free_move as a back-compat alias.
                "reach_point": "reach_point_residual",
                "reach_distance": "reach_distance_residual",
                "directional_move": "directional_move_residual",
                "axis_align": "axis_align_residual",
                "directional_rotate": "directional_rotate_residual",
                "effort_reached": "effort_reached_residual",
                "freeze_position": "freeze_position_residual",
                "freeze_orientation": "freeze_orientation_residual",
                "open_gripper": "open_gripper_residual",
                "close_gripper": "close_gripper_residual",
                "object_grasped": "close_gripper_residual",
                "point_toward": "point_toward_residual",
            }
        )

        # Map op name → category bucket used for normalization.
        #
        # Categories:
        # - "position"
        # - "orientation"  (averaged within the scope: divided by number of orientation terms)
        # - "gripper"
        # - "other" (not normalized / not counted; e.g. "effort_reached")
        self.op_to_category: Dict[str, str] = (
            dict(op_to_category)
            if op_to_category is not None
            else {
                "reach_point": "position",
                "reach_distance": "position",
                "directional_move": "position",
                "freeze_position": "position",
                "axis_align": "orientation",
                "directional_rotate": "orientation",
                "freeze_orientation": "orientation",
                "point_toward": "orientation",
                "open_gripper": "gripper",
                "close_gripper": "gripper",
                "object_grasped": "gripper",
                # explicitly ignored for counting/normalization purposes
                "effort_reached": "other",
            }
        )

    # -------------------------------------------------------------------------
    # 1. Compile a single IR (for one subtask) into a scope-level residual
    # -------------------------------------------------------------------------

    def compile_scope(
        self,
        scope: Dict[str, Any],
        scope_name: str,
        ir: Mapping[str, Any],
        print_function: bool = False,
    ) -> str:
        """
        Compile a flat-Constraint IR into a named scope-level residual function.

        Parameters
        ----------
        scope : dict
            Dict used as the global scope for the generated function. It must
            already contain primitive residuals and any other dependencies.
        scope_name : str
            A human-readable label for this subtask/scope
            (e.g. the original subtask description).
        ir : Mapping[str, Any]
            IR with keys "Reasoning" (optional) and "Constraints" (required).
        print_function : bool, default False
            If True, print the generated function source.

        Returns
        -------
        str
            The generated function name, which is also the key used in `scope`
            to retrieve the function object.
        """
        constraints = ir.get("Constraints")
        if constraints is None:
            raise ValueError("IR must contain a 'Constraints' key")
        if not isinstance(constraints, list) or len(constraints) == 0:
            raise ValueError("'Constraints' must be a non-empty list")

        reasoning = (ir.get("Reasoning") or "").strip()

        # Make a stable function name from the scope name
        func_name = scope_name

        lines: List[str] = []
        lines.append(
            f"def {func_name}(state: Sym, residual_weights: Optional[Dict[int, float]] = None, "
            "return_per_term: bool = False):"
        )
        lines.append('    """')
        lines.append(f"    <<Auto-generated residual for scope '{scope_name}'.")
        if reasoning:
            for line in reasoning.splitlines():
                lines.append(f"    Reasoning: {line}")
        lines.append("    Returns a weighted sum of primitive residuals over")
        lines.append("    IR['Constraints'] (logic fields are ignored).")
        lines.append("    If return_per_term=True, returns a tuple of weighted costs per constraint.")
        lines.append("    >>")
        lines.append('    """')
        lines.append("")

        # check for any violations of the constraints
        n_position, n_orientation = self.constraint_validation(constraints)

        # Use weights parameter, default to all 1.0 if not provided
        lines.append("    # Use provided weights, or default to all 1.0 if None")
        lines.append("    _weights = residual_weights if residual_weights is not None else {}")
        lines.append("")

        # PERF: deduplicate identical primitive calls within this generated scope function.
        call_cache: Dict[str, str] = {}

        # Accumulators by category.
        lines.append("    _pos_cost = 0")
        lines.append("    _ori_cost = 0")
        lines.append("    _grip_cost = 0")
        lines.append("    _other_cost = 0")
        lines.append("")

        # Terms: accumulate into category buckets with weights.
        for idx, term in enumerate(constraints):
            op = term.get("op")
            cat = self.op_to_category.get(str(op), "other")

            call = self._build_term_call(term)
            var = call_cache.get(call)
            if var is None:
                var = f"_term_{idx}"
                call_cache[call] = var
                lines.append(f"    {var} = {call}")

            # Get weight for this term (default 1.0 if not specified)
            lines.append(f"    # Term {idx}: op={op!r}, category={cat!r}")
            # Use dict.get to avoid KeyError when a weight is not provided.
            lines.append(f"    _weight_{idx} = _weights.get({idx}, 1.0)")
            lines.append(f"    _weighted_term_{idx} = {var} * _weight_{idx}")
            if cat == "orientation":
                lines.append(f"    _ori_cost = _ori_cost + _weighted_term_{idx}")
            elif cat == "position":
                lines.append(f"    _pos_cost = _pos_cost + _weighted_term_{idx}")
            elif cat == "gripper":
                lines.append(f"    _grip_cost = _grip_cost + _weighted_term_{idx}")
            else:
                lines.append(f"    _other_cost = _other_cost + _weighted_term_{idx}")
            lines.append("")

        # Optional: return per-term weighted costs (same order as Constraints).
        weighted_terms = ", ".join(f"_weighted_term_{i}" for i in range(len(constraints)))
        lines.append(f"    _weighted_terms = ({weighted_terms})")
        lines.append("    if return_per_term:")
        lines.append("        return _weighted_terms")
        lines.append("    return _pos_cost + _ori_cost + _grip_cost + _other_cost")
        lines.append("")

        func_src = "\n".join(lines)

        if print_function:
            print(func_src)

        # Inject the generated function into the given scope.
        exec(func_src, scope)

        return func_name

    def adjust_scope(
        self,
        scope: Dict[str, Any],
        scope_name: str,
        ir: Mapping[str, Any],
        print_function: bool = False,
    ) -> str:
        return self.compile_scope(scope, scope_name, ir, print_function)

    # -------------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------------

    def _build_term_call(self, term: Mapping[str, Any]) -> str:
        """
        Turn a single constraint entry into a primitive residual call.

        term ~
            {
              "logic": "and" | "or",       # IGNORED
              "condition": "...",          # IGNORED
              "op": "free_move | ...",
              "args": { ... }
            }

        We map 'op' → primitive residual name (e.g. 'free_move_residual') via
        self.op_to_residual_func, and then emit:

            free_move_residual(state, controlled_entity=..., controlled_target=..., ...)
        """
        op = term.get("op")
        if op is None:
            raise ValueError(f"Constraint term is missing 'op': {term}")

        if op not in self.op_to_residual_func:
            raise ValueError(f"Unknown op '{op}' in term: {term}")

        func_name = self.op_to_residual_func[op]

        args = term.get("args", {})
        if not isinstance(args, dict):
            raise ValueError(f"'args' must be a dict for term {term}")

        parts: List[str] = []
        # state is always the first positional argument
        parts.append("state")
        # term args as keyword arguments
        for key, val in args.items():
            parts.append(f"{key}={self._format_value(val)}")
        return f"{func_name}({', '.join(parts)})"

    @staticmethod
    def _sanitize_identifier(name: str) -> str:
        """
        Turn an arbitrary scope name into a safe Python identifier suffix.
        """
        import re

        base = name.strip().lower()
        base = re.sub(r"[^0-9a-zA-Z_]+", "_", base)
        base = re.sub(r"_+", "_", base)
        base = base.strip("_") or "unnamed"
        if base[0].isdigit():
            base = "_" + base
        return base

    @staticmethod
    def _format_value(value: Any) -> str:
        """
        Convert a Python IR value into a literal string.
        """
        if isinstance(value, str):
            escaped = value.replace("'", "\\'")
            return f"'{escaped}'"
        if isinstance(value, bool):
            return "True" if value else "False"
        if isinstance(value, (int, float)):
            return repr(value)
        if isinstance(value, (list, tuple)):
            inner = ", ".join(ResidualModuleGenerator._format_value(v) for v in value)
            return f"[{inner}]"
        if value is None:
            return "None"
        return repr(value)

    def constraint_validation(self, constraints: List[Mapping[str, Any]]) -> None:
        # ---------------------------------------------------------------------
        # Validate potential constraint conflicts (per-scope).
        #
        # Conflicts to catch:
        # - more than 1 position constraint
        # - more than 2 orientation constraints
        # - freeze_orientation mixed with ANY other orientation constraint
        # - more than 1 gripper constraint
        # ---------------------------------------------------------------------
        position_ops = {
            op for op, cat in self.op_to_category.items() if cat == "position"
        }
        orientation_ops = {
            op for op, cat in self.op_to_category.items() if cat == "orientation"
        }
        gripper_ops = {
            op for op, cat in self.op_to_category.items() if cat == "gripper"
        }

        errors: List[str] = []
        pos_hits: List[tuple[int, Any]] = []
        ori_hits: List[tuple[int, Any]] = []
        grip_hits: List[tuple[int, Any]] = []

        for idx, term in enumerate(constraints):
            if not isinstance(term, Mapping):
                errors.append(
                    f"Constraint term {idx} must be a mapping/dict, got {type(term)!r}: {term!r}"
                )
                continue
            op = term.get("op")
            if op is None:
                errors.append(
                    f"Constraint term {idx} is missing required key 'op': {term!r}"
                )
                continue
            if op in position_ops:
                pos_hits.append((idx, op))
            if op in orientation_ops:
                ori_hits.append((idx, op))
            if op in gripper_ops:
                grip_hits.append((idx, op))

        if len(pos_hits) > 1:
            errors.append(
                f"Conflicting position constraints (expected <= 1): {pos_hits}"
            )

        if len(ori_hits) > 3:
            errors.append(
                f"Too many orientation constraints (expected <= 2): {ori_hits}"
            )

        if ori_hits:
            has_freeze = any(op == "freeze_orientation" for _, op in ori_hits)
            has_other_orientation = any(
                op != "freeze_orientation" for _, op in ori_hits
            )
            if has_freeze and has_other_orientation:
                errors.append(
                    f"Mixes 'freeze_orientation' with other orientation constraints (not allowed): {ori_hits}"
                )

        if len(grip_hits) > 1:
            errors.append(
                f"Conflicting gripper constraints (expected <= 1): {grip_hits}"
            )

        errors.extend(find_directional_rotate_axis_align_conflicts(constraints))
        errors.extend(find_manipulator_effort_condition_conflicts(constraints))

        if errors:
            raise ValueError(
                "Constraint validation failed due to conflicts" f"\n- ".join(errors)
            )

        # Precompute (at compile-time) how many position/orientation terms exist in this scope.
        # 'effort_reached' does not count (category "other") and is excluded here.
        n_position = sum(1 for t in constraints if (t.get("op") in position_ops))
        n_orientation = sum(1 for t in constraints if (t.get("op") in orientation_ops))

        return n_position, n_orientation
