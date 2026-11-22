from __future__ import annotations
from typing import Any, Dict, List, Mapping, Optional

import re
import torch

from kuafu.programs.compiler.constraint_validation import (
    find_directional_rotate_axis_align_conflicts,
    find_manipulator_effort_condition_conflicts,
)


class PredicateModuleGenerator:
    """
    Compile flat Constraint IR into named scope-level predicate functions that call
    primitive predicates which already exist in a provided Python scope.

    IR structure (per scope / subtask):

        {
          "Reasoning": "free-form explanation string",
          "Constraints": [
            {
              "condition": "geometry"
                           | "gripper_opened" | "gripper_closed"
                           | "manipulator_high_effort" | "manipulator_low_effort",
              "op": "free_move | directional_move | axis_align | directional_rotate | "
                    "freeze_position | freeze_orientation | open_gripper | close_gripper | "
                    "manipulator_high_effort | manipulator_low_effort",
              "args": { ... }  # matches TermAPI[op].args
            },
            ...
          ]
        }

    Assumptions
    -----------
    * The caller provides a `scope` dict that already contains:
        - All primitive predicate callables referenced in `op_to_predicate_func`.
        - Any other globals they rely on (e.g. `torch`, API helpers, device, etc.).

    * op → primitive mapping:
        Each TermAPI `op` name is mapped to a primitive predicate function name via
        `self.op_to_predicate_func`. These primitive names must be present in
        the provided `scope` when the compiled function is executed.

    Main method
    -----------
        compile_scope(scope, scope_name, ir, print_function=False) -> List[str]
            - Generate one predicate function per constraint term (named as `<op><idx>`),
              inject them into `scope`, and return the list of function names.
            - Also generates the usual combined reducer function named by `scope_name`
              (for backwards compatibility).
    """

    def __init__(
        self,
        op_to_predicate_func: Optional[Mapping[str, str]] = None,
    ) -> None:
        # Map TermAPI op → primitive predicate function name
        self.op_to_predicate_func: Dict[str, str] = (
            dict(op_to_predicate_func)
            if op_to_predicate_func is not None
            else {
                # reach_point is the new name; keep free_move as a back-compat alias.
                "reach_point": "pred_reach_point",
                "reach_distance": "pred_reach_distance",
                "directional_move": "pred_directional_move",
                "axis_align": "pred_axis_align",
                "point_toward": "pred_point_toward",
                "directional_rotate": "pred_directional_rotate",
                "freeze_position": "pred_freeze_position",
                "freeze_orientation": "pred_freeze_orientation",
                "open_gripper": "pred_open_gripper",
                "close_gripper": "pred_close_gripper",
                "object_grasped": "pred_object_grasped",
                "manipulator_high_effort": "pred_manipulator_high_effort",
                "manipulator_low_effort": "pred_manipulator_low_effort",
            }
        )
        # Map condition label → no-arg predicate name (for OR-term sub-predicates).
        # "geometry" uses the constraint's op/args and is not listed here.
        # Gripper conditions (gripper_opened, gripper_closed) are handled separately.
        self._condition_to_predicate: Dict[str, str] = {
            "manipulator_high_effort": "pred_manipulator_high_effort",
            "manipulator_low_effort": "pred_manipulator_low_effort",
        }

    # -------------------------------------------------------------------------
    # 1. Compile a single IR (for one subtask) into a scope-level predicate
    # -------------------------------------------------------------------------

    def compile_scope(
        self,
        scope: Dict[str, Any],
        scope_name: str,
        ir: Mapping[str, Any],
        print_function: bool = False,
    ) -> List[str]:
        """
        Compile a flat-Constraint IR into a named scope-level predicate function.

        Parameters
        ----------
        scope : dict
            Dict used as the global scope for the generated function. It must
            already contain the primitive predicates and any other dependencies
            (e.g. `torch`).
        scope_name : str
            A human-readable label for this subtask/scope
            (e.g. the original subtask description).
        ir : Mapping[str, Any]
            IR with keys "Reasoning" and "Constraints".
        print_function : bool, default False
            If True, print the generated function source.

        Returns
        -------
        List[str]
            Names of the generated per-term predicate functions in order.
        """
        constraints = ir.get("Constraints")
        if constraints is None:
            raise ValueError("IR must contain a 'Constraints' key")
        if not isinstance(constraints, list) or len(constraints) == 0:
            raise ValueError("'Constraints' must be a non-empty list")

        # Validate for conflicting constraints
        self._validate_constraint_conflicts(constraints)

        reasoning = (ir.get("Reasoning") or "").strip()

        # Make a stable function name from the scope name
        func_name = self._sanitize_identifier(scope_name)

        # Compile each constraint term into its own function for debugging.
        # Per-term functions must mirror the same OR/AND combination of geometry
        # and effort/condition sub-predicates that the combined `predicate_function`
        # uses, otherwise the per-term diagnostic disagrees with the actual term
        # value (e.g. a `directional_move` constraint with
        # `condition: ["geometry", "manipulator_high_effort"]` should also report
        # True when high-effort contact is detected, just like the combined reducer).
        term_func_names: List[str] = []
        for idx, constraint in enumerate(constraints):
            op = constraint.get("op")
            if not isinstance(op, str) or len(op.strip()) == 0:
                raise ValueError(f"Constraint term is missing valid 'op': {constraint}")
            op_name = self._sanitize_identifier(op)
            term_func_name = f"{op_name}{idx}"
            calls, combinator = self._build_term_calls(constraint)
            term_func_names.append(term_func_name)

            term_lines: List[str] = []
            term_lines.append(f"def {term_func_name}():")
            term_lines.append('    """')
            term_lines.append(
                f"    Auto-generated predicate term for scope '{scope_name}'."
            )
            term_lines.append(f"    Term {idx}: op={op!r}")
            term_lines.append('    """')
            if len(calls) == 1:
                term_lines.append(f"    return {calls[0]}")
            else:
                for i, call in enumerate(calls):
                    term_lines.append(f"    _t{i} = {call}")
                combine_fn = (
                    "torch.logical_or" if combinator == "or" else "torch.logical_and"
                )
                expr = "_t0"
                for i in range(1, len(calls)):
                    expr = f"{combine_fn}({expr}, _t{i})"
                term_lines.append(f"    return {expr}")
            term_lines.append("")
            term_src = "\n".join(term_lines)
            # Do not print per-term (component) functions; only the combined predicate is printed below.
            exec(term_src, scope)

        lines: List[str] = []
        lines.append(f"def {func_name}(return_per_term=False):")
        lines.append('    """')
        lines.append(f"    Auto-generated predicate for scope '{scope_name}'.")
        if reasoning:
            for line in reasoning.splitlines():
                lines.append(f"    Reasoning: {line}")
        lines.append("    Left-to-right logical reduction over primitive predicates.")
        lines.append("    If return_per_term=True, returns tuple of per-constraint results.")
        lines.append('    """')
        lines.append("")

        # PERF: deduplicate identical primitive calls within this scope function.
        # Many learned IRs include repeated constraints (same op + args). We preserve
        # semantics by computing each unique call once and reusing the cached value.
        # This reduces redundant API lookups + math across both predicates and residuals.
        call_cache: Dict[str, str] = {}
        term_vars: List[str] = []

        # First constraint initializes the accumulator
        first = constraints[0]
        first_calls, first_combinator = self._build_term_calls(first)
        first_op = first.get("op", "unknown")
        first_cond = first.get("condition", "unknown")

        lines.append(
            f"    # Constraint 0: condition={first_cond!r}, op={first_op!r} (initializer)"
        )
        if len(first_calls) == 1:
            first_call = first_calls[0]
            if first_call in call_cache:
                first_var = call_cache[first_call]
            else:
                first_var = "_term_0"
                call_cache[first_call] = first_var
            if call_cache.get(first_call) == "_term_0":
                lines.append(f"    {first_var} = {first_call}")
        else:
            first_var = "_term_0"
            for i, call in enumerate(first_calls):
                lines.append(f"    _term_0_{i} = {call}")
            combine_fn = "torch.logical_or" if first_combinator == "or" else "torch.logical_and"
            expr = "_term_0_0"
            for i in range(1, len(first_calls)):
                expr = f"{combine_fn}({expr}, _term_0_{i})"
            lines.append(f"    {first_var} = {expr}")
        term_vars.append(first_var)
        lines.append(f"    result = {first_var}")
        lines.append("")

        # Remaining constraints: term is OR of sub-predicates when condition is a list
        # with effort alternatives; combination with result is always AND.
        for idx, constraint in enumerate(constraints[1:], start=1):
            op = constraint.get("op", "unknown")
            cond = constraint.get("condition", "unknown")
            # Honor explicit "logic" when present; otherwise infer from condition/op
            explicit_logic = constraint.get("logic")
            if isinstance(explicit_logic, str) and explicit_logic.strip().lower() in ("and", "or"):
                logic = explicit_logic.strip().lower()
            else:
                logic = self._infer_logic_from_condition(cond, op)

            calls, combinator = self._build_term_calls(constraint)
            if len(calls) == 1:
                call = calls[0]
                var = call_cache.get(call)
                if var is None:
                    var = f"_term_{idx}"
                    call_cache[call] = var
                    lines.append(f"    {var} = {call}")
            else:
                var = f"_term_{idx}"
                for i, call in enumerate(calls):
                    lines.append(f"    {var}_{i} = {call}")
                combine_fn = "torch.logical_or" if combinator == "or" else "torch.logical_and"
                expr = f"{var}_0"
                for i in range(1, len(calls)):
                    expr = f"{combine_fn}({expr}, {var}_{i})"
                lines.append(f"    {var} = {expr}")
            term_vars.append(var)
            lines.append(
                f"    # Constraint {idx}: condition={cond!r}, op={op!r}, logic={logic!r}"
            )
            # Logic only dictates how the term was built (OR of conditions); combine with result always via AND.
            lines.append(f"    result = torch.logical_and(result, {var})")
            lines.append("")

        lines.append("    if return_per_term:")
        # Optional: return per-term results (same order as Constraints).
        term_results_expr = ", ".join(term_vars)
        lines.append(f"        _term_results = ({term_results_expr})")
        lines.append("        return _term_results")
        lines.append("    return result")
        lines.append("")

        func_src = "\n".join(lines)

        if print_function:
            print(func_src)

        # Inject the generated function into the given scope.
        exec(func_src, scope)

        return term_func_names

    def adjust_scope(
        self,
        scope: Dict[str, Any],
        scope_name: str,
        ir: Mapping[str, Any],
        existing_term_func_names: List[str],
        print_function: bool = False,
    ) -> List[str]:
        self.compile_scope(scope, scope_name, ir, print_function)
        
    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def _validate_constraint_conflicts(self, constraints: List[Mapping[str, Any]]) -> None:
        """
        Validate invalid or conflicting geometric constraints.
        
        Detects:
        0. axis_align constraints that relate axes on the exact same frame,
           which is an invalid self-orientation relation.
        1. Two axis_align constraints with same controlled_target, reference_origin,
           controlled_axis, and alignment, but different reference_axis.
        2. Two axis_align constraints with same controlled_target, reference_origin,
           controlled_axis, and reference_axis, but different alignment.
        3. Two non-bipolar axis_align constraints with same controlled_target, reference_origin,
           opposite controlled_axis (e.g., "+y" vs "-y"), same reference_axis,
           and same alignment (geometrically impossible).
        
        Parameters
        ----------
        constraints : List[Mapping[str, Any]]
            List of constraint dictionaries to validate.
        
        Raises
        ------
        ValueError
            If conflicting constraints are detected.
        """
        errors: List[str] = []

        # close_gripper and object_grasped are mutually exclusive
        ops = [c.get("op", "") for c in constraints]
        if "close_gripper" in ops and "object_grasped" in ops:
            errors.append(
                "close_gripper and object_grasped cannot appear together: "
                "use object_grasped when an object must be held, close_gripper when the gripper must be shut (no object)."
            )

        def _are_opposite_axes(axis1: str, axis2: str) -> bool:
            """Check if two axis strings are opposites (e.g., "+y" vs "-y")."""
            if not isinstance(axis1, str) or not isinstance(axis2, str):
                return False
            axis1 = axis1.strip()
            axis2 = axis2.strip()
            # Check if they have opposite signs but same letter
            if len(axis1) >= 2 and len(axis2) >= 2:
                sign1 = axis1[0]
                sign2 = axis2[0]
                letter1 = axis1[1:].lower()
                letter2 = axis2[1:].lower()
                return (sign1 == "+" and sign2 == "-" or sign1 == "-" and sign2 == "+") and letter1 == letter2
            return False

        def _same_or_opposite_axes(axis1: str, axis2: str) -> bool:
            """Check if two axis strings name the same geometric line."""
            return axis1 == axis2 or _are_opposite_axes(axis1, axis2)

        for idx, constraint in enumerate(constraints):
            op = constraint.get("op")
            args = constraint.get("args", {})
            if op != "axis_align" or not isinstance(args, dict):
                continue

            controlled_entity = args.get("controlled_entity")
            controlled_target = args.get("controlled_target")
            reference_basis = args.get("reference_basis")
            reference_origin = args.get("reference_origin")
            if (
                isinstance(controlled_entity, str)
                and isinstance(controlled_target, str)
                and isinstance(reference_basis, str)
                and isinstance(reference_origin, str)
                and controlled_entity.strip()
                and controlled_target.strip()
                and reference_basis.strip()
                and controlled_entity == reference_basis
                and controlled_target == reference_origin
            ):
                errors.append(
                    f"Invalid axis_align constraint at index {idx}: "
                    f"controlled_entity={controlled_entity!r}, controlled_target={controlled_target!r}, "
                    f"reference_basis={reference_basis!r}, and reference_origin={reference_origin!r} "
                    f"refer to the same frame, so the orientation relation is self-referential"
                )
        
        # Collect all axis_align constraints with their indices
        axis_align_constraints: List[tuple[int, Mapping[str, Any]]] = []
        for idx, constraint in enumerate(constraints):
            op = constraint.get("op")
            if op == "axis_align":
                axis_align_constraints.append((idx, constraint))
        
        # Check for conflicts between pairs of axis_align constraints
        for i, (idx1, constraint1) in enumerate(axis_align_constraints):
            args1 = constraint1.get("args", {})
            if not isinstance(args1, dict):
                continue
            
            for idx2, constraint2 in axis_align_constraints[i + 1:]:
                args2 = constraint2.get("args", {})
                if not isinstance(args2, dict):
                    continue
                
                # Extract relevant fields
                target1 = args1.get("controlled_target")
                target2 = args2.get("controlled_target")
                origin1 = args1.get("reference_origin")
                origin2 = args2.get("reference_origin")
                c_axis1 = args1.get("controlled_axis")
                c_axis2 = args2.get("controlled_axis")
                r_axis1 = args1.get("reference_axis")
                r_axis2 = args2.get("reference_axis")
                align1 = args1.get("alignment")
                align2 = args2.get("alignment")
                align1_n = align1.strip().lower() if isinstance(align1, str) else align1
                align2_n = align2.strip().lower() if isinstance(align2, str) else align2
                
                # Check if they share the same base parameters
                if (target1 == target2 and 
                    origin1 == origin2 and 
                    c_axis1 == c_axis2):
                    
                    # Case 1: Same alignment but different reference_axis
                    if (
                        align1_n == align2_n
                        and r_axis1 != r_axis2
                        and not (
                            align1_n == "bipolar"
                            and _same_or_opposite_axes(r_axis1, r_axis2)
                        )
                    ):
                        errors.append(
                            f"Conflicting axis_align constraints at indices {idx1} and {idx2}: "
                            f"same controlled_target={target1!r}, reference_origin={origin1!r}, "
                            f"controlled_axis={c_axis1!r}, alignment={align1!r}, "
                            f"but different reference_axis ({r_axis1!r} vs {r_axis2!r})"
                        )
                    
                    # Case 2: Same reference_axis but different alignment
                    if r_axis1 == r_axis2 and align1_n != align2_n:
                        errors.append(
                            f"Conflicting axis_align constraints at indices {idx1} and {idx2}: "
                            f"same controlled_target={target1!r}, reference_origin={origin1!r}, "
                            f"controlled_axis={c_axis1!r}, reference_axis={r_axis1!r}, "
                            f"but different alignment ({align1!r} vs {align2!r})"
                        )
                
                # Case 3: Opposite controlled axes with same target, origin, reference_axis, and alignment
                if (target1 == target2 and 
                    origin1 == origin2 and 
                    _are_opposite_axes(c_axis1, c_axis2) and
                    r_axis1 == r_axis2 and 
                    align1_n == align2_n and
                    align1_n != "bipolar"):
                    errors.append(
                        f"Conflicting axis_align constraints at indices {idx1} and {idx2}: "
                        f"same controlled_target={target1!r}, reference_origin={origin1!r}, "
                        f"reference_axis={r_axis1!r}, alignment={align1!r}, "
                        f"but opposite controlled_axis ({c_axis1!r} vs {c_axis2!r}) - "
                        f"cannot simultaneously align opposite axes to the same reference axis"
                    )

        errors.extend(find_directional_rotate_axis_align_conflicts(constraints))
        errors.extend(find_manipulator_effort_condition_conflicts(constraints))
        
        if errors:
            raise ValueError(
                "Constraint validation failed due to conflicts:\n" + "\n".join(f"- {e}" for e in errors)
            )

    # -------------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------------

    def _build_term_call(self, term: Mapping[str, Any]) -> str:
        """
        Turn a single constraint entry into a primitive predicate call.

        term ~
            {
              "condition": "...",
              "op": "free_move | ...",
              "args": { ... }
            }

        We map 'op' → primitive predicate name (e.g. 'pred_free_move') via
        self.op_to_predicate_func, and then emit:

            pred_free_move(controlled_entity=..., controlled_target=..., ...)
        """
        op = term.get("op")
        if op is None:
            raise ValueError(f"Constraint term is missing 'op': {term}")

        if op not in self.op_to_predicate_func:
            raise ValueError(f"Unknown op '{op}' in term: {term}")

        func_name = self.op_to_predicate_func[op]

        args = term.get("args", {})
        if not isinstance(args, dict):
            raise ValueError(f"'args' must be a dict for term {term}")

        if not args:
            return f"{func_name}()"

        parts: List[str] = []
        for key, val in args.items():
            parts.append(f"{key}={self._format_value(val)}")
        return f"{func_name}({', '.join(parts)})"

    def _build_term_calls(self, constraint: Mapping[str, Any]) -> tuple[List[str], str]:
        """
        Build one or more primitive predicate call strings for a constraint,
        plus the combinator ("and" or "or") used to join them within the term.

        open_gripper and close_gripper ignore the condition list entirely: they
        only imply pred_open_gripper() and pred_close_gripper() respectively.
        We never add pred_manipulator_high_effort / pred_manipulator_low_effort
        (or other condition keywords) for these ops.

        For all other ops with a list-valued condition we ignore the LLM's
        declared "logic" and apply a fixed rule:
        - condition contains "manipulator_high_effort" → combinator = "or"
          (geometry goal OR high-effort detection suffices)
        - condition contains "manipulator_low_effort"  → combinator = "and"
          (geometry goal AND low-effort both required)
        - otherwise fall back to the declared logic ("or" emits sub-calls
          OR-combined; anything else collapses to a single geometry call).

        Sub-calls per condition entry:
        - "geometry" → the constraint's op/args (e.g. pred_reach_point(...))
        - "manipulator_high_effort" → pred_manipulator_high_effort()
        - "manipulator_low_effort"  → pred_manipulator_low_effort()
        - etc. via _condition_to_predicate.
        """
        op = constraint.get("op")
        if isinstance(op, str) and op.strip().lower() in ("open_gripper", "close_gripper", "object_grasped"):
            return [self._build_term_call(constraint)], "and"
        cond = constraint.get("condition")
        if isinstance(cond, (list, tuple)):
            conds = [c.strip().lower() for c in cond if isinstance(c, str)]
            if len(conds) <= 1:
                return [self._build_term_call(constraint)], "and"

            if "manipulator_high_effort" in conds:
                combinator = "or"
            elif "manipulator_low_effort" in conds:
                combinator = "and"
            else:
                logic = constraint.get("logic")
                if isinstance(logic, str) and logic.strip().lower() == "or":
                    combinator = "or"
                else:
                    return [self._build_term_call(constraint)], "and"

            calls: List[str] = []
            for c in conds:
                if c == "geometry":
                    calls.append(self._build_term_call(constraint))
                else:
                    pred = self._condition_to_predicate.get(c)
                    if pred:
                        calls.append(f"{pred}()")
            if not calls:
                return [self._build_term_call(constraint)], "and"
            return calls, combinator
        return [self._build_term_call(constraint)], "and"


    @staticmethod
    def _infer_logic_from_condition(condition: Any, op: Any) -> str:
        """
        Infer logical combination ('and'/'or') from the condition label and op.

        Design (per Zhi's spec)
        -----------------------
        * Default: use AND.
        * Only use OR when:
            - the condition is *not* "geometry", AND
            - the op is *not* one of the gripper/effort ops
              (`open_gripper`, `close_gripper`,
               `manipulator_high_effort`, `manipulator_low_effort`).

        In particular, even if the condition is e.g. "gripper_closed",
        we still use AND when the op itself is one of the gripper/effort
        primitives. This matches the rule:

            "Or should only happen if they use any op that are not
             lines 67-70 and decide to use condition that is not geometry."
        """
        # Normalise condition/op to strings
        if isinstance(condition, str):
            cond = condition.strip().lower()
        elif isinstance(condition, (list, tuple)):
            # List condition e.g. ["manipulator_high_effort", "geometry"] means
            # "satisfy when (geometry) OR (effort)" → treat as non-geometry so OR is used.
            conds = [c.strip().lower() for c in condition if isinstance(c, str)]
            cond = "geometry" if all(c == "geometry" for c in conds) else (conds[0] if conds else "")
        else:
            cond = ""

        if isinstance(op, str):
            op_name = op.strip().lower()
        else:
            op_name = ""

        gripper_effort_ops = {
            "open_gripper",
            "close_gripper",
            "object_grasped",
            "manipulator_high_effort",
            "manipulator_low_effort",
        }

        # OR only when:
        #   - condition is explicitly non-geometry, AND
        #   - op is *not* a gripper/effort op.
        if cond not in ("", "geometry") and op_name not in gripper_effort_ops:
            return "or"

        # Otherwise, default to AND.
        return "and"
    
    @staticmethod
    def _sanitize_identifier(name: str) -> str:
        """
        Turn an arbitrary scope name into a safe Python identifier suffix.
        """
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
            inner = ", ".join(PredicateModuleGenerator._format_value(v) for v in value)
            # If you care about distinguishing tuple vs list, do it here:
            if isinstance(value, tuple):
                # Handle 1-element tuple syntax
                if len(value) == 1:
                    inner = inner + ","
                return f"({inner})"
            return f"[{inner}]"

        if value is None:
            return "None"

        # Fallback: use repr
        return repr(value)
