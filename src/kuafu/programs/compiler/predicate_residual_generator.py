from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import re
import torch

from kuafu.programs.compiler.constraint_validation import (
    find_manipulator_effort_condition_conflicts,
)


class PredicateResidualModuleGenerator:
    """
    Compile a flat Constraint IR into a single scope-level function that returns BOTH:
      - a predicate truth value (logical reduction over primitive predicates), and
      - a residual cost (sum over primitive residuals)

    This lets callers evaluate predicates and residuals in one compiled function while
    sharing a per-evaluation cache (via APIWrapper.eval_cache + memo()) across both.

    IMPORTANT:
    - This generator does NOT replace PredicateModuleGenerator / ResidualModuleGenerator.
      Those are still needed elsewhere (e.g. predicate_function() is used independently).
    - The caller must provide a `scope` that already contains:
        * primitive predicates (pred_free_move, ...)
        * primitive residuals (free_move_residual, ...)
        * torch
    """

    def __init__(
        self,
        op_to_predicate_func: Optional[Mapping[str, str]] = None,
        op_to_residual_func: Optional[Mapping[str, str]] = None,
    ) -> None:
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
                "manipulator_high_effort": "manipulator_high_effort_residual",
                "manipulator_low_effort": "manipulator_low_effort_residual",
            }
        )

    def compile_scope(
        self,
        scope: Dict[str, Any],
        scope_name: str,
        ir: Mapping[str, Any],
        *,
        function_name: str = "predicate_residual_function",
        print_function: bool = False,
    ) -> str:
        constraints = ir.get("Constraints")
        if constraints is None:
            raise ValueError("IR must contain a 'Constraints' key")
        if not isinstance(constraints, list) or len(constraints) == 0:
            raise ValueError("'Constraints' must be a non-empty list")
        effort_conflicts = find_manipulator_effort_condition_conflicts(constraints)
        if effort_conflicts:
            raise ValueError(
                "Constraint validation failed due to conflicts:\n"
                + "\n".join(f"- {error}" for error in effort_conflicts)
            )

        reasoning = (ir.get("Reasoning") or "").strip()
        func_name = self._sanitize_identifier(function_name)

        lines: List[str] = []
        lines.append(f"def {func_name}(state):")
        lines.append('    """')
        lines.append(f"    Auto-generated combined predicate+residual for scope '{scope_name}'.")
        if reasoning:
            for line in reasoning.splitlines():
                lines.append(f"    Reasoning: {line}")
        lines.append("    Returns: (predicate_bool, residual_cost)")
        lines.append('    """')
        lines.append("")

        # -------- predicate reduction (same logic rule as PredicateModuleGenerator) -----
        first = constraints[0]
        first_pred_call = self._build_predicate_call(first)
        first_op = first.get("op", "unknown")
        first_cond = first.get("condition", "unknown")
        lines.append(f"    # Predicate term 0: condition={first_cond!r}, op={first_op!r}")
        lines.append(f"    pred = {first_pred_call}")
        lines.append("")

        for idx, term in enumerate(constraints[1:], start=1):
            op = term.get("op", "unknown")
            cond = term.get("condition", "unknown")
            # Prefer explicit IR-provided "logic" (e.g. "and"/"or") if present.
            logic_raw = term.get("logic")
            if isinstance(logic_raw, str) and logic_raw.strip().lower() in ("and", "or"):
                logic = logic_raw.strip().lower()
            else:
                logic = self._infer_logic_from_condition(cond, op)
            call = self._build_predicate_call(term)
            lines.append(f"    # Predicate term {idx}: condition={cond!r}, op={op!r}, logic={logic!r}")
            if logic == "and":
                lines.append(f"    pred = torch.logical_and(pred, {call})")
            else:
                lines.append(f"    pred = torch.logical_or(pred, {call})")
            lines.append("")

        # -------- residual sum ---------------------------------------------------------
        first_res_call = self._build_residual_call(first)
        lines.append("    # Residual term 0")
        lines.append(f"    cost = {first_res_call}")
        lines.append("")

        for idx, term in enumerate(constraints[1:], start=1):
            call = self._build_residual_call(term)
            lines.append(f"    # Residual term {idx}")
            lines.append(f"    cost = cost + {call}")
            lines.append("")

        lines.append("    return pred, cost")
        lines.append("")

        func_src = "\n".join(lines)
        if print_function:
            print(func_src)

        if "torch" not in scope:
            scope["torch"] = torch

        exec(func_src, scope)
        return func_name

    # ---------------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------------
    def _build_predicate_call(self, term: Mapping[str, Any]) -> str:
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
        for k, v in args.items():
            parts.append(f"{k}={self._format_value(v)}")
        return f"{func_name}({', '.join(parts)})"

    def _build_residual_call(self, term: Mapping[str, Any]) -> str:
        op = term.get("op")
        if op is None:
            raise ValueError(f"Constraint term is missing 'op': {term}")
        if op not in self.op_to_residual_func:
            raise ValueError(f"Unknown op '{op}' in term: {term}")
        func_name = self.op_to_residual_func[op]
        args = term.get("args", {})
        if not isinstance(args, dict):
            raise ValueError(f"'args' must be a dict for term {term}")
        parts: List[str] = ["state"]
        for k, v in args.items():
            parts.append(f"{k}={self._format_value(v)}")
        return f"{func_name}({', '.join(parts)})"

    @staticmethod
    def _infer_logic_from_condition(condition: Any, op: Any) -> str:
        if isinstance(condition, str):
            cond = condition.strip().lower()
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
        if cond not in ("", "geometry") and op_name not in gripper_effort_ops:
            return "or"
        return "and"

    @staticmethod
    def _sanitize_identifier(name: str) -> str:
        base = name.strip().lower()
        base = re.sub(r"[^0-9a-zA-Z_]+", "_", base)
        base = re.sub(r"_+", "_", base)
        base = base.strip("_") or "unnamed"
        if base[0].isdigit():
            base = "_" + base
        return base

    @staticmethod
    def _format_value(value: Any) -> str:
        if isinstance(value, str):
            escaped = value.replace("'", "\\'")
            return f"'{escaped}'"
        if isinstance(value, bool):
            return "True" if value else "False"
        if isinstance(value, (int, float)):
            return repr(value)
        if isinstance(value, (list, tuple)):
            inner = ", ".join(PredicateResidualModuleGenerator._format_value(v) for v in value)
            if isinstance(value, tuple):
                if len(value) == 1:
                    inner = inner + ","
                return f"({inner})"
            return f"[{inner}]"
        if value is None:
            return "None"
        return repr(value)


