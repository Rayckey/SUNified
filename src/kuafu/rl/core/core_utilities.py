from typing import Any
from kuafu.rl.llm.prompts import FAILURE_FEEDBACK
from kuafu.common.env_prompt_helpers import format_single_run_feedback_block

def extract_tolerances_from_constraints_raw(
    constraints_raw: list[dict[str, Any]] | None,
) -> dict[int, dict[str, object]]:
    """Extract tolerance parameters from constraints_raw and convert to current_parameters format.
    
    Args:
        constraints_raw: List of constraint dicts with structure [{"op": "...", "args": {...}}, ...]
        
    Returns:
        Dict mapping constraint index to tolerance parameters, e.g.:
        {0: {"tolerance_m": 0.05}, 1: {"tolerance_rad": 0.2}}
    """
    if not constraints_raw:
        return {}
    
    # Allowlist of tolerance fields that can be extracted
    allowed_tolerance_fields = {"tolerance_m", "tolerance_rad"}
    
    parameters: dict[int, dict[str, object]] = {}
    
    for idx, constraint in enumerate(constraints_raw):
        if not isinstance(constraint, dict):
            continue
        
        args = constraint.get("args", {})
        if not isinstance(args, dict):
            continue
        
        # Extract tolerance fields from args
        tolerance_params: dict[str, object] = {}
        for field in allowed_tolerance_fields:
            if field in args:
                value = args[field]
                # Only include numeric values
                try:
                    float_value = float(value)
                    if float_value >= 0:  # Tolerances should be non-negative
                        tolerance_params[field] = float_value
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"Invalid tolerance field {field!r}: {value!r}") from exc
        
        # Only add entry if we found at least one tolerance field
        if tolerance_params:
            parameters[idx] = tolerance_params
    
    return parameters

def format_correcting_prompt(
    r:  Any,
    stage_id: int,
    run_idx: int,
    llm_resp: Any,
    feedback_subsampling: int,
    fb: str | None = None,
) -> str:
    return FAILURE_FEEDBACK.format(
        traceback_msg=getattr(r, "exception", "") or "",
        previous_response=(
            llm_resp.raw_outputs[run_idx] if run_idx < len(llm_resp.raw_outputs) else ""
        ),
    )
