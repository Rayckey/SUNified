from typing import Dict, Any, List

def format_constraint_description(constraint: Dict[str, Any]) -> str:
    """
    Generate a human-readable description of a constraint based on its operator and arguments.
    
    Args:
        constraint: Dictionary containing "op", "args", "logic", and "condition"
    
    Returns:
        Brief, concise description of what the constraint does
    """
    op = constraint.get("op", "unknown")
    args = constraint.get("args", {})
    logic = constraint.get("logic", "and")
    
    # Helper to format offset/axis references
    def format_reference(root, basis):
        if root == "world":
            return "world frame"
        return f"{root} ({basis} basis)"
    
    # Helper to format axis direction
    def format_axis(axis_str):
        if axis_str.startswith("+"):
            return f"{axis_str[1:].upper()}-axis"
        else:
            return f"negative {axis_str[1:].upper()}-axis"
    
    # Helper to format alignment type
    def format_alignment(align):
        align_map = {
            "codirectional": "aligned in the same direction",
            "antiparallel": "aligned in opposite directions",
            "bipolar": "aligned in either direction",
            "orthogonal": "perpendicular"
        }
        return align_map.get(align, align)
    
    descriptions = {
        "reach_point": lambda: (
            f"Reach target point at {format_reference(args.get('reference_origin', 'world'), args.get('reference_basis', 'world'))} "
            f"with offset {args.get('offset_from_origin_xyz', [0,0,0])} "
            f"(tolerance: {args.get('tolerance_m', 0.0):.3f} m)"
        ),
        
        "reach_distance": lambda: (
            f"Maintain distance {args.get('distance_m', 0.0):.3f} m "
            f"(±{args.get('tolerance_m', 0.0):.3f} m) from reference point at "
            f"{format_reference(args.get('reference_origin', 'world'), args.get('reference_basis', 'world'))} "
            f"with offset {args.get('offset_from_origin_xyz', [0,0,0])}"
        ),
        
        "directional_move": lambda: (
            f"Move along {format_axis(args.get('axis', '+x'))} by {args.get('travel_distance_m', 0.0):.3f} m "
            f"(tolerance: {args.get('tolerance_m', 0.0):.3f} m)"
        ),
        
        "axis_align": lambda: (
            f"Align {format_axis(args.get('controlled_axis', '+x'))} to be {format_alignment(args.get('alignment', 'codirectional'))} "
            f"with {format_axis(args.get('reference_axis', '+x'))} at {format_reference(args.get('reference_origin', 'world'), args.get('reference_basis', 'world'))} "
            f"(tolerance: {args.get('tolerance_rad', 0.2):.3f} rad)"
        ),
        
        "point_toward": lambda: (
            f"Point {format_axis(args.get('controlled_axis', '+x'))} toward target at "
            f"{format_reference(args.get('reference_origin', 'world'), args.get('reference_basis', 'world'))} "
            f"with offset {args.get('offset_from_origin_xyz', [0,0,0])} "
            f"(tolerance: {args.get('tolerance_rad', 0.2):.3f} rad)"
        ),
        
        "directional_rotate": lambda: (
            f"Rotate about {format_axis(args.get('rotation_axis', '+z'))} by {args.get('rotation_angle_rad', 0.0):.3f} rad "
            f"at {format_reference(args.get('reference_origin', 'world'), args.get('reference_basis', 'world'))} "
            f"(tolerance: {args.get('tolerance_rad', 0.2):.3f} rad)"
        ),
        
        "freeze_position": lambda: (
            f"Keep position fixed (tolerance: {args.get('tolerance_m', 0.0):.3f} m)"
        ),
        
        "freeze_orientation": lambda: (
            f"Keep orientation fixed (tolerance: {args.get('tolerance_rad', 0.2):.3f} rad)"
        ),
        
        "open_gripper": lambda: "Gripper should be open",
        
        "close_gripper": lambda: "Gripper fingers are fully shut (no object between them)",
        "object_grasped": lambda: "Gripper is physically grasping an object (contact on both fingers, not fully closed)",
    }
    
    formatter = descriptions.get(op)
    if formatter:
        try:
            desc = formatter()
            # Add controlled entity info if available
            controlled_entity = args.get("controlled_entity", "")
            controlled_target = args.get("controlled_target", "")
            if controlled_entity and controlled_target:
                entity_desc = "gripped object" if controlled_entity == "gripped_object" else "robot frame"
                desc = f"[{entity_desc} {controlled_target}] {desc}"
            return desc
        except Exception as e:
            return f"{op} constraint (error formatting: {e})"
    else:
        return f"{op} constraint (unknown operator)"

def explain_constraints(constraints) -> List[str]:
    """
    Generate brief, concise explanations of all constraints.
    
    Uses templates based on the constraint prompt to generate human-readable
    descriptions of what each constraint does.
    
    Args:
        constraints: List of constraint dictionaries
    
    Returns:
        List of strings, where each string describes one constraint.
        The index of each string matches the constraint ID.
    
    Example:
        >>> descriptions = explain_constraints(constraints)
        >>> print(descriptions[0])
        "Component 0: [robot frame panda_fingertip] Reach target point..."
    """
    descriptions = []
    for idx, constraint in enumerate(constraints):
        logic_raw = constraint.get("logic", "and")
        condition_tags = constraint.get("condition", []) or []
        desc = format_constraint_description(constraint)

        # Remove purely geometric tagging from the printed condition tags
        filtered_tags = [tag for tag in condition_tags if tag != "geometry"]

        # If this is an OR-condition that becomes active when the manipulator reaches
        # high effort, surface that fact in the natural-language description rather
        # than as a raw condition tag.
        if logic_raw.lower() == "or" and "manipulator_high_effort" in condition_tags:
            desc = f"{desc}, or when manipulator reaches high effort"
            filtered_tags = [tag for tag in filtered_tags if tag != "manipulator_high_effort"]
        
        condition_str = f" [{', '.join(filtered_tags)}]" if filtered_tags else ""
        descriptions.append(f"Component {idx}{condition_str}: {desc}")
    
    return descriptions
