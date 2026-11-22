"""Data-only contracts shared by Python, JSON CLI and MCP callers.

Geometry follows the existing constraint prompt. Numeric acceptance tolerances
are deliberately absent: the receiving controller, not the proposing agent,
assigns them through the common tolerance policy.
"""



def obj(properties, required=None):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties) if required is None else required,
        "additionalProperties": False,
    }


TEXT = {"type": "string", "minLength": 1, "maxLength": 16000}
NUMBER = {"type": "number", "minimum": -1000, "maximum": 1000}
VECTOR = {"type": "array", "items": NUMBER, "minItems": 3, "maxItems": 3}
AXIS = {"enum": ["+x", "-x", "+y", "-y", "+z", "-z"]}
TARGET = {"controlled_entity": {"enum": ["robot_frame", "gripped_object"]}, "controlled_target": TEXT}
REFERENCE = {"reference_basis": {"enum": ["world", "object", "robot_frame"]}, "reference_origin": TEXT}
OP_ARGS = {
    "reach_point": {**TARGET, **REFERENCE, "offset_from_origin_xyz": VECTOR},
    "directional_move": {
        **TARGET,
        **REFERENCE,
        "axis": AXIS,
        "travel_distance_m": {"type": "number", "exclusiveMinimum": 0, "maximum": 1000},
    },
    "axis_align": {
        **TARGET,
        **REFERENCE,
        "controlled_axis": AXIS,
        "reference_axis": AXIS,
        "alignment": {"enum": ["codirectional", "antiparallel", "bipolar", "orthogonal"]},
    },
    "point_toward": {**TARGET, **REFERENCE, "controlled_axis": AXIS, "offset_from_origin_xyz": VECTOR},
    "directional_rotate": {**TARGET, **REFERENCE, "rotation_axis": AXIS, "rotation_angle_rad": NUMBER},
    "freeze_position": TARGET,
    "freeze_orientation": TARGET,
    "open_gripper": {},
    "close_gripper": {},
    "object_grasped": {},
}
TERM = {
    "oneOf": [
        obj(
            {
                "logic": {"const": "and"},
                "condition": {
                    "type": "array",
                    "minItems": 1,
                    "uniqueItems": True,
                    "items": {
                        "enum": [
                            "geometry",
                            "gripper_opened",
                            "gripper_closed",
                            "manipulator_high_effort",
                            "manipulator_low_effort",
                        ]
                    },
                },
                "op": {"const": op},
                "args": obj(args),
            }
        )
        for op, args in OP_ARGS.items()
    ]
}
PROGRAM = obj(
    {
        "stages": {
            "type": "array",
            "minItems": 1,
            "maxItems": 32,
            "items": obj(
                {
                    "task": TEXT,
                    "atomic_actions": {"type": "array", "items": TEXT, "minItems": 1, "maxItems": 32},
                    "constraints": obj(
                        {
                            "Reasoning": TEXT,
                            "Constraints": {"type": "array", "items": TERM, "minItems": 1, "maxItems": 32},
                        }
                    ),
                }
            ),
        }
    }
)
