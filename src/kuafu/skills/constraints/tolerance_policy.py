"""Receiving-side tolerance policies shared by MPC formulation and learning.

The learned constraint program still carries ``tolerance_m`` and
``tolerance_rad`` fields because the predicate/residual compilers require them.
This module assigns reproducible values from received operator geometry,
stage semantics, and scene metadata. The LLM does not select numeric thresholds.
"""

from __future__ import annotations

from copy import deepcopy
import math
import re
from typing import Any, Iterable


POLICY_VERSION = "operator_geometry_v2"

# The LLM-facing geometry schema deliberately excludes these fields. Ownership
# is determined by the operator, never by whichever keys the model returned.
OP_TOLERANCE_FIELDS = {
    "reach_point": "tolerance_m",
    "reach_distance": "tolerance_m",
    "directional_move": "tolerance_m",
    "freeze_position": "tolerance_m",
    "axis_align": "tolerance_rad",
    "point_toward": "tolerance_rad",
    "directional_rotate": "tolerance_rad",
    "freeze_orientation": "tolerance_rad",
    "open_gripper": None,
    "close_gripper": None,
    "object_grasped": None,
    "manipulator_high_effort": None,
    "manipulator_low_effort": None,
}

POSITION_MIN_M = 0.01
POSITION_CONTACT_DEFAULT_M = 0.02
POSITION_PRECONTACT_M = 0.03
POSITION_FREE_SPACE_M = 0.05

ANGULAR_PRECISE_RAD = 0.30
ANGULAR_FREE_SPACE_RAD = 0.50

MPC_POSITION_PRECONTACT_M = 0.02
MPC_POSITION_PRECISE_M = 0.01
MPC_ANGULAR_RAD = 0.20

_TOLERANCE_KEYS = frozenset({"tolerance_m", "tolerance_rad"})
_EXPLANATION_TOLERANCE_RE = re.compile(
    r"\s*\((?:tolerance:\s*|±\s*)"
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?\s*(?:m|rad)\)",
    re.IGNORECASE,
)
_SIZE_PATTERN = re.compile(
    r"\bsize(?:\s*\(approx\.?\)|\s*\(approximate\))?\s*:?\s*\(([^)]*)\)",
    re.IGNORECASE,
)
_FEATURE_SIZE_PATTERN = re.compile(
    r"\b(?:diameter|thickness|width|cross[- ]section)\s*(?:is|=|:)?\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*(m|cm|mm)?\b",
    re.IGNORECASE,
)
_SHAPE_SIZE_PATTERN = re.compile(
    r"\b(?:as|is|about|approximately)\s+(?:a\s+)?"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*(m|cm|mm)\s*"
    r"(?:cube|sphere|wide|thick|cross[- ]section)\b",
    re.IGNORECASE,
)
_NUMBER_WITH_UNIT = re.compile(
    r"^\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*(m|cm|mm)?\s*$",
    re.IGNORECASE,
)

_RECEIVER_TERMS = (
    "insert",
    "insertion",
    "thread",
    "hook",
    "engage",
    "engagement",
    "seat",
    "seating",
    "receiver",
    "opening",
    "slot",
    "holder",
    "place into",
    "place on",
    "placement",
    "stack on",
    "supported pose",
)
_COARSE_MOTION_TERMS = (
    "lift",
    "lower",
    "raise",
    "carry",
    "transport",
    "clearance",
    "retract",
    "back away",
    "back-away",
    "free-space",
    "free space",
    "non-contact configuration",
    "reposition",
)
_RETRACT_TERMS = ("retract", "back away", "back-away", "move away")
_PREENGAGEMENT_TERMS = (
    "alignment",
    "aligned with",
    "pre-engagement",
    "pre-grasp",
    "pre-insertion",
    "pre-placement",
    "placement configuration",
    "standoff",
)
_CONSTRAINED_FEATURE_TERMS = (
    "branch",
    "button",
    "cap",
    "edge",
    "handle",
    "hook",
    "knob",
    "rim",
    "shaft",
    "slot",
)
_SECURED_RELEASE_TERMS = (
    "attached",
    "captured",
    "fixed feature",
    "hooked",
    "hanging",
    "inserted",
    "inside",
    "mechanism",
    "seated",
    "stacked",
    "stacking",
    "supported",
    "secured",
)
_SECURED_ROLE_TERMS = (
    "attached",
    "branch",
    "button",
    "cabinet",
    "drawer",
    "fixture",
    "handle",
    "holder",
    "mechanism",
    "mounted",
    "opening",
    "receiver",
)
_ROTATION_CRITICAL_TERMS = (
    "insert",
    "insertion",
    "thread",
    "hook",
    "engage",
    "seat",
    "twist",
    "turn",
    "torque",
    "press",
    "upright",
    "final orientation",
    "orientation validity",
)
_ROTATION_PROGRESS_TERMS = (
    "actuat",
    "door",
    "hinge",
    "insert",
    "mechanism",
    "seat",
    "twist",
    "turn",
    "unlock",
    "upright",
)
_SLENDER_TERMS = (
    "slender",
    "long and thin",
    "long/thin",
    "rod-like",
    "rodlike",
    "narrow shaft",
    "thin shaft",
)


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), float(lower)), float(upper))


def _rounded(value: float) -> float:
    return round(float(value), 6)


def _to_metres(value: float, unit: str | None) -> float:
    normalized = (unit or "m").lower()
    if normalized == "cm":
        return float(value) * 0.01
    if normalized == "mm":
        return float(value) * 0.001
    return float(value)


def _object_description_text(blob: Any) -> str:
    if isinstance(blob, str):
        return blob
    if isinstance(blob, dict):
        return str(blob["description"]) if "description" in blob else " ".join(str(value) for value in blob.values())
    if isinstance(blob, (list, tuple)):
        # Environment metadata is (description, USD path). Paths/identifiers
        # are not semantic descriptions and must not affect classification.
        return str(blob[0]) if blob else ""
    return str(blob)


def parse_detectable_object_sizes(detectable_objects: Any) -> dict[str, float]:
    """Return the smallest declared physical dimension for each object/frame."""

    if not isinstance(detectable_objects, dict):
        return {}

    sizes: dict[str, float] = {}
    for raw_name, blob in detectable_objects.items():
        text = _object_description_text(blob)
        tuple_match = _SIZE_PATTERN.search(text)
        dimensions: list[float] = []
        if isinstance(blob, dict) and isinstance(blob.get("size_m"), (list, tuple)):
            for raw_value in blob["size_m"]:
                value = float(raw_value)
                if math.isfinite(value) and value > 0:
                    dimensions.append(value)
        if tuple_match and not dimensions:
            for part in tuple_match.group(1).split(","):
                match = _NUMBER_WITH_UNIT.fullmatch(part)
                if not match:
                    continue
                value = _to_metres(float(match.group(1)), match.group(2))
                if math.isfinite(value) and value > 0.0:
                    dimensions.append(value)
        if not dimensions:
            feature_match = _FEATURE_SIZE_PATTERN.search(text)
            if feature_match:
                value = _to_metres(float(feature_match.group(1)), feature_match.group(2))
                if math.isfinite(value) and value > 0.0:
                    dimensions.append(value)
        if not dimensions:
            shape_match = _SHAPE_SIZE_PATTERN.search(text)
            if shape_match:
                value = _to_metres(float(shape_match.group(1)), shape_match.group(2))
                if math.isfinite(value) and value > 0.0:
                    dimensions.append(value)
        if dimensions:
            sizes[str(raw_name)] = min(dimensions)
    return sizes


def _detectable_object_descriptions(detectable_objects: Any) -> dict[str, str]:
    if not isinstance(detectable_objects, dict):
        return {}
    return {
        str(name): _object_description_text(blob).lower()
        for name, blob in detectable_objects.items()
    }


def _is_receiver_description(description: str) -> bool:
    return _contains_any(
        description,
        (
            "cavity",
            "container interior",
            "holder",
            "handle opening",
            "hole/opening",
            "hanger branch",
            "support surface",
            "insertion opening",
            "opening center",
            "openings along",
            "placement target",
            "receiver",
            "slot",
            "support target",
        ),
    )


def _contains_any(text: str, terms: Iterable[str]) -> bool:
    return any(term in text for term in terms)


def _mentioned_names(text: str, names: Iterable[str]) -> set[str]:
    """Match complete entity identifiers, not overlapping name prefixes."""
    return {name for name in names if name and re.search(
        rf"(?<!\w){re.escape(name)}(?!\w)", text, re.IGNORECASE
    )}


def _semantic_text(text: str, descriptions: dict[str, str]) -> str:
    """Prevent an identifier such as insertion_7 from acting as a verb."""
    if not descriptions:
        return text.lower()
    pattern = r"(?<!\w)(?:" + "|".join(
        re.escape(name) for name in sorted(descriptions, key=len, reverse=True)
    ) + r")(?!\w)"
    return re.sub(pattern, "entity", text, flags=re.IGNORECASE).lower()


def _conditions(constraint: dict[str, Any]) -> set[str]:
    raw = constraint.get("condition", [])
    if isinstance(raw, str):
        return {raw}
    if isinstance(raw, (list, tuple, set)):
        return {str(value) for value in raw}
    return set()


def _offset_is_zero(args: dict[str, Any]) -> bool:
    offset = args.get("offset_from_origin_xyz")
    if not isinstance(offset, (list, tuple)) or len(offset) != 3:
        return False
    try:
        return math.sqrt(sum(float(value) ** 2 for value in offset)) <= 1e-9
    except (TypeError, ValueError):
        return False


def strip_tolerance_annotation(value: Any) -> str:
    """Remove the generated parenthetical threshold from an explanation."""

    return _EXPLANATION_TOLERANCE_RE.sub("", str(value)).strip()


def summarize_atomic_action(value: Any) -> str:
    """Return only the semantic action headline from stored atomic-action text."""

    text = str(value).strip()
    if not text:
        return ""
    first_line = re.split(r"(?:\r?\n|(?:\\r)?\\n)", text, maxsplit=1)[0].strip()
    match = re.search(r"\[(?:Approach|Manipulate|Release)\].*", first_line, re.IGNORECASE)
    if match:
        first_line = match.group(0)
    return first_line.rstrip("',) ").strip()


def _stage_phase(atomic_action: str) -> str:
    first_line = summarize_atomic_action(atomic_action).lower()
    if "[approach]" in first_line:
        return "approach"
    if "[manipulate]" in first_line:
        return "manipulate"
    if "[release]" in first_line:
        return "release"
    return "unknown"


def _absolute_numeric_arg(args: dict[str, Any], key: str) -> float | None:
    try:
        value = abs(float(args[key]))
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _active_receiver_motion(text: str) -> bool:
    if _contains_any(
        text,
        ("insert", "seat", "hook", "engage", "placement contact"),
    ):
        return True
    if re.search(r"\bplac(?:e|ing)\b.*\b(?:on|onto|into|inside)\b", text):
        return True
    if re.search(r"\blower\b.*(?:\bonto\b|\bcontact\b|\binside\b)", text):
        return True
    return False


def _referenced_size(
    constraint: dict[str, Any],
    *,
    semantic_text: str,
    object_sizes: dict[str, float],
    preferred_origin: bool = False,
) -> tuple[float | None, str | None]:
    args = constraint.get("args")
    if not isinstance(args, dict):
        args = {}

    controlled_target = args.get("controlled_target")
    reference_origin = args.get("reference_origin")
    if preferred_origin and reference_origin in object_sizes:
        return float(object_sizes[reference_origin]), str(reference_origin)
    explicit_matches: list[tuple[float, str]] = []
    for name in (controlled_target, reference_origin):
        if isinstance(name, str) and name in object_sizes:
            explicit_matches.append((float(object_sizes[name]), name))
    if explicit_matches:
        # For receiver/fit metrics, the smaller of the held-object cross-section
        # and receiver feature is the safest available clearance proxy.
        return min(explicit_matches, key=lambda item: item[0])

    matches = [
        (float(size), name)
        for name, size in object_sizes.items()
        if name in _mentioned_names(semantic_text, object_sizes)
    ]
    if not matches:
        return None, None
    return min(matches, key=lambda item: item[0])


def _size_limited_position_tolerance(size_m: float | None, *, unknown: float) -> float:
    if size_m is None:
        return float(unknown)
    return _clamp(0.5 * float(size_m), POSITION_MIN_M, POSITION_PRECONTACT_M)


def _position_tolerance(
    constraint: dict[str, Any],
    *,
    phase: str,
    semantic_text: str,
    receiver_role_text: str,
    feature_role_text: str,
    object_sizes: dict[str, float],
    reference_text: str,
    stage_has_grasp: bool,
    stage_has_open_gripper: bool,
    stage_secured_release: bool,
) -> tuple[float, str]:
    op = str(constraint.get("op") or "")
    args_raw = constraint.get("args")
    args = args_raw if isinstance(args_raw, dict) else {}
    conditions = _conditions(constraint)
    high_effort = "manipulator_high_effort" in conditions
    receiver = _contains_any(f"{semantic_text} {receiver_role_text}", _RECEIVER_TERMS)
    active_receiver = _active_receiver_motion(semantic_text) or (
        "lower" in semantic_text and receiver and "non-contact" not in semantic_text
    )
    coarse_motion = _contains_any(semantic_text, _COARSE_MOTION_TERMS)
    retract = _contains_any(semantic_text, _RETRACT_TERMS)
    preengagement = _contains_any(semantic_text, _PREENGAGEMENT_TERMS)
    constrained_feature = _contains_any(
        f"{semantic_text} {feature_role_text}",
        _CONSTRAINED_FEATURE_TERMS,
    )

    size_m, size_name = _referenced_size(
        constraint,
        semantic_text=reference_text,
        object_sizes=object_sizes,
    )
    size_note = f" from {size_name} size" if size_name else ""

    tool_pose = args.get("controlled_entity") == "robot_frame"
    final_object = not tool_pose and _contains_any(semantic_text, (
        "final", "remain", "stay", "supported", "seated", "upright",
    ))
    if retract and not final_object:
        return POSITION_FREE_SPACE_M, "release/retract free-space tolerance"

    if phase == "release" and tool_pose:
        if op == "freeze_position" and stage_has_open_gripper:
            if stage_secured_release:
                return POSITION_FREE_SPACE_M, "secured-result tool release guard"
            return POSITION_PRECONTACT_M, "near-object release hold"
        return POSITION_FREE_SPACE_M, "release-stage free-space tolerance"

    if op in {"reach_point", "reach_distance"}:
        distance_m = _absolute_numeric_arg(args, "distance_m")
        zero_target = _offset_is_zero(args) or (
            op == "reach_distance" and distance_m is not None and distance_m <= 1e-9
        )
        controlled_entity = str(args.get("controlled_entity") or "")
        if high_effort or (phase == "approach" and zero_target):
            value = _size_limited_position_tolerance(
                size_m,
                unknown=(POSITION_CONTACT_DEFAULT_M if constrained_feature else POSITION_PRECONTACT_M),
            )
            return value, f"first-contact size rule{size_note}"
        if phase == "approach" or preengagement:
            return POSITION_PRECONTACT_M, "specific pre-contact setup tolerance"
        if coarse_motion and not active_receiver:
            return POSITION_FREE_SPACE_M, "broad free-space waypoint tolerance"
        if controlled_entity == "gripped_object" and (active_receiver or (receiver and zero_target)):
            receiver_size, receiver_name = _referenced_size(
                constraint,
                semantic_text=reference_text,
                object_sizes=object_sizes,
                preferred_origin=True,
            )
            value = _size_limited_position_tolerance(
                receiver_size,
                unknown=(POSITION_CONTACT_DEFAULT_M if _contains_any(
                    f"{semantic_text} {receiver_role_text}", ("narrow", "fit-sensitive", "slender")
                ) else POSITION_PRECONTACT_M),
            )
            receiver_note = f" from {receiver_name} size" if receiver_name else ""
            return value, f"receiver/final-placement rule{receiver_note}"
        if controlled_entity == "gripped_object":
            return POSITION_FREE_SPACE_M, "broad free-space waypoint tolerance"
        return POSITION_PRECONTACT_M, "ordinary position-target tolerance"

    if op == "directional_move":
        if coarse_motion and not active_receiver:
            return POSITION_FREE_SPACE_M, "coarse free-space directional-motion tolerance"
        if active_receiver:
            receiver_size, receiver_name = _referenced_size(
                constraint,
                semantic_text=reference_text,
                object_sizes=object_sizes,
                preferred_origin=True,
            )
            value = _size_limited_position_tolerance(
                receiver_size,
                unknown=(POSITION_CONTACT_DEFAULT_M if _contains_any(
                    f"{semantic_text} {receiver_role_text}", ("narrow", "fit-sensitive", "slender")
                ) else POSITION_PRECONTACT_M),
            )
            receiver_note = f" from {receiver_name} size" if receiver_name else ""
            return value, f"receiver/engagement directional-motion rule{receiver_note}"
        distance_m = _absolute_numeric_arg(args, "travel_distance_m")
        if distance_m is None:
            return POSITION_FREE_SPACE_M, "directional-motion fallback tolerance"
        value = _clamp(0.20 * distance_m, POSITION_MIN_M, POSITION_FREE_SPACE_M)
        return value, "20% directional-progress rule"

    if op == "freeze_position":
        free_hold = _contains_any(semantic_text, ("free space", "free-space")) and not high_effort
        if free_hold and not active_receiver and not final_object:
            return POSITION_FREE_SPACE_M, "free-space held-object pose-hold tolerance"
        compliant_bar = tool_pose and _contains_any(feature_role_text, ("bar", "protrusion", "protruding"))
        fit_text = re.sub(r"\bnot (?:a |an )?centered fit\b", "", semantic_text)
        centered = _contains_any(fit_text, ("centered fit", "centered axis", "pivot", "press", "twist"))
        if compliant_bar and not centered and (stage_has_grasp or high_effort):
            value = POSITION_CONTACT_DEFAULT_M if size_m is None else _clamp(size_m, POSITION_MIN_M, POSITION_PRECONTACT_M)
            return value, f"compliant protrusion tool-hold rule{size_note}"
        if stage_has_grasp or high_effort or final_object:
            value = _size_limited_position_tolerance(
                size_m,
                unknown=(
                    POSITION_CONTACT_DEFAULT_M
                    if high_effort or constrained_feature
                    else POSITION_PRECONTACT_M
                ),
            )
            return value, f"active grasp/contact pose-hold rule{size_note}"
        return POSITION_FREE_SPACE_M, "ordinary free-space pose-hold tolerance"

    return POSITION_PRECONTACT_M, "default metric-position tolerance"


def _angular_tolerance(
    constraint: dict[str, Any],
    *,
    phase: str,
    semantic_text: str,
    receiver_role_text: str,
    object_sizes: dict[str, float],
    reference_text: str,
    stage_has_grasp: bool,
) -> tuple[float, str]:
    op = str(constraint.get("op") or "")
    args_raw = constraint.get("args")
    args = args_raw if isinstance(args_raw, dict) else {}
    conditions = _conditions(constraint)
    high_effort = "manipulator_high_effort" in conditions
    receiver = _contains_any(f"{semantic_text} {receiver_role_text}", _RECEIVER_TERMS)
    active_receiver = _active_receiver_motion(semantic_text) or (
        "lower" in semantic_text and receiver and "non-contact" not in semantic_text
    )
    rotation_critical = _contains_any(semantic_text, _ROTATION_CRITICAL_TERMS)
    coarse_motion = _contains_any(semantic_text, _COARSE_MOTION_TERMS)
    retract = _contains_any(semantic_text, _RETRACT_TERMS)

    if (phase == "release" or retract) and args.get("controlled_entity") == "robot_frame":
        return ANGULAR_FREE_SPACE_RAD, "release/retract angular tolerance"

    if op == "directional_rotate":
        precision_rotation = _contains_any(semantic_text, _ROTATION_PROGRESS_TERMS)
        if not (precision_rotation or active_receiver or high_effort):
            return ANGULAR_FREE_SPACE_RAD, "coarse free-space rotation tolerance"
        angle_rad = _absolute_numeric_arg(args, "rotation_angle_rad")
        if angle_rad is None:
            return ANGULAR_PRECISE_RAD, "directional-rotation fallback tolerance"
        value = _clamp(0.20 * angle_rad, ANGULAR_PRECISE_RAD, ANGULAR_FREE_SPACE_RAD)
        return value, "20% directional-rotation rule"

    if phase == "approach":
        return ANGULAR_FREE_SPACE_RAD, "non-contact/contact-setup angular tolerance"

    if coarse_motion and not active_receiver:
        return ANGULAR_FREE_SPACE_RAD, "coarse-motion angular tolerance"

    if (receiver or active_receiver) and phase == "manipulate":
        return ANGULAR_PRECISE_RAD, "receiver/engagement angular tolerance"
    if rotation_critical:
        return ANGULAR_PRECISE_RAD, "actuation/final-orientation angular tolerance"

    size_m, _size_name = _referenced_size(
        constraint,
        semantic_text=reference_text,
        object_sizes=object_sizes,
    )
    slender = (size_m is not None and size_m <= 0.03) or _contains_any(
        semantic_text,
        _SLENDER_TERMS,
    )
    if stage_has_grasp and phase == "manipulate" and slender:
        return ANGULAR_PRECISE_RAD, "small/slender grasp angular tolerance"

    return ANGULAR_FREE_SPACE_RAD, "ordinary angular-maintenance tolerance"


def derive_automatic_tolerances(
    constraints_raw: list[dict[str, Any]],
    *,
    atomic_action: str,
    stage_reasoning: str = "",
    explanations: Iterable[Any] = (),
    detectable_objects: Any = None,
    profile: str = "learning",
) -> tuple[dict[int, dict[str, float]], dict[int, dict[str, str]]]:
    """Derive operator-owned tolerance fields without consulting authored values.

    Operator ownership determines the required keys, even when the received
    geometry contains no tolerance fields. LLM-authored values are never read.
    """

    if not isinstance(constraints_raw, list) or not constraints_raw:
        raise ValueError("constraints_raw must be a non-empty list")
    if profile not in {"mpc", "learning"}:
        raise ValueError(f"Unknown tolerance profile: {profile!r}")

    explanation_list = [str(value) for value in explanations]
    stage_text = summarize_atomic_action(atomic_action)
    # The authored reasoning can contain stale numeric thresholds and future
    # stage discussion. Keep it out of classification just as it is kept out
    # of the LLM weight query; the action headline and structured operator are
    # the stable semantic contract.
    _ = stage_reasoning
    phase = _stage_phase(atomic_action)
    object_sizes = parse_detectable_object_sizes(detectable_objects)
    object_descriptions = _detectable_object_descriptions(detectable_objects)
    stage_ops = {
        str(constraint.get("op") or "")
        for constraint in constraints_raw
        if isinstance(constraint, dict)
    }
    stage_has_grasp = "object_grasped" in stage_ops
    stage_has_open_gripper = "open_gripper" in stage_ops
    full_action_text = _semantic_text(str(atomic_action), object_descriptions)
    mentioned_in_action = _mentioned_names(str(atomic_action), object_descriptions)
    # Whole words distinguish supported/secured from unsupported/unsecured.
    stage_secured_release = any(
        re.search(rf"\b{re.escape(term)}\b", full_action_text)
        for term in _SECURED_RELEASE_TERMS
    ) or any(
        name in mentioned_in_action
        and _contains_any(_semantic_text(description, object_descriptions), _SECURED_ROLE_TERMS)
        for name, description in object_descriptions.items()
    )

    parameters: dict[int, dict[str, float]] = {}
    decisions: dict[int, dict[str, str]] = {}
    for index, constraint in enumerate(constraints_raw):
        if not isinstance(constraint, dict):
            raise TypeError(f"constraint {index} must be a dict")
        args = constraint.get("args")
        if not isinstance(args, dict):
            raise TypeError(f"constraint {index} args must be a dict")
        op = constraint.get("op")
        if op not in OP_TOLERANCE_FIELDS:
            raise ValueError(f"Unknown constraint operator: {op!r}")
        field = OP_TOLERANCE_FIELDS[op]
        if field is None:
            continue

        component_text = stage_text
        if index < len(explanation_list):
            component_text = f"{stage_text} {strip_tolerance_annotation(explanation_list[index])}"
        reference_text = component_text
        component_text = _semantic_text(component_text, object_descriptions)
        args = constraint["args"]
        referenced_names = {
            name
            for name in (args.get("controlled_target"), args.get("reference_origin"))
            if isinstance(name, str) and name in object_descriptions
        }
        referenced_names.update(_mentioned_names(reference_text, object_descriptions))
        feature_role_text = " ".join(
            _semantic_text(object_descriptions[name], object_descriptions) for name in sorted(referenced_names)
        )
        receiver_role_text = " ".join(
            _semantic_text(object_descriptions[name], object_descriptions)
            for name in sorted(referenced_names)
            if _is_receiver_description(object_descriptions[name])
        )

        patch: dict[str, float] = {}
        component_decisions: dict[str, str] = {}
        if field == "tolerance_m":
            value, rule = _position_tolerance(
                constraint,
                phase=phase,
                semantic_text=component_text,
                receiver_role_text=receiver_role_text,
                feature_role_text=feature_role_text,
                object_sizes=object_sizes,
                reference_text=reference_text,
                stage_has_grasp=stage_has_grasp,
                stage_has_open_gripper=stage_has_open_gripper,
                stage_secured_release=stage_secured_release,
            )
            if profile == "mpc":
                # Original formulation defaults: contact/pose acquisition at
                # 1 cm, non-contact reach at 2 cm. Widen only for learning.
                zero_target = _offset_is_zero(args) or (
                    op == "reach_distance" and _absolute_numeric_arg(args, "distance_m") == 0.0
                )
                cap = (
                    MPC_POSITION_PRECONTACT_M
                    if op in {"reach_point", "reach_distance"} and not zero_target
                    else MPC_POSITION_PRECISE_M
                )
                value = min(value, cap)
                rule = f"MPC precision cap; {rule}"
            patch["tolerance_m"] = _rounded(value)
            component_decisions["tolerance_m"] = rule
        if field == "tolerance_rad":
            value, rule = _angular_tolerance(
                constraint,
                phase=phase,
                semantic_text=component_text,
                receiver_role_text=receiver_role_text,
                object_sizes=object_sizes,
                reference_text=reference_text,
                stage_has_grasp=stage_has_grasp,
            )
            if profile == "mpc":
                value, rule = MPC_ANGULAR_RAD, f"MPC angular default; {rule}"
            patch["tolerance_rad"] = _rounded(value)
            component_decisions["tolerance_rad"] = rule
        parameters[int(index)] = patch
        decisions[int(index)] = component_decisions

    return parameters, decisions


def geometry_for_prompt(payload: dict[str, Any]) -> dict[str, Any]:
    """Hide application-owned thresholds and provenance in correction queries."""
    if not isinstance(payload, dict):
        raise TypeError("Constraint response must be a JSON object")
    clean = deepcopy(payload)
    clean.pop("TolerancePolicy", None)
    constraints = clean.get("Constraints")
    # An initial parse failure may leave no valid previous program. Feedback
    # must still be constructible; derive_automatic_tolerances validates newly
    # received programs before either compiler can see them.
    if not isinstance(constraints, list):
        return clean
    for constraint in constraints:
        if not isinstance(constraint, dict):
            continue
        args = constraint.get("args")
        if isinstance(args, dict):
            for key in _TOLERANCE_KEYS:
                args.pop(key, None)
    return clean


def receive_constraint_geometry(
    payload: dict[str, Any], *, atomic_action: str,
    detectable_objects: Any, profile: str = "mpc",
) -> dict[str, Any]:
    """Normalize freshly received LLM geometry before either code generator.

    Stored legacy programs are not passed here on replay. This function owns
    only tolerance fields and their provenance; it preserves geometry/order.
    """
    if not isinstance(payload, dict):
        raise TypeError("Constraint response must be a JSON object")
    received = geometry_for_prompt(payload)
    constraints = received.get("Constraints")
    parameters, decisions = derive_automatic_tolerances(
        constraints, atomic_action=atomic_action, detectable_objects=detectable_objects,
        profile=profile,
    )
    for index, patch in parameters.items():
        constraints[index]["args"].update(patch)
    received["TolerancePolicy"] = {
        "version": POLICY_VERSION, "profile": profile, "source": "receiver",
        "decisions": decisions,
    }
    return received


def validate_tolerance_policy_metadata(metadata: dict[str, Any]) -> None:
    """Reject stale automatic artifacts; explicit unstamped user files remain valid."""
    policy = metadata.get("automatic_tolerance_policy")
    if "automatic_tolerance_policy" in metadata and (not isinstance(policy, dict) or policy.get("version") != POLICY_VERSION):
        raise ValueError("Automatic tolerance policy changed; regenerate parameter/weight artifacts.")


def assign_automatic_tolerances(task_info: dict[str, Any]) -> None:
    """Overwrite every stage parameter map in a collected task-info payload."""

    stages = task_info.get("stages")
    if not isinstance(stages, dict):
        raise RuntimeError("task_info is missing stage metadata")
    detectable_objects = task_info.get("detectable_objects")
    for stage_id, stage_meta in stages.items():
        if not isinstance(stage_meta, dict):
            raise TypeError(f"stage {stage_id} metadata must be a dict")
        constraints_raw = stage_meta.get("constraints_raw")
        explanations = stage_meta.get("explanations")
        if not isinstance(constraints_raw, list) or not constraints_raw:
            raise ValueError(f"stage {stage_id} constraints_raw must be a non-empty list")
        if not isinstance(explanations, list):
            raise TypeError(f"stage {stage_id} explanations must be a list")
        parameters, decisions = derive_automatic_tolerances(
            constraints_raw,
            atomic_action=str(stage_meta.get("atomic_action") or ""),
            stage_reasoning=str(stage_meta.get("reasoning") or ""),
            explanations=explanations,
            detectable_objects=detectable_objects,
        )
        stage_meta["parameters"] = parameters
        stage_meta["tolerance_source"] = "automatic"
        stage_meta["tolerance_decisions"] = decisions


def policy_metadata() -> dict[str, Any]:
    """Return a JSON-serializable description stored with sweep artifacts."""

    return {
        "version": POLICY_VERSION,
        "mpc": {
            "position_precision_cap_m": MPC_POSITION_PRECISE_M,
            "nonzero_reach_cap_m": MPC_POSITION_PRECONTACT_M,
            "angular_rad": MPC_ANGULAR_RAD,
        },
        "learning_profile": "bounded_broadening",
        "position_m": {
            "minimum": POSITION_MIN_M,
            "contact_default": POSITION_CONTACT_DEFAULT_M,
            "precontact": POSITION_PRECONTACT_M,
            "free_space": POSITION_FREE_SPACE_M,
            "directional_progress_fraction": 0.20,
        },
        "angular_rad": {
            "precise": ANGULAR_PRECISE_RAD,
            "free_space": ANGULAR_FREE_SPACE_RAD,
            "directional_progress_fraction": 0.20,
        },
    }
