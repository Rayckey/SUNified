from __future__ import annotations

from typing import Any, List, Mapping, Sequence


def _name(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _has_nonzero_rotation(args: Mapping[str, Any]) -> bool:
    angle = args.get("rotation_angle_rad")
    try:
        return abs(float(angle)) > 1e-9
    except (TypeError, ValueError):
        return True


def _condition_tags(condition: Any) -> List[str]:
    if isinstance(condition, str):
        return [condition.strip().lower()] if condition.strip() else []
    if isinstance(condition, Sequence) and not isinstance(condition, (str, bytes)):
        return [
            value.strip().lower()
            for value in condition
            if isinstance(value, str) and value.strip()
        ]
    return []


def find_manipulator_effort_condition_conflicts(
    constraints: Sequence[Mapping[str, Any]],
) -> List[str]:
    """Return errors when one stage mixes contact-expected and non-contact effort tags."""
    high_hits: List[int] = []
    low_hits: List[int] = []

    for idx, constraint in enumerate(constraints):
        if not isinstance(constraint, Mapping):
            continue
        op = constraint.get("op")
        tags = set(_condition_tags(constraint.get("condition")))
        if isinstance(op, str):
            tags.add(op.strip().lower())

        if "manipulator_high_effort" in tags:
            high_hits.append(idx)
        if "manipulator_low_effort" in tags:
            low_hits.append(idx)

    if not high_hits or not low_hits:
        return []

    return [
        "Conflicting manipulator effort conditions in the same stage: "
        f"manipulator_high_effort appears at constraint indices {high_hits}, "
        f"while manipulator_low_effort appears at constraint indices {low_hits}. "
        "A single atomic action must decide whether contact/high effort is expected; "
        "correct the atomic action so it uses exactly one of these effort conditions."
    ]


def find_directional_rotate_axis_align_conflicts(
    constraints: Sequence[Mapping[str, Any]],
) -> List[str]:
    """
    Return errors for stages that command a finite rotation while also holding
    an axis alignment involving that rotating frame.

    `directional_rotate` is a stage-level orientation goal. An `axis_align` on
    the same frame, or one that uses the rotating frame as the live reference,
    asks the optimizer to maintain an orientation relation at the same time.
    That tends to overconstrain door/handle rotation stages, so force the LLM
    to separate pre-rotation alignment from the rotation itself.
    """
    rotate_terms: List[tuple[int, Mapping[str, Any], str]] = []
    axis_terms: List[tuple[int, Mapping[str, Any]]] = []

    for idx, constraint in enumerate(constraints):
        if not isinstance(constraint, Mapping):
            continue
        op = constraint.get("op")
        args = constraint.get("args", {})
        if not isinstance(args, Mapping):
            continue

        if op == "directional_rotate" and _has_nonzero_rotation(args):
            rotating_target = _name(args.get("controlled_target"))
            if rotating_target is not None:
                rotate_terms.append((idx, args, rotating_target))
        elif op == "axis_align":
            axis_terms.append((idx, args))

    errors: List[str] = []
    for rotate_idx, rotate_args, rotating_target in rotate_terms:
        for axis_idx, axis_args in axis_terms:
            axis_controlled_target = _name(axis_args.get("controlled_target"))
            axis_reference_origin = _name(axis_args.get("reference_origin"))
            if (
                axis_controlled_target != rotating_target
                and axis_reference_origin != rotating_target
            ):
                continue

            errors.append(
                "Conflicting orientation constraints at indices "
                f"{rotate_idx} and {axis_idx}: directional_rotate rotates "
                f"controlled_target={rotating_target!r} by "
                f"rotation_angle_rad={rotate_args.get('rotation_angle_rad')!r}, "
                "but axis_align simultaneously maintains an axis relation "
                "involving that rotating frame. Put any approach/pre-grasp "
                "axis_align in an earlier stage, then keep only directional_rotate "
                "for the rotation stage."
            )

    return errors
