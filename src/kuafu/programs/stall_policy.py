"""Small, simulator-independent helpers for learn-programs stall policy."""

from __future__ import annotations


def _as_bool(value) -> bool:
    """Collapse scalar or batched tensor-like values to one Python boolean."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "any"):
        value = value.any()
    if hasattr(value, "item"):
        value = value.item()
    return bool(value)


def evaluate_controller_stall(program_runtime, *, enabled: bool) -> tuple[bool, str | None]:
    """Evaluate a controller stall only when the task explicitly enables it.

    Returning early while disabled is important: ``hasStalled`` mutates its
    internal counters, so even a diagnostic call would change controller state.
    """
    if not enabled:
        return False, None

    stalled = _as_bool(program_runtime.hasStalled())
    if not stalled:
        return False, None
    return True, program_runtime.stallFailureReason()
