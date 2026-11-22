"""Decision-time SUN stage labels for demonstration supervision."""

import torch

STAGE_TIMING = "pre_action"


def snapshot_decision_stage(env, stage=None):
    """Clone the teacher stage before stepping; -1 denotes unavailable supervision."""
    raw = env.unwrapped
    if stage is None:
        stage = getattr(raw, "current_stage", None)
    count = None
    runtime = getattr(raw, "program_runtime", None)
    program = getattr(runtime, "compiled_program", None)
    if program is not None:
        count = int(program.stage_count())
    if stage is None:
        labels = torch.full((raw.num_envs,), -1, device=raw.device, dtype=torch.long)
    else:
        values = torch.as_tensor(stage, device=raw.device).reshape(-1)
        if values.numel() == 1:
            values = values.expand(raw.num_envs)
        if values.numel() != raw.num_envs or not torch.isfinite(values).all():
            raise ValueError("Decision stages must contain one finite integer per environment")
        if not torch.equal(values, values.long().to(values.dtype)):
            raise ValueError("Decision stages must be integers")
        labels = values.long().clone()
    valid = labels >= 0
    if count is not None:
        valid &= labels < count
    labels[~valid] = -1
    return labels, valid, count


def program_stage_schema(env):
    """Return the ordered semantic stage definitions used by the teacher monitor."""
    runtime = getattr(env.unwrapped, "program_runtime", None)
    program = getattr(runtime, "compiled_program", None)
    if program is None:
        return None
    return [{"name": stage.name, "constraints": stage.constraints.get("Constraints", [])}
            for stage in program._program_stages]
