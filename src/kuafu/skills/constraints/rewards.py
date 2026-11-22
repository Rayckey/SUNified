import torch
from typing import Union, Optional

# <utility>
def quad_norm(x: torch.Tensor) -> torch.Tensor:
    """
    << >> 
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
    return 0.5 * torch.sum(x**2, dim=-1, keepdim=True)

# <end-of-utility>

# <utility>
def _keyify_offset(offset) -> tuple:
    """<< Convert small (3,) offsets into a stable, hashable tuple key.>>"""
    if isinstance(offset, torch.Tensor):
        off = offset.detach().flatten().tolist()
    elif isinstance(offset, (list, tuple)):
        off = list(offset)
    else:
        off = [offset]
    return tuple(float(x) for x in off)
# <end-of-utility>

# <utility>
def _keyify_scaler(x: Union[float, torch.Tensor]) -> float:
    """<<Stable float key for memoization (matches residuals_torch.py).>>"""
    if isinstance(x, torch.Tensor):
        return float(x.detach().flatten()[0].item())
    return float(x)

# <end-of-utility>

# <utility>
def _sigmoid(x: torch.Tensor, param: float) -> torch.Tensor:
    """<< Sigmoid function for reward shaping.>>"""
    return x ** 2 / (x ** 2 + param ** 2)
# <end-of-utility>


# <residual>
def free_move_residual(
    state: torch.Tensor,
    controlled_entity: str,
    controlled_target: str,
    reference_basis: str,
    reference_origin: str,
    offset_from_origin_xyz: torch.Tensor,
    tolerance_m: float
) -> torch.Tensor:
    """
    << >> 
    """
    # Back-compat alias: free_move is now reach_point.
    return reach_point_residual(
        state=state,
        controlled_entity=controlled_entity,
        controlled_target=controlled_target,
        reference_basis=reference_basis,
        reference_origin=reference_origin,
        offset_from_origin_xyz=offset_from_origin_xyz,
        tolerance_m=tolerance_m,
    )

# <end-of-residual>


# <residual>
def reach_point_residual(
    state: torch.Tensor,
    controlled_entity: str,
    controlled_target: str,
    reference_basis: str,
    reference_origin: str,
    offset_from_origin_xyz: torch.Tensor,
    tolerance_m: float,
) -> torch.Tensor:
    """
    <<Reward shaping for reach_point: uses memoized distance-to-point.>>
    """

    e = memo(
        (
            "reach_point",
            controlled_entity,
            controlled_target,
            reference_basis,
            reference_origin,
            _keyify_offset(offset_from_origin_xyz),
        )
    )
    res = _sigmoid(e, 0.5)
    return res.flatten()


# <end-of-residual>


# <residual>
def reach_distance_residual(
    state: torch.Tensor,
    controlled_entity: str,
    controlled_target: str,
    reference_basis: str,
    reference_origin: str,
    offset_from_origin_xyz: torch.Tensor,
    distance_m: float,
    tolerance_m: float,
) -> torch.Tensor:
    """
    <<Reward shaping for reach_distance: uses memoized |dist - distance_m|.>>
    """

    e = memo(
        (
            "reach_distance",
            controlled_entity,
            controlled_target,
            reference_basis,
            reference_origin,
            _keyify_offset(offset_from_origin_xyz),
            _keyify_scaler(distance_m),
        )
    )
    res = _sigmoid(e, 0.5)
    return res.flatten()

# <end-of-residual>


# <residual>
def directional_move_residual(
    state: torch.Tensor,
    controlled_entity: str,
    controlled_target: str,
    reference_basis: str,
    reference_origin: str,
    axis: str,
    travel_distance_m: float,
    tolerance_m: float 
) -> torch.Tensor:
    """
    << >> 
    """

    dist = memo(("directional_move",       controlled_entity,
    controlled_target,
    reference_basis,
    reference_origin,
    axis,
    _keyify_scaler(travel_distance_m)))

    res = _sigmoid(dist, travel_distance_m/2.0)

    return res.flatten()

# <end-of-residual>


# <residual>
def axis_align_residual(
    state: torch.Tensor,
    controlled_entity: str,
    controlled_target: str,
    reference_basis: str,
    reference_origin: str,
    controlled_axis: str,
    reference_axis: str,
    alignment: str,
    tolerance_rad: float
) -> torch.Tensor:
    """
    << >> 
    """
    alignment = alignment.strip().lower()
    err =   memo(("axis_align",   controlled_entity,
    controlled_target,
    reference_basis,
    reference_origin,
    controlled_axis,
    reference_axis,
    alignment))

    if alignment == "orthogonal":
        res = torch.abs(err)

    elif alignment in {"codirectional", "antiparallel", "bipolar"}:
        res = err/2.0
    else:
        raise ValueError(f"Unsupported alignment mode '{alignment}'")

    res = _sigmoid(res, 0.25)

    return res.flatten()

# <end-of-residual>


# <residual>
def point_toward_residual(
    state: torch.Tensor,
    controlled_entity: str,
    controlled_target: str,
    controlled_axis: str,
    reference_basis: str,
    reference_origin: str,
    offset_from_origin_xyz: torch.Tensor, # (B, 3) expected if passed from batched context
    tolerance_rad: float
) -> torch.Tensor:
    """
    << >> 
    """
    if controlled_entity != "robot_frame":
        raise ValueError(f"Only robot_frame is supported in controlled_entity for point_toward")

    err = memo(("point_toward",         controlled_entity,
        controlled_target,
        controlled_axis,
        reference_basis,
        reference_origin,
    _keyify_offset(offset_from_origin_xyz)))
    _ = tolerance_rad
    res = err/2.0

    res = _sigmoid(res, 0.25)

    return res.flatten()

# <end-of-residual>


# <residual>
def directional_rotate_residual(
    state: torch.Tensor,
    controlled_entity: str,
    controlled_target: str,
    reference_basis: str,
    reference_origin: str,
    rotation_axis: str,
    rotation_angle_rad: float,
    tolerance_rad: float
) -> torch.Tensor:
    """
    << >> 
    """

    c = memo(("directional_rotate",     controlled_entity,
    controlled_target,
    reference_basis,
    reference_origin,
    rotation_axis,
    _keyify_scaler(rotation_angle_rad)))

    err = torch.linalg.norm(c, dim=-1)
    res = _sigmoid(err, 0.25)

    return res.flatten()

# <end-of-residual>


# <residual>
def close_gripper_residual(
    state: torch.Tensor
) -> torch.Tensor:
    """
    << >> 
    """
    gripper_state = state # (B, 2)
    device = state.device
    target = torch.tensor([0.0, 0.0], device=device).unsqueeze(0)  # (1, 2)
    res = _sigmoid(torch.norm(gripper_state - target, dim=-1), 1.0)

    return res.flatten()

# <end-of-residual>


# <residual>
def open_gripper_residual(
    state: torch.Tensor
) -> torch.Tensor:
    """
    << >> 
    """
    gripper_state = state  # (B, 2)
    target = torch.tensor([1.0, 1.0], device=DEVICE).unsqueeze(0)  # (1, 2)
    res = _sigmoid(torch.norm(gripper_state - target, dim=-1), 1.0)

    return res.flatten()

# <end-of-residual>


# <residual>
def freeze_position_residual(
    state: torch.Tensor,
    controlled_entity: str,
    controlled_target: str,
    tolerance_m: float
) -> torch.Tensor:
    """
    << >> 
    """

    error = memo(("freeze_position", controlled_entity, controlled_target))

    res = _sigmoid(torch.norm(error, dim=-1), 0.25)

    return res.flatten()

# <end-of-residual>


# <residual>
def freeze_orientation_residual(
    state: torch.Tensor,
    controlled_entity: str,
    controlled_target: str,
    tolerance_rad: float
) -> torch.Tensor:
    """
    << >> 
    """
    error = memo(("freeze_orientation", controlled_entity, controlled_target))
    # `pred_freeze_orientation()` memoizes one angular error per environment, so
    # this path must preserve that batch axis instead of reducing across it.
    res = _sigmoid(error.abs().view(-1), 0.25)

    return res.flatten()

# <end-of-residual>
