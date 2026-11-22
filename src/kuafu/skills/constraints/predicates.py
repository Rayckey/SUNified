from typing import Union, TYPE_CHECKING, Any, Callable
import torch

# Robot-specific gripper predicate constants. APIWrapper injects these into
# generated predicate scopes; module defaults preserve non-Kinova behavior.
GRIPPER_MAX_FORCE = 30.0
GRIPPER_MAX_GRASP_OPENNESS = 0.8
ROBOT_GRIPPER_NAME = "contact_sensor"

# <utility>
def _get_controlled_position_snapshot_world(
    controlled_entity: str,
    controlled_target: str,
) -> torch.Tensor:
    """
    << >> 
    """
    controlled_entity = controlled_entity.strip().lower()

    if controlled_entity == "robot_frame":
        # e.g. TCP frame
        return lookup_robot_frame_position(controlled_target, "current")  # (B×3)

    elif controlled_entity == "gripped_object":
        # Use object position API for the grasped object
        return get_object_xyz_position(controlled_target, "current")  # (B×3)

    else:
        raise ValueError(f"Unsupported controlled_entity '{controlled_entity}'")

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
def _normalize_axis(v: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """<<Match residuals_torch.py: safe axis normalization.>>"""
    nrm_sq = torch.sum(v * v, dim=-1, keepdim=True) + eps
    nrm = torch.sqrt(nrm_sq)
    return v / nrm
# <end-of-utility>

# <utility>
def _world_axis_from_label(
    axis: str
) -> torch.Tensor:
    """<<Match residuals_torch.py: batched world axis basis.>>"""
    axis = axis.strip()
    if axis == "+x":
        base = torch.tensor([1.0, 0.0, 0.0], device=DEVICE)
    elif axis == "-x":
        base = torch.tensor([-1.0, 0.0, 0.0], device=DEVICE)
    elif axis == "+y":
        base = torch.tensor([0.0, 1.0, 0.0], device=DEVICE)
    elif axis == "-y":
        base = torch.tensor([0.0, -1.0, 0.0], device=DEVICE)
    elif axis == "+z":
        base = torch.tensor([0.0, 0.0, 1.0], device=DEVICE)
    elif axis == "-z":
        base = torch.tensor([0.0, 0.0, -1.0], device=DEVICE)
    else:
        raise ValueError(f"Unsupported axis label '{axis}'")
    return base.unsqueeze(0)
# <end-of-utility>

# <utility>
def _ensure_batch(t: torch.Tensor) -> torch.Tensor:
    """
    <<Ensure input tensor has a leading batch dimension.

    If input is (n,), return (1, n). Otherwise return unchanged.>>
    """
    if t.dim() == 1:
        return t.unsqueeze(0)
    return t
# <end-of-utility>

# <utility>
def _target_world_from_offset(
    reference_basis: str,
    reference_origin: str,
    offset_from_origin_xyz: list,

) -> torch.Tensor:
    """<<Shared memoized target world point; matches residuals_torch.py.>>"""
    reference_basis_n = reference_basis

    offset_local = _broadcast_offset(offset_from_origin_xyz)
    if reference_basis_n == "world":
        root_pos = get_object_xyz_position(reference_origin, "real time")
        return root_pos + offset_local

    if reference_basis_n == "object":
        return object_to_world_xyz_position(reference_origin, offset_local, "real time")

    if reference_basis_n == "robot_frame":
        root_pos = lookup_robot_frame_position(reference_origin, "current")
        q_root = lookup_robot_frame_rotation(reference_origin, "current")
        target_world = quat_multiply_pos(q_root, offset_local) + root_pos
        return target_world

    raise ValueError(f"Unknown reference_basis: {reference_basis}")


# <end-of-utility>

# <utility>
def _get_reference_axis_world(
    basis: str, root: str, axis: str, time_stamp: str = "real time"
) -> torch.Tensor:
    """
    << >> 
    """
    basis = basis.strip().lower()

    if basis == "world":
        return _world_axis_from_label(axis)
    if basis == "object":
        q_obj = get_object_wxyz_quaternion(root, time_stamp)  # (B×4)
        return _axis_from_quat_label(q_obj, axis) # (B×3)
    if basis == "robot_frame":
        q_frame = lookup_robot_frame_rotation(root, time_stamp)  # (B×4)
        return _axis_from_quat_label(q_frame, axis)  # (B×3)
    raise ValueError(f"Unsupported reference_basis '{basis}'")

# <end-of-utility>


# <utility>
def _axis_from_quat_label(q: torch.Tensor, axis: str) -> torch.Tensor:
    """
    << >> 
    """
    sign = 1.0
    core = axis

    if axis.startswith("+") or axis.startswith("-"):
        if axis.startswith("-"):
            sign = -1.0
        core = axis[1:]

    if core == "x":
        v = x_axis_from_quat(q)
    elif core == "y":
        v = y_axis_from_quat(q)
    elif core == "z":
        v = z_axis_from_quat(q)
    else:
        raise ValueError(f"Unsupported axis component '{axis}'")
    return sign * v  # (B×3)


# <end-of-utility>

# <utility>
def _broadcast_offset(
    offset: Union[torch.Tensor, list, tuple]
) -> torch.Tensor:
    """
    <<Broadcast a 3D offset to match a given batch size.

    Accepts (3,), (batch, 3), list, or tuple and returns (1, 3) or (B, 3)
    on the specified device and dtype.  We intentionally keep constant
    offsets as (1, 3) so that PyTorch broadcasting handles alignment
    with any batch dimension (full or stage-masked).>>
    """

    off = torch.as_tensor(offset, device=DEVICE, dtype=DTYPE)

    if off.ndim == 1:
        if off.shape[0] != 3:
            raise ValueError(f"offset_from_origin_xyz must have length 3, got {tuple(off.shape)}")
        off = off.unsqueeze(0)  # (1,3)

    if off.ndim != 2 or off.shape[1] != 3:
        raise ValueError(f"offset_from_origin_xyz must be (3,) or (B,3), got {tuple(off.shape)}")

    return off

# <end-of-utility>

# <utility>
def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """ << Quaternion multiply for wxyz quaternions. Shapes (...,4).>> """
    q1 = _ensure_batch(q1)
    q2 = _ensure_batch(q2)
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack((w, x, y, z), dim=-1)
# <end-of-utility>


# <utility>
def axis_angle_to_rotvec_torch(
    rotation_axis: str, rotation_angle_rad: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    <<Convert an axis label and rotation angle into a (batched) 3D rotation vector
      (axis * angle).

      rotation_axis : "+x", "-x", "+y", "-y", "+z", "-z"
      rotation_angle_rad : (...,) or (...,1) tensor (radians)
      returns : (...,3) tensor of rotation vectors>>
    """
    a = rotation_axis.strip().lower()

    if a == "+x":
        axis_vec = torch.tensor(
            [1.0, 0.0, 0.0],
            device=DEVICE,
            dtype=DTYPE,
        )
    elif a == "-x":
        axis_vec = torch.tensor(
            [-1.0, 0.0, 0.0],
            device=DEVICE,
            dtype=DTYPE,
        )
    elif a == "+y":
        axis_vec = torch.tensor(
            [0.0, 1.0, 0.0],
            device=DEVICE,
            dtype=DTYPE,
        )
    elif a == "-y":
        axis_vec = torch.tensor(
            [0.0, -1.0, 0.0],
            device=DEVICE,
            dtype=DTYPE,
        )
    elif a == "+z":
        axis_vec = torch.tensor(
            [0.0, 0.0, 1.0],
            device=DEVICE,
            dtype=DTYPE,
        )
    elif a == "-z":
        axis_vec = torch.tensor(
            [0.0, 0.0, -1.0],
            device=DEVICE,
            dtype=DTYPE,
        )
    else:
        raise ValueError(f"Invalid rotation_axis label: {rotation_axis!r}")

    angle = rotation_angle_rad
    rotvec = angle * axis_vec.unsqueeze(0)
    return rotvec

# <end-of-utility>


# <utility>
def axis_angle_to_rotvec(
    rotation_axis: str,
    rotation_angle_rad
) -> torch.Tensor:
    """
    <<Match residuals_torch.py: build (B,3) rotvec = axis * angle.>>
    """
    a = rotation_axis.strip().lower()

    if a == "+x":
        axis_vec = torch.tensor([1.0, 0.0, 0.0], device=DEVICE)
    elif a == "-x":
        axis_vec = torch.tensor([-1.0, 0.0, 0.0], device=DEVICE)
    elif a == "+y":
        axis_vec = torch.tensor([0.0, 1.0, 0.0], device=DEVICE)
    elif a == "-y":
        axis_vec = torch.tensor([0.0, -1.0, 0.0], device=DEVICE)
    elif a == "+z":
        axis_vec = torch.tensor([0.0, 0.0, 1.0], device=DEVICE)
    elif a == "-z":
        axis_vec = torch.tensor([0.0, 0.0, -1.0], device=DEVICE)
    else:
        raise ValueError(f"Invalid rotation_axis label: {rotation_axis!r}")

    axis_vec = axis_vec.unsqueeze(0) #(1,3)

    angle = torch.as_tensor(rotation_angle_rad, device=DEVICE)
    if angle.dim() == 0:
        angle = angle.unsqueeze(0).unsqueeze(0)
    elif angle.dim() == 1:
        angle = angle.unsqueeze(-1)
    return angle * axis_vec


# <end-of-utility>


# <utility>
def _get_gripped_object_rel_quat(controlled_target: str, time_stamp: str) -> torch.Tensor:
    """
    <<Return the constant relative quaternion r_obj_in_ee for a gripped object.

      r_obj_in_ee rotates from end-effector frame to object frame.
      It is computed once at grasp time from observed orientations:

          r_obj_in_ee = quat_conj(q_ee_grasp_world) * q_obj_grasp_world

      where q_ee_grasp_world and q_obj_grasp_world are the end-effector and
      object quaternions in world frame at the moment the object is grasped.

      Implementation detail:
        - In practice this should be a casadi.DM(4×1) parameter looked up by
          controlled_target (e.g., from a small dictionary populated at
          compile time), so that this function is free of side effects and
          returns a Sym.
    >>
    """
    # Placeholder: user code should provide a mapping from object name to
    # a casadi.DM 4×1 relative quaternion r_obj_in_ee.
    # Example (to be defined elsewhere in the module):
    #   _GRIPPED_OBJECT_REL_QUAT = {
    #       "mug": ca.DM([...]),
    #       "cube": ca.DM([...]),
    #   }
    q_ee_grasp_world = get_object_wxyz_quaternion(state_frame, time_stamp) # (4×1)
    q_obj_grasp_world = get_object_wxyz_quaternion(
        controlled_target, time_stamp
    )
    quat_diff = quat_mul(quat_conj(q_ee_grasp_world), q_obj_grasp_world)

    return quat_diff  # (4×1)


# <end-of-utility>

# ============================================================
# Primitive predicates used by the constraint parser / generator
# ============================================================


# <predicate>
def pred_free_move(
    controlled_entity: str,
    controlled_target: str,
    reference_basis: str,
    reference_origin: str,
    offset_from_origin_xyz: torch.Tensor,
    tolerance_m: float
) -> torch.Tensor:
    """
    <<Check if a controlled entity is within a positional tolerance of a target point
    defined by a reference basis and offset.

    The target point is defined by:
        - reference_basis / reference_origin      : which coordinate system and origin to use
        - offset_from_origin_xyz               : 3D offset in that basis (meters)

    The controlled entity is:
        - a robot frame (e.g., TCP), or
        - a gripped object rigidly attached to the gripper.

    Inputs
        controlled_entity : (str)
            "robot_frame" or "gripped_object".
        controlled_target : (str)
            If controlled_entity == "robot_frame": name of a robot frame (e.g. "tool0").
            If controlled_entity == "gripped_object": name of a grasped object.
        reference_basis : (str)
            "world", "object", or "robot_frame".
        reference_origin : (str)
            "world" when reference_basis == "world";
            object name when reference_basis == "object";
            robot frame name when reference_basis == "robot_frame".
        offset_from_origin_xyz : (3,) or (batch, 3)
            Desired offset in meters, expressed in the reference_basis coordinates.
        tolerance_m : (float)
            Maximum allowed Euclidean distance (meters) between the current position
            of the controlled entity and the target point.

    Returns
        (batch, ) torch.bool
            True where the controlled entity is within tolerance of the target point.>>
    """
    # Back-compat alias: free_move is now reach_point.
    return pred_reach_point(
        controlled_entity=controlled_entity,
        controlled_target=controlled_target,
        reference_basis=reference_basis,
        reference_origin=reference_origin,
        offset_from_origin_xyz=offset_from_origin_xyz,
        tolerance_m=tolerance_m,
    )


# <end-of-predicate>


# <predicate>
def pred_reach_point(
    controlled_entity: str,
    controlled_target: str,
    reference_basis: str,
    reference_origin: str,
    offset_from_origin_xyz: torch.Tensor,
    tolerance_m: float
) -> torch.Tensor:
    """
    <<Goal: reach a specific target point (reference_origin + offset_from_origin_xyz in reference_basis)
      within tolerance_m. Path is unconstrained.>>
    """
    controlled_entity_n = controlled_entity

    target_world = _target_world_from_offset(
        reference_basis=reference_basis,
        reference_origin=reference_origin,
        offset_from_origin_xyz=offset_from_origin_xyz
    )

    if controlled_entity_n == "robot_frame":
        pos_world = lookup_robot_frame_position(controlled_target, "real time")
        error = pos_world - target_world
        error_dist = torch.norm(error, dim=-1)  # (B,)

    elif controlled_entity_n == "gripped_object":
        obj_pos = get_object_xyz_position(controlled_target, "real time")
        error = obj_pos - target_world
        error_dist = torch.norm(error, dim=-1)  # (B,)

    else:
        raise ValueError(f"Unsupported controlled_entity '{controlled_entity}'")

    # Memoize the scalar distance-to-point error for residuals/rewards.
    memo(
        (
            "reach_point",
            controlled_entity,
            controlled_target,
            reference_basis,
            reference_origin,
            _keyify_offset(offset_from_origin_xyz),
        ),
        error_dist,
    )

    return (error_dist <= tolerance_m).flatten()


# <end-of-predicate>


# <predicate>
def pred_reach_distance(
    controlled_entity: str,
    controlled_target: str,
    reference_basis: str,
    reference_origin: str,
    offset_from_origin_xyz: Union[torch.Tensor, list],
    distance_m: float,
    tolerance_m: float
) -> torch.Tensor:
    """
    <<Goal: reach any position such that distance(controlled_target, reference_point)
      is distance_m (± tolerance_m). The reference_point is reference_origin + offset_from_origin_xyz
      expressed in reference_basis.>>
    """
    controlled_entity_n = controlled_entity.strip().lower()

    target_world = _target_world_from_offset(
        reference_basis=reference_basis,
        reference_origin=reference_origin,
        offset_from_origin_xyz=offset_from_origin_xyz,
    )  # (B,3)

    if controlled_entity_n == "robot_frame":
        ctrl_pos_world = lookup_robot_frame_position(controlled_target, "real time")  # (B,3)
    elif controlled_entity_n == "gripped_object":
        ctrl_pos_world = get_object_xyz_position(controlled_target, "real time")  # (B,3)
    else:
        raise ValueError(f"Unsupported controlled_entity '{controlled_entity}'")

    dist = torch.norm(ctrl_pos_world - target_world, dim=-1)  # (B,)
    error = torch.abs(dist - float(distance_m))  # (B,)

    memo(
        (
            "reach_distance",
            controlled_entity,
            controlled_target,
            reference_basis,
            reference_origin,
            _keyify_offset(offset_from_origin_xyz),
            _keyify_scaler(distance_m),
        ),
        error,
    )


    return (error <= tolerance_m).flatten()


# <end-of-predicate>


# <predicate>
def pred_directional_move(
    controlled_entity: str,
    controlled_target: str,
    reference_basis: str,
    reference_origin: str,
    axis: str,
    travel_distance_m: float,
    tolerance_m: float
) -> torch.Tensor:
    """
    <<Check if a controlled entity has moved a desired distance along a given axis,
    while staying within a lateral deviation tolerance.

    The motion is measured as displacement between the "current" snapshot and "real time"
    projected onto a chosen axis.

    Inputs
        controlled_entity : (str)
            "robot_frame" or "gripped_object".
        controlled_target : (str)
            Frame or object name, as in pred_free_move.
        reference_basis : (str)
            "world", "object", or "robot_frame".
        reference_origin : (str)
            Root object/frame defining the axis basis, as in pred_free_move.
        axis : (str)
            One of "+x", "-x", "+y", "-y", "+z", "-z" in the reference_basis coordinates.
        travel_distance_m : (float)
            Target displacement along the chosen axis (non-negative, meters).
        tolerance_m : (float)
            Allowed error in axial displacement (meters).

    Returns
        (batch, ) torch.bool
            True where the axial displacement is within tolerance_m.>>
    """


    # Match `directional_move_residual` and threshold with tolerance_m.
    controlled_entity_n = controlled_entity
    if controlled_entity_n not in ("robot_frame", "gripped_object"):
        raise ValueError(f"Unsupported controlled_entity '{controlled_entity}'")

    # 1) World-space axis (uses 'current' snapshot in residual).
    a_hat = _get_reference_axis_world(reference_basis, reference_origin, axis, "current")  # (B,3)

    # 2) Reference snapshot position in world (current).
    pos_ref = _get_controlled_position_snapshot_world(controlled_entity, controlled_target)  # (B,3)

    # 3) Live position consistent with residual's state composition.
    if controlled_entity_n == "gripped_object":
        pos_now = get_object_xyz_position(controlled_target, "real time")  # (B,3)
    else:
        pos_now = lookup_robot_frame_position(controlled_target, "real time")  # (B,3)

    # 4) Decompose displacement.
    disp_world = pos_now - pos_ref  # (B,3)
    d_parallel = torch.sum(disp_world * a_hat, dim=-1, keepdim=True)  # (B,1)
    d_axial = d_parallel - travel_distance_m  # (B,1)
    d_parallel_vec = a_hat * d_parallel  # (B,3)
    d_lateral_vec = disp_world - d_parallel_vec  # (B,3)

    dist = torch.sqrt((d_axial * d_axial).flatten() + (torch.sum(d_lateral_vec * d_lateral_vec, dim=-1))/2 )

    memo(("directional_move",       controlled_entity,
    controlled_target,
    reference_basis,
    reference_origin,
    axis,
    _keyify_scaler(travel_distance_m)), dist)

    return dist <= tolerance_m

# <end-of-predicate>


# <predicate>
def pred_axis_align(
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
    <<Check if a chosen axis of the controlled entity has a specified relation
    to a reference axis within an angular tolerance.

    Inputs
        controlled_entity : (str)
            "robot_frame" or "gripped_object".
        controlled_target : (str)
            Frame or object name being oriented.
        reference_basis : (str)
            "world", "object", or "robot_frame" for the reference axis.
        reference_origin : (str)
            Root defining the reference axis:
                - "world"               when reference_basis == "world"
                - object name           when reference_basis == "object"
                - robot frame name      when reference_basis == "robot_frame"
        controlled_axis : (str)
            Local axis of the controlled entity: "+x", "-x", "+y", "-y", "+z", "-z".
        reference_axis : (str)
            Axis of the reference_origin (in reference_basis) to align with, same format as controlled_axis.
        alignment : (str)
            "codirectional", "antiparallel", "bipolar", or "orthogonal".
        tolerance_rad : (float)
            Maximum allowed angular deviation (radians) from the desired alignment.

    Returns
        (batch, ) torch.bool
            True where the axis relation satisfies the alignment condition within tolerance.>>
    """

    controlled_entity_n = controlled_entity
    if controlled_entity_n == "gripped_object":
        q_for_axis = get_object_wxyz_quaternion(controlled_target, "real time")  # (B,4)
    elif controlled_entity_n == "robot_frame":
        q_for_axis = lookup_robot_frame_rotation(controlled_target, "real time")  # (B,4)
    else:
        raise ValueError(f"Unsupported controlled_entity '{controlled_entity}'")

    a_ctrl_u = _axis_from_quat_label(q_for_axis, controlled_axis)
    a_ref_u = _get_reference_axis_world(reference_basis, reference_origin, reference_axis, "real time")

    if a_ctrl_u.shape[0] != a_ref_u.shape[0] and a_ref_u.shape[0] != 1:
        raise RuntimeError(
            f"pred_axis_align batch mismatch: a_ctrl_u{tuple(a_ctrl_u.shape)} vs a_ref_u{tuple(a_ref_u.shape)}. "
            "This typically indicates a mask/caching inconsistency (e.g., memoized values reused across different "
            "batch masks)."
        )

    alignment_n = alignment.strip().lower()
    if alignment_n == "codirectional":
        dot = torch.sum(a_ctrl_u * a_ref_u, dim=-1, keepdim=True)  # (B,1)
        err = 1.0 - dot
        tol = 1 - torch.cos(torch.tensor(tolerance_rad, device=DEVICE, dtype=DTYPE)).view(1, 1) 


    elif alignment_n == "antiparallel":
        dot = torch.sum(a_ctrl_u * (-a_ref_u), dim=-1, keepdim=True)  # (B,1)
        err = 1.0 - dot
        tol = 1 - torch.cos(torch.tensor(tolerance_rad, device=DEVICE, dtype=DTYPE)).view(1, 1) 


    elif alignment_n == "bipolar":
        dot = torch.sum(a_ctrl_u * a_ref_u, dim=-1, keepdim=True)  # (B,1)
        err = 1.0 - torch.abs(dot)
        tol = 1 - torch.cos(torch.tensor(tolerance_rad, device=DEVICE, dtype=DTYPE)).view(1, 1)


    elif alignment_n == "orthogonal":
        delta = 1e-9
        c = torch.sum(a_ctrl_u * a_ref_u, dim=-1, keepdim=True)  # (B,1)
        err = torch.sqrt(c * c + (delta * delta))
        tol = torch.sin(torch.tensor(tolerance_rad, device=DEVICE, dtype=DTYPE)).view(1, 1)

    else:
        raise ValueError(f"Unsupported alignment mode '{alignment}'")

    memo(("axis_align",   controlled_entity,
    controlled_target,
    reference_basis,
    reference_origin,
    controlled_axis,
    reference_axis,
    alignment_n), err)

    return (err <= tol).flatten()

# <end-of-predicate>


# <predicate>
def pred_point_toward(
    controlled_entity: str,
    controlled_target: str,
    controlled_axis: str,
    reference_basis: str,
    reference_origin: str,
    offset_from_origin_xyz: Union[torch.Tensor, list],
    tolerance_rad: float
) -> torch.Tensor:
    """
    <<Check if a chosen axis of the controlled entity points toward the object with offset within an angular tolerance.

    Inputs
        controlled_entity : (str)
            "robot_frame" or "gripped_object".
        controlled_target : (str)
            Frame or object name being oriented.
        controlled_axis : (str)
             Axis of controlled entity to point.
        reference_basis: str,
        reference_origin: str,
        offset_from_origin_xyz: (3,) or (batch, 3)
        tolerance_rad: float

    Returns
        (batch, ) torch.bool
            True where the axis relation satisfies the alignment condition within tolerance.>>
    """


    if controlled_entity == "gripped_object":
        # Current object orientation in world frame, expressed via state
        q_ctrl = get_object_wxyz_quaternion(controlled_target, "real time")  # (4×1)

    else:
        # Default: controlled entity is the EE itself (or any frame encoded as state[3:7])
        q_ctrl = lookup_robot_frame_rotation(state_frame, "real time")  # (B,4)

    a_ctrl_u = _axis_from_quat_label(q_ctrl, controlled_axis)
    offset_vec = _broadcast_offset(offset_from_origin_xyz)  # (B,3)
    target_pos = get_object_xyz_position(reference_origin, "real time") + offset_vec

    ee_pos = lookup_robot_frame_position(state_frame, "real time")  # (B,3)
    target_vec = target_pos - ee_pos
    target_vec_u = _normalize_axis(target_vec)

    error = 1.0 - torch.sum(a_ctrl_u * target_vec_u, dim=-1, keepdim=True)  # (B,1)

    memo(("point_toward",         controlled_entity,
        controlled_target,
        controlled_axis,
        reference_basis,
        reference_origin,
    _keyify_offset(offset_from_origin_xyz)), error)

    tol = torch.as_tensor(tolerance_rad, device=DEVICE, dtype=DTYPE)
    c_thr = 1.0 - torch.cos(tol)

    return (error <= c_thr).flatten()

# <end-of-predicate>


# <predicate>
def pred_directional_rotate(
    controlled_entity: str,
    controlled_target: str,
    reference_basis: str,
    reference_origin: str,
    rotation_axis: str,
    rotation_angle_rad: float,
    tolerance_rad: float
) -> torch.Tensor:
    """
    <<Check if the controlled entity has rotated by a desired angle about a given axis,
    while keeping off-axis rotation small.

    The rotation is measured as the change in orientation between "current" and "real time"
    (or whichever snapshots you use), decomposed into a component about rotation_axis
    and a residual off-axis component.

    Inputs
        controlled_entity : (str)
            "robot_frame" or "gripped_object".
        controlled_target : (str)
            Frame or object name being rotated.
        reference_basis : (str)
            "world", "object", or "robot_frame" for the rotation axis.
        reference_origin : (str)
            Root defining the rotation axis, similar to reference_basis / reference_origin in pred_axis_align.
        rotation_axis : (str)
            Axis label "+x", "-x", "+y", "-y", "+z", "-z" in the reference_basis coordinates.
        rotation_angle_rad : (float)
            Target rotation magnitude about rotation_axis (non-negative).
        tolerance_rad : (float)
            Allowed angular error in radians.

    Returns
        (batch, ) torch.bool
            True where rotation error is within tolerance_rad.>>
    """

    # Match `directional_rotate_residual` and threshold using tolerance_rad on ||quat - target_quat||.
    controlled_entity_n = controlled_entity
    reference_basis_n = reference_basis.strip().lower()


    # The CasADi residual in residuals.py builds this as
    #   quat_from_euler_xyz(axis_angle_to_rotvec(rotation_axis, rotation_angle_rad))
    # which only works because the rotvec is axis-aligned (a single nonzero Euler
    # component). On the torch side use isaaclab's `quat_from_angle_axis` directly,
    # which is mathematically equivalent and reuses an existing utility (injected
    # into the predicate scope via APIWrapper.get_all_predicate_apis()).
    angle_t = torch.as_tensor(rotation_angle_rad, device=DEVICE, dtype=DTYPE).reshape(1)  # (1,)
    axis_unit = axis_angle_to_rotvec(rotation_axis, 1.0)  # (1,3) unit vector along rotation_axis
    adj_quat = quat_from_angle_axis(angle_t, axis_unit)  # (1,4)

    if controlled_entity_n == "robot_frame":
        quat_now = lookup_robot_frame_rotation(state_frame, "real time")  # (B,4)
        # Snapshot of the controlled entity's orientation at constraint apply
        # time; used as the rotation source for the world-frame branch below.
        q_init = lookup_robot_frame_rotation(state_frame, "current")  # (B,4)
    elif controlled_entity_n == "gripped_object":
        ee_quat_now = lookup_robot_frame_rotation(state_frame, "real time")  # (B,4)
        q_ctrl = _get_gripped_object_rel_quat(controlled_target, "current")  # (B,4)
        quat_now = quat_mul(ee_quat_now, q_ctrl)  # (B,4)
        ee_q_init = lookup_robot_frame_rotation(state_frame, "current")  # (B,4)
        q_init = quat_mul(ee_q_init, q_ctrl)  # (B,4)
    else:
        raise ValueError(f"Unsupported controlled_entity '{controlled_entity}'")

    if reference_basis_n == "robot_frame":
        q_ref = lookup_robot_frame_rotation(reference_origin, "current")
        q_delta_world = quat_mul(quat_mul(q_ref, adj_quat), quat_conj(q_ref))
        target_quat = quat_mul(q_delta_world, q_init)
    elif reference_basis_n == "object" or reference_basis_n == "gripped_object":
        q_ref = get_object_wxyz_quaternion(reference_origin, "current")
        q_delta_world = quat_mul(quat_mul(q_ref, adj_quat), quat_conj(q_ref))
        target_quat = quat_mul(q_delta_world, q_init)
    elif reference_basis_n == "world":
        # World-frame rotation: target = adj_quat ⊗ q_init, where q_init is a
        # SNAPSHOT of the controlled entity's orientation. Mirrors the fix in
        # `directional_rotate_residual` so the residual and predicate share a
        # consistent fixed target for the same memo key.
        target_quat = quat_mul(adj_quat, q_init)
    else:
        raise ValueError(f"Unsupported reference_basis '{reference_basis}'")

    c = quat_now - target_quat

    memo(("directional_rotate",     controlled_entity,
    controlled_target,
    reference_basis,
    reference_origin,
    rotation_axis,
    _keyify_scaler(rotation_angle_rad)), c)

    c_n = torch.linalg.norm(c, dim=-1)  # (batch,)

    return (c_n < tolerance_rad).flatten()


# <end-of-predicate>


# <predicate>
def pred_close_gripper() -> torch.Tensor:
    """
    <<Check if the gripper fingers are physically closed (fully shut, no object).

    This checks purely the mechanical state — fingers are together.
    It does NOT verify an object is held (use pred_object_grasped for that).

    Threshold: openness <= 0.1 (fully closed).
    >>
    """
    pos = gripper_openness().flatten()  # (batch,)
    return pos <= 0.1

# <end-of-predicate>


# <predicate>
def _pred_object_grasped_force_threshold() -> torch.Tensor:
    openness = gripper_openness().flatten()
    left_mag, right_mag = finger_contact_force()
    max_grasp_openness = GRIPPER_MAX_GRASP_OPENNESS
    openness_clamped = torch.clamp(openness, min=0.03, max=max_grasp_openness)
    openness_alpha = (openness_clamped - 0.03) / (max_grasp_openness - 0.03)
    required_force = GRIPPER_MAX_FORCE * (0.1 + 0.6 * openness_alpha)
    both_in_contact = (left_mag >= required_force) & (right_mag >= required_force)
    gripper_partially_closed = (openness > 0.1) & (openness <= max_grasp_openness)
    high_force_threshold = 10.0
    one_high_contact = (
        (left_mag >= high_force_threshold) | (right_mag >= high_force_threshold)
    ) & ~(
        (left_mag >= high_force_threshold) & (right_mag >= high_force_threshold)
    )
    gripper_sufficiently_closed = openness < 0.3
    asymmetric_grasp = one_high_contact & gripper_sufficiently_closed & gripper_partially_closed
    return (both_in_contact & gripper_partially_closed) | asymmetric_grasp


def pred_object_grasped_fr3_finray() -> torch.Tensor:
    return _pred_object_grasped_force_threshold()


def pred_object_grasped_franka_panda() -> torch.Tensor:
    """Use the calibrated Panda joint-effort contact rule for Panda.

    Calibration with the production controller found stable empty closes at
    no more than 1.605 effort units and real bilateral grasps at no less than
    8.276.  A fixed 5.0 threshold cleanly separates those distributions.
    ``finger_contact_force`` already zeros moving fingers, so retain the
    partial-openness guard here and require both fingers.  In particular, do
    not use the generic one-finger shortcut for Panda.
    """
    openness = gripper_openness().flatten()
    left_mag, right_mag = finger_contact_force()
    gripper_partially_closed = (
        (openness > 0.1) & (openness <= GRIPPER_MAX_GRASP_OPENNESS)
    )
    min_bilateral_effort = 5.0
    both_in_contact = (
        (left_mag >= min_bilateral_effort)
        & (right_mag >= min_bilateral_effort)
    )
    return both_in_contact & gripper_partially_closed


def pred_object_grasped_kinova_robotiq() -> torch.Tensor:
    return _pred_object_grasped_force_threshold()


def pred_object_grasped() -> torch.Tensor:
    """
    <<
    True when the current robot-specific gripper model reports a valid object grasp.

    Dispatch is by ROBOT_GRIPPER_NAME so FR3FinRay, Panda, and Kinova do not
    silently share the same gripper-contact interpretation.
    >>
    """
    if ROBOT_GRIPPER_NAME == "fr3_finray":
        return pred_object_grasped_fr3_finray()
    if ROBOT_GRIPPER_NAME == "franka_panda":
        return pred_object_grasped_franka_panda()
    if ROBOT_GRIPPER_NAME == "kinova_robotiq":
        return pred_object_grasped_kinova_robotiq()
    return _pred_object_grasped_force_threshold()
# <end-of-predicate>


# <predicate>
def pred_open_gripper() -> torch.Tensor:
    """
    << >>
    """
    pos = gripper_openness().flatten()  # (batch, )

    return pos >= 0.9
# <end-of-predicate>

# <predicate>
def pred_manipulator_high_effort() -> torch.Tensor:
    """
    <<  >>
    """
    return  high_effort_contact()
# <end-of-predicate>


# <predicate>
def pred_manipulator_low_effort() -> torch.Tensor:
    """
    <<  >>
    """
    eff = manipulator_effort()  # (batch, )
    out = eff <= 10.0

    return out.flatten()  # (batch,1)
# <end-of-predicate>


# <predicate>
def pred_freeze_position(
    controlled_entity: str,
    controlled_target: str,
    tolerance_m: float
) -> torch.Tensor:
    """
    <<  >>
    """
    controlled_entity_n = controlled_entity.strip().lower()
    if controlled_entity_n == "robot_frame":
        rt = lookup_robot_frame_position(controlled_target, "real time")
        c = lookup_robot_frame_position(controlled_target, "current")
    elif controlled_entity_n == "gripped_object":
        rt = get_object_xyz_position(controlled_target, "real time")
        c = get_object_xyz_position(controlled_target, "current")
    else:
        raise ValueError(f"Unsupported controlled_entity '{controlled_entity}'")

    diff = rt - c
    memo(("freeze_position", controlled_entity, controlled_target), diff)
    dist = torch.linalg.norm(diff, dim=-1)  # (batch,)
    out = dist <= tolerance_m
    return out .flatten()# (batch,1)
# <end-of-predicate>


# <predicate>
def pred_freeze_orientation(
    controlled_entity: str,
    controlled_target: str,
    tolerance_rad: float
) -> torch.Tensor:
    """
    <<  >>
    """
    controlled_entity_n = controlled_entity.strip().lower()
    if controlled_entity_n == "robot_frame":
        rt = lookup_robot_frame_rotation(controlled_target, "real time")
        c = lookup_robot_frame_rotation(controlled_target, "current")
    elif controlled_entity_n == "gripped_object":
        rt = get_object_wxyz_quaternion(controlled_target, "real time")
        c = get_object_wxyz_quaternion(controlled_target, "current")
    else:
        raise ValueError(f"Unsupported controlled_entity '{controlled_entity}'")

    # Angle between unit quaternions (robust + interpretable vs L2 diff).
    rt_u = rt / (torch.linalg.norm(rt, dim=-1, keepdim=True) + 1e-9)
    c_u = c / (torch.linalg.norm(c, dim=-1, keepdim=True) + 1e-9)
    dot = torch.sum(rt_u * c_u, dim=-1).abs().clamp(0.0, 1.0)
    ang = 2.0 * torch.acos(dot)  # (batch,)

    memo(("freeze_orientation", controlled_entity, controlled_target), ang)
    out = ang <= tolerance_rad
    return out.flatten()
# <end-of-predicate>
