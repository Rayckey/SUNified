from typing import Union
import math
import casadi as ca

Sym = Union[ca.MX, ca.SX, ca.DM]  # convenience alias


# <utility>
def quad_norm(x: Sym) -> Sym:
    """
    <<Quadratic form 0.5 * xᵀ x for a single environment.

    Inputs
        x : (n×1), (1×n), or flat (n,)
            Error vector.

    Returns
        Sym
            1×1 scalar residual 0.5 * ||x||².>>
    """
    v = ca.reshape(x, x.numel(), 1)  # (n×1)
    return 0.5 * ca.mtimes(v.T, v)  # (1×1)


# <end-of-utility>


# ----------------------------------------------------------------------
# Small helpers (utilities) - single-environment versions
# ----------------------------------------------------------------------


# <utility>
def _world_axis_from_label(axis: str) -> Sym:
    """
    <<Return a world-frame axis vector for "+x", "-x", "+y", "-y", "+z", "-z"
      for a single environment (3×1 DM).>>
    """
    axis = axis.strip()
    if axis == "+x":
        base = ca.DM([1.0, 0.0, 0.0])
    elif axis == "-x":
        base = ca.DM([-1.0, 0.0, 0.0])
    elif axis == "+y":
        base = ca.DM([0.0, 1.0, 0.0])
    elif axis == "-y":
        base = ca.DM([0.0, -1.0, 0.0])
    elif axis == "+z":
        base = ca.DM([0.0, 0.0, 1.0])
    elif axis == "-z":
        base = ca.DM([0.0, 0.0, -1.0])
    else:
        raise ValueError(f"Unsupported axis label '{axis}'")

    return ca.reshape(base, 3, 1)  # (3×1)


# <end-of-utility>


# <utility>
def _axis_from_quat_label(q: Sym, axis: str) -> Sym:
    """
    <<Extract a labeled axis ("+x", "-x", "+y", "-y", "+z", "-z") from a
      quaternion q (4×1) using provided axis-extraction utilities.>>
    """
    axis = axis.strip()
    sign = 1.0
    core = axis

    if axis.startswith("+") or axis.startswith("-"):
        if axis.startswith("-"):
            sign = -1.0
        core = axis[1:]

    core = core.lower()
    if core == "x":
        v = x_axis_from_quat(q)
    elif core == "y":
        v = y_axis_from_quat(q)
    elif core == "z":
        v = z_axis_from_quat(q)
    else:
        raise ValueError(f"Unsupported axis component '{axis}'")

    return sign * v  # (3×1)


# <end-of-utility>


# <utility>
def _normalize_axis(v: Sym, eps: float = 1e-9) -> Sym:
    """
    <<Normalize a 3×1 (or 3×N) axis vector to unit length column-wise.>>
    """
    nrm_sq = ca.sum1(v * v) + eps  # (1×N)
    nrm = ca.sqrt(nrm_sq)  # (1×N)
    return v / ca.repmat(nrm, 3, 1)


# <end-of-utility>


# <utility>
def _get_gripped_object_rel_quat(controlled_target: str, time_stamp: str) -> Sym:
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
    q_ee_grasp_world = ca.reshape(ca.vertcat(get_object_wxyz_quaternion(state_frame, time_stamp)), 4, 1)  # (4×1)
    q_obj_grasp_world = ca.reshape(ca.vertcat(get_object_wxyz_quaternion(
        controlled_target, time_stamp
    )), 4, 1)  # (4×1)
    quat_diff = quat_mul(quat_conj(q_ee_grasp_world), q_obj_grasp_world)

    return quat_diff  # (4×1)


# <end-of-utility>


# <utility>
def _get_gripped_object_rel_pos(controlled_target: str, time_stamp: str) -> Sym:
    """
    <<Return the constant relative position ee_obj_pos for a gripped object.>>
    """
    ee_pos = ca.reshape(ca.vertcat(lookup_robot_frame_position(state_frame, "current")), 3, 1)  # (3×1)
    ee_quat = ca.reshape(ca.vertcat(lookup_robot_frame_rotation(state_frame, "current")), 4, 1)  # (4×1)
    obj_pos = ca.reshape(ca.vertcat(get_object_xyz_position(controlled_target, "current")), 3, 1)  # (3×1)

    ee_obj_pos = quat_to_matrix(quat_conj(ee_quat)) @ (obj_pos - ee_pos)
    return ee_obj_pos  # (3×1)


# <end-of-utility>


# <utility>
def _get_reference_axis_world(
    state: Sym, basis: str, root: str, axis: str, time_stamp: str = "real time"
) -> Sym:
    """
    <<Get a reference axis in the world frame for a single environment.
      basis: "world", "object", or "robot_frame".
      root : "world" / object name / robot frame name.
      axis : "+x", "-x", "+y", "-y", "+z", "-z".>>
    """
    _ = state  # kept for interface symmetry

    basis = basis.strip().lower()
    if basis == "world":
        return _world_axis_from_label(axis)

    elif basis == "object":
        # Object orientation in "real time" so the axis moves with the object.
        q_obj = ca.reshape(ca.vertcat(get_object_wxyz_quaternion(root, time_stamp)), 4, 1)  # (4×1)
        v = _axis_from_quat_label(q_obj, axis)  # (3×1)
        return _normalize_axis(v)

    elif basis == "robot_frame":
        # Reference snapshot of the robot frame.
        q_frame = ca.reshape(ca.vertcat(lookup_robot_frame_rotation(root, time_stamp)), 4, 1)  # (4×1)
        v = _axis_from_quat_label(q_frame, axis)  # (3×1)
        return _normalize_axis(v)

    else:
        raise ValueError(f"Unsupported reference_basis '{basis}'")


# <end-of-utility>


# <utility>
def _get_controlled_axis_in_state(
    state: Sym,
    controlled_entity: str,
    controlled_target: str,
    axis: str,
    time_stamp: str = "real time",
) -> Sym:
    """
    <<Get the controlled axis in world frame for a single environment.

      If controlled_entity == "robot_frame":
        - We assume the controlled frame is the end-effector whose
          orientation is encoded directly in state[3:7] as a world-frame
          quaternion q_ee_world.

      If controlled_entity == "gripped_object":
        - We assume the object is rigidly attached to the end-effector.
        - At grasp time, we know both:
            q_ee_grasp_world  : EE orientation in world frame
            q_obj_grasp_world : object orientation in world frame
        - Their relative orientation is the constant quaternion:
            r_obj_in_ee = quat_conj(q_ee_grasp_world) * q_obj_grasp_world
        - During optimization, the current EE orientation is q_ee_world = state[3:7].
          The current object orientation in world frame is:
            q_obj_world = q_ee_world * r_obj_in_ee
        - We then extract the requested axis from q_obj_world.

      In both cases the returned axis is expressed purely as a function of
      the optimization state and constant parameters, suitable for CasADi.

      Inputs
        state : (n_state×1)
            Full state vector for one environment.
        controlled_entity : str
            "robot_frame" or "gripped_object".
        controlled_target : str
            Name of the controlled frame or gripped object (used to look up
            the relative orientation when controlled_entity == "gripped_object").
        axis : str
            Axis label understood by _axis_from_quat_label (e.g. "x", "y", "z",
            or any agreed label set).

      Returns
        Sym
            Normalized 3×1 axis vector in world frame.
    >>
    """
    # End-effector orientation in world, taken from the optimization state
    q_ee_world = state[3:7]  # (4×1)

    if controlled_entity == "gripped_object":
        # Current object orientation in world frame, expressed via state
        rela_ = _get_gripped_object_rel_quat(controlled_target, "current")  # (4×1)
        q_ctrl = quat_mul(q_ee_world, rela_)

    else:
        # Default: controlled entity is the EE itself (or any frame encoded as state[3:7])
        q_ctrl = q_ee_world  # (4×1)

    v = _axis_from_quat_label(q_ctrl, axis)  # (3×1)
    return _normalize_axis(v)


# <end-of-utility>


# <utility>
def _get_controlled_position_snapshot_world(
    controlled_entity: str,
    controlled_target: str,
) -> Sym:
    """
    <<Lookup controlled entity position in world frame at a given timestamp
      ("current" snapshot or otherwise), without depending on the optimization
      state, for a single environment.>>
    """
    controlled_entity = controlled_entity.strip().lower()

    if controlled_entity == "robot_frame":
        # e.g. TCP frame
        val = lookup_robot_frame_position(controlled_target, "current")  # (3×1)
        return ca.reshape(ca.vertcat(val), 3, 1)

    elif controlled_entity == "gripped_object":
        # Use object position API for the grasped object
        val = get_object_xyz_position(controlled_target, "current")  # (3×1)
        return ca.reshape(ca.vertcat(val), 3, 1)

    else:
        raise ValueError(f"Unsupported controlled_entity '{controlled_entity}'")


# <end-of-utility>


# <utility>
def _get_controlled_quaternion_snapshot_world(
    controlled_entity: str,
    controlled_target: str,
) -> Sym:
    """
    <<Lookup controlled entity quaternion in world frame at a given timestamp
      ("current" snapshot or otherwise), without depending on the optimization
      state, for a single environment.>>
    """
    controlled_entity = controlled_entity.strip().lower()

    if controlled_entity == "robot_frame":
        # e.g. TCP frame
        val = lookup_robot_frame_rotation(controlled_target, "current")  # (4×1)
        return ca.reshape(ca.vertcat(val), 4, 1)

    elif controlled_entity == "gripped_object":
        # Use object quaternion API for the grasped object
        val = get_object_wxyz_quaternion(controlled_target, "current")  # (4×1)
        return ca.reshape(ca.vertcat(val), 4, 1)

    else:
        raise ValueError(f"Unsupported controlled_entity '{controlled_entity}'")


# <end-of-utility>

# <utility>
def axis_angle_to_rotvec(rotation_axis: str, rotation_angle_rad: Sym) -> Sym:
    """
    <<Convert an axis label and rotation angle into a 3D rotation vector
      (axis * angle) in the given basis.

      rotation_axis : "+x", "-x", "+y", "-y", "+z", "-z"
      rotation_angle_rad : scalar (float or Sym)
      returns : (3×1) rotation vector>>
    """
    a = rotation_axis.strip().lower()

    if a == "+x":
        axis_vec = ca.vertcat(1, 0, 0)
    elif a == "-x":
        axis_vec = ca.vertcat(-1, 0, 0)
    elif a == "+y":
        axis_vec = ca.vertcat(0, 1, 0)
    elif a == "-y":
        axis_vec = ca.vertcat(0, -1, 0)
    elif a == "+z":
        axis_vec = ca.vertcat(0, 0, 1)
    elif a == "-z":
        axis_vec = ca.vertcat(0, 0, -1)
    else:
        raise ValueError(f"Invalid rotation_axis label: {rotation_axis!r}")

    return rotation_angle_rad * axis_vec


# <end-of-utility>

# ----------------------------------------------------------------------
# Translation residuals (single-environment)
# ----------------------------------------------------------------------


# <residual>
def free_move_residual(
    state: Sym,
    controlled_entity: str,
    controlled_target: str,
    reference_basis: str,
    reference_origin: str,
    offset_from_origin_xyz: Sym,
    tolerance_m
) -> Sym:
    """
    <<Residual driving a controlled entity toward a target point, with no path constraint,
      for a single environment.

    The target is defined by:
        - reference_basis / reference_origin : which coordinate system and origin to use
        - offset_from_origin_xyz          : 3D offset (meters) in that basis

    The controlled entity is:
        - a robot frame (e.g., TCP), or
        - a gripped object rigidly attached to the gripper.

    Inputs
        state : (n_state×1)
            Full state vector for one environment.
        controlled_entity : (str)
            "robot_frame" or "gripped_object".
        controlled_target : (str)
            If controlled_entity == "robot_frame": name of a robot frame (e.g. "tool0").
            If controlled_entity == "gripped_object": name of a grasped object.
        reference_basis : (str)
            "world", "object", or "robot_frame".
        reference_origin : (str)
            - "world"               when reference_basis == "world"
            - an object name        when reference_basis == "object"
            - a robot frame name    when reference_basis == "robot_frame"
        offset_from_origin_xyz : (3×1) or (3×1 DM)
            Desired offset in meters, expressed in the reference_basis coordinates.

    Returns
        Sym
            Scalar positional residual (1×1).>>
    """


    pos_world = state[0:3]  # (3×1)

    offset_local = ca.reshape(offset_from_origin_xyz, 3, 1)  # (3×1)

    if reference_basis == "world":
        # Offset is already expressed in world frame.
        pos = get_object_xyz_position(reference_origin, "real time")
        target_world = offset_local + pos # (3×1)

    elif reference_basis == "object":
        # Offset is in object frame; transform to world in real time so the
        # target moves with the object.
        target_world = object_to_world_xyz_position(
            reference_origin,
            offset_local,
            "real time",
        )  # (3×1)

    elif reference_basis == "robot_frame":
        # Offset is in some robot frame's local coordinates. Use the reference
        # snapshot ("current") of that frame as the root.
        root_pos = ca.reshape(ca.vertcat(lookup_robot_frame_position(reference_origin, "current")), 3, 1)  # (3×1)
        q_root = ca.reshape(ca.vertcat(lookup_robot_frame_rotation(reference_origin, "current")), 4, 1)  # (4×1)
        rot_root = quat_to_matrix(q_root)  # (3×3)
        world_offset = rot_root @ offset_local
        target_world = root_pos + world_offset  # (3×1)

    else:
        raise ValueError(f"Unsupported reference_basis '{reference_basis}'")


    if controlled_entity == "gripped_object":
        # If controlling a gripped object, we may need to adjust the position
        # based on the relative pose between the gripper and the object.
        ee_obj_pos = _get_gripped_object_rel_pos(
            controlled_target, time_stamp="current"
        )  # (3×1)

        # Only needed for gripped-object composition; keep symbolic work minimal.
        rot_world = quat_to_matrix(state[3:7])  # (3×3)
        obj_pos_state = pos_world + rot_world @ ee_obj_pos
        res = quad_norm(obj_pos_state - target_world)

    elif controlled_entity == "robot_frame":
        # Error and residual
        err = pos_world - target_world  # (3×1)
        res = quad_norm(err)

    else:
        raise ValueError(f"Unsupported controlled_entity '{controlled_entity}'")

    return res


# <end-of-residual>


# <residual>
def reach_point_residual(
    state: Sym,
    controlled_entity: str,
    controlled_target: str,
    reference_basis: str,
    reference_origin: str,
    offset_from_origin_xyz: Sym,
    tolerance_m,
) -> Sym:
    """
    <<Residual for reach_point: reach a concrete target point defined by
      reference_origin + offset_from_origin_xyz (expressed in reference_basis). Path is unconstrained.

      This is the renamed successor of free_move_residual.>>
    """
    return free_move_residual(
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
def reach_distance_residual(
    state: Sym,
    controlled_entity: str,
    controlled_target: str,
    reference_basis: str,
    reference_origin: str,
    offset_from_origin_xyz: Sym,
    distance_m: float,
    tolerance_m: float,
) -> Sym:
    """
    <<Residual for reach_distance: reach any pose such that the controlled target is
      at distance distance_m (± tolerance_m) from the reference point defined by
      reference_origin + offset_from_origin_xyz (in reference_basis).>>
    """
    pos_world = state[0:3]  # (3×1)
    offset_local = ca.reshape(offset_from_origin_xyz, 3, 1)  # (3×1)

    reference_basis_n = reference_basis
    if reference_basis_n == "object":
        target_world = object_to_world_xyz_position(
            reference_origin,
            offset_local,
            "real time",
        )  # (3×1)
    elif reference_basis_n == "robot_frame":
        root_pos = ca.reshape(
            ca.vertcat(lookup_robot_frame_position(reference_origin, "current")), 3, 1
        )  # (3×1)
        q_root = ca.reshape(
            ca.vertcat(lookup_robot_frame_rotation(reference_origin, "current")), 4, 1
        )  # (4×1)
        rot_root = quat_to_matrix(q_root)  # (3×3)
        world_offset = rot_root @ offset_local
        target_world = root_pos + world_offset  # (3×1)
    elif reference_basis_n == 'world':
        root_pos = ca.reshape(
            ca.vertcat(get_object_xyz_position(reference_origin, "real time")), 3, 1
        ) 
        target_world = root_pos + offset_local 
    else:
        raise ValueError(f"Unsupported reference_basis '{reference_basis}'")

    controlled_entity_n = controlled_entity
    if controlled_entity_n == "gripped_object":
        ee_obj_pos = _get_gripped_object_rel_pos(controlled_target, time_stamp="current")  # (3×1)
        rot_world = quat_to_matrix(state[3:7])  # (3×3)
        ctrl_pos_world = pos_world + rot_world @ ee_obj_pos
    elif controlled_entity_n == "robot_frame":
        ctrl_pos_world = pos_world
    else:
        raise ValueError(f"Unsupported controlled_entity '{controlled_entity}'")

    diff = ctrl_pos_world - target_world
    eps = 1e-8
    dist = ca.sqrt(ca.sumsqr(diff) + eps)  # Soft norm: avoids gradient singularity at zero
    err = dist - distance_m

    return quad_norm(err)


# <end-of-residual>


# <residual>
def directional_move_residual(
    state: Sym,
    controlled_entity: str,
    controlled_target: str,
    reference_basis: str,
    reference_origin: str,
    axis: str,
    travel_distance_m: float,
    tolerance_m: float
) -> Sym:
    """
    <<Residual encouraging motion along a specified axis by a desired distance,
    while penalizing lateral deviation, for a single environment.

    Inputs
        state : (n_state×1)
            Full state vector.
        controlled_entity : (str)
            "robot_frame" or "gripped_object".
        controlled_target : (str)
            Frame or grasped object name.
        reference_basis : (str)
            "world", "object", or "robot_frame" for the axis definition.
        reference_origin : (str)
            Root object/frame defining the axis.
        axis : (str)
            "+x", "-x", "+y", "-y", "+z", "-z" in the reference_basis coordinates.
        travel_distance_m : (float)
            Target displacement along axis (non-negative).
        travel_tolerance_m : (float)
            Tolerance on axial displacement (meters).
        lateral_tolerance_m : (float)
            Tolerance on lateral displacement (meters).

    Returns
        Sym
            Scalar residual combining axial and lateral errors.>>
    """
    a_hat = _get_reference_axis_world(
        state, reference_basis, reference_origin, axis, "current"
    )  # (3×1)

    # 2) Displacement of the controlled entity between a reference snapshot
    #    ("current") and the live optimization state ("real time").
    pos_ref = _get_controlled_position_snapshot_world(
        controlled_entity,
        controlled_target,
    )  # (3×1)

    pos_world = state[0:3]  # (3×1)
    if controlled_entity.strip().lower() == "gripped_object":
        ee_obj_pos = _get_gripped_object_rel_pos(
            controlled_target, time_stamp="current"
        )  # (3×1)
        rot_world = quat_to_matrix(state[3:7])  # (3×3)
        obj_pos_state = pos_world + rot_world @ ee_obj_pos
        disp_world = obj_pos_state - pos_ref  # (3×1)
    else:
        disp_world = pos_world - pos_ref  # (3×1)

    # 3) Decompose into axial and lateral components.
    d_parallel = ca.dot(disp_world, a_hat)  # (1×1), dot
    d_axial = d_parallel - travel_distance_m

    d_parallel_vec = a_hat * d_parallel  # (3×1)
    d_lateral_vec = disp_world - d_parallel_vec  # (3×1)

    # Compute squared distance directly (avoids norm_2/sqrt gradient issues at 0,
    # and avoids any shape ambiguity with ca.vertcat on scalar + vector).
    # Mathematically equivalent to: 0.5 * (d_axial² + ||d_lateral_vec||²)
    d_axial_sq = d_axial * d_axial  # scalar
    d_lateral_sq = ca.dot(d_lateral_vec, d_lateral_vec)  # scalar: ||d_lateral_vec||²
    res = 0.5 * (d_axial_sq + d_lateral_sq)
    return res


# <end-of-residual>


# ----------------------------------------------------------------------
# Orientation residuals (single-environment)
# ----------------------------------------------------------------------


# <residual>
def axis_align_residual(
    state: Sym,
    controlled_entity: str,
    controlled_target: str,
    reference_basis: str,
    reference_origin: str,
    controlled_axis: str,
    reference_axis: str,
    alignment: str,
    tolerance_rad: float
) -> Sym:
    """
    <<Residual driving an axis of the controlled entity toward a specified relation
    with a reference axis, for a single environment.

    Inputs
        state : (n_state×1)
            Full state vector.
        controlled_entity : (str)
            "robot_frame" or "gripped_object".
        controlled_target : (str)
            Frame or object whose orientation is being controlled.
        reference_basis : (str)
            "world", "object", or "robot_frame" for the reference axis.
        reference_origin : (str)
            Root defining the reference axis.
        controlled_axis : (str)
            Local axis on the controlled entity: "+x", "-x", "+y", "-y", "+z", "-z".
        reference_axis : (str)
            Axis label on the reference_origin in reference_basis.
        alignment : (str)
            "codirectional", "antiparallel", "bipolar", or "orthogonal".
        tolerance_rad : (float)
            Tolerance band for alignment.

    Returns
        Sym
            Scalar residual on the axis misalignment.>>
    """
    a_ctrl = _get_controlled_axis_in_state(
        state,
        controlled_entity,
        controlled_target,
        controlled_axis,
        "real time",
    )  # (3×1)

    # Reference axis in world frame.
    a_ref = _get_reference_axis_world(
        state,
        reference_basis,
        reference_origin,
        reference_axis,
        "real time",
    )  # (3×1)

    # Ensure unit length.
    a_ctrl_u = _normalize_axis(a_ctrl)
    a_ref_u = _normalize_axis(a_ref)

    alignment = alignment.strip().lower()
    if alignment == "codirectional":
        # Good when angle <= tol => dot >= cos(tol).
        # Penalize only if dot < cos_tol.
        err = 1.0 - ca.dot(a_ctrl_u, a_ref_u)

    elif alignment == "antiparallel":
        # Good when angle >= pi - tol => dot <= -cos(tol).
        # Penalize only if dot > -cos_tol.
        err = 1.0 - ca.dot(a_ctrl_u, -a_ref_u)

    elif alignment == "bipolar":
        # Good when the axes are collinear in either direction.
        err = 1.0 - ca.fabs(ca.dot(a_ctrl_u, a_ref_u))

    elif alignment == "orthogonal":
        # Good when |angle - pi/2| <= tol; equivalently |dot| <= sin(tol) for small tol.
        delta = 1e-9
        c = ca.sum1(a_ctrl_u * a_ref_u)
        err = ca.sqrt(c * c + (delta * delta))
    else:
        raise ValueError(f"Unsupported alignment mode '{alignment}'")

    res = quad_norm(err)
    return res


# <end-of-residual>


# <residual>
def point_toward_residual(
    state: Sym,
    controlled_entity: str,
    controlled_target: str,
    controlled_axis: str,
    reference_basis: str,
    reference_origin: str,
    offset_from_origin_xyz: Sym,
    tolerance_rad: float
) -> Sym:
    """
    <<Residual driving a robot-frame axis toward a target point for a single
    environment.

    Inputs
        state : (n_state×1)
            Full state vector.
        controlled_entity : (str)
            "robot_frame" or "gripped_object".
        controlled_target : (str)
            Frame or object whose orientation is being controlled.
        reference_basis : (str)
            "world", "object", or "robot_frame" for the target point.
        reference_origin : (str)
            Root defining the target point.
        controlled_axis : (str)
            Local axis on the controlled entity: "+x", "-x", "+y", "-y", "+z", "-z".
        offset_from_origin_xyz : Sym
            Offset from the reference origin in reference_basis.
        tolerance_rad : (float)
            Tolerance band for pointing.

    Returns
        Sym
            Scalar residual on the pointing error.>>
    """


    if controlled_entity != "robot_frame":
        raise ValueError(f"Only robot_frame is supported in controlled_entity for point_toward")


    # Controlled axis in world frame from the optimization state.
    a_ctrl = _get_controlled_axis_in_state(
        state,
        controlled_entity,
        controlled_target,
        controlled_axis,
        "real time",
    )  # (3×1)

    # Ensure unit length.
    a_ctrl_u = _normalize_axis(a_ctrl)

    target_pos = get_object_xyz_position(reference_origin, "real time") + ca.reshape(offset_from_origin_xyz, 3, 1)

    target_vec = target_pos - state[0:3] 
    target_vec_u = _normalize_axis(target_vec)

    c = 1.0 - ca.dot(a_ctrl_u, target_vec_u)
    res = quad_norm(c)
    return res


# <end-of-residual>


# <residual>
def directional_rotate_residual(
    state: Sym,
    controlled_entity: str,
    controlled_target: str,
    reference_basis: str,
    reference_origin: str,
    rotation_axis: str,
    rotation_angle_rad: float,
    tolerance_rad: float,
) -> Sym:
    """
    <<Residual encouraging rotation about a given axis by a specific angle,
    while discouraging off-axis rotation, for a single environment.

    Inputs
        state : (n_state×1)
            Full state vector.
        controlled_entity : (str)
            "robot_frame" or "gripped_object".
        controlled_target : (str)
            Frame or object being rotated.
        reference_basis : (str)
            "world", "object", or "robot_frame" for the rotation axis.
        reference_origin : (str)
            Root defining the rotation axis.
        rotation_axis : (str)
            Axis label "+x", "-x", "+y", "-y", "+z", "-z" in reference_basis.
        rotation_angle_rad : (float)
            Target rotation magnitude about rotation_axis.
        tolerance_rad : (float)
            Allowed angular error in radians.

    Returns
        Sym
            Scalar residual combining on-axis and off-axis components.>>
    """


    adjustment = axis_angle_to_rotvec(rotation_axis, rotation_angle_rad)
    adj_quat = quat_from_euler_xyz(adjustment)

    if controlled_entity == "robot_frame":
        quat = state[3:7]
        # Snapshot of controlled entity's orientation at constraint apply time;
        # used as the rotation source for the world-frame branch below.
        q_init = ca.reshape(
            ca.vertcat(lookup_robot_frame_rotation(state_frame, "current")), 4, 1
        )
    elif controlled_entity == "gripped_object":
        q_ctrl = _get_gripped_object_rel_quat(controlled_target, "current")  # (4×1)
        quat = quat_mul(state[3:7], q_ctrl)
        ee_q_init = ca.reshape(
            ca.vertcat(lookup_robot_frame_rotation(state_frame, "current")), 4, 1
        )
        q_init = quat_mul(ee_q_init, q_ctrl)
    else:
        raise ValueError(f"Unsupported controlled_entity '{controlled_entity}'")

    reference_basis_n = reference_basis.strip().lower()
    if reference_basis_n == "robot_frame":
        q_ref = ca.reshape(
            ca.vertcat(lookup_robot_frame_rotation(reference_origin, "current")), 4, 1
        )
        q_delta_world = quat_mul(quat_mul(q_ref, adj_quat), quat_conj(q_ref))
        target_quat = quat_mul(q_delta_world, q_init)

    elif reference_basis_n == "object" or reference_basis_n == "gripped_object":
        q_ref = ca.reshape(
            ca.vertcat(get_object_wxyz_quaternion(reference_origin, "current")), 4, 1
        )
        q_delta_world = quat_mul(quat_mul(q_ref, adj_quat), quat_conj(q_ref))
        target_quat = quat_mul(q_delta_world, q_init)

    elif reference_basis_n == "world":
        # World-frame rotation: target = adj_quat ⊗ q_init, where q_init is a
        # SNAPSHOT of the controlled entity's orientation. Previously this used
        # `quat` (the live optimization variable) which made target track the
        # optimizer and produced no useful gradient, so the manipulator never
        # moved.
        target_quat = quat_mul(adj_quat, q_init)
    else:
        raise ValueError(f"Unsupported reference_basis '{reference_basis}'")

    c = quat - target_quat

    res = quad_norm(c)
    return res


# <end-of-residual>


# <residual>
def close_gripper_residual(
    state: Sym
) -> Sym:
    """
    <<Simple residual to drive the gripper joints toward a desired closed
    configuration for a single environment.>>
    """
    gripper_state = state[7:]  # assume last 2 entries are gripper joints (2×1)
    target = ca.vertcat(0.0, 0.0)  # (2×1) desired closed configuration
    res = quad_norm(gripper_state - target)
    return res


# <end-of-residual>


# <residual>
def open_gripper_residual(
    state: Sym
) -> Sym:
    """
    <<Simple residual to drive the gripper joints toward a desired closed
    configuration for a single environment.>>
    """

    gripper_state = state[7:]  # assume last 2 entries are gripper joints (2×1)
    target = ca.vertcat(1.0, 1.0)  # (2×1) desired closed configuration
    res = quad_norm(gripper_state - target)
    return res


# <end-of-residual>


# <residual>
def freeze_position_residual(
    state: Sym,
    controlled_entity: str,
    controlled_target: str,
    tolerance_m: float
) -> Sym:
    """
    <<Simple residual to drive the gripper joints toward a desired closed
    configuration for a single environment.>>
    """

    _ = tolerance_m
    pos = state[:3]

    controlled_entity_n = controlled_entity.strip().lower()
    if controlled_entity_n == "robot_frame":
        ref_pos = lookup_robot_frame_position(controlled_target, "current")
        pos_state = ca.reshape(pos, 3, 1)
        ref_pos = ca.reshape(ca.vertcat(ref_pos), 3, 1)
        return quad_norm(pos_state - ref_pos)

    if controlled_entity_n == "gripped_object":
        # Compare composed object position (via EE state + rel pose) to snapshot object position.
        ref_pos = get_object_xyz_position(controlled_target, "current")
        ref_pos = ca.reshape(ca.vertcat(ref_pos), 3, 1)
        ee_obj_pos = _get_gripped_object_rel_pos(controlled_target, time_stamp="current")
        rot_world = quat_to_matrix(state[3:7])
        pos_state = ca.reshape(pos, 3, 1) + rot_world @ ee_obj_pos
        return quad_norm(pos_state - ref_pos)

    raise ValueError(f"Unsupported controlled_entity '{controlled_entity}'")

    # Ensure dimensions match (3x1)
    pos = ca.reshape(pos, 3, 1)
    ee_pos = ca.reshape(ca.vertcat(ee_pos), 3, 1)

    res = quad_norm(pos - ee_pos)

    return res


# <end-of-residual>


# <residual>
def freeze_orientation_residual(
    state: Sym,
    controlled_entity: str,
    controlled_target: str,
    tolerance_rad: float
) -> Sym:
    """
    <<Simple residual to drive the gripper joints toward a desired closed
    configuration for a single environment.>>
    """

    quat = ca.reshape(state[3:7], 4, 1)

    ref_q = lookup_robot_frame_rotation(state_frame, "current")
    ref_q = ca.reshape(ca.vertcat(ref_q), 4, 1)
    q1 = quat
    q2 = ref_q


    res = quad_norm(q1 - q2)
    return res


# <end-of-residual>
