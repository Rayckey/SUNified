import concurrent.futures
import contextlib
import h5py
import json
import numpy as np
import os
import shutil
import time
import torch
from typing import Any, Sequence, cast

from isaaclab.utils.math import matrix_from_quat
from kuafu.data.stage_labels import STAGE_TIMING, program_stage_schema, snapshot_decision_stage

POINT_CLOUD_MIN_Z = 0.01
# If <= 0, keep all valid cropped points and pad unused slots with NaN.
# If > 0, uniformly subsample that many points per frame (random subset of valid depths).
POINT_CLOUD_NUM_POINTS = -1
try:
    POINT_CLOUD_TIMESTEP_CHUNK_SIZE = max(1, int(os.environ.get("KUAFU_POINT_CLOUD_TIMESTEP_CHUNK_SIZE", "16")))
except ValueError:
    POINT_CLOUD_TIMESTEP_CHUNK_SIZE = 16

# Reduce allocator fragmentation for repeated large CUDA allocations.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

WORKSPACE_LIMITS = [
    [-1.5, -1.5, POINT_CLOUD_MIN_Z],
    [1.5, 1.5, 1.5],
]


def _make_quaternions_temporally_continuous(
    quaternions: np.ndarray,
    previous_quaternion: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Choose quaternion signs so adjacent samples have non-negative dot products.

    Quaternions q and -q encode the same rotation, but sudden sign changes look
    like large jumps to sequence models. This preserves the recorded rotation
    and only changes the representative sign.
    """
    quats = np.asarray(quaternions)
    if quats.ndim != 2 or quats.shape[-1] != 4 or quats.shape[0] == 0:
        return quats, previous_quaternion

    out = quats.copy()
    prev = None if previous_quaternion is None else np.asarray(previous_quaternion, dtype=out.dtype)
    for i in range(out.shape[0]):
        current = out[i]
        if not np.isfinite(current).all():
            continue
        if prev is not None and np.isfinite(prev).all() and float(np.dot(prev, current)) < 0.0:
            out[i] = -current
        prev = out[i].copy()
    return out, prev.copy() if prev is not None else None


def _get_robot_dof_targets(env) -> torch.Tensor | None:
    """Return the active robot joint-position target buffer when the env exposes one."""
    for attr_name in ("robot_dof_targets", "_robot_dof_targets", "_logged_joint_targets"):
        targets = getattr(env, attr_name, None)
        if isinstance(targets, torch.Tensor):
            return targets
    return None


def _gripper_binary_from_joint_targets(
    joint_targets: np.ndarray,
    gripper_opened: np.ndarray | None,
    gripper_closed: np.ndarray | None,
) -> np.ndarray | None:
    """Map gripper joint targets to the reference binary action: 1=open, 0=closed."""
    if joint_targets.ndim != 2 or joint_targets.shape[-1] < 2:
        return None
    if gripper_opened is None or gripper_closed is None:
        return None

    grip_targets = joint_targets[:, -2:].astype(np.float32, copy=False)
    open_pos = np.asarray(gripper_opened, dtype=np.float32).reshape(1, -1)
    closed_pos = np.asarray(gripper_closed, dtype=np.float32).reshape(1, -1)
    n_gripper = min(grip_targets.shape[-1], open_pos.shape[-1], closed_pos.shape[-1])
    if n_gripper <= 0:
        return None

    grip_targets = grip_targets[:, -n_gripper:]
    open_pos = open_pos[:, -n_gripper:]
    closed_pos = closed_pos[:, -n_gripper:]
    open_dist = np.mean(np.abs(grip_targets - open_pos), axis=-1)
    closed_dist = np.mean(np.abs(grip_targets - closed_pos), axis=-1)
    return (open_dist < closed_dist).astype(np.float32).reshape(-1, 1)


def _find_body_index(robot_art, candidates: Sequence[str]) -> int:
    body_names = list(getattr(robot_art, "body_names", ()))
    for candidate in candidates:
        if candidate in body_names:
            return int(body_names.index(candidate))
    for candidate in candidates:
        for idx, body_name in enumerate(body_names):
            if candidate in body_name:
                return int(idx)
    raise ValueError(
        "Could not resolve any body name from "
        f"{list(candidates)}. Available robot bodies: {body_names}"
    )


def resolve_arm_body_indices(env) -> tuple[int, int, int, float]:
    """Resolve arm and fingertip body indices for supported Kuafu robots.

    Returns ``(link7_idx, left_finger_idx, right_finger_idx, width_offset)``.
    ``width_offset`` is subtracted from fingertip-link distance; the Robotiq
    tip-link frames sit outside the pad surfaces, while Franka finger links
    already give a useful opening distance without an extra correction.
    """
    robot_art = env.scene._articulations["robot"]
    link7_idx = _find_body_index(
        robot_art,
        (
            "fr3_link7",
            "panda_link7",
            "bracelet_link",
            "wrist_3_link",
            "link7",
            "link_7",
        ),
    )
    left_idx = _find_body_index(
        robot_art,
        (
            "robotiq_85_left_finger_tip_link_finray",
            "robotiq_85_left_finger_tip_link",
            "panda_leftfinger",
            "left_inner_finger_pad",
            "left_inner_finger",
            "left_outer_finger",
            "left_finger_tip",
        ),
    )
    right_idx = _find_body_index(
        robot_art,
        (
            "robotiq_85_right_finger_tip_link_finray",
            "robotiq_85_right_finger_tip_link",
            "panda_rightfinger",
            "right_inner_finger_pad",
            "right_inner_finger",
            "right_outer_finger",
            "right_finger_tip",
        ),
    )
    body_names = list(getattr(robot_art, "body_names", ()))
    left_name = body_names[left_idx]
    right_name = body_names[right_idx]
    width_offset = 0.0505 if "robotiq_85" in left_name and "robotiq_85" in right_name else 0.0
    return link7_idx, left_idx, right_idx, width_offset


def get_measured_fingertip_pose_and_gripper_width(
    env,
    left_finger_idx: int,
    right_finger_idx: int,
    finger_width_offset: float = 0.0505,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return measured panda_fingertip pose and observed gripper width.

    The fingertip pose follows the same measured frame source used by learned
    policy observations: ``lookup_robot_frame_position/rotation("panda_fingertip")``.
    This keeps recorded ``eef_pose`` aligned with the observation frame instead
    of the IK body frame (bracelet/link7).

    Shapes: position (N, 3), quaternion-wxyz (N, 4), rotation matrix (N, 3, 3),
    gripper width (N,).
    """
    robot_art = env.scene._articulations["robot"]
    body_pos_w = robot_art.data.body_pos_w  # (N, B, 3)

    fingertip_pos = None
    fingertip_quat_wxyz = None
    api = getattr(getattr(env, "program_runtime", None), "controller_api_wrapper", None)
    if api is not None:
        try:
            fingertip_pos = api.lookup_robot_frame_position("panda_fingertip")
            fingertip_quat_wxyz = api.lookup_robot_frame_rotation("panda_fingertip")
        except Exception:
            fingertip_pos = None
            fingertip_quat_wxyz = None

    if fingertip_pos is None or fingertip_quat_wxyz is None:
        sensor = None
        sensors = getattr(env.scene, "sensors", {})
        with contextlib.suppress(Exception):
            sensor = sensors.get("panda_fingertip") if hasattr(sensors, "get") else sensors["panda_fingertip"]
        if sensor is None:
            with contextlib.suppress(Exception):
                sensor = sensors.get("ee_frame") if hasattr(sensors, "get") else sensors["ee_frame"]
        if sensor is None:
            raise RuntimeError("Cannot record measured eef_pose: panda_fingertip frame source is unavailable.")

        fingertip_pos = sensor.data.target_pos_w[:, 0, :] - env.scene.env_origins
        fingertip_quat_wxyz = sensor.data.target_quat_w[:, 0, :]

    fingertip_pos = fingertip_pos.to(device=body_pos_w.device, dtype=torch.float32)
    fingertip_quat_wxyz = fingertip_quat_wxyz.to(device=body_pos_w.device, dtype=torch.float32)
    fingertip_rot_mat = matrix_from_quat(fingertip_quat_wxyz)
    gripper_width = (
        torch.linalg.norm(
            body_pos_w[:, left_finger_idx, :] - body_pos_w[:, right_finger_idx, :], dim=-1
        )
        - float(finger_width_offset)
    ).clamp(min=0.0)
    return fingertip_pos, fingertip_quat_wxyz, fingertip_rot_mat, gripper_width


def get_absolute_fingertip_command_pose_from_action(actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return the absolute panda_fingertip command pose encoded by policy actions.

    Assumes ``action_mode == 'absolute_ee'``: ``actions`` lays out as
    ``[x, y, z, rot6d(6), grip(2)]`` (and possibly extra trailing dims that
    are ignored).

    Args:
        actions: (N, A) policy actions on any device.

    Returns:
        ``(fingertip_pos (N, 3), fingertip_rot_mat (N, 3, 3))`` in the robot
        base frame, or ``None`` if ``actions`` does not have at least 9 columns.
    """
    if actions.shape[-1] < 9:
        return None
    # Lazy import: pulls in kuafu/torch._dynamo only when needed.
    from kuafu.mpc.simulation.spatial_utils import rotation_6d_to_quat

    actions_dev = actions.to(dtype=torch.float32)
    ft_pos = actions_dev[:, :3]
    ft_quat = rotation_6d_to_quat(actions_dev[:, 3:9])
    ft_rot_mat = matrix_from_quat(ft_quat)
    return ft_pos, ft_rot_mat


def _infer_episode_shapes(depths: list[torch.Tensor]) -> tuple[int, int, int, int, torch.device]:
    """Infer B, H, W, TC and device from a list of depth tensors (each B,H,W or B,H,W,1)."""
    d0 = depths[0]
    if isinstance(d0, torch.Tensor) and d0.is_cuda:
        device = d0.device
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # B, H, W from first tensor (handles both (B,H,W) and (B,H,W,1))
    B = int(d0.shape[0])
    H = int(d0.shape[1])
    W = int(d0.shape[2])
    TC = len(depths)
    return B, H, W, TC, device


def _to_tensor_list(
    x: Sequence[torch.Tensor | np.ndarray],
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> list[torch.Tensor]:
    """Convert a list of tensors or arrays to a list of torch.Tensor on the given device."""
    out = []
    for t in x:
        if isinstance(t, torch.Tensor):
            tensor = cast(torch.Tensor, t)
            out.append(tensor.to(device=device, dtype=dtype or tensor.dtype))
        else:
            arr = np.asarray(t, dtype=np.float32)
            t_t = torch.from_numpy(arr).to(device=device, dtype=dtype or torch.float32)
            out.append(t_t)
    return out


def fused_pc(
    depths: list[torch.Tensor],  # flat list length T*C, each (B,H,W,1) or (B,H,W)
    rgbs: list[torch.Tensor] | None,  # flat list length T*C, each (B,H,W,3/4) or None
    intrinsics: list[torch.Tensor],  # flat list length T*C, each (B,3,3)
    extrinsics: list[torch.Tensor],  # flat list length T*C, each (B,4,4)
    num_cams: int,
    num_points: int = -1,
    depth_scale: float = 1.0,
    crop_to_workspace: bool = True,
    crop_z_only: bool = False,
    crop_area_min_override: Sequence[float | None] | None = None,
    crop_area_max_override: Sequence[float | None] | None = None,
    exclude_box_min_override: Sequence[float | None] | None = None,
    exclude_box_max_override: Sequence[float | None] | None = None,
    exclude_cylinder_center_xy_override: Sequence[float] | None = None,
    exclude_cylinder_radius: float | None = None,
    exclude_cylinder_z_min: float | None = None,
    exclude_cylinder_z_max: float | None = None,
    min_depth_m: float | None = None,
    max_depth_m: float | None = None,
    table_plane_normal: Sequence[float] | None = None,
    table_plane_d: float | None = None,
    table_plane_clearance_m: float = 0.0,
) -> torch.Tensor:
    """
    Build fused point clouds per timestep (fusing across all cameras), without
    a Python loop over timesteps.

    Args:
        depths:      [T*C] depth maps; each (B,H,W,1) or (B,H,W)
        rgbs:        [T*C] RGB images; each (B,H,W,3/4), or None for XYZ-only clouds
        intrinsics:  [T*C] camera intrinsics; each (B,3,3)
        extrinsics:  [T*C] camera extrinsics; each (B,4,4)
        num_cams:    C
        num_points:  P sampled points per timestep per env; if <= 0, keep all
                     valid points up to the full camera pixel budget
        depth_scale: 1.0 if meters, 1000.0 if mm, etc.
        crop_to_workspace: if True, keep only points inside WORKSPACE_LIMITS.
        crop_z_only: if True and crop_to_workspace is False, keep only points
                     inside the WORKSPACE_LIMITS z range.

    Returns:
        (T, B, P, 3) or (T, B, P, 6) fused point clouds.
    """
    if len(depths) == 0:
        raise ValueError("depths is empty.")
    if not (len(depths) == len(intrinsics) == len(extrinsics)):
        raise ValueError("depths, intrinsics, extrinsics must have equal length T*C.")
    if num_cams <= 0:
        raise ValueError("num_cams must be > 0.")

    B, H, W, TC, device = _infer_episode_shapes(depths)
    if TC % num_cams != 0:
        raise ValueError(f"len(depths)={TC} not divisible by num_cams={num_cams}.")
    T = TC // num_cams

    chunk_size = max(1, POINT_CLOUD_TIMESTEP_CHUNK_SIZE)
    if T > chunk_size:
        pc_t_b_pf = None
        for start_t in range(0, T, chunk_size):
            end_t = min(start_t + chunk_size, T)
            start_i = start_t * num_cams
            end_i = end_t * num_cams
            chunk = fused_pc(
                depths=depths[start_i:end_i],
                rgbs=rgbs[start_i:end_i] if rgbs is not None else None,
                intrinsics=intrinsics[start_i:end_i],
                extrinsics=extrinsics[start_i:end_i],
                num_cams=num_cams,
                num_points=num_points,
                depth_scale=depth_scale,
                crop_to_workspace=crop_to_workspace,
                crop_z_only=crop_z_only,
                crop_area_min_override=crop_area_min_override,
                crop_area_max_override=crop_area_max_override,
                exclude_box_min_override=exclude_box_min_override,
                exclude_box_max_override=exclude_box_max_override,
                exclude_cylinder_center_xy_override=exclude_cylinder_center_xy_override,
                exclude_cylinder_radius=exclude_cylinder_radius,
                exclude_cylinder_z_min=exclude_cylinder_z_min,
                exclude_cylinder_z_max=exclude_cylinder_z_max,
                min_depth_m=min_depth_m,
                max_depth_m=max_depth_m,
                table_plane_normal=table_plane_normal,
                table_plane_d=table_plane_d,
                table_plane_clearance_m=table_plane_clearance_m,
            )
            if pc_t_b_pf is None:
                pc_t_b_pf = torch.empty(
                    (T, *chunk.shape[1:]),
                    device=chunk.device,
                    dtype=chunk.dtype,
                )
            pc_t_b_pf[start_t:end_t] = chunk
            del chunk
        if pc_t_b_pf is None:
            raise RuntimeError("Point cloud chunking produced no output chunks.")
        return pc_t_b_pf

    depths = _to_tensor_list(depths, device)
    intrinsics = _to_tensor_list(intrinsics, device)
    extrinsics = _to_tensor_list(extrinsics, device)
    rgbs = _to_tensor_list(rgbs, device, dtype=torch.float32) if rgbs is not None else None

    # Depths: (T*C, B, H, W[,1]) -> (B*T, C, H, W)
    depth_tc_bhw = torch.stack([d.squeeze(-1) if d.ndim == 4 else d for d in depths], dim=0)
    depth_btchw = (
        depth_tc_bhw.reshape(T, num_cams, B, H, W)
        .permute(2, 0, 1, 3, 4)  # (B, T, C, H, W)
        .contiguous()
        .reshape(B * T, num_cams, H, W)
    )  # (B*T, C, H, W)

    # Intrinsics: (T*C, B, 3, 3) -> (B*T, C, 3, 3)
    intr_tc_b33 = torch.stack(intrinsics, dim=0)
    intr_bt_c33 = (
        intr_tc_b33.reshape(T, num_cams, B, 3, 3).permute(2, 0, 1, 3, 4).contiguous().reshape(B * T, num_cams, 3, 3)
    )

    # Extrinsics: (T*C, B, 4, 4) -> (B*T, C, 4, 4)
    extr_tc_b44 = torch.stack(extrinsics, dim=0)
    extr_bt_c44 = (
        extr_tc_b44.reshape(T, num_cams, B, 4, 4).permute(2, 0, 1, 3, 4).contiguous().reshape(B * T, num_cams, 4, 4)
    )

    rgb_btchw3 = None
    if rgbs is not None:
        rgb_tc_bhw3 = torch.stack(
            [(rgb[..., :3] if rgb.shape[-1] > 3 else rgb) for rgb in rgbs],
            dim=0,
        )
        rgb_btchw3 = (
            rgb_tc_bhw3.reshape(T, num_cams, B, H, W, 3)
            .permute(2, 0, 1, 3, 4, 5)
            .contiguous()
            .reshape(B * T, num_cams, H, W, 3)
        )

    # Single fused call with batch = B*T, N = C
    def _bounds_to_tensor(bounds, default_bounds, unbounded_value):
        source = default_bounds if bounds is None else bounds
        if len(source) != 3:
            raise ValueError(f"Point-cloud crop bounds must have 3 values, got {source}.")
        values = [unbounded_value if value is None else float(value) for value in source]
        return torch.tensor(values, device=device, dtype=torch.float32)

    if crop_to_workspace:
        crop_area_min = _bounds_to_tensor(crop_area_min_override, WORKSPACE_LIMITS[0], float("-inf"))
        crop_area_max = _bounds_to_tensor(crop_area_max_override, WORKSPACE_LIMITS[1], float("inf"))
    elif crop_z_only:
        crop_area_min = torch.tensor(
            [float("-inf"), float("-inf"), WORKSPACE_LIMITS[0][2]],
            device=device,
            dtype=torch.float32,
        )
        crop_area_max = torch.tensor(
            [float("inf"), float("inf"), WORKSPACE_LIMITS[1][2]],
            device=device,
            dtype=torch.float32,
        )
    else:
        crop_area_min = None
        crop_area_max = None

    exclude_box_min = None
    exclude_box_max = None
    if exclude_box_min_override is not None or exclude_box_max_override is not None:
        exclude_box_min = _bounds_to_tensor(exclude_box_min_override, [None, None, None], float("-inf"))
        exclude_box_max = _bounds_to_tensor(exclude_box_max_override, [None, None, None], float("inf"))

    exclude_cylinder_center_xy = None
    exclude_cylinder_radius_value = -1.0
    exclude_cylinder_z_min_value = -1.0e30
    exclude_cylinder_z_max_value = 1.0e30
    if exclude_cylinder_radius is not None:
        if exclude_cylinder_center_xy_override is None:
            raise ValueError("exclude_cylinder_center_xy_override is required when exclude_cylinder_radius is set.")
        if len(exclude_cylinder_center_xy_override) != 2:
            raise ValueError(
                "exclude_cylinder_center_xy_override must contain exactly 2 values, "
                f"got {exclude_cylinder_center_xy_override}."
            )
        exclude_cylinder_radius_value = float(exclude_cylinder_radius)
        if exclude_cylinder_radius_value <= 0.0:
            raise ValueError(f"exclude_cylinder_radius must be positive, got {exclude_cylinder_radius}.")
        exclude_cylinder_center_xy = torch.tensor(
            [float(value) for value in exclude_cylinder_center_xy_override],
            device=device,
            dtype=torch.float32,
        )
        if exclude_cylinder_z_min is not None:
            exclude_cylinder_z_min_value = float(exclude_cylinder_z_min)
        if exclude_cylinder_z_max is not None:
            exclude_cylinder_z_max_value = float(exclude_cylinder_z_max)
        if exclude_cylinder_z_max_value <= exclude_cylinder_z_min_value:
            raise ValueError(
                "exclude_cylinder_z_max must be greater than exclude_cylinder_z_min, "
                f"got {exclude_cylinder_z_min_value}..{exclude_cylinder_z_max_value}."
            )

    table_plane_normal_tensor = None
    if table_plane_normal is not None:
        if len(table_plane_normal) != 3:
            raise ValueError(f"table_plane_normal must contain exactly 3 values, got {table_plane_normal}.")
        table_plane_normal_tensor = torch.tensor(
            [float(value) for value in table_plane_normal],
            device=device,
            dtype=torch.float32,
        )

    pc_bt_pf = depth_to_point_cloud_torch(
        intrinsics=intr_bt_c33,
        depth=depth_btchw,
        extrinsics=extr_bt_c44,
        depth_scale=depth_scale,
        crop_area_min=crop_area_min,
        crop_area_max=crop_area_max,
        exclude_box_min=exclude_box_min,
        exclude_box_max=exclude_box_max,
        exclude_cylinder_center_xy=exclude_cylinder_center_xy,
        exclude_cylinder_radius=exclude_cylinder_radius_value,
        exclude_cylinder_z_min=exclude_cylinder_z_min_value,
        exclude_cylinder_z_max=exclude_cylinder_z_max_value,
        min_depth_m=-1.0 if min_depth_m is None else float(min_depth_m),
        max_depth_m=-1.0 if max_depth_m is None else float(max_depth_m),
        table_plane_normal=table_plane_normal_tensor,
        table_plane_d=0.0 if table_plane_d is None else float(table_plane_d),
        table_plane_clearance_m=float(table_plane_clearance_m),
        num_points=num_points,
        features=rgb_btchw3,
    )  # (B*T, P, 3) or (B*T, P, 6)

    feature_dim = pc_bt_pf.shape[-1]
    output_points = pc_bt_pf.shape[1]

    # Reshape back to (T, B, P, F)
    pc_t_b_pf = pc_bt_pf.reshape(B, T, output_points, feature_dim).permute(1, 0, 2, 3).contiguous()
    return pc_t_b_pf
@torch.jit.script
def depth_to_point_cloud_torch(
    intrinsics: torch.Tensor,  # (B, N, 3, 3)
    depth: torch.Tensor,  # (B, N, H, W)
    extrinsics: torch.Tensor,  # (B, N, 4, 4)
    depth_scale: float = 1000.0,
    crop_area_min: torch.Tensor | None = None,  # (3,)
    crop_area_max: torch.Tensor | None = None,  # (3,)
    exclude_box_min: torch.Tensor | None = None,  # (3,)
    exclude_box_max: torch.Tensor | None = None,  # (3,)
    exclude_cylinder_center_xy: torch.Tensor | None = None,  # (2,)
    exclude_cylinder_radius: float = -1.0,
    exclude_cylinder_z_min: float = -1.0e30,
    exclude_cylinder_z_max: float = 1.0e30,
    min_depth_m: float = -1.0,
    max_depth_m: float = -1.0,
    table_plane_normal: torch.Tensor | None = None,  # (3,)
    table_plane_d: float = 0.0,
    table_plane_clearance_m: float = 0.0,
    num_points: int = -1,
    features: torch.Tensor | None = None,  # (B, N, H, W, F) or None
) -> torch.Tensor:
    """
    Vectorized depth->point cloud with optional camera fusion across N and sampling.

    Returns:
        (B, P, 3) if features is None, else (B, P, 3+F)
    """
    B, N = depth.shape[0], depth.shape[1]
    H, W = depth.shape[-2], depth.shape[-1]
    device = depth.device

    # Build pixel grid on the fly (TorchScript-friendly)
    # u in [0..W-1], v in [0..H-1], with ij indexing
    v = torch.arange(H, device=device).view(H, 1).expand(H, W)
    u = torch.arange(W, device=device).view(1, W).expand(H, W)
    uv = torch.stack((u, v), dim=-1)  # (H, W, 2)
    uv = uv.view(1, 1, H, W, 2).expand(B, N, H, W, 2)

    fx = intrinsics[..., 0, 0].view(B, N, 1, 1)
    fy = intrinsics[..., 1, 1].view(B, N, 1, 1)
    cx = intrinsics[..., 0, 2].view(B, N, 1, 1)
    cy = intrinsics[..., 1, 2].view(B, N, 1, 1)

    # Depth to z; validity is checked on projected XYZ below before sampling.
    z = depth / depth_scale  # (B, N, H, W)
    valid = torch.isfinite(z) & (z > 0.0)
    if min_depth_m >= 0.0:
        valid &= z >= min_depth_m
    if max_depth_m >= 0.0:
        valid &= z <= max_depth_m

    u_img = uv[..., 0]  # (B, N, H, W)
    v_img = uv[..., 1]

    x = (u_img - cx) * z / fx
    y = (v_img - cy) * z / fy

    pc = torch.stack((x, y, z), dim=-1)  # (B, N, H, W, 3)

    # Apply extrinsics: (B*N, H*W, 4) @ (B*N, 4, 4)^T
    if extrinsics is not None:
        E = extrinsics.reshape(B * N, 4, 4)
        homog = torch.cat(
            (pc.reshape(B * N, H * W, 3), torch.ones(B * N, H * W, 1, device=device)),
            dim=-1,
        )
        world = torch.bmm(homog, E.transpose(1, 2))[..., :3]  # (B*N, H*W, 3)
        pc = world.reshape(B, N, H, W, 3)

    # Optional crop in XYZ (world coordinates)
    if (crop_area_min is not None) or (crop_area_max is not None):
        m = torch.ones_like(pc[..., 0], dtype=torch.bool)  # (B, N, H, W)
        if crop_area_min is not None:
            m &= pc[..., 0] > crop_area_min[0]
            m &= pc[..., 1] > crop_area_min[1]
            m &= pc[..., 2] > crop_area_min[2]
        if crop_area_max is not None:
            m &= pc[..., 0] < crop_area_max[0]
            m &= pc[..., 1] < crop_area_max[1]
            m &= pc[..., 2] < crop_area_max[2]
        valid &= m

    # Optional exclusion box in XYZ (world coordinates). This removes only the
    # intersection of the provided bounds instead of cropping whole halfspaces.
    if (exclude_box_min is not None) or (exclude_box_max is not None):
        exclude = torch.ones_like(pc[..., 0], dtype=torch.bool)
        if exclude_box_min is not None:
            exclude &= pc[..., 0] > exclude_box_min[0]
            exclude &= pc[..., 1] > exclude_box_min[1]
            exclude &= pc[..., 2] > exclude_box_min[2]
        if exclude_box_max is not None:
            exclude &= pc[..., 0] < exclude_box_max[0]
            exclude &= pc[..., 1] < exclude_box_max[1]
            exclude &= pc[..., 2] < exclude_box_max[2]
        valid &= ~exclude

    # Optional vertical cylinder exclusion in XYZ (world coordinates).
    if exclude_cylinder_center_xy is not None and exclude_cylinder_radius > 0.0:
        dx = pc[..., 0] - exclude_cylinder_center_xy[0]
        dy = pc[..., 1] - exclude_cylinder_center_xy[1]
        exclude_cylinder = (dx * dx + dy * dy) < (exclude_cylinder_radius * exclude_cylinder_radius)
        exclude_cylinder &= pc[..., 2] > exclude_cylinder_z_min
        exclude_cylinder &= pc[..., 2] < exclude_cylinder_z_max
        valid &= ~exclude_cylinder

    # Optional table plane removal in world/robot-frame coordinates.
    # Keeps points above the plane by a positive signed-distance clearance.
    if table_plane_normal is not None:
        signed_distance = (
            pc[..., 0] * table_plane_normal[0]
            + pc[..., 1] * table_plane_normal[1]
            + pc[..., 2] * table_plane_normal[2]
            + table_plane_d
        )
        valid &= signed_distance > table_plane_clearance_m

    valid &= torch.isfinite(pc).all(dim=-1)

    # Concatenate features if provided
    if features is not None:
        # features: (B, N, H, W, F)
        valid &= torch.isfinite(features).all(dim=-1)
        pc = torch.cat((pc, features), dim=-1)

    F = pc.shape[-1]
    pc_flat = pc.reshape(B, N * H * W, F)  # (B, M, F)
    mask_flat = valid.reshape(B, N * H * W)  # (B, M)

    max_points = N * H * W
    if num_points <= 0:
        out = torch.full((B, max_points, F), float("nan"), device=device, dtype=pc.dtype)
        for b in range(B):
            pts_b = pc_flat[b, mask_flat[b]]
            num_valid = pts_b.size(0)
            if num_valid == 0:
                continue
            out[b, :num_valid] = pts_b
        return out  # (B, max_points, 3) or (B, max_points, 3+F)

    # Output. Keep all rows NaN for a batch item with no finite points so the
    # saver records a count of zero instead of fabricating origin points.
    out = torch.full((B, num_points, F), float("nan"), device=device, dtype=pc.dtype)

    # Per-batch sampling
    for b in range(B):
        pts_b = pc_flat[b, mask_flat[b]]
        num_valid = pts_b.size(0)
        if num_valid == 0:
            continue
        if num_valid >= num_points:
            idx = torch.randperm(num_valid, device=device)[:num_points]
        else:
            idx = torch.randint(0, num_valid, (num_points,), device=device)
        out[b] = pts_b[idx]

    return out  # (B, num_points, 3) or (B, num_points, 3+F)


def _replace_nan_with_samples(pc: torch.Tensor) -> torch.Tensor:
    """Fill non-finite rows in a fused point cloud (T, B, P, F) by sampling
    with replacement from finite rows of the same (t, b). A (t, b) with zero
    finite rows is left untouched.
    """
    if pc.numel() == 0:
        return pc
    out = pc
    T, B, P, F = out.shape
    flat = out.reshape(T * B, P, F)
    for i in range(T * B):
        invalid_mask = ~torch.isfinite(flat[i]).all(dim=-1)
        num_invalid = int(invalid_mask.sum().item())
        if num_invalid == 0:
            continue
        valid_idx = (~invalid_mask).nonzero(as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            continue
        sample_idx = valid_idx[torch.randint(0, valid_idx.numel(), (num_invalid,), device=flat.device)]
        flat[i, invalid_mask] = flat[i, sample_idx]
    return flat.reshape(T, B, P, F)


class DemonstrationRecorder:
    """Record synchronized robot trajectories and optional camera/point-cloud data."""

    def __init__(
        self,
        base_dir: str,
        fps: float = 60,
        chunk_size: int = -1,
        compression_filter: str | None = "lzf",
        compression_opts: int | None = None,
        save_states_only: bool = False,
        filter_skipped_steps: bool = False,
        point_cloud_num_points: int | None = None,
        point_cloud_crop_to_workspace: bool = True,
        point_cloud_crop_z_only: bool = False,
        point_cloud_workspace_min: Sequence[float | None] | None = None,
        point_cloud_workspace_max: Sequence[float | None] | None = None,
        point_cloud_exclude_box_min: Sequence[float | None] | None = None,
        point_cloud_exclude_box_max: Sequence[float | None] | None = None,
        point_cloud_exclude_cylinder_center_xy: Sequence[float] | None = None,
        point_cloud_exclude_cylinder_radius: float | None = None,
        point_cloud_exclude_cylinder_z_min: float | None = None,
        point_cloud_exclude_cylinder_z_max: float | None = None,
        point_cloud_min_depth_m: float | None = None,
        point_cloud_max_depth_m: float | None = None,
        point_cloud_table_plane_normal: Sequence[float] | None = None,
        point_cloud_table_plane_d: float | None = None,
        point_cloud_table_object_clearance_m: float = 0.0,
        camera_buffer_device: str | None = None,
        point_cloud_buffer_device: str | None = None,
        camera_keys: Sequence[str] | None = None,
        eager_point_cloud: bool = False,
        write_rgb: bool = False,
        write_point_cloud: bool = True,
        write_point_cloud_flat: bool = True,
        record_stride: int = 1,
    ):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        self.metadata_info = {"num_timesteps": [], "num_episodes": 0}
        self.fps = fps
        self.record_stride = int(record_stride)
        if self.record_stride < 1:
            raise ValueError("record_stride must be positive")
        self.save_states_only = save_states_only
        self.filter_skipped_steps = filter_skipped_steps
        self.point_cloud_num_points = (
            POINT_CLOUD_NUM_POINTS if point_cloud_num_points is None else int(point_cloud_num_points)
        )
        self.point_cloud_crop_to_workspace = bool(point_cloud_crop_to_workspace)
        self.point_cloud_crop_z_only = bool(point_cloud_crop_z_only)
        self.point_cloud_workspace_min = (
            tuple(point_cloud_workspace_min) if point_cloud_workspace_min is not None else None
        )
        self.point_cloud_workspace_max = (
            tuple(point_cloud_workspace_max) if point_cloud_workspace_max is not None else None
        )
        self.point_cloud_exclude_box_min = (
            tuple(point_cloud_exclude_box_min) if point_cloud_exclude_box_min is not None else None
        )
        self.point_cloud_exclude_box_max = (
            tuple(point_cloud_exclude_box_max) if point_cloud_exclude_box_max is not None else None
        )
        self.point_cloud_exclude_cylinder_center_xy = (
            tuple(float(value) for value in point_cloud_exclude_cylinder_center_xy)
            if point_cloud_exclude_cylinder_center_xy is not None
            else None
        )
        self.point_cloud_exclude_cylinder_radius = (
            None if point_cloud_exclude_cylinder_radius is None else float(point_cloud_exclude_cylinder_radius)
        )
        self.point_cloud_exclude_cylinder_z_min = (
            None if point_cloud_exclude_cylinder_z_min is None else float(point_cloud_exclude_cylinder_z_min)
        )
        self.point_cloud_exclude_cylinder_z_max = (
            None if point_cloud_exclude_cylinder_z_max is None else float(point_cloud_exclude_cylinder_z_max)
        )
        self.point_cloud_min_depth_m = None if point_cloud_min_depth_m is None else float(point_cloud_min_depth_m)
        self.point_cloud_max_depth_m = None if point_cloud_max_depth_m is None else float(point_cloud_max_depth_m)
        self.point_cloud_table_plane_normal = (
            tuple(float(value) for value in point_cloud_table_plane_normal)
            if point_cloud_table_plane_normal is not None
            else None
        )
        self.point_cloud_table_plane_d = (
            None if point_cloud_table_plane_d is None else float(point_cloud_table_plane_d)
        )
        self.point_cloud_table_object_clearance_m = float(point_cloud_table_object_clearance_m)
        self.camera_buffer_device = str(
            camera_buffer_device if camera_buffer_device is not None else os.environ.get("KUAFU_CAMERA_BUFFER_DEVICE", "cpu")
        ).lower()
        self.point_cloud_buffer_device = str(
            point_cloud_buffer_device
            if point_cloud_buffer_device is not None
            else os.environ.get("KUAFU_POINT_CLOUD_BUFFER_DEVICE", "cpu")
        ).lower()
        self.camera_keys_filter = tuple(str(key) for key in camera_keys) if camera_keys is not None else None
        self.write_rgb = bool(write_rgb)
        self.write_point_cloud = bool(write_point_cloud)
        self.write_point_cloud_flat = bool(write_point_cloud_flat) and self.write_point_cloud
        self.eager_point_cloud = bool(eager_point_cloud) and self.write_point_cloud
        for label, bounds in (
            ("point_cloud_workspace_min", self.point_cloud_workspace_min),
            ("point_cloud_workspace_max", self.point_cloud_workspace_max),
            ("point_cloud_exclude_box_min", self.point_cloud_exclude_box_min),
            ("point_cloud_exclude_box_max", self.point_cloud_exclude_box_max),
            ("point_cloud_table_plane_normal", self.point_cloud_table_plane_normal),
        ):
            if bounds is not None and len(bounds) != 3:
                raise ValueError(f"{label} must contain exactly 3 values, got {bounds}.")
        if self.point_cloud_exclude_cylinder_radius is not None:
            if self.point_cloud_exclude_cylinder_center_xy is None:
                raise ValueError(
                    "point_cloud_exclude_cylinder_center_xy is required when "
                    "point_cloud_exclude_cylinder_radius is set."
                )
            if len(self.point_cloud_exclude_cylinder_center_xy) != 2:
                raise ValueError(
                    "point_cloud_exclude_cylinder_center_xy must contain exactly 2 values, "
                    f"got {self.point_cloud_exclude_cylinder_center_xy}."
                )
            if self.point_cloud_exclude_cylinder_radius <= 0.0:
                raise ValueError(
                    "point_cloud_exclude_cylinder_radius must be positive, "
                    f"got {self.point_cloud_exclude_cylinder_radius}."
                )
            if (
                self.point_cloud_exclude_cylinder_z_min is not None
                and self.point_cloud_exclude_cylinder_z_max is not None
                and self.point_cloud_exclude_cylinder_z_max <= self.point_cloud_exclude_cylinder_z_min
            ):
                raise ValueError(
                    "point_cloud_exclude_cylinder_z_max must be greater than "
                    "point_cloud_exclude_cylinder_z_min."
                )

        if self.point_cloud_table_plane_normal is not None:
            normal_norm = float(np.linalg.norm(np.asarray(self.point_cloud_table_plane_normal, dtype=np.float32)))
            if normal_norm < 1e-8:
                raise ValueError("point_cloud_table_plane_normal must be non-zero.")
            self.point_cloud_table_plane_normal = tuple(
                float(value) / normal_norm for value in self.point_cloud_table_plane_normal
            )
            if self.point_cloud_table_plane_d is None:
                raise ValueError("point_cloud_table_plane_d is required when point_cloud_table_plane_normal is set.")
        self.chunk_size = chunk_size
        self._file_handles = {}
        self._created_datasets = {}
        self._episode_tmp_paths = {}
        self.max_workers = min(int(os.cpu_count() or 1) // 2, 16)

        self._timestamp_counter = {}
        self._initial_sample_skipped = {}
        self._previous_action_with_gripper = {}
        self._previous_quaternions: dict[tuple[int, str], np.ndarray] = {}
        self.cam_keys = None
        self._actuated_joint_ids = None

        self.env_origins = None
        self.task_description = None

        # Lazily-resolved body indices for link7 and fingertip links, plus width offset.
        self._arm_body_indices: tuple[int, int, int, float] | None = None

        # Lazily-cached env action mode and gripper calibration for mapping
        # commanded gripper targets to the reference-style binary action bit.
        self._action_mode: str | None = None
        self._gripper_opened_np: np.ndarray | None = None
        self._gripper_closed_np: np.ndarray | None = None

        self.reset_buffers()

    def reset_buffers(self):
        self.arm_ee_pose = []
        self.ee_positions = []
        # Public dataset: measured panda_fingertip quaternion in Isaac order [qw, qx, qy, qz].
        self.ee_rotations = []
        # Internal measured panda_fingertip pose-matrix source used for eef_pose.
        self.ee_rotation_matrices = []
        # Absolute panda_fingertip command pose from action_mode='absolute_ee'.
        self.ee_command_positions = []
        self.ee_command_rotations = []
        self.arm_gripper_widths = []
        self.arm_joint_positions = []
        self.all_joint_positions = []
        self.all_joint_velocities = []
        self.joint_positions = []
        self.joint_velocities = []
        self.joint_targets = []  # Store joint targets (actions) passed to store()
        self.robot_dof_targets = []
        self.rigid_obj_positions = {}
        self.rigid_obj_rotations = {}
        self.articulated_joint_positions = {}
        self.articulated_joint_velocities = {}
        self.articulated_root_positions = {}
        self.articulated_root_rotations = {}
        self.rgbs = []
        self.depths = []
        self.extrinsics = []
        self.intrinsics = []
        self.point_cloud = None
        self.point_cloud_counts = None
        self.point_cloud_frames = []
        self.point_cloud_count_frames = []
        self._point_cloud_env_ids = None
        self.stored_env_ids_per_timestep = []  # Track which env_ids were stored at each timestep
        self.skill_dones = []
        self.decision_stages = []
        self.decision_valid = []
        self.observed_grippers = []

    def store(
        self, actions: torch.Tensor, env, env_ids: list[int] | torch.Tensor | None = None,
        *, decision_stage=None,
    ) -> None:
        """Record the current observation and command before env.step()."""
        stages, stage_valid, stage_count = snapshot_decision_stage(env, decision_stage)

        if len(self.ee_positions) == 0:
            print("[DemonstrationRecorder] Starting a new data chunk ...")

        # Handle env_ids parameter
        if env_ids is None:
            env_ids_list: list[int] = list(range(actions.shape[0]))
        elif isinstance(env_ids, torch.Tensor):
            env_ids_list = env_ids.tolist()  # type: ignore[union-attr]
        else:
            env_ids_list = env_ids
        # env_ids_list is now a list of ints

        for env_id in env_ids_list:
            self._open_episode_if_needed(env_id)
            self._file_handles[env_id].attrs["decision_stage_timing"] = STAGE_TIMING
            if stage_count is not None:
                self._file_handles[env_id].attrs["decision_stage_count"] = stage_count

        # save env origins once
        if self.env_origins is None:
            self.env_origins = env.unwrapped.scene.env_origins

        # save task description once
        if self.task_description is None:
            cfg = env.unwrapped.cfg
            episode_cfg = getattr(cfg, "episode_cfg", {}) or {}
            self.task_description = episode_cfg.get("task_description", getattr(cfg, "task_description", ""))

        # Cache the env's action mode and gripper calibration once. Used by
        # ``write()`` to derive the binary gripper channel using the same rule
        # the env applies in ``_process_gripper_actions``.
        if self._action_mode is None:
            self._action_mode = str(getattr(env.unwrapped.cfg, "action_mode", "absolute_ee"))
            try:
                api = env.unwrapped.program_runtime.controller_api_wrapper
                self._gripper_opened_np = api._gripper_opened.detach().cpu().numpy().astype(np.float32).reshape(-1)
                self._gripper_closed_np = api._gripper_closed.detach().cpu().numpy().astype(np.float32).reshape(-1)
            except Exception:
                # Fallback: leave unset; ``write()`` will fall back to a
                # sign-based interpretation if calibration is unavailable.
                self._gripper_opened_np = None
                self._gripper_closed_np = None

        # ee position/rotation and joint positions/velocities
        robot_art = env.unwrapped.scene._articulations["robot"]
        if self._actuated_joint_ids is None:
            robot_cfg = getattr(env.unwrapped.cfg, "robot", None)
            if robot_cfg is None:
                robot_cfg = getattr(env.unwrapped.scene.cfg, "robot", None)
            if robot_cfg is None:
                raise AttributeError("Could not find robot config on env.cfg or scene.cfg")
            actuated_joints_names = list(robot_cfg.arm_actuated_joints) + list(robot_cfg.gripper_actuated_joints)
            self._actuated_joint_ids, _ = robot_art.find_joints(actuated_joints_names, preserve_order=True)
        joint_ids = self._actuated_joint_ids

        joint_pos = robot_art.data.joint_pos[:, joint_ids]
        joint_vel = robot_art.data.joint_vel[:, joint_ids]
        all_joint_pos = robot_art.data.joint_pos
        all_joint_vel = robot_art.data.joint_vel
        arm_joint_pos = joint_pos[:, :-2]

        # Resolve body indices once. The arm body index is kept for backwards
        # compatibility with older robots; recorded EE pose below is measured
        # panda_fingertip, matching the policy observation frame.
        if self._arm_body_indices is None:
            self._arm_body_indices = resolve_arm_body_indices(env.unwrapped)
        _arm_body_idx, left_finger_idx, right_finger_idx, finger_width_offset = self._arm_body_indices

        ee_position, ee_quat_wxyz, ee_rotation, gripper_width = get_measured_fingertip_pose_and_gripper_width(
            env.unwrapped, left_finger_idx, right_finger_idx, finger_width_offset
        )

        # Build the measured panda_fingertip pose as [x, y, z, qx, qy, qz, qw]
        # (xyzw — note this differs from Isaac Sim's wxyz convention).
        ee_quat_xyzw = ee_quat_wxyz[..., [1, 2, 3, 0]]
        arm_ee_pose = torch.cat([ee_position, ee_quat_xyzw], dim=-1)  # (N, 7)

        ee_cmd_pose: tuple[torch.Tensor, torch.Tensor] | None = None
        if (self._action_mode or "").lower() == "absolute_ee":
            ee_cmd_pose = get_absolute_fingertip_command_pose_from_action(actions)

        # Index data for specified env_ids
        env_ids_tensor = torch.tensor(env_ids_list, device=ee_position.device)
        self.arm_ee_pose.append(arm_ee_pose[env_ids_tensor].clone())
        self.ee_positions.append(ee_position[env_ids_tensor].clone())
        self.ee_rotations.append(ee_quat_wxyz[env_ids_tensor].clone())
        self.ee_rotation_matrices.append(ee_rotation[env_ids_tensor].clone())
        self.arm_gripper_widths.append(gripper_width[env_ids_tensor].clone())
        if ee_cmd_pose is not None:
            ee_cmd_position, ee_cmd_rotation = ee_cmd_pose
            ee_cmd_position = ee_cmd_position.to(ee_position.device)
            ee_cmd_rotation = ee_cmd_rotation.to(ee_position.device)
            self.ee_command_positions.append(ee_cmd_position[env_ids_tensor].clone())
            self.ee_command_rotations.append(ee_cmd_rotation[env_ids_tensor].clone())
        self.arm_joint_positions.append(arm_joint_pos[env_ids_tensor].clone())
        self.all_joint_positions.append(all_joint_pos[env_ids_tensor].clone())
        self.all_joint_velocities.append(all_joint_vel[env_ids_tensor].clone())
        self.joint_positions.append(joint_pos[env_ids_tensor].clone())
        self.joint_velocities.append(joint_vel[env_ids_tensor].clone())
        # Save the command/target tensor passed to store(), aligned to env_ids.
        actions_for_save = actions.to(ee_position.device)
        if actions_for_save.shape[0] == ee_position.shape[0]:
            actions_for_save = actions_for_save[env_ids_tensor]
        self.joint_targets.append(actions_for_save.clone())
        self.decision_stages.append(stages[env_ids_tensor].clone())
        self.decision_valid.append(stage_valid[env_ids_tensor].clone())
        api = getattr(getattr(env.unwrapped, "program_runtime", None), "controller_api_wrapper", None)
        if api is not None:
            observed_gripper = api.gripper_openness().reshape(-1).to(ee_position.device).clamp(0, 1)
        else:
            observed_gripper = torch.full_like(stages, float("nan"), dtype=torch.float32)
        self.observed_grippers.append(observed_gripper[env_ids_tensor].clone())
        # Track which env_ids were stored at this timestep
        self.stored_env_ids_per_timestep.append(env_ids_list.copy())

        # Save robot DOF targets when the env exposes them. Learned FR3FinRay envs
        # keep these in ``_robot_dof_targets``; older envs use ``robot_dof_targets``.
        robot_dof_targets = _get_robot_dof_targets(env.unwrapped)
        if robot_dof_targets is not None:
            robot_dof_targets = robot_dof_targets.to(ee_position.device)
            if robot_dof_targets.shape[0] == ee_position.shape[0]:
                robot_dof_targets = robot_dof_targets[env_ids_tensor]
            self.robot_dof_targets.append(robot_dof_targets.clone())

        # save skill done
        skill_done = getattr(env, "skill_done", None)
        if skill_done is None:
            skill_done = torch.zeros(actions.shape[0], dtype=torch.bool, device=actions.device)
        self.skill_dones.append(skill_done.clone())

        # rigid object positions and rotations
        for object_id in env.unwrapped.scene.rigid_objects:
            if object_id not in self.rigid_obj_positions:
                self.rigid_obj_positions[object_id] = []
                self.rigid_obj_rotations[object_id] = []
            pos = env.unwrapped.scene._rigid_objects[object_id].data.root_pos_w[..., :3]
            pos -= env.unwrapped.scene.env_origins
            rot = env.unwrapped.scene._rigid_objects[object_id].data.root_quat_w
            self.rigid_obj_positions[object_id].append(pos[env_ids_tensor].clone())
            self.rigid_obj_rotations[object_id].append(rot[env_ids_tensor].clone())

        # articulated object positions and rotations
        for object_id in env.unwrapped.scene._articulations:
            if object_id == "robot":
                continue
            if object_id not in self.articulated_joint_positions:
                self.articulated_joint_positions[object_id] = []
                self.articulated_joint_velocities[object_id] = []
                self.articulated_root_positions[object_id] = []
                self.articulated_root_rotations[object_id] = []
            joint_pos = env.unwrapped.scene._articulations[object_id].data.joint_pos
            joint_vel = env.unwrapped.scene._articulations[object_id].data.joint_vel
            root_pos = env.unwrapped.scene._articulations[object_id].data.root_pos_w[..., :3]
            root_pos -= env.unwrapped.scene.env_origins
            root_rot = env.unwrapped.scene._articulations[object_id].data.root_quat_w
            self.articulated_joint_positions[object_id].append(joint_pos[env_ids_tensor].clone())
            self.articulated_joint_velocities[object_id].append(joint_vel[env_ids_tensor].clone())
            self.articulated_root_positions[object_id].append(root_pos[env_ids_tensor].clone())
            self.articulated_root_rotations[object_id].append(root_rot[env_ids_tensor].clone())

        if not self.save_states_only:
            # save cam keys once
            if self.cam_keys is None:
                available_cam_keys = [
                    sensor_key for sensor_key in env.unwrapped.scene.sensors.keys() if "cam" in sensor_key.lower()
                ]
                if self.camera_keys_filter is None:
                    self.cam_keys = available_cam_keys
                else:
                    available_set = set(available_cam_keys)
                    missing = [cam_key for cam_key in self.camera_keys_filter if cam_key not in available_set]
                    if missing:
                        raise ValueError(
                            "Configured DemonstrationRecorder camera_keys are not available: "
                            f"{missing}. Available camera sensors: {available_cam_keys}"
                        )
                    self.cam_keys = [cam_key for cam_key in self.camera_keys_filter if cam_key in available_set]
                    print(f"[DemonstrationRecorder] Camera allowlist active: {self.cam_keys}")

            metadata_env_ids = [
                env_id for env_id in env_ids_list
                if "point_cloud_recording" not in self._file_handles[env_id].attrs
            ]
            if metadata_env_ids:
                recording = {
                    "program_stages": program_stage_schema(env),
                    "camera_keys": [key for key in self.cam_keys if "wrist" not in key.lower()],
                    "control_dt": float(env.unwrapped.step_dt),
                    "record_stride": self.record_stride,
                    "fusion": {
                        "num_points": self.point_cloud_num_points,
                        "crop_to_workspace": self.point_cloud_crop_to_workspace,
                        "crop_z_only": self.point_cloud_crop_z_only,
                        "crop_area_min_override": self.point_cloud_workspace_min,
                        "crop_area_max_override": self.point_cloud_workspace_max,
                        "exclude_box_min_override": self.point_cloud_exclude_box_min,
                        "exclude_box_max_override": self.point_cloud_exclude_box_max,
                        "exclude_cylinder_center_xy_override": self.point_cloud_exclude_cylinder_center_xy,
                        "exclude_cylinder_radius": self.point_cloud_exclude_cylinder_radius,
                        "exclude_cylinder_z_min": self.point_cloud_exclude_cylinder_z_min,
                        "exclude_cylinder_z_max": self.point_cloud_exclude_cylinder_z_max,
                        "min_depth_m": self.point_cloud_min_depth_m,
                        "max_depth_m": self.point_cloud_max_depth_m,
                        "table_plane_normal": self.point_cloud_table_plane_normal,
                        "table_plane_d": self.point_cloud_table_plane_d,
                        "table_plane_clearance_m": self.point_cloud_table_object_clearance_m,
                    },
                }
                for env_id in metadata_env_ids:
                    self._file_handles[env_id].attrs["point_cloud_recording"] = json.dumps(recording)

            def buffer_tensor(tensor: torch.Tensor, device_name: str) -> torch.Tensor:
                tensor = tensor.detach()
                if device_name in {"cuda", "gpu"} and tensor.is_cuda:
                    return tensor.clone()
                return tensor.cpu().clone()

            eager_depths = []
            eager_rgbs = []
            eager_intrinsics = []
            eager_extrinsics = []

            for cam_key in self.cam_keys:
                sensor = env.unwrapped.scene.sensors[cam_key]

                # save extrinsics per timestep
                position = sensor.data.pos_w - env.unwrapped.scene.env_origins
                rotation_ros = sensor.data.quat_w_ros
                rotation_matrix_world = matrix_from_quat(rotation_ros)
                extrinsic_matrix = self.get_extrinsics(rotation_matrix_world, position)
                extrinsic_slice = extrinsic_matrix[env_ids_tensor].detach()
                self.extrinsics.append(buffer_tensor(extrinsic_slice, self.camera_buffer_device))

                # save intrinsic matrices per timestep
                intrinsic_slice = sensor.data.intrinsic_matrices[env_ids_tensor].detach()
                self.intrinsics.append(buffer_tensor(intrinsic_slice, self.camera_buffer_device))

                rgb_slice = sensor.data.output["rgb"][env_ids_tensor].detach()
                depth_slice = sensor.data.output["distance_to_image_plane"][env_ids_tensor].detach()

                if self.write_rgb:
                    self.rgbs.append(buffer_tensor(rgb_slice, self.camera_buffer_device))

                if self.eager_point_cloud and "wrist" not in cam_key.lower():
                    eager_depths.append(depth_slice)
                    eager_rgbs.append(rgb_slice)
                    eager_intrinsics.append(intrinsic_slice)
                    eager_extrinsics.append(extrinsic_slice)
                elif not self.eager_point_cloud:
                    if not self.write_rgb:
                        self.rgbs.append(buffer_tensor(rgb_slice, self.camera_buffer_device))
                    self.depths.append(buffer_tensor(depth_slice, self.camera_buffer_device))

            if self.eager_point_cloud:
                if len(eager_depths) == 0:
                    raise ValueError("No external cameras found for eager point cloud fusion.")
                with torch.inference_mode():
                    point_cloud = fused_pc(
                        depths=eager_depths,
                        rgbs=eager_rgbs,
                        intrinsics=eager_intrinsics,
                        extrinsics=eager_extrinsics,
                        num_cams=len(eager_depths),
                        num_points=self.point_cloud_num_points,
                        depth_scale=1.0,
                        crop_to_workspace=self.point_cloud_crop_to_workspace,
                        crop_z_only=self.point_cloud_crop_z_only,
                        crop_area_min_override=self.point_cloud_workspace_min,
                        crop_area_max_override=self.point_cloud_workspace_max,
                        exclude_box_min_override=self.point_cloud_exclude_box_min,
                        exclude_box_max_override=self.point_cloud_exclude_box_max,
                        exclude_cylinder_center_xy_override=self.point_cloud_exclude_cylinder_center_xy,
                        exclude_cylinder_radius=self.point_cloud_exclude_cylinder_radius,
                        exclude_cylinder_z_min=self.point_cloud_exclude_cylinder_z_min,
                        exclude_cylinder_z_max=self.point_cloud_exclude_cylinder_z_max,
                        min_depth_m=self.point_cloud_min_depth_m,
                        max_depth_m=self.point_cloud_max_depth_m,
                        table_plane_normal=self.point_cloud_table_plane_normal,
                        table_plane_d=self.point_cloud_table_plane_d,
                        table_plane_clearance_m=self.point_cloud_table_object_clearance_m,
                    )
                    point_cloud_counts = (
                        torch.isfinite(point_cloud[..., :3]).all(dim=-1).sum(dim=-1).to(torch.int32).squeeze(0)
                    )
                    point_cloud_frame = _replace_nan_with_samples(point_cloud).squeeze(0)
                self.point_cloud_frames.append(
                    buffer_tensor(point_cloud_frame, self.point_cloud_buffer_device)
                )
                self.point_cloud_count_frames.append(point_cloud_counts.cpu().clone())

        if len(self.ee_positions) == self.chunk_size:
            print("[DemonstrationRecorder] Saving and flushing data chunk ...")
            # Get all unique env_ids that were stored across all timesteps
            all_stored_env_ids = set()
            for env_ids_at_t in self.stored_env_ids_per_timestep:
                all_stored_env_ids.update(env_ids_at_t)
            self._flush_chunks_parallel(
                compute_point_cloud=(not self.save_states_only and self.write_point_cloud),
                env_ids=list(all_stored_env_ids),
            )

    def _flush_chunks_parallel(self, compute_point_cloud: bool = True, env_ids: list[int] | None = None):
        # Get all unique env_ids that were stored, or use provided env_ids.
        if env_ids is None:
            if len(self.stored_env_ids_per_timestep) > 0:
                env_ids = sorted(set().union(*self.stored_env_ids_per_timestep))
            else:
                batch_size = self.ee_positions[0].shape[0] if len(self.ee_positions) > 0 else 0
                env_ids = list(range(batch_size))
        else:
            env_ids = [int(env_id) for env_id in env_ids]

        batch_size = len(env_ids)
        self._point_cloud_env_ids = None

        if batch_size == 0:
            self.reset_buffers()
            print("[DemonstrationRecorder] No env chunks to flush; buffers cleared.")
            return

        def selected_row_indices(timestep: int) -> list[int]:
            stored_env_ids = self.stored_env_ids_per_timestep[timestep]
            missing = [env_id for env_id in env_ids if env_id not in stored_env_ids]
            if missing:
                raise ValueError(
                    f"Cannot point-cloud flush envs {missing}; timestep {timestep} only has {stored_env_ids}."
                )
            return [stored_env_ids.index(env_id) for env_id in env_ids]

        def select_rows(tensor: torch.Tensor, indices: list[int]) -> torch.Tensor:
            if len(indices) == int(tensor.shape[0]) and all(idx == pos for pos, idx in enumerate(indices)):
                return tensor
            index_tensor = torch.as_tensor(indices, device=tensor.device, dtype=torch.long)
            return tensor.index_select(0, index_tensor)

        if compute_point_cloud:
            self._point_cloud_env_ids = env_ids.copy()
            if self.eager_point_cloud:
                num_timesteps = len(self.point_cloud_frames)
                if num_timesteps == 0:
                    raise ValueError("No eager point-cloud frames found for point-cloud flush.")

                selected_frames = []
                selected_counts = []
                for timestep in range(num_timesteps):
                    row_indices = selected_row_indices(timestep)
                    selected_frames.append(select_rows(self.point_cloud_frames[timestep], row_indices))
                    selected_counts.append(select_rows(self.point_cloud_count_frames[timestep], row_indices))

                self.point_cloud = torch.stack(selected_frames, dim=0).cpu()
                self.point_cloud_counts = torch.stack(selected_counts, dim=0).to(torch.int32).cpu()
                print(
                    "[DemonstrationRecorder] Using eager point cloud buffer: "
                    f"{num_timesteps} timesteps, {batch_size} environments, "
                    f"{self.point_cloud_num_points} points/frame"
                )
            else:
                print(f"[DemonstrationRecorder] Computing point cloud for {batch_size} environments...")
                # Filter out wrist camera - only use external cameras for point cloud fusion.
                if self.cam_keys is not None:
                    external_cam_indices = [
                        idx for idx, cam_key in enumerate(self.cam_keys)
                        if "wrist" not in cam_key.lower()
                    ]
                    num_external_cams = len(external_cam_indices)

                    if num_external_cams == 0:
                        raise ValueError("No external cameras found for point cloud fusion.")

                    total_cams = len(self.cam_keys)
                    num_timesteps = len(self.depths) // total_cams if total_cams > 0 else 0

                    filtered_depths = []
                    filtered_rgbs = []
                    filtered_intrinsics = []
                    filtered_extrinsics = []

                    for t in range(num_timesteps):
                        row_indices = selected_row_indices(t)
                        for cam_idx in external_cam_indices:
                            list_idx = t * total_cams + cam_idx
                            filtered_depths.append(select_rows(self.depths[list_idx], row_indices))
                            filtered_rgbs.append(select_rows(self.rgbs[list_idx], row_indices))
                            filtered_intrinsics.append(select_rows(self.intrinsics[list_idx], row_indices))
                            filtered_extrinsics.append(select_rows(self.extrinsics[list_idx], row_indices))

                    has_custom_crop = (
                        self.point_cloud_workspace_min is not None
                        or self.point_cloud_workspace_max is not None
                    )
                    has_exclude_box = (
                        self.point_cloud_exclude_box_min is not None
                        or self.point_cloud_exclude_box_max is not None
                    )
                    has_exclude_cylinder = self.point_cloud_exclude_cylinder_radius is not None
                    crop_label = (
                        "custom"
                        if self.point_cloud_crop_to_workspace and has_custom_crop
                        else "workspace"
                        if self.point_cloud_crop_to_workspace
                        else "z-only"
                        if self.point_cloud_crop_z_only
                        else "disabled"
                    )
                    if has_custom_crop:
                        crop_label += f" min={self.point_cloud_workspace_min} max={self.point_cloud_workspace_max}"
                    if has_exclude_box:
                        crop_label += (
                            f" exclude_min={self.point_cloud_exclude_box_min}"
                            f" exclude_max={self.point_cloud_exclude_box_max}"
                        )
                    if has_exclude_cylinder:
                        crop_label += (
                            f" exclude_cylinder_center_xy={self.point_cloud_exclude_cylinder_center_xy}"
                            f" exclude_cylinder_radius={self.point_cloud_exclude_cylinder_radius}"
                            f" exclude_cylinder_z_min={self.point_cloud_exclude_cylinder_z_min}"
                            f" exclude_cylinder_z_max={self.point_cloud_exclude_cylinder_z_max}"
                        )
                    print(
                        "[DemonstrationRecorder] Point cloud fusion: "
                        f"{num_timesteps} timesteps, {num_external_cams} cameras, "
                        f"{self.point_cloud_num_points} points/frame, "
                        f"xyz_crop={crop_label}, "
                        f"chunk={POINT_CLOUD_TIMESTEP_CHUNK_SIZE} timesteps"
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    with torch.inference_mode():
                        point_cloud = fused_pc(
                            depths=filtered_depths,
                            rgbs=filtered_rgbs,
                            intrinsics=filtered_intrinsics,
                            extrinsics=filtered_extrinsics,
                            num_cams=num_external_cams,
                            num_points=self.point_cloud_num_points,
                            depth_scale=1.0,
                            crop_to_workspace=self.point_cloud_crop_to_workspace,
                            crop_z_only=self.point_cloud_crop_z_only,
                            crop_area_min_override=self.point_cloud_workspace_min,
                            crop_area_max_override=self.point_cloud_workspace_max,
                            exclude_box_min_override=self.point_cloud_exclude_box_min,
                            exclude_box_max_override=self.point_cloud_exclude_box_max,
                            exclude_cylinder_center_xy_override=self.point_cloud_exclude_cylinder_center_xy,
                            exclude_cylinder_radius=self.point_cloud_exclude_cylinder_radius,
                            exclude_cylinder_z_min=self.point_cloud_exclude_cylinder_z_min,
                            exclude_cylinder_z_max=self.point_cloud_exclude_cylinder_z_max,
                            min_depth_m=self.point_cloud_min_depth_m,
                            max_depth_m=self.point_cloud_max_depth_m,
                            table_plane_normal=self.point_cloud_table_plane_normal,
                            table_plane_d=self.point_cloud_table_plane_d,
                            table_plane_clearance_m=self.point_cloud_table_object_clearance_m,
                        )  # -> (T, B, P, 3/6)
                        self.point_cloud_counts = (
                            torch.isfinite(point_cloud[..., :3]).all(dim=-1).sum(dim=-1).to(torch.int32).cpu()
                        )
                        self.point_cloud = _replace_nan_with_samples(point_cloud).cpu()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    raise ValueError("No cameras found for point cloud fusion.")
                print("[DemonstrationRecorder] Point cloud computation completed.")

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._flush_chunk_for_env, env_id) for env_id in env_ids]
            for env_id, future in zip(env_ids, futures):
                future.result()
        print("[DemonstrationRecorder] All chunks flushed successfully.")
        self.reset_buffers()

    def get_extrinsics(self, rotation_matrices: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Compute batched extrinsic matrices.


        Args:
            rotation_matrices: (batch_size, 3, 3) tensor of rotation matrices
            positions: (batch_size, 3) tensor of translation vectors

        Returns:
            extrinsics: (batch_size, 4, 4) tensor of extrinsic matrices
        """
        batch_size = rotation_matrices.shape[0]
        extrinsics = torch.eye(4, device=rotation_matrices.device).unsqueeze(0).repeat(batch_size, 1, 1)
        extrinsics[:, :3, :3] = rotation_matrices
        extrinsics[:, :3, 3] = positions
        return extrinsics

    def write(self, dones, successes, extra_trajectory_level_info: dict[str, Any] = {}) -> int:
        """Finalize any episodes whose env has done=True."""
        dones_list = (
            [bool(v) for v in dones.detach().cpu().tolist()]
            if isinstance(dones, torch.Tensor)
            else [bool(v) for v in dones]
        )
        successes_list = (
            [bool(v) for v in successes.detach().cpu().tolist()]
            if isinstance(successes, torch.Tensor)
            else [bool(v) for v in successes]
        )
        if len(self.ee_positions) > 0:
            stored_env_ids = set()
            for env_ids_at_t in self.stored_env_ids_per_timestep:
                stored_env_ids.update(int(env_id) for env_id in env_ids_at_t)

            envs_to_flush = []
            for env_id in sorted(stored_env_ids):
                done = env_id < len(dones_list) and dones_list[env_id]
                success = env_id < len(successes_list) and successes_list[env_id]
                # Failed completed episodes are about to be deleted, so avoid
                # spending point-cloud/HDF5 time on their buffered frames.
                if (done and success) or not done:
                    envs_to_flush.append(env_id)

            if envs_to_flush:
                compute_point_cloud = not self.save_states_only and self.write_point_cloud
                self._flush_chunks_parallel(compute_point_cloud=compute_point_cloud, env_ids=envs_to_flush)
            else:
                self.reset_buffers()
        kept_paths = []
        for i, done in enumerate(dones_list):
            if not done:
                continue
            success_val = successes_list[i]

            kept, tmp_h5 = self._finalize_episode(i, success=success_val)
            if kept and tmp_h5:
                kept_paths.append((i, tmp_h5, os.path.join(self.base_dir, f"episode_tmp_env{i}"), success_val))

        for env_id, tmp_h5, tmp_root, success_val in kept_paths:
            episode_index = self.metadata_info["num_episodes"]
            # Create episode directory
            episode_dir = os.path.join(self.base_dir, f"episode_{episode_index}")
            os.makedirs(episode_dir, exist_ok=True)
            final_h5 = os.path.join(episode_dir, f"episode_{episode_index}.h5")
            final_dir = episode_dir

            # Rename temporary h5 file to final h5 file
            rename_success = False
            if os.path.exists(tmp_h5):
                try:
                    # Check if target already exists (shouldn't happen, but handle it)
                    if os.path.exists(final_h5):
                        print(f"[DemonstrationRecorder] Warning: Target file {final_h5} already exists, removing it...")
                        os.remove(final_h5)
                    os.rename(tmp_h5, final_h5)
                    rename_success = True
                except OSError as e:
                    print(f"[DemonstrationRecorder] Failed to rename {tmp_h5} to {final_h5}: {e}")
            else:
                print(f"[DemonstrationRecorder] Warning: Temporary file {tmp_h5} does not exist, skipping episode {episode_index}")

            # Clean up temporary path entry
            self._episode_tmp_paths.pop(env_id, None)

            # Skip processing if rename failed
            if not rename_success:
                # Still increment counter to avoid getting stuck on the same index
                self.metadata_info["num_episodes"] += 1
                continue

            if extra_trajectory_level_info:
                with h5py.File(final_h5, "a") as file_handle:
                    for key in extra_trajectory_level_info:
                        value = extra_trajectory_level_info[key][env_id]
                        # Convert complex Python objects (lists, dicts) to JSON strings for HDF5 compatibility
                        # Also handle numpy arrays with object dtype
                        if value is None:
                            continue
                        # Convert complex Python objects (lists, dicts) to JSON strings for HDF5 compatibility
                        # Also handle numpy arrays with object dtype
                        if isinstance(value, (list, dict)):
                            file_handle.attrs[key] = json.dumps(value)
                        elif isinstance(value, np.ndarray) and value.dtype == object:
                            # Convert object dtype arrays to JSON
                            file_handle.attrs[key] = json.dumps(value.tolist())
                        else:
                            try:
                                file_handle.attrs[key] = value
                            except TypeError:
                                # Fallback: try to json dump any other non-native HDF5 types
                                try:
                                    file_handle.attrs[key] = json.dumps(value)
                                except Exception:
                                    print(f"[DemonstrationRecorder] Warning: Could not save attribute {key} with value {value}")
            with h5py.File(final_h5, "r") as file_handle:
                action_dataset = file_handle["action"]
                num_timesteps = action_dataset.shape[0] if isinstance(action_dataset, h5py.Dataset) else 0
            self.metadata_info["num_timesteps"].append(num_timesteps)
            self.metadata_info["num_episodes"] += 1
            with open(os.path.join(self.base_dir, "metadata.json"), "w") as json_file:
                json.dump(self.metadata_info, json_file, indent=4)

        return self.metadata_info["num_episodes"]

    def stop(self):
        """Close all open file handles."""
        for env_id, file_handle in list(self._file_handles.items()):
            if file_handle is not None:
                with contextlib.suppress(Exception):
                    file_handle.close()

        self._file_handles.clear()
        self._created_datasets.clear()
        self._episode_tmp_paths.clear()
        self._timestamp_counter.clear()
        self._initial_sample_skipped.clear()
        self._previous_action_with_gripper.clear()
        self._previous_quaternions.clear()

    def _clear_previous_quaternions(self, env_id: int):
        for key in list(self._previous_quaternions):
            if key[0] == env_id:
                self._previous_quaternions.pop(key, None)

    def _open_episode_if_needed(self, env_id: int):
        if env_id in self._file_handles:
            return
        os.makedirs(self.base_dir, exist_ok=True)
        tmp_path = os.path.join(self.base_dir, f"episode_tmp_env{env_id}.h5")
        self._episode_tmp_paths[env_id] = tmp_path
        file_handle = h5py.File(tmp_path, "w")
        self._file_handles[env_id] = file_handle
        self._created_datasets[env_id] = set()
        self._timestamp_counter[env_id] = 0
        self._initial_sample_skipped[env_id] = False
        self._previous_action_with_gripper[env_id] = None
        self._clear_previous_quaternions(env_id)

    def _create_or_append(self, file_handle: h5py.File, name: str, chunk: np.ndarray, dtype=None, compression="lzf"):
        """Append along axis=0 to an extendable dataset. Create if missing."""
        if name not in file_handle:
            maxshape = (None,) + tuple(chunk.shape[1:])
            chunks = (max(1, min(8, chunk.shape[0])),) + tuple(chunk.shape[1:])
            dataset = file_handle.create_dataset(
                name,
                shape=(0,) + tuple(chunk.shape[1:]),
                maxshape=maxshape,
                chunks=chunks,
                dtype=dtype or chunk.dtype,
                compression=compression,
            )
        else:
            dataset = file_handle[name]
        if isinstance(dataset, h5py.Dataset):
            old_size = dataset.shape[0]
            dataset.resize(old_size + chunk.shape[0], axis=0)
            dataset[old_size : old_size + chunk.shape[0]] = chunk

    def _flush_chunk_for_env(self, env_id: int):  # noqa: C901
        """Append the current in-memory chunk to env_id's episode file."""
        try:
            file_handle = self._file_handles.get(env_id)
            if file_handle is None or len(self.ee_positions) == 0:
                return
            num_timesteps = len(self.ee_positions)

            # Determine indices to keep based on skill_done
            if self.filter_skipped_steps and len(self.skill_dones) == num_timesteps:
                # Extract skill_done for this env across all timesteps
                skill_dones_env = torch.stack([self.skill_dones[t][env_id] for t in range(num_timesteps)]).cpu().numpy()
                keep_indices = np.where(~skill_dones_env.astype(bool))[0]
            else:
                keep_indices = np.arange(num_timesteps)

            # Drop the raw first sample of each episode. The first command can be
            # a reset/startup target that has not been applied to the scene yet.
            if not self._initial_sample_skipped.get(env_id, False):
                keep_indices = keep_indices[keep_indices != 0]
                self._initial_sample_skipped[env_id] = True

            if len(keep_indices) == 0:
                return

            valid_timesteps = []
            env_indices_for_timestep = []
            for t in keep_indices:
                t = int(t)
                stored_env_ids = self.stored_env_ids_per_timestep[t]
                if env_id not in stored_env_ids:
                    continue
                valid_timesteps.append(t)
                env_indices_for_timestep.append(stored_env_ids.index(env_id))
            if len(valid_timesteps) == 0:
                return
            keep_indices = np.asarray(valid_timesteps, dtype=np.int64)

            def stack_env_slice(lst):
                timestep_slices = [
                    lst[int(t)][env_idx]
                    for t, env_idx in zip(keep_indices, env_indices_for_timestep)
                ]
                if len(timestep_slices) == 0:
                    return np.array([])
                return torch.stack(timestep_slices, dim=0).cpu().numpy()

            def continuous_quat(stream_name: str, sequence_np: np.ndarray) -> np.ndarray:
                previous = self._previous_quaternions.get((env_id, stream_name))
                fixed, last = _make_quaternions_temporally_continuous(sequence_np, previous)
                if last is not None:
                    self._previous_quaternions[(env_id, stream_name)] = last.astype(np.float32, copy=True)
                return fixed

            ee_pos_np = stack_env_slice(self.ee_positions)
            if ee_pos_np.shape[0] == 0:
                return
            saved_timestep_count = int(ee_pos_np.shape[0])
            self._create_or_append(
                file_handle, "observed_gripper", stack_env_slice(self.observed_grippers).astype(np.float32)
            )
            ee_rot_np = continuous_quat("ee_rotation", stack_env_slice(self.ee_rotations))
            ee_rot_mat_np = stack_env_slice(self.ee_rotation_matrices)
            arm_joint_pos_np = stack_env_slice(self.arm_joint_positions)
            all_joint_pos_np = stack_env_slice(self.all_joint_positions)
            all_joint_vel_np = stack_env_slice(self.all_joint_velocities)
            joint_pos_np = stack_env_slice(self.joint_positions)
            joint_vel_np = stack_env_slice(self.joint_velocities)
            joint_targets_np = stack_env_slice(self.joint_targets)
            skill_done_np = stack_env_slice(self.skill_dones)
            self._create_or_append(
                file_handle, "decision_stage", stack_env_slice(self.decision_stages).astype(np.int64)
            )
            self._create_or_append(
                file_handle, "decision_valid", stack_env_slice(self.decision_valid).astype(bool)
            )

            self._create_or_append(file_handle, "ee_position", ee_pos_np.astype(np.float32), dtype=np.float32)
            self._create_or_append(file_handle, "ee_rotation", ee_rot_np.astype(np.float32), dtype=np.float32)
            # ``arm_ee_pose``: (T, 7) measured panda_fingertip pose as
            # [x, y, z, qx, qy, qz, qw]. Quaternion ordering is xyzw
            # (NOT Isaac Sim's wxyz).
            if len(self.arm_ee_pose) == num_timesteps:
                arm_ee_pose_np = stack_env_slice(self.arm_ee_pose)
                if arm_ee_pose_np.ndim == 2 and arm_ee_pose_np.shape[1] >= 7:
                    arm_ee_pose_np = arm_ee_pose_np.copy()
                    arm_ee_pose_np[:, 3:7] = continuous_quat("arm_ee_pose", arm_ee_pose_np[:, 3:7])
                self._create_or_append(
                    file_handle, "arm_ee_pose", arm_ee_pose_np.astype(np.float32), dtype=np.float32
                )
            # Measured gripper width in meters (continuous, 0 closed to ~0.085 m
            # fully open). The reference visuomotor format stores this raw value
            # in ``arm_gripper_width`` and a normalized width in ``gripper_width``.
            arm_gripper_width_np = stack_env_slice(self.arm_gripper_widths).astype(np.float32).reshape(-1)
            self._create_or_append(
                file_handle, "arm_gripper_width", arm_gripper_width_np, dtype=np.float32
            )
            _ROBOTIQ_85_MAX_WIDTH_M = 0.085
            gripper_width_np = (arm_gripper_width_np / _ROBOTIQ_85_MAX_WIDTH_M).astype(np.float32).reshape(-1, 1)

            robot_dof_targets_np = None
            if len(self.robot_dof_targets) > 0 and len(self.robot_dof_targets) == num_timesteps:
                robot_dof_targets_np = stack_env_slice(self.robot_dof_targets)

            # Reference convention for /action[:, 16]: 1 = open, 0 = closed.
            # Absolute modes provide gripper joint targets directly; never infer
            # the action from measured gripper width.
            action_mode = (self._action_mode or "").lower()
            if not action_mode.startswith("absolute_"):
                raise RuntimeError(
                    f"Cannot derive /action gripper bit for action_mode={self._action_mode!r}; only absolute_* modes are supported."
                )

            gripper_binary = _gripper_binary_from_joint_targets(
                joint_targets_np, self._gripper_opened_np, self._gripper_closed_np
            )
            if gripper_binary is None and robot_dof_targets_np is not None:
                gripper_binary = _gripper_binary_from_joint_targets(
                    robot_dof_targets_np, self._gripper_opened_np, self._gripper_closed_np
                )
            if gripper_binary is None:
                raise RuntimeError(
                    "Cannot derive /action gripper bit from absolute gripper joint target; refusing to fall back to measured width."
                )
            ee_cmd_pos_np = None
            ee_cmd_rot_np = None
            if len(self.ee_command_positions) == num_timesteps:
                ee_cmd_pos_np = stack_env_slice(self.ee_command_positions)
                ee_cmd_rot_np = stack_env_slice(self.ee_command_rotations)
                self._create_or_append(
                    file_handle, "ee_command_position", ee_cmd_pos_np.astype(np.float32), dtype=np.float32
                )
                self._create_or_append(
                    file_handle, "ee_command_rotation", ee_cmd_rot_np.astype(np.float32), dtype=np.float32
                )
            self._create_or_append(file_handle, "arm_joint_positions", arm_joint_pos_np.astype(np.float32), dtype=np.float32)
            self._create_or_append(file_handle, "all_joint_pos", all_joint_pos_np.astype(np.float32), dtype=np.float32)
            self._create_or_append(file_handle, "all_joint_vel", all_joint_vel_np.astype(np.float32), dtype=np.float32)
            self._create_or_append(file_handle, "joint_pos", joint_pos_np.astype(np.float32), dtype=np.float32)
            self._create_or_append(file_handle, "joint_vel", joint_vel_np.astype(np.float32), dtype=np.float32)
            self._create_or_append(file_handle, "joint_targets", joint_targets_np.astype(np.float32), dtype=np.float32)
            self._create_or_append(file_handle, "skill_done", skill_done_np.astype(bool), dtype=bool)

            # Save robot_dof_targets if available
            if robot_dof_targets_np is not None:
                self._create_or_append(
                    file_handle, "robot_dof_targets", robot_dof_targets_np.astype(np.float32), dtype=np.float32
                )

            def flatten_pose_matrices(positions_np: np.ndarray, rotations_np: np.ndarray) -> np.ndarray:
                pos_tensor = torch.from_numpy(positions_np)
                rotation_matrix = torch.from_numpy(rotations_np)
                transformation_matrix = torch.eye(
                    4, dtype=rotation_matrix.dtype
                ).repeat(rotation_matrix.shape[0], 1, 1)
                transformation_matrix[:, :3, :3] = rotation_matrix
                transformation_matrix[:, :3, 3] = pos_tensor
                # Avoid einops here: importing einops' torch backend triggers torch._dynamo/_inductor imports,
                # which can crash in Isaac Sim environments due to torch/typing_extensions version mismatches.
                return transformation_matrix.reshape(transformation_matrix.shape[0], -1).cpu().numpy().astype(np.float32)

            eef_pose_np = flatten_pose_matrices(ee_pos_np, ee_rot_mat_np)
            action_with_gripper = np.concatenate([eef_pose_np, gripper_binary], axis=1).astype(np.float32)

            # Reference action format: measured panda_fingertip pose plus the
            # command-derived gripper bit.
            self._previous_action_with_gripper[env_id] = action_with_gripper[-1].copy()

            self._create_or_append(file_handle, "eef_pose", eef_pose_np, dtype=np.float32)
            self._create_or_append(file_handle, "gripper_width", gripper_width_np, dtype=np.float32)
            self._create_or_append(file_handle, "action", action_with_gripper, dtype=np.float32)
            with contextlib.suppress(Exception):
                file_handle["action"].attrs["frame"] = "panda_fingertip"
                file_handle["action"].attrs["pose_source"] = "measured_panda_fingertip_observation"
                file_handle["action"].attrs["gripper_source"] = "commanded_gripper_target_binary"
                file_handle["action"].attrs["format"] = "flattened_4x4_pose_plus_gripper_command"
                file_handle["eef_pose"].attrs["frame"] = "panda_fingertip"
                file_handle["eef_pose"].attrs["pose_source"] = "measured_panda_fingertip_observation"
                file_handle["ee_position"].attrs["frame"] = "panda_fingertip"
                file_handle["ee_rotation"].attrs["frame"] = "panda_fingertip"

            initial_timestamp = self._timestamp_counter[env_id]
            timestamps = np.round(
                np.arange(initial_timestamp, initial_timestamp + saved_timestep_count) / self.fps, 6
            ).astype(np.float32)
            self._create_or_append(file_handle, "timestamp", timestamps, dtype=np.float32)
            self._timestamp_counter[env_id] += saved_timestep_count

            def write_art_group(group_name, store_dict, quaternion: bool = False):
                if not store_dict:
                    return
                file_handle.require_group(group_name)
                object_keys = list(store_dict.keys())
                if len(object_keys) == 0:
                    return
                for object_id in object_keys:
                    sequence_np = stack_env_slice(store_dict[object_id])
                    if quaternion:
                        sequence_np = continuous_quat(f"{group_name}/{object_id}", sequence_np)
                    self._create_or_append(
                        file_handle, f"{group_name}/{object_id}", sequence_np.astype(np.float32), dtype=np.float32
                    )

            write_art_group("articulated_joint_pos", self.articulated_joint_positions)
            write_art_group("articulated_joint_vel", self.articulated_joint_velocities)
            write_art_group("articulated_root_pos", self.articulated_root_positions)
            write_art_group("articulated_root_rot", self.articulated_root_rotations, quaternion=True)

            def write_rigid_group(group_name, store_dict, quaternion: bool = False):
                if not store_dict:
                    return
                file_handle.require_group(group_name)
                object_keys = list(store_dict.keys())
                for object_id in object_keys:
                    sequence_np = stack_env_slice(store_dict[object_id])
                    if quaternion:
                        sequence_np = continuous_quat(f"{group_name}/{object_id}", sequence_np)
                    self._create_or_append(
                        file_handle, f"{group_name}/{object_id}", sequence_np.astype(np.float32), dtype=np.float32
                    )

            write_rigid_group("obj_positions", self.rigid_obj_positions)
            write_rigid_group("obj_rotations", self.rigid_obj_rotations, quaternion=True)

            if (
                not self.save_states_only
                and self.cam_keys is not None
                and len(self.cam_keys) > 0
                and self.extrinsics is not None
                and len(self.extrinsics) > 0
            ):
                num_cameras = len(self.cam_keys)

                # Calculate actual number of timesteps with camera data
                # Camera data is stored for all timesteps where store() was called and cameras were available
                actual_timesteps_with_camera = len(self.extrinsics) // num_cameras

                def reshape_timestep_camera(lst):
                    output = []
                    # Iterate through kept timesteps where camera data exists
                    for timestep in keep_indices:
                        timestep = int(timestep)
                        if timestep >= actual_timesteps_with_camera:
                            continue
                        # Find the index of env_id in the stored env_ids for this timestep
                        stored_env_ids = self.stored_env_ids_per_timestep[timestep]
                        if env_id not in stored_env_ids:
                            # Skip this timestep if env_id wasn't stored
                            continue
                        env_idx = stored_env_ids.index(env_id)
                        row = []
                        base_index = timestep * num_cameras
                        for camera_idx in range(num_cameras):
                            row.append(lst[base_index + camera_idx][env_idx])
                        output.append(torch.stack(row, dim=0))
                    if len(output) == 0:
                        return torch.empty((0, num_cameras, *lst[0].shape[1:]), device=lst[0].device)
                    return torch.stack(output, dim=0)

                extrinsics_timestep_camera = reshape_timestep_camera(self.extrinsics).cpu().numpy().astype(np.float32)
                intrinsics_timestep_camera = (
                    reshape_timestep_camera(self.intrinsics).cpu().numpy().astype(np.float32)
                    if len(self.intrinsics) == len(self.extrinsics)
                    else None
                )

                self._create_or_append(file_handle, "extrinsics", extrinsics_timestep_camera, dtype=np.float32)
                if intrinsics_timestep_camera is not None:
                    self._create_or_append(file_handle, "intrinsics", intrinsics_timestep_camera, dtype=np.float32)

                if self.write_rgb:
                    if len(self.rgbs) != len(self.extrinsics):
                        raise ValueError(
                            "RGB persistence requires one RGB frame per camera extrinsic: "
                            f"rgb={len(self.rgbs)} extrinsics={len(self.extrinsics)}"
                        )
                    rgb_timestep_camera = reshape_timestep_camera(self.rgbs).cpu().numpy()
                    if rgb_timestep_camera.shape[-1] == 4:
                        rgb_timestep_camera = rgb_timestep_camera[..., :3]
                    self._create_or_append(
                        file_handle,
                        "rgb",
                        rgb_timestep_camera.astype(np.uint8, copy=False),
                        dtype=np.uint8,
                    )

                if self.write_point_cloud and self.point_cloud is not None:
                    # Point cloud shape is (T, B, P, F); extract rows for this env_id across timesteps.
                    point_cloud_slices = []
                    point_cloud_count_values = []
                    point_cloud_flat_slices = [] if self.write_point_cloud_flat else None
                    point_cloud_env_ids = self._point_cloud_env_ids
                    for timestep in keep_indices:
                        timestep = int(timestep)
                        if point_cloud_env_ids is None:
                            stored_env_ids = self.stored_env_ids_per_timestep[timestep]
                            if env_id not in stored_env_ids:
                                continue
                            env_idx = stored_env_ids.index(env_id)
                        else:
                            if env_id not in point_cloud_env_ids:
                                continue
                            env_idx = point_cloud_env_ids.index(env_id)
                        point_cloud_t = self.point_cloud[timestep, env_idx]
                        point_cloud_slices.append(point_cloud_t)
                        if self.point_cloud_counts is not None:
                            point_count = int(self.point_cloud_counts[timestep, env_idx].item())
                        else:
                            point_count = int(point_cloud_t.shape[0])
                        point_cloud_count_values.append(point_count)
                        if self.write_point_cloud_flat and point_count > 0 and point_cloud_flat_slices is not None:
                            point_cloud_flat_slices.append(point_cloud_t[:point_count, :3])
                    if len(point_cloud_count_values) > 0:
                        point_cloud_count_np = np.asarray(point_cloud_count_values, dtype=np.int32)
                        self._create_or_append(
                            file_handle, "point_cloud_count", point_cloud_count_np, dtype=np.int32
                        )
                        if self.write_point_cloud_flat and point_cloud_flat_slices is not None:
                            if len(point_cloud_flat_slices) > 0:
                                point_cloud_flat_np = (
                                    torch.cat(point_cloud_flat_slices, dim=0).cpu().numpy().astype(np.float32)
                                )
                            else:
                                point_cloud_flat_np = np.empty((0, 3), dtype=np.float32)
                            self._create_or_append(file_handle, "point_cloud_flat", point_cloud_flat_np, dtype=np.float32)
                    # ``POINT_CLOUD_NUM_POINTS=-1`` means keep the full valid variable-length cloud.
                    # In that mode, the reference-compatible flat/count datasets above are the intended output;
                    # writing the padded dense tensor can create multi-GB HDF5 files and exhaust disk space.
                    if len(point_cloud_slices) > 0 and self.point_cloud_num_points > 0:
                        point_cloud_np = torch.stack(point_cloud_slices, dim=0).cpu().numpy().astype(np.float32)
                        self._create_or_append(file_handle, "point_cloud", point_cloud_np, dtype=np.float32)
        except Exception as e:
            print(f"[flush][env={env_id}] ERROR: {repr(e)}")
            raise

    def _finalize_episode(self, env_id: int, success: bool):
        kept = False
        file_handle = self._file_handles.get(env_id)
        if file_handle is None:
            return kept, None

        if "env_origins" not in file_handle and self.env_origins is not None:
            file_handle.create_dataset(
                "env_origins", data=self.env_origins[env_id].detach().cpu().numpy().astype(np.float32)
            )
        if "task_description" not in file_handle and self.task_description is not None:
            string_dtype = h5py.special_dtype(vlen=str)
            file_handle.create_dataset(
                "task_description", data=np.array([self.task_description], dtype=object), dtype=string_dtype
            )

        file_handle.close()

        if success:
            kept = True
        else:
            tmp_root = os.path.join(self.base_dir, f"episode_tmp_env{env_id}")
            tmp_h5 = self._episode_tmp_paths.get(env_id)
            try:
                if tmp_h5 and os.path.exists(tmp_h5):
                    os.remove(tmp_h5)
                if os.path.isdir(tmp_root):
                    shutil.rmtree(tmp_root, ignore_errors=True)
            except Exception:
                pass
            # Clean up the path entry for failed episodes
            self._episode_tmp_paths.pop(env_id, None)

        for dictionary in (
            self._file_handles,
            self._created_datasets,
            self._timestamp_counter,
            self._initial_sample_skipped,
            self._previous_action_with_gripper,
        ):
            dictionary.pop(env_id, None)
        self._clear_previous_quaternions(env_id)

        return kept, self._episode_tmp_paths.get(env_id)
