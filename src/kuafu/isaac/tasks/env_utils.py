import torch
from isaaclab.utils import math as math_utils

def random_pose_radial(
    z_height: float,
    dist_min: float,
    dist_max: float,
    angle_min_rad: float,
    angle_max_rad: float,
    n: int,
    device: torch.device,
    origins: torch.Tensor,
    rotate_randomly: bool = True,
):
    dist = dist_min + torch.rand(n, device=device) * (dist_max - dist_min)
    angle_rad = angle_min_rad + torch.rand(n, device=device) * (
        angle_max_rad - angle_min_rad
    )
    x = dist * torch.cos(angle_rad)
    y = dist * torch.sin(angle_rad)
    pos = origins + torch.stack(
        [x, y, torch.full((n,), z_height, device=device)], dim=1
    )

    if rotate_randomly:
        rot_rad = torch.rand(n, device=device) * 2 * torch.pi
    else:
        rot_rad = angle_rad
    quat = math_utils.quat_from_euler_xyz(
        torch.zeros(n, device=device),
        torch.zeros(n, device=device),
        rot_rad,
    )
    return pos, quat
