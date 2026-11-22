import numpy as np
import torch
from torch import Tensor
from isaaclab.utils.math import (
    quat_from_euler_xyz,
    quat_from_angle_axis,
    quat_from_matrix,
    matrix_from_euler,
    matrix_from_quat,
    euler_xyz_from_quat,
    make_pose,
    pose_inv,
    unmake_pose,
    pose_in_A_to_pose_in_B,
)

class Transform:
    __slots__ = ("data", "_translation", "_rotation", "_quaternion", "_inverse")

    def __init__(self, matrix: torch.Tensor):
        # All transforms stored in float32 for consistency/performance
        self.data = matrix
        self._translation = None
        self._rotation = None
        self._quaternion = None
        self._inverse = None

    def _invalidate_cache(self) -> None:
        """Invalidate cached decomposition/inverse derived from self.data.

        IMPORTANT: If you mutate/replace `self.data`, you must call this to keep
        `translation/rotation/quaternion/inverse` consistent.
        """
        self._translation = None
        self._rotation = None
        self._quaternion = None
        self._inverse = None

    def set_matrix(self, matrix: torch.Tensor) -> None:
        """Replace internal transform matrix and invalidate cached values."""
        self.data = matrix
        self._invalidate_cache()

    def _ensure_components(self):
        if self._translation is None or self._rotation is None:
            self._translation, self._rotation = unmake_pose(self.data)

    @property
    def translation(self):
        """Return the translation vector."""
        self._ensure_components()
        return self._translation

    @property
    def matrix(self):
        """Return the 4x4 transformation matrix."""
        return self.data

    @property
    def rotation(self):
        """Return the rotation matrix."""
        self._ensure_components()
        return self._rotation

    @property
    def quaternion(self):
        """Return the quaternion representation (w, x, y, z)."""
        if self._quaternion is None:
            self._quaternion = quat_from_matrix(self.rotation)
        return self._quaternion

    @property
    def inverse(self):
        """Return the inverse transformation."""
        if self._inverse is None:
            self._inverse = Transform(matrix=pose_inv(self.data))
        return self._inverse

    def __matmul__(self, other):
        """Compose with another Transform: self * other."""
        return Transform(self.matrix @ other.matrix)


class Twist:
    def __init__(self, vec):
        self.vec = np.asarray(vec, dtype=float).reshape(6)  # [ωx,ωy,ωz,vx,vy,vz]

    # Accessors -------------------------------------------------------------
    @property
    def w(self):
        return self.vec[:3]  # angular

    @property
    def v(self):
        return self.vec[3:]  # linear

    # Hat operator: 6‑vector → 4×4 matrix ----------------------------------
    def hat(self):
        ω_hat = Twist.skew(self.w)
        Xi = np.zeros((4, 4))
        Xi[:3, :3] = ω_hat
        Xi[:3, 3] = self.v
        return Xi

    # Frame change: ξ' = Ad_T ξ --------------------------------------------
    def transform(self, T: Transform):
        # This functionality isn't used in the current codebase; keep it explicit
        # to avoid relying on non-existent Transform methods.
        raise NotImplementedError("Twist.transform requires Transform.adjoint().")

    # Algebraic ops ---------------------------------------------------------
    def __add__(self, other):
        return Twist(self.vec + other.vec)

    def __sub__(self, other):
        return Twist(self.vec - other.vec)

    def __repr__(self):
        return f"Twist({self.vec})"

    # Static utility: skew‑symmetric matrix of a 3‑vector -------------------
    @staticmethod
    def skew(a):
        ax, ay, az = a
        return np.array([[0, -az, ay], [az, 0, -ax], [-ay, ax, 0]])


def euler_xyz_from_matrix(m):
    return euler_xyz_from_quat(quat_from_matrix(m))


def quat_to_matrix(q: Tensor) -> Tensor:
    return matrix_from_quat(q)


# @torch.compile
def quat_to_rotation_6d(q: Tensor) -> Tensor:
    """Convert wxyz quaternion(s) to the canonical 6D rotation representation.

    The layout is explicit column-major-in-vector form:
        [R00, R01, R10, R11, R20, R21]
    i.e. the first two columns of the rotation matrix, written as
    [col0.x, col1.x, col0.y, col1.y, col0.z, col1.z].

    We spell this out explicitly instead of relying on reshape semantics so the
    quat <-> rot6d convention stays unambiguous across the codebase.
    """
    # Compute the 6 needed elements directly from quaternion components,
    # avoiding the full 3x3 matrix_from_quat allocation.
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    x2, y2, z2 = x * x, y * y, z * z
    return torch.stack(
        [
            1.0 - 2.0 * (y2 + z2),  # R00
            2.0 * (x * y - w * z),  # R01
            2.0 * (x * y + w * z),  # R10
            1.0 - 2.0 * (x2 + z2),  # R11
            2.0 * (x * z - w * y),  # R20
            2.0 * (y * z + w * x),  # R21
        ],
        dim=-1,
    )


# @torch.compile
def rotation_6d_to_quat(r6: Tensor) -> Tensor:
    """Convert 6D rotation representation to wxyz quaternion(s).

    Expected input layout matches `quat_to_rotation_6d` exactly:
        [R00, R01, R10, R11, R20, R21]
    i.e. the first two columns of a 3x3 rotation matrix written in explicit
    vector form.

    This function is intentionally defensive: policy outputs can contain tiny or
    nearly parallel 6D columns. We therefore repair the basis with a stabilized
    Gram-Schmidt step and deterministic fallback axes, so the result is always a
    valid right-handed rotation before conversion to quaternion.
    """
    eps = 1e-6
    a1 = r6[..., [0, 2, 4]]  # col0 = [R00, R10, R20]
    a2 = r6[..., [1, 3, 5]]  # col1 = [R01, R11, R21]

    def _safe_normalize(v: Tensor, fallback: Tensor) -> Tensor:
        norm = v.norm(dim=-1, keepdim=True)
        # torch.where broadcasts fallback implicitly; no broadcast_to needed.
        return torch.where(norm > eps, v / norm.clamp_min(eps), fallback)

    # Build x_axis without in-place mutation so torch.compile can trace this.
    x_axis = torch.nn.functional.pad(torch.zeros_like(a1)[..., 1:], (1, 0), value=1.0)
    b1 = _safe_normalize(a1, x_axis)

    u2 = a2 - (a2 * b1).sum(dim=-1, keepdim=True) * b1

    # Compute cross(e_i, b1) analytically for each standard basis vector e_i,
    # then gather the one for the least-aligned axis. This avoids one_hot +
    # torch.cross (two extra kernel launches).
    b1x, b1y, b1z = b1[..., 0], b1[..., 1], b1[..., 2]
    _z = torch.zeros_like(b1x)
    cross_e = torch.stack(
        [
            torch.stack([_z,   -b1z,  b1y], dim=-1),  # cross(e0, b1)
            torch.stack([b1z,   _z,  -b1x], dim=-1),  # cross(e1, b1)
            torch.stack([-b1y,  b1x,  _z ], dim=-1),  # cross(e2, b1)
        ],
        dim=-2,
    )  # (..., 3, 3)
    least_aligned_axis = b1.abs().argmin(dim=-1)
    idx = least_aligned_axis[..., None, None].expand(*least_aligned_axis.shape, 1, 3)
    b2_fallback_raw = cross_e.gather(-2, idx).squeeze(-2)
    b2_fallback = _safe_normalize(b2_fallback_raw, torch.roll(x_axis, shifts=1, dims=-1))
    b2 = _safe_normalize(u2, b2_fallback)

    b3 = _safe_normalize(
        torch.cross(b1, b2, dim=-1),
        torch.cross(b1, b2_fallback, dim=-1),
    )
    b2 = _safe_normalize(torch.cross(b3, b1, dim=-1), b2_fallback)

    R = torch.stack([b1, b2, b3], dim=-1)  # columns are b1, b2, b3
    return quat_from_matrix(R)


def quat_multiply_pos(q: Tensor, v: Tensor) -> Tensor:
    """Rotate vector(s) v by quaternion(s) q (wxyz) using a matrix-free formula.

    This is significantly faster than building a rotation matrix for hot paths
    (predicates/residuals) and matches the standard quaternion rotation:
        v' = q ⊗ (0,v) ⊗ q_conj
    for unit quaternions.
    """
    q = q.to(dtype=torch.float32)
    v = v.to(dtype=torch.float32)
    # Ensure (...,3) for v and (...,4) for q
    qw = q[..., :1]
    qv = q[..., 1:4]
    # t = 2 * cross(qv, v)
    t = 2.0 * torch.cross(qv, v, dim=-1)
    # v' = v + qw * t + cross(qv, t)
    return v + qw * t + torch.cross(qv, t, dim=-1)

def z_axis_from_quat(q):
    # PERFORMANCE: avoid building a full rotation matrix.
    # For wxyz quaternion q, the z-axis is column 2 of R(q):
    #   [2(xz + wy), 2(yz - wx), 1 - 2(x^2 + y^2)]
    if q.shape[-1] == 1 and q.shape[-2] == 3:
        q = q.squeeze(-1)
    if q.ndim == 1:
        q = q.unsqueeze(0)
    w, x, y, z = q.unbind(-1)
    two = 2.0
    return torch.stack(
        (
            two * (x * z + w * y),
            two * (y * z - w * x),
            1.0 - two * (x * x + y * y),
        ),
        dim=-1,
    )


def x_axis_from_quat(q):
    # PERFORMANCE: avoid building a full rotation matrix.
    # For wxyz quaternion q, the x-axis is column 0 of R(q):
    #   [1 - 2(y^2 + z^2), 2(xy + wz), 2(xz - wy)]
    if q.shape[-1] == 1 and q.shape[-2] == 3:
        q = q.squeeze(-1)
    if q.ndim == 1:
        q = q.unsqueeze(0)
    w, x, y, z = q.unbind(-1)
    two = 2.0
    return torch.stack(
        (
            1.0 - two * (y * y + z * z),
            two * (x * y + w * z),
            two * (x * z - w * y),
        ),
        dim=-1,
    )


def y_axis_from_quat(q):
    # PERFORMANCE: avoid building a full rotation matrix.
    # For wxyz quaternion q, the y-axis is column 1 of R(q):
    #   [2(xy - wz), 1 - 2(x^2 + z^2), 2(yz + wx)]
    if q.shape[-1] == 1 and q.shape[-2] == 3:
        q = q.squeeze(-1)
    if q.ndim == 1:
        q = q.unsqueeze(0)
    w, x, y, z = q.unbind(-1)
    two = 2.0
    return torch.stack(
        (
            two * (x * y - w * z),
            1.0 - two * (x * x + z * z),
            two * (y * z + w * x),
        ),
        dim=-1,
    )


def codirectional_angle_between_vectors(a, b) -> torch.Tensor:
    """
    Angle between 3D vectors a and b (radians) that are meant to be pointing in the same direction. Raises if a or b is ~zero.
    """
    return angle_between_vectors(a, b)


def angle_between_vectors(a, b) -> torch.Tensor:
    """
    Angle between 3D vectors a and b (radians). Raises if a or b is ~zero.
    """
    dot = (a * b).sum(dim=-1)
    na = a.norm(dim=-1)
    nb = b.norm(dim=-1)
    cosang = (dot / (na * nb).clamp_min(1e-12)).clamp(-1.0, 1.0)
    return torch.acos(cosang)


def antiparallel_angle_between_vectors(a, b) -> torch.Tensor:
    """
    Angle between 3D vectors a and b (radians) that are meant to be pointing in the opposite direction. Raises if a or b is ~zero.
    """
    return angle_between_vectors(a, -b)


def bipolar_angle_between_vector_and_line(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """
    Smallest angle (radians) from a to the axis defined by ±b. Range [0, π/2].
    a, b: (...,3) or (...,3,1)
    """
    if a.shape[-1] == 1 and len(a.shape) > 2:
        a = a.squeeze(-1)
    if b.shape[-1] == 1 and len(b.shape) > 2:
        b = b.squeeze(-1)
    eps: float = 1e-12
    dot = (a * b).sum(dim=-1)  # (...,)
    cross_norm = torch.linalg.norm(torch.cross(a, b, dim=-1), dim=-1)  # (...,)
    angle = torch.atan2(cross_norm, torch.abs(dot) + eps)  # (...,)
    return angle


def orthogonality_angle_between_vector_and_line(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Smallest angular deviation from 90° (π/2) between a and the axis defined by ±b.
    0 means perfectly perpendicular; π/2 means (anti)parallel. Range [0, π/2].
    a, b: (...,3) or (...,3,1)
    """
    if a.shape[-1] == 1 and a.ndim > 2:
        a = a.squeeze(-1)
    if b.shape[-1] == 1 and b.ndim > 2:
        b = b.squeeze(-1)

    eps = 1e-12
    dot = (a * b).sum(dim=-1)
    a_n = torch.linalg.norm(a, dim=-1)
    b_n = torch.linalg.norm(b, dim=-1)
    cos = torch.abs(dot) / (a_n * b_n + eps)          # |cos θ|
    cos = torch.clamp(cos, 0.0, 1.0)                  # numerical safety
    return torch.arcsin(cos)         # (batch,)


def inverse_quat(q: Tensor) -> Tensor:
    """
    Inverse of a unit (normalized) quaternion.
    q: (batch, 4) in (w, x, y, z)
    returns: (batch, 4) in (w, x, y, z)
    """
    if q.ndim != 2 or q.shape[-1] != 4:
        raise ValueError(f"Expected q with shape (batch, 4), got {tuple(q.shape)}")

    q_inv = q.clone()
    q_inv[:, 1:] = -q_inv[:, 1:]  # conjugate
    return q_inv
