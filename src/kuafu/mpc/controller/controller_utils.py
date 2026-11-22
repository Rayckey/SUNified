import numpy as np
import casadi as ca
import pinocchio as pin
import torch

# ---------------------------------------------------------------------------
# Basic helpers --------------------------------------------------------------
# ---------------------------------------------------------------------------


def _skew(v):
    """Skew-symmetric matrix for a 3-vector (v is a CasADi MX or DM)."""
    return ca.vcat(
        [
            ca.hcat([0, -v[2], v[1]]),
            ca.hcat([v[2], 0, -v[0]]),
            ca.hcat([-v[1], v[0], 0]),
        ]
    )


def _rodrigues(axis, theta):
    """
    Rotation matrix e^(axis×theta) with CasADi algebra.
    `axis` is a *constant* 3-vector (DM); `theta` is symbolic (MX/SX scalar).
    """
    axis = axis / ca.norm_2(axis)  # make sure it is unit length
    K = _skew(axis)
    return ca.MX.eye(3) + ca.sin(theta) * K + (1 - ca.cos(theta)) * (K @ K)


def _se3_to_casadi(T_se3):
    """Convert a pin.SE3 to a 4×4 CasADi constant matrix."""
    R, p = T_se3.rotation, T_se3.translation
    T = ca.DM.zeros(4, 4)
    T[:3, :3] = ca.DM(R)
    T[:3, 3] = ca.DM(p)
    T[3, 3] = 1.0
    return T


def omega(w):
    """Ω(w) so that  q̇ = 0.5·Ω(w)·q   (w = angular velocity)."""
    wx, wy, wz = w[0], w[1], w[2]
    return ca.vertcat(
        ca.hcat([0, -wx, -wy, -wz]),
        ca.hcat([wx, 0, -wz, wy]),
        ca.hcat([wy, wz, 0, -wx]),
        ca.hcat([wz, -wy, wx, 0]),
    )

def quat_multiply_matrix(q):
    """
    Left-multiplication matrix L(q) for Hamilton product with wxyz quaternion.
    For any quaternion p (4x1), L(q) @ p == q ⊗ p.
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    return ca.vertcat(
        ca.hcat([ w, -x, -y, -z]),
        ca.hcat([ x,  w, -z,  y]),
        ca.hcat([ y,  z,  w, -x]),
        ca.hcat([ z, -y,  x,  w])
    )

def matrix_to_quat(R: np.ndarray):
    if isinstance(R, torch.Tensor):
        R = R.cpu().numpy( )
    _q = pin.Quaternion(R)
    return np.array([_q.w, _q.x, _q.y, _q.z])

def quat_to_matrix(q):
    """
    <<Convert quaternion (w, x, y, z) to rotation matrix.
    Inputs
        q : Quaternion in (w, x, y, z) format.>>
    """
    w, x, y, z = q[0], q[1], q[2], q[3]

    # Build row-wise to preserve the input symbolic type (MX/SX/DM) and avoid
    # DM-set errors when q is symbolic.
    row0 = ca.hcat([1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)])
    row1 = ca.hcat([2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)])
    row2 = ca.hcat([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)])
    return ca.vertcat(row0, row1, row2)

# <utility>
def quat_mul(q1, q2):
    """<< Quaternion product q = q1 ⊗ q2, both in wxyz. >>"""
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    return ca.vcat(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


# <end-of-utility>

# <utility>
def quat_from_euler_xyz(xyz):
    """
    << Build q from local roll(X), pitch(Y), yaw(Z) increments (radian).>>
    """
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    hx, hy, hz = 0.5 * x, 0.5 * y, 0.5 * z
    cx, sx = ca.cos(hx), ca.sin(hx)
    cy, sy = ca.cos(hy), ca.sin(hy)
    cz, sz = ca.cos(hz), ca.sin(hz)

    # Intrinsic X->Y->Z (equivalently extrinsic Z->Y->X)
    w = cx * cy * cz + sx * sy * sz
    qx = sx * cy * cz - cx * sy * sz
    qy = cx * sy * cz + sx * cy * sz
    qz = cx * cy * sz - sx * sy * cz

    # normalize for robustness
    n = ca.sqrt(w * w + qx * qx + qy * qy + qz * qz)
    return ca.vcat([w / n, qx / n, qy / n, qz / n])


# <end-of-utility>

# <utility>
def z_axis_from_quat(q):
    """
    <<extract z-axis vector from quaternion (w, x, y, z)
    Inputs
        q : Quaternion in (w, x, y, z) format.>>

    """
    w, x, y, z = q[0], q[1], q[2], q[3]

    vz0 = 2 * (x * z + y * w)
    vz1 = 2 * (y * z - x * w)
    vz2 = 1 - 2 * (x * x + y * y)
    return ca.vertcat(vz0, vz1, vz2)


# <end-of-utility>


# <utility>
def x_axis_from_quat(q):
    """
    <<extract x-axis vector from quaternion (w, x, y, z)
    Inputs
        q : Quaternion in (w, x, y, z) format.>>
    """
    w, x, y, z = q[0], q[1], q[2], q[3]

    vx0 = 1 - 2 * (y * y + z * z)
    vx1 = 2 * (x * y + z * w)
    vx2 = 2 * (x * z - y * w)
    return ca.vertcat(vx0, vx1, vx2)


# <end-of-utility>


# <utility>
def y_axis_from_quat(q):
    """
    <<extract y-axis vector from quaternion (w, x, y, z)
    Inputs
        q : Quaternion in (w, x, y, z) format.>>
    """
    w, x, y, z = q[0], q[1], q[2], q[3]

    vy0 = 2 * (x * y - z * w)
    vy1 = 1 - 2 * (x * x + z * z)
    vy2 = 2 * (y * z + x * w)
    return ca.vertcat(vy0, vy1, vy2)


# <end-of-utility>


# <utility>
def inverse_quat(q):
    """
    << Invert a quaternion.
    :param q: Quaternion as a CasADi vector [w, x, y, z].
    :return: Inverted quaternion as a CasADi vector [w, -x, -y, -z]. >>
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    return ca.vertcat(w, -x, -y, -z)


# <end-of-utility>

# ---------------------------------------------------------------------------
# Main builder ---------------------------------------------------------------
# ---------------------------------------------------------------------------


def build_symbolic_fk(
    model, ee_frame_name: str = "panda_fingertip"
) -> ca.Function:
    """
    Return a CasADi FK(q) → 4×4 homogeneous pose of `ee_frame_name`.

    Works on aligned + un-aligned revolute / prismatic joints without
    importing pinocchio.casadi.
    """

    # --- 1  collect joints on the root→EE path -----------------------
    fid = model.getFrameId(ee_frame_name)
    if fid == len(model.frames):
        raise ValueError(f"Frame '{ee_frame_name}' not in model.")
    chain = []
    j = model.frames[fid].parent
    while j > 0:
        chain.append(j)
        j = model.parents[j]
    chain.reverse()

    # --- 2  symbolic joint vector ------------------------------------
    nq = model.nq
    q = ca.MX.sym("q", nq)

    # --- 3  forward-propagate transforms -----------------------------
    T = ca.MX.eye(4)
    idx = 0  # cursor in q

    for jid in chain:
        # constant placement parent→joint
        T = T @ _se3_to_casadi(model.jointPlacements[jid])

        # joint model & single coordinate
        jm = model.joints[jid]
        if jm.nq != 1:
            raise NotImplementedError(f"Joint {jid} has {jm.nq} DoF.")
        theta = q[idx]
        idx += 1
        jname = jm.shortname()

        # -------- axis retrieval (new code) --------------------------
        if jname.startswith(("JointModelR", "JointModelP")):
            if hasattr(jm, "axis"):
                axis = ca.DM(jm.axis)
            elif hasattr(jm, "getMotionAxis"):
                axis = ca.DM(jm.getMotionAxis())
            else:  # belt-and-braces
                if jname.endswith("RX"):
                    axis = ca.DM([1, 0, 0])
                elif jname.endswith("RY"):
                    axis = ca.DM([0, 1, 0])
                elif jname.endswith("RZ"):
                    axis = ca.DM([0, 0, 1])
                elif jname.endswith("PX"):
                    axis = ca.DM([1, 0, 0])
                elif jname.endswith("PY"):
                    axis = ca.DM([0, 1, 0])
                elif jname.endswith("PZ"):
                    axis = ca.DM([0, 0, 1])
                else:
                    raise NotImplementedError(f"No axis for '{jname}'.")
            # -------------------------------------------------------------

            if jname.startswith("JointModelR"):  # revolute
                Rvar = _rodrigues(axis, theta)
                Tvar = ca.MX.eye(4)
                Tvar[:3, :3] = Rvar
            else:  # prismatic
                Tvar = ca.MX.eye(4)
                Tvar[:3, 3] = axis * theta

            T = T @ Tvar
        else:
            raise NotImplementedError(f"Unhandled joint type '{jname}'.")

    # --- 4  final joint→frame placement ------------------------------
    T = T @ _se3_to_casadi(model.frames[fid].placement)

    # --- 5  wrap up --------------------------------------------------
    return ca.Function(f"fk_{ee_frame_name}", [q], [T], ["q"], ["T"])


# ---------------------------------------------------------------------------
# Example usage --------------------------------------------------------------
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    urdf_path = "resources/robots/franka_description/robots/panda_arm_hand_fixed.urdf"
    model = pin.buildModelFromUrdf(urdf_path)
    fk = build_symbolic_fk(model, "panda_fingertip")  # returns CasADi Function

    q_zero = np.random.rand(7, 1)  # home config
    T_sym = fk(q_zero)  # CasADi → DM(4,4)
    data = model.createData()
    pin.framesForwardKinematics(model, data, q_zero)
    T_num = pin.updateFramePlacement(model, data, model.getFrameId("panda_fingertip"))

    print("symbolic FK\n", np.array(T_sym))
    print("pinocchio FK\n", T_num)
    # They should match up to numerical precision.
