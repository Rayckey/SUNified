import casadi as ca
import numpy as np
import pinocchio as pin
import torch
import copy
from isaaclab.utils.math import quat_box_minus
from kuafu.mpc.simulation.struct import (
    StateEntity,
    ControlEntity,
    FrameEntity,
    VisibleEntity,
    SensedEntity,
)

from kuafu.mpc.simulation.spatial_utils import Transform, quat_to_rotation_6d
from kuafu.mpc.controller.controller_utils import (
    build_symbolic_fk,
    matrix_to_quat,
    quat_to_matrix,
    quat_multiply_matrix,
)

from typing import Dict, Any


class Controller:
    """
    Base class for controllers. Provides methods to get state and control descriptions.
    """

    def __init__(self):
        """
        Initializes the controller.
        """
        self.entity_dict = {}
        self.name = ""
        pass

    def getStateDescription(self):
        """
        Returns a list of StateEntity objects describing the state of the controller.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def getControlDescription(self):
        """
        Returns a list of ControlEntity objects describing the control inputs of the controller.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def save_odometry_checkpoint(self):
        raise NotImplementedError("Subclasses should implement this method.")


class DifferentialDriveController(Controller):
    """
    State x_k := [px, py, pz, qw, qx, qy, qz, wl, wr]  (len=9)
       - wl, wr are wheel angular velocities (rad/s) stored in the state.
    Control u_k := [wl_cmd, wr_cmd]                    (len=2)
       - controls are wheel angular velocities commanded at step k.
       - We impose rate limits on u_k to respect max wheel acceleration.

    Discrete dynamics (dt):
       1) Integrate base pose using *state* wheel rates wl_k, wr_k.
       2) Enforce wheel state update with zero-order hold:
              wl_{k+1} = u_k[0],   wr_{k+1} = u_k[1]
       3) Apply rate constraints on u_k - u_{k-1}.
    """

    def __init__(
        self,
        wheel_radius: float,
        base_length: float,
        horizon_length: int,
        time_step: float,
        max_wheel_velocity: float,
        max_wheel_acceleration: float,
        device="cpu",
    ):
        self.radius = float(wheel_radius)
        self.base = float(base_length)
        self.N = int(horizon_length)
        self.dt = float(time_step)
        self.max_velocity = float(max_wheel_velocity)
        self.max_acceleration = float(max_wheel_acceleration)
        self.device = device

        # Custom cost function (call update_cost_function to set)
        self._cost_function = None

        # CasADi problem sizes
        self.num_states = 9  # [x,y,z,qw,qx,qy,qz, wl, wr]
        self.num_controls = 2  # [wl_cmd, wr_cmd]

        # Symbolic handles (used mainly for user reference; solver uses packed variables)
        self.state = ca.SX.sym("state", self.num_states)
        self.control = ca.SX.sym("control", self.num_controls)

        # NLP solver + caches
        self.solver = None
        self._pack_shapes: Dict[str, Any] = {}
        self._lbx = None
        self._ubx = None
        self._lbg = None
        self._ubg = None

        # Debug trajectory store
        self.last_traj = None

    # -----------------------------
    # Descriptions for your UI
    # -----------------------------
    def getStateDescription(self):
        pos_x = StateEntity("position x", 0, "base frame position x", True)
        pos_y = StateEntity("position y", 1, "base frame position y", True)
        pos_z = StateEntity("position z", 2, "base frame position z", True)
        rot_w = StateEntity("quaternion w", 3, "base frame quaternion w", True)
        rot_x = StateEntity("quaternion x", 4, "base frame quaternion x", True)
        rot_y = StateEntity("quaternion y", 5, "base frame quaternion y", True)
        rot_z = StateEntity("quaternion z", 6, "base frame quaternion z", True)
        wl = StateEntity("left wheel ω", 7, "left wheel angular velocity (rad/s)", True)
        wr = StateEntity(
            "right wheel ω", 8, "right wheel angular velocity (rad/s)", True
        )
        return [pos_x, pos_y, pos_z, rot_w, rot_x, rot_y, rot_z, wl, wr]

    def getControlDescription(self):
        # Keeping your labels; units corrected in the description
        T_l = ControlEntity(
            "W_l", 0, "Commanded left wheel angular velocity (rad/s)", True
        )
        T_r = ControlEntity(
            "W_r", 1, "Commanded right wheel angular velocity (rad/s)", True
        )
        return [T_l, T_r]

    # -----------------------------
    # State & control extraction
    # -----------------------------
    def getState(
        self,
        xyz: np.ndarray,
        quaternion: np.ndarray,
        vel_xyz: np.ndarray,
        vel_euler: np.ndarray,
    ):
        """
        Build the 9D MPC state from sim:
          xyz:        [3] position
          quaternion: [4] wxyz (unit)
          vel_xyz:    [3] linear velocity in base/world consistent with your data (m/s)
          vel_euler:  [3] angular velocity; we use yaw rate = vel_euler[2] (rad/s)

        Wheel angular velocities are computed from (v, yaw_rate):
            v  = sqrt(vx^2 + vy^2)
            wl = (v - yaw_rate * L/2) / r
            wr = (v + yaw_rate * L/2) / r
        """
        xyz = np.asarray(xyz, dtype=float).reshape(3)
        quat = np.asarray(quaternion, dtype=float).reshape(4)
        vel_xyz = np.asarray(vel_xyz, dtype=float).reshape(3)
        vel_euler = np.asarray(vel_euler, dtype=float).reshape(3)

        v = float(np.sqrt(vel_xyz[0] ** 2 + vel_xyz[1] ** 2))
        yaw_rate = float(vel_euler[2])
        wl = (v - yaw_rate * self.base * 0.5) / self.radius
        wr = (v + yaw_rate * self.base * 0.5) / self.radius

        return np.concatenate([xyz, quat, np.array([wl, wr], dtype=float)])

    def getControl(self, vel_xyz: np.ndarray, vel_euler: np.ndarray):
        """
        Convert current motion to the control vector [wl_cmd, wr_cmd].
        You can also plug your policy output directly here if you prefer.
        """
        vel_xyz = np.asarray(vel_xyz, dtype=float).reshape(3)
        vel_euler = np.asarray(vel_euler, dtype=float).reshape(3)
        v = float(np.sqrt(vel_xyz[0] ** 2 + vel_xyz[1] ** 2))
        yaw_rate = float(vel_euler[2])
        left = (v - yaw_rate * self.base * 0.5) / self.radius
        right = (v + yaw_rate * self.base * 0.5) / self.radius
        return np.array([left, right], dtype=float)

    # -----------------------------
    # Cost plumbing
    # -----------------------------
    def update_cost_function(self, name_scope, cost_only=False) -> None:
        """
        Plug in a new CasADi scalar-valued objective with signature:

            residual_function(X, U) -> SX/MX scalar

        shapes:
            X : (9, N+1)
            U : (2, N)
        """
        self._cost_function = name_scope["residual_function"]
        if cost_only:
            return
        self.compile_solver()

    @property
    def cost_function(self):
        if self._cost_function is None:
            raise RuntimeError("Cost function is not set.")
        return self._cost_function

    # -----------------------------
    # Solver build
    # -----------------------------
    def compile_solver(self):
        if self._cost_function is None:
            raise RuntimeError(
                "Set the cost function via update_cost_function() before compiling the solver."
            )

        nx = self.num_states
        nu = self.num_controls
        N = self.N
        dt = self.dt
        r = self.radius
        L = self.base

        # Decision variables (stacked over horizon)
        X = ca.SX.sym("X", nx, N + 1)  # [x,y,z,qw,qx,qy,qz, wl, wr]
        U = ca.SX.sym("U", nu, N)  # [wl_cmd, wr_cmd]

        # Parameters: initial state and previous control
        X0 = ca.SX.sym("X0", nx)
        Uprev = ca.SX.sym("U_prev", nu)

        g = []
        lbg = []
        ubg = []

        # Initial state equality
        g.append(X[:, 0] - X0)
        lbg += [0.0] * nx
        ubg += [0.0] * nx

        # Control bounds
        umin = np.array([-self.max_velocity, -self.max_velocity], dtype=float)
        umax = np.array([self.max_velocity, self.max_velocity], dtype=float)

        # Rate limits on controls (acceleration constraint)
        du_lim = self.max_acceleration * dt

        # Dynamics across the horizon
        for k in range(N):
            xk = X[:, k]
            uk = U[:, k]

            # unpack state
            pk = xk[0:3]  # pos
            qk = xk[3:7]  # quat (wxyz)
            wl = xk[7]
            wr = xk[8]

            # base kinematics using *state* wheel rates
            v = 0.5 * r * (wl + wr)  # forward speed (body +X)
            wz = (r / L) * (wr - wl)  # yaw rate (about body +Z)

            # move along body +X mapped to world by quaternion
            ex_world = self._ex_from_quat(qk)
            dp = v * dt * ex_world
            dp = ca.vertcat(dp[0], dp[1], 0.0)  # planar integration (keep z)

            p_next = pk + dp

            # orientation update by body-yaw increment
            delta_yaw = wz * dt
            dq = self._dq_yaw_body(delta_yaw)
            q_next = self._quat_mul(qk, dq)
            q_next = self._quat_normalize(q_next)

            # wheel state next = commanded control (zero-order hold)
            wl_next = uk[0]
            wr_next = uk[1]

            x_next = ca.SX.zeros(nx)
            x_next[0:3] = p_next
            x_next[3:7] = q_next
            x_next[7] = wl_next
            x_next[8] = wr_next

            # model equality
            g.append(X[:, k + 1] - x_next)
            lbg += [0.0] * nx
            ubg += [0.0] * nx

            # wheel acceleration (rate) constraint on controls
            du = (U[:, 0] - Uprev) if k == 0 else (U[:, k] - U[:, k - 1])
            g.append(du)
            lbg += [-du_lim, -du_lim]
            ubg += [du_lim, du_lim]

        # Variable bounds: X free (you can add wheel-state clamps if desired), U box-bounded
        lbx = [-ca.inf] * (nx * (N + 1))
        ubx = [ca.inf] * (nx * (N + 1))
        for _ in range(N):
            lbx += list(umin)
            ubx += list(umax)

        # Objective from user
        try:
            J = self._cost_function(X, U)
        except TypeError as e:
            raise RuntimeError(
                "Your residual_function must have signature cost(X, U)."
            ) from e

        # Pack NLP
        w = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        gall = ca.vertcat(*g)
        nlp = {"x": w, "f": J, "g": gall, "p": ca.vertcat(X0, Uprev)}

        opts = {
            "ipopt": {
                "print_level": 0,
                "max_iter": 300,
                "tol": 1e-4,
                "linear_solver": "mumps",
            },
            "print_time": 0,
        }
        self.solver = ca.nlpsol("solver", "ipopt", nlp, opts)

        # caches
        self._pack_shapes = {"nx": nx, "nu": nu, "N": N}
        self._lbx = np.array(lbx, dtype=float)
        self._ubx = np.array(ubx, dtype=float)
        self._lbg = np.array(lbg, dtype=float)
        self._ubg = np.array(ubg, dtype=float)

    # -----------------------------
    # Solve
    # -----------------------------
    def compute_control(self, initial_state, previous_control):
        """
        Solve the MPC and return (wl_cmd, wr_cmd) for the current step.

        initial_state: array-like (9,)  -> [x,y,z,qw,qx,qy,qz, wl, wr]
        previous_control: array-like (2,) -> [wl_cmd_prev, wr_cmd_prev]
        """
        if self.solver is None:
            raise RuntimeError("Call compile_solver() after setting the cost function.")

        nx = self._pack_shapes["nx"]
        nu = self._pack_shapes["nu"]
        N = self._pack_shapes["N"]

        X0 = np.asarray(initial_state, dtype=float).reshape(nx)
        Uprev = np.asarray(previous_control, dtype=float).reshape(nu)

        # warm-start
        X_guess = np.tile(X0.reshape(nx, 1), (1, N + 1))
        U_guess = np.tile(Uprev.reshape(nu, 1), (1, N))

        w0 = np.concatenate(
            [X_guess.reshape(-1, order="F"), U_guess.reshape(-1, order="F")]
        )

        p_vec = np.concatenate([X0, Uprev])

        sol = self.solver(
            x0=w0,
            lbx=self._lbx,
            ubx=self._ubx,
            lbg=self._lbg,
            ubg=self._ubg,
            p=p_vec,
        )

        w_opt = np.array(sol["x"]).reshape(-1)
        nXblk = nx * (N + 1)
        U_opt = w_opt[nXblk:].reshape(nu, N, order="F")
        u0 = U_opt[:, 0]

        self.last_traj = {
            "X": w_opt[:nXblk].reshape(nx, N + 1, order="F"),
            "U": U_opt.copy(),
            "cost": float(sol["f"]),
        }
        return u0

    # -----------------------------
    # Quaternion helpers
    # -----------------------------
    def _ex_from_quat(self, q):
        """
        Forward axis (body +X) expressed in world, from quaternion q=(w,x,y,z).
        Returns 3x1 CasADi vector.
        """
        w, x, y, z = q[0], q[1], q[2], q[3]
        ww, xx, yy, zz = w * w, x * x, y * y, z * z
        wx, wy, wz = w * x, w * y, w * z
        xy, xz, yz = x * y, x * z, y * z
        return ca.vertcat(
            ww + xx - yy - zz,
            2 * (xy + wz),
            2 * (xz - wy),
        )

    def _quat_mul(self, q, p):
        """Hamilton product (w,x,y,z)."""
        w1, x1, y1, z1 = q[0], q[1], q[2], q[3]
        w2, x2, y2, z2 = p[0], p[1], p[2], p[3]
        return ca.vertcat(
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )

    def _quat_normalize(self, q):
        return q / ca.sqrt(ca.sumsqr(q) + 1e-12)

    def _dq_yaw_body(self, delta_yaw):
        """Quaternion for rotation about body +Z by delta_yaw (w,x,y,z)."""
        c = ca.cos(0.5 * delta_yaw)
        s = ca.sin(0.5 * delta_yaw)
        return ca.vertcat(c, 0.0, 0.0, s)

    # -----------------------------
    # Optional: frame lookups (unchanged)
    # -----------------------------
    def lookup_frame_transformation(self, frame_name, timestamp="real time"):
        if timestamp == "real time":
            return self.entity_dict[frame_name].transform_real_time
        else:
            return self.entity_dict[frame_name].transform_current

    def lookup_frame_transformation_casadi(self, frame_name, timestamp="real time"):
        if timestamp == "real time":
            return self.entity_dict[frame_name].transform_realtime.data.cpu().numpy()
        else:
            return self.entity_dict[frame_name].transform_current.data.cpu().numpy()

    def lookup_frame_position(self, frame_name, timestamp="real time"):
        if timestamp == "real time":
            return self.entity_dict[frame_name].transform_real_time.translation
        else:
            return self.entity_dict[frame_name].transform_current.translation

    def lookup_frame_rotation(self, frame_name, timestamp="real time"):
        if timestamp == "real time":
            return self.entity_dict[frame_name].transform_real_time.quaternion
        else:
            return self.entity_dict[frame_name].transform_current.quaternion

    def lookup_frame_position_casadi(self, frame_name, timestamp="real time"):
        if timestamp == "real time":
            return self.entity_dict[frame_name].transform_real_time.translation
        else:
            return (
                self.entity_dict[frame_name].transform_current.translation.cpu().numpy()
            )

    def lookup_frame_rotation_casadi(self, frame_name, timestamp="real time"):
        if timestamp == "real time":
            return (
                self.entity_dict[frame_name]
                .transform_real_time.quaternion.cpu()
                .numpy()
            )
        else:
            return (
                self.entity_dict[frame_name].transform_current.quaternion.cpu().numpy()
            )


class ManipulatorPositionController(Controller):
    """
    Horizon-based position controller with joint-velocity limits and
    pluggable residual (cost) functions.
    """

    def __init__(
        self, urdf_path, ee_frame="end_effector", time_step=0.02, horizon_length=40
    ):
        # --- kinematics & limits ------------------------------------------------
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.ee_id = self.model.getFrameId(ee_frame)

        self.nq = self.model.nq
        self.num_states = self.nq
        self.dt = time_step
        self.N = horizon_length
        self.num_controls = self.nq

        # URDF-provided limits
        self.q_min = self.model.lowerPositionLimit
        self.q_max = self.model.upperPositionLimit
        self.v_max = self.model.velocityLimit  # [rad/s]

        self._residual_fns = []  # user costs
        self._solver = None  # built once
        self._has_gripper = False

    @property
    def has_gripper(self):
        """
        Returns True if the robot has a gripper, False otherwise.
        """
        return self._has_gripper

    def getFrameDescription(self):
        return NotImplementedError("Subclasses should implement this method.")

    def getStateDescription(self):
        return NotImplementedError("Subclasses should implement this method.")

    def getControlDescription(self):
        return NotImplementedError("Subclasses should implement this method.")

    def getState(self, joint_pos, joint_vel):
        return np.array(joint_pos)[: self.nq]

    def getControl(self, joint_pos, joint_vel):
        return np.array(joint_pos)[: self.nq]

    # ---------- public ----------------------------------------------------------

    def calculate_frame_position(self, frame_name, joint_state):
        """
        Calculate the position of a frame on the robot using kinematics.
        :param frame_name: Name of the frame.
        :param joint_state: Joint state of the robot.
        :return: Position of the frame in the robot's base frame.
        """
        self.forward_kinematics(joint_state)
        return self.lookup_frame_position(frame_name)

    def calculate_frame_rotation(self, frame_name, joint_state):
        """
        Calculate the position of a frame on the robot using kinematics.
        :param frame_name: Name of the frame.
        :param joint_state: Joint state of the robot.
        :return: Position of the frame in the robot's base frame.
        """
        self.forward_kinematics(joint_state)
        return matrix_to_quat(self.lookup_frame_rotation(frame_name))

    def forward_kinematics(self, q: np.array):
        """
        Compute the end-effector position given joint angles.

        Parameters:
        - q: (nq,) numpy array of joint angles

        Returns:
        - (3,) numpy array of end-effector position in world frame
        """
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return

    def lookup_frame_position(self, frame_name, timestamp="real time", numpy=False):
        if timestamp == "real time":
            return self.data.oMf[frame_name].translation
        else:
            # Handle other timestamps if needed
            pass

    def lookup_frame_rotation(self, frame_name, timestamp="real time", numpy=False):
        if timestamp == "real time":
            return matrix_to_quat(self.data.oMf[frame_name].rotation.cpu().numpy())
        else:
            # Handle other timestamps if needed
            pass

    def update_cost_function(self, name_scope) -> None:
        """
        Plug in a new CasADi scalar-valued objective.

        cost_fn(X, U, P) → SX/MX
            X : 3×(N+1)   state matrix
            U : 2×N       control matrix
            P : parameter vector
        """
        self._residual_fns = name_scope["residual_function"]
        self._build_nlp()

    def compute_control(self, initial_state, previous_control):

        sol = self._solver(
            x0=self._initial_guess(initial_state),
            p=np.zeros(1) if previous_control is None else previous_control,
            lbg=self._lbg,
            ubg=self._ubg,
            lbx=self._lbx,
            ubx=self._ubx,
        )
        q_plan = ca.reshape(sol["x"], self.nq, self.N + 1).full().T
        return q_plan[1].flatten(), q_plan

    # ---------- internals -------------------------------------------------------
    def _build_nlp(self):
        q = ca.SX.sym("q", self.nq, self.N + 1)
        J, g = 0, []

        # --- accumulate user residuals -----------------------------------------
        for k in range(self.N):
            qk = q[:, k]
            dq = q[:, k + 1] - qk
            J += self._residual_fns(qk, dq) * self.dt

            # velocity-limit constraints:  q_{k+1} - q_k  ∈ [-v_max*dt, v_max*dt]
            g.append(dq - self.v_max * self.dt)  # ≤ 0
            g.append(-dq - self.v_max * self.dt)  # ≤ 0

        # decision vector
        w = ca.vec(q)
        nlp = {"x": w, "f": J, "g": ca.vertcat(*g)}
        opts = {"ipopt.print_level": 0, "print_time": 0}
        self._solver = ca.nlpsol("s", "ipopt", nlp, opts)

        # --- bounds -------------------------------------------------------------
        # inequality constraints: all g  ≤ 0
        m = (self.N) * 2 * self.nq
        self._lbg = -np.inf * np.ones(m)
        self._ubg = np.zeros(m)

        # position bounds per time step
        lbq = np.tile(self.q_min.reshape(-1, 1), (1, self.N + 1))
        ubq = np.tile(self.q_max.reshape(-1, 1), (1, self.N + 1))
        self._lbx = lbq.flatten()
        self._ubx = ubq.flatten()

    def _initial_guess(self, q0):
        return np.tile(q0.reshape(-1, 1), (1, self.N + 1)).flatten()


class FrankaPositionController(ManipulatorPositionController):
    """
    Horizon-based position controller with joint-velocity limits and
    pluggable residual (cost) functions.
    """

    _frames = [
        FrameEntity(
            "panda_fingertip",
            "End-effector frame, finger direction is positive z in tool frame",
        ),
        FrameEntity("panda_link4", "Elbow frame at middle of the arm"),
        FrameEntity("panda_link0", "Base frame of the robot, mounted on the ground"),
    ]

    def __init__(
        self, urdf_path, ee_frame="panda_fingertip", time_step=0.02, horizon_length=40
    ):
        # --- kinematics & limits ------------------------------------------------
        super().__init__(
            urdf_path=urdf_path,
            ee_frame=ee_frame,
            time_step=time_step,
            horizon_length=horizon_length,
        )

        fk_fingertip = build_symbolic_fk(self.model, "panda_fingertip")
        fk_link4 = build_symbolic_fk(self.model, "panda_link4")
        fk_link0 = build_symbolic_fk(self.model, "panda_link0")
        self._fk_fun = {
            "panda_fingertip": fk_fingertip,
            "panda_link4": fk_link4,
            "panda_link0": fk_link0,
        }
        self._has_gripper = True

        # Optimization variables
        self.opti = ca.Opti()
        self.q = self.opti.variable(self.nq, self.N + 1)
        self.dq = self.opti.variable(self.nq, self.N)

        self.q_init = np.zeros(self.nq)
        self.last_solution = None

    def getFrameDescription(self):
        """
        Returns a list of StateEntity objects describing the state of the robot.
        """
        return self._frames

    def getStateDescription(self):
        theta0 = StateEntity("theta0", 0, "Joint 0 position", True)
        theta1 = StateEntity("theta1", 1, "Joint 1 position", True)
        theta2 = StateEntity("theta2", 2, "Joint 2 position", True)
        theta3 = StateEntity("theta3", 3, "Joint 3 position", True)
        theta4 = StateEntity("theta4", 4, "Joint 4 position", True)
        theta5 = StateEntity("theta5", 5, "Joint 5 position", True)
        theta6 = StateEntity("theta6", 6, "Joint 6 position", True)
        return [theta0, theta1, theta2, theta3, theta4, theta5, theta6]

    def getControlDescription(self):
        theta0 = ControlEntity("theta0", 0, "Joint 0 velocity", True)
        theta1 = ControlEntity("theta1", 1, "Joint 1 velocity", True)
        theta2 = ControlEntity("theta2", 2, "Joint 2 velocity", True)
        theta3 = ControlEntity("theta3", 3, "Joint 3 velocity", True)
        theta4 = ControlEntity("theta4", 4, "Joint 4 velocity", True)
        theta5 = ControlEntity("theta5", 5, "Joint 5 velocity", True)
        theta6 = ControlEntity("theta6", 6, "Joint 6 velocity", True)

        return [theta0, theta1, theta2, theta3, theta4, theta5, theta6]

    def frame_index(self, frame_name):
        if frame_name in self._frame_index:
            return self._frame_index[frame_name]
        else:
            raise ValueError(
                f"Frame {frame_name} not found in FrankaPositionController frame index."
            )

    # ---------- public ----------------------------------------------------------
    @staticmethod
    def state_trimming(state):
        """
        Trims the state to the first 7 joint positions and the gripper position.
        :param state: The state vector containing joint positions and gripper position.
        :return: Trimmed state vector.
        """
        return state[:7]

    def calculate_frame_position_symbolic(self, frame_name, joint_state):
        """
        Calculate the position of a frame on the robot using kinematics.
        :param frame_name: Name of the frame.
        :param joint_state: Joint state of the robot.
        :return: Position of the frame in the robot's base frame.
        """
        if frame_name in self._fk_fun:
            return self._fk_fun[frame_name](joint_state)[0]
        raise NotImplementedError(
            "Symbolic frame position calculation not implemented for this frame."
        )

    def calculate_frame_rotation_symbolic(self, frame_name, joint_state):
        """
        Calculate the position of a frame on the robot using kinematics.
        :param frame_name: Name of the frame.
        :param joint_state: Joint state of the robot.
        :return: Position of the frame in the robot's base frame.
        """
        if frame_name in self._fk_fun:
            return self._fk_fun[frame_name](joint_state)[1]
        raise NotImplementedError(
            "Symbolic frame rotation calculation not implemented for this frame."
        )

    def calculate_frame_position(self, frame_name, joint_state):
        """
        Calculate the position of a frame on the robot using kinematics.
        :param frame_name: Name of the frame.
        :param joint_state: Joint state of the robot.
        :return: Position of the frame in the robot's base frame.
        """
        self.forward_kinematics(joint_state)
        return self.lookup_frame_position(frame_name)

    def calculate_frame_rotation(self, frame_name, joint_state):
        """
        Calculate the position of a frame on the robot using kinematics.
        :param frame_name: Name of the frame.
        :param joint_state: Joint state of the robot.
        :return: Position of the frame in the robot's base frame.
        """
        self.forward_kinematics(joint_state)
        return matrix_to_quat(self.lookup_frame_rotation(frame_name))

    def forward_kinematics(self, q: np.array):
        """
        Compute the end-effector position given joint angles.

        Parameters:
        - q: (nq,) numpy array of joint angles

        Returns:
        - (3,) numpy array of end-effector position in world frame
        """
        pin.forwardKinematics(
            self.model, self.data, FrankaPositionController.state_trimming(q)
        )
        pin.updateFramePlacements(self.model, self.data)
        return

    def lookup_frame_position(self, frame_name):

        return self.data.oMf[self.model.getFrameId(frame_name)].translation

    def lookup_frame_rotation(self, frame_name):

        return matrix_to_quat(self.data.oMf[self.model.getFrameId(frame_name)].rotation)

    def update_cost_function(self, name_scope) -> None:
        """
        Plug in a new CasADi scalar-valued objective.

        cost_fn(X, U, P) → SX/MX
            X : 3×(N+1)   state matrix
            U : 2×N       control matrix
            P : parameter vector
        """
        self._residual_fns = name_scope["residual_function"]
        self._build_nlp()

    def compute_control(self, initial_state, previous_control):

        if self.last_solution is not None:
            self.opti.set_initial(self.q, self.last_solution["q"])
            self.opti.set_initial(self.dq, self.last_solution["dq"])
        else:
            q_guess = ca.repmat(self.q_init, 1, self.N + 1)
            dq_guess = ca.DM.zeros(self.nq, self.N)
            self.opti.set_initial(self.q, q_guess)
            self.opti.set_initial(self.dq, dq_guess)


        self.opti.solver("ipopt", {"print_level": 0}, {"max_iter": 100})
        sol = self.opti.solve()

        q_sol = sol.value(self.q)
        dq_sol = sol.value(self.dq)
        self.last_solution = {"q": q_sol, "dq": dq_sol}
        return q_sol


    # ---------- internals -------------------------------------------------------
    def _build_nlp(self) -> None:


        cost = 0
        for k in range(self.N):
            qk = self.q[:, k]
            dqk = self.dq[:, k]
            cost += self._residual_fns(qk, dqk)
        self.opti.minimize(cost)
        self.add_constraints()

    def add_constraints(self):
        # Initial state constraint
        self.opti.subject_to(self.q[:, 0] == self.q_init)

        for k in range(self.N):
            qk = self.q[:, k]
            dqk = self.dq[:, k]
            q_next = self.q[:, k + 1]

            # State integration: q_{k+1} = q_k + dt * dq_k
            self.opti.subject_to(q_next == qk + self.dt * dqk)

            # Joint position limits
            self.opti.subject_to(qk <= self.q_max)
            self.opti.subject_to(qk >= self.q_min)

            # Joint velocity limits
            self.opti.subject_to(dqk <= self.v_max)
            self.opti.subject_to(dqk >= -self.v_max)

        # Final position limits
        self.opti.subject_to(self.q[:, self.N] <= self.q_max)
        self.opti.subject_to(self.q[:, self.N] >= self.q_min)

    def warm_start(self, q_now, v_prev=None):
        """
        Build an initial guess for the next NLP solve.

        Parameters
        ----------
        q_now   : (nq,)   current measured joint positions
        v_prev  : (nq*N,) previous optimal velocity vector (flattened).
                        Pass None on the very first call.

        Returns
        -------
        x0 : (nq*(2*N+1),)  warm-start decision vector
        """
        nq, N, dt = self.nq, self.N, self.dt

        # ---- 1. velocity guess --------------------------------------------------
        if v_prev is None:  # first MPC iteration
            v_mat = np.zeros((nq, N))
        elif (
            len(v_prev.shape) == 1 or v_prev.shape[0] * v_prev.shape[1] == self.nq
        ):  # previous v is flattened
            v_mat = np.tile(v_prev.reshape((nq, 1)), (1, N))  # repeat to fill N cols
        else:
            v_mat = v_prev.reshape((nq, N))  #  ➜  nq × N
            v_mat = np.hstack(
                [v_mat[:, 1:], v_mat[:, -1:]]  # shift left
            )  # pad with last col (hold)

        # ---- 2. position guess --------------------------------------------------
        q_mat = np.empty((nq, N + 1))
        q_mat[:, 0] = q_now
        for k in range(N):
            q_mat[:, k + 1] = q_mat[:, k] + v_mat[:, k] * dt

        # ---- 3. flatten to decision vector -------------------------------------
        return np.concatenate(
            [
                q_mat.ravel(order="F"),  # col-major flatten = CasADi
                v_mat.ravel(order="F"),
            ]
        )


class ManipulatorIKController(Controller):
    """
    Horizon-based position controller with joint-velocity limits and
    pluggable residual (cost) functions.
    """

    def __init__(
        self,
        urdf_path,
        ee_frame="end_effector",
        time_step=0.02,
        horizon_length=40,
        ik_fun=None,
        fk_fun=None,
        control_output_mode="relative_joint",
        device="cpu",
    ):
        super().__init__()

        self.device = device
        # --- kinematics & limits ------------------------------------------------
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.ee_id = self.model.getFrameId(ee_frame)
        self.ee_frame = ee_frame

        self.nq = 9
        self.nu = 8
        self.num_states = self.nq
        self.dt = time_step
        self.N = horizon_length
        self.num_controls = self.nq

        # end effector position and speed limits
        self.v_lin_max = 0.5  # m/s
        self.v_ang_max = np.pi/2  # rad/s
        self.v_gripper_max = 1.0  # m/s

        # symbolic variables
        self.ee_pos = ca.SX.sym("ee_pos", 3, horizon_length)
        self.ee_quat = ca.SX.sym(
            "ee_quat", 4, horizon_length
        )  # quaternion (w, x, y, z)
        self.ee_vel = ca.SX.sym("ee_vel", 6, horizon_length - 1)  # linear + angular

        self.opti = ca.Opti()
        self.X = self.opti.variable(
            self.nq, horizon_length
        )  # [pos (3), quat (4), gripper (1)]
        self.U = self.opti.variable(
            self.nu, horizon_length - 1
        )  # [vel_lin (3), vel_ang (3), gripper vel(1)]

        self.last_sol = None
        self._residual_fns = []  # user costs
        self._solver = None  # built once
        self._has_gripper = False

        if ik_fun is None:
            self._ik_fun = self.default_ik
        else:
            self._ik_fun = ik_fun
        if fk_fun is None:
            self._fk_fun = self.calculate_frame_transformation
        else:
            self._fk_fun = fk_fun

        self.last_solution = None  # for warm start

        self.offset_trans = np.eye(4)
        self.control_output_mode = control_output_mode

        self._frames = []

    def save_odometry_checkpoint(self):
        for frame in self._frames:
            self.entity_dict[frame.name].transform_current = copy.deepcopy(self.entity_dict[frame.name].transform_real_time)

    def forward_kinematics(self, q):
        """
        Compute the end-effector position given joint angles.

        Parameters:
        - q: (nq,) torch tensor of joint angles

        Returns:
        - (3,) torch tensor of end-effector position in world frame
        """
        pin.forwardKinematics(
            self.model, self.data, FrankaPositionController.state_trimming(q)
        )
        pin.updateFramePlacements(self.model, self.data)

        for frame in self._frames:
            self.entity_dict[frame.name].transform_calculated = Transform(
                torch.from_numpy(
                    self.data.oMf[self.model.getFrameId(frame.name)].homogeneous
                )
                .to(self.device)
                .unsqueeze(0)
            )
        return

    def calculate_frame_transformation(self, frame_name, joint_state):
        """
        Calculate the transformation of a frame on the robot using kinematics.
        :param frame_name: Name of the frame.
        :param joint_state: Joint state of the robot.
        :return: Transformation of the frame in the robot's base frame.
        """
        if isinstance(joint_state, torch.Tensor):
            self.forward_kinematics(joint_state.cpu().numpy())
            return self.lookup_frame_transformation(frame_name)
        else:
            self.forward_kinematics(joint_state)
            return self.lookup_frame_transformation_casadi(frame_name)

    def lookup_frame_transformation_casadi(self, frame_name, timestamp="real time"):
        if timestamp == "real time":
            return self.data.oMf[self.model.getFrameId(frame_name)].homogeneous
        else:
            return self.lookup_frame_transformation(frame_name, timestamp).data.cpu().numpy()[0]

    def lookup_frame_transformation(self, frame_name, timestamp="real time"):
        if frame_name == 'world':
            raise ValueError("World frame cannot be used as root as it has no origin")
        if timestamp == "real time":
            return self.entity_dict["agent"].transform_real_time.inverse @ self.entity_dict[frame_name].transform_real_time
        else:
            return  self.entity_dict["agent"].transform_current.inverse @ self.entity_dict[frame_name].transform_current

    def lookup_frame_position(self, frame_name, timestamp="real time"):
        t = self.lookup_frame_transformation(frame_name, timestamp)
        return t.translation


    def lookup_frame_rotation(self, frame_name, timestamp="real time"):
        t = self.lookup_frame_transformation(frame_name, timestamp)
        return t.quaternion

    def lookup_frame_position_casadi(self, frame_name, timestamp="real time"):
        if timestamp == "real time":
            return self.data.oMf[self.model.getFrameId(frame_name)].translation
        else:
            return (
                self.lookup_frame_transformation_casadi(frame_name, timestamp)[:3,3]
            )

    def lookup_frame_rotation_casadi(self, frame_name, timestamp="real time"):
        if timestamp == "real time":
            return matrix_to_quat(
                self.data.oMf[self.model.getFrameId(frame_name)].rotation
            )
        else:
            return matrix_to_quat(
                self.lookup_frame_transformation_casadi(frame_name, timestamp)[:3,:3]
            )

    def default_ik(
        self,
        p_des,
        quat_des,
        q_init=None,
        frame_name="tool",
        max_iters=1000,
        tol=1e-4,
        alpha=1.0,  # start with full step
        lam=1e-6,  # task–space weights
    ):
        raise NotImplementedError("Default IK not implemented yet.")

    @property
    def has_gripper(self):
        """
        Returns True if the robot has a gripper, False otherwise.
        """
        return self._has_gripper

    def getFrameDescription(self):
        return NotImplementedError("Subclasses should implement this method.")

    def getStateDescription(self):
        pos_x = StateEntity("position x", 0, "end effector position x", True)
        pos_y = StateEntity("position y", 1, "end effector position y", True)
        pos_z = StateEntity("position z", 2, "end effector position z", True)
        rot_w = StateEntity("quaternion w", 3, "end effector quaternion w", True)
        rot_x = StateEntity("quaternion x", 4, "end effector quaternion x", True)
        rot_y = StateEntity("quaternion y", 5, "end effector quaternion y", True)
        rot_z = StateEntity("quaternion z", 6, "end effector quaternion z", True)
        gripper_l = StateEntity(
            "gripper left finger",
            7,
            "gripper joint state, 1.0 is fully open, 0.0 is fully closed, effort feedback only",
            True,
        )
        gripper_r = StateEntity(
            "gripper right finger",
            8,
            "gripper joint state, 1.0 is fully open, 0.0 is fully closed, effort feedback only",
            True,
        )
        return [pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, rot_w, gripper_l, gripper_r]

    def getControlDescription(self):
        pos_x = StateEntity("position x", 0, "end effector velocity x", True)
        pos_y = StateEntity("position y", 1, "end effector velocity y", True)
        pos_z = StateEntity("position z", 2, "end effector velocity z", True)
        rot_x = StateEntity("rotation x", 3, "end effector angular velocity x", True)
        rot_y = StateEntity("rotation y", 4, "end effector angular velocity y", True)
        rot_z = StateEntity("rotation z", 5, "end effector angular velocity z", True)
        left_gripper = StateEntity(
            "gripper left finger", 6, "gripper left finger velocity", True
        )
        right_gripper = StateEntity(
            "gripper right finger", 7, "gripper right finger velocity", True
        )
        return [pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, left_gripper, right_gripper]

    def getState(self, joint_pos, joint_vel=None):
        return joint_pos[0].cpu().numpy().flatten()

    def getControl(self, joint_pos, joint_vel=None):
        # Expose current joint velocity to downstream callers (e.g., as previous_control),
        # so the IK seeding can use q + dq*dt when available.
        if joint_vel is None:
            return None
        return joint_vel[0].cpu().numpy().flatten()

    def _add_dynamics_constraints(self):
        self.opti.subject_to(self.X[:, 0] == self.X0)

        for k in range(self.N - 1):
            pos_k = self.X[:3, k]
            quat_k = self.X[3:7, k]
            gripper_k = self.X[7:, k]
            vel_k = self.U[:, k]

            pos_next = pos_k + self.dt * vel_k[:3]
            gripper_next = gripper_k + self.dt * vel_k[6:]
            quat_dot = 0.5 * quat_multiply_matrix(quat_k) @ ca.vertcat(0, vel_k[3:6])
            quat_next = quat_k + self.dt * quat_dot
            quat_next = quat_next / ca.norm_2(quat_next)

            self.opti.subject_to(self.X[:3, k + 1] == pos_next)
            self.opti.subject_to(self.X[3:7, k + 1] == quat_next)
            self.opti.subject_to(self.X[7:, k + 1] == gripper_next)

            self.opti.subject_to(ca.sumsqr(vel_k[:3]) <= (self.v_lin_max) ** 2)
            self.opti.subject_to(ca.sumsqr(vel_k[3:6]) <= (self.v_ang_max) ** 2)
            self.opti.subject_to(ca.sumsqr(vel_k[6:]) <= (self.v_gripper_max) ** 2)

    def update_cost_function(self, name_scope, cost_only=False) -> None:
        """
        Plug in a new CasADi scalar-valued objective.

        cost_fn(X, U, P) → SX/MX
            X : 3×(N+1)   state matrix
            U : 2×N       control matrix
            P : parameter vector
        """

        if cost_only:
            self._residual_fns = name_scope["residual_function"]
            return
        self.opti = ca.Opti()

        # Decision variables: absolute EE pose state and gripper state.
        self.X = self.opti.variable(self.nq, self.N)  # [pos(3); quat(4); gripper(2)]
        self.U = self.opti.variable(
            self.nu, self.N - 1
        )  # [linear vel(3); angular vel(3); gripper vel(2)]

        # Parameter for initial absolute EE pose / gripper state.
        self.X0 = self.opti.parameter(self.nq)

        self._add_dynamics_constraints()

        self._residual_fns = name_scope["residual_function"]
        cost = 0
        for k in range(self.N - 1):
            state_k = self.X[:, k]
            control_k = self.U[:, k]
            cost += self._residual_fns(state_k)
        self.opti.minimize(cost)

        self.opti.solver(
            "ipopt", {"print_time": 0, "ipopt.print_level": 0}, {"max_iter": 1000}
        )


    def _set_initial_guess(
        self, ee_pos_init, ee_quat_init, gripper_init, u_init=None, warm_start: bool = True
    ):
        if warm_start and self.last_solution:
            # Warm start from previous solution
            for k in range(self.N):
                self.opti.set_initial(self.X[:3, k], self.last_solution["X"][k, :3])
                self.opti.set_initial(self.X[3:7, k], self.last_solution["X"][k, 3:7])
                self.opti.set_initial(self.X[7:, k], self.last_solution["X"][k, 7:])
            for k in range(self.N - 1):
                self.opti.set_initial(self.U[:, k], self.last_solution["U"][k, :])
            return

        # Flat initial guess based on current absolute state.
        for k in range(self.N):
            self.opti.set_initial(self.X[:3, k], ee_pos_init)
            self.opti.set_initial(self.X[3:7, k], ee_quat_init)
            self.opti.set_initial(self.X[7:, k], gripper_init)
        for k in range(self.N - 1):
            self.opti.set_initial(self.U[:, k], 0.0 if u_init is None else u_init)

    # pure Cartesian-level solve (identical to previous revision)
    def solve(
        self,
        p_curr: np.ndarray,
        quat_curr: np.ndarray,
        gripper_curr: np.ndarray,
        u_init: np.ndarray | None = None,
        warm_start: bool = True,
    ):


        self._set_initial_guess(
            p_curr, quat_curr, gripper_curr, u_init=u_init, warm_start=warm_start
        )
        self.opti.set_value(self.X0, np.concatenate([p_curr, quat_curr, gripper_curr]))

        sol = self.opti.solve()
        x_abs = sol.value(self.X).T
        u = sol.value(self.U).T
        self.last_solution = {"X": x_abs, "U": u}
        return x_abs, u

    def _serialize_relative_ee_actions(self, x_abs: np.ndarray, u: np.ndarray):
        actions = np.zeros((x_abs.shape[0], 8), dtype=float)
        actions[:-1, :6] = u[:, :6]
        actions[-1, :6] = u[-1, :6]
        actions[:-1, 6:] = np.where(u[:, 6:] > 0.0, 1.0, -1.0)
        actions[-1, 6:] = actions[-2, 6:] if actions.shape[0] > 1 else np.where(u[-1, 6:] > 0.0, 1.0, -1.0)
        return actions

    def _serialize_absolute_ee_actions(self, x_abs: np.ndarray, u: np.ndarray):
        actions = np.zeros((x_abs.shape[0], 11), dtype=float)
        quat = torch.as_tensor(x_abs[:, 3:7], dtype=torch.float32)
        rot6d = quat_to_rotation_6d(quat).cpu().numpy()
        actions[:, :3] = x_abs[:, :3]
        actions[:, 3:9] = rot6d
        actions[:-1, 9:] = np.where(u[:, 6:] > 0.0, 1.0, -1.0)
        actions[-1, 9:] = actions[-2, 9:] if actions.shape[0] > 1 else np.where(u[-1, 6:] > 0.0, 1.0, -1.0)
        return actions

    def compute_control(self, q_curr, dq_curr):

        if self._ik_fun is None:
            raise RuntimeError(
                "IK function must be set using set_ik_function() before compiling the solver."
            )
        if self._fk_fun is None:
            raise RuntimeError(
                "FK function must be set using set_fk_function() before compiling the solver."
            )
        t_curr = self._fk_fun(self.ee_frame, q_curr)

        u_init = None
        if dq_curr is not None:
            try:
                q_pin = np.asarray(q_curr).reshape(-1)[: self.model.nq]
                dq_pin = np.asarray(dq_curr).reshape(-1)[: self.model.nv]
                if q_pin.shape[0] == self.model.nq and dq_pin.shape[0] == self.model.nv:
                    pin.forwardKinematics(self.model, self.data, q_pin, dq_pin)
                    pin.updateFramePlacements(self.model, self.data)
                    J = pin.computeFrameJacobian(
                        self.model,
                        self.data,
                        q_pin,
                        self.ee_id,
                        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
                    )
                    twist6 = (J @ dq_pin).reshape(-1)
                    u_init = np.zeros(self.nu, dtype=float)
                    u_init[:6] = twist6[:6]
                    dq_full = np.asarray(dq_curr).reshape(-1)
                    if dq_full.shape[0] >= 2:
                        u_init[6:] = dq_full[-2:]

                    # Clamp to configured EE/gripper speed limits for a feasible seed.
                    lin = u_init[:3]
                    ang = u_init[3:6]
                    lin_norm = np.linalg.norm(lin)
                    ang_norm = np.linalg.norm(ang)
                    if lin_norm > float(self.v_lin_max) and lin_norm > 1e-12:
                        u_init[:3] = lin * (float(self.v_lin_max) / lin_norm)
                    if ang_norm > float(self.v_ang_max) and ang_norm > 1e-12:
                        u_init[3:6] = ang * (float(self.v_ang_max) / ang_norm)
                    u_init[6:] = np.clip(u_init[6:], -float(self.v_gripper_max), float(self.v_gripper_max))
            except Exception:
                u_init = None

        x_abs, u = self.solve(
            t_curr[:3, 3].copy(),
            matrix_to_quat(t_curr[:3, :3].copy()),
            q_curr[-2:].copy(),
            u_init=u_init,
            warm_start=False,
        )

        if self.control_output_mode == "relative_ee":
            return self._serialize_relative_ee_actions(x_abs, u)
        if self.control_output_mode == "absolute_ee":
            return self._serialize_absolute_ee_actions(x_abs, u)

        x_q = np.zeros((x_abs.shape[0], len(q_curr)))

        # Seed IK with a one-step-forward joint prediction when velocity is available.
        q_seed0 = q_curr.copy()
        if dq_curr is not None:
            dq_curr = np.asarray(dq_curr).reshape(-1)
            if dq_curr.shape[0] == q_seed0.shape[0]:
                q_seed0[:-2] = q_seed0[:-2] + dq_curr[:-2] * self.dt

        x_q[0, :] = q_curr
        for k in range(0, x_abs.shape[0]):
            x_se3 = pin.SE3(
                quat=pin.Quaternion(x_abs[k, 3], x_abs[k, 4], x_abs[k, 5], x_abs[k, 6]),
                translation=x_abs[k, :3],
            )
            x_se3_ = x_se3.homogeneous @ np.linalg.pinv(self.offset_trans)
            q_seed = q_seed0[:-2] if k == 0 else x_q[k - 1, :-2]
            x_q_k = self._ik_fun(
                x_se3_[:3, 3],
                matrix_to_quat(x_se3_[:3, :3]),
                q_seed,
                self.ee_frame,
            )[0]
            x_q[k, :-2] = x_q_k
            x_q[k, -2:] = x_abs[k, 7:]

        if self.control_output_mode in ("relative_joint", "absolute_joint"):
            return x_q
        raise ValueError(f"Unsupported control_output_mode: {self.control_output_mode}")

    def getFrameDescription(self):
        """
        Returns a list of StateEntity objects describing the state of the robot.
        """
        return self._frames

    @property
    def cost_function(self):
        if self._residual_fns is None:
            raise RuntimeError("Cost function is not set.")
        return self._residual_fns


class FrankaIKController(ManipulatorIKController):
    """
    Horizon-based position controller with joint-velocity limits and
    pluggable residual (cost) functions.
    """

    def __init__(
        self,
        urdf_path,
        ee_frame="panda_fingertip",
        time_step=0.02,
        horizon_length=40,
        ik_fun=None,
        fk_fun=None,
        control_output_mode="relative_joint",
        device="cpu",
    ):
        super().__init__(
            urdf_path=urdf_path,
            ee_frame=ee_frame,
            time_step=time_step,
            horizon_length=horizon_length,
            ik_fun=ik_fun,
            fk_fun=fk_fun,
            control_output_mode=control_output_mode,
            device=device,
        )
        self.name = "Franka IK Controller"
        self._frames = [
            FrameEntity(
                "panda_fingertip",
                "End-effector frame, +Z = tool pointing/aiming direction; +Y = along the gripper fingers. CAN ONLY GRASP ONE OBJECT AT A TIME.",
            ),
            FrameEntity("panda_link4", "Elbow frame at middle of the arm"),
        ]

        for frame in self._frames:
            self.entity_dict[frame.name] = FrameEntity(frame.name, frame.description)

        self._has_gripper = True

        self.forward_kinematics(np.zeros(self.nq))
        self.data.oMf[self.model.getFrameId("panda_hand")]
        self.offset_trans = (
            self.data.oMf[self.model.getFrameId("panda_hand")].inverse()
            * self.data.oMf[self.model.getFrameId("panda_fingertip")]
        ).homogeneous

    def appendTrajectory(self, joint_pos, joint_vel, joint_eff):
        batch_size = joint_pos.shape[0]
        pos_list, quat_list = [], []

        for i in range(batch_size):
            # Forward kinematics only works on one row at a time
            self.forward_kinematics(joint_pos[i].cpu().numpy().flatten())

            quat = self.lookup_frame_rotation("panda_fingertip")
            pos = self.lookup_frame_position("panda_fingertip")

            quat_list.append(
                torch.as_tensor(quat, dtype=joint_pos.dtype, device=joint_pos.device)
            )
            pos_list.append(
                torch.as_tensor(pos, dtype=joint_pos.dtype, device=joint_pos.device)
            )

        # Stack back into (B, …) tensors
        ee_quat = torch.stack(quat_list, dim=0)
        ee_pos = torch.stack(pos_list, dim=0)


class FR3FinRayIKController(ManipulatorIKController):
    """
    Horizon-based position controller with joint-velocity limits and
    pluggable residual (cost) functions.
    """

    def __init__(
        self,
        urdf_path,
        ee_frame="panda_fingertip",
        time_step=0.02,
        horizon_length=40,
        ik_fun=None,
        fk_fun=None,
        control_output_mode="relative_joint",
        device="cpu",
    ):
        super().__init__(
            urdf_path=urdf_path,
            ee_frame=ee_frame,
            time_step=time_step,
            horizon_length=horizon_length,
            ik_fun=ik_fun,
            fk_fun=fk_fun,
            control_output_mode=control_output_mode,
            device=device,
        )
        self.name = "FR3FinRay Manipulator IK Controller"
        self._frames = [
            FrameEntity(
                "panda_fingertip",
                "End-effector frame, +Z = tool pointing/aiming direction; +Y = along the gripper fingers. CAN ONLY GRASP ONE OBJECT AT A TIME.",
            ),
            FrameEntity("panda_link4", "Elbow frame at middle of the arm"),
        ]

        for frame in self._frames:
            self.entity_dict[frame.name] = FrameEntity(frame.name, frame.description)

        self._has_gripper = True

        self.forward_kinematics(np.zeros(self.nq))
        self.offset_trans = (
            self.data.oMf[self.model.getFrameId("panda_link7")].inverse()
            * self.data.oMf[self.model.getFrameId("panda_fingertip")]
        ).homogeneous

    def appendTrajectory(self, joint_pos, joint_vel, joint_eff):
        batch_size = joint_pos.shape[0]
        pos_list, quat_list = [], []

        for i in range(batch_size):
            # Forward kinematics only works on one row at a time
            self.forward_kinematics(joint_pos[i].cpu().numpy().flatten())

            quat = self.lookup_frame_rotation("panda_fingertip")
            pos = self.lookup_frame_position("panda_fingertip")

            quat_list.append(
                torch.as_tensor(quat, dtype=joint_pos.dtype, device=joint_pos.device)
            )
            pos_list.append(
                torch.as_tensor(pos, dtype=joint_pos.dtype, device=joint_pos.device)
            )

        # Stack back into (B, …) tensors
        ee_quat = torch.stack(quat_list, dim=0)
        ee_pos = torch.stack(pos_list, dim=0)


class UR5IKController(ManipulatorIKController):
    """Six-joint UR5 FK/IK with two logical gripper slots in Cartesian MPC."""

    def __init__(self, urdf_path, ee_frame="panda_fingertip", time_step=0.02,
                 horizon_length=40, ik_fun=None, fk_fun=None,
                 control_output_mode="relative_joint", device="cpu"):
        super().__init__(urdf_path, ee_frame, time_step, horizon_length,
                         ik_fun, fk_fun, control_output_mode, device)
        if self.model.nq != 6:
            raise ValueError("UR5 controller requires the prepared six-joint controller URDF")
        self.name = "UR5 Robotiq IK Controller"
        self._frames = [
            FrameEntity("panda_fingertip", "UR5 Robotiq grasp center; +Z points along the tool."),
            FrameEntity("panda_link4", "UR5 forearm_link frame."),
            FrameEntity("panda_link7", "UR5 wrist_3_link frame."),
        ]
        for frame in self._frames:
            self.entity_dict[frame.name] = FrameEntity(frame.name, frame.description)
        self._has_gripper = True
        self.forward_kinematics(np.zeros(6))
        self.offset_trans = (self.data.oMf[self.model.getFrameId("panda_link7")].inverse()
                             * self.data.oMf[self.ee_id]).homogeneous
        if ik_fun is None:
            from kuafu.mpc.controller.bounded_ik import BoundedPoseIK
            self._ik_fun = BoundedPoseIK(self.model, frame_name="wrist_3_link", time_step=time_step)

    def forward_kinematics(self, q):
        pin.forwardKinematics(self.model, self.data, np.asarray(q, dtype=float).reshape(-1)[:6])
        pin.updateFramePlacements(self.model, self.data)
        for frame in self._frames:
            self.entity_dict[frame.name].transform_calculated = Transform(
                torch.from_numpy(self.data.oMf[self.model.getFrameId(frame.name)].homogeneous.copy())
                .to(self.device).unsqueeze(0)
            )

    def calculate_frame_transformation(self, frame_name, q):
        self.forward_kinematics(q)
        return self.data.oMf[self.model.getFrameId(frame_name)].homogeneous.copy()


class KinovaIKController(ManipulatorIKController):
    """Kinova Gen3 controller model for SUNProgramExecutor MPC collection."""

    def __init__(
        self,
        urdf_path,
        ee_frame="panda_fingertip",
        time_step=0.02,
        horizon_length=40,
        ik_fun=None,
        fk_fun=None,
        control_output_mode="relative_joint",
        device="cpu",
    ):
        super().__init__(
            urdf_path=urdf_path,
            ee_frame=ee_frame,
            time_step=time_step,
            horizon_length=horizon_length,
            ik_fun=ik_fun,
            fk_fun=fk_fun,
            control_output_mode=control_output_mode,
            device=device,
        )
        self.name = "Kinova Gen3 IK Controller"
        self._frames = [
            FrameEntity(
                "panda_fingertip",
                "Kinova Robotiq virtual fingertip/tool frame: +Z = tool pointing/aiming direction; +Y = along the gripper fingers.",
            ),
            FrameEntity("panda_link4", "Kinova forearm_link frame exposed as panda_link4 for learned constraints."),
            FrameEntity("panda_link7", "Kinova bracelet_link frame exposed as panda_link7 for learned constraints."),
        ]

        for frame in self._frames:
            self.entity_dict[frame.name] = FrameEntity(frame.name, frame.description)

        self._has_gripper = True
        self._pin_q_index = self._build_pin_q_index()

        self.forward_kinematics(np.zeros(self.nq))
        self.offset_trans = (
            self.data.oMf[self.model.getFrameId("panda_link7")].inverse()
            * self.data.oMf[self.model.getFrameId("panda_fingertip")]
        ).homogeneous

    def _build_pin_q_index(self):
        q_index = {}
        joint_names = (
            *[f"joint_{i}" for i in range(1, 8)],
            "finger_joint",
            "right_outer_knuckle_joint",
            "left_inner_knuckle_joint",
            "right_inner_knuckle_joint",
            "left_inner_finger_joint",
            "right_inner_finger_joint",
            # Legacy USD-converted Robotiq names kept so older diagnostic URDFs
            # still build a valid Pinocchio q vector.
            "robotiq_85_left_knuckle_joint",
            "robotiq_85_right_knuckle_joint",
            "robotiq_85_left_inner_knuckle_joint",
            "robotiq_85_right_inner_knuckle_joint",
            "robotiq_85_left_finger_tip_joint",
            "robotiq_85_right_finger_tip_joint",
            "left_inner_finger_knuckle_joint",
            "right_inner_finger_knuckle_joint",
        )
        for name in joint_names:
            if not self.model.existJointName(name):
                continue
            joint_id = self.model.getJointId(name)
            joint = self.model.joints[joint_id]
            if joint.nq in (1, 2):
                q_index[name] = (joint.idx_q, joint.nq)
        return q_index

    def _set_pin_joint(self, q_pin, joint_name, value):
        entry = self._pin_q_index.get(joint_name)
        if entry is None:
            return
        idx, nq = entry
        value = float(value)
        if nq == 1:
            q_pin[idx] = value
        elif nq == 2:
            q_pin[idx] = np.cos(value)
            q_pin[idx + 1] = np.sin(value)

    def _to_pin_q(self, q):
        q_in = np.asarray(q, dtype=float).reshape(-1)
        q_pin = pin.neutral(self.model)
        for i in range(min(7, q_in.shape[0])):
            self._set_pin_joint(q_pin, f"joint_{i + 1}", q_in[i])

        left_knuckle = q_in[7] if q_in.shape[0] > 7 else 0.0
        # Official Kortex Robotiq kinematics use finger_joint as the master.
        # Isaac still reports the right driver with the opposite sign, so only
        # the left/master close amount is used for Pinocchio FK here.
        self._set_pin_joint(q_pin, "finger_joint", left_knuckle)
        self._set_pin_joint(q_pin, "right_outer_knuckle_joint", left_knuckle)
        self._set_pin_joint(q_pin, "left_inner_knuckle_joint", left_knuckle)
        self._set_pin_joint(q_pin, "right_inner_knuckle_joint", left_knuckle)
        self._set_pin_joint(q_pin, "left_inner_finger_joint", -left_knuckle)
        self._set_pin_joint(q_pin, "right_inner_finger_joint", -left_knuckle)

        # Legacy USD-converted Robotiq kinematics used opposite signs on the
        # right side and different inner-link joint names.
        self._set_pin_joint(q_pin, "robotiq_85_left_knuckle_joint", left_knuckle)
        self._set_pin_joint(q_pin, "robotiq_85_right_knuckle_joint", -left_knuckle)
        self._set_pin_joint(q_pin, "robotiq_85_left_inner_knuckle_joint", left_knuckle)
        self._set_pin_joint(q_pin, "robotiq_85_right_inner_knuckle_joint", -left_knuckle)
        self._set_pin_joint(q_pin, "robotiq_85_left_finger_tip_joint", -left_knuckle)
        self._set_pin_joint(q_pin, "robotiq_85_right_finger_tip_joint", left_knuckle)
        self._set_pin_joint(q_pin, "left_inner_finger_knuckle_joint", left_knuckle)
        self._set_pin_joint(q_pin, "right_inner_finger_knuckle_joint", -left_knuckle)
        return q_pin

    def forward_kinematics(self, q):
        pin.forwardKinematics(self.model, self.data, self._to_pin_q(q))
        pin.updateFramePlacements(self.model, self.data)

        for frame in self._frames:
            self.entity_dict[frame.name].transform_calculated = Transform(
                torch.from_numpy(
                    self.data.oMf[self.model.getFrameId(frame.name)].homogeneous
                )
                .to(self.device)
                .unsqueeze(0)
            )
        return
