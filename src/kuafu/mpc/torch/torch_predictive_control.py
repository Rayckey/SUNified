import numpy as np
from kuafu.mpc.simulation.struct import StateEntity, ControlEntity, FrameEntity
from kuafu.mpc.controller.controller_utils import (
    matrix_to_quat,
    quat_to_matrix,
    quat_multiply_matrix,
)
from isaaclab.utils.math import (
    quat_from_euler_xyz,
    quat_from_matrix,
    matrix_from_euler,
    matrix_from_quat,
    euler_xyz_from_quat,
    make_pose,
    pose_inv,
    unmake_pose,
    pose_in_A_to_pose_in_B,
)


class Controller:
    """
    Base class for controllers. Provides methods to get state and control descriptions.
    """

    def __init__(self, device: str = "cpu", num_envs: int = 1):
        """
        Initializes the controller.
        """
        self.entity_dict = {}
        self.device = device
        self.num_envs = num_envs
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

    def update_cost_function(self, name_scope) -> None:
        """
        Plug in a new CasADi scalar-valued objective.

        cost_fn(X, U, P) → SX/MX
            X : state matrix
            U : control matrix
            P : parameter vector
        """
        raise NotImplementedError("Subclasses should implement this method.")


class DifferentialDriveMPC(Controller):
    def __init__(
        self,
        wheel_radius,
        base_length,
        horizon_length,
        time_step,
        max_wheel_velocity,
        max_wheel_acceleration,
    ):
        return NotImplementedError("Subclasses should implement this method.")

    def getState(self, xyz, euler, vel_xyz, vel_euler):
        """
        Convert the state from the simulation to the MPC state representation.

        Parameters:
        - xyz: 3D position [x, y, z].
        - euler: 3D orientation [roll, pitch, yaw].

        Returns:
        - A numpy array representing the state in the form [x, y, theta, v, omega].
        """
        x = xyz[0]
        y = xyz[1]
        theta = euler[2]

        return np.array([x, y, theta])

    def getControl(self, xyz, euler, vel_xyz, vel_euler):
        return NotImplementedError("Subclasses should implement this method.")

    def update_cost_function(self, name_scope) -> None:
        """
        Plug in a new CasADi scalar-valued objective.

        cost_fn(X, U, P) → SX/MX
            X : 3×(N+1)   state matrix
            U : 2×N       control matrix
            P : parameter vector
        """
        self.cost_function = name_scope["residual_function"]
        self.compile_solver()

    def compile_solver(self):
        return NotImplementedError("Subclasses should implement this method.")

    def compute_control(self, initial_state, previous_control):
        return NotImplementedError("Subclasses should implement this method.")


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
        return NotImplementedError("You should not be here")

    def getStateDescription(self):
        return NotImplementedError("You should not be here")

    def getControlDescription(self):
        return NotImplementedError("You should not be here")

    def getState(self, joint_pos, joint_vel):
        return np.array(joint_pos)[: self.nq]

    def getControl(self, joint_pos, joint_vel):
        return np.array(joint_pos)[: self.nq]

    # ---------- public ----------------------------------------------------------

    def calculate_frame_position(self, frame_name, joint_state):
        return NotImplementedError("Subclasses should implement this method.")

    def calculate_frame_rotation(self, frame_name, joint_state):
        return NotImplementedError("Subclasses should implement this method.")

    def forward_kinematics(self, q: np.array):
        return NotImplementedError("Subclasses should implement this method.")

    def lookup_frame_position(self, frame_name, timestamp="real time"):
        return NotImplementedError("Subclasses should implement this method.")

    def lookup_frame_rotation(self, frame_name, timestamp="real time"):
        return NotImplementedError("Subclasses should implement this method.")

    def update_cost_function(self, name_scope):
        return NotImplementedError("Subclasses should implement this method.")

    def compute_control(self, initial_state, previous_control):
        return NotImplementedError("Subclasses should implement this method.")


class ManipulatorIKController(Controller):
    """
    Horizon-based position controller with joint-velocity limits and
    pluggable residual (cost) functions.
    """

    def __init__(
        self,
        ee_frame: str = "panda_fingertip",
        device: str = "cpu",
        num_envs: int = 1,
    ):
        """Minimal IK controller for batched learned constraints.

        Notes:
            The original implementation accepted URDF, time_step and horizon
            parameters but did not use them. Those have been removed so callers
            no longer need to provide unused configuration.
        """
        super().__init__(device=device, num_envs=num_envs)
        self.ee_frame = ee_frame

        # Fixed-size minimal state used by APIWrapper.getMinimumState()
        self.nq = 9
        self.nu = 8
        self.num_states = self.nq
        self.num_controls = self.nq

        self._residual_fns = []  # user costs
        self._has_gripper = False
        self._frames = []

        # End-effector speed limits (matches kuafu/controller/mpc.py ManipulatorIKController)
        self.v_lin_max = 0.5   # m/s
        self.v_ang_max = np.pi   # rad/s

    def save_odometry_checkpoint(self):
        for frame in self._frames:
            self.entity_dict[frame.name].transform_current = (
                self.entity_dict[frame.name].transform_real_time.clone().detach()
            )

    def lookup_frame_transformation(self, frame_name, timestamp="real time"):
        if timestamp == "real time":
            return self.entity_dict[frame_name].transform_real_time
        else:
            return self.entity_dict[frame_name].transform_current

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

    @property
    def has_gripper(self):
        """
        Returns True if the robot has a gripper, False otherwise.
        """
        return self._has_gripper

    def getFrameDescription(self):
        return NotImplementedError("You should not be here")

    def getStateDescription(self):
        return NotImplementedError("You should not be here")

    def getControlDescription(self):
        return NotImplementedError("You should not be here")

    def getState(self, joint_pos):
        return joint_pos

    def getControl(self, joint_pos):
        return NotImplementedError("You should not be here")

    def update_cost_function(self, name_scope) -> None:
        self._residual_fns = name_scope["residual_function"]

    @property
    def cost_function(self):
        return self._residual_fns

    # pure Cartesian-level solve (identical to previous revision)
    def solve(
        self, p_curr: np.ndarray, quat_curr: np.ndarray, gripper_curr: np.ndarray
    ):
        raise NotImplementedError("You should not be here")

    def compute_control(self, q_curr, _):
        raise NotImplementedError("You should not be here")


class FrankaIKController(ManipulatorIKController):
    """
    Horizon-based position controller with joint-velocity limits and
    pluggable residual (cost) functions.
    """

    def __init__(
        self,
        ee_frame: str = "panda_fingertip",
        device: str = "cpu",
        num_envs: int = 1,
    ):
        """Franka-specific IK controller for learned constraint evaluation.

        This variant intentionally does not depend on a URDF path or horizon
        config; it only exposes the frames used by the API wrapper.
        """
        super().__init__(ee_frame=ee_frame, device=device, num_envs=num_envs)

        self._frames = [
            FrameEntity(
                "panda_fingertip",
                "End-effector frame, tool pointing direction is positive z, flat side is positive x. Never align x with object direction",
            ),
            FrameEntity("panda_link4", "Elbow frame at middle of the arm"),
        ]

        self._has_gripper = True


    def getFrameDescription(self):
        return self._frames


class UR5IKController(ManipulatorIKController):
    """UR5 tool and arm frames exposed through Kuafu's shared constraint aliases."""

    def __init__(self, ee_frame="panda_fingertip", device="cpu", num_envs=1):
        super().__init__(ee_frame=ee_frame, device=device, num_envs=num_envs)
        self._frames = [
            FrameEntity("panda_fingertip", "UR5 Robotiq grasp center; +Z points along the tool."),
            FrameEntity("panda_link4", "UR5 forearm_link frame."),
            FrameEntity("panda_link7", "UR5 wrist_3_link frame."),
        ]
        self._has_gripper = True

    def getFrameDescription(self):
        return self._frames


class KinovaIKController(ManipulatorIKController):
    """Kinova Gen3 frame descriptors for learned constraint evaluation."""

    def __init__(
        self,
        ee_frame: str = "panda_fingertip",
        device: str = "cpu",
        num_envs: int = 1,
    ):
        super().__init__(ee_frame=ee_frame, device=device, num_envs=num_envs)

        self._frames = [
            FrameEntity(
                "panda_fingertip",
                "Kinova Robotiq virtual fingertip/tool frame.",
            ),
            FrameEntity("panda_link4", "Kinova forearm_link frame exposed as panda_link4."),
            FrameEntity("panda_link7", "Kinova bracelet_link frame exposed as panda_link7."),
        ]
        self._has_gripper = True

    def getFrameDescription(self):
        return self._frames
