"""Single-environment Isaac adapter using the recorded camera and action contract."""

import copy

import numpy as np
import torch

from .artifacts import json_digest
from .object_frame import OBJECT_RELATIVE, ObjectFrame, validate_relative_contract


def recorded_tolerance_updates(current, recorded):
    """Recover teacher tolerances while requiring identical task semantics."""
    candidate = copy.deepcopy(current)
    updates = {}
    if len(candidate) != len(recorded):
        raise ValueError("Simulator SUN Program stage count differs from the recording")
    for sid, (actual, expected) in enumerate(zip(candidate, recorded)):
        if len(actual["constraints"]) != len(expected["constraints"]):
            raise ValueError("Simulator SUN Program constraints differ from the recording")
        for term, (constraint, target) in enumerate(zip(actual["constraints"], expected["constraints"])):
            args = constraint.setdefault("args", {})
            for key in ("tolerance_m", "tolerance_rad"):
                if key in target.get("args", {}) and args.get(key) != target["args"][key]:
                    value = target["args"][key]
                    if type(value) not in (int, float) or not np.isfinite(value) or not 0 <= value <= 10:
                        raise ValueError("Recorded tolerances must be finite scalars in [0, 10]")
                    args[key] = value
                    updates.setdefault(sid, {}).setdefault(term, {})[key] = value
    if json_digest(candidate) != json_digest(recorded):
        raise ValueError("Simulator SUN Program definitions differ beyond recorded tolerances")
    return updates


class IsaacAdapter:
    def __init__(self, *, experiment, dataset, protocol, device):
        self.object_frame = None
        self.relative = (validate_relative_contract(dataset["export"])
                         if dataset["export"]["action_format"] == OBJECT_RELATIVE else None)
        recording = dataset.get("recording")
        if not recording or not recording["camera_keys"]:
            raise ValueError("Recollect with point_cloud_recording metadata for matched Isaac preprocessing")
        from isaaclab.app import AppLauncher
        self.app = AppLauncher(headless=True, enable_cameras=True, device=device, fast_shutdown=False).app
        self.env = None
        try:
            import gymnasium as gym
            import isaaclab_tasks  # noqa: F401
            from isaaclab_tasks.utils import parse_env_cfg
            import kuafu.isaac.learned  # noqa: F401
            from kuafu.isaac.learned import kinova, panda  # noqa: F401
            self.cfg = parse_env_cfg(protocol["task"], device=device, num_envs=1)
            self.cfg.seed = int(protocol.get("seeds", [0])[0])
            self.cfg.enable_cameras = True
            self.cfg.action_mode = "absolute_ee"
            self.cfg.action_space = 11
            if hasattr(self.cfg, "mpc_config"):
                self.cfg.mpc_config["num_envs"] = 1
            self.env = gym.make(protocol["task"], cfg=self.cfg)
            self.raw = self.env.unwrapped
            if abs(float(self.raw.step_dt) - recording["control_dt"]) > 1e-8:
                raise ValueError("Simulator control period differs from the demonstrations")
            if self.raw.program_runtime.compiled_program.stage_count() != len(dataset["stage_names"]):
                raise ValueError("Simulator SUN Program stage count differs from the dataset")
            if not callable(getattr(self.raw, "get_true_success_count", None)):
                raise ValueError("Simulator must expose true complete-task success counts")
            from kuafu.data.stage_labels import program_stage_schema
            self.restored_tolerances = {}
            if recording.get("program_stages") is not None:
                self.restored_tolerances = recorded_tolerance_updates(program_stage_schema(self.raw), recording["program_stages"])
                if self.restored_tolerances:
                    from kuafu.rl.isaac.update_utilities import update_all_parameters
                    update_all_parameters(self.raw, all_stages_parameters=self.restored_tolerances)
                if json_digest(recording["program_stages"]) != json_digest(program_stage_schema(self.raw)):
                    raise ValueError("Simulator did not restore the recorded SUN Program tolerances")
            self.recording, self.dataset = recording, dataset
            self.device = device
            self.export = dataset["export"]
            self.format = self.export["action_format"]
            from kuafu2real.clip_zone import PointCloudClipZone
            self.zone = PointCloudClipZone.from_mapping(self.export["clip_zone"])
            self.camera_warmup_frames = int(protocol.get("camera_warmup_frames", 3))
            self.arm_ids = None
            if self.format == "bc":
                robot = self.raw.scene.articulations["robot"]
                self.arm_ids, names = robot.find_joints(self.cfg.robot.arm_actuated_joints, preserve_order=True)
                if len(self.arm_ids) != 7 or list(names) != list(self.cfg.robot.arm_actuated_joints):
                    raise ValueError("BC requires seven ordered measured arm joints")
        except BaseException:
            import traceback
            traceback.print_exc()
            self.close()
            raise

    def reset(self, *, seed, condition):
        if set(condition) - {"object_offsets"}:
            raise ValueError("Isaac conditions support object_offsets in environment-frame meters")
        self.env.reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        for name, offset in condition.get("object_offsets", {}).items():
            if name == "robot":
                raise ValueError("Choose a task object for the controlled placement variation")
            asset = self.raw.scene[name]
            displacement = torch.as_tensor(offset, device=self.raw.device, dtype=torch.float32)
            if displacement.shape != (3,) or not torch.isfinite(displacement).all():
                raise ValueError("Object offsets must be three finite coordinates")
            pose = asset.data.root_state_w[:, :7].clone()
            pose[:, :3] += displacement
            asset.write_root_pose_to_sim(pose)
        self.raw.scene.update(self.raw.physics_dt)
        self.raw.update_data_in_controller()
        if self.relative:
            self.object_frame = ObjectFrame(self._reference_pose())
        for _ in range(self.camera_warmup_frames):
            self.raw.sim.render()
            self.raw.scene.update(self.raw.physics_dt)
        state = {}
        for name, asset in {**self.raw.scene.articulations, **self.raw.scene.rigid_objects}.items():
            state[name] = {"root": asset.data.root_state_w.detach().cpu().tolist()}
            if hasattr(asset.data, "joint_pos"):
                state[name]["joints"] = asset.data.joint_pos.detach().cpu().tolist()
        api = self.raw.program_runtime.controller_api_wrapper
        self.successes_before = self.raw.get_true_success_count()
        return {"obs": self._observation(), "stage_id": self._stage(), "initial_state_id": json_digest(state)}

    def _stage(self):
        return int(self.raw.current_stage.reshape(-1)[0].item())

    def _reference_pose(self):
        name = self.relative["reference_object"]
        asset = self.raw.scene[name]
        pose = asset.data.root_state_w[0, :7].detach().cpu().numpy().copy()
        pose[:3] -= self.raw.scene.env_origins[0].detach().cpu().numpy()
        return pose

    def _check_reference(self):
        if self.relative:
            if self.object_frame is None:
                raise RuntimeError("Reset the object-relative adapter before using it")
            self.object_frame.check_stationary(self._reference_pose())

    def encode_observation(self, observation):
        """Apply the dataset frame after camera fusion and workspace cropping."""
        self._check_reference()
        if self.relative:
            return {"point_cloud": self.object_frame.encode_cloud(observation["point_cloud"]),
                    "agent_pos": self.object_frame.encode_pose(observation["agent_pos"])}
        return observation

    def decode_action(self, action):
        """Restore the environment-frame target before gripper/IK conversion."""
        self._check_reference()
        return self.object_frame.decode_pose(action) if self.relative else action

    def _observation(self):
        from isaaclab.utils.math import matrix_from_quat
        from kuafu.isaac.learned.saver import fused_pc
        from kuafu2real.dataset import _fixed_point_cloud_frame
        from kuafu2real.action import rotation_matrix_to_rot6d
        depths, intrinsics, extrinsics = [], [], []
        for name in self.recording["camera_keys"]:
            sensor = self.raw.scene.sensors[name]
            depths.append(sensor.data.output["distance_to_image_plane"].detach())
            intrinsics.append(sensor.data.intrinsic_matrices.detach())
            extrinsic = torch.eye(4, device=self.raw.device).unsqueeze(0)
            extrinsic[:, :3, :3] = matrix_from_quat(sensor.data.quat_w_ros)
            extrinsic[:, :3, 3] = sensor.data.pos_w - self.raw.scene.env_origins
            extrinsics.append(extrinsic)
        cloud = fused_pc(depths=depths, rgbs=None, intrinsics=intrinsics, extrinsics=extrinsics,
                         num_cams=len(depths), depth_scale=1.0, **self.recording["fusion"])[0, 0]
        cloud = _fixed_point_cloud_frame(cloud.cpu().numpy(), self.export["num_points"], self.rng,
                                        clip_zone=self.zone)[..., :3]
        api = self.raw.program_runtime.controller_api_wrapper
        if self.format == "bc":
            joints = self.raw.scene.articulations["robot"].data.joint_pos[:, self.arm_ids]
            state = torch.cat((joints, api.gripper_openness().reshape(1, 1).clamp(0, 1)), dim=-1)[0].cpu().numpy()
        else:
            pos = api.lookup_robot_frame_position("panda_fingertip")[0].cpu().numpy()
            quat = api.lookup_robot_frame_rotation("panda_fingertip")
            rotation = matrix_from_quat(quat)[0].cpu().numpy()
            state = np.r_[pos, rotation_matrix_to_rot6d(rotation), api.gripper_openness().reshape(-1).clamp(0, 1).cpu().numpy()]
        return self.encode_observation({"point_cloud": cloud.astype(np.float32), "agent_pos": state.astype(np.float32)})

    def step(self, action):
        from kuafu2real.gripper import normalized_gripper_to_joint_targets
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        action = self.decode_action(action)
        action = torch.as_tensor(action, device=self.raw.device, dtype=torch.float32).reshape(1, -1)
        if action.shape != (1, 10) or not torch.isfinite(action).all():
            raise ValueError("Expected one finite Kuafu2Real 10D EE/gripper command")
        api = self.raw.program_runtime.controller_api_wrapper
        gripper = normalized_gripper_to_joint_targets(action[:, 9:], api._gripper_opened, api._gripper_closed)
        command = torch.cat((action[:, :9], gripper), dim=-1)
        advanced_before, _ = self.raw.get_stage_completion_cumulative()
        for _ in range(self.recording["record_stride"]):
            _, _, terminated, truncated, _ = self.env.step(command)
            done = bool((terminated | truncated).item())
            if done:
                break
        success = self.raw.get_true_success_count() > self.successes_before
        advanced_after, _ = self.raw.get_stage_completion_cumulative()
        advanced = [stage for stage, count in advanced_after.items() if count > advanced_before.get(stage, 0)]
        return {"obs": None if done else self._observation(), "stage_id": -1 if done else self._stage(),
                "done": done, "success": success, "advanced_stages": advanced}

    def close(self):
        if self.env is not None:
            self.env.close()
        if getattr(self, "app", None) is not None:
            self.app.close()


def create_adapter(**kwargs):
    return IsaacAdapter(**kwargs)
