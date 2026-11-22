from isaaclab.utils import convert_dict_to_backend
from isaaclab.utils.math import unproject_depth, transform_points, subtract_frame_transforms
from pxr import Sdf, UsdGeom, Gf, Sdf
import numpy as np
import torch
import os
import shutil
import subprocess
from omni.usd import get_prim_at_path
from PIL import Image, ImageColor, ImageDraw, ImageFont
import omni.usd
from kuafu.mpc.simulation.sun_program_executor import SUNProgramExecutor
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg

# Try to import the underlying set_camera_view function
try:
    from isaacsim.core.utils.viewports import set_camera_view as _set_camera_view_direct
    HAS_DIRECT_SET_CAMERA = True
except ImportError:
    HAS_DIRECT_SET_CAMERA = False

def zoom_in_camera(env, stage, zoom_factor=0.5):
    """Zoom in the camera by moving it closer to the target.
    
    Args:
        env: The gymnasium environment
        stage: The USD stage
        zoom_factor: Factor to zoom in (0.0 = no change, 1.0 = move to target, default 0.5)
    """
    camera_prim_path = "/OmniverseKit_Persp"
    camera_prim = stage.GetPrimAtPath(camera_prim_path)

    if not camera_prim or not camera_prim.IsValid():
        print(f"Warning: Camera prim at {camera_prim_path} not found or invalid")
        return

    try:
        # Get camera world transform
        xformable = UsdGeom.Xformable(camera_prim)
        world_transform = xformable.ComputeLocalToWorldTransform(0)

        # Extract position (translation) from transform matrix
        # The translation is in the last column, first 3 rows
        eye = np.array([
            world_transform[3][0],
            world_transform[3][1],
            world_transform[3][2]
        ])

        # Get camera target (center of interest) from the camera prim
        # The target is typically stored in the omni:kit:centerOfInterest attribute
        coi_attr = camera_prim.GetAttribute("omni:kit:centerOfInterest")
        if coi_attr and coi_attr.IsValid():
            coi_local = coi_attr.Get()
            # Transform the local center of interest to world space
            coi_world = world_transform.Transform(Gf.Vec3d(coi_local))
            target = np.array([coi_world[0], coi_world[1], coi_world[2]])
        else:
            # If center of interest is not available, use a default target
            # Calculate forward direction from camera transform
            forward = np.array([
                -world_transform[2][0],  # -Z axis in camera space
                -world_transform[2][1],
                -world_transform[2][2]
            ])
            # Use a default distance for the target
            target = eye + forward * 5.0

        # Calculate direction vector from eye to target
        direction = target - eye
        distance = np.linalg.norm(direction)

        if distance < 1e-6:
            print(f"Warning: Camera eye and target are too close (distance: {distance})")
            return

        # Zoom in by moving the eye closer to the target
        new_eye = eye + direction * zoom_factor

        print(f"Zooming camera: eye {eye} -> {new_eye}, target {target}, zoom_factor={zoom_factor}")

        # Try multiple approaches to set the camera view
        success = False

        # Approach 1: Use the environment's set_camera_view (wrapper)
        try:
            env.unwrapped.sim.set_camera_view(
                eye=tuple(new_eye),
                target=tuple(target)
            )
            success = True
            print("Camera view updated via env.unwrapped.sim.set_camera_view")
        except Exception as e:
            print(f"Failed to set camera via env wrapper: {e}")

        # Approach 2: Try direct call to underlying function
        if not success and HAS_DIRECT_SET_CAMERA:
            try:
                _set_camera_view_direct(tuple(new_eye), tuple(target), camera_prim_path)
                success = True
                print("Camera view updated via direct set_camera_view call")
            except Exception as e:
                print(f"Failed to set camera via direct call: {e}")

        # Approach 3: Try using viewport camera controller if available
        if not success and hasattr(env.unwrapped, 'viewport_camera_controller') and env.unwrapped.viewport_camera_controller is not None:
            try:
                env.unwrapped.viewport_camera_controller.update_view_location(
                    eye=tuple(new_eye),
                    lookat=tuple(target)
                )
                success = True
                print("Camera view updated via viewport_camera_controller")
            except Exception as e:
                print(f"Failed to set camera via viewport_camera_controller: {e}")

        if not success:
            print("Warning: All methods to set camera view failed. The camera may not be updated.")
            print("This might happen if viewport rendering is not enabled.")
    except Exception as e:
        print(f"Error zooming camera: {e}")
        import traceback
        traceback.print_exc()

def extract_data_from_camera(scene_camera, camera_index):

    single_cam_data = convert_dict_to_backend(
        {
            "rgb": scene_camera.data.output["rgb"][camera_index],
            "distance_to_image_plane": scene_camera.data.output[
                "distance_to_image_plane"
            ][camera_index],
            "instance_segmentation_fast": scene_camera.data.output[
                "instance_segmentation_fast"
            ][camera_index],
        },
        backend="numpy",
    )

    # Get camera info
    single_cam_info = {}

    if hasattr(scene_camera.data, "info"):
        single_cam_info = scene_camera.data.info
    else:
        single_cam_info = {"rgb": None}  # Provide empty info if not available

    return single_cam_data, single_cam_info


def write_camera_image(scene_camera, camera_index, rgb_save_dir):

    single_cam_data, single_cam_info = extract_data_from_camera(
        scene_camera, camera_index
    )

    # Pack data for replicator format
    rep_output = {"annotators": {}}
    for key, data in single_cam_data.items():
        info = single_cam_info.get(key, None)
        if info is not None:
            rep_output["annotators"][key] = {"render_product": {"data": data, **info}}
        else:
            rep_output["annotators"][key] = {"render_product": {"data": data}}


    from torchvision.utils import make_grid, save_image

    rgb = single_cam_data["rgb"]
    depth = single_cam_data["distance_to_image_plane"]

    im = Image.fromarray(single_cam_data["rgb"])
    im.save(f"{rgb_save_dir}_rgb_cam.png", format="PNG")

    im = Image.fromarray(reframe(single_cam_data["distance_to_image_plane"].squeeze()))
    im.save(f"{rgb_save_dir}_depth.png")

    return single_cam_data, single_cam_info


def reframe(picture):
    picture -= np.min(picture)
    picture /= np.max(picture) - np.min(picture)
    picture *= 255
    return picture.astype(np.uint8)


def get_object_id_from_prim_path(prim_path):
    return Sdf.Path(prim_path).name.lower()


def flip_segmentation_ids(dig: dict, allowed: list):
    """
    Flip the segmentation IDs in the dictionary to match the allowed set.

    Parameters
    ----------
    dig : dict
        Dictionary containing segmentation IDs.
    allowed : set
        Set of allowed segmentation IDs.

    Returns
    -------
    dict
        Dictionary with flipped segmentation IDs.
    """
    flipped = {}
    for k, v in dig.items():
        short_v = get_object_id_from_prim_path(v)
        if short_v in allowed:
            flipped[short_v] = k

    return flipped


def prim_paths_above_area_rgba(
    seg: np.ndarray, id_to_prim4: dict[tuple[int], str], area_thresh: float = 0.05
) -> list[str]:
    """
    Return prim-paths whose (R,G,B,A) ID occupies at least `area_thresh`
    fraction of the pixels in a (H,W,4) uint8 segmentation image.

    Parameters
    ----------
    seg : np.ndarray, shape (H, W, 4), dtype uint8
        Segmentation image in RGBA format.
    id_to_prim4 : dict[(int,int,int,int), str]
        Maps RGBA tuples to prim paths (e.g. {(1,0,0,255): "/World/chair"}).
    area_thresh : float, default 0.05
        Minimum fraction of total image area an object must cover.

    Returns
    -------
    List[str]
        Prim paths that meet the area criterion.
    """

    if seg.ndim != 3 or seg.shape[2] != 4 or seg.dtype != np.uint8:
        raise ValueError("seg must be a (H,W,4) uint8 array.")

    # ----------------------------------------------------------------
    # 1. Pack each RGBA pixel into a single uint32 for fast counting.
    #    Little-endian packing:  ID = R + 256*G + 256**2*B + 256**3*A
    # ----------------------------------------------------------------
    multipliers = np.array([1, 256, 256**2, 256**3], dtype=np.uint32)
    packed_seg = (seg.astype(np.uint32) @ multipliers).reshape(-1)

    # ----------------------------------------------------------------
    # 2. Build a packed version of the id→prim map
    # ----------------------------------------------------------------
    packed_map = {
        int(r) + 256 * int(g) + 256**2 * int(b) + 256**3 * int(a): prim
        for (r, g, b, a), prim in id_to_prim4.items()
    }

    # ----------------------------------------------------------------
    # 3. Histogram: pixel counts for each packed ID
    # ----------------------------------------------------------------
    ids, counts = np.unique(packed_seg, return_counts=True)
    frac = counts / packed_seg.size
    large_ids = ids[frac >= area_thresh]

    # ----------------------------------------------------------------
    # 4. Translate back to prim paths (ignore IDs absent from map)
    # ----------------------------------------------------------------
    return [packed_map[i] for i in large_ids if i in packed_map]


def reciprocal_barrier(d_vec, thr):
    """
    <<Creates a virtual repulsion that is strong when the position vector is short and disappears when its length is above the threshold.
    Inputs
        d_vec : vector from the robot to the obstacle
        thr   : distance threshold where the repulsion turns off>>
    """
    _EPS = 1e-9
    dist = np.linalg.norm(d_vec) + _EPS
    gate = 1.0 / (1.0 + np.exp(10.0 * (dist - thr)))
    return gate / dist


def quad_norm(x, Q):
    """
    <<Penalises deviations proportionally to their squared magnitude for smooth tracking.
    Inputs
        x : state-error vector (n×1)
        Q : symmetric positive-semidefinite weight (n×n)>>
    """
    return 0.5 * x.T @ (Q @ x)


def compile_odom_objects(env, visible_objects, env_id: int = -1):
    odom = {}

    if env_id == -1:
        for visible_object in visible_objects:
            state = env.unwrapped.scene.state["rigid_object"][visible_object][
                "root_pose"
            ]
            pos = state[:, :3]
            quat = state[:, 3:7]
            odom[visible_object] = [pos, quat, visible_objects[visible_object]]
        state = env.unwrapped.scene.state["articulation"]["robot"]["root_pose"]
        pos = state[:, :3]
        quat = state[:, 3:7]
        odom["agent"] = [pos, quat, "robot"]
    else:
        for visible_object in visible_objects:
            state = (
                env.unwrapped.scene.state["rigid_object"][visible_object]["root_pose"]
                .cpu()
                .numpy()[env_id]
                .flatten()
            )
            pos = state[:3]
            quat = state[3:7]
            odom[visible_object] = [pos, quat, visible_objects[visible_object]]

        state = (
            env.unwrapped.scene.state["articulation"]["robot"]["root_pose"]
            .cpu()
            .numpy()[env_id]
            .flatten()
        )
        pos = state[:3]
        quat = state[3:7]
        odom["agent"] = [pos, quat, "robot"]
    return odom


def get_env_local_pose(
    env_pos: torch.Tensor, xformable: UsdGeom.Xformable, device: torch.device
):
    """Compute pose in env-local coordinates"""
    world_transform = xformable.ComputeLocalToWorldTransform(0)
    world_pos = world_transform.ExtractTranslation()
    world_quat = world_transform.ExtractRotationQuat()

    px = world_pos[0] - env_pos[0]
    py = world_pos[1] - env_pos[1]
    pz = world_pos[2] - env_pos[2]
    qx = world_quat.imaginary[0]
    qy = world_quat.imaginary[1]
    qz = world_quat.imaginary[2]
    qw = world_quat.real

    return torch.tensor([px, py, pz], device=device), torch.tensor(
        [qw, qx, qy, qz], device=device
    )


def extract_odom_objects(env, include_objects, env_id: int = -1):
    # when not looking at the camera, we only extract the objects we care about
    odom = {}

    scene_state = env.unwrapped.scene.state
    rigid_state = scene_state["rigid_object"]
    art_state = scene_state["articulation"]

    # If env_id is given, only slice that env (keeps shape [1, ...] for consistency)
    if env_id == -1:
        env_slice = slice(None)
    else:
        env_slice = slice(env_id, env_id + 1)

    # Iterate over names + metadata directly, avoid repeated dict lookups
    for name, meta in include_objects.items():
        # Look up in rigid or articulation maps once
        if name in rigid_state:
            root_pose = rigid_state[name]["root_pose"][env_slice]
        elif name in art_state:
            root_pose = art_state[name]["root_pose"][env_slice]
        else:
            # Not found anywhere, skip
            continue

        pos = root_pose[..., :3]
        quat = root_pose[..., 3:7]

        odom[name] = [pos, quat, name, meta]

    return odom


def compile_frame_sensors(env, odom, allowed_names, env_id: int = -1):
    if env_id == -1:
        for s in env.unwrapped.scene.sensors:
            if "frame" in s:
                frame_name = env.unwrapped.scene.sensors[s].data.target_frame_names[0]
                if frame_name in allowed_names:
                    pos = env.unwrapped.scene.sensors[s].data.target_pos_w.squeeze(1)
                    quat = env.unwrapped.scene.sensors[s].data.target_quat_w.squeeze(1)
                    odom[frame_name] = [
                        pos,
                        quat,
                        env.unwrapped.scene.sensors[s].cfg.prim_path,
                    ]

    return odom


def extract_frame_sensors(env, odom, allowed_names, env_id: int = -1):
    scene = env.unwrapped.scene
    sensors = scene.sensors
    robot = scene.articulations["robot"]

    all_envs = env_id == -1

    # --- Agent pose (robot base) ---
    if all_envs:
        agent_pos = robot.data.root_pos_w  # (num_envs, 3)
        agent_quat = robot.data.root_quat_w  # (num_envs, 4)
    else:
        agent_pos = robot.data.root_pos_w[env_id]  # (3,)
        agent_quat = robot.data.root_quat_w[env_id]  # (4,)

    # --- Sensor poses ---
    # Iterate over allowed_names to avoid scanning every sensor each call
    for name, meta in allowed_names.items():
        sensor = sensors.get(name)
        if sensor is None:
            # allowed in config but not present in scene
            continue

        data = sensor.data
        if all_envs:
            # world-frame target pose for all envs
            pos = data.target_pos_w.squeeze(1)
            quat = data.target_quat_w.squeeze(1)
        else:
            # per-env source-frame pose, as in your original code
            pos = data.target_pos_source[env_id].squeeze(1)
            quat = data.target_quat_source[env_id].squeeze(1)

        odom[name] = [pos, quat, name, meta]

    # --- Add agent ---
    odom["agent"] = [agent_pos, agent_quat, "robot", "robot"]

    return odom


def compile_odom_manipulators(env, visible_objects, env_id: int = -1):
    odom = compile_odom_objects(env, visible_objects, env_id)
    return odom


def extract_odom_manipulators(env, visible_objects, env_id: int = -1):
    odom = extract_odom_objects(env, visible_objects, env_id)
    return odom


def get_all_visible_objects_from_camera(scene_camera, allowed_names):
    id2labels = scene_camera.data.info["instance_segmentation_fast"]["idToLabels"]
    visible_objects = {
        get_object_id_from_prim_path(v): v
        for v in id2labels.values()
        if get_object_id_from_prim_path(v) in allowed_names
    }

    return visible_objects, id2labels


def update_meta_in_controller(program_runtime, cam_data, id2labels, visible_objects):
    program_runtime.callbackImage(cam_data["rgb"])
    program_runtime.callbackSegmentation(
        cam_data["instance_segmentation_fast"],
        flip_segmentation_ids(id2labels, visible_objects),
    )


def write_txt(file, content):
    with open(file, "w") as f:
        f.write(content)


FRANKA_PANDA_GRIPPER_MPC_CONFIG = {
    "gripper_opened": [0.04, 0.04],
    "gripper_closed": [0.0, 0.0],
    "gripper_max_force": 40.0,
    "robot_gripper_name": "franka_panda",
    "has_finger_contact_sensors": False,
    "object_grasp_debug": False,
    "object_grasp_debug_every": 25,
}


def apply_franka_panda_gripper_config(mpc_config: dict) -> dict:
    """Apply Panda hand grasp settings used by the learned Franka Panda envs."""
    mpc_config.update(FRANKA_PANDA_GRIPPER_MPC_CONFIG)
    return mpc_config


def _duplicate_kinova_drive_gripper_state(joint_state: dict) -> dict:
    agent_state = joint_state["agent"]
    for key in ("joint_position", "joint_velocity", "joint_effort"):
        values = agent_state.get(key)
        if values is None or values.shape[-1] < 9:
            continue
        values = values.clone()
        drive_value = values[:, -2:-1]
        if key == "joint_effort":
            drive_value = drive_value.abs()
        values[:, -2:] = drive_value.repeat(1, 2)
        agent_state[key] = values
    return joint_state


def combine_joint_states(env, robot: str):
    if robot == "panda" or robot == "franka":
        joint_state = {}
        effort = torch.concat(
            [
                env.unwrapped.scene.articulations["robot"]
                .actuators["panda_shoulder"]
                .computed_effort,
                env.unwrapped.scene.articulations["robot"]
                .actuators["panda_forearm"]
                .computed_effort,
                torch.abs(env.unwrapped.scene.articulations["robot"]
                .actuators["panda_hand"]
                .computed_effort),
            ],
            axis=1,
        )
        joint_state["agent"] = {
            "joint_position": env.unwrapped.scene.state["articulation"]["robot"][
                "joint_position"
            ],
            "joint_velocity": env.unwrapped.scene.state["articulation"]["robot"][
                "joint_velocity"
            ],
            "joint_effort": effort,
        }


    if robot == "jetbot":
        joint_state = {}
        effort = (
            env.unwrapped.scene.articulations["robot"]
            .actuators["wheel_acts"]
            .computed_effort
        )
        joint_state["agent"] = {
            "joint_position": env.unwrapped.scene.state["articulation"]["robot"][
                "joint_position"
            ],
            "joint_velocity": env.unwrapped.scene.state["articulation"]["robot"][
                "joint_velocity"
            ],
            "joint_effort": effort,
        }


    if robot == "fr3":
        joint_state = {}
        # Read contact sensor forces (net normal contact force in world frame)
        left_contact_force = env.unwrapped.scene.sensors["contact_sensor_left"].data.net_forces_w   # (num_envs, 1, 3)
        right_contact_force = env.unwrapped.scene.sensors["contact_sensor_right"].data.net_forces_w  # (num_envs, 1, 3)
        hand_contact_force = env.unwrapped.scene.sensors["contact_sensor_hand"].data.net_forces_w    # (num_envs, 1, 3)
        left_mag = torch.norm(left_contact_force, dim=-1)  # (num_envs,)
        right_mag = torch.norm(right_contact_force, dim=-1)
        hand_mag = torch.norm(hand_contact_force, dim=-1)

        effort = torch.concat(
            [
                env.unwrapped.scene.articulations["robot"]
                .actuators["fr3_shoulder"]
                .computed_effort,
                env.unwrapped.scene.articulations["robot"]
                .actuators["fr3_forearm"]
                .computed_effort,
                left_mag,right_mag
            ],
            axis=1,
        )
        joint_state["agent"] = {
            "joint_position": env.unwrapped.scene.state["articulation"]["robot"][
                "joint_position"
            ][:,:9],
            "joint_velocity": env.unwrapped.scene.state["articulation"]["robot"][
                "joint_velocity"
            ][:,:9],
            "joint_effort": effort,
        }

    if robot in ("ur5", "ur5_robotiq"):
        return env.unwrapped._combine_kinova_joint_states()

    if robot == "kinova":
        if hasattr(env.unwrapped, "_combine_kinova_joint_states"):
            return _duplicate_kinova_drive_gripper_state(
                env.unwrapped._combine_kinova_joint_states()
            )

        joint_state = {}
        hand_effort = torch.abs(
            env.unwrapped.scene.articulations["robot"]
            .actuators["kinova_hand"]
            .computed_effort
        )
        if hand_effort.shape[1] == 1:
            hand_effort = hand_effort.repeat(1, 2)
        effort = torch.concat(
            [
                env.unwrapped.scene.articulations["robot"]
                .actuators["kinova_shoulder"]
                .computed_effort,
                env.unwrapped.scene.articulations["robot"]
                .actuators["kinova_forearm"]
                .computed_effort,
                hand_effort,
            ],
            axis=1,
        )
        joint_state["agent"] = {
            "joint_position": env.unwrapped.scene.state["articulation"]["robot"]["joint_position"][:, :9],
            "joint_velocity": env.unwrapped.scene.state["articulation"]["robot"]["joint_velocity"][:, :9],
            "joint_effort": effort,
        }
        joint_state = _duplicate_kinova_drive_gripper_state(joint_state)

    return joint_state


def create_video_writers(env, env_cfg, video_save_dir, camera_names, sun_program_executor_envs, depth_clip_m=2.0):
    """Create VideoWriter objects for recording RGB and depth from camera sensors.

    Returns:
        tuple: (video_writers dict, depth_clip_m float)
    """
    os.makedirs(video_save_dir, exist_ok=True)
    sim_dt = float(getattr(getattr(env_cfg, "sim", None), "dt", 1.0 / 60.0))
    decimation = int(getattr(env_cfg, "decimation", getattr(getattr(env_cfg, "sim", None), "decimation", 1)))
    denom = sim_dt * max(decimation, 1)
    video_fps = max(1, int(round(1.0 / denom))) if denom > 0.0 else 30

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH; required for MP4 recording.")

    class _FFmpegWriter:
        """Stream raw BGR frames to ffmpeg -> H.264 MP4 (faststart)."""

        def __init__(self, path: str, fps: int, width: int, height: int):
            self.path = path
            self._proc = subprocess.Popen(
                [
                    ffmpeg,
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "bgr24",
                    "-s:v",
                    f"{width}x{height}",
                    "-r",
                    str(fps),
                    "-i",
                    "-",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-crf",
                    "18",
                    "-preset",
                    "veryfast",
                    "-movflags",
                    "+faststart",
                    path,
                ],
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

        def write(self, frame_bgr: np.ndarray) -> None:
            if self._proc.stdin is None:
                raise RuntimeError("ffmpeg stdin is closed.")
            self._proc.stdin.write(frame_bgr.tobytes())

        def release(self) -> None:
            if self._proc.stdin is not None:
                try:
                    self._proc.stdin.close()
                except Exception:
                    pass
            rc = self._proc.wait()
            if rc != 0:
                err = b""
                if self._proc.stderr is not None:
                    try:
                        err = self._proc.stderr.read()
                    except Exception:
                        err = b""
                msg = err.decode("utf-8", errors="replace")[-2000:]
                raise RuntimeError(f"ffmpeg failed (exit={rc}) while writing {self.path}:\n{msg}")

    video_writers: dict[str, _FFmpegWriter] = {}
    for cam_name in camera_names:
        cam_sensor = env.unwrapped.scene.sensors[cam_name]
        if not hasattr(cam_sensor, "image_shape"):
            raise AttributeError(
                f"Sensor '{cam_name}' is not an image-producing camera (got {type(cam_sensor).__name__}); "
                "expected a camera sensor with `image_shape` and `data.output['rgb']`."
            )
        cam_h, cam_w = cam_sensor.image_shape
        for eid in sun_program_executor_envs:
            rgb_key = f"{cam_name}_env{eid}_rgb"
            rgb_path = os.path.join(video_save_dir, f"{rgb_key}.mp4")
            video_writers[rgb_key] = _FFmpegWriter(rgb_path, video_fps, cam_w, cam_h)

            depth_key = f"{cam_name}_env{eid}_depth"
            depth_path = os.path.join(video_save_dir, f"{depth_key}.mp4")
            video_writers[depth_key] = _FFmpegWriter(depth_path, video_fps, cam_w, cam_h)

            print(f"[Video] Recording {cam_name} env {eid} → rgb + depth")
    return video_writers, depth_clip_m


def write_camera_frames_to_video(env, video_writers, camera_names, sun_program_executor_envs, depth_clip_m):
    """Write one frame of RGB and depth per camera per env to the VideoWriters."""
    for cam_name in camera_names:
        cam_sensor = env.unwrapped.scene.sensors[cam_name]
        rgb = cam_sensor.data.output["rgb"]
        depth = cam_sensor.data.output["distance_to_image_plane"]
        for eid in sun_program_executor_envs:
            frame = rgb[eid].detach().cpu().numpy()
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
            # RGB -> BGR for ffmpeg's bgr24
            frame_bgr = np.ascontiguousarray(frame[..., ::-1])
            video_writers[f"{cam_name}_env{eid}_rgb"].write(frame_bgr)

            d = depth[eid, :, :, 0].detach().cpu().numpy()
            d_clipped = np.clip(d, 0.0, depth_clip_m)
            d_norm = (d_clipped / depth_clip_m * 255).astype(np.uint8)
            d_bgr = np.ascontiguousarray(np.repeat(d_norm[:, :, None], 3, axis=2))
            video_writers[f"{cam_name}_env{eid}_depth"].write(d_bgr)


def release_video_writers(video_writers, video_save_dir):
    """Release all VideoWriter objects and print summary."""
    for key, writer in video_writers.items():
        writer.release()
    print(f"[Video] Saved {len(video_writers)} videos to {video_save_dir}")


def depth_maps_to_pointclouds(
    env,
    camera_names,
    env_ids,
    *,
    stride=2,
    depth_min=0.01,
    depth_max=4.0,
    merge_cameras=True,
    keep_invalid=False,
    return_rgb=True,
):
    """Convert depth maps (and optionally RGB) from multiple cameras into world-frame point clouds.

    Uses camera intrinsics and poses from the environment's camera sensors. Depth is
    ``distance_to_image_plane`` (orthogonal depth). ``unproject_depth`` produces points
    in the pinhole/CV camera frame (X-right, Y-down, Z-forward) which matches the ROS
    convention, so we use ``pos_w`` and ``quat_w_ros`` to transform to world.

    When ``stride > 1`` the intrinsic matrix is scaled so that the sub-sampled pixel
    indices still map to the correct original pixel coordinates.

    Args:
        env: Gymnasium environment with scene.sensors containing the cameras.
        camera_names: List of camera sensor names (e.g. ["table_cam_0", "table_cam_1"]).
        env_ids: List of environment indices to compute point clouds for.
        stride: Sample every Nth pixel in height and width to reduce points. Default 2.
        depth_min: Minimum valid depth in meters. Default 0.01.
        depth_max: Maximum valid depth in meters. Default 100.0.
        merge_cameras: If True, merge all cameras into one point cloud per env. If False,
            return a dict per env mapping camera_name -> (N, 3) array.
        keep_invalid: If True, keep NaN/inf points; otherwise drop them. Default False.
        return_rgb: If True, sample RGB from camera output and return (pointclouds, colors).
            Default True. Requires "rgb" in camera data_types.

    Returns:
        If return_rgb is False: pointclouds only (dict as below).
        If return_rgb is True: (pointclouds, colors) where colors[env_id] is (N, 3) uint8 RGB.
        Pointclouds: if merge_cameras, dict[int, np.ndarray] (N, 3); else dict[int, dict[str, np.ndarray]].
    """
    scene = env.unwrapped.scene
    device = scene.sensors[camera_names[0]].data.pos_w.device
    out_merged = {} if merge_cameras else {}
    out_per_cam = {} if not merge_cameras else None
    colors_merged = {} if return_rgb else None
    colors_per_cam = {} if return_rgb and not merge_cameras else None
    has_rgb = return_rgb and "rgb" in scene.sensors[camera_names[0]].data.output

    for eid in env_ids:
        if merge_cameras:
            all_pts = []
            all_rgb = [] if has_rgb else None
        else:
            out_per_cam[eid] = {}
            colors_per_cam[eid] = {} if has_rgb else None

        for cam_name in camera_names:
            sensor = scene.sensors[cam_name]
            data = sensor.data
            depth_full = data.output["distance_to_image_plane"][eid]  # (H, W, 1)
            rgb_full = data.output["rgb"][eid] if has_rgb else None  # (H, W, 3)
            K = data.intrinsic_matrices[eid].clone()  # (3, 3)
            pos_w = data.pos_w[eid]  # (3,)
            quat_w = data.quat_w_ros[eid]  # (4,) w,x,y,z

            if stride > 1:
                depth = depth_full[::stride, ::stride, :]  # (H', W', 1)
                rgb = rgb_full[::stride, ::stride, :] if rgb_full is not None else None  # (H', W', 3)
                K[0, 0] /= stride
                K[1, 1] /= stride
                K[0, 2] /= stride
                K[1, 2] /= stride
            else:
                depth = depth_full
                rgb = rgb_full

            pts_cam = unproject_depth(depth, K.unsqueeze(0), is_ortho=True)
            if pts_cam.dim() == 3:
                pts_cam = pts_cam.squeeze(0)

            # Same flatten order as unproject_depth (depth.T then row-major)
            d_flat = depth.squeeze(-1).T.reshape(-1)
            valid = torch.isfinite(d_flat) & (d_flat >= depth_min) & (d_flat <= depth_max)
            if not keep_invalid:
                pts_cam = pts_cam[valid]

            if rgb is not None:
                # unproject_depth flattens depth as (W,H) row-major, so point i -> (u,v)=(i//H, i%H)
                # rgb is (H, W, 3); we need rgb_flat[i] = rgb[v, u, :] = rgb[i%H, i//H, :]
                H_, W_ = rgb.shape[0], rgb.shape[1]
                idx = torch.arange(H_ * W_, device=rgb.device, dtype=torch.long)
                rgb_flat = rgb[idx % H_, idx // H_, :]  # (H'*W', 3)
                if not keep_invalid:
                    rgb_flat = rgb_flat[valid]
                rgb_np = rgb_flat.cpu().numpy().astype(np.uint8)

            pts_world = transform_points(pts_cam, pos_w.unsqueeze(0), quat_w.unsqueeze(0))
            if pts_world.dim() == 3:
                pts_world = pts_world.squeeze(0)
            pts_np = pts_world.detach().cpu().numpy().astype(np.float32)

            if merge_cameras:
                all_pts.append(pts_np)
                if has_rgb and rgb is not None:
                    all_rgb.append(rgb_np)
            else:
                out_per_cam[eid][cam_name] = pts_np
                if has_rgb and rgb is not None:
                    colors_per_cam[eid][cam_name] = rgb_np

        if merge_cameras:
            out_merged[eid] = np.concatenate(all_pts, axis=0) if all_pts else np.zeros((0, 3), dtype=np.float32)
            if has_rgb and all_rgb:
                colors_merged[eid] = np.concatenate(all_rgb, axis=0)
            elif has_rgb:
                colors_merged[eid] = np.zeros((0, 3), dtype=np.uint8)

    if not return_rgb:
        return out_merged if merge_cameras else out_per_cam
    return (out_merged if merge_cameras else out_per_cam, colors_merged if merge_cameras else colors_per_cam)


def depth_maps_to_pointclouds_every_n_frames(
    env,
    camera_names,
    env_ids,
    step_index,
    every_n_frames,
    **kwargs,
):
    """Compute point clouds from depth maps only every N frames.

    Convenience wrapper around :func:`depth_maps_to_pointclouds`. Call this each step;
    when ``step_index % every_n_frames == 0`` it computes and returns point clouds,
    otherwise returns None (so you can skip heavy work most steps).

    Args:
        env: Gymnasium environment.
        camera_names: List of camera sensor names.
        env_ids: List of environment indices.
        step_index: Current simulation step index (e.g. ``count`` in the main loop).
        every_n_frames: Compute point clouds when step_index % every_n_frames == 0.
        **kwargs: Passed through to :func:`depth_maps_to_pointclouds` (e.g. stride, depth_min, merge_cameras).

    Returns:
        Result of :func:`depth_maps_to_pointclouds` when step_index % every_n_frames == 0, else None.
    """
    if step_index % every_n_frames != 0:
        return None
    return depth_maps_to_pointclouds(env, camera_names, env_ids, **kwargs)


def save_pointcloud_ply(points: np.ndarray, path: str, colors: np.ndarray | None = None) -> None:
    """Save a point cloud to an ASCII PLY file, optionally with vertex colors.

    Args:
        points: (N, 3) array of x, y, z coordinates (float32 or float64).
        path: Output file path (e.g. .ply).
        colors: Optional (N, 3) array of RGB in [0, 255], uint8. If provided, PLY includes red/green/blue.
    """
    n = points.shape[0]
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex %d\n" % n)
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            line = "%.6f %.6f %.6f" % (float(points[i, 0]), float(points[i, 1]), float(points[i, 2]))
            if colors is not None:
                line += " %d %d %d" % (int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2]))
            f.write(line + "\n")


def merge_meshes_under(
    src_root: str, dst_mesh_path: str, deactivate_sources: bool = False
) -> str:
    """
    Merge all UsdGeom.Mesh prims under `src_root` into a single mesh at `dst_mesh_path`.
    - Flattens transforms into points (world-space).
    - Keeps only basic geometry (points / faces). UVs/material subsets are not preserved.
    - Good for RayCaster (needs a single static mesh).
    Returns the created mesh path (dst_mesh_path).
    """
    stage = omni.usd.get_context().get_stage()
    src_prim = stage.GetPrimAtPath(src_root)
    if not src_prim:
        raise RuntimeError(f"merge_meshes_under: missing source root {src_root}")

    # Ensure parent exists
    parent_path = Sdf.Path(dst_mesh_path).GetParentPath()
    if not stage.GetPrimAtPath(parent_path):
        stage.DefinePrim(parent_path, "Xform")

    # If mesh already exists, delete/redefine it
    if stage.GetPrimAtPath(dst_mesh_path):
        stage.RemovePrim(dst_mesh_path)
    merged = UsdGeom.Mesh.Define(stage, dst_mesh_path)

    all_points = []
    all_counts = []
    all_indices = []
    idx_offset = 0

    # Walk all meshes under src_root
    for prim in stage.Traverse(src_prim):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)

        loc_points = mesh.GetPointsAttr().Get()
        loc_counts = mesh.GetFaceVertexCountsAttr().Get()
        loc_indices = mesh.GetFaceVertexIndicesAttr().Get()
        if not loc_points or not loc_counts or not loc_indices:
            continue

        # Bring points to world space
        xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        tr_points = [Gf.Vec3f(xf.Transform(Gf.Vec3d(p))) for p in loc_points]

        all_points.extend(tr_points)
        all_counts.extend(loc_counts)
        all_indices.extend([i + idx_offset for i in loc_indices])
        idx_offset += len(loc_points)

    if not all_points or not all_counts or not all_indices:
        raise RuntimeError(f"merge_meshes_under: no triangles found under {src_root}")

    merged.CreatePointsAttr(all_points)
    merged.CreateFaceVertexCountsAttr(all_counts)
    merged.CreateFaceVertexIndicesAttr(all_indices)

    # Set extent (AABB)
    mn = Gf.Vec3f(all_points[0])
    mx = Gf.Vec3f(all_points[0])
    for p in all_points[1:]:
        if p[0] < mn[0]:
            mn[0] = p[0]
        if p[1] < mn[1]:
            mn[1] = p[1]
        if p[2] < mn[2]:
            mn[2] = p[2]
        if p[0] > mx[0]:
            mx[0] = p[0]
        if p[1] > mx[1]:
            mx[1] = p[1]
        if p[2] > mx[2]:
            mx[2] = p[2]
    merged.CreateExtentAttr([mn, mx])

    # Ensure identity xform on merged
    UsdGeom.XformCommonAPI(merged).SetTranslate((0, 0, 0))
    UsdGeom.XformCommonAPI(merged).SetRotate((0, 0, 0))
    UsdGeom.XformCommonAPI(merged).SetScale((1, 1, 1))

    # Optionally deactivate the original hierarchy
    if deactivate_sources:
        src_prim.SetActive(False)

    return dst_mesh_path


def create_differential_ik_pose_function(
    env,
    env_cfg,
    robot_name: str,
):
    """Create an inverse kinematics function for desired end-effector pose in robot-base coordinates.

    Args:
        env: The gymnasium environment (unwrapped scene must contain "robot").
        env_cfg: Environment config (used for sim.device).

    Returns:
        A function (p_des, quat_des, q_init=None, frame_name="tool") -> joint_pos_des (cpu tensor).
    """
    if robot_name in ("ur5", "ur5_robotiq"):
        import pinocchio as pin
        from kuafu.mpc.controller.bounded_ik import BoundedPoseIK
        return BoundedPoseIK(
            pin.buildModelFromUrdf(env_cfg.mpc_config["urdf_path"]),
            frame_name="wrist_3_link", time_step=env_cfg.sim.dt * env_cfg.decimation,
        )
    if robot_name in ("panda", "franka"):
        joint_names = ["panda_joint.*"]
        body_names = ["panda_hand"]
    elif robot_name == "fr3":
        joint_names = ["fr3.*"]
        body_names = ["fr3_link7"]
    elif robot_name == "kinova":
        joint_names = ["joint_.*"]
        body_names = ["bracelet_link"]
    else:
        raise ValueError(f"Invalid robot name, only panda/franka, fr3, and kinova are supported")

    robot = env.unwrapped.scene["robot"]  # type: ignore[attr-defined]
    diff_ik_cfg = DifferentialIKControllerCfg(
        command_type="pose", use_relative_mode=False, ik_method="dls"
    )
    diff_ik_controller = DifferentialIKController(
        diff_ik_cfg,
        num_envs=env.unwrapped.scene.num_envs,  # type: ignore[attr-defined]
        device=env_cfg.sim.device,
    )
    robot_entity_cfg = SceneEntityCfg(
        "robot", joint_names=joint_names, body_names=body_names
    )
    robot_entity_cfg.resolve(env.unwrapped.scene)  # type: ignore[attr-defined]
    ee_body_id = robot_entity_cfg.body_ids[0]
    ee_jacobi_idx = ee_body_id - 1

    def inverse_kinematics(p_des, quat_des, q_init=None, frame_name="tool"):
        """Compute joint positions to reach desired end-effector pose in robot-base coordinates."""
        diff_ik_controller.set_command(
            torch.concatenate((torch.Tensor(p_des), torch.Tensor(quat_des)), dim=-1)
        )
        jacobian = robot.root_physx_view.get_jacobians()[
            :, ee_jacobi_idx, :, robot_entity_cfg.joint_ids
        ]
        ee_pose_w = robot.data.body_pose_w[:, ee_body_id]
        root_pose_w = robot.data.root_pose_w
        joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3],
            root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3],
            ee_pose_w[:, 3:7],
        )
        joint_pos_des = diff_ik_controller.compute(
            ee_pos_b, ee_quat_b, jacobian, joint_pos
        )
        return joint_pos_des.detach().cpu()

    return inverse_kinematics


def update_sun_program_executor_data_for_env(env, program_executor: SUNProgramExecutor, robot_name: str):
    """Update a single SUNProgramExecutor instance with odom + joint state for one Isaac env."""
    detectable_objects = env.unwrapped.cfg.detectable_objects
    odom = extract_odom_manipulators(env, detectable_objects)
    sensors = extract_frame_sensors(env, {}, detectable_objects)
    program_executor.callbackOdom(odom, sensors)
    program_executor.callbackJointState(
        combine_joint_states(env, robot=robot_name)
    )
    env_mpc_config = getattr(env.unwrapped.cfg, "mpc_config", {}) or {}
    has_finger_contact_sensors = env_mpc_config.get(
        "has_finger_contact_sensors", True
    )
    if has_finger_contact_sensors:
        scene_sensors = env.unwrapped.scene.sensors
        contact_forces = {}
        for key in ("contact_sensor_left", "contact_sensor_right"):
            if key in scene_sensors:
                contact_forces[key] = scene_sensors[key].data.net_forces_w[:, 0, :].detach()
        if contact_forces:
            program_executor.callbackContactForces(contact_forces)
