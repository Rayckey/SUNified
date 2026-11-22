import argparse
import json
import math
import sys
import os
import time

from kuafu.paths import resolve_artifact_path

from isaaclab.app import AppLauncher

DEFAULT_POINT_CLOUD_NUM_POINTS = 92160
PANDA_POINT_CLOUD_WORKSPACE_MIN = (0.15, -0.6, -0.05)
PANDA_POINT_CLOUD_WORKSPACE_MAX = (0.8, 0.2, 0.5)
PANDA_POINT_CLOUD_TABLE_NORMAL = (
    -0.014077403631341417,
    0.0046943986465211585,
    0.999889888601914,
)
PANDA_POINT_CLOUD_TABLE_D = 0.040399580987183284
PANDA_POINT_CLOUD_TABLE_CLEARANCE_M = 0.015
PANDA_OFFICIAL_CAMERA_NAMES = ("d415_0", "d415_1", "d415_2")
KINOVA_OFFICIAL_CAMERA_PAIR = ("d415_1", "d415_2")


def _parse_optional_float(value: str) -> float | None:
    if value.lower() in {"none", "null"}:
        return None
    return float(value)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


parser = argparse.ArgumentParser(description="Collect point-cloud demos using a learned RL checkpoint.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="FR3-FinRay-Table-Drawer-Long-Long-U", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rl_games_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .pth checkpoint file.")
parser.add_argument(
    "--checkpoint_params_dir",
    type=str,
    default=None,
    help=(
        "Optional explicit checkpoint playback params directory containing agent.yaml and env.yaml. "
        "When set, do not discover params through archive metadata."
    ),
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="External shared output root supplied by the collection launcher.",
)
parser.add_argument(
    "--prior_checkpoint",
    type=str,
    default=None,
    help="Path to BC prior checkpoint (required for residual_rl playback).",
)
parser.add_argument(
    "--prior_mode",
    type=str,
    choices=["full_actor", "residual_rl"],
    default="full_actor",
    help="How to execute checkpoint policy. Use residual_rl for Kuafu residual runs.",
)
parser.add_argument(
    "--weights_params_json",
    type=str,
    default=None,
    help="Optional Kuafu weights/parameters JSON (same format as best.json or plain stages dict).",
)
parser.add_argument("--seed", type=int, default=-1, help="Seed used for the environment.")
parser.add_argument("--worker_id", type=int, default=0, help="Worker ID for parallel collection (controls output subdir).")
parser.add_argument("--num_demos", type=int, default=10, help="Number of successful demos to collect.")
parser.add_argument(
    "--state_only", action="store_true",
    help="Record aligned full-state/action HDF5 demonstrations without camera rendering.",
)
parser.add_argument(
    "--rgb_transitions", action="store_true",
    help="Record synchronized 128x72 RGB from both Kinova cameras and applied joint commands.",
)
parser.add_argument(
    "--pointcloud_transitions", action="store_true",
    help="Also record synchronized depth-derived XYZ point clouds for a visual world model.",
)
parser.add_argument(
    "--stochastic_policy",
    action="store_true",
    default=False,
    help="Sample policy actions stochastically (default is deterministic playback).",
)
parser.add_argument(
    "--camera_keys",
    type=str,
    nargs="+",
    default=None,
    metavar="SENSOR_KEY",
    help="Ordered simulated-camera sensor allowlist used for RGB/depth capture and point-cloud fusion.",
)
parser.add_argument(
    "--use_tiled_cameras",
    action="store_true",
    default=False,
    help="Use TiledCamera sensors instead of standard Camera sensors for collection.",
)
parser.add_argument(
    "--stride",
    type=int,
    default=1,
    help="Only store every N-th simulation step (default: 1, store every step).",
)
parser.add_argument(
    "--saver_chunk_size",
    type=int,
    default=int(os.environ.get("KUAFU_SAVER_CHUNK_SIZE", "0")),
    help="Flush DemonstrationRecorder every N stored frames to bound camera/pointcloud memory. "
    "Use <=0 to flush only on episode done (default).",
)
parser.add_argument(
    "--eager_point_cloud",
    action="store_true",
    default=_env_flag("KUAFU_EAGER_POINT_CLOUD", False),
    help="Fuse and downsample each saved camera frame immediately, buffering compact point clouds "
    "instead of raw RGB/depth frames.",
)
parser.add_argument(
    "--skip_point_cloud_write",
    action="store_true",
    default=not _env_flag("KUAFU_WRITE_POINT_CLOUD", True),
    help="Do not write point_cloud, point_cloud_flat, or point_cloud_count datasets to HDF5.",
)
parser.add_argument(
    "--save_rgb",
    action="store_true",
    default=_env_flag("KUAFU_WRITE_RGB", False),
    help="Persist camera RGB frames as uint8 HDF5 dataset /rgb with shape (T,C,H,W,3).",
)
parser.add_argument(
    "--point_cloud_num_points",
    type=int,
    default=DEFAULT_POINT_CLOUD_NUM_POINTS,
    help="Cap stored points per frame via uniform subsample of fused depth/RGB. "
    f"Default: {DEFAULT_POINT_CLOUD_NUM_POINTS}. Use <=0 to keep all valid fused pixels.",
)
parser.add_argument(
    "--disable_point_cloud_workspace_crop",
    action="store_true",
    default=False,
    help="Disable all saver.py WORKSPACE_LIMITS point-cloud cropping before sampling.",
)
parser.add_argument(
    "--point_cloud_workspace_min",
    type=_parse_optional_float,
    nargs=3,
    metavar=("X_MIN", "Y_MIN", "Z_MIN"),
    default=None,
    help="Optional workspace crop min bounds. Use none for an unbounded axis.",
)
parser.add_argument(
    "--point_cloud_workspace_max",
    type=_parse_optional_float,
    nargs=3,
    metavar=("X_MAX", "Y_MAX", "Z_MAX"),
    default=None,
    help="Optional workspace crop max bounds. Use none for an unbounded axis.",
)
parser.add_argument(
    "--point_cloud_exclude_box_min",
    type=_parse_optional_float,
    nargs=3,
    metavar=("X_MIN", "Y_MIN", "Z_MIN"),
    default=None,
    help="Optional XYZ min bounds for a box to exclude after any workspace crop. Use none for an unbounded axis.",
)
parser.add_argument(
    "--point_cloud_exclude_box_max",
    type=_parse_optional_float,
    nargs=3,
    metavar=("X_MAX", "Y_MAX", "Z_MAX"),
    default=None,
    help="Optional XYZ max bounds for a box to exclude after any workspace crop. Use none for an unbounded axis.",
)
parser.add_argument(
    "--point_cloud_exclude_cylinder_center_xy",
    type=float,
    nargs=2,
    metavar=("CENTER_X", "CENTER_Y"),
    default=None,
    help="Optional XY center for a vertical cylinder to exclude after any workspace crop.",
)
parser.add_argument(
    "--point_cloud_exclude_cylinder_radius",
    type=float,
    default=None,
    help="Optional radius for the vertical point-cloud exclusion cylinder.",
)
parser.add_argument(
    "--point_cloud_exclude_cylinder_z_min",
    type=_parse_optional_float,
    default=None,
    help="Optional lower z bound for the exclusion cylinder. Omit for all z.",
)
parser.add_argument(
    "--point_cloud_exclude_cylinder_z_max",
    type=_parse_optional_float,
    default=None,
    help="Optional upper z bound for the exclusion cylinder. Omit for all z.",
)
parser.add_argument(
    "--point_cloud_min_depth_m",
    type=float,
    default=None,
    help="Optional camera-depth lower bound in meters before point-cloud fusion.",
)
parser.add_argument(
    "--point_cloud_max_depth_m",
    type=float,
    default=None,
    help="Optional camera-depth upper bound in meters before point-cloud fusion.",
)
parser.add_argument(
    "--disable_point_cloud_table_plane_filter",
    action="store_true",
    default=False,
    help="Disable the Panda/Beatrix table-plane point-cloud filter.",
)
parser.add_argument(
    "--camera_warmup_frames_after_reset",
    type=int,
    default=0,
    help="Render this many frames after reset/auto-reset before the next saved camera frame.",
)
parser.add_argument(
    "--disable_left_right_randomization",
    action="store_true",
    default=False,
    help="Use the task's deterministic swap=False object-side assignment for this collection run.",
)
parser.add_argument(
    "--force_left_right_swap",
    action="store_true",
    default=False,
    help="Use deterministic swap=True object-side assignment for this collection run.",
)
parser.add_argument(
    "--hide_table_visuals",
    action="store_true",
    default=False,
    help="Hide table render geometry while keeping table/floor collision behavior intact.",
)
parser.add_argument(
    "--skip_dense_reward_computation",
    action="store_true",
    default=False,
    help="Return zero env rewards during collection while keeping SUNProgramEvaluator stage evaluation enabled.",
)
parser.add_argument(
    "--dry_run",
    action="store_true",
    default=False,
    help="Print the resolved checkpoint collector setup and exit before launching Isaac.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
for _artifact_attr, _artifact_label in (
    ("checkpoint", "Collection policy checkpoint"),
    ("checkpoint_params_dir", "Checkpoint playback parameters"),
    ("prior_checkpoint", "Collection BC prior checkpoint"),
    ("weights_params_json", "Collection residual parameters"),
):
    _artifact_value = getattr(args_cli, _artifact_attr, None)
    if _artifact_value:
        setattr(args_cli, _artifact_attr, str(resolve_artifact_path(_artifact_value, _artifact_label)))

if args_cli.dry_run:
    print(
        "[dry-run] checkpoint pointcloud collector "
        f"task={args_cli.task} num_envs={args_cli.num_envs} "
        f"checkpoint={args_cli.checkpoint} prior_mode={args_cli.prior_mode} "
        f"device={args_cli.device} camera_keys={args_cli.camera_keys}"
    )
    raise SystemExit(0)

# State-only collection uses the same checkpoint and physics without camera sensors.
if args_cli.pointcloud_transitions:
    args_cli.rgb_transitions = True
args_cli.enable_cameras = not args_cli.state_only
if args_cli.state_only and args_cli.rgb_transitions:
    raise ValueError("--state_only and --rgb_transitions are mutually exclusive")
if args_cli.rgb_transitions and args_cli.stride != 1:
    raise ValueError("--rgb_transitions requires --stride 1 for action alignment")
if args_cli.state_only and args_cli.stride != 1:
    raise ValueError("--state_only requires --stride 1 to retain transition alignment")

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

# Runtime patch for omni.syntheticdata-0.6.13 bug: dep_attrib_data.get() returns a
# malformed dtype for freshly-created UINT64 state attributes, and generic Python
# / NumPy writes can still fail in some Isaac Sim builds. Normalize reads and
# write back the explicit USD `Vt.UInt64Array` type. Patched here (not in site-packages).
def _patch_syntheticdata():
    import numpy as np
    from pxr import Vt
    from omni.syntheticdata.scripts import SyntheticData as _sdm
    import omni.graph.core as og
    import omni.graph.tools.ogn

    def _coerce_uint64_list(value):
        if value is None:
            return []
        if isinstance(value, np.ndarray):
            if value.dtype.itemsize == 0:
                return []
            value = value.reshape(-1).tolist()
        elif isinstance(value, (list, tuple, set)):
            value = list(value)
        else:
            try:
                value = list(value)
            except TypeError:
                value = [value]
        return [int(item) for item in value]

    def _to_uint64_array(value):
        return Vt.UInt64Array(_coerce_uint64_list(value))

    @staticmethod
    def _fixed(node, downstream_node_handle):
        if (not node.is_valid()) or (downstream_node_handle is None):
            return 0
        attr_name = "state:_sdp_intergraph_downstream_node_handles_"
        if not node.get_attribute_exists(attr_name):
            dep_attrib = og.Controller.create_attribute(
                node=node,
                attr_name=attr_name,
                attr_type=og.Type(og.BaseDataType.UINT64, 1, 1),
                attr_port=og.AttributePortType.ATTRIBUTE_PORT_TYPE_STATE,
                attr_extended_type=og.ExtendedAttributeType.REGULAR,
            )
            if dep_attrib is None or not node.get_attribute_exists(attr_name):
                raise _sdm.SyntheticDataException(
                    f"failed to create node dependency for {node.get_prim_path()} @ {downstream_node_handle}."
                )
            dep_attrib.set_metadata(omni.graph.tools.ogn.MetadataKeys.INTERNAL, "1")
            dep_attrib.set_metadata(omni.graph.tools.ogn.MetadataKeys.LITERAL_ONLY, "1")
        else:
            dep_attrib = node.get_attribute(attr_name)
        existing_handles = _coerce_uint64_list(dep_attrib.get())
        existing_handles.append(int(downstream_node_handle))
        dep_attrib.set(_to_uint64_array(existing_handles))
        return len(existing_handles)

    _sdm.SyntheticData._add_node_downstream_intergraph_dependency = _fixed

    @staticmethod
    def _fixed_remove(node, active_node_handles):
        if not node.is_valid():
            return 0
        attr_name = "state:_sdp_intergraph_downstream_node_handles_"
        if not node.get_attribute_exists(attr_name):
            return 0
        dep_attrib = node.get_attribute(attr_name)
        active_handles = set(_coerce_uint64_list(active_node_handles))
        dep_data = [handle for handle in _coerce_uint64_list(dep_attrib.get()) if handle in active_handles]
        dep_attrib.set(_to_uint64_array(dep_data))
        return len(dep_data)

    _sdm.SyntheticData._remove_inactive_node_downstream_intergraph_dependencies = _fixed_remove

if args_cli.use_tiled_cameras:
    _patch_syntheticdata()

import kuafu.isaac.learned  # noqa: F401
from kuafu.isaac.learned import kinova, panda  # noqa: F401

import datetime
import re
from pathlib import Path

import gymnasium as gym
import torch

from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
try:
    from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
except ImportError:
    from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401

from kuafu.rl.isaac.update_utilities import update_all_parameters, update_weights

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

from kuafu.isaac.learned.saver import DemonstrationRecorder
from kuafu.data.collection.checkpoint_options import (
    PANDA_CAMERA_KEYS as PANDA_OFFICIAL_CAMERA_KEYS,
    checkpoint_params_directory,
    find_checkpoint_params,
    collection_camera_keys as select_collection_camera_keys,
    configure_object_sides,
    limit_completed_successes,
)

script_dir = Path(__file__).resolve().parents[4]
WORKER_ID = args_cli.worker_id
NUM_DEMOS = args_cli.num_demos


# ---------------------------------------------------------------------------
# Helpers copied from play.py
# ---------------------------------------------------------------------------

def _set_env_obs_mode(env_cfg, obs_mode: str) -> str:
    set_obs_mode = getattr(env_cfg, "set_obs_mode", None)
    if callable(set_obs_mode):
        set_obs_mode(obs_mode)
    else:
        setattr(env_cfg, "obs_mode", obs_mode)
        obs_dims = getattr(env_cfg, "obs_dims", None)
        if isinstance(obs_dims, dict) and obs_mode in obs_dims and hasattr(env_cfg, "observation_space"):
            setattr(env_cfg, "observation_space", int(obs_dims[obs_mode]))
    return obs_mode


def _find_checkpoint_params_dir(checkpoint_path: str) -> Path | None:
    return find_checkpoint_params(retrieve_file_path(checkpoint_path), script_dir)


def _load_checkpoint_agent_cfg(params_dir: Path | None) -> dict | None:
    if params_dir is None:
        return None
    agent_yaml_path = params_dir / "agent.yaml"
    if not agent_yaml_path.exists():
        return None
    import yaml

    with agent_yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError(f"checkpoint agent.yaml must load to a dict: {agent_yaml_path}")
    return data


def _parse_yaml_scalar_value(raw: str):
    value = raw.split("#", 1)[0].strip()
    if value == "":
        return None
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in {"null", "none"}:
        return None
    try:
        if re.fullmatch(r"[-+]?\d+", value):
            return int(value)
        if re.fullmatch(r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", value):
            return float(value)
    except ValueError:
        pass
    return value.strip('"\'')


def _read_simple_env_yaml_settings(params_dir: Path | None) -> dict[str, object]:
    if params_dir is None:
        return {}
    env_yaml_path = params_dir / "env.yaml"
    try:
        lines = env_yaml_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}

    top_level_keys = {
        "obs_mode",
        "observation_space",
        "action_space",
        "decimation",
        "episode_length_s",
        "action_scale",
        "ee_action_scale",
        "action_mode",
        "use_absolute_progress_rewards",
        "use_relative_progress_rewards",
        "use_time_penalty",
        "use_velocity_penalty",
        "regularize_velocity_scale",
        "regularize_time_scale",
        "progress_rewards_scale",
        "stage_transition_bonus_scale",
        "stage_transition_bonus_use_remaining_horizon",
        "penalize_false_completions",
        "sun_program_evaluator_stage_override",
        "ignore_stage_gate_for_success",
        "use_external_rewards",
        "external_rewards_scale",
    }
    mpc_keys = {
        "use_per_env_weight",
        "use_absolute_progress_rewards",
        "use_relative_progress_rewards",
        "use_velocity_penalty",
        "use_time_penalty",
        "regularize_velocity_scale",
        "regularize_time_scale",
        "progress_rewards_scale",
        "stage_transition_bonus_scale",
        "stage_transition_bonus_use_remaining_horizon",
        "penalize_false_completions",
        "ignore_stage_gate_for_success",
        "use_external_rewards",
        "external_rewards_scale",
        "reward_print_every",
        "episode_outcome_print_every",
        "collect_terminal_stuck_constraint_stats",
        "robot_gripper_name",
        "finger_contact_force_mode",
        "object_grasp_debug",
        "object_grasp_debug_every",
        "ee_frame",
        "gripper_max_force",
    }

    settings: dict[str, object] = {}
    mpc_settings: dict[str, object] = {}
    in_mpc = False
    for line in lines:
        if line.startswith("mpc_config:"):
            in_mpc = True
            continue
        if in_mpc and line and not line.startswith(" "):
            in_mpc = False
        if in_mpc:
            match = re.match(r"^  ([A-Za-z0-9_]+):\s*(.*)$", line)
            if match and match.group(1) in mpc_keys:
                parsed = _parse_yaml_scalar_value(match.group(2))
                if parsed is not None:
                    mpc_settings[match.group(1)] = parsed
            continue

        match = re.match(r"^([A-Za-z0-9_]+):\s*(.*)$", line)
        if match and match.group(1) in top_level_keys:
            parsed = _parse_yaml_scalar_value(match.group(2))
            if parsed is not None:
                settings[match.group(1)] = parsed
    if mpc_settings:
        settings["mpc_config"] = mpc_settings
    return settings


def _apply_checkpoint_env_settings(env_cfg, params_dir: Path | None) -> dict[str, object]:
    settings = _read_simple_env_yaml_settings(params_dir)
    if not settings:
        return {}
    applied: dict[str, object] = {}
    obs_mode = settings.get("obs_mode")
    if isinstance(obs_mode, str):
        applied["obs_mode"] = _set_env_obs_mode(env_cfg, obs_mode)

    for key, value in settings.items():
        if key in {"obs_mode", "mpc_config"}:
            continue
        if hasattr(env_cfg, key):
            setattr(env_cfg, key, value)
            applied[key] = value
    mpc_settings = settings.get("mpc_config")
    if isinstance(mpc_settings, dict) and isinstance(getattr(env_cfg, "mpc_config", None), dict):
        mpc_attr_keys = {
            "use_absolute_progress_rewards",
            "use_relative_progress_rewards",
            "use_velocity_penalty",
            "use_time_penalty",
            "regularize_velocity_scale",
            "regularize_time_scale",
            "progress_rewards_scale",
            "stage_transition_bonus_scale",
            "stage_transition_bonus_use_remaining_horizon",
            "penalize_false_completions",
            "ignore_stage_gate_for_success",
            "use_external_rewards",
            "external_rewards_scale",
            "reward_print_every",
            "episode_outcome_print_every",
            "collect_terminal_stuck_constraint_stats",
        }
        for key, value in mpc_settings.items():
            env_cfg.mpc_config[key] = value
            if key in mpc_attr_keys and hasattr(env_cfg, key):
                setattr(env_cfg, key, value)
        applied["mpc_config"] = dict(mpc_settings)
    return applied


def _apply_checkpoint_obs_mode(env_cfg, checkpoint_path: str) -> str | None:
    params_dir = _find_checkpoint_params_dir(checkpoint_path)
    settings = _read_simple_env_yaml_settings(params_dir)
    obs_mode = settings.get("obs_mode")
    if not isinstance(obs_mode, str):
        return None
    return _set_env_obs_mode(env_cfg, obs_mode)


def _load_all_stages_weights_params(path: str):
    resolved_path = retrieve_file_path(path)
    with open(resolved_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {resolved_path}, got {type(data).__name__}")
    stages_data = data.get("stages") if isinstance(data.get("stages"), dict) else data
    if not isinstance(stages_data, dict):
        raise ValueError(f"Expected a stages dict in {resolved_path}")
    valid_stage_ids = []
    for k, v in stages_data.items():
        if not isinstance(v, dict):
            continue
        try:
            valid_stage_ids.append(int(k))
        except (TypeError, ValueError):
            continue
    if not valid_stage_ids:
        raise ValueError(f"No valid stage entries found in {resolved_path}.")
    all_stages_weights = {}
    all_stages_parameters = {}
    for sid in sorted(valid_stage_ids):
        stage_data = stages_data[str(sid)]
        w_raw = stage_data.get("weights") or {}
        p_raw = stage_data.get("parameters") or {}
        if not isinstance(w_raw, dict) or not w_raw:
            raise ValueError(f"Stage {sid} in {resolved_path} has no valid 'weights' dict.")
        if not isinstance(p_raw, dict):
            raise ValueError(f"Stage {sid} in {resolved_path} has invalid 'parameters'.")
        all_stages_weights[sid] = {int(k): float(v) for k, v in dict(w_raw).items()}
        all_stages_parameters[sid] = {int(k): dict(v) for k, v in dict(p_raw).items()}
    return all_stages_weights, all_stages_parameters


RESIDUAL_AGENT_HPARAM_KEYS = {
    "clip_actions",
    "residual_bounds",
    "clip_composed_actions",
    "alpha_final",
    "alpha_warmup_epochs",
}


def _load_residual_agent_hyperparameters(path: str) -> dict[str, object]:
    resolved_path = retrieve_file_path(path)
    with open(resolved_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {resolved_path}, got {type(data).__name__}")
    raw = data.get("agent_hyperparameters")
    if raw is None and isinstance(data.get("best_summary_global"), dict):
        raw = data["best_summary_global"].get("agent_hyperparameters")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"Expected agent_hyperparameters dict in {resolved_path}")
    unknown = set(raw) - RESIDUAL_AGENT_HPARAM_KEYS
    if unknown:
        raise ValueError(f"Unsupported residual agent hyperparameters in {resolved_path}: {sorted(unknown)}")
    return dict(raw)


def _finite_float(value, *, name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return parsed


def _nonnegative_int(value, *, name: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be nonnegative, got {value!r}.")
    return parsed


def _bool_like(value, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be boolean-like, got {value!r}.")


def _apply_residual_agent_hyperparameters(
    agent_cfg: dict, overrides: dict[str, object]
) -> dict[str, object]:
    if not overrides:
        return {}
    params = agent_cfg.setdefault("params", {})
    if not isinstance(params, dict):
        raise TypeError("agent_cfg[params] must be a dict")
    env_params = params.setdefault("env", {})
    if not isinstance(env_params, dict):
        raise TypeError("agent_cfg[params][env] must be a dict")
    config = params.setdefault("config", {})
    if not isinstance(config, dict):
        raise TypeError("agent_cfg[params][config] must be a dict")
    residual_rl = config.setdefault("residual_rl", {})
    if not isinstance(residual_rl, dict):
        raise TypeError("agent_cfg[params][config][residual_rl] must be a dict")

    applied: dict[str, object] = {}
    if "clip_actions" in overrides:
        applied["clip_actions"] = _finite_float(overrides["clip_actions"], name="clip_actions")
        env_params["clip_actions"] = applied["clip_actions"]
    if "alpha_final" in overrides:
        applied["alpha_final"] = _finite_float(overrides["alpha_final"], name="alpha_final")
        residual_rl["alpha_final"] = applied["alpha_final"]
    if "alpha_warmup_epochs" in overrides:
        applied["alpha_warmup_epochs"] = _nonnegative_int(
            overrides["alpha_warmup_epochs"], name="alpha_warmup_epochs"
        )
        residual_rl["alpha_warmup_epochs"] = applied["alpha_warmup_epochs"]
    if "clip_composed_actions" in overrides:
        applied["clip_composed_actions"] = _bool_like(
            overrides["clip_composed_actions"], name="clip_composed_actions"
        )
        residual_rl["clip_composed_actions"] = applied["clip_composed_actions"]
    if "residual_bounds" in overrides:
        raw_bounds = overrides["residual_bounds"]
        if not isinstance(raw_bounds, (list, tuple)):
            raise ValueError("residual_bounds must be a list")
        bounds = [_finite_float(v, name=f"residual_bounds[{i}]") for i, v in enumerate(raw_bounds)]
        applied["residual_bounds"] = bounds
        residual_rl["residual_bounds"] = bounds
    return applied


def _resolve_prior_mode(agent_cfg: dict, explicit_mode: str) -> str:
    mode = str(explicit_mode or "full_actor").strip() or "full_actor"
    residual_cfg = agent_cfg.get("params", {}).get("config", {}).get("residual_rl", {})
    if mode == "full_actor" and isinstance(residual_cfg, dict) and residual_cfg.get("enabled", False):
        return "residual_rl"
    return mode


def _is_panda_task(task_name: str) -> bool:
    return str(task_name).split(":")[-1].startswith("Panda-Table-")


def _is_kinova_task_or_cfg(task_name: str, env_cfg) -> bool:
    haystacks = [
        str(task_name),
        type(env_cfg).__name__,
        type(env_cfg).__module__,
    ]
    for attr_name in ("robot", "scene"):
        attr_value = getattr(env_cfg, attr_name, None)
        if attr_value is not None:
            haystacks.extend((type(attr_value).__name__, type(attr_value).__module__))
    mpc_config = getattr(env_cfg, "mpc_config", None)
    if isinstance(mpc_config, dict):
        for key in ("robot", "robot_name", "robot_type", "ee_frame"):
            value = mpc_config.get(key)
            if value is not None:
                haystacks.append(str(value))
    return any(("kinova" in value.lower() or "gen3" in value.lower()) for value in haystacks)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Collect point-cloud demos using a trained RL checkpoint."""
    task_name = args_cli.task.split(":")[-1]

    resume_path = retrieve_file_path(args_cli.checkpoint)
    if args_cli.checkpoint_params_dir is not None:
        checkpoint_params_dir = checkpoint_params_directory(args_cli.checkpoint_params_dir)
    else:
        checkpoint_params_dir = _find_checkpoint_params_dir(resume_path)
    if checkpoint_params_dir is not None:
        print(f"[INFO] Loading checkpoint playback params from: {checkpoint_params_dir}")
        checkpoint_agent_cfg = _load_checkpoint_agent_cfg(checkpoint_params_dir)
        if checkpoint_agent_cfg is not None:
            agent_cfg = checkpoint_agent_cfg
            print(f"[INFO] Loaded checkpoint agent config: {checkpoint_params_dir / 'agent.yaml'}")
        applied_env_settings = _apply_checkpoint_env_settings(env_cfg, checkpoint_params_dir)
        if applied_env_settings:
            visible_settings = {k: v for k, v in applied_env_settings.items() if k != "mpc_config"}
            print(f"[INFO] Applied checkpoint env settings: {json.dumps(visible_settings, sort_keys=True)}")
    else:
        print(f"[WARN] Could not locate checkpoint params directory for: {resume_path}")

    # --- env config overrides ---
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.mpc_config["num_envs"] = env_cfg.scene.num_envs
    env_cfg.mpc_config["use_per_env_weight"] = False
    env_cfg.enable_cameras = not args_cli.state_only
    if args_cli.rgb_transitions:
        from kuafu.data.collection.rgb_recorder import configure_rgb_cameras
        configure_rgb_cameras(env_cfg)
    if args_cli.pointcloud_transitions:
        from kuafu.data.collection.pointcloud_recorder import configure_pointcloud_cameras
        configure_pointcloud_cameras(env_cfg)
    if hasattr(env_cfg, "use_tiled_cameras"):
        env_cfg.use_tiled_cameras = args_cli.use_tiled_cameras
    if hasattr(env_cfg, "hide_table_visuals"):
        env_cfg.hide_table_visuals = bool(args_cli.hide_table_visuals)
    if hasattr(env_cfg, "skip_dense_reward_computation"):
        env_cfg.skip_dense_reward_computation = bool(args_cli.skip_dense_reward_computation)
    elif args_cli.skip_dense_reward_computation:
        print("[WARN] Env config does not support skip_dense_reward_computation; dense rewards may still run.")
    configure_object_sides(
        env_cfg,
        disable_randomization=args_cli.disable_left_right_randomization,
        force_swap=args_cli.force_left_right_swap,
    )
    if args_cli.force_left_right_swap or args_cli.disable_left_right_randomization:
        print(f"[INFO] Left/right object assignment fixed at swap={env_cfg.fixed_left_right_swap}.")

    if "mpc" in agent_cfg:
        env_cfg.mpc_config["use_per_env_weight"] = agent_cfg["mpc"]["use_per_env_weight"]
        env_cfg.mpc_config["use_absolute_progress_rewards"] = agent_cfg["mpc"]["use_absolute_progress_rewards"]
        env_cfg.mpc_config["use_relative_progress_rewards"] = agent_cfg["mpc"]["use_relative_progress_rewards"]
        env_cfg.mpc_config["use_velocity_penalty"] = agent_cfg["mpc"]["use_velocity_penalty"]
        env_cfg.mpc_config["regularize_velocity_scale"] = agent_cfg["mpc"]["regularize_velocity_scale"]
        env_cfg.mpc_config["regularize_time_scale"] = agent_cfg["mpc"]["regularize_time_scale"]
        env_cfg.mpc_config["progress_rewards_scale"] = agent_cfg["mpc"]["progress_rewards_scale"]

    if args_cli.device is not None:
        agent_cfg["params"]["config"]["device"] = args_cli.device
        agent_cfg["params"]["config"]["device_name"] = args_cli.device

    if args_cli.seed == -1:
        # Default to a runtime-derived seed when user does not provide one.
        args_cli.seed = int(time.time_ns() % (2**31 - 1))

    agent_cfg["params"]["seed"] = args_cli.seed
    env_cfg.seed = args_cli.seed

    # --- output directory ---
    project_root = Path(script_dir).resolve()
    configured = os.environ.get("KUAFU_ARTIFACT_ROOT", "").strip()
    external_root = (
        Path(configured).expanduser().resolve()
        if configured
        else project_root.parent / "KuafuArtifacts"
    )
    if args_cli.output_dir:
        output_root = Path(args_cli.output_dir).expanduser()
        if not output_root.is_absolute():
            output_root = external_root / output_root
        output_root = output_root.resolve()
        dataset_dir = output_root / "workers" / f"worker_{WORKER_ID:02d}" / "datasets"
    else:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dataset_dir = (
            external_root
            / "standalone"
            / "checkpoint_collection"
            / task_name
            / f"{timestamp}_worker{WORKER_ID:02d}"
            / "datasets"
        )
    dataset_dir = dataset_dir.resolve()
    if dataset_dir == project_root or project_root in dataset_dir.parents:
        raise ValueError(
            f"Trajectory output must be outside the source project ({project_root}); got {dataset_dir}"
        )
    os.makedirs(dataset_dir, exist_ok=True)

    # --- checkpoint ---
    log_dir = os.path.dirname(os.path.dirname(resume_path))
    env_cfg.log_dir = log_dir

    checkpoint_obs_mode = _apply_checkpoint_obs_mode(env_cfg, resume_path)
    if checkpoint_obs_mode is not None:
        print(f"[INFO] Set env obs_mode from checkpoint metadata: {checkpoint_obs_mode}")
    print(
        "[INFO] Camera backend for collection: "
        + ("tiled" if getattr(env_cfg, "use_tiled_cameras", True) else "standard")
    )
    requested_camera_keys = tuple(args_cli.camera_keys) if args_cli.camera_keys else None
    is_panda_task = _is_panda_task(task_name)
    is_kinova_task = _is_kinova_task_or_cfg(task_name, env_cfg)
    collection_camera_keys = select_collection_camera_keys(
        requested_camera_keys, is_panda=is_panda_task, is_kinova=is_kinova_task,
    )
    if is_panda_task:
        print(
            "[INFO] Panda collection cameras (3 simulated calibrated views): "
            + ", ".join(
                f"{camera_key}={camera_name}"
                for camera_key, camera_name in zip(
                    PANDA_OFFICIAL_CAMERA_KEYS, PANDA_OFFICIAL_CAMERA_NAMES, strict=True
                )
            )
        )
    elif requested_camera_keys is not None:
        print(
            f"[INFO] Explicit collection camera allowlist ({len(collection_camera_keys)}): "
            + ", ".join(collection_camera_keys)
        )
    elif is_kinova_task:
        print(
            "[INFO] Kinova collection cameras: "
            f"{collection_camera_keys[0]}={KINOVA_OFFICIAL_CAMERA_PAIR[0]}, "
            f"{collection_camera_keys[1]}={KINOVA_OFFICIAL_CAMERA_PAIR[1]}; "
            "third camera disabled for this collection path."
        )
    else:
        print("[INFO] Collection cameras: auto-discovering registered camera sensors.")
    print(f"[INFO] Table visuals hidden: {bool(getattr(env_cfg, 'hide_table_visuals', False))}")
    if bool(getattr(env_cfg, "skip_dense_reward_computation", False)):
        print("[INFO] Environment rewards disabled; SUNProgramEvaluator stage evaluation remains enabled.")
    else:
        print("[INFO] Environment rewards enabled.")

    # --- rl-games wrapper setup ---
    prior_mode = _resolve_prior_mode(agent_cfg, args_cli.prior_mode)
    if prior_mode == "residual_rl" and args_cli.weights_params_json:
        residual_hparams = _load_residual_agent_hyperparameters(args_cli.weights_params_json)
        if residual_hparams:
            applied_hparams = _apply_residual_agent_hyperparameters(agent_cfg, residual_hparams)
            print(
                "[INFO] Applying residual agent hyperparameters from "
                f"{args_cli.weights_params_json}: {json.dumps(applied_hparams, sort_keys=True)}"
            )

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    # --- create environment ---
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.weights_params_json:
        all_stages_weights, all_stages_parameters = _load_all_stages_weights_params(args_cli.weights_params_json)
        print(f"[INFO] Applying weights/parameters from: {args_cli.weights_params_json}")
        update_all_parameters(env.unwrapped, all_stages_parameters=all_stages_parameters)
        update_weights(env.unwrapped, stage_id=0, weights=None, all_stages_weights=all_stages_weights)

    # --- DemonstrationRecorder ---
    raw_env = env.unwrapped
    saver_chunk_size = int(args_cli.saver_chunk_size)
    if saver_chunk_size > 0:
        print(f"[INFO] DemonstrationRecorder chunk flush: every {saver_chunk_size} stored frame(s)")
    if args_cli.eager_point_cloud:
        print("[INFO] DemonstrationRecorder eager point-cloud buffering enabled.")
    if args_cli.skip_point_cloud_write:
        print("[INFO] DemonstrationRecorder point-cloud HDF5 writes disabled.")
    if args_cli.save_rgb:
        print("[INFO] DemonstrationRecorder RGB HDF5 writes enabled.")

    use_panda_real_clip = _is_panda_task(args_cli.task)
    point_cloud_workspace_min = args_cli.point_cloud_workspace_min
    point_cloud_workspace_max = args_cli.point_cloud_workspace_max
    point_cloud_min_depth_m = args_cli.point_cloud_min_depth_m
    point_cloud_max_depth_m = args_cli.point_cloud_max_depth_m
    point_cloud_table_plane_normal = None
    point_cloud_table_plane_d = None
    point_cloud_table_clearance_m = 0.0
    if use_panda_real_clip:
        if not args_cli.disable_point_cloud_workspace_crop:
            point_cloud_workspace_min = point_cloud_workspace_min or PANDA_POINT_CLOUD_WORKSPACE_MIN
            point_cloud_workspace_max = point_cloud_workspace_max or PANDA_POINT_CLOUD_WORKSPACE_MAX
        if not args_cli.disable_point_cloud_table_plane_filter:
            point_cloud_table_plane_normal = PANDA_POINT_CLOUD_TABLE_NORMAL
            point_cloud_table_plane_d = PANDA_POINT_CLOUD_TABLE_D
            point_cloud_table_clearance_m = PANDA_POINT_CLOUD_TABLE_CLEARANCE_M
        depth_filter_label = (
            "disabled"
            if point_cloud_min_depth_m is None and point_cloud_max_depth_m is None
            else f"[{point_cloud_min_depth_m}, {point_cloud_max_depth_m}]"
        )
        print(
            "[INFO] Panda real-camera point-cloud clip: "
            f"depth={depth_filter_label}, "
            f"workspace_min={point_cloud_workspace_min if not args_cli.disable_point_cloud_workspace_crop else None}, "
            f"workspace_max={point_cloud_workspace_max if not args_cli.disable_point_cloud_workspace_crop else None}, "
            f"table_plane_filter={point_cloud_table_plane_normal is not None}"
        )

    saver = None if (args_cli.state_only or args_cli.rgb_transitions) else DemonstrationRecorder(
        base_dir=dataset_dir,
        fps=1.0 / (env_cfg.sim.dt * env_cfg.decimation * max(1, args_cli.stride)),
        record_stride=max(1, args_cli.stride),
        chunk_size=saver_chunk_size if saver_chunk_size > 0 else -1,
        save_states_only=False,
        point_cloud_num_points=args_cli.point_cloud_num_points,
        point_cloud_crop_to_workspace=not args_cli.disable_point_cloud_workspace_crop,
        point_cloud_crop_z_only=False,
        point_cloud_workspace_min=point_cloud_workspace_min,
        point_cloud_workspace_max=point_cloud_workspace_max,
        point_cloud_exclude_box_min=args_cli.point_cloud_exclude_box_min,
        point_cloud_exclude_box_max=args_cli.point_cloud_exclude_box_max,
        point_cloud_exclude_cylinder_center_xy=args_cli.point_cloud_exclude_cylinder_center_xy,
        point_cloud_exclude_cylinder_radius=args_cli.point_cloud_exclude_cylinder_radius,
        point_cloud_exclude_cylinder_z_min=args_cli.point_cloud_exclude_cylinder_z_min,
        point_cloud_exclude_cylinder_z_max=args_cli.point_cloud_exclude_cylinder_z_max,
        point_cloud_min_depth_m=point_cloud_min_depth_m,
        point_cloud_max_depth_m=point_cloud_max_depth_m,
        point_cloud_table_plane_normal=point_cloud_table_plane_normal,
        point_cloud_table_plane_d=point_cloud_table_plane_d,
        point_cloud_table_object_clearance_m=point_cloud_table_clearance_m,
        camera_buffer_device=os.environ.get("KUAFU_CAMERA_BUFFER_DEVICE", "cuda"),
        point_cloud_buffer_device=os.environ.get("KUAFU_POINT_CLOUD_BUFFER_DEVICE", "cpu"),
        camera_keys=collection_camera_keys,
        eager_point_cloud=args_cli.eager_point_cloud,
        write_rgb=args_cli.save_rgb,
        write_point_cloud=not args_cli.skip_point_cloud_write,
        write_point_cloud_flat=False,
    )
    if saver is not None:
        saver.task_description = getattr(env_cfg, "task_description", "")

    # --- wrap for rl-games ---
    residual_cfg = None
    residual_controller = None
    if prior_mode == "residual_rl":
        if not args_cli.prior_checkpoint:
            raise ValueError("Residual RL playback requires --prior_checkpoint.")
        from kuafu.rl.isaac.train_rl_games import (
            _configure_residual_rl_agent_cfg,
            _make_kuafu_residual_rl_games_wrapper,
            _make_residual_action_controller,
            _resolve_env_num_stages,
        )
        from kuafu.bc.training.residual_rl import ResidualRlConfig

        env_num_stages = _resolve_env_num_stages(env)
        residual_mapping = dict(agent_cfg.get("params", {}).get("config", {}).get("residual_rl", {}) or {})
        residual_mapping["enabled"] = True
        action_dim = int(getattr(env.unwrapped.single_action_space, "shape")[0])  # pyright: ignore[reportAttributeAccessIssue]
        residual_cfg = ResidualRlConfig.from_mapping(
            residual_mapping,
            action_dim=action_dim,
            env_num_stages=env_num_stages,
        )
        _configure_residual_rl_agent_cfg(agent_cfg, residual_cfg)
        env = _make_kuafu_residual_rl_games_wrapper(
            env,
            rl_device,
            clip_obs,
            clip_actions,
            residual_cfg,
            obs_groups,
            concate_obs_groups,
        )
        residual_controller = _make_residual_action_controller(
            prior_checkpoint=args_cli.prior_checkpoint,
            residual_cfg=residual_cfg,
            rl_env=env,
        )
        # Playback should use the post-warmup residual scale, not alpha_init.
        playback_epoch = int(residual_cfg.critic_warmup_epochs) + int(residual_cfg.alpha_warmup_epochs)
        residual_controller.set_epoch(playback_epoch)
        print(f"[INFO] Residual RL playback armed with prior: {args_cli.prior_checkpoint}")
        print(f"[INFO] Residual RL playback alpha set to: {residual_controller.alpha:.3f}")
    else:
        env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # --- load checkpoint ---
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    print(f"[INFO] Loading model checkpoint from: {resume_path}")

    # Flush any pending RTX renderer CUDA work before loading the checkpoint.
    # Without this, torch.load/load_state_dict can deadlock against the Isaac
    # Sim render loop which holds CUDA stream locks during camera initialization.
    for _ in range(5):
        simulation_app.update()
    torch.cuda.synchronize()

    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    # --- reset ---
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)
    agent_is_rnn = bool(getattr(agent, "is_rnn", False))
    if agent_is_rnn:
        agent.init_rnn()

    if args_cli.state_only or args_cli.rgb_transitions:
        from kuafu.data.collection.state_recorder import collect_state_episodes, StateTransitionRecorder
        from kuafu.data.collection.rgb_recorder import RGBTransitionRecorder
        from kuafu.data.collection.pointcloud_recorder import PointCloudTransitionRecorder
        report = collect_state_episodes(
            raw_env, env, agent, simulation_app,
            dataset_dir / ("rgb_demos.hdf5" if args_cli.rgb_transitions else "state_demos.hdf5"), NUM_DEMOS, obs,
            deterministic=not args_cli.stochastic_policy,
            recorder_class=(PointCloudTransitionRecorder if args_cli.pointcloud_transitions
                            else RGBTransitionRecorder if args_cli.rgb_transitions else StateTransitionRecorder),
            metadata={"env_name": task_name, "seed": args_cli.seed, "checkpoint": resume_path,
                      "prior_checkpoint": args_cli.prior_checkpoint, "prior_mode": prior_mode,
                      "control_dt": float(raw_env.step_dt), "action_mode": env_cfg.action_mode},
        )
        print(f"[INFO] {'RGB' if args_cli.rgb_transitions else 'State'} collection complete: {json.dumps(report)}", flush=True)
        env.close()
        return

    camera_warmup_frames = max(0, int(args_cli.camera_warmup_frames_after_reset))

    def _warm_camera_tensors(reason: str) -> None:
        if camera_warmup_frames <= 0:
            return
        print(f"[INFO] Refreshing camera tensors after {reason}: {camera_warmup_frames} render frame(s)")
        for _ in range(camera_warmup_frames):
            raw_env.sim.render()
            raw_env.scene.update(raw_env.physics_dt)

    _warm_camera_tensors("initial reset")

    print(f"\n[INFO] Collecting {NUM_DEMOS} demos into: {dataset_dir}\n")
    deterministic_actions = not args_cli.stochastic_policy
    print(f"[INFO] Action selection: {'deterministic' if deterministic_actions else 'stochastic'}")
    stride = max(1, int(args_cli.stride))
    if stride > 1:
        print(f"[INFO] Storing every {stride} simulation steps.")

    # --- main loop ---
    step_count = 0
    while simulation_app.is_running() and saver.metadata_info["num_episodes"] < NUM_DEMOS:
        # Match Kuafu eval: env.step() mutates env-owned tensors in-place, so
        # do not run the rollout under torch.inference_mode().
        with torch.no_grad():
            obs = agent.obs_to_torch(obs)
            actions = agent.get_action(obs, is_deterministic=deterministic_actions)

            # In residual_rl mode the env wrapper internally composes the
            # actual command sent to physics as ``clamp(bc_prior(obs) + alpha *
            # bound(residual))``. ``actions`` here is just the bare residual,
            # so saving it would record tiny noise around 0 instead of the
            # real per-step command. Mirror the wrapper's composition for the
            # stored action; pass the original ``actions`` to ``env.step`` so
            # the wrapper's own composition still runs identically.
            if residual_controller is not None:
                actions_to_store = residual_controller.compose_env_action(
                    raw_residual_action=actions.to(device=env._sim_device),
                    bc_obs=env._latest_policy_obs,
                    clip_actions=clip_actions,
                )
            else:
                actions_to_store = actions

            if step_count % stride == 0:
                saver.store(actions_to_store, raw_env)

            obs, _, dones, _ = env.step(actions)
            if isinstance(obs, dict):
                obs = obs["obs"]
            step_count += 1

            if len(dones) > 0:
                if isinstance(dones, torch.Tensor):
                    n_done = int(dones.sum().item())
                else:
                    n_done = int(sum(dones))

                if n_done > 0:
                    successes = getattr(raw_env, "_cached_success", None)
                    if successes is None:
                        successes = dones if isinstance(dones, torch.Tensor) else torch.tensor(dones)
                    remaining_demos = max(0, NUM_DEMOS - int(saver.metadata_info["num_episodes"]))
                    successes_for_saver, successful = limit_completed_successes(dones, successes, remaining_demos)
                    accepted = min(successful, remaining_demos)
                    num_saved = saver.write(
                        dones if isinstance(dones, torch.Tensor) else torch.tensor(dones),
                        successes_for_saver,
                    )
                    print(
                        f"[COLLECT] Episodes done: {n_done} | "
                        f"Successful: {successful} | "
                        f"Accepted: {accepted} | "
                        f"Total saved: {num_saved} / {NUM_DEMOS}"
                    )
                    _warm_camera_tensors("episode reset")

                    # reset RNN states for terminated envs
                    if agent_is_rnn and agent.states is not None:
                        for s in agent.states:
                            s[:, dones, :] = 0.0

    saver.stop()
    print(f"\n[INFO] Collection complete. Saved {saver.metadata_info['num_episodes']} demos to {dataset_dir}")
    env.close()


if __name__ == "__main__":
    main()  # pyright: ignore[reportCallIssue]
    simulation_app.close()
