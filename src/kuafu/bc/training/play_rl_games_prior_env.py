"""Debug companion to play_bc_env.py that rolls out the BC prior AFTER it has
been loaded into an RL-Games A2C agent (same object residual-RL's freeze period uses).

Pipeline mirrored from src/kuafu/rl/isaac/train_rl_games.py (prior-loading branch):

    1. gym.make(task, cfg=env_cfg)                   -- same env creation as play_bc_env.py
    2. RlGamesVecEnvWrapper(env, clip_obs, clip_actions)
    3. runner.algo_factory.create(...)               -- builds the rl_games A2C agent
    4. apply_bc_prior_to_rl_games_agent(agent, ckpt) -- copies BC weights into
           agent.model.a2c_network.{actor_mlp, mu}
           agent.model.running_mean_std  <- BC obs mean/std
    5. rollout loop: agent.model(input_dict)['mus'] -> env.step

Compared to play_bc_env.py (raw PyTorch BC model + Normalizer), this script
exposes exactly what changes when the BC policy is rehosted inside rl_games:
the RlGamesVecEnvWrapper action/obs clipping and the rl_games RunningMeanStd
(with its built-in [-5, 5] clamp). Running both on the same env should give the
same success rate; any gap between the two is a rollout-path bug.

Args:
    task:             Isaac Lab task id, e.g. "FR3-FinRay-Table-Pen-Cup-ff".
    checkpoint:       Path to BC .pth checkpoint (same file used as INITIAL_PRIOR_CHECKPOINT).
    duration_sec:     Simulation time (seconds) to run.
    num_envs:         Number of parallel envs.
    seed:             Random seed.
    params_path:      Optional best.json to overwrite env constraints (same as play_bc_env.py).
    disable_wrapper_clips:  If set, passes clip_obs=clip_actions=inf to the wrapper
                            (matches play_bc_env.py's no-wrapper rollout).
    disable_rms_clamp:      If set, monkey-patches RunningMeanStd.forward to drop
                            the hardcoded clamp(y, -5, 5) (matches BC's plain
                            (x-mean)/std normalization).
    stochastic:       Use sampled actions (Normal(mu, sigma)) instead of mu.
"""

import argparse
import sys
from pathlib import Path as _Path


from kuafu.rl.isaac.task_registry import register_task_modules, resolve_execution_robot_id  # noqa: E402
from kuafu.rl.isaac.update_utilities import update_weights, update_parameters  # noqa: E402

from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# CLI (AppLauncher must parse before any Isaac imports)
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Play a BC prior after it has been loaded into an RL-Games A2C model.",
)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--task", type=str, default="FR3-FinRay-Table-Pen-Cup-ff")
parser.add_argument(
    "--execution_robot_id",
    type=str,
    default=None,
    help="Robot task package to register. Defaults to inference from --task, then FR3FinRay-Franka.",
)
parser.add_argument("--dry_run", action="store_true", default=False, help="Resolve task registration and exit before launching Isaac.")
parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="Path to BC .pth checkpoint (the same file used as INITIAL_PRIOR_CHECKPOINT in residual-RL).",
)
parser.add_argument("--duration_sec", type=float, default=120.0)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument(
    "--params_path",
    type=str,
    default=None,
    help="Optional best.json (residual-RL format) to override env constraint parameters and residual weights.",
)
parser.add_argument(
    "--obs_stats_count",
    type=float,
    default=1e6,
    help="Running-mean-std count written into rl_games RMS after setting BC mean/std (1e6 = effectively frozen).",
)
parser.add_argument(
    "--disable_wrapper_clips",
    action="store_true",
    default=False,
    help="Override clip_obs and clip_actions in RlGamesVecEnvWrapper to +inf (matches play_bc_env.py).",
)
parser.add_argument(
    "--disable_rms_clamp",
    action="store_true",
    default=False,
    help="Patch rl_games RunningMeanStd.forward to drop the built-in clamp(y, -5, 5) (matches BC Normalizer).",
)
parser.add_argument(
    "--use_raw_obs_prior",
    action="store_true",
    default=False,
    help=(
        "Force normalize_input=False in the rl_games config before building the agent. "
        "This makes apply_bc_prior_to_rl_games_agent pick the model_state_dict_prior_raw_obs "
        "variant (BC's explicit (x-mean)/std transform in the first linear layer) and "
        "makes rl_games' running_mean_std a no-op. Mathematically equivalent to "
        "kuafu.bc.training.dataset.Normalizer -- avoids the sqrt(var + 1e-5) vs /clamp(std, 1e-8) "
        "discrepancy that rl_games' RunningMeanStd introduces for low-variance obs dims."
    ),
)
parser.add_argument(
    "--diag_compare_pipelines",
    action="store_true",
    default=False,
    help=(
        "Before rollout, load the plain BC pipeline (model_state_dict_norm + Normalizer) and "
        "compare its action output against the rl_games pipeline (agent.model -> mus) on the "
        "first env.reset() observation batch. Prints per-stage L2 / L_inf stats so we can "
        "tell whether the two pipelines agree mathematically or whether the bake/rl_games "
        "plumbing introduces a divergence before any environment dynamics."
    ),
)
parser.add_argument(
    "--stochastic",
    action="store_true",
    default=False,
    help="Sample actions from Normal(mu, sigma) instead of using mu deterministically.",
)
parser.add_argument(
    "--sigma_override",
    type=float,
    default=None,
    help=(
        "If set, overwrites the learned log-std (sigma parameter) to this constant value. "
        "residual-RL's freeze period uses -10. Only meaningful with --stochastic; with deterministic "
        "rollout sigma is ignored."
    ),
)
parser.add_argument(
    "--bypass_rl_wrapper",
    action="store_true",
    default=False,
    help=(
        "Drive the env the same way kuafu.bc.training.play_bc_env.py does: use env.unwrapped.step(action) "
        "and env.unwrapped.reset() directly, skipping RlGamesVecEnvWrapper and any gym wrappers "
        "stacked by gym.make. Keeps the rl_games agent.model for action computation. Diagnostic "
        "switch to check whether the wrapper/step path -- not the model -- is responsible for "
        "the success-rate gap vs play_bc_env.py."
    ),
)
parser.add_argument(
    "--freeze_input_rms",
    action="store_true",
    default=False,
    help=(
        "Flavor B: after apply_bc_prior_to_rl_games_agent seeds the observation running_mean_std "
        "with the BC expert obs stats, pin those stats so PPO's per-rollout updates never drift "
        "them toward the rollout distribution. Without this flag, rl_games' RunningMeanStd is "
        "refreshed on every forward pass (self.training=True), which at num_envs=8192 converges "
        "the normalizer to the live rollout distribution within a few hundred steps and silently "
        "pushes the frozen BC weights out-of-distribution. Equivalent in intent to "
        "--use_raw_obs_prior (flavor A: bake normalization into the first Linear and leave RMS as "
        "identity), but keeps normalize_input=True so the post-freeze PPO fine-tuning still gets "
        "normalized inputs via the (now fixed) expert stats."
    ),
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.dry_run:
    resolved_robot_id = resolve_execution_robot_id(args_cli.task, args_cli.execution_robot_id)
    print(
        "[dry-run] play_rl_games_prior_env.py "
        f"task={args_cli.task} execution_robot_id={resolved_robot_id} "
        f"num_envs={args_cli.num_envs} device={args_cli.device}"
    )
    raise SystemExit(0)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

resolved_execution_robot_id = register_task_modules(
    task=args_cli.task,
    execution_robot_id=args_cli.execution_robot_id,
)
print(f"[INFO] Registered Kuafu task package for execution_robot_id={resolved_execution_robot_id}", flush=True)

"""Rest of imports must come after AppLauncher."""

import math  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.common.algo_observer import IsaacAlgoObserver  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

from kuafu.bc.training.rl_games_prior import apply_bc_prior_to_rl_games_agent  # noqa: E402
from kuafu.bc.training.dataset import Normalizer  # noqa: E402
from kuafu.bc.training.model import MLPActorCriticPrior, MLPActorPrior  # noqa: E402


def _bc_load_checkpoint(ckpt_path: str, device: torch.device):
    """Inlined copy of kuafu.bc.training.play_bc_env.load_checkpoint.

    Cannot import play_bc_env directly because that module runs argparse + AppLauncher
    at import time, which steals this script's CLI args and double-launches Isaac Lab.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)

    ckpt = torch.load(ckpt_path, map_location="cpu")

    obs_norm = Normalizer.from_state_dict(ckpt["obs_normalizer"])
    act_norm = Normalizer.from_state_dict(ckpt["act_normalizer"])

    model_state_dict = ckpt["model_state_dicts"]["model_state_dict_norm"]
    uses_actor_critic_layout = any(
        key.startswith("actor_trunk.") or key.startswith("critic_trunk.") or key.startswith("value_layer.")
        for key in model_state_dict
    )
    model_cls = MLPActorCriticPrior if uses_actor_critic_layout else MLPActorPrior

    model = model_cls(
        obs_dim=ckpt["obs_dim"],
        act_dim=ckpt["act_dim"],
        hidden_dims=tuple(ckpt["hidden_dims"]),
        activation=str(ckpt.get("activation", "leaky_relu")),
        negative_slope=float(ckpt.get("negative_slope", 0.01)),
        dropout=float(ckpt.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(model_state_dict)
    model.eval()

    return model, obs_norm, act_norm, ckpt


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _register_rl_env(rl_env) -> None:
    """Register the wrapped Isaac Lab env with rl_games (idempotent-ish)."""
    try:
        vecenv.register(
            "IsaacRlgWrapper",
            lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
        )
    except Exception:
        pass
    try:
        env_configurations.register(
            "rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: rl_env}
        )
    except Exception:
        try:
            env_configurations.configurations["rlgpu"] = {
                "vecenv_type": "IsaacRlgWrapper",
                "env_creator": lambda **kwargs: rl_env,
            }
        except Exception:
            pass


def _patch_rms_no_clamp(rms) -> None:
    """Replace rms.forward with a copy that omits rl_games' clamp(y, -5, 5).

    This mimics kuafu.bc.training.dataset.Normalizer exactly for finite inputs.
    No NaN/Inf sanitization is applied here -- this script is a debugging tool
    and we *want* to see numerical blowups if they happen.
    """
    import types
    from rl_games.algos_torch import torch_ext

    def _unclamped_forward(self, input, denorm=False, mask=None):
        if self.training:
            if mask is not None:
                mean, var = torch_ext.get_mean_var_with_masks(input, mask)
            else:
                mean = input.mean(self.axis)
                var = input.var(self.axis)
            (self.running_mean, self.running_var, self.count) = self._update_mean_var_count_from_moments(
                self.running_mean, self.running_var, self.count, mean, var, input.size()[0]
            )

        if self.per_channel:
            if len(self.insize) == 3:
                current_mean = self.running_mean.view([1, self.insize[0], 1, 1]).expand_as(input)
                current_var = self.running_var.view([1, self.insize[0], 1, 1]).expand_as(input)
            elif len(self.insize) == 2:
                current_mean = self.running_mean.view([1, self.insize[0], 1]).expand_as(input)
                current_var = self.running_var.view([1, self.insize[0], 1]).expand_as(input)
            else:
                current_mean = self.running_mean.view([1, self.insize[0]]).expand_as(input)
                current_var = self.running_var.view([1, self.insize[0]]).expand_as(input)
        else:
            current_mean = self.running_mean
            current_var = self.running_var

        if denorm:
            y = torch.sqrt(current_var.float() + self.epsilon) * input + current_mean.float()
        else:
            if self.norm_only:
                y = input / torch.sqrt(current_var.float() + self.epsilon)
            else:
                y = (input - current_mean.float()) / torch.sqrt(current_var.float() + self.epsilon)
        return y

    rms.forward = types.MethodType(_unclamped_forward, rms)


def _silence_sigma_head(agent, sigma_override: float | None) -> None:
    """Replicate residual-RL freeze-period sigma handling.

    - fixed_sigma=True: overwrite the scalar `sigma` Parameter with sigma_override.
    - fixed_sigma=False: zero the sigma-head Linear's weight and set its bias to sigma_override.
    """
    import torch.nn as nn

    net = agent.model.a2c_network
    if sigma_override is None:
        return

    sigma_attr = getattr(net, "sigma", None)
    fixed_sigma = bool(getattr(net, "fixed_sigma", False))

    with torch.no_grad():
        if fixed_sigma and isinstance(sigma_attr, nn.Parameter):
            sigma_attr.data.fill_(float(sigma_override))
            print(f"[play_rl_prior] Fixed-sigma: set log-std to {sigma_override}", flush=True)
        elif isinstance(sigma_attr, nn.Linear):
            sigma_attr.weight.zero_()
            if sigma_attr.bias is not None:
                sigma_attr.bias.fill_(float(sigma_override))
            print(
                f"[play_rl_prior] Learned-sigma head: weight=0, bias={sigma_override}",
                flush=True,
            )
        else:
            print(
                f"[play_rl_prior] WARNING: unrecognized sigma attribute type ({type(sigma_attr).__name__}); "
                "ignoring --sigma_override.",
                flush=True,
            )


@torch.no_grad()
def _diag_compare_pipelines(agent, checkpoint_path: str, obs_tensor: torch.Tensor, device: str) -> None:
    """One-shot comparison: BC-Normalizer+BC-model vs rl_games agent.model on the same obs batch."""
    try:
        bc_model, obs_norm, act_norm, _ = _bc_load_checkpoint(checkpoint_path, torch.device(device))
        bc_model.eval()
    except Exception as exc:  # pragma: no cover - diagnostic path
        print(f"[diag] Failed to build BC reference pipeline: {exc!r}", flush=True)
        return

    raw_obs = obs_tensor.detach().to(device).float()
    print(
        f"[diag] raw_obs shape={tuple(raw_obs.shape)} "
        f"min={raw_obs.min().item():.4e} max={raw_obs.max().item():.4e} "
        f"mean={raw_obs.mean().item():.4e}",
        flush=True,
    )

    # --- BC pipeline (matches play_bc_env.py) ---
    bc_obs_in = obs_norm.normalize(raw_obs)
    bc_out = bc_model(bc_obs_in)
    bc_mu_norm = bc_out[0] if isinstance(bc_out, (tuple, list)) else bc_out
    bc_action_full = act_norm.denormalize(bc_mu_norm)  # (B, act_dim * chunk_size)

    # --- rl_games pipeline (matches the rollout below) ---
    input_dict = {"is_train": False, "prev_actions": None, "obs": raw_obs, "rnn_states": None}
    rl_out = agent.model(input_dict)
    rl_action = rl_out["mus"]  # (B, act_dim)

    # Align shapes: BC outputs the full chunk (e.g. 30 * 11 = 330), rl_games uses only the
    # first chunk (11). rl_games_prior.py bakes mu_weight[:rl_act_dim] / mu_bias[:rl_act_dim],
    # which is equivalent to taking action chunk #0 (play_bc_env.py's `act_full.view(B, cs, act_dim)[:, 0, :]`).
    rl_act_dim = int(rl_action.shape[-1])
    bc_total_dim = int(bc_action_full.shape[-1])
    if bc_total_dim == rl_act_dim:
        bc_action = bc_action_full
        chunk_size = 1
    elif bc_total_dim % rl_act_dim == 0:
        chunk_size = bc_total_dim // rl_act_dim
        bc_action = bc_action_full.reshape(bc_action_full.shape[0], chunk_size, rl_act_dim)[:, 0, :]
    else:
        print(
            f"[diag] BC/RL action dims incompatible: bc_total_dim={bc_total_dim} rl_act_dim={rl_act_dim}; "
            "aborting comparison.",
            flush=True,
        )
        return
    if chunk_size > 1:
        print(f"[diag] action chunking detected: chunk_size={chunk_size}, comparing chunk 0 only", flush=True)

    diff = (bc_action - rl_action).abs()
    print(
        "[diag] === action pipeline comparison (BC vs rl_games) ===\n"
        f"[diag] BC action:   min={bc_action.min().item():.4e} max={bc_action.max().item():.4e} "
        f"mean_abs={bc_action.abs().mean().item():.4e}\n"
        f"[diag] RL action:   min={rl_action.min().item():.4e} max={rl_action.max().item():.4e} "
        f"mean_abs={rl_action.abs().mean().item():.4e}\n"
        f"[diag] |BC - RL|:   max={diff.max().item():.4e} mean={diff.mean().item():.4e} "
        f"L2={diff.pow(2).sum().sqrt().item():.4e}\n"
        f"[diag] rel_err:     {(diff.mean() / (bc_action.abs().mean() + 1e-12)).item():.4e}",
        flush=True,
    )

    # also log per-dim max diff so we see if the divergence is localized
    per_dim_max = diff.reshape(-1, diff.shape[-1]).max(dim=0).values
    topk = min(8, per_dim_max.numel())
    worst = torch.topk(per_dim_max, topk)
    print(
        f"[diag] top-{topk} worst action dims (|BC-RL| max): "
        + ", ".join(f"dim{int(i)}={v.item():.4e}" for v, i in zip(worst.values, worst.indices)),
        flush=True,
    )

    # also compare the post-first-layer activation to check if obs normalization diverges
    try:
        bc_trunk = bc_model.actor_trunk if hasattr(bc_model, "actor_trunk") else None
        if bc_trunk is None and hasattr(bc_model, "trunk"):
            bc_trunk = bc_model.trunk
        if bc_trunk is not None and len(bc_trunk) > 0:
            bc_first = bc_trunk[0]
            bc_h0 = bc_first(bc_obs_in)
            a2c_net = agent.model.a2c_network
            actor_mlp = getattr(a2c_net, "actor_mlp", None)
            if actor_mlp is not None and len(actor_mlp) > 0:
                rl_first = actor_mlp[0]
                # raw obs goes through rl_games first Linear (possibly after norm_obs identity)
                rl_obs_in = agent.model.norm_obs(raw_obs) if hasattr(agent.model, "norm_obs") else raw_obs
                rl_h0 = rl_first(rl_obs_in)
                h_diff = (bc_h0 - rl_h0).abs()
                print(
                    "[diag] --- first-Linear output comparison ---\n"
                    f"[diag] BC h0:   min={bc_h0.min().item():.4e} max={bc_h0.max().item():.4e} mean_abs={bc_h0.abs().mean().item():.4e}\n"
                    f"[diag] RL h0:   min={rl_h0.min().item():.4e} max={rl_h0.max().item():.4e} mean_abs={rl_h0.abs().mean().item():.4e}\n"
                    f"[diag] |h0diff|: max={h_diff.max().item():.4e} mean={h_diff.mean().item():.4e}",
                    flush=True,
                )
    except Exception as exc:
        print(f"[diag] first-Linear comparison skipped: {exc!r}", flush=True)


@torch.no_grad()
def _get_action(agent, obs_tensor: torch.Tensor, deterministic: bool) -> torch.Tensor:
    """Query agent.model for an action the same way rl_games' BasePlayer does."""
    input_dict = {
        "is_train": False,
        "prev_actions": None,
        "obs": obs_tensor,
        "rnn_states": None,
    }
    res_dict = agent.model(input_dict)
    if deterministic:
        act = res_dict["mus"]
    else:
        act = res_dict["actions"]
    return act


# -----------------------------------------------------------------------------
# Params/weights overrides (copied from play_bc_env.py)
# -----------------------------------------------------------------------------


def _apply_params_path(env, params_path: str) -> None:
    import json

    params_path = os.path.abspath(params_path)
    if not os.path.isfile(params_path):
        raise FileNotFoundError(f"[params_path] File not found: {params_path}")
    print(f"[INFO]: Loading params/weights from: {params_path}")
    with open(params_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"[params_path] Expected a JSON object, got {type(raw).__name__}: {params_path}")
    if "stages" not in raw:
        raise ValueError(f"[params_path] Missing 'stages' key in JSON: {params_path}")
    stages_data = raw["stages"]
    if not isinstance(stages_data, dict) or not stages_data:
        raise ValueError(f"[params_path] 'stages' must be a non-empty dict: {params_path}")
    all_stages_weights = {}
    for stage_id_str, items in stages_data.items():
        stage_id = int(stage_id_str)
        if "weights" not in items or not items["weights"]:
            raise ValueError(f"[params_path] Stage {stage_id_str} missing non-empty 'weights': {params_path}")
        all_stages_weights[stage_id] = items["weights"]
        update_parameters(env, stage_id=stage_id, parameters=items.get("parameters", {}))
    update_weights(env, stage_id=None, weights=None, all_stages_weights=all_stages_weights)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>")
    print(f"CUDA_VISIBLE_DEVICES={cuda_visible}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(
            f"Using GPU: cuda:{torch.cuda.current_device()} ({torch.cuda.get_device_name(0)})",
            flush=True,
        )
    else:
        print("Using device: cpu", flush=True)

    # ------------------------------------------------------------------ env
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    # Same env-config knobs play_bc_env.py uses so the two scripts see an identical env.
    env_cfg.action_mode = "absolute_ee"
    env_cfg.enable_cameras = False
    env_cfg.debug_rewards = False
    env_cfg.sun_program_evaluator_stage_override = False
    env_cfg.seed = args_cli.seed

    env = gym.make(args_cli.task, cfg=env_cfg)  # keep the gym wrapper so env.step/reset are standard

    if args_cli.params_path is not None:
        _apply_params_path(env.unwrapped, args_cli.params_path)

    if args_cli.seed is not None:
        torch.manual_seed(args_cli.seed)
        if hasattr(env.unwrapped, "seed"):
            env.unwrapped.seed(args_cli.seed)
        random.seed(args_cli.seed)

    # -------------------------------------------------------------- rl_games
    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")
    assert agent_cfg is not None

    # Force a single-actor num_actors matching the vec env.
    agent_cfg["params"]["config"]["num_actors"] = args_cli.num_envs
    agent_cfg["params"]["config"]["device"] = args_cli.device
    agent_cfg["params"]["config"]["device_name"] = args_cli.device
    agent_cfg["params"]["load_checkpoint"] = False

    # --use_raw_obs_prior: switch the agent to the "explicitly normalized first Linear" variant.
    # With normalize_input=False:
    #   * rl_games builds the model without a running_mean_std module (norm_obs becomes a no-op).
    #   * apply_bc_prior_to_rl_games_agent selects model_state_dict_prior_raw_obs, whose first
    #     linear layer has (x - mean) / std computed explicitly before its matrix multiplication -- the exact
    #     math kuafu.bc.training.dataset.Normalizer performs. Avoids the sqrt(var + 1e-5) vs
    #     /clamp(std, 1e-8) discrepancy for low-variance obs dims.
    original_normalize_input = bool(agent_cfg["params"]["config"].get("normalize_input", False))
    if args_cli.use_raw_obs_prior:
        agent_cfg["params"]["config"]["normalize_input"] = False
        print(
            f"[play_rl_prior] --use_raw_obs_prior: overrode normalize_input "
            f"{original_normalize_input} -> False. Agent will pick model_state_dict_prior_raw_obs "
            "(explicit normalization in first linear layer, matches play_bc_env.py's Normalizer math).",
            flush=True,
        )
    # The training-algo constructor asserts `batch_size % minibatch_size == 0`, where
    # batch_size = num_actors * horizon_length (num_agents=1 here). The yaml ships
    # a minibatch_size tuned for 8192 envs (= 65536), which fails the assertion for
    # any small --num_envs we debug with. We never actually train here, so shrink
    # minibatch_size to match batch_size (one minibatch per update). No effect on
    # rollout since `agent.model(...)` is called directly.
    _horizon = int(agent_cfg["params"]["config"].get("horizon_length", 64))
    _debug_batch_size = int(args_cli.num_envs) * _horizon
    _orig_minibatch = agent_cfg["params"]["config"].get("minibatch_size")
    agent_cfg["params"]["config"]["minibatch_size"] = _debug_batch_size
    print(
        f"[play_rl_prior] Overrode minibatch_size {_orig_minibatch} -> {_debug_batch_size} "
        f"(num_envs={args_cli.num_envs} * horizon_length={_horizon}) so the A2CAgent constructor "
        "assertion passes. Rollout-only script; this does not affect behavior.",
        flush=True,
    )

    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    if args_cli.disable_wrapper_clips:
        print(
            f"[play_rl_prior] Disabling wrapper clips (was clip_obs={clip_obs}, "
            f"clip_actions={clip_actions}) -> both set to +inf.",
            flush=True,
        )
        clip_obs = math.inf
        clip_actions = math.inf
    print(f"[play_rl_prior] Wrapper clips: clip_obs={clip_obs}, clip_actions={clip_actions}", flush=True)

    rl_env = RlGamesVecEnvWrapper(env, args_cli.device, clip_obs, clip_actions)
    _register_rl_env(rl_env)

    # Build the A2C agent the same way train_rl_games.py's prior branch does.
    runner = Runner(IsaacAlgoObserver())
    runner.load(agent_cfg)
    runner.reset()

    agent = runner.algo_factory.create(runner.algo_name, base_name="run", params=runner.params)

    # Apply BC prior -- identical call to train_rl_games.py.
    report = apply_bc_prior_to_rl_games_agent(
        agent,
        args_cli.checkpoint,
        obs_stats_count=args_cli.obs_stats_count,
        verbose=True,
        load_critic_prior=False,
        require_critic_prior=False,
        freeze_input_running_mean_std=bool(args_cli.freeze_input_rms),
    )
    print(f"[play_rl_prior] Prior load report: {report}", flush=True)

    if args_cli.disable_rms_clamp:
        if hasattr(agent.model, "running_mean_std") and agent.model.running_mean_std is not None:
            _patch_rms_no_clamp(agent.model.running_mean_std)
            print(
                "[play_rl_prior] Patched running_mean_std.forward to skip clamp(y, -5, 5) "
                "(matches kuafu.bc.training.dataset.Normalizer).",
                flush=True,
            )
        else:
            print(
                "[play_rl_prior] --disable_rms_clamp is a no-op: agent.model has no "
                "running_mean_std (likely because normalize_input=False / --use_raw_obs_prior).",
                flush=True,
            )

    _silence_sigma_head(agent, args_cli.sigma_override)

    # Put the model in eval mode so RunningMeanStd doesn't accumulate stats over rollout.
    agent.model.eval()

    # --------------------------------------------------------------- rollout
    step_dt = (
        getattr(env.unwrapped, "step_dt", None)
        or getattr(env.unwrapped, "dt", None)
        or (env_cfg.sim.dt * getattr(env_cfg, "decimation", 1))
    )
    print(f"Running for {args_cli.duration_sec}s simulation time (step_dt={step_dt:.4f}s)")

    raw_env = env.unwrapped
    if hasattr(raw_env, "clear_end_to_end_rate_stats"):
        raw_env.clear_end_to_end_rate_stats()  # type: ignore[attr-defined]

    # --------------------------------------------------------- env reset path
    # `play_bc_env.py` does `env = gym.make(...).unwrapped; obs_dict, _ = env.reset()` and then
    # pulls `obs_dict["policy"]`. That skips every gym wrapper (OrderEnforcing, PassiveEnvChecker,
    # TimeLimit, ...) AND skips RlGamesVecEnvWrapper's action/obs processing. We replicate that
    # path when --bypass_rl_wrapper is set; otherwise we drive the env through rl_env (wrapper).
    # The obs key used by rl_games for this task (e.g. "state" / "policy") is read from the env
    # cfg so the bypass path picks exactly the same tensor the wrapper would have returned.
    policy_obs_key = "policy"
    try:
        obs_groups = getattr(raw_env.cfg, "observation_groups", None) or getattr(
            raw_env.cfg, "obs_groups", None
        )
        if isinstance(obs_groups, dict) and "obs" in obs_groups:
            g = obs_groups["obs"]
            if isinstance(g, (list, tuple)) and len(g) > 0:
                policy_obs_key = str(g[0])
            elif isinstance(g, str):
                policy_obs_key = g
    except Exception:
        pass

    if args_cli.bypass_rl_wrapper:
        print(
            f"[play_rl_prior] --bypass_rl_wrapper: driving env.unwrapped directly "
            f"(obs key='{policy_obs_key}'), skipping RlGamesVecEnvWrapper.step/reset and any "
            "gym wrappers stacked by gym.make. Mirrors play_bc_env.py's env = gym.make(...).unwrapped path.",
            flush=True,
        )
        if hasattr(raw_env, "seed"):
            raw_env.seed(int(args_cli.seed) if args_cli.seed is not None else 0)  # type: ignore[attr-defined]
        _reset_ret = raw_env.reset()
        obs_dict = _reset_ret[0] if isinstance(_reset_ret, tuple) else _reset_ret
        obs = obs_dict[policy_obs_key]
    else:
        rl_env.seed(int(args_cli.seed) if args_cli.seed is not None else 0)
        obs = rl_env.reset()
        if isinstance(obs, dict):
            obs = obs.get("obs", obs)

    # If diagnostic is enabled, also build the BC reference pipeline once so we can compare
    # BC-action vs RL-action at EVERY step during the rollout (not just reset). This lets us
    # see if the <=1% per-step divergence stays bounded or compounds into trajectory-level drift.
    bc_ref = None
    if args_cli.diag_compare_pipelines:
        obs_t = obs if isinstance(obs, torch.Tensor) else torch.as_tensor(obs, device=args_cli.device)
        _diag_compare_pipelines(agent, args_cli.checkpoint, obs_t, args_cli.device)
        try:
            bc_model, bc_obs_norm, bc_act_norm, _ = _bc_load_checkpoint(
                args_cli.checkpoint, torch.device(args_cli.device)
            )
            bc_model.eval()
            bc_ref = (bc_model, bc_obs_norm, bc_act_norm)
            print("[diag] continuous-diff mode: logging |BC-RL| action stats every 50 steps", flush=True)
        except Exception as exc:
            print(f"[diag] Could not enable continuous-diff mode: {exc!r}", flush=True)

    sim_time = 0.0
    total_episodes = 0
    deterministic = not args_cli.stochastic
    diag_step = 0
    diag_sum_max = 0.0
    diag_sum_mean = 0.0
    diag_n = 0

    while sim_time < args_cli.duration_sec and simulation_app.is_running() and not simulation_app.is_exiting():
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, device=args_cli.device)

        action = _get_action(agent, obs, deterministic=deterministic)

        if bc_ref is not None:
            with torch.no_grad():
                bc_model, bc_obs_norm, bc_act_norm = bc_ref
                bc_in = bc_obs_norm.normalize(obs.float())
                bc_out = bc_model(bc_in)
                bc_mu_norm = bc_out[0] if isinstance(bc_out, (tuple, list)) else bc_out
                bc_full = bc_act_norm.denormalize(bc_mu_norm)
                rl_act_dim = int(action.shape[-1])
                bc_total_dim = int(bc_full.shape[-1])
                if bc_total_dim % rl_act_dim == 0:
                    cs = bc_total_dim // rl_act_dim
                    bc_first = (
                        bc_full.reshape(bc_full.shape[0], cs, rl_act_dim)[:, 0, :]
                        if cs > 1
                        else bc_full
                    )
                    d = (bc_first - action).abs()
                    diag_sum_max += float(d.max().item())
                    diag_sum_mean += float(d.mean().item())
                    diag_n += 1
                if diag_step % 50 == 0:
                    print(
                        f"[diag][t={diag_step:04d}] |BC-RL| mean={d.mean().item():.4e} "
                        f"max={d.max().item():.4e} "
                        f"rl_act_abs={action.abs().mean().item():.4e}",
                        flush=True,
                    )
            diag_step += 1

        if args_cli.bypass_rl_wrapper:
            # Mirror play_bc_env.py: unwrapped env returns (obs_dict, rew, terminated, truncated, extras).
            step_ret = raw_env.step(action)
            if len(step_ret) == 5:
                obs_dict, _rew, terminated, truncated, _extras = step_ret
            else:
                obs_dict, _rew, dones_scalar, _extras = step_ret
                terminated = dones_scalar
                truncated = torch.zeros_like(dones_scalar)
            obs = obs_dict[policy_obs_key]
            dones = (terminated | truncated)
        else:
            obs, _rew, dones, _extras = rl_env.step(action)
            if isinstance(obs, dict):
                obs = obs.get("obs", obs)

        done_mask = dones.to(dtype=torch.bool).view(-1)
        n_done = int(done_mask.sum().item())
        if n_done > 0:
            total_episodes += n_done
            successful = (
                raw_env.get_true_success_count() if hasattr(raw_env, "get_true_success_count") else 0
            )
            print(
                f"[PLAY-RL-PRIOR] Episodes: {total_episodes} | "
                f"Successful: {successful} | "
                f"Success rate: {successful / max(total_episodes, 1) * 100:.1f}%",
                flush=True,
            )

        sim_time += step_dt

    successful = raw_env.get_true_success_count() if hasattr(raw_env, "get_true_success_count") else 0
    success_rate_pct = successful / max(total_episodes, 1) * 100.0
    print("\n" + "=" * 60)
    print("  PLAY-RL-PRIOR SUMMARY")
    print(f"  Checkpoint               : {args_cli.checkpoint}")
    print(f"  Task                     : {args_cli.task}")
    print(f"  Num envs                 : {args_cli.num_envs}")
    print(f"  Simulation time (s)      : {sim_time:.2f}")
    print(f"  Wrapper clip_obs         : {clip_obs}")
    print(f"  Wrapper clip_actions     : {clip_actions}")
    print(f"  RMS clamp disabled       : {args_cli.disable_rms_clamp}")
    print(f"  Use raw-obs prior        : {args_cli.use_raw_obs_prior}")
    print(f"  Bypass rl wrapper        : {args_cli.bypass_rl_wrapper}")
    print(f"  Freeze input RMS (asked) : {args_cli.freeze_input_rms}")
    print(f"  Freeze input RMS (applied): {report.input_running_mean_std_frozen}")
    print(f"  normalize_input (effective): {bool(agent_cfg['params']['config'].get('normalize_input', False))}")
    print(f"  Deterministic rollout    : {deterministic}")
    print(f"  Sigma override           : {args_cli.sigma_override}")
    print(f"  Total episodes completed : {total_episodes}")
    print(f"  Successful terminations  : {successful}")
    print(f"  Success rate             : {success_rate_pct:.1f}%")
    if diag_n > 0:
        print(
            f"  |BC-RL| over rollout     : mean_of_means={diag_sum_mean / diag_n:.4e}  "
            f"mean_of_maxes={diag_sum_max / diag_n:.4e}  n_steps={diag_n}"
        )
    print("=" * 60 + "\n")
    print(f"SUCCESS_RATE:{success_rate_pct:.1f}", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
