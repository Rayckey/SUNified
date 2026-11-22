"""RL-Games training/evaluation wrappers used inside worker processes."""

from __future__ import annotations

import copy
import glob
import math
import os
from datetime import datetime
from typing import Any, cast

from kuafu.paths import external_artifact_root


def _checkpoint_epoch(path: str) -> int:
    """Read the epoch counter stored inside an RL-Games checkpoint."""
    import torch

    state = torch.load(path, map_location="cpu", weights_only=False)
    return int(state.get("epoch", 0))


def _dump_run_yaml_configs(
    *,
    run_dir: str,
    env,
    agent_cfg: dict,
    task: str,
    worker_idx: int,
    prior_mode: str,
    effective_max_epochs: int,
) -> None:
    """Persist resolved configs for reproducibility, mirroring IsaacLab train.py."""
    from isaaclab.utils.io import dump_yaml

    params_dir = os.path.join(run_dir, "params")
    os.makedirs(params_dir, exist_ok=True)

    env_cfg = getattr(getattr(env, "unwrapped", env), "cfg", None)
    if env_cfg is not None and hasattr(env_cfg, "log_dir"):
        env_cfg.log_dir = run_dir

    if env_cfg is not None:
        dump_yaml(os.path.join(params_dir, "env.yaml"), env_cfg)
    dump_yaml(os.path.join(params_dir, "agent.yaml"), agent_cfg)
    dump_yaml(
        os.path.join(params_dir, "runtime.yaml"),
        {
            "task": task,
            "worker_idx": int(worker_idx),
            "prior_mode": str(prior_mode),
            "effective_max_epochs": int(effective_max_epochs),
            "kuafu_run_dir": os.path.dirname(os.path.dirname(run_dir)),
            "run_dir": run_dir,
            "env_vars": {
                key: os.environ.get(key)
                for key in sorted(os.environ)
                if key.startswith("KUAFU_") or key in {"CUDA_VISIBLE_DEVICES"}
            },
        },
    )
    print(f"[residual_rl] Wrote resolved YAML configs to {params_dir}", flush=True)


def _load_bc_supervised_dataset(
    prior_checkpoint_path: str,
    device: str,
    *,
    dataset_path_override: str | None = None,
) -> dict:
    """Load the BC demonstration dataset for supervised regularization during PPO.

    This reads the BC run's adjacent ``train_config.yaml`` to locate the HDF5
    dataset and preprocessing settings (obs_key, obs_history, action_chunk_size,
    time_shift, action_mode), loads the in-memory (obs, action) pairs, applies the
    same history/chunk preprocessing the BC model was trained with, and extracts
    the obs normalizer stats (mean/std) from the checkpoint.  Everything is
    returned as float32 tensors pinned to ``device``.

    The returned dict has:
    - obs_raw:  (N, rl_obs_dim) raw (un-normalized) obs after history stacking.
    - actions_raw: (N, rl_act_dim) raw (un-normalized) actions after chunking.
    - obs_mean, obs_std: (rl_obs_dim,) snapshot stats used to pre-normalize obs
      before the actor trunk (RMS-drift-safe; do NOT use live running_mean_std).
    - dataset_path, obs_key, obs_history, action_chunk_size, action_mode, time_shift,
      num_pairs:  metadata for logging.

    The obs-normalization snapshot is pinned at load time and never updates, which
    is what makes the regularizer immune to live RMS drift after it unfreezes.
    """
    import torch
    import yaml

    from kuafu.bc.training.rl_games_prior import _resolve_prior_checkpoint_path
    from kuafu.bc.training.train_bc_prior_mlp import (
        _apply_preprocessing,
        _build_dataset,
    )

    resolved_ckpt_path, _ = _resolve_prior_checkpoint_path(prior_checkpoint_path)
    run_dir = os.path.dirname(resolved_ckpt_path)
    cfg_path = os.path.join(run_dir, "train_config.yaml")
    if not os.path.isfile(cfg_path):
        alt = os.path.join(run_dir, "meta", "config.yaml")
        if os.path.isfile(alt):
            cfg_path = alt
        else:
            raise FileNotFoundError(
                f"Could not find BC run config next to {resolved_ckpt_path} "
                f"(looked for train_config.yaml and meta/config.yaml)."
            )

    with open(cfg_path, encoding="utf-8") as f:
        run_cfg = yaml.safe_load(f)
    if not isinstance(run_cfg, dict):
        raise ValueError(f"BC train_config did not deserialize to a dict: {cfg_path}")
    dataset_cfg = run_cfg.get("dataset") or {}
    preproc_cfg = run_cfg.get("preprocessing") or {}

    dataset_path = str(dataset_path_override or dataset_cfg.get("dataset") or "")
    if not dataset_path:
        raise ValueError(
            f"BC train_config is missing `dataset.dataset` and no override was provided: {cfg_path}"
        )
    if not os.path.isabs(dataset_path) and not os.path.isfile(dataset_path):
        # Relative dataset paths travel with their config or artifact store.
        for base in (os.path.dirname(cfg_path), str(external_artifact_root())):
            candidate = os.path.join(base, dataset_path)
            if os.path.isfile(candidate):
                dataset_path = candidate
                break
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"BC dataset HDF5 not found: {dataset_path}")

    obs_key = str(dataset_cfg.get("obs_key") or "state")
    action_mode = str(dataset_cfg.get("action_mode") or "relative_joint")
    time_shift = int(dataset_cfg.get("time_shift", 0) or 0)
    obs_history = int(preproc_cfg.get("obs_history", 1) or 1)
    action_chunk_size = int(preproc_cfg.get("action_chunk_size", 1) or 1)
    # Mirror BC's legacy alias resolution in `_run_config_from_yaml`: accept `skip_step`
    # when `action_chunk_stride` is absent, so older BC train_configs load identically.
    action_chunk_stride = int(
        preproc_cfg.get("action_chunk_stride", preproc_cfg.get("skip_step", 1)) or 1
    )
    demo = dataset_cfg.get("demo")
    val_fraction = float(dataset_cfg.get("val_fraction", 0.0) or 0.0)
    split_mode = str(dataset_cfg.get("split_mode", "contiguous") or "contiguous")
    seed = int(dataset_cfg.get("seed", 1999) or 1999)

    # Use BC's own dataset + preprocessing builders so the regularizer stays in lock-step
    # with any future change to BC's training pipeline.  `load_returns=False` because the
    # BC-on-dataset loss only needs (obs, action); returns are only used for critic
    # pretraining in BC.  Mirrors how `train_one` builds `train_ds` (same split/seed so
    # the ckpt's obs_normalizer stats — which were fit on this same train split — line up).
    ds = _build_dataset(
        dataset_path=dataset_path,
        demo=demo,
        obs_key=obs_key,
        action_mode=action_mode,
        time_shift=time_shift,
        split="train",
        val_fraction=val_fraction,
        split_mode=split_mode,
        seed=seed,
        load_returns=False,
    )
    _apply_preprocessing(ds, obs_history, action_chunk_size, action_chunk_stride)
    if len(ds) == 0:
        raise ValueError(f"BC dataset loaded from {dataset_path} is empty after preprocessing.")

    obs_raw = ds.obs.detach().clone().to(device=device, dtype=torch.float32).contiguous()
    actions_raw = ds.actions.detach().clone().to(device=device, dtype=torch.float32).contiguous()

    ckpt = torch.load(resolved_ckpt_path, map_location="cpu", weights_only=False)
    obs_norm = ckpt.get("obs_normalizer")
    if not isinstance(obs_norm, dict) or "mean" not in obs_norm or "std" not in obs_norm:
        raise KeyError(
            f"BC checkpoint at {resolved_ckpt_path} is missing `obs_normalizer[mean|std]`; "
            "cannot RMS-safely run the BC dataset loss."
        )
    obs_mean = obs_norm["mean"].detach().clone().to(device=device, dtype=torch.float32).flatten()
    obs_std = obs_norm["std"].detach().clone().to(device=device, dtype=torch.float32).flatten().clamp_min(1e-6)

    if obs_mean.numel() != obs_raw.shape[1]:
        raise ValueError(
            "BC obs_normalizer mean size does not match preprocessed obs width: "
            f"mean={obs_mean.numel()} vs dataset_obs_dim={obs_raw.shape[1]} "
            f"(obs_history={obs_history}, obs_key={obs_key})"
        )

    return {
        "obs_raw": obs_raw,
        "actions_raw": actions_raw,
        "obs_mean": obs_mean,
        "obs_std": obs_std,
        "dataset_path": dataset_path,
        "obs_key": obs_key,
        "obs_history": obs_history,
        "action_chunk_size": action_chunk_size,
        "action_mode": action_mode,
        "time_shift": time_shift,
        "num_pairs": int(obs_raw.shape[0]),
    }


def _stabilize_sigma_head_after_prior(agent_cfg: dict, agent) -> None:
    """Reset learned sigma head to a safe constant after copying BC trunk weights.

    BC prior loading only initializes the actor trunk and mean head. When RL-Games uses
    `fixed_sigma=False`, the sigma head is another Linear layer whose default constant-weight
    init can interact badly with the copied BC features and produce inf/NaN std on the first step.
    """
    import torch
    import torch.nn as nn

    net = agent.model.a2c_network
    fixed_sigma = bool(getattr(net, "fixed_sigma", False))
    if fixed_sigma or not hasattr(net, "sigma"):
        return

    sigma_head = net.sigma
    if not isinstance(sigma_head, nn.Linear):
        return

    sigma_cfg = (
        agent_cfg.get("params", {})
        .get("network", {})
        .get("space", {})
        .get("continuous", {})
        .get("sigma_init", {})
    )
    sigma_init_val = float(sigma_cfg.get("val", 0.0)) if isinstance(sigma_cfg, dict) else 0.0

    with torch.no_grad():
        sigma_head.weight.zero_()
        if sigma_head.bias is not None:
            sigma_head.bias.fill_(sigma_init_val)

    print(
        f"[residual_rl] Reset learned sigma head after BC prior load: weight=0 bias={sigma_init_val}",
        flush=True,
    )


def _freeze_actor_for_epochs(
    agent,
    freeze_epochs: int,
    *,
    unfreeze_input_rms_at_end: bool = False,
    rms_unfreeze_after_actor_lr_warmup: bool = False,
    sigma_ramp_epochs: int = 0,
    silenced_sigma_val: float = -10.0,
    sigma_final_val: float | None = None,
    actor_lr_warmup_epochs: int = 0,
    actor_lr_cap: float | None = None,
    lr_schedule_freeze_during_warmup: bool = False,
    fixed_lr: float | None = None,
    bc_anchor_epochs: int = 0,
    bc_anchor_coef: float = 0.0,
    bc_dataset: dict | None = None,
    bc_dataset_coef: float = 0.0,
    bc_dataset_decay_epochs: int = 0,
    bc_dataset_batch_size: int = 1024,
) -> None:
    """Protect a BC/MPC-initialized actor during PPO fine-tuning.

    Sigma is silenced at the start of the freeze and (optionally) ramped back to the
    ramp target over the LAST ``sigma_ramp_epochs`` of the freeze window, so that by
    the time the actor is unfrozen sigma has already reached that target.  Critically,
    this means PPO never sees the actor unfrozen with a degenerate near-zero sigma —
    the importance ratio / log-prob math stays well-conditioned the very first epoch
    the actor starts learning, and the critic has already calibrated against the
    noisier distribution it will face.  An earlier design ramped sigma AFTER unfreeze;
    that blew up because the first unfrozen epochs ran with sigma ≈ exp(-10), giving
    enormous gradients/ratios and a catastrophic policy update before BC anchor or
    LR warmup could damp it.

    The ramp target defaults to the sigma value loaded from the RL-Games config
    (``sigma_init``, typically ``-3`` ⇒ std ≈ 0.05).  For a trained BC prior that was
    collected on effectively deterministic demonstrations, 0.05 action std (≈ 5 cm of
    EE noise at clip_actions=1.0) exceeds many stage predicates' tolerances (e.g. the
    2 cm freeze_position check), so restoring sigma to that value destroys the policy's
    stage-completion competence even though its weights are unchanged.  Pass
    ``sigma_final_val`` to override the ramp target: for example, ``-5.5`` gives
    std ≈ 4 mm which matches BC-scale precision and lets PPO then expand sigma
    upward if more exploration helps.  When ``sigma_final_val`` is None, behavior is
    backward-compatible (restore to the checkpoint's saved sigma).

    The other important part is the optimizer-step hook.  rl_games can reset all
    optimizer parameter groups to ``self.last_lr`` inside its PPO minibatch path.  If
    we only scale the actor LR at epoch boundaries, the first unfrozen PPO minibatch
    can still run at the full base LR and blow up the BC mean head.  This function
    therefore enforces the actor LR immediately before and immediately after every
    ``optimizer.step()`` while warmup/capping is active.

    When ``lr_schedule_freeze_during_warmup`` is True, we also pin rl_games'
    ``agent.scheduler.update(...)`` to a no-op for the duration of the actor LR
    warmup window (between the actor-unfreeze epoch and
    ``unfreeze_epoch + actor_lr_warmup_epochs``).  This is needed because at the
    unfreeze boundary PPO often reports a massive one-shot KL (empirically ~130
    for a BC-initialized actor) — regardless of whether the actor's weights
    actually moved — and the default ``AdaptiveScheduler`` reacts by halving
    ``self.last_lr`` at every mini-epoch with ``kl > 2*kl_threshold``, crushing
    the base LR by 100x+ in a single epoch.  Once the base LR is tiny, our
    ``actor_lr = warmup_factor * base_lr`` formula multiplies by near-zero and
    the 200-epoch warmup never delivers a real learning rate to the actor; the
    critic can't learn either, and the policy dies.  Freezing the scheduler
    keeps ``last_lr`` pinned at its pre-unfreeze value (typically the config
    ``learning_rate``) through the whole warmup, so the actor warms up against a
    stable base LR and the critic trains at full LR throughout.  The scheduler
    is restored unchanged at the end of warmup, at which point the actor has
    had 200 epochs of real gradient updates and any subsequent KL is of natural
    PPO scale rather than an unfreeze artifact.

    Finally, when ``unfreeze_input_rms_at_end`` is True the observation
    ``running_mean_std`` is unfrozen at the actor-unfreeze boundary by default.
    That causes a subtle disaster: PPO's minibatch update pass calls
    ``_preproc_obs(stored_obs)`` which in turn calls ``RunningMeanStd.forward`` in
    training mode, so the normalizer's ``running_mean/var/count`` shift with every
    minibatch it processes.  Because the BC ``mu`` head was trained against the
    expert-seeded RMS stats, even a small normalizer shift changes the network's
    output *with weights unchanged*, which makes ``old_log_prob != new_log_prob``,
    producing a fake KL that triggers the PPO adaptive LR scheduler to crush the
    base LR, and the policy's stage-completion competence collapses within a few
    epochs — even though the LR warmup correctly held actor_lr at 0.  Passing
    ``rms_unfreeze_after_actor_lr_warmup=True`` (together with
    ``actor_lr_warmup_epochs > 0``) defers the RMS unfreeze to
    ``unfreeze_epoch + actor_lr_warmup_epochs``.  By then the actor has had its
    full LR warmup window with a stable normalizer and has had time to actually
    learn with real gradients, so subsequent RMS drift is absorbed gracefully
    instead of invalidating a never-updated BC mean head.
    """
    import torch
    import torch.nn as nn

    net = agent.model.a2c_network
    actor_params: list[nn.Parameter] = []
    seen: set[int] = set()
    for attr in ("actor_mlp", "mu", "sigma"):
        module_or_param = getattr(net, attr, None)
        if module_or_param is None:
            continue
        params = [module_or_param] if isinstance(module_or_param, nn.Parameter) else list(module_or_param.parameters())
        for p in params:
            if p.requires_grad and id(p) not in seen:
                actor_params.append(p)
                seen.add(id(p))

    if not actor_params:
        print("[residual_rl] WARNING: no actor params found for prior safety controls — skipping.", flush=True)
        return

    effective_freeze_epochs = max(0, int(freeze_epochs))
    effective_warmup_epochs = max(0, int(actor_lr_warmup_epochs))
    effective_bc_anchor_epochs = max(0, int(bc_anchor_epochs))
    effective_bc_anchor_coef = max(0.0, float(bc_anchor_coef))
    effective_bc_dataset_coef = max(0.0, float(bc_dataset_coef))
    effective_bc_dataset_decay_epochs = max(0, int(bc_dataset_decay_epochs))
    effective_bc_dataset_batch_size = max(1, int(bc_dataset_batch_size))
    effective_actor_lr_cap: float | None = None
    if actor_lr_cap is not None:
        try:
            cap = float(actor_lr_cap)
            if math.isfinite(cap) and cap > 0.0:
                effective_actor_lr_cap = cap
        except (TypeError, ValueError):
            effective_actor_lr_cap = None

    # --- Fixed LR install (disables rl_games' adaptive scheduler) ---------------
    # When `fixed_lr` is provided, we permanently pin the agent's LR to that value
    # and neutralize rl_games' scheduler.  This eliminates the entire class of
    # failures where the AdaptiveScheduler reacts to transient KL spikes (from
    # RMS unfreeze, sigma ramp boundary, BC-dataset gradient, actor-unfreeze
    # one-shot, etc.) by halving `base_lr` every minibatch until it collapses
    # below the actor LR cap and kills both actor and critic training.  For a
    # BC-warm-start regime the scheduler's assumptions are inverted (the policy
    # starts good, early KL is transient artifact not "step too big"), so
    # running at a constant small LR is both simpler and more robust than the
    # coordination dance of actor_lr_cap + actor_lr_warmup + lr_schedule_freeze
    # + rms_unfreeze_after_actor_lr_warmup.  When this is set, those other LR-
    # control knobs become no-ops (by design): the fixed LR is the LR.
    effective_fixed_lr: float | None = None
    if fixed_lr is not None:
        try:
            flr = float(fixed_lr)
            if math.isfinite(flr) and flr > 0.0:
                effective_fixed_lr = flr
        except (TypeError, ValueError):
            effective_fixed_lr = None

    if effective_fixed_lr is not None:
        orig_last_lr = float(getattr(agent, "last_lr", 0.0))
        setattr(agent, "last_lr", effective_fixed_lr)
        optimizer_obj = getattr(agent, "optimizer", None)
        if optimizer_obj is not None and hasattr(optimizer_obj, "param_groups"):
            for group in optimizer_obj.param_groups:
                group["lr"] = effective_fixed_lr

        scheduler = getattr(agent, "scheduler", None)
        if scheduler is not None and hasattr(scheduler, "update"):
            def _patched_scheduler_update_fixed(
                current_lr, entropy_coef, epoch_num, frames, kl_dist, *args, **kwargs
            ):
                return effective_fixed_lr, entropy_coef
            scheduler.update = _patched_scheduler_update_fixed
        print(
            f"[residual_rl] Fixed LR installed: lr={effective_fixed_lr:.2e} "
            f"(was agent.last_lr={orig_last_lr:.2e}); rl_games scheduler neutralized "
            "(scheduler.update() pinned; optimizer param-group LRs set). "
            "actor_lr_cap/actor_lr_warmup_epochs/lr_schedule_freeze_during_warmup are "
            "redundant and have no effect while fixed_lr is active.",
            flush=True,
        )

    actor_param_ids = {id(p) for p in actor_params}

    if effective_freeze_epochs > 0:
        for p in actor_params:
            p.requires_grad_(False)

    # Silence sigma only when we actually freeze the actor.  This keeps frozen rollouts
    # close to deterministic BC behavior while the critic calibrates.  Supports both
    # fixed-sigma Parameter and learned-sigma Linear head layouts used by rl_games.
    sigma_obj = getattr(net, "sigma", None)
    sigma_saved_param: torch.Tensor | None = None
    sigma_saved_state: dict[str, torch.Tensor] | None = None
    sigma_silenced_state: dict[str, torch.Tensor] | None = None

    def _sigma_mean_text() -> str:
        if isinstance(sigma_obj, nn.Parameter):
            return f"{float(sigma_obj.data.mean().item()):.3f}"
        if isinstance(sigma_obj, nn.Linear) and sigma_obj.bias is not None:
            return f"{float(sigma_obj.bias.data.mean().item()):.3f}"
        return "n/a"

    if effective_freeze_epochs > 0 and isinstance(sigma_obj, nn.Parameter):
        sigma_saved_param = sigma_obj.data.clone()
        with torch.no_grad():
            sigma_obj.data.fill_(float(silenced_sigma_val))
        print(
            f"[residual_rl] Sigma parameter silenced to {float(silenced_sigma_val):.3f} "
            f"during freeze (was {sigma_saved_param.mean().item():.3f}); rollout uses BC mean.",
            flush=True,
        )
    elif effective_freeze_epochs > 0 and isinstance(sigma_obj, nn.Linear):
        sigma_saved_state = {k: v.detach().clone() for k, v in sigma_obj.state_dict().items()}
        with torch.no_grad():
            sigma_obj.weight.zero_()
            if sigma_obj.bias is not None:
                sigma_obj.bias.fill_(float(silenced_sigma_val))
        sigma_silenced_state = {k: v.detach().clone() for k, v in sigma_obj.state_dict().items()}
        saved_bias = sigma_saved_state.get("bias")
        saved_bias_mean = float(saved_bias.mean().item()) if saved_bias is not None else float("nan")
        print(
            f"[residual_rl] Sigma Linear head silenced to {float(silenced_sigma_val):.3f} "
            f"during freeze (saved bias mean={saved_bias_mean:.3f}); rollout uses BC mean.",
            flush=True,
        )

    # --- Sigma ramp/restore TARGET ---------------------------------------------
    # The ramp endpoint and the unfreeze-time restore value.  Defaults to the saved
    # checkpoint value (backward-compat); when `sigma_final_val` is set, both the
    # ramp target and the restored value use that instead.  A lower target keeps
    # action noise below the BC prior's stage-completion tolerance so that
    # unfreezing the actor does not immediately poison rollouts with noise.
    sigma_target_param: torch.Tensor | None = sigma_saved_param
    sigma_target_state: dict[str, torch.Tensor] | None = sigma_saved_state
    sigma_target_value: float | None = None
    if (
        effective_freeze_epochs > 0
        and sigma_final_val is not None
        and (sigma_saved_param is not None or sigma_saved_state is not None)
    ):
        try:
            sigma_target_value = float(sigma_final_val)
        except (TypeError, ValueError):
            sigma_target_value = None
        if sigma_target_value is not None and math.isfinite(sigma_target_value):
            if sigma_saved_param is not None:
                sigma_target_param = torch.full_like(sigma_saved_param, sigma_target_value)
                print(
                    f"[residual_rl] Sigma ramp target overridden: will end at "
                    f"{sigma_target_value:.3f} (std≈{math.exp(sigma_target_value):.4g}) "
                    f"instead of the checkpoint's saved value "
                    f"{float(sigma_saved_param.mean().item()):.3f}.",
                    flush=True,
                )
            elif sigma_saved_state is not None:
                sigma_target_state = {k: v.clone() for k, v in sigma_saved_state.items()}
                if "bias" in sigma_target_state and sigma_target_state["bias"] is not None:
                    sigma_target_state["bias"] = torch.full_like(
                        sigma_target_state["bias"], sigma_target_value
                    )
                if "weight" in sigma_target_state and sigma_target_state["weight"] is not None:
                    # Zero the weight so the Linear head produces a state-independent
                    # bias-only sigma equal to `sigma_target_value`.  Preserves the
                    # originally-saved weight if the caller does not override the
                    # target.
                    sigma_target_state["weight"] = torch.zeros_like(sigma_target_state["weight"])
                saved_bias = sigma_saved_state.get("bias")
                saved_bias_mean = (
                    float(saved_bias.mean().item()) if saved_bias is not None else float("nan")
                )
                print(
                    f"[residual_rl] Sigma Linear head ramp target overridden: will end at bias="
                    f"{sigma_target_value:.3f} (std≈{math.exp(sigma_target_value):.4g}) with "
                    f"weight zeroed; saved bias mean was {saved_bias_mean:.3f}.",
                    flush=True,
                )

    freeze_start = int(agent.epoch_num)
    # rl_games increments epoch_num before train_epoch().  With freeze_start=0 and
    # freeze_epochs=200, unfreeze at 201 so epochs 1..200 are fully frozen.
    unfreeze_epoch = freeze_start + effective_freeze_epochs + 1
    effective_ramp_epochs = max(0, min(int(sigma_ramp_epochs), effective_freeze_epochs))
    ramp_start = unfreeze_epoch - effective_ramp_epochs
    unfrozen = [effective_freeze_epochs <= 0]
    last_ramp_logged = [-1]

    bc_anchor_model = None
    if effective_bc_anchor_epochs > 0 and effective_bc_anchor_coef > 0.0:
        bc_anchor_model = copy.deepcopy(agent.model)
        bc_anchor_model.eval()
        for p in bc_anchor_model.parameters():
            p.requires_grad_(False)
        setattr(agent, "_kuafu_bc_anchor_loss", torch.zeros(1, device=agent.ppo_device))
        setattr(agent, "_kuafu_bc_anchor_coef", effective_bc_anchor_coef)
        setattr(agent, "_kuafu_bc_anchor_epochs", effective_bc_anchor_epochs)
    bc_anchor_last_logged = [-1]

    # --- BC-on-dataset supervised loss state ----------------------------------
    # The user-selected regularizer adds a supervised BC loss computed on rows
    # sampled from the actual demonstration dataset (not the rollout buffer):
    # ``coef(t) * || mu_theta(obs_expert_normalized) - action_expert ||^2``.
    # The obs-normalization used on the dataset pathway is pinned to the BC
    # checkpoint's ``obs_normalizer`` stats at load time (``bc_dataset['obs_mean']``
    # / ``obs_std``), NOT ``agent.model.running_mean_std``.  This keeps the
    # regularizer correct after RMS unfreezes (epoch = unfreeze + warmup): the
    # live RMS drifts with on-policy distribution while the BC-dataset loss keeps
    # pulling the policy toward dataset actions computed under expert-distribution
    # stats, which is exactly the behavior we want.
    #
    # The coefficient decays linearly from ``coef`` at unfreeze_epoch to
    # ``coef * 0.1`` over ``bc_dataset_decay_epochs`` and holds at 0.1 * coef
    # afterward (always-on floor). When ``decay_epochs == 0`` the coefficient is
    # constant at ``coef``.
    bc_dataset_active = (
        bc_dataset is not None and effective_bc_dataset_coef > 0.0
    )
    bc_dataset_state: dict = {
        "obs_raw": None,
        "actions_raw": None,
        "obs_mean": None,
        "obs_std": None,
        "num_pairs": 0,
        "generator": None,
        "last_logged": -1,
    }
    if bc_dataset_active:
        assert bc_dataset is not None  # for the type checker; guarded by bc_dataset_active
        bc_dataset_state["obs_raw"] = bc_dataset["obs_raw"]
        bc_dataset_state["actions_raw"] = bc_dataset["actions_raw"]
        bc_dataset_state["obs_mean"] = bc_dataset["obs_mean"]
        bc_dataset_state["obs_std"] = bc_dataset["obs_std"]
        bc_dataset_state["num_pairs"] = int(bc_dataset["num_pairs"])
        # Sanity check: the pinned obs width must match the first-layer input of
        # the live actor_mlp, else the BC weights aren't loaded with a matching
        # preprocessing layout and running the regularizer would silently
        # misnormalize.
        try:
            first_linear = None
            for m in net.actor_mlp.children():
                if isinstance(m, nn.Linear):
                    first_linear = m
                    break
            if first_linear is not None:
                rl_in = int(first_linear.weight.shape[1])
                if rl_in != int(bc_dataset_state["obs_raw"].shape[1]):
                    raise ValueError(
                        "BC dataset obs width does not match actor_mlp first-layer input: "
                        f"dataset_obs_dim={int(bc_dataset_state['obs_raw'].shape[1])} "
                        f"vs actor_in={rl_in}. Check obs_history / obs_key alignment."
                    )
            mu_head = getattr(net, "mu", None)
            if isinstance(mu_head, nn.Linear):
                rl_out = int(mu_head.weight.shape[0])
                dataset_act_dim = int(bc_dataset_state["actions_raw"].shape[1])
                if rl_out == dataset_act_dim:
                    pass  # no chunking mismatch
                elif dataset_act_dim > rl_out and dataset_act_dim % rl_out == 0:
                    # Mirrors kuafu.bc.training.rl_games_prior.apply_bc_prior_to_rl_games_agent:
                    # when the BC checkpoint was trained with action_chunk_size > 1 but the
                    # RL env executes only one action per step, the prior loader copies
                    # ONLY the first rl_out rows of mean_layer into net.mu.  The BC policy
                    # on RL-Games therefore predicts the FIRST action of each chunk.  Our
                    # dataset regularizer must compare against the same slice — i.e., the
                    # first raw_act_dim columns of the chunked dataset actions.
                    chunk_mult = dataset_act_dim // rl_out
                    bc_dataset_state["actions_raw"] = (
                        bc_dataset_state["actions_raw"][:, :rl_out].contiguous()
                    )
                    print(
                        f"[residual_rl] BC dataset: sliced chunked actions "
                        f"({dataset_act_dim} -> {rl_out} dims, chunk_size={chunk_mult}) "
                        "to match the live RL-Games mu head (BC prior uses only the first "
                        "action of each chunk at rollout time).",
                        flush=True,
                    )
                else:
                    raise ValueError(
                        "BC dataset action width does not match mu head output: "
                        f"dataset_act_dim={dataset_act_dim} vs mu_out={rl_out} "
                        "(not an exact multiple — cannot reconcile). "
                        "Check action_chunk_size / action_mode alignment."
                    )
        except AttributeError as exc:
            raise RuntimeError(
                "[residual_rl] Failed to validate BC dataset dimensions against the live actor network."
            ) from exc

    def _bc_dataset_coef_at(current_epoch: int) -> float:
        """Linear decay from ``coef`` -> ``coef * 0.1`` over decay_epochs.

        Always-on; before unfreeze returns 0 (PPO gradients aren't flowing yet
        so the regularizer would only slow BC init copy into Adam). After
        unfreeze + decay_epochs, stays at the 0.1*coef floor indefinitely.
        """
        if not bc_dataset_active:
            return 0.0
        if current_epoch < unfreeze_epoch:
            return 0.0
        if effective_bc_dataset_decay_epochs <= 0:
            return effective_bc_dataset_coef
        progress = (current_epoch - unfreeze_epoch) / float(effective_bc_dataset_decay_epochs)
        progress = max(0.0, min(1.0, progress))
        # Linear interpolation from coef (progress=0) -> 0.1*coef (progress=1)
        return effective_bc_dataset_coef * (1.0 - 0.9 * progress)

    lr_controls_requested = effective_warmup_epochs > 0 or effective_actor_lr_cap is not None
    lr_state: dict = {
        "installed": False,
        "completed": False,
        "actor_group_idx": None,
        "orig_update_lr": None,
        "orig_optimizer_step": None,
        "last_factor_logged": -1.0,
        "step_logs": 0,
    }

    # LR-schedule freeze during warmup.  Only meaningful when there is actually
    # a warmup window AND the caller asked for it.  See the function docstring
    # for the full rationale; in short, we pin agent.scheduler.update(...) to
    # return the pre-unfreeze last_lr so the AdaptiveScheduler cannot react to
    # the one-shot unfreeze-boundary KL spike and crush the base LR.
    schedule_freeze_requested = bool(
        lr_schedule_freeze_during_warmup and effective_warmup_epochs > 0
    )
    schedule_freeze_state: dict = {
        "installed": False,
        "completed": False,
        "orig_update": None,
        "pinned_lr": None,
        "call_count": 0,
        "kl_spike_logs": 0,
    }

    # RMS-unfreeze scheduling.  When rms_unfreeze_after_actor_lr_warmup is True AND
    # an actor LR warmup window is actually configured (>0 epochs), we defer the
    # running_mean_std unfreeze from "actor unfreeze" to "actor unfreeze + warmup".
    # Otherwise we keep the legacy behavior (unfreeze at actor unfreeze).  The target
    # epoch is the first training epoch at which the warmup window has fully elapsed,
    # mirroring the boundary used by _compute_warmup_factor (progress >= 1.0).
    rms_defer_mode = bool(rms_unfreeze_after_actor_lr_warmup and effective_warmup_epochs > 0)
    rms_deferred_unfreeze_epoch = unfreeze_epoch + effective_warmup_epochs if rms_defer_mode else unfreeze_epoch
    rms_unfreeze_state: dict = {
        "target_epoch": None,  # populated at actor-unfreeze boundary when defer_mode is on
        "done": False,
    }

    def _do_rms_unfreeze(current_epoch: int, *, reason: str) -> None:
        """Unfreeze the observation running_mean_std and mark the state as done.

        ``reason`` is either ``"actor_unfreeze"`` (legacy immediate path) or
        ``"actor_lr_warmup_end"`` (deferred path) and is included in the log line
        so the collapse post-mortem in worker logs is unambiguous.
        """
        if rms_unfreeze_state["done"]:
            return
        rms = getattr(agent.model, "running_mean_std", None)
        if rms is not None and getattr(rms, "_bc_prior_frozen", False):
            from kuafu.bc.training.rl_games_prior import unfreeze_running_mean_std

            unfreeze_running_mean_std(rms)
            print(
                f"[residual_rl] Observation running_mean_std unfrozen at epoch {current_epoch} "
                f"(reason={reason}); PPO rollouts will now update the normalizer.",
                flush=True,
            )
        rms_unfreeze_state["done"] = True

    def _compute_warmup_factor(current_epoch: int) -> float:
        if effective_warmup_epochs <= 0:
            return 1.0
        progress = float(current_epoch - unfreeze_epoch) / float(effective_warmup_epochs)
        return min(max(progress, 0.0), 1.0)

    def _resolve_base_lr(base_lr: float | None = None) -> float:
        if base_lr is not None:
            try:
                lr = float(base_lr)
                if math.isfinite(lr) and lr >= 0.0:
                    return lr
            except (TypeError, ValueError) as exc:
                raise ValueError(f"[residual_rl] Invalid base_lr value: {base_lr!r}") from exc
            raise ValueError(f"[residual_rl] base_lr must be finite and non-negative, got {base_lr!r}")
        # Prefer a non-actor optimizer group; the actor group may already be scaled/capped.
        for group in getattr(agent.optimizer, "param_groups", []):
            if not group.get("_kuafu_is_actor_group", False):
                try:
                    lr = float(group["lr"])
                    if math.isfinite(lr) and lr >= 0.0:
                        return lr
                except (KeyError, TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"[residual_rl] Invalid non-actor optimizer lr group: {group!r}"
                    ) from exc
                raise RuntimeError(f"[residual_rl] Non-actor optimizer lr must be finite and non-negative: {group!r}")
        last_lr_error: BaseException | None = None
        try:
            lr = float(getattr(agent, "last_lr"))
            if math.isfinite(lr) and lr >= 0.0:
                return lr
            last_lr_error = RuntimeError(
                f"agent.last_lr must be finite and non-negative, got {getattr(agent, 'last_lr', None)!r}"
            )
        except (TypeError, ValueError, AttributeError) as exc:
            last_lr_error = exc
        idx = lr_state.get("actor_group_idx")
        if idx is not None:
            return float(agent.optimizer.param_groups[int(idx)]["lr"])
        if last_lr_error is not None:
            raise RuntimeError("[residual_rl] Invalid or missing agent.last_lr while resolving actor LR.") from last_lr_error
        return float(agent.optimizer.param_groups[0]["lr"])

    def _target_actor_lr(full_lr: float, factor: float) -> float:
        # Ramp from 0 toward the effective *target* LR, where the target is the
        # stricter of (base LR, actor LR cap).  The previous formulation
        # ``min(full_lr * factor, cap)`` made the warmup finish at
        # ``factor = cap / full_lr``: with base_lr=1e-2 and cap=1e-4 the 300-epoch
        # warmup was effectively a 3-epoch warmup because the cap clipped at
        # factor >= 0.01.  That short-circuited the warmup entirely and landed the
        # actor at the full cap by epoch ~54, while the critic had only seen
        # `freeze_actor_epochs` of training — producing fast collapse.  Ramping
        # toward ``min(full_lr, cap)`` makes the warmup length actually control
        # how long the actor takes to reach its target LR.
        target = float(full_lr)
        if effective_actor_lr_cap is not None:
            target = min(target, float(effective_actor_lr_cap))
        return target * float(factor)

    def _apply_actor_lr_control(base_lr: float | None = None) -> tuple[float, float, float] | None:
        if not lr_state["installed"]:
            return None
        if lr_state["completed"] and effective_actor_lr_cap is None:
            return None
        idx = lr_state["actor_group_idx"]
        if idx is None:
            return None
        full_lr = _resolve_base_lr(base_lr)
        factor = _compute_warmup_factor(int(agent.epoch_num))
        actor_lr = _target_actor_lr(full_lr, factor)
        agent.optimizer.param_groups[int(idx)]["lr"] = actor_lr
        return full_lr, factor, actor_lr

    def _split_optimizer_param_groups() -> None:
        optimizer = agent.optimizer
        new_groups: list[dict] = []
        for group in optimizer.param_groups:
            actor_ps = [p for p in group["params"] if id(p) in actor_param_ids]
            other_ps = [p for p in group["params"] if id(p) not in actor_param_ids]
            if other_ps:
                other_group = {k: v for k, v in group.items() if k != "params"}
                other_group["params"] = other_ps
                other_group["_kuafu_is_actor_group"] = False
                new_groups.append(other_group)
            if actor_ps:
                actor_group = {k: v for k, v in group.items() if k != "params"}
                actor_group["params"] = actor_ps
                actor_group["_kuafu_is_actor_group"] = True
                new_groups.append(actor_group)
        optimizer.param_groups = new_groups
        lr_state["actor_group_idx"] = None
        for i, group in enumerate(optimizer.param_groups):
            if group.get("_kuafu_is_actor_group", False):
                lr_state["actor_group_idx"] = i
                return

    def _install_update_lr_wrapper() -> None:
        orig_update_lr = agent.update_lr
        lr_state["orig_update_lr"] = orig_update_lr

        def patched_update_lr(lr):
            orig_update_lr(lr)
            _apply_actor_lr_control(float(lr))

        agent.update_lr = patched_update_lr

    def _install_optimizer_step_wrapper() -> None:
        orig_optimizer_step = agent.optimizer.step
        lr_state["orig_optimizer_step"] = orig_optimizer_step

        def patched_optimizer_step(*args, **kwargs):
            before = _apply_actor_lr_control()
            result = orig_optimizer_step(*args, **kwargs)
            after = _apply_actor_lr_control()
            if lr_state["step_logs"] < 8:
                lr_state["step_logs"] += 1
                applied = after or before
                if applied is not None:
                    full_lr, factor, actor_lr = applied
                    cap_msg = f" cap={effective_actor_lr_cap:.2e}" if effective_actor_lr_cap is not None else ""
                    print(
                        f"[residual_rl] Actor LR control optimizer-step {lr_state['step_logs']}: "
                        f"epoch={int(agent.epoch_num)} base_lr={full_lr:.2e} "
                        f"factor={factor:.3f} actor_lr={actor_lr:.2e}{cap_msg}.",
                        flush=True,
                    )
            return result

        agent.optimizer.step = patched_optimizer_step

    def _install_actor_lr_control(current_epoch: int) -> None:
        if not lr_controls_requested or lr_state["installed"]:
            return
        _split_optimizer_param_groups()
        idx = lr_state["actor_group_idx"]
        if idx is None:
            print(
                "[residual_rl] WARNING: actor LR control requested but no actor param group "
                "was detected after optimizer split; skipping.",
                flush=True,
            )
            return
        _install_update_lr_wrapper()
        _install_optimizer_step_wrapper()
        lr_state["installed"] = True
        applied = _apply_actor_lr_control()
        if applied is None:
            full_lr = _resolve_base_lr()
            factor = _compute_warmup_factor(current_epoch)
            actor_lr = _target_actor_lr(full_lr, factor)
        else:
            full_lr, factor, actor_lr = applied
        cap_msg = f", actor_lr_cap={effective_actor_lr_cap:.2e}" if effective_actor_lr_cap is not None else ""
        print(
            f"[residual_rl] Actor LR control installed at epoch {current_epoch}: "
            f"warmup_epochs={effective_warmup_epochs}, initial_factor={factor:.3f}, "
            f"base_lr={full_lr:.2e}, initial_actor_lr={actor_lr:.2e}, "
            f"actor_group_idx={idx}{cap_msg}. Control is enforced before every "
            f"optimizer.step(), so rl_games cannot reset actor LR to full last_lr inside a minibatch.",
            flush=True,
        )

    def _install_lr_schedule_freeze(current_epoch: int) -> None:
        if not schedule_freeze_requested or schedule_freeze_state["installed"]:
            return
        scheduler = getattr(agent, "scheduler", None)
        if scheduler is None or not hasattr(scheduler, "update"):
            print(
                "[residual_rl] WARNING: lr_schedule_freeze_during_warmup requested but "
                "agent.scheduler is missing or has no update(); skipping.",
                flush=True,
            )
            schedule_freeze_state["completed"] = True
            return
        pinned_lr = float(getattr(agent, "last_lr", 0.0))
        if pinned_lr <= 0.0:
            pinned_lr = float(_resolve_base_lr() or 0.0)
        schedule_freeze_state["pinned_lr"] = pinned_lr
        orig_update = scheduler.update
        schedule_freeze_state["orig_update"] = orig_update

        def patched_scheduler_update(
            current_lr, entropy_coef, epoch_num, frames, kl_dist, *args, **kwargs
        ):
            schedule_freeze_state["call_count"] += 1
            try:
                kl_val = float(kl_dist)
            except (TypeError, ValueError):
                kl_val = float("nan")
            if (
                kl_val == kl_val  # not NaN
                and kl_val > 1.0
                and schedule_freeze_state["kl_spike_logs"] < 4
            ):
                schedule_freeze_state["kl_spike_logs"] += 1
                print(
                    f"[residual_rl] LR-schedule freeze absorbed spurious KL at "
                    f"epoch={epoch_num} frames={frames}: kl={kl_val:.2e} "
                    f"(would have adapted last_lr; pinned at "
                    f"{schedule_freeze_state['pinned_lr']:.2e}).",
                    flush=True,
                )
            return schedule_freeze_state["pinned_lr"], entropy_coef

        scheduler.update = patched_scheduler_update
        schedule_freeze_state["installed"] = True
        print(
            f"[residual_rl] LR-schedule freeze installed at epoch {current_epoch}: "
            f"pinned_last_lr={pinned_lr:.2e} for the actor LR warmup window "
            f"({effective_warmup_epochs} epochs). The AdaptiveScheduler will "
            f"not react to KL during this interval; it is restored verbatim "
            f"once the warmup completes.",
            flush=True,
        )

    def _maybe_complete_lr_schedule_freeze(current_epoch: int) -> None:
        if (
            not schedule_freeze_state["installed"]
            or schedule_freeze_state["completed"]
        ):
            return
        factor = _compute_warmup_factor(current_epoch)
        if factor < 1.0:
            return
        scheduler = getattr(agent, "scheduler", None)
        if scheduler is not None and schedule_freeze_state["orig_update"] is not None:
            scheduler.update = schedule_freeze_state["orig_update"]
        schedule_freeze_state["orig_update"] = None
        schedule_freeze_state["completed"] = True
        print(
            f"[residual_rl] LR-schedule freeze released at epoch {current_epoch}; "
            f"AdaptiveScheduler control restored (absorbed "
            f"{schedule_freeze_state['call_count']} update() calls during "
            f"warmup).",
            flush=True,
        )

    def _maybe_complete_lr_control(current_epoch: int) -> None:
        if not lr_state["installed"] or lr_state["completed"]:
            return
        factor = _compute_warmup_factor(current_epoch)
        if factor < 1.0:
            return
        lr_state["completed"] = True
        if effective_actor_lr_cap is None:
            if lr_state["orig_update_lr"] is not None:
                agent.update_lr = lr_state["orig_update_lr"]
                lr_state["orig_update_lr"] = None
            if lr_state["orig_optimizer_step"] is not None:
                agent.optimizer.step = lr_state["orig_optimizer_step"]
                lr_state["orig_optimizer_step"] = None
            print(
                f"[residual_rl] Actor LR warmup complete at epoch {current_epoch}; "
                "actor group now tracks the full adaptive LR and wrappers were restored.",
                flush=True,
            )
        else:
            print(
                f"[residual_rl] Actor LR warmup complete at epoch {current_epoch}; "
                f"actor LR cap remains active at {effective_actor_lr_cap:.2e}.",
                flush=True,
            )

    def _ramp_sigma(progress: float) -> None:
        progress = min(max(float(progress), 0.0), 1.0)
        with torch.no_grad():
            if isinstance(sigma_obj, nn.Parameter) and sigma_target_param is not None:
                sigma_obj.data.copy_(
                    float(silenced_sigma_val)
                    + progress * (sigma_target_param - float(silenced_sigma_val))
                )
            elif (
                isinstance(sigma_obj, nn.Linear)
                and sigma_target_state is not None
                and sigma_silenced_state is not None
            ):
                mixed = {}
                for key, target in sigma_target_state.items():
                    silenced = sigma_silenced_state[key]
                    mixed[key] = silenced + progress * (target - silenced)
                sigma_obj.load_state_dict(mixed)

    def _restore_sigma() -> None:
        with torch.no_grad():
            if isinstance(sigma_obj, nn.Parameter) and sigma_target_param is not None:
                sigma_obj.data.copy_(sigma_target_param)
            elif isinstance(sigma_obj, nn.Linear) and sigma_target_state is not None:
                sigma_obj.load_state_dict(sigma_target_state)

    if bc_anchor_model is not None or bc_dataset_active:
        from rl_games.common import common_losses
        from rl_games.algos_torch import torch_ext

        def patched_calc_gradients(input_dict):
            value_preds_batch = input_dict["old_values"]
            old_action_log_probs_batch = input_dict["old_logp_actions"]
            advantage = input_dict["advantages"]
            old_mu_batch = input_dict["mu"]
            old_sigma_batch = input_dict["sigma"]
            return_batch = input_dict["returns"]
            actions_batch = input_dict["actions"]
            obs_batch = input_dict["obs"]
            obs_batch = agent._preproc_obs(obs_batch)

            lr_mul = 1.0
            curr_e_clip = agent.e_clip

            batch_dict = {
                "is_train": True,
                "prev_actions": actions_batch,
                "obs": obs_batch,
            }

            rnn_masks = None
            if agent.is_rnn:
                rnn_masks = input_dict["rnn_masks"]
                batch_dict["rnn_states"] = input_dict["rnn_states"]
                batch_dict["seq_length"] = agent.seq_length
                if agent.zero_rnn_on_done:
                    batch_dict["dones"] = input_dict["dones"]

            current_epoch = int(agent.epoch_num)
            anchor_active = (
                effective_bc_anchor_epochs > 0
                and effective_bc_anchor_coef > 0.0
                and current_epoch >= unfreeze_epoch
                and current_epoch < (unfreeze_epoch + effective_bc_anchor_epochs)
            )

            with torch.cuda.amp.autocast(enabled=agent.mixed_precision):
                res_dict = agent.model(batch_dict)
                action_log_probs = res_dict["prev_neglogp"]
                values = res_dict["values"]
                entropy = res_dict["entropy"]
                mu = res_dict["mus"]
                sigma = res_dict["sigmas"]

                a_loss = agent.actor_loss_func(
                    old_action_log_probs_batch, action_log_probs, advantage, agent.ppo, curr_e_clip
                )

                if agent.has_value_loss:
                    c_loss = common_losses.critic_loss(
                        agent.model, value_preds_batch, values, curr_e_clip, return_batch, agent.clip_value
                    )
                else:
                    c_loss = torch.zeros(1, device=agent.ppo_device)

                if agent.bound_loss_type == "regularisation":
                    b_loss = agent.reg_loss(mu)
                elif agent.bound_loss_type == "bound":
                    b_loss = agent.bound_loss(mu)
                else:
                    b_loss = torch.zeros(1, device=agent.ppo_device)

                losses, _sum_mask = torch_ext.apply_masks(
                    [a_loss.unsqueeze(1), c_loss, entropy.unsqueeze(1), b_loss.unsqueeze(1)], rnn_masks
                )
                a_loss, c_loss, entropy, b_loss = losses[0], losses[1], losses[2], losses[3]

                bc_anchor_loss = torch.zeros(1, device=agent.ppo_device)
                if anchor_active:
                    assert bc_anchor_model is not None  # guarded by anchor_active
                    with torch.no_grad():
                        ref_res_dict = bc_anchor_model(batch_dict)
                        ref_mu = ref_res_dict["mus"]
                    bc_anchor_per_sample = torch.mean((mu - ref_mu) ** 2, dim=-1)
                    bc_anchor_losses, _ = torch_ext.apply_masks([bc_anchor_per_sample.unsqueeze(1)], rnn_masks)
                    bc_anchor_loss = bc_anchor_losses[0]

                # --- BC-on-dataset supervised loss ---------------------------
                # Sample a minibatch from the demonstration HDF5 (pre-loaded to
                # device), normalize with the PINNED BC obs stats (RMS-drift
                # safe), forward through a2c_network.actor_mlp + mu head
                # DIRECTLY — bypassing the live running_mean_std — and compute
                # MSE against dataset actions.  This pulls the policy toward
                # the actual demonstrations, not just toward a frozen copy of
                # the initial policy.  Unlike the rollout-based bc_anchor,
                # whose reference targets reflect whatever bc_anchor_model
                # predicts on out-of-distribution rollout states, this loss is
                # always computed on genuine expert (obs, action) pairs.
                bc_dataset_loss = torch.zeros(1, device=agent.ppo_device)
                bc_dataset_coef_now = _bc_dataset_coef_at(current_epoch)
                if bc_dataset_active and bc_dataset_coef_now > 0.0:
                    obs_all = bc_dataset_state["obs_raw"]
                    act_all = bc_dataset_state["actions_raw"]
                    n_pairs = bc_dataset_state["num_pairs"]
                    idx = torch.randint(
                        low=0,
                        high=n_pairs,
                        size=(effective_bc_dataset_batch_size,),
                        device=obs_all.device,
                    )
                    obs_sample = obs_all.index_select(0, idx)
                    act_sample = act_all.index_select(0, idx)
                    obs_norm = (obs_sample - bc_dataset_state["obs_mean"]) / bc_dataset_state["obs_std"]
                    mu_pred = net.mu(net.actor_mlp(obs_norm))
                    bc_dataset_loss = torch.mean((mu_pred - act_sample) ** 2)

                loss = (
                    a_loss
                    + 0.5 * c_loss * agent.critic_coef
                    - entropy * agent.entropy_coef
                    + b_loss * agent.bounds_loss_coef
                    + effective_bc_anchor_coef * bc_anchor_loss
                    + bc_dataset_coef_now * bc_dataset_loss
                )

                aux_loss = agent.model.get_aux_loss()
                agent.aux_loss_dict = {}
                if aux_loss is not None:
                    for k, v in aux_loss.items():
                        loss += v
                        if k in agent.aux_loss_dict:
                            agent.aux_loss_dict[k].append(v.detach())
                        else:
                            agent.aux_loss_dict[k] = [v.detach()]
                # rl_games write_stats() calls torch_ext.mean_list() on every aux loss entry,
                # so each value must be a list of tensors, not a bare tensor.
                agent.aux_loss_dict["kuafu_bc_anchor"] = [bc_anchor_loss.detach()]
                agent.aux_loss_dict["kuafu_bc_dataset"] = [bc_dataset_loss.detach()]

                # --- Contribution tracking --------------------------------------
                # Log each loss term's WEIGHTED magnitude (what actually enters
                # the sum above) plus BC-dataset's share of the ACTOR-side
                # gradient magnitude.  In RL-Games continuous PPO with separate
                # actor_mlp and critic_mlp (no shared trunk), critic loss only
                # updates critic params — so the meaningful "how much is BC
                # pulling on the actor" number must EXCLUDE the value term.
                # We log both: `bc_dataset_actor_share` is the actionable one
                # for tuning COEF; `bc_dataset_total_share` is kept for
                # completeness / historical comparison but is misleading when
                # c_loss is large (typical with a randomly-initialized critic).
                # All values are |weighted component| for a fair "who owns the
                # update" comparison (a_loss is signed; entropy is subtracted).
                with torch.no_grad():
                    w_policy = a_loss.detach().abs()
                    w_value = (0.5 * c_loss.detach() * agent.critic_coef).abs()
                    w_entropy = (entropy.detach() * agent.entropy_coef).abs()
                    w_bounds = (b_loss.detach() * agent.bounds_loss_coef).abs()
                    w_bc_anchor = (effective_bc_anchor_coef * bc_anchor_loss.detach()).abs()
                    w_bc_dataset = (bc_dataset_coef_now * bc_dataset_loss.detach()).abs()
                    # Actor-side gradient magnitude: everything that touches
                    # actor_mlp / mu / sigma.  Excludes c_loss (critic-only).
                    actor_mag = (
                        w_policy + w_entropy + w_bounds + w_bc_anchor + w_bc_dataset
                    ).clamp_min(1e-12)
                    total_mag = (actor_mag + w_value).clamp_min(1e-12)
                    bc_dataset_actor_share = w_bc_dataset / actor_mag
                    bc_dataset_total_share = w_bc_dataset / total_mag
                agent.aux_loss_dict["kuafu_w_policy"] = [w_policy]
                agent.aux_loss_dict["kuafu_w_value"] = [w_value]
                agent.aux_loss_dict["kuafu_w_entropy"] = [w_entropy]
                agent.aux_loss_dict["kuafu_w_bounds"] = [w_bounds]
                agent.aux_loss_dict["kuafu_w_bc_anchor"] = [w_bc_anchor]
                agent.aux_loss_dict["kuafu_w_bc_dataset"] = [w_bc_dataset]
                agent.aux_loss_dict["kuafu_bc_dataset_coef"] = [
                    torch.tensor(bc_dataset_coef_now, device=agent.ppo_device)
                ]
                agent.aux_loss_dict["kuafu_bc_dataset_actor_share"] = [bc_dataset_actor_share]
                agent.aux_loss_dict["kuafu_bc_dataset_total_share"] = [bc_dataset_total_share]

                if agent.multi_gpu:
                    agent.optimizer.zero_grad()
                else:
                    for param in agent.model.parameters():
                        param.grad = None

            agent.scaler.scale(loss).backward()
            agent.trancate_gradients_and_step()

            with torch.no_grad():
                reduce_kl = rnn_masks is None
                kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
                if rnn_masks is not None:
                    kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()

            if anchor_active and (
                bc_anchor_last_logged[0] < 0
                or current_epoch == unfreeze_epoch
                or current_epoch >= bc_anchor_last_logged[0] + max(1, effective_bc_anchor_epochs // 5)
            ):
                bc_anchor_last_logged[0] = current_epoch
                print(
                    f"[residual_rl] BC anchor epoch {current_epoch}: coef={effective_bc_anchor_coef:.4f} "
                    f"loss={float(bc_anchor_loss.detach().item()):.6f}.",
                    flush=True,
                )

            if bc_dataset_active and bc_dataset_coef_now > 0.0:
                last_logged = bc_dataset_state["last_logged"]
                log_period = max(1, (effective_bc_dataset_decay_epochs or 100) // 10)
                if (
                    last_logged < 0
                    or current_epoch == unfreeze_epoch
                    or current_epoch >= last_logged + log_period
                ):
                    bc_dataset_state["last_logged"] = current_epoch
                    print(
                        f"[residual_rl] BC dataset epoch {current_epoch}: "
                        f"coef={bc_dataset_coef_now:.4f} (base={effective_bc_dataset_coef:.4f}, "
                        f"decay_epochs={effective_bc_dataset_decay_epochs}, "
                        f"batch={effective_bc_dataset_batch_size}) "
                        f"loss={float(bc_dataset_loss.detach().item()):.6f} "
                        f"weighted={float(w_bc_dataset.item()):.4e} "
                        f"actor_share={float(bc_dataset_actor_share.item()):.1%} "
                        f"total_share={float(bc_dataset_total_share.item()):.1%} "
                        f"[actor: policy={float(w_policy.item()):.4e} "
                        f"entropy={float(w_entropy.item()):.4e} "
                        f"bounds={float(w_bounds.item()):.4e} "
                        f"anchor={float(w_bc_anchor.item()):.4e}] "
                        f"[critic: value={float(w_value.item()):.4e}].",
                        flush=True,
                    )

            agent.diagnostics.mini_batch(
                agent,
                {
                    "values": value_preds_batch,
                    "returns": return_batch,
                    "new_neglogp": action_log_probs,
                    "old_neglogp": old_action_log_probs_batch,
                    "masks": rnn_masks,
                },
                curr_e_clip,
                0,
            )

            setattr(agent, "_kuafu_bc_anchor_loss", bc_anchor_loss.detach())
            agent.train_result = (
                a_loss,
                c_loss,
                entropy,
                kl_dist,
                agent.last_lr,
                lr_mul,
                mu.detach(),
                sigma.detach(),
                b_loss,
            )

        agent.calc_gradients = patched_calc_gradients

    orig_train_epoch = agent.train_epoch

    def patched_train_epoch():
        current_epoch = int(agent.epoch_num)

        # Ramp sigma back BEFORE the actor is unfrozen.  This way PPO's first
        # unfrozen step sees a numerically well-conditioned sigma (close to the
        # saved value) and the critic has had the ramp window to re-calibrate V
        # against the noisier rollouts.  Both rollout collection (controlled by
        # the live sigma) and value targets adjust gradually, instead of being
        # hit by a step-function change at the unfreeze boundary.
        if (
            effective_freeze_epochs > 0
            and not unfrozen[0]
            and effective_ramp_epochs > 0
            and (sigma_saved_param is not None or sigma_saved_state is not None)
            and current_epoch >= ramp_start
            and current_epoch < unfreeze_epoch
        ):
            progress = float(current_epoch - ramp_start) / float(effective_ramp_epochs)
            _ramp_sigma(progress)
            if last_ramp_logged[0] != current_epoch:
                last_ramp_logged[0] = current_epoch
                print(
                    f"[residual_rl] Sigma ramp epoch {current_epoch} "
                    f"(ramp_start={ramp_start}, unfreeze_epoch={unfreeze_epoch}): "
                    f"progress={progress:.3f} sigma_mean={_sigma_mean_text()} "
                    "(actor still frozen; critic is re-calibrating V).",
                    flush=True,
                )

        if not unfrozen[0] and current_epoch >= unfreeze_epoch:
            for p in actor_params:
                p.requires_grad_(True)
            for group in agent.optimizer.param_groups:
                for p in group["params"]:
                    if id(p) in actor_param_ids:
                        agent.optimizer.state.pop(p, None)
            if sigma_saved_param is not None or sigma_saved_state is not None:
                _restore_sigma()
                print(f"[residual_rl] Sigma restored to {_sigma_mean_text()} at actor unfreeze.", flush=True)
            if unfreeze_input_rms_at_end:
                if rms_defer_mode:
                    rms_unfreeze_state["target_epoch"] = int(rms_deferred_unfreeze_epoch)
                    print(
                        f"[residual_rl] Observation running_mean_std unfreeze DEFERRED to epoch "
                        f"{rms_deferred_unfreeze_epoch} "
                        f"(actor unfreeze + actor_lr_warmup_epochs={effective_warmup_epochs}). "
                        "Rationale: unfreezing RMS at the actor-unfreeze boundary shifts the "
                        "normalizer during PPO minibatch updates and invalidates the BC mu head "
                        "with weights unchanged, producing a fake KL that crushes the base LR.",
                        flush=True,
                    )
                else:
                    _do_rms_unfreeze(current_epoch, reason="actor_unfreeze")
            unfrozen[0] = True
            _install_actor_lr_control(current_epoch)
            _install_lr_schedule_freeze(current_epoch)
            print(
                f"[residual_rl] Actor unfrozen at epoch {current_epoch} "
                f"(unfreeze_epoch={unfreeze_epoch}; requested full frozen epochs={effective_freeze_epochs}); "
                "Adam state cleared for actor params.",
                flush=True,
            )

        # Deferred RMS unfreeze: fires once current_epoch reaches the scheduled target.
        # Keeps the BC-seeded normalizer pinned throughout the actor LR warmup window so
        # that the actor's warmed-up gradient updates happen against a stable input
        # distribution.  After the actor has had a full warmup of real learning, we let
        # PPO rollouts update the RMS normally.
        if (
            unfrozen[0]
            and unfreeze_input_rms_at_end
            and rms_unfreeze_state["target_epoch"] is not None
            and not rms_unfreeze_state["done"]
            and current_epoch >= int(rms_unfreeze_state["target_epoch"])
        ):
            _do_rms_unfreeze(current_epoch, reason="actor_lr_warmup_end")

        if unfrozen[0] and lr_controls_requested and not lr_state["installed"]:
            _install_actor_lr_control(current_epoch)
        if (
            unfrozen[0]
            and schedule_freeze_requested
            and not schedule_freeze_state["installed"]
            and not schedule_freeze_state["completed"]
        ):
            _install_lr_schedule_freeze(current_epoch)

        if lr_state["installed"]:
            applied = _apply_actor_lr_control()
            if applied is not None:
                full_lr, factor, actor_lr = applied
                if (
                    abs(factor - lr_state["last_factor_logged"]) >= 0.1
                    or lr_state["last_factor_logged"] < 0.0
                    or factor >= 1.0
                ):
                    lr_state["last_factor_logged"] = factor
                    cap_msg = f" cap={effective_actor_lr_cap:.2e}" if effective_actor_lr_cap is not None else ""
                    print(
                        f"[residual_rl] Actor LR control epoch {current_epoch}: "
                        f"factor={factor:.3f} base_lr={full_lr:.2e} actor_lr={actor_lr:.2e}{cap_msg}.",
                        flush=True,
                    )
            _maybe_complete_lr_control(current_epoch)

        if schedule_freeze_state["installed"] and not schedule_freeze_state["completed"]:
            _maybe_complete_lr_schedule_freeze(current_epoch)

        return orig_train_epoch()

    agent.train_epoch = patched_train_epoch

    # Human-readable ramp/restore target for the banner.
    def _sigma_target_text() -> str:
        if sigma_target_value is not None:
            return f"{sigma_target_value:.3f} (overridden via sigma_final_val)"
        if sigma_target_param is not None:
            return f"{float(sigma_target_param.mean().item()):.3f} (checkpoint saved sigma)"
        if sigma_target_state is not None and sigma_target_state.get("bias") is not None:
            return (
                f"{float(sigma_target_state['bias'].mean().item()):.3f} "
                "(checkpoint saved sigma head bias)"
            )
        return "n/a"

    if effective_freeze_epochs > 0:
        if effective_ramp_epochs > 0:
            print(
                f"[residual_rl] Froze {len(actor_params)} actor param tensors for {effective_freeze_epochs} full epochs "
                f"(first actor-trainable epoch={unfreeze_epoch}); sigma ramps from "
                f"{float(silenced_sigma_val):.3f} to {_sigma_target_text()} over the last "
                f"{effective_ramp_epochs} frozen epochs (ramp_start={ramp_start}).",
                flush=True,
            )
        else:
            print(
                f"[residual_rl] Froze {len(actor_params)} actor param tensors for {effective_freeze_epochs} full epochs "
                f"(first actor-trainable epoch={unfreeze_epoch}); sigma restores to {_sigma_target_text()} "
                "at unfreeze (no ramp).",
                flush=True,
            )
    else:
        print(
            "[residual_rl] Actor freeze disabled; prior safety hook will install actor LR controls "
            "at the first train epoch if requested.",
            flush=True,
        )

    if lr_controls_requested:
        cap_msg = f", actor_lr_cap={effective_actor_lr_cap:.2e}" if effective_actor_lr_cap is not None else ""
        print(
            f"[residual_rl] Actor LR control armed: starts at epoch {unfreeze_epoch}, "
            f"warmup_epochs={effective_warmup_epochs}{cap_msg}. Controls are enforced "
            "around every optimizer.step().",
            flush=True,
        )
    if schedule_freeze_requested:
        print(
            f"[residual_rl] LR-schedule freeze armed: starts at epoch {unfreeze_epoch}, "
            f"released at epoch {unfreeze_epoch + effective_warmup_epochs} "
            f"(= actor unfreeze + {effective_warmup_epochs} warmup epochs). "
            "agent.scheduler.update() will be pinned to the pre-unfreeze last_lr "
            "during this window so the AdaptiveScheduler cannot crush the base LR "
            "in response to a one-shot unfreeze-boundary KL spike.",
            flush=True,
        )
    if bc_anchor_model is not None:
        print(
            f"[residual_rl] BC anchor armed: starts at epoch {unfreeze_epoch} for "
            f"{effective_bc_anchor_epochs} epochs with coef={effective_bc_anchor_coef:.4f}.",
            flush=True,
        )
    if bc_dataset_active:
        assert bc_dataset is not None  # guarded by bc_dataset_active
        floor_coef = effective_bc_dataset_coef * 0.1
        if effective_bc_dataset_decay_epochs > 0:
            decay_desc = (
                f"linear decay {effective_bc_dataset_coef:.4f} -> "
                f"{floor_coef:.4f} over {effective_bc_dataset_decay_epochs} epochs, "
                f"then held at floor"
            )
        else:
            decay_desc = f"constant at {effective_bc_dataset_coef:.4f}"
        print(
            f"[residual_rl] BC dataset loss armed: starts at epoch {unfreeze_epoch}, "
            f"always-on afterward; coef={decay_desc}; "
            f"minibatch={effective_bc_dataset_batch_size} sampled per PPO step from "
            f"{bc_dataset_state['num_pairs']} expert (obs, action) pairs "
            f"({bc_dataset['dataset_path']}, obs_key={bc_dataset['obs_key']}, "
            f"obs_history={bc_dataset['obs_history']}, "
            f"action_chunk_size={bc_dataset['action_chunk_size']}). "
            f"Obs-normalization pinned to BC expert stats (RMS-drift safe).",
            flush=True,
        )

def _zero_critic_after_prior(agent) -> None:
    import torch
    import torch.nn as nn

    net = agent.model.a2c_network
    critic_mlp = getattr(net, "critic_mlp", None)
    value_head = getattr(net, "value", None)
    if critic_mlp is None or not isinstance(value_head, nn.Linear):
        raise AttributeError("RL-Games network is missing `critic_mlp` or `value`; cannot zero critic.")

    with torch.no_grad():
        for module in critic_mlp.modules():
            if isinstance(module, nn.Linear):
                module.weight.zero_()
                if module.bias is not None:
                    module.bias.zero_()
        value_head.weight.zero_()
        if value_head.bias is not None:
            value_head.bias.zero_()

        value_rms = getattr(agent.model, "value_mean_std", None)
        if value_rms is not None:
            value_rms.running_mean.zero_()
            value_rms.running_var.fill_(1.0)
            value_rms.count.fill_(1.0)

    print("[residual_rl] Zero-initialized critic trunk, value head, and value normalization stats.", flush=True)


def _resolve_prior_mode(agent_cfg: dict, prior_mode: str | None) -> str:
    mode = str(prior_mode or "full_actor").strip() or "full_actor"
    valid_modes = {"full_actor", "residual_rl"}
    if mode not in valid_modes:
        raise ValueError(f"Unsupported prior_mode={mode!r}. Expected one of {sorted(valid_modes)}.")
    residual_cfg = agent_cfg.get("params", {}).get("config", {}).get("residual_rl", {})
    if mode == "full_actor" and isinstance(residual_cfg, dict) and residual_cfg.get("enabled", False):
        return "residual_rl"
    return mode


def _configure_residual_rl_agent_cfg(agent_cfg: dict, residual_cfg, *, fixed_lr_override: float | None = None) -> None:
    network_cfg = agent_cfg["params"]["network"]
    config_cfg = agent_cfg["params"]["config"]
    network_cfg["separate"] = True
    continuous_cfg = network_cfg["space"]["continuous"]
    continuous_cfg["fixed_sigma"] = False
    continuous_cfg.setdefault("sigma_init", {})
    continuous_cfg["sigma_init"]["name"] = "const_initializer"
    continuous_cfg["sigma_init"]["val"] = float(residual_cfg.residual_logstd_init)
    config_cfg["e_clip"] = float(residual_cfg.e_clip)

    fixed_lr = fixed_lr_override
    if fixed_lr is None:
        fixed_lr = residual_cfg.fixed_lr
    if fixed_lr is not None and fixed_lr > 0.0:
        prev_lr = config_cfg.get("learning_rate", None)
        prev_sched = config_cfg.get("lr_schedule", None)
        config_cfg["learning_rate"] = float(fixed_lr)
        config_cfg["lr_schedule"] = "identity"
        print(
            f"[residual_rl] Residual-RL fixed LR preflight: learning_rate {prev_lr} -> {fixed_lr:.2e}, "
            f"lr_schedule {prev_sched!r} -> 'identity'.",
            flush=True,
        )
    elif not residual_cfg.use_adaptive_lr:
        config_cfg["lr_schedule"] = "identity"
        print(
            f"[residual_rl] Residual-RL disabled adaptive LR without overriding learning_rate "
            f"(using {config_cfg.get('learning_rate')}).",
            flush=True,
        )


def _make_kuafu_residual_rl_games_wrapper(
    env,
    rl_device: str,
    clip_obs: float,
    clip_actions: float,
    residual_cfg,
    obs_groups: dict[str, list[str]] | None = None,
    concate_obs_group: bool = True,
):
    import gym.spaces
    import torch

    from isaaclab_rl.rl_games import RlGamesVecEnvWrapper

    class KuafuResidualRlGamesVecEnvWrapper(RlGamesVecEnvWrapper):
        def __init__(self, *args, **kwargs):
            self._action_controller = None
            self._latest_policy_obs: torch.Tensor | None = None
            self._rollout_reward_epoch_key: int | None = None
            self._rollout_reward_sum: torch.Tensor | None = None
            self._rollout_reward_count = 0
            window_steps_raw = os.getenv("KUAFU_ROLLOUT_REWARD_WINDOW_STEPS")
            window_steps = (
                int(window_steps_raw)
                if window_steps_raw not in (None, "")
                else int(getattr(residual_cfg, "rollout_reward_window_steps", 100) or 100)
            )
            self._rollout_reward_window_steps = max(1, window_steps)
            self._rollout_reward_window_sum: torch.Tensor | None = None
            self._rollout_reward_window_values: list[torch.Tensor] = []
            self._residual_use_stage_conditioning = bool(getattr(residual_cfg, "use_stage_conditioning", False))
            self._configured_stage_dim = (
                int(getattr(residual_cfg, "num_stages", 0)) if self._residual_use_stage_conditioning else 0
            )
            self._residual_mask_stage_observation = bool(
                getattr(residual_cfg, "mask_stage_observation", False)
            )
            super().__init__(*args, **kwargs)

        @property
        def observation_space(self):
            space = super().observation_space
            stage_dim = self._residual_stage_dim
            if stage_dim <= 0:
                return space
            if isinstance(space, gym.spaces.Box):
                if len(space.shape) != 1:
                    raise NotImplementedError(
                        "Residual stage conditioning currently requires flat actor observation spaces."
                    )
                return gym.spaces.Box(-self._clip_obs, self._clip_obs, (space.shape[0] + stage_dim,))
            if isinstance(space, gym.spaces.Dict):
                if len(space.spaces) != 1:
                    raise NotImplementedError(
                        "Residual action controller currently requires concatenated observations or a single actor "
                        "observation group."
                    )
                group_name, group_space = next(iter(space.spaces.items()))
                if not isinstance(group_space, gym.spaces.Box) or len(group_space.shape) != 1:
                    raise NotImplementedError(
                        "Residual stage conditioning currently requires flat actor observation spaces."
                    )
                new_spaces = dict(space.spaces)
                new_spaces[group_name] = gym.spaces.Box(
                    -self._clip_obs, self._clip_obs, (group_space.shape[0] + stage_dim,)
                )
                return gym.spaces.Dict(new_spaces)
            return space

        def set_action_controller(self, controller) -> None:
            self._action_controller = controller

        def _rollout_reward_diagnostics(self, rew: torch.Tensor) -> dict[str, torch.Tensor]:
            epoch_key_raw = getattr(self.unwrapped, "_Kuafu_current_train_epoch", None)
            epoch_key = int(epoch_key_raw) if epoch_key_raw is not None else None
            if (
                self._rollout_reward_sum is None
                or self._rollout_reward_sum.device != rew.device
                or self._rollout_reward_epoch_key != epoch_key
            ):
                self._rollout_reward_epoch_key = epoch_key
                self._rollout_reward_sum = torch.zeros((), device=rew.device, dtype=torch.float32)
                self._rollout_reward_count = 0

            step_mean = rew.detach().to(dtype=torch.float32).mean()
            self._rollout_reward_sum = self._rollout_reward_sum + step_mean
            self._rollout_reward_count += 1
            epoch_step_mean = self._rollout_reward_sum / max(self._rollout_reward_count, 1)

            if self._rollout_reward_window_sum is None or self._rollout_reward_window_sum.device != rew.device:
                self._rollout_reward_window_sum = torch.zeros((), device=rew.device, dtype=torch.float32)
                self._rollout_reward_window_values = []
            window_value = step_mean.detach()
            self._rollout_reward_window_values.append(window_value)
            self._rollout_reward_window_sum = self._rollout_reward_window_sum + window_value
            while len(self._rollout_reward_window_values) > self._rollout_reward_window_steps:
                self._rollout_reward_window_sum = self._rollout_reward_window_sum - self._rollout_reward_window_values.pop(0)
            window_step_mean = self._rollout_reward_window_sum / max(len(self._rollout_reward_window_values), 1)
            return {
                "Kuafu/rollout_step_reward_mean": step_mean,
                "Kuafu/rollout_step_reward_mean_window": window_step_mean.detach(),
                "Kuafu/rollout_epoch_step_reward_mean": epoch_step_mean.detach(),
            }

        def step(self, actions):
            actions = actions.detach().clone().to(device=self._sim_device)
            if self._action_controller is not None:
                if self._latest_policy_obs is None:
                    raise RuntimeError(
                        "Residual action controller was armed before wrapper.reset() produced observations."
                    )
                actions = self._action_controller.compose_env_action(
                    raw_residual_action=actions,
                    bc_obs=self._latest_policy_obs,
                    clip_actions=self._clip_actions,
                )
            else:
                actions = torch.clamp(actions, -self._clip_actions, self._clip_actions)
            obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
            if not self.unwrapped.cfg.is_finite_horizon:
                extras["time_outs"] = truncated.to(device=self._rl_device)
            obs_and_states = self._process_obs(obs_dict)
            rew = rew.to(device=self._rl_device)
            dones = (terminated | truncated).to(device=self._rl_device)
            extras = {
                k: v.to(device=self._rl_device, non_blocking=True) if hasattr(v, "to") else v for k, v in extras.items()
            }
            extras.update(self._rollout_reward_diagnostics(rew))
            if "log" in extras:
                extras["episode"] = extras.pop("log")
            return obs_and_states, rew, dones, extras

        def _process_obs(self, obs_dict):
            if self._rl_device != self._sim_device:
                obs_dict = {key: obs.to(device=self._rl_device) for key, obs in obs_dict.items()}
            for key, obs in obs_dict.items():
                obs_dict[key] = torch.clamp(obs, -self._clip_obs, self._clip_obs)

            rl_games_obs: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
                "obs": {group: obs_dict[group] for group in self._obs_groups["obs"]}
            }
            policy_obs = None
            if self._concate_obs_groups:
                policy_obs = self._obs_concat_fn(list(rl_games_obs["obs"].values()))
            elif self._action_controller is not None:
                if len(rl_games_obs["obs"]) != 1:
                    raise NotImplementedError(
                        "Residual action controller currently requires concatenated observations or a single actor "
                        "observation group."
                    )
                policy_obs = next(iter(rl_games_obs["obs"].values()))
            if policy_obs is not None and self._residual_mask_stage_observation:
                if policy_obs.ndim != 2:
                    raise NotImplementedError(
                        "Residual stage-observation masking currently requires rank-2 flat actor observations."
                    )
                policy_obs = policy_obs.clone()
                policy_obs[:, -1] = 0.0
                if not self._concate_obs_groups:
                    obs_groups_dict = rl_games_obs["obs"]
                    if isinstance(obs_groups_dict, dict):
                        only_obs_group = next(iter(obs_groups_dict))
                        obs_groups_dict[only_obs_group] = policy_obs
            self._latest_policy_obs = policy_obs
            if self._residual_stage_dim > 0:
                if policy_obs is None:
                    raise RuntimeError("Residual action controller could not resolve the actor observation tensor.")
                if self._action_controller is not None:
                    policy_obs = self._action_controller.augment_observation(policy_obs, self._get_stage_tensor())
                else:
                    policy_obs = torch.cat([policy_obs, self._make_stage_one_hot(policy_obs)], dim=-1)
            if len(self._obs_groups["states"]) > 0:
                rl_games_obs["states"] = {group: obs_dict[group] for group in self._obs_groups["states"]}

            if self._concate_obs_groups:
                assert policy_obs is not None
                rl_games_obs["obs"] = policy_obs
                if "states" in rl_games_obs:
                    rl_games_obs["states"] = self._states_concat_fn(list(rl_games_obs["states"].values()))
            elif self._action_controller is not None:
                assert policy_obs is not None
                obs_groups_dict = cast(dict[str, torch.Tensor], rl_games_obs["obs"])
                only_obs_group = next(iter(obs_groups_dict))
                rl_games_obs["obs"] = {only_obs_group: policy_obs}

            return rl_games_obs

        @property
        def _residual_stage_dim(self) -> int:
            return int(getattr(self, "_configured_stage_dim", 0))

        def _get_stage_tensor(self) -> torch.Tensor:
            stage_tensor = getattr(self.unwrapped, "current_stage", None)
            if stage_tensor is None:
                return torch.zeros((self.num_envs, 1), device=self._rl_device, dtype=torch.float32)
            return stage_tensor.to(device=self._rl_device, dtype=torch.float32)

        def _make_stage_one_hot(self, policy_obs: torch.Tensor) -> torch.Tensor:
            stage_tensor = self._get_stage_tensor().to(device=policy_obs.device, dtype=torch.long).view(-1)
            stage_tensor = stage_tensor.clamp_(0, max(self._residual_stage_dim - 1, 0))
            out = torch.zeros(
                (stage_tensor.shape[0], self._residual_stage_dim),
                device=policy_obs.device,
                dtype=policy_obs.dtype,
            )
            if self._residual_stage_dim > 0:
                out.scatter_(1, stage_tensor.unsqueeze(1), 1.0)
            return out

    return KuafuResidualRlGamesVecEnvWrapper(
        env,
        rl_device,
        clip_obs,
        clip_actions,
        obs_groups,
        concate_obs_group,
    )


def _resolve_env_num_stages(env) -> int:
    """Return the task's stage count from the env's program_runtime/compiled_program.

    residual_rl.num_stages must always come from the env so that the one-hot stage
    conditioning concatenated to the observation has the correct width. Trusting a
    yaml-hardcoded value silently corrupts conditioning whenever the yaml drifts from
    the actual task (or whenever the same yaml gets reused across tasks).
    """
    raw = env.unwrapped if hasattr(env, "unwrapped") else env
    program_runtime = getattr(raw, "program_runtime", None)
    if program_runtime is None:
        raise RuntimeError(
            "[residual_rl] residual_rl requires env.unwrapped.program_runtime, but none is attached. "
            "Stage count cannot be resolved."
        )
    compiled_program = getattr(program_runtime, "compiled_program", None)
    if compiled_program is None or not hasattr(compiled_program, "stage_count"):
        raise RuntimeError(
            "[residual_rl] residual_rl requires program_runtime.compiled_program.stage_count(); "
            "none available on this env."
        )
    n = int(compiled_program.stage_count())
    if n <= 0:
        raise RuntimeError(f"[residual_rl] residual_rl: env reports non-positive stage_count={n}.")
    return n


def _make_bc_action_reward_wrapper(
    env,
    rl_device: str,
    clip_obs: float,
    clip_actions: float,
    prior_checkpoint: str,
    bc_action_reward_coef: float,
    obs_groups: dict[str, list[str]] | None = None,
    concate_obs_group: bool = True,
):
    """Wrap env so every step subtracts `coef * ||a - a_BC(obs)||²` from the reward.

    This is a reward-level BC proximity constraint: the penalty flows through GAE into
    the value function, making it fundamentally harder for PPO to forget BC than a
    gradient-level anchor loss alone.  The BC prior is frozen for the entire run.
    """
    import torch
    from isaaclab_rl.rl_games import RlGamesVecEnvWrapper
    from kuafu.bc.training.residual_rl import FrozenBCPrior

    action_dim = int(env.unwrapped.single_action_space.shape[0])
    bc_prior = FrozenBCPrior.from_checkpoint(
        prior_checkpoint,
        action_dim=action_dim,
        use_bc_rms=True,
        device=rl_device,
    )
    bc_prior.eval()
    coef = float(bc_action_reward_coef)

    class KuafuBCRewardVecEnvWrapper(RlGamesVecEnvWrapper):
        def __init__(self, *args, **kwargs):
            self._latest_policy_obs: torch.Tensor | None = None
            super().__init__(*args, **kwargs)

        def _capture_obs(self, obs_or_dict):
            if isinstance(obs_or_dict, dict):
                obs_tensor = obs_or_dict.get("obs", None)
                if isinstance(obs_tensor, dict):
                    obs_tensor = next(iter(obs_tensor.values()), None)
            else:
                obs_tensor = obs_or_dict
            if isinstance(obs_tensor, torch.Tensor):
                self._latest_policy_obs = obs_tensor

        def reset(self):
            result = super().reset()
            self._capture_obs(result)
            return result

        def step(self, actions):
            obs_and_states, rew, dones, extras = super().step(actions)
            if self._latest_policy_obs is not None and coef > 0.0:
                with torch.no_grad():
                    bc_action = bc_prior.get_mean(self._latest_policy_obs)
                    a = actions.to(device=rl_device).clamp(-self._clip_actions, self._clip_actions)
                    penalty = coef * (a - bc_action).pow(2).mean(dim=-1)
                    rew = rew - penalty.to(device=rew.device)
            self._capture_obs(obs_and_states)
            return obs_and_states, rew, dones, extras

    print(
        f"[residual_rl] BC-action reward penalty armed: coef={coef:.4f} "
        f"action_dim={action_dim}.",
        flush=True,
    )
    return KuafuBCRewardVecEnvWrapper(
        env,
        rl_device,
        clip_obs,
        clip_actions,
        obs_groups,
        concate_obs_group,
    )


def _make_residual_action_controller(
    *,
    prior_checkpoint: str,
    residual_cfg,
    rl_env,
):
    from kuafu.bc.training.residual_rl import FrozenBCPrior, ResidualActionController

    action_dim = int(rl_env.unwrapped.single_action_space.shape[0])
    bc_prior = FrozenBCPrior.from_checkpoint(
        prior_checkpoint,
        action_dim=action_dim,
        use_bc_rms=bool(residual_cfg.use_bc_rms),
        device=rl_env.device,
    )
    controller = ResidualActionController(
        cfg=residual_cfg,
        bc_prior=bc_prior,
        action_dim=action_dim,
        sim_device=rl_env.device,
    )
    rl_env.set_action_controller(controller)
    print(
        f"[residual_rl] Residual RL controller armed: action_dim={action_dim} "
        f"num_stages={residual_cfg.num_stages} bc_rms={residual_cfg.use_bc_rms} "
        f"alpha=[{residual_cfg.alpha_init:.3f}->{residual_cfg.alpha_final:.3f}] "
        f"critic_warmup_epochs={residual_cfg.critic_warmup_epochs} "
        f"alpha_warmup_epochs={residual_cfg.alpha_warmup_epochs} "
        f"logstd_clamp=[{residual_cfg.residual_logstd_min},{residual_cfg.residual_logstd_max}] "
        f"clip_composed_actions={residual_cfg.clip_composed_actions} "
        f"mask_stage_observation={getattr(residual_cfg, 'mask_stage_observation', False)}.",
        flush=True,
    )
    return controller


def _install_residual_rl_hooks(agent, raw_env, controller, residual_cfg) -> None:
    import torch
    from rl_games.algos_torch import torch_ext
    from rl_games.common import common_losses

    from kuafu.bc.training.residual_rl import (
        initialize_residual_logstd,
        install_residual_logstd_clamp,
        set_residual_mean_trainable,
        zero_residual_mean_head,
    )

    zero_residual_mean_head(agent)
    initialize_residual_logstd(agent, float(residual_cfg.residual_logstd_init))
    logstd_clamp_handle = install_residual_logstd_clamp(
        agent,
        logstd_min=residual_cfg.residual_logstd_min,
        logstd_max=residual_cfg.residual_logstd_max,
    )
    if logstd_clamp_handle is not None:
        print(
            f"[residual_rl] Residual logstd clamp installed: "
            f"min={residual_cfg.residual_logstd_min} max={residual_cfg.residual_logstd_max}.",
            flush=True,
        )
    set_residual_mean_trainable(agent, False)

    orig_train_epoch = agent.train_epoch
    warmup_state = {"trainable": False}

    # Rolling-window per-stage completion rate.
    # _stage_advanced_cumul[N] and _episodes_started_cumul never reset, so we can safely diff them.
    _window_max = 10  # epoch buckets
    _sc_state: dict = {
        "initialized": False,
        "prev_advanced": {},   # stage -> int
        "prev_eps_started": 0,
        "rings": {},           # stage -> list of (delta_advanced, delta_entered)
        "ring_sums": {},       # stage -> [sum_advanced, sum_entered]
        "last_epoch_rates": {},  # stage -> float
    }

    def _stage_completion_window_tensors(device, stage_count):
        if not hasattr(raw_env, "get_stage_completion_cumulative"):
            return {}
        advanced_cumul, eps_started = raw_env.get_stage_completion_cumulative()

        if not _sc_state["initialized"]:
            _sc_state["prev_advanced"] = {s: advanced_cumul.get(s, 0) for s in range(stage_count)}
            _sc_state["prev_eps_started"] = eps_started
            for s in range(stage_count):
                _sc_state["rings"][s] = []
                _sc_state["ring_sums"][s] = [0, 0]
                _sc_state["last_epoch_rates"][s] = 0.0
            _sc_state["initialized"] = True
            return {}

        # Compute per-stage deltas from monotonically-increasing cumulative counters.
        delta_adv: dict[int, int] = {}
        for s in range(stage_count):
            curr = advanced_cumul.get(s, 0)
            prev = _sc_state["prev_advanced"].get(s, 0)
            delta_adv[s] = max(0, curr - prev)
            _sc_state["prev_advanced"][s] = curr

        delta_eps = max(0, eps_started - _sc_state["prev_eps_started"])
        _sc_state["prev_eps_started"] = eps_started

        result: dict[int, torch.Tensor] = {}
        for s in range(stage_count):
            delta_entered = delta_eps if s == 0 else delta_adv[s - 1]
            delta_advanced = delta_adv[s]

            if delta_entered > 0:
                rate = delta_advanced / float(delta_entered)
                _sc_state["last_epoch_rates"][s] = rate
                ring = _sc_state["rings"][s]
                ring.append((delta_advanced, delta_entered))
                _sc_state["ring_sums"][s][0] += delta_advanced
                _sc_state["ring_sums"][s][1] += delta_entered
                while len(ring) > _window_max:
                    old_adv, old_ent = ring.pop(0)
                    _sc_state["ring_sums"][s][0] -= old_adv
                    _sc_state["ring_sums"][s][1] -= old_ent

            sum_adv, sum_ent = _sc_state["ring_sums"][s]
            window_rate = sum_adv / float(sum_ent) if sum_ent > 0 else 0.0
            result[s] = torch.tensor(window_rate, dtype=torch.float32, device=device)

        return result

    def patched_calc_gradients(input_dict):
        value_preds_batch = input_dict["old_values"]
        old_action_log_probs_batch = input_dict["old_logp_actions"]
        advantage = input_dict["advantages"]
        if residual_cfg.advantage_clip is not None:
            advantage = advantage.clamp(
                -float(residual_cfg.advantage_clip),
                float(residual_cfg.advantage_clip),
            )
        old_mu_batch = input_dict["mu"]
        old_sigma_batch = input_dict["sigma"]
        return_batch = input_dict["returns"]
        actions_batch = input_dict["actions"]
        obs_batch = agent._preproc_obs(input_dict["obs"])

        lr_mul = 1.0
        curr_e_clip = agent.e_clip
        batch_dict = {"is_train": True, "prev_actions": actions_batch, "obs": obs_batch}

        rnn_masks = None
        if agent.is_rnn:
            rnn_masks = input_dict["rnn_masks"]
            batch_dict["rnn_states"] = input_dict["rnn_states"]
            batch_dict["seq_length"] = agent.seq_length
            if agent.zero_rnn_on_done:
                batch_dict["dones"] = input_dict["dones"]

        with torch.cuda.amp.autocast(enabled=agent.mixed_precision):
            res_dict = agent.model(batch_dict)
            action_log_probs = res_dict["prev_neglogp"]
            values = res_dict["values"]
            entropy = res_dict["entropy"]
            mu = res_dict["mus"]
            sigma = res_dict["sigmas"]

            a_loss = agent.actor_loss_func(
                old_action_log_probs_batch, action_log_probs, advantage, agent.ppo, curr_e_clip
            )
            if agent.has_value_loss:
                c_loss = common_losses.critic_loss(
                    agent.model, value_preds_batch, values, curr_e_clip, return_batch, agent.clip_value
                )
            else:
                c_loss = torch.zeros(1, device=agent.ppo_device)

            if agent.bound_loss_type == "regularisation":
                b_loss = agent.reg_loss(mu)
            elif agent.bound_loss_type == "bound":
                b_loss = agent.bound_loss(mu)
            else:
                b_loss = torch.zeros(1, device=agent.ppo_device)

            residual_per_sample = torch.mean(controller.bound_residual(mu) ** 2, dim=-1)
            losses, _sum_mask = torch_ext.apply_masks(
                [a_loss.unsqueeze(1), c_loss, entropy.unsqueeze(1), b_loss.unsqueeze(1), residual_per_sample.unsqueeze(1)],
                rnn_masks,
            )
            a_loss, c_loss, entropy, b_loss, residual_l2 = losses
            loss = (
                a_loss
                + 0.5 * c_loss * agent.critic_coef
                - entropy * agent.entropy_coef
                + b_loss * agent.bounds_loss_coef
                + float(residual_cfg.residual_l2_coef) * residual_l2
            )

            aux_loss = agent.model.get_aux_loss()
            agent.aux_loss_dict = {}
            if aux_loss is not None:
                for k, v in aux_loss.items():
                    loss += v
                    agent.aux_loss_dict.setdefault(k, []).append(v.detach())

            bounded_mu = controller.bound_residual(mu.detach())
            delta_mu = controller.scaled_delta(mu.detach())
            sigma_detached = sigma.detach().clamp_min(1e-8)
            logstd = sigma_detached.log()
            controller_stats = controller.latest_stats(device=agent.ppo_device)
            stage_tensor = getattr(raw_env, "current_stage", None)
            if stage_tensor is not None:
                avg_stage = stage_tensor.float().mean().to(device=agent.ppo_device)
                max_stage = stage_tensor.float().max().to(device=agent.ppo_device)
            else:
                avg_stage = torch.zeros(1, device=agent.ppo_device)
                max_stage = torch.zeros(1, device=agent.ppo_device)

            ret_var = torch.var(return_batch.detach())
            explained_variance = torch.zeros(1, device=agent.ppo_device)
            if torch.isfinite(ret_var) and float(ret_var.item()) > 1e-8:
                explained_variance = 1.0 - torch.var(return_batch.detach() - values.detach()) / ret_var

            agent.aux_loss_dict["kuafu_residual_alpha"] = [
                torch.tensor(float(controller.alpha), dtype=torch.float32, device=agent.ppo_device)
            ]
            agent.aux_loss_dict["kuafu_residual_mean_magnitude"] = [bounded_mu.abs().mean()]
            agent.aux_loss_dict["kuafu_delta_from_bc_magnitude"] = [delta_mu.abs().mean()]
            agent.aux_loss_dict["kuafu_residual_l2"] = [residual_l2.detach()]
            agent.aux_loss_dict["kuafu_logstd_mean"] = [logstd.mean()]
            agent.aux_loss_dict["kuafu_logstd_min"] = [logstd.min()]
            agent.aux_loss_dict["kuafu_logstd_max"] = [logstd.max()]
            agent.aux_loss_dict["kuafu_avg_stage"] = [avg_stage]
            agent.aux_loss_dict["kuafu_max_stage"] = [max_stage]
            agent.aux_loss_dict["kuafu_value_loss"] = [c_loss.detach()]
            agent.aux_loss_dict["kuafu_entropy"] = [entropy.detach()]
            agent.aux_loss_dict["kuafu_explained_variance"] = [explained_variance.detach()]
            for key, value in controller_stats.items():
                agent.aux_loss_dict[f"kuafu_{key}"] = [value]

            if hasattr(raw_env, "get_highest_stage_reached_per_env"):
                highest = raw_env.get_highest_stage_reached_per_env()
                if highest:
                    highest_t = torch.tensor(highest, dtype=torch.float32, device=agent.ppo_device)
                    stage_count = int(getattr(residual_cfg, "num_stages", 0))
                    sc_window = _stage_completion_window_tensors(agent.ppo_device, stage_count)
                    for stage_idx in range(stage_count):
                        entered = (highest_t >= float(stage_idx)).float().sum()
                        advanced = (highest_t > float(stage_idx)).float().sum()
                        agent.aux_loss_dict[f"kuafu_stage_completion_{stage_idx}"] = [
                            advanced / entered.clamp(min=1.0)
                        ]
                        if stage_idx in sc_window:
                            agent.aux_loss_dict[f"kuafu_stage_completion_{stage_idx}_window"] = [
                                sc_window[stage_idx]
                            ]

            if agent.multi_gpu:
                agent.optimizer.zero_grad()
            else:
                for param in agent.model.parameters():
                    param.grad = None

        agent.scaler.scale(loss).backward()
        agent.trancate_gradients_and_step()

        with torch.no_grad():
            reduce_kl = rnn_masks is None
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if rnn_masks is not None:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()
        agent.aux_loss_dict["kuafu_approx_kl"] = [kl_dist.detach()]

        agent.diagnostics.mini_batch(
            agent,
            {
                "values": value_preds_batch,
                "returns": return_batch,
                "new_neglogp": action_log_probs,
                "old_neglogp": old_action_log_probs_batch,
                "masks": rnn_masks,
            },
            curr_e_clip,
            0,
        )
        agent.train_result = (
            a_loss,
            c_loss,
            entropy,
            kl_dist,
            agent.last_lr,
            lr_mul,
            mu.detach(),
            sigma.detach(),
            b_loss,
        )

    def patched_train_epoch():
        current_epoch = int(agent.epoch_num)
        setattr(raw_env, "_Kuafu_current_train_epoch", current_epoch)
        controller.set_epoch(current_epoch)
        should_train = current_epoch > int(residual_cfg.critic_warmup_epochs)
        if should_train != warmup_state["trainable"]:
            set_residual_mean_trainable(agent, should_train)
            warmup_state["trainable"] = should_train
            print(
                f"[residual_rl] Residual RL actor-mean {'unfrozen' if should_train else 'frozen'} at epoch {current_epoch}; "
                f"alpha={controller.alpha:.3f}.",
                flush=True,
            )
        return orig_train_epoch()

    agent.calc_gradients = patched_calc_gradients
    agent.train_epoch = patched_train_epoch


def train_rl_games(
    *,
    env,
    task: str,
    device: str,
    max_train_epoch: int,
    worker_idx: int,
    seed: int,
    resume_from_checkpoint: str | None = None,
    prior_checkpoint: str | None = None,
    prior_mode: str = "full_actor",
    prior_obs_stats_count: float = 1e6,
    prior_verbose: bool = False,
    prior_critic_init_mode: str = "actor_only_random_critic",
    prior_freeze_actor_epochs: int = 0,
    prior_sigma_ramp_epochs: int = 0,
    prior_sigma_final_val: float | None = None,
    prior_actor_lr_warmup_epochs: int = 0,
    prior_actor_lr_cap: float | None = 0.0,
    prior_lr_schedule_freeze_during_warmup: bool = False,
    prior_fixed_lr: float | None = None,
    prior_freeze_input_running_mean_std: bool = False,
    prior_unfreeze_input_running_mean_std_after_freeze: bool = False,
    prior_rms_unfreeze_after_actor_lr_warmup: bool = False,
    prior_bc_anchor_epochs: int = 0,
    prior_bc_anchor_coef: float = 0.0,
    prior_bc_dataset_coef: float = 0.0,
    prior_bc_dataset_decay_epochs: int = 0,
    prior_bc_dataset_batch_size: int = 1024,
    prior_bc_dataset_path: str | None = None,
    prior_bc_action_reward_coef: float = 0.0,
    kuafu_run_dir: str | None = None,
    agent_cfg: dict | None = None,
) -> tuple[str, str | None]:
    """Train with RL-Games and return (tensorboard_log_dir, checkpoint_path)."""
    from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
    from rl_games.common import env_configurations, vecenv
    from rl_games.common.algo_observer import IsaacAlgoObserver
    from rl_games.torch_runner import Runner

    if agent_cfg is None:
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

        agent_cfg = load_cfg_from_registry(task, "rl_games_cfg_entry_point")
    assert agent_cfg is not None
    prior_mode = _resolve_prior_mode(agent_cfg, prior_mode)
    residual_cfg = None
    action_dim = int(env.unwrapped.single_action_space.shape[0])
    if prior_mode == "residual_rl":
        from kuafu.bc.training.residual_rl import ResidualRlConfig

        residual_mapping = copy.deepcopy(
            agent_cfg.get("params", {}).get("config", {}).get("residual_rl", {})
        )
        if isinstance(residual_mapping, dict):
            residual_mapping["enabled"] = True
        env_num_stages = _resolve_env_num_stages(env)
        residual_cfg = ResidualRlConfig.from_mapping(
            residual_mapping,
            action_dim=action_dim,
            env_num_stages=env_num_stages,
        )
        if prior_checkpoint is None:
            raise ValueError("Residual RL mode requires prior_checkpoint so the frozen BC base policy is available.")
    elif resume_from_checkpoint and prior_checkpoint:
        raise ValueError("resume_from_checkpoint and prior_checkpoint are mutually exclusive outside residual_rl mode.")
    allowed_prior_modes = {
        "actor_only_random_critic",
        "actor_and_critic_prior",
        "actor_only_zero_critic",
    }
    if prior_critic_init_mode not in allowed_prior_modes:
        raise ValueError(
            f"Unsupported prior_critic_init_mode={prior_critic_init_mode!r}. "
            f"Expected one of {sorted(allowed_prior_modes)}."
        )

    # RL-Games restores epoch_num from the checkpoint, so max_epochs must
    # account for the already-elapsed epochs to actually train further.
    effective_max_epochs = int(max_train_epoch)
    if resume_from_checkpoint:
        ckpt_epoch = _checkpoint_epoch(resume_from_checkpoint)
        effective_max_epochs = ckpt_epoch + int(max_train_epoch)
        print(
            f"[residual_rl] Checkpoint epoch={ckpt_epoch}, "
            f"requesting {max_train_epoch} more → max_epochs={effective_max_epochs}",
            flush=True,
        )

    agent_cfg["params"]["config"]["max_epochs"] = effective_max_epochs
    agent_cfg["params"]["config"]["device"] = str(device)
    agent_cfg["params"]["config"]["device_name"] = str(device)
    agent_cfg["params"]["seed"] = int(seed)
    try:
        epoch_steps = int(agent_cfg["params"]["config"].get("horizon_length", 0) or 0)
    except (TypeError, ValueError):
        epoch_steps = 0
    if epoch_steps > 0:
        setattr(env.unwrapped, "_Kuafu_train_epoch_steps", epoch_steps)
    if resume_from_checkpoint:
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = str(resume_from_checkpoint)

    if residual_cfg is not None:
        _configure_residual_rl_agent_cfg(agent_cfg, residual_cfg)
    else:
        # Fixed-LR preflight: if the caller pinned an LR, rewrite the rl_games YAML
        # *before* runner.load() so the agent is built with an identity scheduler and
        # the correct initial LR from the start, instead of being constructed with
        # adaptive+1e-2 and then rewritten at runtime.  The runtime patch in
        # `_freeze_actor_for_epochs` is still applied as a belt-and-suspenders safety
        # net (in case any rl_games version re-reads the original YAML values).
        if prior_fixed_lr is not None:
            try:
                _fixed_lr_preflight = float(prior_fixed_lr)
            except (TypeError, ValueError):
                _fixed_lr_preflight = None
            if _fixed_lr_preflight is not None and _fixed_lr_preflight > 0.0:
                _rl_cfg = agent_cfg["params"]["config"]
                _prev_lr = _rl_cfg.get("learning_rate", None)
                _prev_sched = _rl_cfg.get("lr_schedule", None)
                _rl_cfg["learning_rate"] = _fixed_lr_preflight
                # rl_games treats any lr_schedule value other than 'adaptive' / 'linear'
                # as identity (no-op).  Use a descriptive sentinel so logs are clear.
                _rl_cfg["lr_schedule"] = "identity"
                print(
                    f"[residual_rl] Fixed-LR preflight: rewrote agent_cfg "
                    f"(learning_rate: {_prev_lr} -> {_fixed_lr_preflight:.2e}, "
                    f"lr_schedule: {_prev_sched!r} -> 'identity') "
                    f"so rl_games builds the agent with the final LR + no adaptive scheduler.",
                    flush=True,
                )

    # Logs: co-locate RL-Games output inside the Kuafu run directory when provided.
    if kuafu_run_dir:
        log_root_path = os.path.join(os.path.abspath(kuafu_run_dir), "rl_runs")
    else:
        log_root_path = str(
            external_artifact_root()
            / "standalone"
            / "residual_rl"
            / "rl_runs"
            / agent_cfg["params"]["config"]["name"]
        )
    os.makedirs(log_root_path, exist_ok=True)
    print(f"[residual_rl] Logging experiment in directory: {log_root_path}", flush=True)
    print(
        "[residual_rl] RL-Games config: "
        f"max_epochs={effective_max_epochs} "
        f"resume_from_checkpoint={bool(resume_from_checkpoint)} "
        f"prior_checkpoint={bool(prior_checkpoint)} "
        f"prior_mode={prior_mode} "
        f"prior_critic_init_mode={prior_critic_init_mode} "
        f"prior_freeze_actor_epochs={prior_freeze_actor_epochs} "
        f"prior_sigma_ramp_epochs={prior_sigma_ramp_epochs} "
        f"prior_sigma_final_val={prior_sigma_final_val} "
        f"prior_actor_lr_warmup_epochs={prior_actor_lr_warmup_epochs} "
        f"prior_actor_lr_cap={prior_actor_lr_cap} "
        f"prior_lr_schedule_freeze_during_warmup={prior_lr_schedule_freeze_during_warmup} "
        f"prior_fixed_lr={prior_fixed_lr} "
        f"prior_freeze_input_rms={prior_freeze_input_running_mean_std} "
        f"prior_unfreeze_input_rms_after_freeze={prior_unfreeze_input_running_mean_std_after_freeze} "
        f"prior_rms_unfreeze_after_actor_lr_warmup={prior_rms_unfreeze_after_actor_lr_warmup} "
        f"prior_bc_anchor_epochs={prior_bc_anchor_epochs} "
        f"prior_bc_anchor_coef={prior_bc_anchor_coef} "
        f"prior_bc_dataset_coef={prior_bc_dataset_coef} "
        f"prior_bc_dataset_decay_epochs={prior_bc_dataset_decay_epochs} "
        f"prior_bc_dataset_batch_size={prior_bc_dataset_batch_size} "
        f"prior_bc_dataset_path={prior_bc_dataset_path or '<auto>'} "
        f"prior_bc_action_reward_coef={prior_bc_action_reward_coef} "
        f"residual_cfg={'enabled' if residual_cfg is not None else 'disabled'}",
        flush=True,
    )

    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_Run-{worker_idx}"
    agent_cfg["params"]["config"]["train_dir"] = log_root_path
    agent_cfg["params"]["config"]["full_experiment_name"] = log_dir
    run_dir = os.path.join(log_root_path, log_dir)

    # Tensorboard directory (what feedback reader uses)
    tb_log_dir = os.path.join(run_dir, "summaries")

    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    if residual_cfg is not None:
        rl_env = _make_kuafu_residual_rl_games_wrapper(env, device, clip_obs, clip_actions, residual_cfg)
    elif prior_bc_action_reward_coef > 0.0 and prior_checkpoint is not None:
        _obs_groups = agent_cfg["params"]["env"].get("obs_groups")
        _concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)
        rl_env = _make_bc_action_reward_wrapper(
            env, device, clip_obs, clip_actions,
            prior_checkpoint=prior_checkpoint,
            bc_action_reward_coef=prior_bc_action_reward_coef,
            obs_groups=_obs_groups,
            concate_obs_group=_concate_obs_groups,
        )
    else:
        rl_env = RlGamesVecEnvWrapper(env, device, clip_obs, clip_actions)
    residual_controller = None
    if residual_cfg is not None:
        assert prior_checkpoint is not None
        residual_controller = _make_residual_action_controller(
            prior_checkpoint=prior_checkpoint,
            residual_cfg=residual_cfg,
            rl_env=rl_env,
        )

    # Register env with RL-Games (idempotent-ish per process).
    try:
        vecenv.register(
            "IsaacRlgWrapper",
            lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
        )
    except Exception as exc:
        print(
            "[residual_rl] WARNING: vecenv.register('IsaacRlgWrapper') failed; "
            f"assuming it is already registered. Error: {type(exc).__name__}: {exc}",
            flush=True,
        )
    try:
        env_configurations.register(
            "rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: rl_env}
        )
    except Exception as exc:
        # If already registered, overwrite directly (best-effort).
        print(
            "[residual_rl] WARNING: env_configurations.register('rlgpu') failed; "
            f"overwriting configuration directly. Error: {type(exc).__name__}: {exc}",
            flush=True,
        )
        try:
            env_configurations.configurations["rlgpu"] = {
                "vecenv_type": "IsaacRlgWrapper",
                "env_creator": lambda **kwargs: rl_env,
            }
        except Exception as inner_exc:
            raise RuntimeError("[residual_rl] Failed to register or overwrite RL-Games env configuration 'rlgpu'.") from inner_exc

    agent_cfg["params"]["config"]["num_actors"] = rl_env.unwrapped.num_envs
    _dump_run_yaml_configs(
        run_dir=run_dir,
        env=env,
        agent_cfg=agent_cfg,
        task=task,
        worker_idx=worker_idx,
        prior_mode=prior_mode,
        effective_max_epochs=effective_max_epochs,
    )

    runner = Runner(IsaacAlgoObserver())
    runner.load(agent_cfg)
    runner.reset()
    run_kwargs: dict[str, object] = {"train": True, "play": False, "sigma": None}
    if resume_from_checkpoint:
        # RL-Games supports passing the checkpoint path as part of runner.run kwargs.
        run_kwargs["checkpoint"] = str(resume_from_checkpoint)
    print(f"[residual_rl] RL-Games runner.run kwargs: {run_kwargs}", flush=True)
    if residual_cfg is not None:
        pbt_cfg = agent_cfg.get("pbt")
        if isinstance(pbt_cfg, dict) and pbt_cfg.get("enabled", False):
            raise ValueError("Residual RL mode is not supported with PBT enabled.")
        assert residual_controller is not None
        agent = runner.algo_factory.create(runner.algo_name, base_name="run", params=runner.params)
        _install_residual_rl_hooks(agent, rl_env.unwrapped, residual_controller, residual_cfg)
        if resume_from_checkpoint:
            print(
                f"[residual_rl] Restoring residual RL checkpoint before training: {resume_from_checkpoint}",
                flush=True,
            )
            agent.restore(str(resume_from_checkpoint))
        print(
            "[residual_rl] Residual RL mode bypasses legacy full-actor BC->PPO controls "
            "(actor freeze/unfreeze, sigma silencing, BC anchor, BC dataset loss).",
            flush=True,
        )
        agent.train()
    elif resume_from_checkpoint:
        # Raw-input BC initialization persists explicit normalization buffers.
        # Restore their layer types even though the standard builder makes Linear.
        from kuafu.bc.training.normalization import install_checkpoint_normalizer_loader

        original_create = runner.algo_factory.create

        def create_with_normalizer_loader(*args, **kwargs):
            restored_agent = original_create(*args, **kwargs)
            install_checkpoint_normalizer_loader(restored_agent.model)
            return restored_agent

        runner.algo_factory.create = create_with_normalizer_loader
        runner.run(run_kwargs)
    elif prior_checkpoint is None:
        runner.run(run_kwargs)
    else:
        pbt_cfg = agent_cfg.get("pbt")
        if isinstance(pbt_cfg, dict) and pbt_cfg.get("enabled", False):
            raise ValueError("prior_checkpoint is not supported with PBT enabled.")

        from kuafu.bc.training.rl_games_prior import apply_bc_prior_to_rl_games_agent

        print(f"[residual_rl] Applying BC prior checkpoint: {prior_checkpoint}", flush=True)
        agent = runner.algo_factory.create(runner.algo_name, base_name="run", params=runner.params)
        report = apply_bc_prior_to_rl_games_agent(
            agent,
            prior_checkpoint,
            obs_stats_count=prior_obs_stats_count,
            verbose=prior_verbose,
            load_critic_prior=(prior_critic_init_mode == "actor_and_critic_prior"),
            require_critic_prior=(prior_critic_init_mode == "actor_and_critic_prior"),
            freeze_input_running_mean_std=prior_freeze_input_running_mean_std,
        )
        print(f"[residual_rl] Prior load report: {report}", flush=True)
        _stabilize_sigma_head_after_prior(agent_cfg, agent)
        if prior_critic_init_mode == "actor_only_zero_critic":
            _zero_critic_after_prior(agent)
        prior_actor_lr_cap_value = float(prior_actor_lr_cap or 0.0)
        prior_bc_dataset_coef_value = float(prior_bc_dataset_coef or 0.0)
        prior_fixed_lr_value: float | None = None
        if prior_fixed_lr is not None:
            try:
                _flr = float(prior_fixed_lr)
                if math.isfinite(_flr) and _flr > 0.0:
                    prior_fixed_lr_value = _flr
            except (TypeError, ValueError):
                prior_fixed_lr_value = None
        prior_safety_controls_requested = (
            int(prior_freeze_actor_epochs) > 0
            or int(prior_sigma_ramp_epochs) > 0
            or int(prior_actor_lr_warmup_epochs) > 0
            or prior_actor_lr_cap_value > 0.0
            or int(prior_bc_anchor_epochs) > 0
            or float(prior_bc_anchor_coef) > 0.0
            or prior_bc_dataset_coef_value > 0.0
            or prior_fixed_lr_value is not None
        )
        # Load BC demonstration dataset for the supervised regularizer if requested.
        # This is done once, here, because it needs ``prior_checkpoint`` to auto-resolve
        # dataset path + obs normalizer stats, and we keep the tensors pinned to the
        # agent device for the whole run.
        bc_dataset_bundle = None
        if prior_bc_dataset_coef_value > 0.0:
            bc_dataset_bundle = _load_bc_supervised_dataset(
                prior_checkpoint,
                device=str(agent.ppo_device),
                dataset_path_override=prior_bc_dataset_path or None,
            )
            print(
                f"[residual_rl] BC supervised dataset loaded: "
                f"{bc_dataset_bundle['dataset_path']} "
                f"(num_pairs={bc_dataset_bundle['num_pairs']}, "
                f"obs_dim={int(bc_dataset_bundle['obs_raw'].shape[1])}, "
                f"act_dim={int(bc_dataset_bundle['actions_raw'].shape[1])}, "
                f"obs_key={bc_dataset_bundle['obs_key']}, "
                f"obs_history={bc_dataset_bundle['obs_history']}, "
                f"action_chunk_size={bc_dataset_bundle['action_chunk_size']}).",
                flush=True,
            )
        if prior_safety_controls_requested:
            _freeze_actor_for_epochs(
                agent,
                int(prior_freeze_actor_epochs),
                unfreeze_input_rms_at_end=(
                    report.input_running_mean_std_frozen
                    and prior_unfreeze_input_running_mean_std_after_freeze
                ),
                rms_unfreeze_after_actor_lr_warmup=bool(
                    prior_rms_unfreeze_after_actor_lr_warmup
                ),
                sigma_ramp_epochs=int(prior_sigma_ramp_epochs),
                sigma_final_val=(
                    float(prior_sigma_final_val)
                    if prior_sigma_final_val is not None
                    else None
                ),
                actor_lr_warmup_epochs=int(prior_actor_lr_warmup_epochs),
                actor_lr_cap=prior_actor_lr_cap_value,
                lr_schedule_freeze_during_warmup=bool(
                    prior_lr_schedule_freeze_during_warmup
                ),
                fixed_lr=prior_fixed_lr_value,
                bc_anchor_epochs=int(prior_bc_anchor_epochs),
                bc_anchor_coef=float(prior_bc_anchor_coef),
                bc_dataset=bc_dataset_bundle,
                bc_dataset_coef=prior_bc_dataset_coef_value,
                bc_dataset_decay_epochs=int(prior_bc_dataset_decay_epochs),
                bc_dataset_batch_size=int(prior_bc_dataset_batch_size),
            )
        agent.train()

    # Best-effort: checkpoint path
    ckpts = glob.glob(os.path.join(run_dir, "nn", "*.pth"))
    checkpoint_path = max(ckpts, key=os.path.getmtime) if ckpts else None
    return tb_log_dir, checkpoint_path


def evaluate_bc_prior_policy_rollout(
    *,
    env,
    task: str,
    device: str,
    prior_checkpoint: str,
    agent_cfg: dict | None = None,
    prior_obs_stats_count: float = 1e6,
    prior_verbose: bool = False,
    prior_freeze_input_running_mean_std: bool = False,
    seed: int,
    max_eval_steps: int | None = None,
) -> dict[str, object]:
    """Evaluate a BC prior rehosted inside an RL-Games actor and return env stats."""
    import torch
    from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
    from rl_games.common import env_configurations, vecenv
    from rl_games.common.algo_observer import IsaacAlgoObserver
    from rl_games.torch_runner import Runner
    from kuafu.bc.training.rl_games_prior import apply_bc_prior_to_rl_games_agent

    if not prior_checkpoint or not (os.path.isfile(prior_checkpoint) or os.path.isdir(prior_checkpoint)):
        raise FileNotFoundError(f"[residual_rl] BC prior checkpoint/run-dir missing on disk: {prior_checkpoint}")
    if agent_cfg is None:
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

        agent_cfg = load_cfg_from_registry(task, "rl_games_cfg_entry_point")
    assert agent_cfg is not None

    eval_cfg = copy.deepcopy(agent_cfg)
    raw_env = env.unwrapped
    eval_cfg["params"]["config"]["device"] = str(device)
    eval_cfg["params"]["config"]["device_name"] = str(device)
    eval_cfg["params"]["config"]["num_actors"] = int(raw_env.num_envs)
    eval_cfg["params"]["load_checkpoint"] = False
    eval_cfg["params"]["load_path"] = ""

    # The A2C constructor still validates training batch divisibility even for rollout-only use.
    horizon = int(eval_cfg["params"]["config"].get("horizon_length", 64) or 64)
    batch_size = int(raw_env.num_envs) * horizon
    minibatch = int(eval_cfg["params"]["config"].get("minibatch_size", batch_size) or batch_size)
    if batch_size > 0 and (minibatch <= 0 or batch_size % minibatch != 0):
        eval_cfg["params"]["config"]["minibatch_size"] = batch_size

    clip_obs = eval_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = eval_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = eval_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = eval_cfg["params"]["env"].get("concate_obs_groups", True)
    try:
        rl_env = RlGamesVecEnvWrapper(
            env,
            str(device),
            clip_obs,
            clip_actions,
            obs_groups,
            concate_obs_groups,
        )
    except TypeError:
        rl_env = RlGamesVecEnvWrapper(env, str(device), clip_obs, clip_actions)

    try:
        vecenv.register(
            "IsaacRlgWrapper",
            lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
        )
    except Exception as exc:
        print(
            "[residual_rl] WARNING: vecenv.register('IsaacRlgWrapper') failed during BC-prior eval; "
            f"assuming it is already registered. Error: {type(exc).__name__}: {exc}",
            flush=True,
        )
    try:
        env_configurations.register(
            "rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: rl_env}
        )
    except Exception as exc:
        print(
            "[residual_rl] WARNING: env_configurations.register('rlgpu') failed during BC-prior eval; "
            f"overwriting configuration directly. Error: {type(exc).__name__}: {exc}",
            flush=True,
        )
        try:
            env_configurations.configurations["rlgpu"] = {
                "vecenv_type": "IsaacRlgWrapper",
                "env_creator": lambda **kwargs: rl_env,
            }
        except Exception as inner_exc:
            raise RuntimeError("[residual_rl] Failed to register or overwrite eval RL-Games env configuration 'rlgpu'.") from inner_exc

    runner = Runner(IsaacAlgoObserver())
    runner.load(eval_cfg)
    runner.reset()
    agent = runner.algo_factory.create(runner.algo_name, base_name="run", params=runner.params)
    report = apply_bc_prior_to_rl_games_agent(
        agent,
        prior_checkpoint,
        obs_stats_count=prior_obs_stats_count,
        verbose=prior_verbose,
        load_critic_prior=False,
        require_critic_prior=False,
        freeze_input_running_mean_std=prior_freeze_input_running_mean_std,
    )
    print(f"[residual_rl] BC-prior eval load report: {report}", flush=True)
    _stabilize_sigma_head_after_prior(eval_cfg, agent)
    agent.model.eval()

    if hasattr(raw_env, "clear_stage_transition_stats"):
        raw_env.clear_stage_transition_stats()
    elif hasattr(raw_env, "clear_end_to_end_rate_stats"):
        raw_env.clear_end_to_end_rate_stats()
    if hasattr(raw_env, "clear_terminal_stuck_constraint_stats"):
        raw_env.clear_terminal_stuck_constraint_stats()

    if max_eval_steps is None:
        max_eval_steps = int(getattr(raw_env, "max_episode_length", 0))
    if int(max_eval_steps) <= 0:
        raise RuntimeError(f"[residual_rl] Invalid BC-prior eval horizon: max_eval_steps={max_eval_steps}")

    rl_env.seed(int(seed))
    obs = rl_env.reset()
    if isinstance(obs, dict):
        obs = obs.get("obs", obs)

    completed_episodes = 0
    reward_sum = 0.0
    total_steps = 0

    def _action_from_model(obs_obj: Any) -> torch.Tensor:
        obs_t = obs_obj if torch.is_tensor(obs_obj) else torch.as_tensor(obs_obj, device=device)
        with torch.no_grad():
            out = agent.model(
                {
                    "is_train": False,
                    "prev_actions": None,
                    "obs": obs_t,
                    "rnn_states": None,
                }
            )
        return out["mus"]

    while total_steps < int(max_eval_steps):
        actions = _action_from_model(obs)
        obs, rew, dones, _extras = rl_env.step(actions)
        if isinstance(obs, dict):
            obs = obs.get("obs", obs)

        if torch.is_tensor(rew):
            reward_sum += float(rew.detach().to(dtype=torch.float32).sum().item())
        done_mask = dones.to(dtype=torch.bool).view(-1)
        completed_episodes += int(done_mask.sum().item())
        total_steps += 1

    true_success_count = int(raw_env.get_true_success_count()) if hasattr(raw_env, "get_true_success_count") else 0
    false_complete_count = int(raw_env.get_false_complete_count()) if hasattr(raw_env, "get_false_complete_count") else 0
    timeout_count = int(raw_env.get_timeout_count()) if hasattr(raw_env, "get_timeout_count") else 0
    total_episodes = true_success_count + false_complete_count + timeout_count
    if total_episodes <= 0:
        total_episodes = int(completed_episodes)
    if total_episodes <= 0:
        raise RuntimeError(
            f"[residual_rl] BC-prior eval completed zero episodes after {total_steps} steps."
        )

    return {
        "completed_episodes": int(completed_episodes),
        "total_steps": int(total_steps),
        "average_step_reward": reward_sum / max(1, int(total_steps) * int(raw_env.num_envs)),
        "stage_transition_stats": raw_env.get_stage_transition_stats(),
        "highest_stage_reached_per_env": raw_env.get_highest_stage_reached_per_env(),
        "constraint_met_rate_stats": raw_env.get_constraint_stats(),
        "true_success_count": true_success_count,
        "false_complete_count": false_complete_count,
        "timeout_count": timeout_count,
        "total_episodes": total_episodes,
    }


def evaluate_rl_games(
    *,
    env,
    task: str,
    device: str,
    checkpoint_path: str,
    prior_checkpoint: str | None = None,
    prior_mode: str = "full_actor",
    agent_cfg: dict | None = None,
    progress_metric_expr: str,
    seed: int,
    max_eval_steps: int | None = None,
) -> dict[str, object]:
    """Evaluate a saved checkpoint on a fresh deterministic rollout.

    The returned metrics are used for worker selection, so missing eval outputs are fatal.
    """
    import torch
    from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
    from rl_games.common import env_configurations, vecenv
    from rl_games.common.player import BasePlayer
    from rl_games.torch_runner import Runner

    if agent_cfg is None:
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

        agent_cfg = load_cfg_from_registry(task, "rl_games_cfg_entry_point")
    assert agent_cfg is not None
    prior_mode = _resolve_prior_mode(agent_cfg, prior_mode)

    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"[residual_rl] Eval checkpoint missing on disk: {checkpoint_path}")
    if not progress_metric_expr:
        raise RuntimeError("[residual_rl] Eval requires a progress_metric_expr.")

    eval_cfg = copy.deepcopy(agent_cfg)
    eval_cfg["params"]["config"]["device"] = str(device)
    eval_cfg["params"]["config"]["device_name"] = str(device)
    eval_cfg["params"]["load_checkpoint"] = True
    eval_cfg["params"]["load_path"] = str(checkpoint_path)
    residual_cfg = None
    if prior_mode == "residual_rl":
        if prior_checkpoint is None:
            raise ValueError("Residual RL eval requires prior_checkpoint to reconstruct the frozen BC base policy.")
        from kuafu.bc.training.residual_rl import ResidualRlConfig

        residual_mapping = copy.deepcopy(
            eval_cfg.get("params", {}).get("config", {}).get("residual_rl", {})
        )
        if isinstance(residual_mapping, dict):
            residual_mapping["enabled"] = True
        residual_cfg = ResidualRlConfig.from_mapping(
            residual_mapping,
            action_dim=int(env.unwrapped.single_action_space.shape[0]),
            env_num_stages=_resolve_env_num_stages(env),
        )
        _configure_residual_rl_agent_cfg(eval_cfg, residual_cfg)

    clip_obs = eval_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = eval_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = eval_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = eval_cfg["params"]["env"].get("concate_obs_groups", True)
    if prior_mode == "residual_rl":
        assert residual_cfg is not None
        rl_env = _make_kuafu_residual_rl_games_wrapper(
            env, str(device), clip_obs, clip_actions, residual_cfg, obs_groups, concate_obs_groups
        )
    else:
        rl_env = RlGamesVecEnvWrapper(env, str(device), clip_obs, clip_actions, obs_groups, concate_obs_groups)
    if prior_mode == "residual_rl":
        assert residual_cfg is not None
        assert prior_checkpoint is not None
        residual_controller = _make_residual_action_controller(
            prior_checkpoint=prior_checkpoint,
            residual_cfg=residual_cfg,
            rl_env=rl_env,
        )
        eval_epoch = _checkpoint_epoch(str(checkpoint_path))
        residual_controller.set_epoch(eval_epoch)
        print(
            f"[residual_rl] Residual RL eval controller restored: "
            f"checkpoint_epoch={eval_epoch} alpha={residual_controller.alpha:.3f}.",
            flush=True,
        )

    try:
        vecenv.register(
            "IsaacRlgWrapper",
            lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
        )
    except Exception as exc:
        print(
            "[residual_rl] WARNING: vecenv.register('IsaacRlgWrapper') failed during eval; "
            f"assuming it is already registered. Error: {type(exc).__name__}: {exc}",
            flush=True,
        )
    try:
        env_configurations.register(
            "rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: rl_env}
        )
    except Exception as exc:
        print(
            "[residual_rl] WARNING: env_configurations.register('rlgpu') failed during eval; "
            f"overwriting configuration directly. Error: {type(exc).__name__}: {exc}",
            flush=True,
        )
        try:
            env_configurations.configurations["rlgpu"] = {
                "vecenv_type": "IsaacRlgWrapper",
                "env_creator": lambda **kwargs: rl_env,
            }
        except Exception as inner_exc:
            raise RuntimeError("[residual_rl] Failed to register or overwrite eval RL-Games env configuration 'rlgpu'.") from inner_exc

    raw_env = rl_env.unwrapped
    eval_cfg["params"]["config"]["num_actors"] = raw_env.num_envs
    runner = Runner()
    runner.load(eval_cfg)
    agent: BasePlayer = runner.create_player()
    from kuafu.bc.training.normalization import install_checkpoint_normalizer_loader

    install_checkpoint_normalizer_loader(agent.model)
    agent.restore(str(checkpoint_path))
    agent.reset()

    if hasattr(raw_env, "clear_stage_transition_stats"):
        raw_env.clear_stage_transition_stats()
    elif hasattr(raw_env, "clear_end_to_end_rate_stats"):
        raw_env.clear_end_to_end_rate_stats()
    if hasattr(raw_env, "clear_terminal_stuck_constraint_stats"):
        raw_env.clear_terminal_stuck_constraint_stats()

    if max_eval_steps is None:
        max_eval_steps = int(getattr(raw_env, "max_episode_length", 0))
    if int(max_eval_steps) <= 0:
        raise RuntimeError(f"[residual_rl] Invalid eval horizon: max_eval_steps={max_eval_steps}")

    rl_env.seed(int(seed))
    obs = rl_env.reset()
    if isinstance(obs, dict):
        obs = obs.get("obs", obs)
    agent_is_rnn = bool(getattr(agent, "is_rnn", False))
    if agent_is_rnn:
        agent.init_rnn()
    _ = agent.get_batch_size(obs, 1)

    completed_returns: list[float] = []
    running_returns: torch.Tensor | None = None
    max_external_reward: float | None = None
    external_reward_error: str | None = None
    total_steps = 0
    max_stage_observed = 0
    max_external_reward_observed: float | None = None

    while total_steps < int(max_eval_steps):
        # Do not use inference_mode around env.step(): some env-owned tensors are updated
        # in-place during rollout and must remain normal tensors for the next training phase.
        with torch.no_grad():
            obs_t = agent.obs_to_torch(obs)
            actions = agent.get_action(obs_t, is_deterministic=True)
            obs, rew, dones, _extras = rl_env.step(actions)
            if isinstance(obs, dict):
                obs = obs.get("obs", obs)

            if external_reward_error is None and (
                hasattr(raw_env, "get_last_external_rewards") or hasattr(raw_env, "external_rewards")
            ):
                try:
                    external_rewards = None
                    if hasattr(raw_env, "get_last_external_rewards"):
                        external_rewards = raw_env.get_last_external_rewards()
                    if external_rewards is None:
                        external_rewards = raw_env.external_rewards()
                    external_rewards = external_rewards.detach().to(dtype=torch.float32).view(-1)
                    step_external_max = float(external_rewards.max().item())
                    max_external_reward = (
                        step_external_max
                        if max_external_reward is None
                        else max(max_external_reward, step_external_max)
                    )
                except NotImplementedError as exc:
                    external_reward_error = f"{type(exc).__name__}: {exc}"
                except Exception as exc:
                    external_reward_error = f"{type(exc).__name__}: {exc}"

            rew_t = rew.detach().to(dtype=torch.float32)
            if running_returns is None:
                running_returns = torch.zeros_like(rew_t, dtype=torch.float32)
            running_returns = running_returns + rew_t
            if running_returns is None:
                raise RuntimeError("[residual_rl] Eval running_returns unexpectedly uninitialized.")
            running_returns_t = running_returns

            current_stage = getattr(raw_env, "current_stage", None)
            if torch.is_tensor(current_stage) and current_stage.numel() > 0:
                max_stage_observed = max(max_stage_observed, int(current_stage.detach().max().item()))
            elif current_stage is not None:
                try:
                    max_stage_observed = max(max_stage_observed, int(current_stage))
                except (TypeError, ValueError):
                    pass

            if hasattr(raw_env, "external_rewards"):
                try:
                    external_reward = raw_env.external_rewards()
                except NotImplementedError:
                    external_reward = None
                if torch.is_tensor(external_reward) and external_reward.numel() > 0:
                    external_max = float(
                        external_reward.detach().to(dtype=torch.float32).view(-1).max().item()
                    )
                    if max_external_reward_observed is None:
                        max_external_reward_observed = external_max
                    else:
                        max_external_reward_observed = max(max_external_reward_observed, external_max)
                elif external_reward is not None:
                    try:
                        external_max = float(external_reward)
                    except (TypeError, ValueError):
                        external_max = None
                    if external_max is not None:
                        if max_external_reward_observed is None:
                            max_external_reward_observed = external_max
                        else:
                            max_external_reward_observed = max(max_external_reward_observed, external_max)

            done_mask = dones.to(dtype=torch.bool)
            if done_mask.ndim > 1:
                done_mask = done_mask.view(-1)
            if done_mask.any():
                completed_returns.extend(running_returns_t[done_mask].detach().cpu().tolist())
                running_returns_t[done_mask] = 0.0
                agent_states = getattr(agent, "states", None)
                if agent_is_rnn and agent_states is not None:
                    for s in agent_states:
                        s[:, done_mask, :] = 0.0

        total_steps += 1

    if not completed_returns:
        raise RuntimeError(
            f"[residual_rl] Eval completed zero episodes after {total_steps} steps for checkpoint {checkpoint_path}"
        )

    metric_value = eval(progress_metric_expr, {"torch": torch}, {"self": raw_env})
    if hasattr(metric_value, "item"):
        progress_metric_eval = float(metric_value.item())
    else:
        progress_metric_eval = float(metric_value)
    completed_returns_f = [float(x) for x in completed_returns]
    avg_episode_return = float(sum(completed_returns_f) / len(completed_returns_f))

    timeout_count = raw_env.get_timeout_count() if hasattr(raw_env, "get_timeout_count") else 0
    highest_stage_reached_per_env = raw_env.get_highest_stage_reached_per_env()
    if highest_stage_reached_per_env:
        max_stage_reached = int(max(highest_stage_reached_per_env))
    else:
        max_stage_reached = int(max_stage_observed)
    external_reward_candidates = [
        value for value in (max_external_reward_observed, max_external_reward) if value is not None
    ]
    max_external_reward_value = max(external_reward_candidates) if external_reward_candidates else None
    payload = {
        "progress_metric_eval": progress_metric_eval,
        "average_episodic_return": avg_episode_return,
        "max_episodic_return": float(max(completed_returns_f)),
        "min_episodic_return": float(min(completed_returns_f)),
        "completed_episodes": int(len(completed_returns)),
        "stage_transition_stats": raw_env.get_stage_transition_stats(),
        "highest_stage_reached_per_env": highest_stage_reached_per_env,
        "max_stage_reached": max_stage_reached,
        "max_stage_observed": int(max_stage_observed),
        "max_external_reward": max_external_reward_value,
        "external_reward_error": external_reward_error,
        "constraint_met_rate_stats": raw_env.get_constraint_stats(),
        "true_success_count": raw_env.get_true_success_count(),
        "false_complete_count": raw_env.get_false_complete_count(),
        "timeout_count": timeout_count,
    }
    if hasattr(raw_env, "clear_terminal_stuck_constraint_stats"):
        raw_env.clear_terminal_stuck_constraint_stats()
    return payload
