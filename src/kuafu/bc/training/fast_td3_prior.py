"""Transfer a BC prior checkpoint into a FastTD3 Actor and seed its EmpiricalNormalization.

FastTD3 Actor trunk layout (actor_hidden_dim = H):
    actor.net[0]  = Linear(n_obs, H)
    actor.net[1]  = ReLU
    actor.net[2]  = Linear(H, H//2)
    actor.net[3]  = ReLU
    actor.net[4]  = Linear(H//2, H//4)
    actor.net[5]  = ReLU
    actor.fc_mu[0] = Linear(H//4, n_act)
    actor.fc_mu[1] = Tanh

For weight transfer to succeed the BC model must have been trained with
    hidden_dims = (H, H // 2, H // 4)
i.e. exactly three hidden layers whose sizes match the FastTD3 actor structure.

Weights are loaded from the 'model_state_dict_norm' baked variant, which expects
*normalised* observations and outputs *normalised* actions.  That representation is
compatible with FastTD3's workflow where observations are normalised externally by
EmpiricalNormalization before being fed to the actor, and the actor's Tanh head
maps approximately-unit-variance logits into [-1, 1].
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FastTD3PriorLoadReport:
    source_path: str
    resolved_checkpoint_path: str
    used_state_dict: str
    trunk_layers_copied: int
    output_layer_copied: bool
    obs_normalizer_seeded: bool
    obs_dim_bc: int
    obs_dim_actor: int
    act_dim_bc: int
    act_dim_actor: int
    hidden_dims_bc: tuple[int, ...]
    epoch: int | None = None
    train_loss: float | None = None
    val_loss: float | None = None
    obs_key: str | None = None
    action_mode: str | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_prior_checkpoint_path(prior_path: str) -> Path:
    """Return the .pth file, searching run-dir layouts if a directory is given."""
    p = Path(prior_path).expanduser()
    if p.is_file():
        return p.resolve()
    if p.is_dir():
        for base in (p, p / "checkpoints"):
            for name in ("best.pth", "final.pth"):
                candidate = base / name
                if candidate.is_file():
                    return candidate.resolve()
    raise FileNotFoundError(
        f"Could not resolve BC prior checkpoint from '{prior_path}'. "
        "Pass either a .pth file or a BC run directory containing checkpoints/best.pth."
    )


def _load_adjacent_run_config(ckpt_path: Path) -> dict[str, Any]:
    import yaml

    run_dir = ckpt_path.parent.parent if ckpt_path.parent.name == "checkpoints" else ckpt_path.parent
    for cfg_path in (run_dir / "meta" / "config.yaml", run_dir / "train_config.yaml"):
        if cfg_path.is_file():
            with open(cfg_path, encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                return loaded
    return {}


def _extract_bc_trunk_linear_pairs(bc_sd: dict[str, torch.Tensor]) -> list[tuple[str, str]]:
    """Return (weight_key, bias_key) pairs for every Linear in the BC actor trunk.

    Supports the two layouts used by MLPActorPrior / MLPActorCriticPrior:
      - old:  trunk.0, trunk.2, trunk.4  (no dropout)
      - old:  trunk.0, trunk.3, trunk.6  (with dropout, stride=3)
      - new:  actor_trunk.0, actor_trunk.3, actor_trunk.6
    """
    for prefix in ("actor_trunk", "trunk"):
        indices: dict[int, set[str]] = {}
        pattern = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.(weight|bias)$")
        for key in bc_sd:
            m = pattern.match(key)
            if m:
                indices.setdefault(int(m.group(1)), set()).add(m.group(2))

        pairs: list[tuple[str, str]] = []
        for idx in sorted(indices):
            if {"weight", "bias"}.issubset(indices[idx]):
                pairs.append((f"{prefix}.{idx}.weight", f"{prefix}.{idx}.bias"))
        if pairs:
            return pairs

    raise KeyError(
        "Could not find any BC trunk Linear layers. "
        f"Available keys sample: {list(bc_sd.keys())[:20]}"
    )


def _linear_layers_in_sequential(module: nn.Sequential) -> list[tuple[int, nn.Linear]]:
    """Return (index, layer) pairs for every nn.Linear in a Sequential."""
    return [(i, m) for i, m in enumerate(module) if isinstance(m, nn.Linear)]


def _seed_empirical_normalizer(normalizer: nn.Module, mean: torch.Tensor, std: torch.Tensor, count: int) -> None:
    """Populate EmpiricalNormalization buffers from BC obs statistics.

    EmpiricalNormalization stores:
        _mean  : shape (1, obs_dim)
        _var   : shape (1, obs_dim)  -- variance
        _std   : shape (1, obs_dim)  -- std dev
        count  : scalar long

    The normalizer must already be built with the right obs_dim.
    """
    device = normalizer._mean.device
    mean_f = mean.to(device=device, dtype=torch.float32).flatten()
    std_f  = std .to(device=device, dtype=torch.float32).flatten().clamp(min=1e-6)
    var_f  = std_f.pow(2)

    if normalizer._mean.shape[-1] != mean_f.numel():
        raise ValueError(
            f"EmpiricalNormalization obs_dim mismatch: "
            f"normalizer.shape={normalizer._mean.shape} vs BC mean.numel()={mean_f.numel()}"
        )

    with torch.no_grad():
        normalizer._mean.copy_(mean_f.unsqueeze(0))
        normalizer._var .copy_(var_f .unsqueeze(0))
        normalizer._std .copy_(std_f .unsqueeze(0))
        normalizer.count.copy_(torch.tensor(count, dtype=torch.long, device=device))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_bc_prior_to_fast_td3_actor(
    actor: nn.Module,
    obs_normalizer: nn.Module | None,
    prior_ckpt_path: str,
    *,
    obs_stats_count: int = 1_000_000,
    verbose: bool = True,
) -> FastTD3PriorLoadReport:
    """Copy BC MLP weights into a FastTD3 Actor and optionally seed EmpiricalNormalization.

    Parameters
    ----------
    actor:
        A FastTD3 ``Actor`` instance (from ``FastTD3/fast_td3/fast_td3.py``).
        Its ``net`` Sequential must have exactly 3 Linear layers, and ``fc_mu``
        must be a Sequential whose first element is a Linear.
    obs_normalizer:
        The ``EmpiricalNormalization`` instance used for policy observations.
        Pass ``None`` to skip seeding.  If passed, its ``_mean``/``_var``/``_std``
        buffers are initialised from the BC checkpoint's obs statistics.
    prior_ckpt_path:
        Path to a BC ``.pth`` checkpoint or a BC run directory.
    obs_stats_count:
        Synthetic count assigned to the normalizer so downstream updates are
        down-weighted relative to the pre-loaded statistics.
    verbose:
        Print a summary of what was copied.

    Returns
    -------
    FastTD3PriorLoadReport
        Structured summary of what was transferred.
    """
    resolved = _resolve_prior_checkpoint_path(prior_ckpt_path)
    ckpt = torch.load(str(resolved), map_location="cpu", weights_only=False)

    if "model_state_dicts" not in ckpt:
        raise KeyError(
            "BC checkpoint is missing 'model_state_dicts'. "
            "Expected output of src/kuafu/bc/training/train_bc_prior_mlp.py."
        )

    model_state_dicts: dict[str, dict] = ckpt["model_state_dicts"]
    used_key = "model_state_dict_norm"
    if used_key not in model_state_dicts:
        raise KeyError(
            f"BC checkpoint missing baked weights for key '{used_key}'. "
            "Re-train the BC prior with an up-to-date version of train_bc_prior_mlp.py."
        )

    bc_sd: dict[str, torch.Tensor] = model_state_dicts[used_key]

    # ---- Extract BC linear layer keys -----------------------------------
    trunk_pairs = _extract_bc_trunk_linear_pairs(bc_sd)
    if "mean_layer.weight" not in bc_sd or "mean_layer.bias" not in bc_sd:
        raise KeyError("BC state dict missing 'mean_layer.weight' / 'mean_layer.bias'.")

    # ---- Inspect FastTD3 actor structure --------------------------------
    if not hasattr(actor, "net") or not isinstance(actor.net, nn.Sequential):
        raise AttributeError("actor.net is not an nn.Sequential; unexpected actor architecture.")
    if not hasattr(actor, "fc_mu") or not isinstance(actor.fc_mu, nn.Sequential):
        raise AttributeError("actor.fc_mu is not an nn.Sequential; unexpected actor architecture.")

    actor_trunk_linears = _linear_layers_in_sequential(actor.net)
    if len(actor_trunk_linears) != 3:
        raise ValueError(
            f"FastTD3 actor.net has {len(actor_trunk_linears)} Linear layers but 3 are expected "
            "(hidden_dim, hidden_dim//2, hidden_dim//4). "
            "Ensure you are using the standard FastTD3 Actor, not SimbaV2."
        )
    fc_mu_linears = _linear_layers_in_sequential(actor.fc_mu)
    if len(fc_mu_linears) != 1:
        raise ValueError(
            f"actor.fc_mu has {len(fc_mu_linears)} Linear layers but 1 is expected."
        )

    # ---- Dimension checks -----------------------------------------------
    if len(trunk_pairs) != 3:
        raise ValueError(
            f"BC prior has {len(trunk_pairs)} hidden Linear layer(s) but FastTD3 actor "
            "requires exactly 3.  Train BC with hidden_dims matching "
            "(actor_hidden_dim, actor_hidden_dim//2, actor_hidden_dim//4)."
        )

    bc_obs_dim  = int(ckpt.get("obs_dim",  bc_sd[trunk_pairs[0][0]].shape[1]))
    bc_act_dim  = int(ckpt.get("act_dim",  bc_sd["mean_layer.weight"].shape[0]))

    _, actor_first_linear = actor_trunk_linears[0]
    _, actor_out_linear   = fc_mu_linears[0]
    rl_obs_dim = actor_first_linear.weight.shape[1]
    rl_act_dim = actor_out_linear.weight.shape[0]

    if bc_obs_dim != rl_obs_dim:
        raise ValueError(
            f"Obs-dim mismatch: BC prior obs_dim={bc_obs_dim} vs actor obs_dim={rl_obs_dim}. "
            "Ensure the BC dataset obs_key matches the FastTD3 policy observation space."
        )

    # Trunk hidden dims from BC weight shapes
    hidden_dims_bc: tuple[int, ...] = tuple(
        int(bc_sd[w_key].shape[0]) for w_key, _ in trunk_pairs
    )

    # Check each trunk layer shape matches
    for bc_pair, (actor_idx, actor_linear) in zip(trunk_pairs, actor_trunk_linears):
        w_key, b_key = bc_pair
        if bc_sd[w_key].shape != actor_linear.weight.shape:
            raise ValueError(
                f"Trunk layer shape mismatch at actor.net[{actor_idx}]: "
                f"BC {tuple(bc_sd[w_key].shape)} vs actor {tuple(actor_linear.weight.shape)}. "
                f"BC hidden_dims={hidden_dims_bc}. "
                "Train BC with hidden_dims = (actor_hidden_dim, actor_hidden_dim//2, actor_hidden_dim//4)."
            )

    # Handle action-dim difference: BC may use action_chunk_size > 1
    mu_w = bc_sd["mean_layer.weight"]
    mu_b = bc_sd["mean_layer.bias"]
    if bc_act_dim != rl_act_dim:
        if bc_act_dim > rl_act_dim and bc_act_dim % rl_act_dim == 0:
            mu_w = mu_w[:rl_act_dim]
            mu_b = mu_b[:rl_act_dim]
            if verbose:
                print(
                    f"[fast_td3_prior] BC act_dim={bc_act_dim} > rl_act_dim={rl_act_dim} "
                    f"(ratio={bc_act_dim // rl_act_dim}). Using first {rl_act_dim} rows."
                )
        else:
            raise ValueError(
                f"Action-dim mismatch: BC act_dim={bc_act_dim} vs actor act_dim={rl_act_dim}."
            )

    # ---- Copy trunk weights ---------------------------------------------
    device = actor_first_linear.weight.device
    trunk_layers_copied = 0
    for (w_key, b_key), (actor_idx, actor_linear) in zip(trunk_pairs, actor_trunk_linears):
        w = bc_sd[w_key].to(device=device, dtype=actor_linear.weight.dtype)
        b = bc_sd[b_key].to(device=device, dtype=actor_linear.bias.dtype)
        with torch.no_grad():
            actor_linear.weight.data.copy_(w)
            actor_linear.bias.data.copy_(b)
        trunk_layers_copied += 1
        if verbose:
            print(f"[fast_td3_prior] actor.net[{actor_idx}] <- {w_key}: W{tuple(w.shape)}")

    # ---- Copy output layer weights (mean_layer -> fc_mu[0]) ------------
    fc_mu_idx, fc_mu_linear = fc_mu_linears[0]
    with torch.no_grad():
        fc_mu_linear.weight.data.copy_(
            mu_w.to(device=device, dtype=fc_mu_linear.weight.dtype)
        )
        fc_mu_linear.bias.data.copy_(
            mu_b.to(device=device, dtype=fc_mu_linear.bias.dtype)
        )
    output_layer_copied = True
    if verbose:
        print(f"[fast_td3_prior] actor.fc_mu[{fc_mu_idx}] <- mean_layer: W{tuple(mu_w.shape)}")

    # ---- Seed EmpiricalNormalization from BC obs stats -----------------
    obs_norm_applied = False
    if obs_normalizer is not None and not isinstance(obs_normalizer, nn.Identity):
        obs_norm_sd: dict | None = ckpt.get("obs_normalizer")
        if obs_norm_sd is not None and "mean" in obs_norm_sd and "std" in obs_norm_sd:
            _seed_empirical_normalizer(
                obs_normalizer,
                mean=obs_norm_sd["mean"],
                std=obs_norm_sd["std"],
                count=obs_stats_count,
            )
            obs_norm_applied = True
            if verbose:
                print(
                    f"[fast_td3_prior] EmpiricalNormalization seeded from BC obs stats "
                    f"(count={obs_stats_count:,})"
                )
        else:
            if verbose:
                print("[fast_td3_prior] WARNING: BC checkpoint has no obs_normalizer; skipping normalizer seeding.")

    # ---- Load run config for metadata ----------------------------------
    run_cfg = _load_adjacent_run_config(resolved)
    obs_key    = ckpt.get("obs_key",    (run_cfg.get("dataset") or {}).get("obs_key"))
    action_mode = ckpt.get("action_mode", (run_cfg.get("dataset") or {}).get("action_mode"))

    return FastTD3PriorLoadReport(
        source_path=str(prior_ckpt_path),
        resolved_checkpoint_path=str(resolved),
        used_state_dict=used_key,
        trunk_layers_copied=trunk_layers_copied,
        output_layer_copied=output_layer_copied,
        obs_normalizer_seeded=obs_norm_applied,
        obs_dim_bc=bc_obs_dim,
        obs_dim_actor=rl_obs_dim,
        act_dim_bc=bc_act_dim,
        act_dim_actor=rl_act_dim,
        hidden_dims_bc=hidden_dims_bc,
        epoch=int(ckpt["epoch"]) if "epoch" in ckpt else None,
        train_loss=float(ckpt["train_loss"]) if "train_loss" in ckpt else None,
        val_loss=float(ckpt["val_loss"]) if "val_loss" in ckpt else None,
        obs_key=str(obs_key) if obs_key is not None else None,
        action_mode=str(action_mode) if action_mode is not None else None,
    )
