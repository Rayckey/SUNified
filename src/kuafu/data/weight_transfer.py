"""Utility to transfer BC actor weights from a robomimic checkpoint into an
SKRL SAC actor model.

Usage (inspect only – no Isaac Sim required):
    python src/kuafu/data/weight_transfer.py path/to/model_epoch_2000.pth

Running this prints every layer name and shape in the checkpoint so you can
verify (or adjust) the DEFAULT_KEY_MAP below before training.

Obs-normalisation note
----------------------
When bc_mlp.json has ``hdf5_normalize_obs: true``, robomimic trains the BC
network on observations that have been whitened with per-dimension statistics
(mean / std) computed from the demo dataset.  Those statistics are saved in
the checkpoint under the key ``obs_normalization_stats``.

At SAC inference time the environment produces *raw* (un-normalised)
observations, so the transferred weights would receive the wrong input
distribution and produce garbage actions.

The fix is to algebraically absorb the normalisation into the first linear
layer of the actor trunk:

    original: output = W @ ((obs - mean) / std) + b
    baked:    output = (W / std) @ obs + (b - W @ (mean / std))

``transfer_weights`` applies this transformation automatically whenever
``obs_normalization_stats`` is found in the checkpoint.
"""

from __future__ import annotations

import argparse
import torch

from kuafu.paths import resolve_artifact_path


# ── Checkpoint loading ────────────────────────────────────────────────────────

def load_robomimic_checkpoint(path: str) -> dict:
    """Return the model state-dict from a robomimic .pth checkpoint."""
    ckpt = torch.load(path, map_location="cpu")
    # robomimic wraps everything under "model" key
    return ckpt.get("model", ckpt)


def load_obs_normalization_stats(path: str) -> dict | None:
    """Extract per-dimension obs normalisation stats from a robomimic checkpoint.

    Returns a dict with keys ``"mean"`` and ``"std"`` (both 1-D tensors of
    length obs_dim) if the checkpoint contains normalisation statistics, or
    ``None`` if the checkpoint was saved without them (i.e. hdf5_normalize_obs
    was false during BC training).

    Robomimic stores these under:
        checkpoint["obs_normalization_stats"]["obs"]["state"]["mean"]
        checkpoint["obs_normalization_stats"]["obs"]["state"]["offset"]   # = std
    The outer key name varies; we also try the top-level fallback location.
    """
    ckpt = torch.load(path, map_location="cpu")

    # Try the most common location first
    stats = ckpt.get("obs_normalization_stats", None)
    if stats is None:
        return None

    # Robomimic 0.4 stores {"state": {"mean": ..., "std": ...}} while
    # older Isaac Lab checkpoints used {"obs": {"state": {...}}}.
    try:
        state_stats = stats["state"] if "state" in stats else stats["obs"]["state"]
        mean = torch.tensor(state_stats["mean"], dtype=torch.float32).flatten()
        # robomimic calls it "offset" (std) or "scale" depending on version
        std_key = next(k for k in ("std", "scale", "offset") if k in state_stats)
        std  = torch.tensor(state_stats[std_key], dtype=torch.float32).flatten()
        # Guard against zero std (constant features)
        std  = std.clamp(min=1e-6)
        print(f"[weight_transfer] Found obs_normalization_stats: mean {mean.shape}, std {std.shape}")
        return {"mean": mean, "std": std}
    except (KeyError, StopIteration) as e:
        print(f"[weight_transfer] WARNING: could not parse obs_normalization_stats ({e}). "
              "Proceeding without baking in normalization.")
        return None


def bake_normalization_into_first_layer(
    state_dict: dict,
    first_weight_key: str,
    first_bias_key: str,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> dict:
    """Absorb obs normalisation stats into the first linear layer's weights.

    Transforms:
        W' = W / std          (each column divided by the corresponding std)
        b' = b - W' @ mean    (bias adjusted so the intercept is correct)

    After this, the layer applied to *raw* obs produces the same output that
    the original layer would produce on *normalised* obs.
    """
    W = state_dict[first_weight_key].clone().float()  # (out, obs_dim)
    b = state_dict[first_bias_key].clone().float()    # (out,)

    mean = mean.to(W.device)
    std  = std.to(W.device)

    W_new = W / std.unsqueeze(0)          # broadcast over output dim
    b_new = b - (W_new @ mean)

    state_dict[first_weight_key] = W_new.to(state_dict[first_weight_key].dtype)
    state_dict[first_bias_key]   = b_new.to(state_dict[first_bias_key].dtype)

    print(f"[weight_transfer] Normalisation baked into {first_weight_key} / {first_bias_key}.")
    return state_dict


def inspect_checkpoint(path: str) -> None:
    """Print every layer name and tensor shape in a robomimic checkpoint.

    Run this first to confirm which key names your checkpoint actually uses,
    then update DEFAULT_KEY_MAP if the auto-detected names differ.
    """
    sd = load_robomimic_checkpoint(path)
    print(f"\n{'='*64}")
    print(f"  Robomimic checkpoint: {path}")
    print(f"{'='*64}")
    for k, v in sd.items():
        shape = tuple(v.shape) if hasattr(v, "shape") else type(v).__name__
        print(f"  {k:<70}  {shape}")
    print(f"{'='*64}\n")


# ── Key mapping ───────────────────────────────────────────────────────────────

# Maps robomimic BC state-dict keys → SKRL Actor state-dict keys.
#
# Architecture assumed on both sides:
#   BC   actor_layer_dims = [256, 512, 64],  obs "state" (49-dim, pass-through encoder)
#   SKRL trunk = Sequential(Linear(49,256), ELU, Linear(256,512), ELU, Linear(512,64), ELU)
#        mean_layer = Linear(64, action_dim)
#
# Robomimic MLP layer naming:
#   robomimic's MLP builds a nn.Sequential stored under self.nets["actor"].
#   Layers are indexed 0, 2, 4 (odd indices are ELU with no parameters).
#   The final mean projection lives under nets["mean"].
#
# If your checkpoint has DIFFERENT key names (visible after running
# inspect_checkpoint), edit this dict accordingly.

DEFAULT_KEY_MAP: dict[str, str] = {
    # ── Actor trunk ──────────────────────────────────────────────────────────
    "policy.nets.mlp._model.0.weight": "trunk.0.weight",
    "policy.nets.mlp._model.0.bias":   "trunk.0.bias",
    "policy.nets.mlp._model.2.weight": "trunk.2.weight",
    "policy.nets.mlp._model.2.bias":   "trunk.2.bias",
    "policy.nets.mlp._model.4.weight": "trunk.4.weight",
    "policy.nets.mlp._model.4.bias":   "trunk.4.bias",
    # ── Output mean layer ────────────────────────────────────────────────────
    "policy.nets.decoder.nets.mean.weight": "mean_layer.weight",
    "policy.nets.decoder.nets.mean.bias":   "mean_layer.bias",
}

# ── Transfer ──────────────────────────────────────────────────────────────────

def transfer_weights(
    actor_state_dict: dict,
    bc_path: str,
    key_map: dict[str, str] | None = None,
) -> tuple[dict, int]:
    """Copy BC weights into the SKRL actor state-dict, baking in obs normalisation.

    Args:
        actor_state_dict: Current state-dict of the SKRL Actor model.
        bc_path:          Path to the robomimic .pth checkpoint.
        key_map:          Override mapping; defaults to DEFAULT_KEY_MAP.

    Returns:
        (updated_state_dict, num_tensors_transferred)

    Notes:
        - Tensors are only copied when both key names exist AND shapes match.
        - A mismatch warning is printed for each skipped tensor so you can
          diagnose encoder-dimension issues quickly.
        - If the checkpoint contains ``obs_normalization_stats`` (produced when
          ``hdf5_normalize_obs: true``), the normalisation is automatically
          baked into the first linear layer so the transferred actor works
          correctly on raw observations from the SAC environment.
        - The critic networks are NOT touched; they are always trained from
          scratch so they reflect the true reward signal.
    """
    if key_map is None:
        key_map = DEFAULT_KEY_MAP

    bc_sd = load_robomimic_checkpoint(bc_path)
    transferred = 0

    for bc_key, skrl_key in key_map.items():
        if bc_key not in bc_sd:
            print(f"  [weight_transfer] SKIP (absent in BC ckpt) : {bc_key}")
            continue
        if skrl_key not in actor_state_dict:
            print(f"  [weight_transfer] SKIP (absent in actor)   : {skrl_key}")
            continue

        bc_t   = bc_sd[bc_key]
        skrl_t = actor_state_dict[skrl_key]

        if bc_t.shape != skrl_t.shape:
            print(
                f"  [weight_transfer] SHAPE MISMATCH           : "
                f"{bc_key} {tuple(bc_t.shape)}  ≠  {skrl_key} {tuple(skrl_t.shape)}\n"
                f"    Hint: encoder may add/remove dimensions. "
                f"Run inspect_checkpoint() and compare first-layer input dims."
            )
            continue

        actor_state_dict[skrl_key] = bc_t.clone()
        transferred += 1
        print(f"  [weight_transfer] OK  {bc_key} → {skrl_key}  {tuple(bc_t.shape)}")

    print(f"\n  [weight_transfer] Transferred {transferred}/{len(key_map)} tensors.")

    # ── Bake obs normalisation into first layer (hdf5_normalize_obs: true) ───
    # The first SKRL trunk layer corresponds to the first entry in key_map
    # whose value targets "trunk.0.weight" / "trunk.0.bias".
    norm_stats = load_obs_normalization_stats(bc_path)
    if norm_stats is not None and transferred > 0:
        first_w_key = "trunk.0.weight"
        first_b_key = "trunk.0.bias"
        if first_w_key in actor_state_dict and first_b_key in actor_state_dict:
            actor_state_dict = bake_normalization_into_first_layer(
                actor_state_dict,
                first_w_key,
                first_b_key,
                norm_stats["mean"],
                norm_stats["std"],
            )
        else:
            print(
                "[weight_transfer] WARNING: could not locate trunk.0.weight/bias "
                "to bake normalisation — actor may behave incorrectly on raw obs."
            )
    elif norm_stats is None:
        print(
            "[weight_transfer] No obs_normalization_stats in checkpoint "
            "(hdf5_normalize_obs was false) — no baking needed."
        )

    print()
    return actor_state_dict, transferred


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect a robomimic BC checkpoint (no Isaac Sim required)."
    )
    parser.add_argument("checkpoint", help="Path to robomimic .pth file")
    args = parser.parse_args()
    inspect_checkpoint(str(resolve_artifact_path(args.checkpoint, "Robomimic checkpoint")))
