"""Offline ensemble fitting, held-out evaluation, and checkpoint provenance."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch

from kuafu.dynamics.data import Transitions, read_transitions, split_episodes
from kuafu.dynamics.model import ProbabilisticEnsemble
from kuafu.paths import resolve_artifact_path


@dataclass
class TrainConfig:
    dataset: str
    output: str
    task: str = ""
    obs_key: str = "full_state"
    action_key: str = "relative_joint"
    exclude_last: int = 1
    skip_first: int = 1
    members: int = 5
    hidden_dim: int = 200
    hidden_layers: int = 4
    epochs: int = 100
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 15
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    seed: int = 42
    device: str = "cpu"
    threads: int = 4
    rollout_horizon: int = 20
    rollout_windows: int = 256
    particles: int = 20

    def validate(self) -> None:
        for field in ("members", "hidden_dim", "hidden_layers", "epochs", "batch_size", "patience",
                      "threads", "rollout_horizon", "rollout_windows", "particles"):
            if getattr(self, field) < 1:
                raise ValueError(f"{field} must be positive")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


@torch.no_grad()
def evaluate(model: ProbabilisticEnsemble, tensors: tuple, batch_size: int = 4096) -> dict:
    """Report prediction error, persistence baseline, and moment-matched coverage."""
    model.eval()
    device = model.input_mean.device
    count, dim = tensors[0].shape
    sums = {key: torch.zeros(dim, device=device, dtype=torch.float64)
            for key in ("squared_error", "baseline", "coverage", "aleatoric", "epistemic")}
    nll = torch.zeros(model.members, device=device, dtype=torch.float64)
    for start in range(0, count, batch_size):
        states, actions, targets = (v[start:start + batch_size].to(device) for v in tensors)
        means, variances = model.predict(states, actions)
        average = means.mean(0)
        aleatoric = variances.mean(0)
        epistemic = means.var(0, unbiased=False)
        error = targets - average
        for key, values in (("squared_error", error.square()), ("baseline", (targets - states).square()),
                            ("coverage", (error.abs() <= 1.96 * (aleatoric + epistemic).sqrt()).float()),
                            ("aleatoric", aleatoric), ("epistemic", epistemic)):
            sums[key] += values.double().sum(0)
        nll += model.nll(states, actions, targets).double() * len(states)
    mse = sums["squared_error"] / count
    baseline = sums["baseline"] / count
    scales = model.delta_std.double().square()
    return {
        "transitions": count, "member_nll_normalized": (nll / count).tolist(),
        "normalized_rmse": (mse / scales).mean().sqrt().item(),
        "persistence_normalized_rmse": (baseline / scales).mean().sqrt().item(),
        "rmse_per_dim": mse.sqrt().tolist(), "persistence_rmse_per_dim": baseline.sqrt().tolist(),
        "moment_matched_95pct_coverage": (sums["coverage"] / count).mean().item(),
        "aleatoric_variance_per_dim": (sums["aleatoric"] / count).tolist(),
        "epistemic_variance_per_dim": (sums["epistemic"] / count).tolist(),
    }


@torch.no_grad()
def evaluate_rollouts(model: ProbabilisticEnsemble, data: Transitions, names: list[str], *,
                      horizon: int, windows: int, particles: int, seed: int) -> dict:
    """Evaluate TS-infinity on recorded actions, without future-state feedback."""
    rng = np.random.default_rng(seed)
    selected = set(names)
    candidates = []
    for episode in data.episodes:
        if episode.name not in selected:
            continue
        # Terminal filtering can leave gaps; such windows must never bridge resets.
        for start in range(len(episode.states) - horizon + 1):
            if int(episode.steps[start + horizon - 1] - episode.steps[start]) == horizon - 1:
                candidates.append((episode, start))
    if not candidates:
        return {"windows": 0, "reason": "No continuous episodes long enough for requested horizon"}
    indices = rng.choice(len(candidates), min(windows, len(candidates)), replace=False)
    chosen = [candidates[i] for i in indices]
    device = model.input_mean.device
    states = torch.stack([e.states[s] for e, s in chosen]).to(device)
    actions = torch.stack([e.actions[s:s + horizon] for e, s in chosen]).to(device)
    targets = torch.stack([e.next_states[s:s + horizon] for e, s in chosen]).to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    predictions = model.rollout(states, actions, particles=particles, generator=generator)[:, :, 1:].mean(1)
    result = {"windows": len(chosen), "particles": particles, "propagation": "tsinf", "horizons": {}}
    for step in sorted({1, min(5, horizon), min(10, horizon), horizon}):
        mse = (predictions[:, step - 1] - targets[:, step - 1]).square().mean(0)
        baseline = (states - targets[:, step - 1]).square().mean(0)
        result["horizons"][str(step)] = {
            "normalized_rmse": (mse / model.delta_std.square()).mean().sqrt().item(),
            "persistence_normalized_rmse": (baseline / model.delta_std.square()).mean().sqrt().item(),
            "rmse_per_dim": mse.sqrt().tolist(), "persistence_rmse_per_dim": baseline.sqrt().tolist(),
        }
    return result


def run_training(config: TrainConfig) -> dict:
    """Train bootstrapped members, select each on validation NLL, then test once."""
    config.validate()
    output = resolve_artifact_path(config.output, "Dynamics output")
    if output.exists():
        raise ValueError(f"Use a new output directory; refusing to overwrite {output}")
    torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    if str(config.device).startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    data = read_transitions(config.dataset, obs_key=config.obs_key, action_key=config.action_key,
                            exclude_last=config.exclude_last, skip_first=config.skip_first)
    split = split_episodes(data, seed=config.seed, val_fraction=config.val_fraction, test_fraction=config.test_fraction)
    train = tuple(t.to(config.device) for t in data.tensors(split["train"]))
    validation = data.tensors(split["validation"])
    model = ProbabilisticEnsemble(data.metadata["state_dim"], data.metadata["action_dim"],
                                  members=config.members, hidden_dim=config.hidden_dim,
                                  hidden_layers=config.hidden_layers).to(config.device)
    model.fit_normalizers(*train)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    # One independent bootstrap dataset per member; reshuffle its order each epoch.
    generator = torch.Generator(device=config.device).manual_seed(config.seed + 1)
    bootstrap = torch.randint(len(train[0]), (config.members, len(train[0])),
                              device=config.device, generator=generator)
    best_nll = torch.full((config.members,), float("inf"), device=config.device)
    best_epoch = torch.zeros(config.members, dtype=torch.long, device=config.device)
    best_weights = {key: value.detach().clone() for key, value in model.state_dict().items()}
    output.mkdir(parents=True)
    run = {"config": asdict(config), "dataset": data.metadata, "split": split,
           "model_config": model.config, "torch_version": str(torch.__version__),
           "variance_bounds": "fixed soft log-variance bounds [-10, 0.5] in normalized delta units",
           "parameters": sum(p.numel() for p in model.parameters())}
    _json(output / "run.json", run)
    print(json.dumps({"output": str(output), "dataset": data.metadata,
                      "split_episodes": {key: len(value) for key, value in split.items()}}, allow_nan=False), flush=True)
    started = time.monotonic()
    stale = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        order = torch.randperm(len(train[0]), device=config.device, generator=generator)
        train_nll = torch.zeros(config.members, device=config.device)
        for start in range(0, len(order), config.batch_size):
            index = bootstrap[:, order[start:start + config.batch_size]]
            batch = tuple(t[index] for t in train)
            optimizer.zero_grad(set_to_none=True)
            losses = model.nll(*batch)
            if not torch.isfinite(losses).all():
                raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}")
            losses.sum().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0, error_if_nonfinite=True)
            optimizer.step()
            train_nll += losses.detach() * index.shape[1]
        metrics = evaluate(model, validation)
        values = torch.tensor(metrics["member_nll_normalized"], device=config.device)
        if not torch.isfinite(values).all():
            raise FloatingPointError(f"Nonfinite validation loss at epoch {epoch}")
        improved = values < best_nll
        significant = values < best_nll - 1e-3
        for key, value in model.state_dict().items():
            if key.startswith("layers."):
                best_weights[key][improved] = value.detach()[improved]
        best_epoch[improved] = epoch
        best_nll = torch.minimum(values, best_nll)
        stale = 0 if significant.any().item() else stale + 1
        record = {"epoch": epoch, "train_nll": (train_nll / len(train[0])).tolist(),
                  "validation_nll": values.tolist(), "validation_normalized_rmse": metrics["normalized_rmse"],
                  "elapsed_seconds": time.monotonic() - started}
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        checkpoint = {"format": "kuafu-pets-v1", "model_config": model.config,
                      "model": {key: value.cpu() for key, value in best_weights.items()},
                      "run": run, "best_epochs": best_epoch.tolist(), "best_validation_nll": best_nll.tolist()}
        _save(output / "best.pt", checkpoint)
        print(json.dumps(record, allow_nan=False), flush=True)
        if stale >= config.patience:
            break
    model.load_state_dict(best_weights)
    report = {"epochs_completed": epoch, "elapsed_seconds": time.monotonic() - started,
              "best_epochs": best_epoch.tolist(), "validation": evaluate(model, validation),
              "test": evaluate(model, data.tensors(split["test"])),
              "test_rollout": evaluate_rollouts(model, data, split["test"], horizon=config.rollout_horizon,
                  windows=config.rollout_windows, particles=config.particles, seed=config.seed + 2)}
    _json(output / "evaluation.json", report)
    print(json.dumps({"finished": str(output / "best.pt"), "test_normalized_rmse": report["test"]["normalized_rmse"],
                      "persistence_normalized_rmse": report["test"]["persistence_normalized_rmse"]}), flush=True)
    return report
