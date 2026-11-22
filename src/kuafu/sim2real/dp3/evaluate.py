"""Paired simulator evaluation; monitor stages are diagnostics outside policy inputs."""

from collections import deque
import importlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from kuafu.paths import resolve_artifact_path

from .artifacts import file_digest, json_digest, write_json
from .dataset import StageDataset
from .metrics import add_confusion, stage_metrics
from .train import load_checkpoint, load_policy, offline_metrics, seed_all


def validate_protocol(protocol):
    required = {"task", "conditions", "seeds", "max_steps", "chunk_exec"}
    if not required.issubset(protocol):
        raise ValueError(f"Evaluation protocol needs {sorted(required)}")
    conditions = protocol["conditions"]
    if "nominal" not in conditions or len(conditions) < 2:
        raise ValueError("Declare nominal and at least one controlled varied condition")
    if len({json_digest(value) for value in conditions.values()}) != len(conditions):
        raise ValueError("Evaluation conditions must describe distinct settings")
    if not protocol["seeds"] or len(set(protocol["seeds"])) != len(protocol["seeds"]):
        raise ValueError("Use distinct, explicit held-out simulator seeds")
    if any(type(seed) is not int or seed < 0 for seed in protocol["seeds"]):
        raise ValueError("Simulator seeds must be nonnegative integers")
    if any(type(protocol[key]) is not int or protocol[key] < 1 for key in ("max_steps", "chunk_exec")):
        raise ValueError("Evaluation step/chunk budgets must be positive integers")


def _synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


@torch.inference_mode()
def rollout_episode(policy, adapter, *, seed, condition, max_steps, chunk_exec, action_stride=1, device="cpu"):
    """Adapters return observations and diagnostic stage IDs as separate fields.

    reset(seed, condition) returns obs, stage_id, initial_state_id. step(action)
    returns obs, stage_id, done, success, and optional advanced_stages captured
    before automatic resets. Each call controls one environment.
    """
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    if not 1 <= chunk_exec <= policy.n_action_steps:
        raise ValueError("chunk_exec exceeds the policy's predicted action chunk")
    if action_stride != 1 and chunk_exec != 1:
        raise ValueError("Strided-action checkpoints require receding-horizon chunk_exec=1")
    seed_all(seed)
    current = adapter.reset(seed=seed, condition=condition)
    if not current.get("initial_state_id"):
        raise ValueError("Simulator adapter must identify the actual reset state")
    initial_state_id = current["initial_state_id"]
    history = {key: deque([value] * policy.n_obs_steps, maxlen=policy.n_obs_steps)
               for key, value in current["obs"].items()}
    queue = deque()
    confusion = np.zeros((policy.num_stages, policy.num_stages), dtype=np.int64)
    latencies, trace, stages = [], [], []
    success, outcome = False, "step_limit"
    advanced, entered = set(), set()
    failure_stage = None
    for step in range(max_steps):
        stage = int(current["stage_id"])
        if not 0 <= stage < policy.num_stages:
            raise ValueError("Simulator decision stage is outside the dataset vocabulary")
        stages.append(stage)
        entered.add(stage)
        predicted = None
        if not queue:
            obs = {key: torch.stack([torch.as_tensor(value, dtype=torch.float32, device=device)
                                     for value in values]).unsqueeze(0) for key, values in history.items()}
            _synchronize(device)
            started = time.perf_counter()
            result = policy.predict_action(obs)
            _synchronize(device)
            latencies.append((time.perf_counter() - started) * 1000)
            if "stage_logits" in result:
                predicted = int(result["stage_logits"].argmax(-1).item())
                add_confusion(confusion, [stage], [predicted])
            actions = result["action"][0, :chunk_exec].detach().cpu()
            if not torch.isfinite(actions).all():
                raise FloatingPointError("Policy produced nonfinite simulator actions")
            queue.extend(actions)
        trace.append({"step": step, "monitor_stage": stage, "predicted_stage": predicted})
        current = adapter.step(queue.popleft())
        # Cumulative monitor counters preserve transitions through automatic resets.
        newly_advanced = {int(value) for value in current.get("advanced_stages", [])}
        if any(value < 0 or value >= policy.num_stages for value in newly_advanced):
            raise ValueError("Transition diagnostic is outside the stage vocabulary")
        advanced.update(newly_advanced)
        entered.update(newly_advanced)
        entered.update(value + 1 for value in newly_advanced if value + 1 < policy.num_stages)
        next_stage = int(current["stage_id"])
        if 0 <= next_stage < policy.num_stages:
            entered.add(next_stage)
            failure_stage = next_stage
        else:
            failure_stage = max([stage, *[min(value + 1, policy.num_stages - 1) for value in newly_advanced]])
        trace[-1]["advanced_stages"] = sorted(newly_advanced)
        if current["done"]:
            success = bool(current["success"])
            outcome = "success" if success else "terminated_without_success"
            break
        if current["success"]:
            raise ValueError("Adapter reported success without ending the attempted episode")
        for key, value in current["obs"].items():
            history[key].append(value)
    transitions = []
    for stage in sorted(entered):
        reached_next = (stage in advanced or any(value > stage for value in entered)) if stage < policy.num_stages - 1 else success
        transitions.append({"stage": stage, "entered": 1, "reached_later_stage_or_final_success": int(reached_next),
                            "episode_failed_before_advancing": int(not success and not reached_next)})
    return {"seed": seed, "condition": condition, "initial_state_id": initial_state_id,
            "success": success, "attempts": 1, "steps": len(stages), "outcome": outcome,
            "failure_stage": None if success else failure_stage, "transitions": transitions,
            "stage": stage_metrics(confusion) if policy.variant != "A" else None,
            "inference_latency_ms": latencies, "trace": trace}


def summarize_episodes(episodes, num_stages):
    successes = sum(episode["success"] for episode in episodes)
    times = [value for episode in episodes for value in episode["inference_latency_ms"]]
    confusion = np.zeros((num_stages, num_stages), dtype=np.int64)
    has_stages = any(episode["stage"] is not None for episode in episodes)
    if has_stages:
        for episode in episodes:
            confusion += np.asarray(episode["stage"]["confusion"])
    transitions = [{"stage": stage, "entered": 0, "reached_later_stage_or_final_success": 0,
                    "episode_failed_before_advancing": 0} for stage in range(num_stages)]
    for episode in episodes:
        for row in episode["transitions"]:
            for key in row:
                if key != "stage":
                    transitions[row["stage"]][key] += row[key]
    return {"successes": successes, "attempts": len(episodes), "success_rate": successes / len(episodes),
            "stage": stage_metrics(confusion) if has_stages else None, "transitions": transitions,
            "latency_ms": {"mean": float(np.mean(times)), "p50": float(np.median(times)),
                           "p95": float(np.percentile(times, 95)), "decisions": len(times)}}


def evaluate_experiment(experiment_dir, dataset_path, protocol_path, output, *, adapter_factory, device="cuda:0"):
    protocol = json.loads(Path(protocol_path).read_text())
    validate_protocol(protocol)
    experiment_dir, output = Path(experiment_dir), resolve_artifact_path(output, "Stage evaluation output")
    campaign = json.loads((experiment_dir / "experiment.json").read_text())
    config = campaign["config"]
    dataset = StageDataset(dataset_path, **config["window"], partition="test", verify_content=True)
    if dataset.manifest["dataset_id"] != config["dataset_id"] or protocol["task"] != dataset.manifest["task"]:
        raise ValueError("Evaluation task/dataset differs from training")
    if output.exists():
        raise FileExistsError(f"Choose a new evaluation output: {output}")
    # Complete the checkpoint preflight before launching Isaac or starting any attempts.
    checkpoints = {}
    for seed in config["seeds"]:
        for variant in "ABC":
            path = experiment_dir / f"seed_{seed}" / variant / "best.ckpt"
            if not path.is_file():
                raise FileNotFoundError(f"Matched comparison requires {path}")
            latest = load_checkpoint(path.with_name("latest.ckpt"))
            metadata = latest["experiment"]
            if (metadata["comparison_id"] != campaign["comparison_id"] or metadata["seed"] != seed
                    or metadata["variant"] != variant or metadata["epoch"] != config["epochs"] - 1):
                raise ValueError(f"Matched comparison requires the complete training budget: {path.parent}")
            del latest
            checkpoints[seed, variant] = path
    module, name = adapter_factory.split(":", 1)
    factory = getattr(importlib.import_module(module), name)
    adapter = factory(experiment=config, dataset=dataset.manifest, protocol=protocol, device=device)
    output.mkdir(parents=True)
    records, initial_states = [], {}
    protocol_id = json_digest(protocol)
    try:
        for (training_seed, variant), checkpoint in checkpoints.items():
            policy, payload = load_policy(checkpoint, device)
            metadata = payload["experiment"]
            if metadata["comparison_id"] != campaign["comparison_id"] or metadata["seed"] != training_seed or metadata["variant"] != variant:
                raise ValueError(f"Checkpoint identity mismatch: {checkpoint}")
            offline = offline_metrics(policy, dataset, batch_size=config["batch_size"], device=device, seed=training_seed + 917)
            for condition_name, condition in protocol["conditions"].items():
                episodes = []
                for rollout_seed in protocol["seeds"]:
                    episode = rollout_episode(
                        policy, adapter, seed=rollout_seed, condition=condition,
                        max_steps=protocol["max_steps"], chunk_exec=protocol["chunk_exec"],
                        action_stride=config["window"]["action_stride"], device=device,
                    )
                    write_json(output / f"attempt_{training_seed}_{variant}_{len(records)}_{rollout_seed}.json", episode)
                    key = condition_name, rollout_seed
                    identity = episode["initial_state_id"]
                    if key in initial_states and initial_states[key] != identity:
                        raise ValueError(f"Unpaired simulator reset state: {key}")
                    initial_states[key] = identity
                    episodes.append(episode)
                row = {"variant": variant, "training_seed": training_seed, "condition": condition_name,
                       "checkpoint": str(checkpoint), "checkpoint_sha256": file_digest(checkpoint),
                       "parameters": metadata["parameters"], "offline_heldout": offline,
                       "rollout": summarize_episodes(episodes, policy.num_stages), "episodes": episodes}
                records.append(row)
                write_json(output / "progress.json", records)
            del policy
    finally:
        adapter.close()
    aggregates = []
    for variant in "ABC":
        for condition_name in protocol["conditions"]:
            runs = [row["rollout"] for row in records if row["variant"] == variant and row["condition"] == condition_name]
            rates = [run["success_rate"] for run in runs]
            aggregates.append({"variant": variant, "condition": condition_name,
                               "successes": sum(run["successes"] for run in runs),
                               "attempts": sum(run["attempts"] for run in runs),
                               "mean_training_seed_success_rate": float(np.mean(rates)),
                               "std_training_seed_success_rate": float(np.std(rates)), "training_seeds": len(runs)})
    result = {"comparison_id": campaign["comparison_id"], "protocol_id": protocol_id, "protocol": protocol,
              "adapter": adapter_factory, "hardware": torch.cuda.get_device_name(device) if str(device).startswith("cuda") else "CPU",
              "latency_method": "synchronized predict_action wall time, including first call; excludes observation preparation",
              "transition_definition": "entered stage followed by a later monitor stage or final success; failure counts episodes ending before that advance",
              "aggregate": aggregates, "runs": records}
    write_json(output / "results.json", result)
    return result
