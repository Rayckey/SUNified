from __future__ import annotations

import types
import copy
from typing import Any

def _to_int_float_dict(d: dict) -> dict[int, float]:
    """Convert a dict with arbitrary numeric-like keys/values to int->float."""
    out: dict[int, float] = {}
    for k, v in d.items():
        out[int(k)] = float(v)
    return out


def _weight_to_float(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "mean"):
        value = value.mean()
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def _verify_residual_weights(program_runtime, stage_id: int, expected: dict[int, float]) -> None:
    actual = program_runtime.get_residual_weights(int(stage_id))
    missing = sorted(set(expected) - set(actual))
    if missing:
        raise RuntimeError(
            f"[residual_rl] residual weights were not stored for stage_id={stage_id}: missing={missing}"
        )
    mismatched: dict[int, tuple[float, float]] = {}
    for idx, expected_value in expected.items():
        actual_value = _weight_to_float(actual[int(idx)])
        if abs(actual_value - float(expected_value)) > 1e-6:
            mismatched[int(idx)] = (actual_value, float(expected_value))
    if mismatched:
        raise RuntimeError(
            f"[residual_rl] residual weights mismatch for stage_id={stage_id}: {mismatched}"
        )


def _verify_stage_parameters(
    program_runtime, *, stage_id: int, expected: dict[int, dict[str, float]]
) -> None:
    actual = program_runtime.get_constraints_by_stage(int(stage_id))
    terms = actual["Constraints"]
    mismatched: dict[int, dict[str, tuple[object, float]]] = {}
    for term_index, patch in expected.items():
        args = terms[int(term_index)].get("args", {})
        for key, expected_value in patch.items():
            actual_value = args.get(key)
            if actual_value is None or abs(float(actual_value) - float(expected_value)) > 1e-6:
                mismatched.setdefault(int(term_index), {})[key] = (actual_value, float(expected_value))
    if mismatched:
        raise RuntimeError(
            f"[residual_rl] stage parameter mismatch for stage_id={stage_id}: {mismatched}"
        )


def update_weights(
    env_u,
    *,
    stage_id: int | None,
    weights: dict[int, float] | None,
    all_stages_weights: dict[int, dict[int, float]] | None,
) -> None:
    """Apply residual weights through the program_runtime API.

    This helper is shared by residual training and evaluation so that
    "update weights" and "update parameters" go through the same program_runtime
    access path.
    """
    if all_stages_weights is None and (stage_id is None or weights is None):
        return

    program_runtime = env_u.program_runtime

    if all_stages_weights is not None:
        print(f"[residual_rl] Applying all-stages weights (stages={list(all_stages_weights.keys())})", flush=True)
        for sid, w in all_stages_weights.items():
            normalized = _to_int_float_dict(dict(w))
            program_runtime.set_residual_weights(int(sid), normalized)
            _verify_residual_weights(program_runtime, int(sid), normalized)
            print(
                f"[residual_rl] Verified residual weights: stage_id={int(sid)} weights={normalized}",
                flush=True,
            )
    elif stage_id is not None and weights is not None:
        normalized = _to_int_float_dict(dict(weights))
        print(
            f"[residual_rl] Applying stage weights: stage_id={int(stage_id)} num_components={len(normalized)}",
            flush=True,
        )
        program_runtime.set_residual_weights(int(stage_id), normalized)
        _verify_residual_weights(program_runtime, int(stage_id), normalized)
        print(
            f"[residual_rl] Verified residual weights: stage_id={int(stage_id)} weights={normalized}",
            flush=True,
        )


def update_parameters(env_u, *, stage_id: int, parameters: dict[int, dict[str, object]] | None) -> None:
    """Apply per-constraint arg overrides for a given stage.

    Behavior:
    - Only modifies a small allowlist of numeric args.
    - Fails fast on invalid indices / value types so caller can debug issues.
    """
    if not parameters or len(parameters) == 0: # no update needed
        return

    program_runtime = env_u.program_runtime

    base = program_runtime.get_constraints_by_stage(int(stage_id))

    modified = copy.deepcopy(base)
    terms = modified["Constraints"]

    # Allowlist of args that we permit to be changed.
    allowed_scalar = {"tolerance_m", "tolerance_rad"}
    verified_patch: dict[int, dict[str, float]] = {}

    for idx, patch in parameters.items():
        term_index = int(idx)
        if not (0 <= term_index < len(terms)):
            raise IndexError(
                f"[residual_rl] update_parameters: term index {term_index} out of range for stage_id={stage_id} "
                f"(num_terms={len(terms)})"
            )

        term = terms[term_index]
        if not isinstance(term, dict):
            raise TypeError(
                f"[residual_rl] update_parameters: term at index {term_index} for stage_id={stage_id} "
                f"has unexpected type {type(term)!r}, expected dict"
            )

        args = term.get("args", {})
        if not isinstance(args, dict):
            raise TypeError(
                f"[residual_rl] update_parameters: 'args' for term index {term_index} and stage_id={stage_id} "
                f"has unexpected type {type(args)!r}, expected dict"
            )
        term["args"] = args

        for k, v in patch.items():
            if k in allowed_scalar:
                fv = float(v)  # type: ignore[arg-type]
                # enforce sane positive range and fail loudly instead of clamping
                if not (0.0 <= fv <= 10.0):
                    raise ValueError(
                        f"[residual_rl] update_parameters: value {fv} for key '{k}' at term index {term_index} "
                        f"and stage_id={stage_id} is out of allowed range [0.0, 10.0]"
                    )
                args[k] = fv
                verified_patch.setdefault(term_index, {})[k] = fv

    program_runtime.reload_constraints(constraints_list=[modified], stage_indices=[int(stage_id)])
    _verify_stage_parameters(program_runtime, stage_id=int(stage_id), expected=verified_patch)
    print(
        f"[residual_rl] Verified stage parameters: stage_id={int(stage_id)} parameters={verified_patch}",
        flush=True,
    )

def update_all_parameters(
    env_u, *, all_stages_parameters: dict[int, dict[int, dict[str, object]]] | None
) -> None:
    """Apply constraint arg overrides for multiple stages.

    This is used to ensure *all prior stages* are consistent when tuning a later stage.
    """
    if not all_stages_parameters:
        return
    print(f"[residual_rl] Applying all-stages parameters (stages={list(dict(all_stages_parameters).keys())})", flush=True)
    for sid, params in dict(all_stages_parameters).items():
        update_parameters(env_u, stage_id=int(sid), parameters=params)

def apply_updates_and_wrap_reset(
    env,
    *,
    stage_id: int,
    all_stages_weights: dict[int, dict[int, float]] | None,
    progress_metric_expr: str | None,
    all_stages_parameters: dict[int, dict[int, dict[str, object]]] | None = None,
) -> None:
    import torch

    # The rolling success-rate window is intentionally epoch-bucketed. Env steps can
    # complete many tiny batches of episodes; a per-step window is too twitchy.
    _episode_rate_window_epochs = 10
    env_u = env.unwrapped
    # Reset per-run best-reward trackers so each training trial starts clean.
    # These are kept for progress expressions that depend on "best reward so far".
    n = int(env_u.num_envs)
    dev = env_u.device
    # Use -inf so this works even if rewards can be negative.
    env_u._Kuafu_best_reward_per_env = torch.full((n,), float("-inf"), device=dev, dtype=torch.float32)
    
    # Parameters need to happen first.
    # When tuning later stages, explicitly re-apply all prior stages' parameters so workers
    # do not drift based on the last trial they happened to run.
    update_all_parameters(env_u, all_stages_parameters=all_stages_parameters)

    update_weights(
        env_u,
        stage_id=stage_id,
        weights=None,
        all_stages_weights=all_stages_weights,
    )

    # Whenever weights/parameters are reset, stage stats must be reset so the new run collects from scratch.
    env_u.clear_stage_transition_stats()

    # Persist the progress expression for worker selection and optional episode logging.
    setattr(env_u, "_Kuafu_progress_metric_expr", progress_metric_expr)
    _prev_ts = int(env_u.get_true_success_count()) if hasattr(env_u, "get_true_success_count") else 0
    _prev_fc = int(env_u.get_false_complete_count()) if hasattr(env_u, "get_false_complete_count") else 0
    _prev_to = int(env_u.get_timeout_count()) if hasattr(env_u, "get_timeout_count") else 0
    setattr(
        env_u,
        "_Kuafu_episode_rate_state",
        {
            "prev_ts": _prev_ts,
            "prev_fc": _prev_fc,
            "prev_to": _prev_to,
            "last_epoch_rate": 0.0,
            "ring": [],
            "ring_sum_ts": 0,
            "ring_sum_total": 0,
            "ema": None,
            "window_epochs": _episode_rate_window_epochs,
            "ema_decay": 0.95,
        },
    )
    # Best-effort: reward reflection via per-constraint satisfaction+reward stats.
    # We throttle these calls because they can be expensive with many envs.
    setattr(env_u, "_Kuafu_constraint_stats_stage_id", stage_id)
    if not hasattr(env_u, "_Kuafu_constraint_stats_stride_steps"):
        setattr(env_u, "_Kuafu_constraint_stats_stride_steps", 50)
    if not hasattr(env_u, "_Kuafu_constraint_stats_step_counter"):
        setattr(env_u, "_Kuafu_constraint_stats_step_counter", 0)
    if not hasattr(env_u, "_step_original"):
        env_u._step_original = env_u.step

        def _step_with_kuafu_logging(self, action):
            obs, rew, terminated, truncated, extras = self._step_original(action)
            # Update "best reward so far" per environment.
            # This tracks the maximum *instantaneous* reward seen in each env over the whole training run.
            # (Per-episode tracking would require integrating with reset semantics; this is simpler and robust.)
            if not torch.is_tensor(rew):
                raise TypeError(f"[residual_rl] Expected rew to be a torch.Tensor, got {type(rew)!r}")
            rew_flat = rew.detach()
            if rew_flat.ndim > 1:
                rew_flat = rew_flat.view(-1)
            if rew_flat.numel() != self._Kuafu_best_reward_per_env.numel():
                raise ValueError(
                    "[residual_rl] Reward shape mismatch: rew has numel={rw} but _Kuafu_best_reward_per_env has numel={bw}".format(
                        rw=int(rew_flat.numel()), bw=int(self._Kuafu_best_reward_per_env.numel())
                    )
                )
            self._Kuafu_best_reward_per_env = torch.maximum(
                self._Kuafu_best_reward_per_env.to(rew_flat.device), rew_flat.to(self._Kuafu_best_reward_per_env.dtype)
            )
            if "log" not in self.extras:
                self.extras["log"] = {}
            log = self.extras["log"]

            expr = getattr(self, "_Kuafu_progress_metric_expr", None)
            if expr:
                log["Kuafu/progress_rate"] = eval(expr, {"torch": torch}, {"self": self})

            if (
                hasattr(self, "get_true_success_count")
                and hasattr(self, "get_false_complete_count")
                and hasattr(self, "get_timeout_count")
            ):
                ts = int(self.get_true_success_count())
                fc = int(self.get_false_complete_count())
                to = int(self.get_timeout_count())
                total = ts + fc + to
                true_rate = ts / float(total) if total > 0 else 0.0
                false_rate = fc / float(total) if total > 0 else 0.0
                timeout_rate = to / float(total) if total > 0 else 0.0

                state = getattr(self, "_Kuafu_episode_rate_state", None)
                if not isinstance(state, dict):
                    state = {
                        "prev_ts": ts,
                        "prev_fc": fc,
                        "prev_to": to,
                        "last_epoch_rate": 0.0,
                        "ring": [],
                        "ring_sum_ts": 0,
                        "ring_sum_total": 0,
                        "ema": None,
                        "window_epochs": _episode_rate_window_epochs,
                        "ema_decay": 0.95,
                    }
                    setattr(self, "_Kuafu_episode_rate_state", state)

                d_ts = ts - int(state.get("prev_ts", 0))
                d_fc = fc - int(state.get("prev_fc", 0))
                d_to = to - int(state.get("prev_to", 0))
                if d_ts < 0 or d_fc < 0 or d_to < 0:
                    d_ts = d_fc = d_to = 0
                    state.update(
                        {
                            "prev_ts": ts,
                            "prev_fc": fc,
                            "prev_to": to,
                            "last_epoch_rate": 0.0,
                            "ring_sum_ts": 0,
                            "ring_sum_total": 0,
                            "ema": None,
                        }
                    )
                    state["ring"] = []

                d_total = d_ts + d_fc + d_to
                if d_total > 0:
                    state["prev_ts"] = ts
                    state["prev_fc"] = fc
                    state["prev_to"] = to

                    epoch_key = getattr(self, "_Kuafu_current_train_epoch", None)
                    if epoch_key is None:
                        epoch_steps = int(getattr(self, "_Kuafu_train_epoch_steps", 0) or 0)
                        if epoch_steps > 0 and hasattr(self, "common_step_counter"):
                            step_counter = getattr(self, "common_step_counter")
                            if torch.is_tensor(step_counter):
                                step_counter = int(step_counter.item())
                            else:
                                step_counter = int(step_counter)
                            epoch_key = int(step_counter // epoch_steps)
                        else:
                            epoch_key = int(state.get("fallback_bucket", -1)) + 1
                            state["fallback_bucket"] = epoch_key
                    else:
                        epoch_key = int(epoch_key)

                    ring = state.setdefault("ring", [])
                    if ring and int(ring[-1][0]) == int(epoch_key):
                        _, prev_ts_bucket, prev_total_bucket = ring[-1]
                        bucket_ts = int(prev_ts_bucket) + d_ts
                        bucket_total = int(prev_total_bucket) + d_total
                        ring[-1] = (int(epoch_key), bucket_ts, bucket_total)
                    else:
                        bucket_ts = d_ts
                        bucket_total = d_total
                        ring.append((int(epoch_key), bucket_ts, bucket_total))

                    state["ring_sum_ts"] = int(state.get("ring_sum_ts", 0)) + d_ts
                    state["ring_sum_total"] = int(state.get("ring_sum_total", 0)) + d_total
                    while len(ring) > int(state.get("window_epochs", _episode_rate_window_epochs)):
                        _, old_ts, old_total = ring.pop(0)
                        state["ring_sum_ts"] = int(state.get("ring_sum_ts", 0)) - int(old_ts)
                        state["ring_sum_total"] = int(state.get("ring_sum_total", 0)) - int(old_total)

                    epoch_rate = bucket_ts / float(bucket_total)
                    state["last_epoch_rate"] = epoch_rate
                    if state.get("ema") is None:
                        state["ema"] = epoch_rate
                    else:
                        decay = float(state.get("ema_decay", 0.95))
                        state["ema"] = decay * float(state["ema"]) + (1.0 - decay) * epoch_rate

                ring_total = int(state.get("ring_sum_total", 0))
                window_rate = int(state.get("ring_sum_ts", 0)) / float(ring_total) if ring_total > 0 else 0.0
                ema_rate = float(state["ema"]) if state.get("ema") is not None else 0.0
                epoch_rate = float(state.get("last_epoch_rate", 0.0))
                tensor = lambda v: torch.tensor(float(v), dtype=torch.float32, device=self.device)
                log["Kuafu/true_success_count"] = tensor(ts)
                log["Kuafu/false_complete_count"] = tensor(fc)
                log["Kuafu/timeout_count"] = tensor(to)
                log["Kuafu/termination_total_count"] = tensor(total)
                log["Kuafu/cum_success_rate"] = tensor(true_rate)
                log["Kuafu/false_complete_rate"] = tensor(false_rate)
                log["Kuafu/timeout_rate"] = tensor(timeout_rate)
                log["Kuafu/success_rate_epoch"] = tensor(epoch_rate)
                log["Kuafu/success_rate_window"] = tensor(window_rate)
                log["Kuafu/success_rate_ema"] = tensor(ema_rate)
            # Reward reflection: log per-constraint satisfaction rate and reward (subsampled in time).
            stride = int(getattr(self, "_Kuafu_constraint_stats_stride_steps", 0) or 0)
            if stride > 0:
                c = int(getattr(self, "_Kuafu_constraint_stats_step_counter", 0) or 0) + 1
                setattr(self, "_Kuafu_constraint_stats_step_counter", c)
                if (c % stride) == 0 and hasattr(self, "get_constraint_stats"):
                    stage_filter = getattr(self, "_Kuafu_constraint_stats_stage_id", None)
                    stats = self.get_constraint_stats()
                    if isinstance(stats, dict):
                        # Log only the tuned stage when stage_id is known; otherwise log whatever stats returns.
                        items = stats.items()
                        for sid, cdict in items:
                            if stage_filter is not None and sid != stage_filter:
                                continue
                            if not isinstance(cdict, dict):
                                continue
                            for cid, s in cdict.items():
                                if not isinstance(s, dict):
                                    continue
                                rate = s.get("rate", None)
                                reward = s.get("reward", None)
                                if rate is not None:
                                    log[f"Kuafu/stage{sid}/component{cid}_rate"] = torch.tensor(
                                        float(rate), device=self.device
                                    )
                                if reward is not None:
                                    log[f"Kuafu/stage{sid}/component{cid}_reward"] = torch.tensor(
                                        float(reward), device=self.device
                                    )
            return obs, rew, terminated, truncated, extras

        setattr(env_u, "step", types.MethodType(_step_with_kuafu_logging, env_u))
