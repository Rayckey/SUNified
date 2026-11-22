# Residual Hparam Defaults

Date: 2026-06-22

## Default Baseline

Residual hyperparameter tuning starts from the selected agent YAML. For Kinova and Panda RL-Games residual agents, the default initial hyperparameters are:

```yaml
params:
  env:
    clip_actions: 2.0

  config:
    residual_rl:
      alpha_final: 1.5
      alpha_warmup_epochs: 600
      critic_warmup_epochs: 150
      residual_bounds: [0.30, 0.30, 0.30, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50]
```

These are defaults, not final tuned values. A residual hparam run may change the allowlisted fields after evaluating rollouts, but the first proposal should be grounded in this baseline unless the experiment intentionally documents another starting point.

## Scope

The defaults apply to RL-Games residual agent configs under:

- `src/kuafu/isaac/learned/kinova/agents/*.yaml`
- `src/kuafu/isaac/learned/panda/agents/*.yaml`

They do not apply to SAC-RLPD or SKRL configs that do not use the `params.env` plus `params.config.residual_rl` schema.

## Tunable Fields

The residual hparam prompt allowlists these fields for LLM edits:

- `params.env.clip_actions`
- `params.config.residual_rl.residual_bounds`
- `params.config.residual_rl.clip_composed_actions`
- `params.config.residual_rl.alpha_final`
- `params.config.residual_rl.alpha_warmup_epochs`

`critic_warmup_epochs` is part of the default residual RL agent config. It is documented here as a required baseline value, but it is not currently part of the residual hparam prompt allowlist.

## Runtime Source of Truth

At runtime, the residual-RL pipeline reads the selected task agent YAML and extracts these initial values from task metadata before asking the LLM for residual hparam proposals. The relevant code paths are:

- `src/kuafu/rl/core/runner_residual_hparams.py` (`_initial_residual_agent_hyperparameters`)
- `src/kuafu/rl/core/worker_pool.py` (`_extract_agent_hyperparameters`)
- `src/kuafu/rl/core/worker_pool.py` (`_apply_agent_hyperparameters`)
- `src/kuafu/rl/llm/prompts_residual_hparams.py` (prompt allowlist and validation bounds)
