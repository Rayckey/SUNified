# Train a dynamics ensemble

Train a PETS-style probabilistic ensemble from Kuafu HDF5 demonstrations without
starting Isaac Sim. Install the optional dependencies in a Python 3.11 environment:

```bash
python -m pip install -e '.[dynamics]'
export KUAFU_ARTIFACT_ROOT=/absolute/path/to/KuafuArtifacts
kuafu train-dynamics \
  --dataset /absolute/path/to/demonstrations.hdf5 \
  --output world_models/my_task/pets_seed42 \
  --task My-Registered-Task \
  --device cuda:0
```

Use a new output directory for each run. `python -m kuafu.dynamics` exposes the
same options. CPU training is supported with `--device cpu`.

## Dataset contract

Each `data/demo_*` group must contain finite, aligned arrays:

| Dataset | Shape | Default selection |
|---|---|---|
| `obs/<obs-key>` | `(T, state_dim)` | `full_state` |
| `actions/<action-key>` | `(T, action_dim)` | `relative_joint` |
| `next_obs/<obs-key>` | `(T, state_dim)` | `full_state` |
| `dones` (optional) | `(T,)` | Exclude terminal rows |

Choose recorded commands with the same timing and scaling you will use at
inference. Some converted datasets derive EE actions from future observations;
those are unsuitable for learning command-conditioned dynamics. Pass
`--action-key ''` for a flat `actions` dataset.

The loader checks `next_obs[t] == obs[t+1]` within numerical tolerance, except at
terminal boundaries. Fix stale or misaligned recordings upstream; the trainer
does not invent next states or shift actions. It excludes terminal rows because
an Isaac autoreset observation is not a physical successor.

Defaults remove the last observation column (`--exclude-last 1`, the SUN Program
stage label) and first transition (`--skip-first 1`, reset artifacts). Set these
to zero for datasets without those conventions. Whole demonstrations are split
80%/10%/10% into training, validation, and test sets. Normalization and each
member's bootstrap sample use only the training set.

## Kinova pen-in-cup

For `Kinova-Table-Gen3-Pen-Cup`, use `full_state` and recorded `relative_joint`
commands. The 59-column observation becomes a 58-column dynamics state when the
stage label is removed. Preserve the recorded coordinate frames, meter units,
rotation-6D values, and joint/action scaling; do not substitute raw simulator
world coordinates. Any source quaternion conversion must retain Kuafu's **wxyz**
convention.

| State indices (Python slices) | Quantity |
|---|---|
| `0:3`, `3:6` | Pen and cup positions |
| `6:9`, `9:12` | Fingertip and link-4 positions |
| `12:18`, `18:24`, `24:30` | Pen, fingertip, and link-4 rotations in 6D |
| `30:39`, `39:48` | Nine joint positions and velocities |
| `48:57` | Normalized joint effort/contact channels |
| `57:58` | Gripper openness |

The nine action coordinates are seven arm commands and two gripper commands.
Check the source dataset's `env_args` for the control timestep. These observation
features omit some simulator state, including object velocities; treat the model
as an observation-dynamics baseline and validate its rollout horizon accordingly.

## Artifacts and prediction

The default ensemble has five independent networks, each with four 200-unit SiLU
layers and Gaussian mean/variance outputs for normalized state changes. It uses
fixed soft log-variance bounds, independent bootstrap datasets, and per-member
validation-NLL checkpoint selection. See the [PETS paper](https://arxiv.org/abs/1805.12114)
for the algorithm and [original implementation](https://github.com/kchua/handful-of-trials).

| Artifact | Contents |
|---|---|
| `best.pt` | Best member weights, normalizers, configuration, and provenance |
| `run.json` | Dataset SHA-256, selected columns, demonstration splits, hyperparameters |
| `metrics.jsonl` | Training and validation metrics after each epoch |
| `evaluation.json` | Held-out one-step and TS-infinity rollout metrics, persistence baselines |

```python
from kuafu.dynamics.model import load_model

model, checkpoint = load_model('/absolute/path/to/run/best.pt', device='cuda:0')
# states: [batch, 58]; commands: [batch, 9], float32 on the model's device
member_means, member_variances = model.predict(states, commands)
mean_next_state = member_means.mean(0)
aleatoric_variance = member_variances.mean(0)
epistemic_variance = member_means.var(0, unbiased=False)

# action_sequences: [batch, horizon, 9]
trajectories = model.rollout(states, action_sequences, particles=20, propagation='tsinf')
# [batch, particles, horizon + 1, 58], including the initial state
```

`tsinf` holds one network per particle; `ts1` resamples at each step. Rollouts use
recorded future actions for evaluation but never feed back true future states.
Reported normalized RMSE scales each coordinate by its training delta standard
deviation; `rmse_per_dim` retains source units. Confidence coverage uses
moment-matched marginal intervals, not calibrated safety bounds.

This command trains the dynamics component. Closed-loop PETS control still needs
a task reward/cost, a CEM planner, and simulator validation. Successful
demonstrations cover a narrow action distribution, so held-out prediction error
does not establish reliability for exploratory actions or physical deployment.
