# Learned-stage DP3 experiment

Run the proposed A/B/C comparison with Kuafu2Real's exporter, action-window
sampler, and DP3 backbone. This is an optional experiment; see
[using Kuafu](USAGE.md) for the agent workflow.

| Variant | Stage supervision | Action conditioning |
|---|---|---|
| A | — | Standard DP3 observation features |
| B | Cross-entropy | Standard DP3 observation features |
| C | Same cross-entropy | Observation features plus predicted soft stage embedding |

C detaches stage probabilities on the action-conditioning route. Labels supervise
the head; deployment uses point clouds and measured robot state. The added FiLM
columns start at zero, preserving A's initial action behavior.

## Set up

Use Python 3.11 and a working [Isaac installation](INSTALL.md#simulation-and-learning)
for simulator evaluation. Configure existing Kuafu2Real and
[DP3](https://github.com/YanjieZe/3D-Diffusion-Policy) checkouts:

```bash
export KUAFU2REAL_ROOT=/absolute/path/to/Kuafu2Real
export KUAFU_DP3_ROOT=/absolute/path/to/3D-Diffusion-Policy/3D-Diffusion-Policy
export KUAFU_ARTIFACT_ROOT=/absolute/path/to/KuafuArtifacts
python -m pip install -e '.[stage-dp3]'
python -m pip install --no-deps -e "$KUAFU2REAL_ROOT"
python -m kuafu.sim2real.dp3 --help
```

The adapter uses Kuafu2Real's compatibility modules and keeps upstream checkouts
unchanged. The implementation targets DP3's XYZ encoder, global FiLM conditioning,
and 10D `eff-simple` or `bc` action contracts.

## Prepare demonstrations

Choose one task after inspecting baseline failures. Record the suspected failure
mechanism and evidence paths in your copy of
[stage_dp3.yaml](../assets/configs/experiments/stage_dp3.yaml). Keep unsuccessful episodes
and their outcomes with that evidence.

Collect point-cloud demonstrations with `collect_with_checkpoint.py` or
`collect_with_pointcloud.py` using the updated recorder. Each HDF5 row now stores:

- `decision_stage`: the zero-based teacher stage before the action, with `-1` for
  unavailable or terminal supervision.
- `decision_valid`: whether the decision has a usable stage label.
- `observed_gripper`: measured gripper openness before the action.
- `point_cloud_recording`: camera, control-period, stride, crop, and SUN Program
  metadata for matching simulator evaluation.

For residual teachers, pass `--prior_mode residual_rl`, the matching
`--prior_checkpoint`, and the saved `--checkpoint_params_dir`. Supply the run's
`--weights_params_json` when it has program parameter overrides.

Save `stages.json` as an array of names in teacher stage-ID order. Use the exact
Gym task ID from collection:

```bash
python -m kuafu.sim2real.dp3 export /path/to/datasets \
  --task YOUR_GYM_TASK --stages /path/to/stages.json \
  --output "$KUAFU_ARTIFACT_ROOT/stage-dp3/data.zarr" \
  --action-format eff-simple --num-points 4096
```

The exporter verifies label timing and lengths, hashes source trajectories, and
saves disjoint trajectory partitions. Inspect `stage_transitions.json` alongside
recorded observations at several boundaries before training. For `eff-simple`,
Kuafu2Real pairs observation row `t` with measured pose target `t+1`; the stage
label stays at `t`. For `bc`, observation and command use the same row. Both use
measured gripper state as input. Export older demonstrations after recovering
verified decision-time labels and recording metadata, or recollect them.

## Use object-relative geometry

Convert an exported `eff-simple` dataset using a stationary object's recorded root:

```bash
python -m kuafu.sim2real.dp3 relative /path/to/data.zarr \
  --reference-object cabinet \
  --output "$KUAFU_ARTIFACT_ROOT/stage-dp3/object-relative.zarr"
```

Train with this new dataset path. The conversion preserves trajectories, splits,
stages, point order, RGB, and gripper values. It expresses measured EE poses, target
poses, and point-cloud XYZ in the cabinet frame: `inverse(T_object) @ T_ee`.
Targets remain absolute poses within that frame; all observations and targets in
a chunk share the stationary root. The converter checks the recorded root across
each full trajectory and rejects moving references. `conversion_validation.json`
records the source identity and full-dataset round-trip errors.

This adapts the object-relative geometry in
[Exo2EgoPolicy](https://www.amazon.science/publications/exo2egopolicy-geometry-aware-policy-transfer-from-exocentric-human-demonstrations)
to StageDP3's existing pose targets. Human-video pose translation and co-training
are separate capabilities. Isaac evaluation reads the cabinet root, applies the
same observation transform after cropping, and maps predictions back before IK.
For hardware, supply a measured or estimated object pose in the same coordinate
frame as the EE pose and point cloud, using `ObjectFrame` to encode and decode.

## Train the comparison

Copy the shared configuration and set the training budget and seeds. All variants
use the same data, windows, optimizer, diffusion sampler, and stage-head settings.
The runner uses float32, cosine learning-rate decay, and the baseline's EMA.

```bash
python -m kuafu.sim2real.dp3 train \
  --config assets/configs/experiments/stage_dp3.yaml \
  --dataset "$KUAFU_ARTIFACT_ROOT/stage-dp3/data.zarr" \
  --output "$KUAFU_ARTIFACT_ROOT/stage-dp3/training"
```

For one seed with the upstream SimpleDP3 backbone and predicted stage conditioning,
use `--config assets/configs/experiments/stage_simple_dp3.yaml --variants C`. Pass the
object-relative dataset path to train in the cabinet frame. Set
`CUDA_VISIBLE_DEVICES=2` to run on physical GPU 2.

Normalization uses training trajectories only. Padded and terminal action targets
are masked. Windows include every valid decision through the end of each
trajectory, preserving brief final stages. Each window's stage is its latest
observation's stage, including when its action chunk crosses a transition.
`experiment.json` records class counts,
settings, dataset identity, and dependency revisions. Each seed/variant saves
`best.ckpt`, `latest.ckpt`, and epoch metrics. Checkpoint selection minimizes the
same validation action loss for A/B/C; held-out test data is reserved for evaluation.

Use `--resume` to continue the identical experiment at an epoch boundary. A
`warm_start` checkpoint in the configuration loads shared baseline weights into
all variants and requires the same training-partition normalizer. Choose head
sizes and loss weight with training/validation data, then freeze the configuration
for the held-out comparison.

## Evaluate in simulation

Copy [the evaluation protocol](../assets/configs/experiments/stage_dp3_evaluation.json).
Set your task, explicit held-out seeds, and a controlled placement variation.
The supplied example shifts the drawer cabinet by 3 cm along environment +x.
`max_steps` counts recorded control intervals; the adapter repeats each command
for the collection stride. Use `chunk_exec: 1` for strided-action checkpoints.

```bash
python -m kuafu.sim2real.dp3 evaluate \
  --experiment "$KUAFU_ARTIFACT_ROOT/stage-dp3/training" \
  --dataset "$KUAFU_ARTIFACT_ROOT/stage-dp3/data.zarr" \
  --protocol /path/to/evaluation.json \
  --output "$KUAFU_ARTIFACT_ROOT/stage-dp3/evaluation" \
  --device cuda:0
```

The evaluator checks actual reset-state identities across models, counts every
attempt including time limits, and keeps nominal and varied conditions separate.
It restores recorded teacher tolerances and verifies the complete SUN Program
definition before stepping.
`results.json` includes raw successes/attempts, variation across training seeds,
transition outcomes, per-stage and macro accuracy, confusion matrices, synchronized
inference latency, and parameter counts. Offline held-out and rollout stage metrics
have separate fields. Transition failures count episodes ending after entering a
stage but before reaching a later stage or final task success.

For another simulator, supply `--adapter your_module:create_adapter`. The factory
receives `experiment`, `dataset`, `protocol`, and `device`. Implement `reset`,
`step`, and `close` using the contract in
[evaluate.py](../src/kuafu/sim2real/dp3/evaluate.py). Keep diagnostic `stage_id`
outside the observation dictionary.

## Validate changes

```bash
python -m pip install pytest
python -m pytest -q tests/stage_dp3
```

The tests use the real upstream DP3 network and scheduler, with small integration
fixtures. Full task-success conclusions require the matched training and simulator
campaign above. Record negative outcomes and remaining failure mechanisms with the
results.

Save test reports and experiment analysis under `KUAFU_ARTIFACT_ROOT`.
