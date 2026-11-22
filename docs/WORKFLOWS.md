# Collect demonstrations and train policies

Give your agent a verified SUN Program and a compatible registered environment.
Configure [training through your agent](AGENT_INTERFACE.md#collection-and-learning),
then use `kuafu_start_stage` to run the stages you need:

```text
collect → train-prior → eval-prior → train-residual
```

The commands below are the CLI equivalents. Run them from the Kuafu checkout in
the [Isaac environment](INSTALL.md#simulation-and-learning).

## Configure a run

Create a [task manifest](../assets/configs/README.md) with your verified program paths,
environment, GPUs, and training settings. Select it explicitly:

```bash
export KUAFU_PYTHON="$(command -v python)"
export ISAACLAB_ROOT=/path/to/IsaacLab
export KUAFU_ARTIFACT_ROOT=/path/to/IsaacLab/KuafuArtifacts
export KUAFU_TASK_CONFIG=/absolute/path/to/my-task.toml

./bin/kuafu -c "$KUAFU_TASK_CONFIG" doctor
./bin/kuafu -c "$KUAFU_TASK_CONFIG" plan --prepare-only
```

Set `KUAFU_PROGRAM_PATH` to pin a program for direct evaluation or residual
training. For a prepared configuration, choose the
[optional drawer workflow](EXAMPLES.md).

Store datasets, checkpoints, generated launchers, logs, videos, and pipeline state
under `KUAFU_ARTIFACT_ROOT`. Use `resources/learned_programs/` for bundled SUN
Programs and `resources/experiment_definitions/` for saved experiment inputs.

## Run the stages

```bash
./bin/kuafu -c "$KUAFU_TASK_CONFIG" stage collect
./bin/kuafu -c "$KUAFU_TASK_CONFIG" stage train-prior
./bin/kuafu -c "$KUAFU_TASK_CONFIG" stage eval-prior
./bin/kuafu -c "$KUAFU_TASK_CONFIG" stage train-residual --prepare-only
```

Inspect the prepared residual launcher, then run the last command with
`--prepare-only` omitted. Set `KUAFU_PROJECT_ROOT` when relocating the checkout.
To run a sequence:

```bash
./bin/kuafu -c "$KUAFU_TASK_CONFIG" \
  run --from-stage collect --through-stage eval-prior
```

Use `--dry-run` to preview a stage's commands and `status` to inspect progress.
State is stored at `$KUAFU_ARTIFACT_ROOT/<task>/pipeline_state.json`. Use a new
run identity for changed inputs, or `--force` to rerun a stage. Completed
per-program datasets can be reused after validation.

| Stage | Input | Output |
|---|---|---|
| `learn-programs` | Task scene, prompts, robot model, provider credentials | Successful run with `program/` and success metadata |
| `collect` | Successful SUN Programs | Exact-count compact HDF5, hash, and provenance |
| `train-prior` | Compact HDF5 | BC configurations, sweep CSV, checkpoints, and published priors |
| `eval-prior` | Sweep CSV and checkpoints | Success rates, selected parameters, and residual weights |
| `train-residual` | Evaluated prior and matching parameters/weights | Launcher, training receipt, logs, and checkpoint |

## Optional built-in program generation

To use Kuafu's built-in generator, set `[learn_programs].enabled = true`, select
its runner, and clear `[collect].program_paths` and `[eval].program_path`:

```bash
./bin/kuafu -c "$KUAFU_TASK_CONFIG" stage learn-programs
```

Run once per program, using `--force` on subsequent runs to append successful
results. Formulation uses Panda scenes for geometry, reset, observations, and actions.
`SUNProgramExecutor` owns candidate predicates and residuals; the scene supplies
zero reward and suppressed done flags. Recovery restores the seeded scene,
completed history, and remaining task plan. Each successful stage records
`task.txt`, `atomic.txt`, and `constraints.json`. Select a runner matching your task:

| Runner in `src/kuafu/programs/` | Gym ID |
|---|---|
| `fr3_finray_drawer_learn_programs.py` | `Formulate-Panda-Drawer-v0` |
| `fr3_finray_drawer_stack_learn_programs.py` | `Formulate-Panda-Drawer-Stack-v0` |
| `can_learn_programs.py` | `Formulate-Panda-Can-v0` |
| `estop_close_learn_programs.py` | `Formulate-Panda-EStop-Close-v0` |
| `estop_open_learn_programs.py` | `Formulate-Panda-EStop-Open-v0` |
| `hang_cups_learn_programs.py` | `Formulate-Panda-Hang-Cup-v0` |
| `pen_cup_learn_programs.py` | `Formulate-Panda-Pen-Cup-v0` |
| `stack2_learn_programs.py` | `Formulate-Panda-Stack-2-v0` |
| `stack3_learn_programs.py` | `Formulate-Panda-Stack-3-v0` |

## Preserve program and prior identity

For a pooled dataset, set `[eval].program_path` to one of its collected programs;
that selection also governs parameter and weight acquisition. A single collected
program is inherited automatically. Keep program paths and content hashes with
datasets, evaluation results, and parameter/weight records. Regenerate records
that lack this provenance, and use `[eval].params_path` for a matching existing
parameter JSON. Reacquire weights after replacing the selected prior checkpoint.

Use the supplied prior loaders to preserve normalization: both raw-input and
normalized-input BC priors use a `1e-8` standard-deviation floor. Raw-input
exports retain `obs_mean`/`obs_std` buffers; a frozen RL-Games BC normalizer
preserves that transform. Unfreezing enables adaptive RL-Games normalization.

The receiver assigns constraint tolerances after parsing each model proposal.
The agent supplies geometry, offsets, travel, rotations, and contact conditions.
Replay uses saved program values. Evaluation and residual training use a broader
profile with position thresholds of 1–5 cm and angular thresholds of 0.30–0.50 rad;
learning model calls tune reward weights.

## Collect larger datasets

Configure collection workers and demonstration counts in your task manifest.
To reuse our saved campaigns, see the
[optional workflow examples](EXAMPLES.md).

Checkpoint collection supports these options in `collect_with_checkpoint.py`:

| Option | Purpose |
|---|---|
| `--checkpoint_params_dir PATH` | Select archived `agent.yaml` and `env.yaml`. |
| `--camera_keys KEY ...` | Select ordered sensors; Panda uses `table_cam_0 table_cam_1 table_cam_2`. |
| `--disable_left_right_randomization` | Fix object sides for Panda Drawer-Stack, Hang-Cup, and Stack-2. |
| `--force_left_right_swap` | Select the swapped layout for those tasks. |

Choose one layout override, or retain randomized placement. Workers cap accepted
demonstrations at their requested counts. Supply archived checkpoints, priors,
and parameters under `KUAFU_ARTIFACT_ROOT` or override their paths.

Collection/upload recipes include `collect_and_upload_loop_kinova_drawer_fff.sh`,
`collect_and_upload_loop_kinova_stack3_fff3.sh`, and the Panda drawer-stack,
hang-cup, and stack-2 loops. Configure their upload destination before running.
Kinova DP3 conversion uses `KUAFU2REAL_ROOT` with
`scripts/export_kuafu_to_dp3.py`, `scripts/create_xyz_only_zarr.py`, and point-cloud
clip configs. Set `LOCAL_ORGANIZE_ZARR_BEFORE_UPLOAD=0` to upload raw shards.

## Reuse evaluation for weight acquisition

Pass `--bc_eval_stats_path PATH` to
`src/kuafu/rl/isaac/initial_weight_sweep.py` to reuse a completed
`play_bc_env.py` evaluation. The sweep checks task, checkpoint hash, and program
identity, then restores stage/component keys from the recorded results.

Set `--reasoning_effort` or `KUAFU_LLM_REASONING_EFFORT` for model reasoning.
Token usage is recorded in `openai_payloads.jsonl`.

## Validate changes

Follow [CONTRIBUTING.md](../CONTRIBUTING.md) for tests and the
[architecture](ARCHITECTURE.md) for stage contracts. The
[cabinet replay](../examples/02_program_to_rollout/README.md)
exercise bundled programs and scenes.
