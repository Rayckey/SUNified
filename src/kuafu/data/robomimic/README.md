# Robomimic training and evaluation

Train native Robomimic BC policies from MPC trajectories and evaluate them in
batched Isaac Lab scenes. For priors used by Kuafu residual learning, follow the
[native prior workflow](../../../../docs/WORKFLOWS.md).

Activate the [Isaac environment](../../../../docs/INSTALL.md#simulation-and-learning)
and run from the Kuafu checkout. Set `KUAFU_ARTIFACT_ROOT` to your artifact store.

## Install

```bash
python -m pip install "robomimic==0.4.0" "numpy==1.26.0" "h5py==3.16.0"
python -c "import robomimic; print(robomimic.__version__)"
```

## Prepare and train

Edit [manifest.json](manifest.json) to select trajectories, tasks, sweep parameters,
and program snapshots. Artifact paths resolve under `KUAFU_ARTIFACT_ROOT`.

```bash
python src/kuafu/data/robomimic/prepare_datasets.py
python src/kuafu/data/robomimic/launch_training.py --run-id my-priors
```

Preparation converts `obs/state` and `actions/absolute_ee` into Robomimic HDF5
under `Trajectories/robomimic/`, alongside `resolved_manifest.json`. Use
`--overwrite` to rebuild prepared data. Training uses the manifest's BC MLP
configuration; set `--epochs N` to change duration or `--dry-run` to preview jobs.

Read configurations, logs, and checkpoints under
`$KUAFU_ARTIFACT_ROOT/standalone/robomimic/training/my-priors/`.

## Evaluate

For one checkpoint:

```bash
python src/kuafu/data/robomimic/eval_policy_env.py \
  --task FR3-FinRay-Table-Stack-2 \
  --checkpoint /absolute/path/to/model_epoch_2000.pth \
  --params-path /absolute/path/to/matching-parameters.json \
  --num-envs 8192 --duration-sec 180 \
  --output /tmp/stack2_robomimic_eval.json --device cuda:0 --headless
```

Use `--program-path` for a snapshot root containing `program/subtask_*`.
For all manifest tasks:

```bash
python src/kuafu/data/robomimic/launch_eval.py \
  --train-run standalone/robomimic/training/my-priors \
  --run-id my-evaluation --gpus 0,1,2,3 \
  --num-envs 8192 --duration-sec 180 --seed 42
```

Read `launch_state.json`, `process_logs/<task>.log`, and `results/<task>.json`
under `$KUAFU_ARTIFACT_ROOT/standalone/robomimic/evaluation/my-evaluation/`.

Match each checkpoint's observation/action dimensions, program stages, and
constraint indices to its sweep parameters. Evaluation uses one forward pass
per environment batch, checks the live stage in each policy observation, and
reports success as `successful / episodes` over the simulated-time window.
SUN Program predicates and stage transitions remain active during playback.

For the May Pen Cup trajectory and sweep, use the manifest's eight-stage
`program_snapshots/pen_cup_may19_b343f54` snapshot from revision `b343f54`.

## Reference results

Run `2026-07-12_robomimic_eval_epoch2000_8192env_180sec_program_sweep_v2`
used 8192 environments per task for 180 simulated seconds (5400 steps).

| Task | Successful / episodes | Success rate |
|---|---:|---:|
| Stack 2 | 12,122 / 99,709 | 12.16% |
| Drawer | 4,223 / 124,951 | 3.38% |
| Pen Cup | 0 / 57,361 | 0.00% |
| Kinova E-Stop Open | 84,414 / 161,705 | 52.20% |
| Kinova E-Stop Close | 15,015 / 134,397 | 11.17% |
| Kinova Can | 3,068 / 57,947 | 5.29% |
| Drawer Stack | 0 / 49,966 | 0.00% |
| Stack 3 | 0 / 33,002 | 0.00% |
| Kinova Hang Cup | 565 / 73,735 | 0.77% |
