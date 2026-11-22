# Panda `chunking_10_fff3` Kuafu handoff

BC training, GPT-5.4 parameter acquisition, parameter-conditioned BC evaluation, and the evaluation-informed GPT-5.4 weight sweep are complete for Pen Cup, Drawer Stack, Stack-2, and Hang Cup. Continue with Kuafu residual-hyperparameter training; do not repeat those phases.

## Published artifacts

| Task | BC eval | Parameter sweep | Final weight sweep |
| --- | ---: | --- | --- |
| Pen Cup | 27.1% | `resources/experiment_definitions/franka_panda/residual_rl/parameter_search/panda_learned_pen_cup/params_best.json` | `resources/experiment_definitions/franka_panda/residual_rl/parameter_search/panda_learned_pen_cup/weights_best.json` |
| Drawer Stack | 56.9% | `resources/experiment_definitions/franka_panda/residual_rl/parameter_search/panda_learned_drawer_stack/params_best.json` | `resources/experiment_definitions/franka_panda/residual_rl/parameter_search/panda_learned_drawer_stack/weights_best.json` |
| Stack-2 | 13.9% | `resources/experiment_definitions/franka_panda/residual_rl/parameter_search/panda_learned_stack_2/params_best.json` | `resources/experiment_definitions/franka_panda/residual_rl/parameter_search/panda_learned_stack_2/weights_best.json` |
| Hang Cup | 4.5% | `resources/experiment_definitions/franka_panda/residual_rl/parameter_search/panda_learned_hang_cup/params_best.json` | `resources/experiment_definitions/franka_panda/residual_rl/parameter_search/panda_learned_hang_cup/weights_best.json` |

The final weight JSON is the `--initial_weights_params` input for kuafu.rl. It contains both the original LLM parameters and the evaluation-informed weights.

## Successful Kinova agent setup transferred to Panda

Each launcher passes `kinova_agent_hyperparameters.json` through `--initial_agent_hyperparameters`. The Panda fff3 YAML defaults were also aligned to the same winning values, while retaining Panda task names and `[4096, 2048, 4096]` actor units for the Panda fff3 priors.

| Panda task | Successful Kinova source | Success | Default training allocation |
| --- | --- | ---: | --- |
| Pen Cup fff3 | `Residuals/Kinova-Table-Gen3-Pen-Cup-fff3/2026-06-23_19-48-07` | 74.74% | 1 worker, GPU 0, 2000 initial epochs |
| Drawer Stack fff3 | `Residuals/Kinova-Table-Gen3-Drawer-Stack-fff3/2026-07-02_01-41-01` | 90.58% | 1 worker, GPU 1, 3000 initial epochs |
| Stack-2 fff3 | `Residuals/Kinova-Table-Gen3-Stack-2-ff3/2026-06-22_09-01-21` | 77.86% | 1 worker, GPU 2, 2000 initial epochs |
| Hang Cup fff3 | `Residuals/Kinova-Table-Gen3-Hang-Cup-fff3/2026-07-09_03-32-32` | 36.02% | 1 worker, GPU 3, 3000 initial epochs |

Stack-2 intentionally borrows the proven Kinova ff3 residual-agent values because that is the successful Kinova run. Only those residual-agent values are transferred; its target policy remains Panda fff3 so the actor architecture matches the Panda fff3 prior.

## Repository and data storage

The Kuafu Git repository contains this README, all four launchers, the LLM parameter and weight JSONs, the Kinova-agent transfer JSONs, and the four Panda agent YAMLs. A clone of Kuafu therefore supplies every small configuration needed to launch training.

Wasabi stores the large BC priors and trajectory datasets. The `Residuals/Panda-Table-*/` folders are optional provenance and contain the full parameter/weight sweep audit outputs; the launchers do not depend on them.

Published BC priors:

```text
Priors/franka_panda/pen_cup/pen_cup_pen_cup_1000_2026-07-25_chunking_10_fff3/best.pth
Priors/franka_panda/drawer_stack/drawer_stack_drawer_stack_1000_2026-07-25_chunking_10_fff3/best.pth
Priors/franka_panda/stack2/stack2_stack2_1000_2026-07-25_chunking_10_fff3/best.pth
Priors/franka_panda/hang_cup/hang_cup_hang_cup_1000_2026-07-25_chunking_10_fff3/best.pth
```

Collected datasets:

```text
Trajectories/panda_mpc_1000_final_2026-07-25/
```

The trajectories are not read by the supplied Kuafu launchers, but keep them if BC retraining or dataset regularization may be needed.

## Download onto the Kuafu server

Run from the target IsaacLab repository root. These commands download only the four required prior directories and the four-task trajectory directory:

```bash
export WASABI_ENDPOINT="https://s3.wasabisys.com"
export WASABI_PROFILE="kuafu"

aws s3 sync s3://kuafu/priors/franka_panda/pen_cup/pen_cup_pen_cup_1000_2026-07-25_chunking_10_fff3/ Priors/franka_panda/pen_cup/pen_cup_pen_cup_1000_2026-07-25_chunking_10_fff3/ --endpoint-url="$WASABI_ENDPOINT" --profile "$WASABI_PROFILE"
aws s3 sync s3://kuafu/priors/franka_panda/drawer_stack/drawer_stack_drawer_stack_1000_2026-07-25_chunking_10_fff3/ Priors/franka_panda/drawer_stack/drawer_stack_drawer_stack_1000_2026-07-25_chunking_10_fff3/ --endpoint-url="$WASABI_ENDPOINT" --profile "$WASABI_PROFILE"
aws s3 sync s3://kuafu/priors/franka_panda/stack2/stack2_stack2_1000_2026-07-25_chunking_10_fff3/ Priors/franka_panda/stack2/stack2_stack2_1000_2026-07-25_chunking_10_fff3/ --endpoint-url="$WASABI_ENDPOINT" --profile "$WASABI_PROFILE"
aws s3 sync s3://kuafu/priors/franka_panda/hang_cup/hang_cup_hang_cup_1000_2026-07-25_chunking_10_fff3/ Priors/franka_panda/hang_cup/hang_cup_hang_cup_1000_2026-07-25_chunking_10_fff3/ --endpoint-url="$WASABI_ENDPOINT" --profile "$WASABI_PROFILE"
aws s3 sync s3://kuafu/trajectories/panda_mpc_1000_final_2026-07-25/ Trajectories/panda_mpc_1000_final_2026-07-25/ --endpoint-url="$WASABI_ENDPOINT" --profile "$WASABI_PROFILE"
```

The target checkout must include the current `Kuafu/` repository and this Kuafu commit. The committed fff3 agent YAMLs contain the transferred Kinova defaults and `[4096, 2048, 4096]` actor units.


## Continue training

Source `init.sh` for the OpenAI key, then reactivate the target server's Isaac environment if needed. Run from the repository root. Launch all four commands concurrently in separate terminals or as background jobs:

```bash
source init.sh
conda activate <ISAAC_ENV>
source _isaac_sim/setup_conda_env.sh
export KUAFU_REPO_ROOT="$PWD"

bash resources/experiment_definitions/franka_panda/residual_rl/prior_conditioned_training/panda_learned_pen_cup_state/pen_cup_pen_cup_1000_2026-07-25_chunking_10_fff3_residual_hparams.sh &
bash resources/experiment_definitions/franka_panda/residual_rl/prior_conditioned_training/panda_learned_drawer_stack/drawer_stack_drawer_stack_1000_2026-07-25_chunking_10_fff3_residual_hparams.sh &
bash resources/experiment_definitions/franka_panda/residual_rl/prior_conditioned_training/panda_learned_stack2_state/stack2_stack2_1000_2026-07-25_chunking_10_fff3_residual_hparams.sh &
bash resources/experiment_definitions/franka_panda/residual_rl/prior_conditioned_training/panda_learned_hang_cup/hang_cup_hang_cup_1000_2026-07-25_chunking_10_fff3_residual_hparams.sh &
wait
```

The four launchers can run concurrently. Each defaults to one worker pinned to a distinct physical GPU (Pen Cup 0, Drawer Stack 1, Stack-2 2, Hang Cup 3); within each masked process that GPU is addressed as `cuda:0`. The task-specific initial-epoch defaults are listed above. The allocation can be overridden with `KUAFU_CUDA_VISIBLE_DEVICES`, `KUAFU_DEVICE`, `KUAFU_NUM_WORKERS`, and, for Hang Cup, `KUAFU_INIT_TRAIN_EPOCH`. Active runs write under `logs/residual_rl/`; an accepted completed run is moved to `Residuals/<task>/<new timestamp>/` and can then be uploaded with `bash Kuafu/upload_data.sh`.
