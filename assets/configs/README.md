# Task configuration

Create a TOML manifest with `schema_version = 1` and the sections below. Select
your registered environment, verified SUN Programs, and available GPUs.

| Section | Configure |
|---|---|
| `workspace` | `isaaclab_root`, `python`, external `artifact_root`, and `gpus` |
| `task` | Run `slug`, registered `task_name`, `robot_name`, and `robot_id` |
| `learn_programs` | Set `enabled = false` for programs supplied by your agent; configure `script`, `output_env`, and `args` for built-in generation |
| `collect` | Verified `program_paths`, positive `successful_demos_per_program`, and `workers_per_gpu_per_program` |
| `prior` | Dataset, task slug, and training settings |
| `eval` | Program selection, checkpoints, parameters, and weight acquisition |
| `residual` | Matching weights, launcher, and training settings |

Resolve relative source paths from the project root and generated output paths
under `KUAFU_ARTIFACT_ROOT`. Keep reusable environment settings in `environments/`
and task registrations in `src/kuafu/isaac/`. See `residual_rl/` for
hyperparameter defaults and [agent training setup](../../docs/AGENT_INTERFACE.md#collection-and-learning)
for supplying generation and tuning through your agent.

To reuse an existing configuration, choose an [optional workflow example](../../docs/EXAMPLES.md).
