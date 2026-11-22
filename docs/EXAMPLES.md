# Four optional examples

Complete [installation](INSTALL.md) and choose an example. Each directory includes
its inputs, recorded outputs, README and reproduction command.
Browse the [task catalog](../assets/tasks/README.md) for scene, asset and
SUN Program references.

| Example | Input | Output |
|---|---|---|
| [Video to rollout](../examples/01_video_to_rollout/README.md) | Videos 28/29 + cabinet description + task | Articulated cabinet, SUN Program, MPC verification and rollout image |
| [Program to rollout](../examples/02_program_to_rollout/README.md) | Prepared cabinet + six-stage SUN Program | Measured drawer travel and rollout image |
| [Demonstrations to policy](../examples/03_demonstrations_to_policy/README.md) | Four recorded Kinova stacking demonstrations | BC checkpoint, validation metrics and training plot |
| [Video to policy: end-to-end](../examples/04_video_to_policy/README.md) | Videos 28/29 + cabinet task + saved agent proposals | Verified asset/program, newly collected demonstrations, BC checkpoint and measured policy rollout |

For your own inputs and agent, start with [usage](USAGE.md). Individual
[module examples](MODULES.md) describe the interfaces used at each handoff.
Saved experimental campaigns remain in `assets/configs/campaigns/` and the
[DP3 experiment guide](STAGE_DP3_EXPERIMENT.md).
