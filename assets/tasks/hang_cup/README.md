# Hang mug

hook the mug handle through the single cup hanger branch

[Task inputs and references](task.json)

| Robot | Environment |
|---|---|
| `fr3_finray` | `FR3-FinRay-Table-Hang-Cup` |
| `franka_panda` | `Panda-Table-Hang-Cup` |
| `kinova_gen3` | `Kinova-Table-Gen3-Hang-Cup` |
| `ur5_robotiq` | `UR5-Table-Hang-Cup` |

Inspect the configuration with `kuafu tasks show hang_cup`.

The mug handle surrounds the hanger branch while the open gripper stays clear for the configured hold duration.

Read the [scene and success implementation](../../../src/kuafu/isaac/learned/fr3_finray_hang_cup_env_learned.py).

Recorded output: [SUN Program (9 stages)](../../../resources/learned_programs/manipulation-hang-cup/2026-07-29_18-42-51/program).
The task JSON lists every bundled program revision and its recorded status.

Program formation: [hang_cups_learn_programs.py](../../../src/kuafu/programs/hang_cups_learn_programs.py) using `Formulate-Panda-Hang-Cup-v0`.

Follow [installation](../../../docs/INSTALL.md) and the [task catalog](../README.md).
