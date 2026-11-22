# Release emergency stop

unlock the e-stop

[Task inputs and references](task.json)

| Robot | Environment |
|---|---|
| `fr3_finray` | `FR3-FinRay-Table-EStop-Open` |
| `franka_panda` | `Panda-Table-EStop-Open` |
| `kinova_gen3` | `Kinova-Table-Gen3-EStop-Open` |
| `ur5_robotiq` | `UR5-Table-EStop-Closed` |

Inspect the configuration with `kuafu tasks show estop_open`.

The initially latched button unlocks, springs open and the end effector withdraws.

Read the [scene and success implementation](../../../src/kuafu/isaac/learned/fr3_finray_estop_closed_env_learned.py).

Recorded output: [SUN Program (6 stages)](../../../resources/learned_programs/manipulation-estop-open/2026-07-29_17-32-14/program).
The task JSON lists every bundled program revision and its recorded status.

Program formation: [estop_open_learn_programs.py](../../../src/kuafu/programs/estop_open_learn_programs.py) using `Formulate-Panda-EStop-Open-v0`.

Follow [installation](../../../docs/INSTALL.md) and the [task catalog](../README.md).
