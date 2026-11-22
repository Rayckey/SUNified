# Press emergency stop

press the e-stop button

[Task inputs and references](task.json)

| Robot | Environment |
|---|---|
| `fr3_finray` | `FR3-FinRay-Table-EStop-Close` |
| `franka_panda` | `Panda-Table-EStop-Close` |
| `kinova_gen3` | `Kinova-Table-Gen3-EStop-Close` |
| `ur5_robotiq` | `UR5-Table-EStop` |

Inspect the configuration with `kuafu tasks show estop_close`.

The button latches closed and the end effector withdraws from the cap.

Read the [scene and success implementation](../../../src/kuafu/isaac/learned/fr3_finray_estop_env_learned.py).

Recorded output: [SUN Program (4 stages)](../../../resources/learned_programs/manipulation-estop-close/2026-07-29_16-54-03/program).
The task JSON lists every bundled program revision and its recorded status.

Program formation: [estop_close_learn_programs.py](../../../src/kuafu/programs/estop_close_learn_programs.py) using `Formulate-Panda-EStop-Close-v0`.

Follow [installation](../../../docs/INSTALL.md) and the [task catalog](../README.md).
