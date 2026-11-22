# Stand up can

Pick up the fallen over can and place it back down in upright position

[Task inputs and references](task.json)

| Robot | Environment |
|---|---|
| `fr3_finray` | `FR3-FinRay-Table-Can` |
| `franka_panda` | `Panda-Table-Can` |
| `kinova_gen3` | `Kinova-Table-Gen3-Can` |
| `ur5_robotiq` | `UR5-Table-Can` |

Inspect the configuration with `kuafu tasks show can`.

The can is upright and nearly stationary, with an open gripper clear of it.

Read the [scene and success implementation](../../../src/kuafu/isaac/learned/fr3_finray_can_env_learned.py).

Recorded output: [SUN Program (8 stages)](../../../resources/learned_programs/manipulation-can/2026-07-29_18-05-40/program).
The task JSON lists every bundled program revision and its recorded status.

Program formation: [can_learn_programs.py](../../../src/kuafu/programs/can_learn_programs.py) using `Formulate-Panda-Can-v0`.

Follow [installation](../../../docs/INSTALL.md) and the [task catalog](../README.md).
