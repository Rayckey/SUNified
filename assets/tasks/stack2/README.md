# Stack two cubes

stack the cubes

[Task inputs and references](task.json)

| Robot | Environment |
|---|---|
| `fr3_finray` | `FR3-FinRay-Table-Stack-2` |
| `franka_panda` | `Panda-Table-Stack-2` |
| `kinova_gen3` | `Kinova-Table-Gen3-Stack-2` |
| `ur5_robotiq` | `UR5-Table-Stack-2` |

Inspect the configuration with `kuafu tasks show stack2`.

The cubes align horizontally at one cube-height separation, with an open gripper clear of both cubes.

Read the [scene and success implementation](../../../src/kuafu/isaac/learned/fr3_finray_stack2_env_learned.py).

Recorded output: [SUN Program (8 stages)](../../../resources/learned_programs/manipulation-stack2/2026-07-29_17-04-22/program).
The task JSON lists every bundled program revision and its recorded status.

Program formation: [stack2_learn_programs.py](../../../src/kuafu/programs/stack2_learn_programs.py) using `Formulate-Panda-Stack-2-v0`.

Follow [installation](../../../docs/INSTALL.md) and the [task catalog](../README.md).
