# Stack three cubes

stack the cubes

[Task inputs and references](task.json)

| Robot | Environment |
|---|---|
| `fr3_finray` | `FR3-FinRay-Table-Stack-3` |
| `franka_panda` | `Panda-Table-Stack-3` |
| `kinova_gen3` | `Kinova-Table-Gen3-Stack-3` |
| `ur5_robotiq` | `UR5-Table-Stack-3` |

Inspect the configuration with `kuafu tasks show stack3`.

Cube 2 is on cube 1 and cube 3 is on cube 2; the gripper releases and clears the stack.

Read the [scene and success implementation](../../../src/kuafu/isaac/learned/fr3_finray_stack3_env_learned.py).

Recorded output: [SUN Program (16 stages)](../../../resources/learned_programs/manipulation-stack3/2026-07-29_18-22-26/program).
The task JSON lists every bundled program revision and its recorded status.

Program formation: [stack3_learn_programs.py](../../../src/kuafu/programs/stack3_learn_programs.py) using `Formulate-Panda-Stack-3-v0`.

Follow [installation](../../../docs/INSTALL.md) and the [task catalog](../README.md).

The base Panda task uses the [face-aligned program](../../../resources/learned_programs/manipulation-stack3/2026-11-05_panda/program).
Its gripper aligns with cube faces before descending. Existing policy variants
keep their trained program.
