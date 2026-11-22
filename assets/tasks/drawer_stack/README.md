# Place object in drawer

open the top drawer fully and put the strawberry in top_drawer_center_frame

[Task inputs and references](task.json)

| Robot | Environment |
|---|---|
| `fr3_finray` | `FR3-FinRay-Table-Drawer-Stack` |
| `franka_panda` | `Panda-Table-Drawer-Stack` |
| `kinova_gen3` | `Kinova-Table-Gen3-Drawer-Stack` |
| `ur5_robotiq` | `UR5-Table-Drawer-Stack` |

Inspect the configuration with `kuafu tasks show drawer_stack`.

The strawberry reaches the drawer interior and the gripper opens; the configured program stage gate also applies.

Read the [scene and success implementation](../../../src/kuafu/isaac/learned/fr3_finray_drawer_stack_env_learned.py).

Recorded output: [SUN Program (14 stages)](../../../resources/learned_programs/manipulation-fr3-finray-drawer-stack/2026-07-29_18-34-48/program).
The task JSON lists every bundled program revision and its recorded status.

Program formation: [fr3_finray_drawer_stack_learn_programs.py](../../../src/kuafu/programs/fr3_finray_drawer_stack_learn_programs.py) using `Formulate-Panda-Drawer-Stack-v0`.

Follow [installation](../../../docs/INSTALL.md) and the [task catalog](../README.md).
