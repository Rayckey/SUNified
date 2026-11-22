# Open drawer

open the top drawer by 14 cm

[Task inputs and references](task.json)

| Robot | Environment |
|---|---|
| `fr3_finray` | `FR3-FinRay-Table-Drawer` |
| `franka_panda` | `Panda-Table-Drawer` |
| `kinova_gen3` | `Kinova-Table-Gen3-Drawer` |
| `ur5_robotiq` | `UR5-Table-Drawer` |

Inspect the configuration with `kuafu tasks show drawer`.

The top drawer opens, the gripper releases it and moves clear of the handle.

Read the [scene and success implementation](../../../src/kuafu/isaac/learned/fr3_finray_drawer_env_learned.py).

Recorded output: [SUN Program (6 stages)](../../../resources/learned_programs/manipulation-fr3-finray-drawer/2026-07-29_17-19-00/program).
The task JSON lists every bundled program revision and its recorded status.

Program formation: [fr3_finray_drawer_learn_programs.py](../../../src/kuafu/programs/fr3_finray_drawer_learn_programs.py) using `Formulate-Panda-Drawer-v0`.

Follow [installation](../../../docs/INSTALL.md) and the [task catalog](../README.md).
