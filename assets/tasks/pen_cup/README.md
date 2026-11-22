# Insert marker into holder

pick up the fallen over marker, stick it upright, and insert it into the pen holder

[Task inputs and references](task.json)

| Robot | Environment |
|---|---|
| `fr3_finray` | `FR3-FinRay-Table-Pen-Cup` |
| `franka_panda` | `Panda-Table-Pen-Cup` |
| `kinova_gen3` | `Kinova-Table-Gen3-Pen-Cup` |
| `ur5_robotiq` | `UR5-Table-Pen-Cup` |

Inspect the configuration with `kuafu tasks show pen_cup`.

The marker is upright and near the holder opening; the configured release and program-stage conditions apply.

Read the [scene and success implementation](../../../src/kuafu/isaac/learned/fr3_finray_pen_cup_env_learned.py).

Recorded output: [SUN Program (10 stages)](../../../resources/learned_programs/manipulation-pen-cup/2026-07-29_18-22-20/program).
The task JSON lists every bundled program revision and its recorded status.

Program formation: [pen_cup_learn_programs.py](../../../src/kuafu/programs/pen_cup_learn_programs.py) using `Formulate-Panda-Pen-Cup-v0`.

Follow [installation](../../../docs/INSTALL.md) and the [task catalog](../README.md).
