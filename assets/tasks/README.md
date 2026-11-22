# Task catalog

The catalog includes nine registered task families across FR3/FinRay, Panda,
Kinova Gen3 and UR5/Robotiq, plus the video-derived drawer cabinet.
Each task folder contains instructions, object references, environment IDs,
success-check sources and available SUN Program outputs.

```bash
kuafu tasks list
kuafu tasks show stack2 --robot kinova_gen3
```

MCP and Python agents use `kuafu_list_tasks` and `kuafu_get_task`.

## Registered tasks

| Task | Objective |
|---|---|
| [Open drawer](drawer/README.md) | open the top drawer by 14 cm |
| [Place object in drawer](drawer_stack/README.md) | open the top drawer fully and put the strawberry in top_drawer_center_frame |
| [Stack two cubes](stack2/README.md) | stack the cubes |
| [Stack three cubes](stack3/README.md) | stack the cubes |
| [Insert marker into holder](pen_cup/README.md) | pick up the fallen over marker, stick it upright, and insert it into the pen holder |
| [Hang mug](hang_cup/README.md) | hook the mug handle through the single cup hanger branch |
| [Stand up can](can/README.md) | Pick up the fallen over can and place it back down in upright position |
| [Press emergency stop](estop_close/README.md) | press the e-stop button |
| [Release emergency stop](estop_open/README.md) | unlock the e-stop |

The task JSON includes the base environment, every registered policy variant,
and each robot's asset paths, interaction frames and default program. FR3/FinRay,
Kinova and UR5 models are bundled; Isaac downloads the Panda model at scene load.
Set the selected Gym ID and matching program in your [pipeline manifest](../configs/README.md).
Use `kuafu tasks show <task> --robot <robot>` to select that robot's configured
program. Replay a native task with its defaults in the Isaac environment:

```bash
python -m kuafu.data.collection.collect_from_mpc \
  --task_name Panda-Table-Pen-Cup --num_envs 1 --num_demos 1 \
  --seed 42 --action_noise_std 0 --max_steps 1600 \
  --output_dir task-checks/pen-cup --headless --device cuda:0
```

Read `collection_result.json` for the physical success count and the HDF5 file
under `datasets/` for aligned demonstrations. Both are saved under
`KUAFU_ARTIFACT_ROOT`.
Robot setup is covered by [installation](../../docs/INSTALL.md) and the [UR5 guide](../../docs/UR5.md).
All four robot families select bundled default SUN Programs for MPC collection.
For policy collection, supply a checkpoint trained for the selected robot.

Saved programs preserve their original recorded results. Check the selected
robot configuration and verify its rollout when reusing a program.

## Video-derived drawer

The [Video 28/29 cabinet](generated_cabinet/README.md) provides a verified
drawer-opening example.

The exported-video inspection/repair tool currently operates on the generated
cabinet. Other tasks use their registered scenes, program-formation runners and
collection/learning interfaces listed above.

The [walkthroughs](../../docs/EXAMPLES.md) illustrate module inputs and outputs.
