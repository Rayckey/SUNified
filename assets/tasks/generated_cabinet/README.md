# Video 28/29 cabinet

Open the top drawer by 10 cm.

[Task inputs and references](task.json)

| Robot | Scene inspection tool |
|---|---|
| `franka_panda` | `kuafu_inspect_drawer` |

Inspect the configuration with `kuafu tasks show generated_cabinet`.

Upper drawer travel 10 cm ± 1 cm; lower drawer motion at most 5 mm.

Read the [scene and success implementation](../../../src/kuafu/programs/generated_drawer.py).

Recorded output: [SUN Program (6 stages)](../../../resources/learned_programs/generated-video28-29-cabinet/2026-10-17_19-34-23/program).
The task JSON lists every bundled program revision and its recorded status.

See the [video input and measured rollout](../../../examples/01_video_to_rollout/README.md).

Follow [installation](../../../docs/INSTALL.md) and the [task catalog](../README.md).
