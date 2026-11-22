# Language and interaction frames

Add an object description, interaction frames, and local axis meanings through
`SceneSpec.semantics`. Each moving joint needs a frame attached to its child body
or a fixed descendant.

| Export | Contents |
|---|---|
| `semantics.json` and `asset.json.semantics` | Descriptions, roles, body-local poses, axes, joints, limits, region sizes, and uncertainty |
| USD `InteractionFrames` | Named Xforms that follow their owning bodies |
| `isaaclab_semantics.py` | Frame-sensor and `detectable_objects` helpers, also exposed by `isaaclab_snippet.py` |

Positions use meters and quaternions use **wxyz** order. Opening/closing axes
come from the joint, opening sign, and frame orientation. Revolute axes follow
the right-hand rule. Transform a local axis to world coordinates with
`quat_apply(sensor.data.target_quat_w[:, 0], local_axis)`; use
`target_quat_source` for source-frame directions.

Place frames at grasp points and task-relevant regions. Name their axes in the
object description so the agent can relate language instructions to geometry.
Read the exported `semantics.json` for each frame's pose, role, and axis meanings.

## Load into an existing environment

After `AppLauncher` starts, load the exported configuration:

```python
import runpy
from isaaclab.assets import Articulation

generated = runpy.run_path("/path/to/asset/isaaclab_snippet.py")
cabinet = generated["ARTICULATED_OBJECT"].replace(
    prim_path="/World/envs/env_.*/Cabinet",
)
```

For `DirectRLEnv`, register the asset during `_setup_scene`, after base setup and
before cloning or resetting simulation:

```python
self.scene.articulations["cabinet"] = Articulation(self.cfg.cabinet)
generated["attach_semantics"](
    self,
    "cabinet",
    source_prim_path="/World/envs/env_.*/Robot/panda_link0",
    debug_vis=True,
)
self.scene.clone_environments(copy_from_source=False)
```

Use the scene asset key (`"cabinet"`) and the robot's rigid base link. The helper
registers sensors and merges descriptions into `cfg.detectable_objects`. Sensors
target owning rigid bodies with local `OffsetCfg` poses. Replace old cabinet
sensors, or set `frame_prefix="cabinet2_"` for a second asset.

For `InteractiveSceneCfg`, add the configurations before constructing the scene:

```python
detectable_objects, frames = generated["make_semantic_cfg"](
    cabinet, "/World/envs/env_.*/Robot/panda_link0", object_name="cabinet"
)
for name, cfg in frames.items():
    setattr(scene_cfg, name, cfg)
env_cfg.detectable_objects = {**env_cfg.detectable_objects, **detectable_objects}
```

Use `DYNAMIC_OBJECT` and `scene.rigid_objects` for rigid assets. Use uniform spawn
scales, or rebuild at a measured scale to update geometry, travel, and frames together.

## Edit and verify

Define input frame poses in the **assembled rest coordinate system before scale
and centering**. Export converts these absolute assembled poses to body-local
poses. The optional `part` field checks ownership. Define unit axes in frame
coordinates and place hinge frames at their revolute joint origin.

Read task points from `SEMANTICS["frames"]` and geometric part origins from
`PART_FRAMES`. Check imported textures and meshes against the interaction frames.

Run `kuafu assets validate` for mechanism physics, then check actual Isaac Lab
sensors on rotated and scaled instances:

```bash
/path/to/isaaclab/python scripts/assets/check_semantics_isaaclab.py \
  --asset-dir /path/to/asset --report /path/to/semantic_check.json
```

The report records artifact hashes, motion, and sensor/axis errors. Video-derived
positions and dimensions are estimates; use the robot task check below to verify
interaction.

## Execute a cabinet task in the robot scene

Activate the [Isaac environment](INSTALL.md). Inspect an exported cabinet from
Kuafu or the standalone asset pipeline:

```bash
KUAFU_PYTHON=/path/to/isaaclab/python ./bin/kuafu verify-drawer \
  --headless --enable_cameras --device cuda:0 \
  --asset /absolute/path/to/exported/cabinet \
  --mode inspect --output drawer_check/inspect
```

Inspection loads the exported collision geometry, registers its frames, settles
the scene, and saves observations and camera views. The default placement is
`(0.6, -0.5, 0.2)` m with 180-degree yaw. Set `--position X Y Z` and `--yaw` to
change it, leaving room for drawer travel and gripper retreat.

Choose how to create the program:

- **Your agent:** use the [inspect/propose/verify loop](AGENT_INTERFACE.md).
- **Built-in agent:** set `OPENAI_API_KEY`, then use `--mode generate
  --max-requests 20 --output drawer_check/generation`. Select `--model` as needed
  (default `gpt-5.5`). The runner sends text observations and constraint context;
  camera images stay local.
- **Saved program:** replay the generated `program/` directory:

```bash
KUAFU_PYTHON=/path/to/isaaclab/python ./bin/kuafu verify-drawer \
  --headless --enable_cameras --device cuda:0 \
  --asset /absolute/path/to/exported/cabinet \
  --mode replay --program /absolute/path/to/generation/program \
  --output drawer_check/replay
```

Give each run a new output directory under `KUAFU_ARTIFACT_ROOT`. Read
`verification.json`: a pass requires every stage, successful finite MPC actions,
final `upper_drawer_joint` travel within 1 cm of `--target-travel` (default
0.10 m), and other-drawer disturbance below 5 mm throughout execution. Keep
`--task` consistent with that target.

The report includes joint/frame traces, solver results, bounded pose-IK errors,
artifact hashes, and stage diagnostics.

To reuse our cabinet and verified SUN Program, follow the
[optional replay example](../examples/02_program_to_rollout/README.md).
