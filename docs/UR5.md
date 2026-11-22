# UR5 tabletop environments

Kuafu provides a six-joint UR5 with a Robotiq 2F-85 gripper under robot ID
`ur5_robotiq` (`ur5` is an alias). It reuses the existing tabletop scenes,
randomization, and physical success checks. The robot, gripper, meshes and
controller URDF are bundled in [assets/robots/ur5_robotiq](../assets/robots/ur5_robotiq/).
Each of the nine tasks selects a bundled SUN Program for MPC collection.
Checkpoint-based collection uses a UR5 policy.

| Task | Gym ID |
|---|---|
| Open drawer | `UR5-Table-Drawer` |
| Drawer and stacking | `UR5-Table-Drawer-Stack` |
| Stack two / three | `UR5-Table-Stack-2`, `UR5-Table-Stack-3` |
| Hang cup | `UR5-Table-Hang-Cup` |
| Can | `UR5-Table-Can` |
| Pen and cup | `UR5-Table-Pen-Cup` |
| Press / release E-stop | `UR5-Table-EStop`, `UR5-Table-EStop-Closed` |

## Run

Use the Isaac Python configured in `KUAFU_ISAAC_PYTHON`. From the Kuafu checkout:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
"$KUAFU_ISAAC_PYTHON" -m kuafu.isaac.learned.ur5.smoke \
  --task UR5-Table-Stack-2 --num-envs 2 --headless --enable_cameras --device cuda:0
```

The default loader reads the bundled model. Set `KUAFU_UR5_ASSET_DIR` to select
a custom model directory containing `ur5_robotiq.usd` and `ur5_controller.urdf`.
The bundled `provenance.json` records the upstream ROS-Industrial revisions and
mesh hashes; `manifest.json` records the conversion and output hashes.
The virtual grasp center is 0.162 m along wrist_3_link's +Z axis. Calibrate the
mount, tool frame and cameras for physical deployment.

Use `--task all` to check every task in separate simulator processes. Add
`--action-mode relative_ee --move-arm --check-mpc` to exercise Cartesian motion
and a real MPC/IK solve. Successful checks write a JSON report and RGB previews.
These checks establish environment/control operation, not a trained policy's
task success rate.

To execute the task's SUN Program and collect successful trajectories, use the
MPC collector. UR5 defaults to joint targets computed with bounded inverse kinematics:

```bash
"$KUAFU_ISAAC_PYTHON" -m kuafu.data.collection.collect_from_mpc \
  --task_name UR5-Table-Drawer-Stack --num_envs 1 --num_demos 1 \
  --seed 42 --max_steps 900 --action_noise_std 0 \
  --headless --device cuda:0 --output_dir ur5/drawer-stack
```

Select another Gym ID from the table to run that task. Read
`collection_result.json` for successful episodes and `datasets/` for trajectories
under `$KUAFU_ARTIFACT_ROOT/ur5/drawer-stack`. Use `--program_path` to supply a
different program. With multiple simulator processes, give each a GPU through
`CUDA_VISIBLE_DEVICES` and use `--device cuda:0` within each process.

## Rebuild the model

To change the geometry, edit the bundled `source/` URDFs and rebuild with Isaac.
The converter reads local meshes and writes a standalone USD plus controller URDF:

```bash
"$KUAFU_ISAAC_PYTHON" -m kuafu.isaac.learned.ur5.prepare_assets \
  --output-dir "$KUAFU_ARTIFACT_ROOT/robots/ur5_robotiq"
export KUAFU_UR5_ASSET_DIR="$KUAFU_ARTIFACT_ROOT/robots/ur5_robotiq"
```

## Observations and actions

The returned observations are `robot_state` and `rgb`, plus `policy` as an alias
of `robot_state` for existing RL wrappers. They contain no cube/object poses,
task labels, or program stages. Object state remains internal to task rewards
and success evaluation.

- `robot_state`: float32 `[num_envs, 8]`, joint positions in radians, ordered
  `shoulder_pan_joint`, `shoulder_lift_joint`, `elbow_joint`, `wrist_1_joint`,
  `wrist_2_joint`, `wrist_3_joint`, `finger_joint`, `right_outer_knuckle_joint`.
  The last two values mirror the single physical gripper driver, matching
  Kuafu's two-slot gripper interface; 0 is open and 0.8 is closed.
- `rgb`: uint8 `[num_envs, 2, 72, 128, 3]`, ordered `table_cam_0`, `table_cam_1`.
  Both are fixed wide views of the UR5 workspace. Launch Isaac with
  `--enable_cameras`; setting `cfg.enable_cameras=False` omits this field.
- Default `relative_joint` actions have eight values: six normalized arm
  increments and two gripper commands. Positive gripper commands open,
  negative commands close, and zero holds.
- `absolute_joint` has eight values in radians. `relative_ee` has eight
  values (translation/axis-angle increments and gripper commands);
  `absolute_ee` has eleven (position, 6D rotation, two gripper joint targets).
  Cartesian poses are in the robot base frame; quaternion interfaces use wxyz.

The control rate is 30 Hz. Existing manual task rewards are used where provided;
tasks without one receive the physical success indicator. A supplied SUN Program
can still drive stage evaluation. The `panda_fingertip`, `panda_link4`, and
`panda_link7` program frame names alias the UR5 grasp center, forearm, and wrist.

To construct a task in your own script, start `AppLauncher` first, import
`kuafu.isaac.learned.ur5`, then call `parse_env_cfg` and `gym.make` as for
other Kuafu tasks. Use `obs_mode="robot_state"`; privileged observation modes
are deliberately unavailable for this family. The supplied RL-games config
uses the `policy` robot-state tensor; Explore2Plan consumes both modalities.

## Collect and train Explore2Plan

Use a checkpoint trained for this UR5 observation/action interface:

```bash
"$KUAFU_ISAAC_PYTHON" -m kuafu.data.collection.collect_with_checkpoint \
  --task UR5-Table-Stack-2 --checkpoint /path/to/ur5/checkpoint.pth \
  --num_envs 2 --num_demos 100 --rgb_transitions --stride 1 \
  --headless --enable_cameras --device cuda:0
```

The native RGB recorder writes `obs/robot_state`, `obs/rgb`, applied actions,
and joint-order metadata. It retains successful episodes and omits terminal
transitions whose next observation would be an autoreset frame. It does not
write object state for UR5. The MPC entry point above uses the task's bundled
SUN Program.

From the Explore2Plan checkout, train on the resulting HDF5:

```bash
python -m manipulation.train_kuafu --dataset /path/to/rgb_demos.hdf5 \
  --output /path/to/new/ur5-run --device cuda:0
```

The adapter infers eight robot inputs from `obs/robot_state`, checks camera and
joint layout, and uses only robot state and RGB. Legacy nine-joint Kinova files
remain supported. Train separate models for differing joint layouts; mixing
UR5 and Kinova recordings in one run is rejected.
