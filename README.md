# Kuafu: Agentic Harness for Real2Sim2Real Robot Policy Learning

[![Project Website](https://img.shields.io/badge/Project-Website-2563eb)](https://kuafu-robo.github.io/) [![arXiv Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b)](https://arxiv.org/abs/2608.31167)

Kuafu is the harness for **SUN: Agentic Robot Policy Learning with Persistent
Task Programs**. It lets your agent create and verify task programs, collect
demonstrations, and train robot policies from task descriptions and object videos.

![SUN Programs across MPC verification, residual learning, and physical robot tasks](https://kuafu-robo.github.io/assets/figures/chase_sun.png?v=20260923-author)

## Installation

Clone with Python 3.11, then choose the capabilities you need:

```bash
git clone https://github.com/Rayckey/SUNified.git Kuafu
cd Kuafu
```

| What you want to do | Install in your Python environment |
|---|---|
| Connect your agent and build articulated assets from video proposals | `python -m pip install -e ".[assets,agents]"` |
| Verify SUN Programs with MPC and return feedback to your agent | `python -m pip install -e ".[agents,programs,mpc,isaac]"` |
| Train a BC policy from demonstrations | `python -m pip install -e ".[bc,data]"` |
| Improve a policy with residual RL | `python -m pip install -e ".[rl,isaac]"` |
| Export point clouds and train DP3 for sim2real | `python -m pip install -e ".[sim2real,data]"` |
| Run the complete agent → asset → SUN Program → policy workflow | `python -m pip install -e ".[all]"` |

For simulation, install **Isaac Sim / Isaac Lab** first and run the selected
install command in that environment. For DP3, configure **Kuafu2Real and DP3**.
The [installation guide](docs/INSTALL.md) gives the tested versions and commands.

## Usage

**End-to-end with your agent.** Connect an MCP client or Python agent, then give
it your object video, description and task—for example, “Open the top drawer
by 10 cm.” Kuafu returns articulated assets, SUN Programs, measured simulation
feedback, demonstrations and policy checkpoints. Start the MCP server with:

```bash
export KUAFU_ISAAC_PYTHON=/absolute/path/to/isaac-environment/bin/python
export KUAFU_ARTIFACT_ROOT=/absolute/path/to/KuafuArtifacts
kuafu agent serve
```

Use the [MCP configuration or Python interface](docs/AGENT_INTERFACE.md) to connect
your agent. Each run keeps inputs, attempts and outputs together; inspect or
resume it with `kuafu handoff status YOUR_RUN_ID --verify`.

The included video-to-policy tools handle **top-drawer opening**. Use registered
scenes for the tasks below; extend the [scene adapter](docs/DEVELOPMENT.md) to
give your agent observations and physical success checks for a new task.

**MPC as an agent tool.** Your agent calls `kuafu_submit_program`, then
`kuafu_verify_program`. It receives failed predicates, solver results and actual
object motion to guide repairs. The [MPC module](src/kuafu/mpc/README.md) explains
the input contract and output files. Browse available scenes with:

```bash
kuafu tasks list
kuafu tasks show stack2 --robot kinova_gen3
```

**BC and residual RL.** Fit a policy to your HDF5 demonstrations using a
[BC configuration](src/kuafu/bc/README.md), or collect and improve policies with a
[task manifest](docs/WORKFLOWS.md):

```bash
python -m kuafu.bc.training.train_bc_prior_mlp --config /path/to/train.yaml
kuafu -c /path/to/task.toml stage collect
kuafu -c /path/to/task.toml stage train-residual
```

**DP3 / sim2real.** Export your demonstrations and train a point-cloud policy
through Kuafu2Real. Follow the [data and runtime setup](docs/STAGE_DP3_EXPERIMENT.md),
then run one SimpleDP3 configuration with predicted SUN stage conditioning:

```bash
python -m kuafu.sim2real.dp3 train \
  --config assets/configs/experiments/stage_simple_dp3.yaml --variants C \
  --dataset /path/to/data.zarr --output "$KUAFU_ARTIFACT_ROOT/dp3-training"
```

**Try a prepared end-to-end example.** After setup, run the supplied cabinet
videos and saved agent proposals through asset construction, MPC verification,
demonstration collection, BC training and policy evaluation:

```bash
python examples/04_video_to_policy/run.py --run-id my-cabinet-policy
```

See its [inputs, outputs and rollout](examples/04_video_to_policy/README.md),
the [four optional examples](docs/EXAMPLES.md), and the full [usage guide](docs/USAGE.md).

## Tasks and robots

The [task catalog](assets/tasks/README.md) covers **nine registered tasks**:
open a drawer, place a strawberry in a drawer, stack two or three cubes, insert
a marker into a holder, hang a mug, stand up a can, and press or unlock an
emergency stop. Each entry links its scene, interaction frames, success check
and available SUN Programs.

| Robot ID | Arm and gripper | Setup |
|---|---|---|
| `fr3_finray` | Franka FR3 with custom FinRay fingers | Bundled model and default programs |
| `franka_panda` | Franka Panda with parallel gripper | Isaac asset download and default programs |
| `kinova_gen3` | Kinova Gen3 with Robotiq 2F-85 | Bundled model and default programs |
| `ur5_robotiq` | UR5 with Robotiq 2F-85 | [Bundled model and default programs](docs/UR5.md) |

All four robots have registrations for these tasks. The generated-cabinet
walkthrough uses Panda.

## Citation

```bibtex
@misc{wang2026sun,
  title = {{SUN}: Agentic Robot Policy Learning with Persistent Task Programs},
  author = {Wang, Weiqi and Li, Zhi and Lei, Yudong and Martinez, David and
            Gao, Xiaofeng and Jiang, Yuxin and Jiang, Chenfanfu and Wu, Yingnian and
            Terzopoulos, Demetri and Gong, Ran},
  year = {2026},
  eprint = {2608.31167},
  archivePrefix = {arXiv},
  primaryClass = {cs.RO},
  url = {https://arxiv.org/abs/2608.31167}
}
```

## License

Original Kuafu code, team-created object assets, reconstruction recipes, and
original FR3/FinRay additions use [Apache-2.0](LICENSE).

## Third-party notices

Upstream code and robot components retain their licenses and copyright notices:

| Component | License |
|---|---|
| [Isaac Lab](https://github.com/isaac-sim/IsaacLab)-derived code | [BSD-3-Clause](LICENSES/IsaacLab-BSD-3-Clause.txt) |
| [Franka](https://github.com/frankaemika/franka_ros) descriptions and converted Panda models | [Apache-2.0](LICENSES/Franka-Apache-2.0.txt) |
| [Kinova Gen3](https://github.com/Kinovarobotics/ros_kortex) descriptions and converted models | [BSD-3-Clause](LICENSES/Kinova-BSD-3-Clause.txt) |
| [UR5](https://github.com/ros-industrial/universal_robot/tree/noetic-devel/ur_description) descriptions, meshes and converted model | [BSD-3-Clause](LICENSES/UniversalRobots-BSD-3-Clause.txt) |
| [DP3](https://github.com/YanjieZe/3D-Diffusion-Policy) (optional dependency) | [MIT](LICENSES/DP3-MIT.txt) |
| [Robotiq components](https://github.com/ros-industrial/robotiq) in the FR3/FinRay and UR5 models | [BSD-2-Clause](LICENSES/Robotiq-BSD-2-Clause.txt) |

Separately installed runtimes, libraries, and downloaded model weights retain
their respective licenses.
