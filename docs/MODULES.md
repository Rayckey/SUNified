# Use individual modules

Install the module extra from the repository root, then open its README for
its input/output example and runtime requirements. Each folder also provides a
`requirements.txt` selecting the same extra.

| Module | Source folder | Installation extra |
|---|---|---|
| [Pipeline and handoffs](../src/kuafu/pipeline/README.md) | `src/kuafu/pipeline/` | `.[pipeline]` |
| [Agent interface](../src/kuafu/agents/README.md) | `src/kuafu/agents/` | `.[agents]` |
| [Asset reconstruction](../src/kuafu/asset_generation/README.md) | `src/kuafu/asset_generation/` | `.[assets]` |
| [SUN Program formation](../src/kuafu/programs/README.md) | `src/kuafu/programs/` | `.[programs]` |
| [SUN skill set](../src/kuafu/skills/README.md) | `src/kuafu/skills/` | `.[skills]` |
| [Model predictive control](../src/kuafu/mpc/README.md) | `src/kuafu/mpc/` | `.[mpc]` |
| [Behavior cloning](../src/kuafu/bc/README.md) | `src/kuafu/bc/` | `.[bc]` |
| [Residual reinforcement learning](../src/kuafu/rl/README.md) | `src/kuafu/rl/` | `.[rl]` |
| [DP3 and simulation-to-real interfaces](../src/kuafu/sim2real/README.md) | `src/kuafu/sim2real/` | `.[sim2real]` |
| [Demonstration data](../src/kuafu/data/README.md) | `src/kuafu/data/` | `.[data]` |
| [Isaac simulation adapters](../src/kuafu/isaac/README.md) | `src/kuafu/isaac/` | `.[isaac]` |
| [Dynamics models](../src/kuafu/dynamics/README.md) | `src/kuafu/dynamics/` | `.[dynamics]` |

Use `python -m pip install -e ".[all]"` for all module dependencies after
[setting up the runtime](INSTALL.md). The [complete agent workflow](PIPELINE.md)
connects these modules with persistent handoff records.
