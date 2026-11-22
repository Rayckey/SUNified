# Install Kuafu

Use Python 3.11 and an editable checkout. Run the installation commands from the
repository root so the packaged code can find `assets/` and `resources/`.

```bash
git clone https://github.com/Rayckey/SUNified.git Kuafu
cd Kuafu
export KUAFU_ROOT="$PWD"
export KUAFU_ARTIFACT_ROOT="$(dirname "$KUAFU_ROOT")/KuafuArtifacts"
```

## Assets and agents (CPU)

```bash
python3.11 -m venv .venv-assets
source .venv-assets/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[assets,agents]"
kuafu agent tools
```

Continue with [your agent](USAGE.md). For geometry tests, add
`python -m pip install -e ".[asset-tests]"` in this CPU environment.

## Module installation

| Module | Command from the repository root |
|---|---|
| Agent interface | `python -m pip install -e ".[agents]"` |
| Asset reconstruction | `python -m pip install -e ".[assets]"` |
| SUN Program formation | `python -m pip install -e ".[programs]"` |
| Skill grammar and constraints | `python -m pip install -e ".[skills]"` |
| MPC | `python -m pip install -e ".[mpc]"` |
| Behavior cloning | `python -m pip install -e ".[bc]"` |
| Residual RL | `python -m pip install -e ".[rl]"` |
| Data collection and conversion | `python -m pip install -e ".[data]"` |
| DP3 / simulation-to-real data | `python -m pip install -e ".[sim2real]"` |
| Isaac adapters | `python -m pip install -e ".[isaac]"` |
| Pipeline state and handoffs | `python -m pip install -e ".[pipeline]"` |
| Dynamics models | `python -m pip install -e ".[dynamics]"` |
| All module dependencies | `python -m pip install -e ".[all]"` |

Each [source module](MODULES.md) also supplies a `requirements.txt` selecting its
extra, plus input/output examples. Combine extras for a workflow, for example
`python -m pip install -e ".[agents,programs,skills,mpc,bc,rl]"`.
`all` includes the optional DreamerV3 and texture-model dependencies. Standalone
USD testing uses the separate CPU `asset-tests` extra. Isaac supplies its own USD.

## Simulation and learning

Install Isaac Sim and Isaac Lab first, then install the selected Kuafu extras in
that Python environment. The recorded cabinet example uses this tested stack:

| Component | Version |
|---|---|
| Ubuntu / Python | 24.04 / 3.11 |
| Isaac Sim / Isaac Lab | 5.1.0 / 0.54.3 |
| PyTorch / Torchvision | 2.7.0 / 0.22.0, CUDA 12.8 |
| NumPy / SciPy | 1.26.0 / 1.15.3 |
| CasADi / Pinocchio | 3.7.0 / 2.7.0 |

Follow the [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)
and [Isaac requirements](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html)
for the runtime and NVIDIA driver. Keep the source checkout available:

```bash
export ISAACLAB_ROOT=/absolute/path/to/IsaacLab
cd "$ISAACLAB_ROOT"
./isaaclab.sh --install
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py

cd "$KUAFU_ROOT"
python -m pip install -e ".[all]"
python -c "import torch, casadi, pinocchio; print(torch.cuda.is_available())"
```

In the CPU agent terminal, select that interpreter:

```bash
export KUAFU_ISAAC_PYTHON=/absolute/path/to/isaac-environment/bin/python
export KUAFU_AGENT_DEVICE=cuda:0
```

For DP3, also configure [Kuafu2Real and the DP3 source checkout](STAGE_DP3_EXPERIMENT.md).
For Robomimic, use the [data adapter](../src/kuafu/data/robomimic/README.md).

## Check the installation

```bash
kuafu --help
kuafu agent tools
python -m kuafu.skills --operator directional_move
```

Run the [examples](EXAMPLES.md) or the environment-appropriate
[development checks](DEVELOPMENT.md#validation).

| Symptom | Action |
|---|---|
| Package import fails | Activate the intended interpreter and reinstall its extras from the repository root. |
| Isaac launcher lookup fails | Set `ISAACLAB_ROOT` to the checkout containing `isaaclab.sh`. |
| Simulator launch fails | Check `nvidia-smi`, display access and the Isaac compatibility checker. |
| Output location rejected | Set `KUAFU_ARTIFACT_ROOT` to an absolute directory outside the checkout. |
| A module stops mid-run | Read its saved request, response and logs using [run status](PIPELINE.md). |
