# Demonstration data

`collection/` records MPC and policy demonstrations. `datasets/` combines and
validates HDF5 episodes; `robomimic/` supplies an optional training adapter.

## Install

From the repository root:

```bash
python -m pip install -r src/kuafu/data/requirements.txt
```

See [runtime setup](../../../docs/INSTALL.md) for the CPU and Isaac environments.

## Input and output

| Direction | Contract | Concrete example |
|---|---|---|
| Input | MPC or checkpoint-driven simulator rollouts | [input.json](examples/input.json) |
| Output | Compact HDF5, point-cloud/RGB recordings and dataset metadata | [output.json](examples/output.json) |

## Use

```bash
python -m kuafu.data.datasets.combine_hdf5 --help
```

Follow the [module guide](../../../docs/WORKFLOWS.md) for a complete invocation.
MPC and checkpoint collectors write aligned observations and applied actions.
Each HDF5 episode records `success`, `num_samples` and
`terminal_transition_omitted`; MPC runs also write `collection_result.json`.
Use `--max_steps` to bound an MPC run across resets. A run that stops before its
demonstration target returns exit code 2.
