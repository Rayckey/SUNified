# Dynamics models

Train a state-transition ensemble from recorded observations and actions.
DreamerV3 and video-model adapters have separate guides and installation extras.

## Install

From the repository root:

```bash
python -m pip install -r src/kuafu/dynamics/requirements.txt
```

See [runtime setup](../../../docs/INSTALL.md) for the CPU and Isaac environments.

## Input and output

| Direction | Contract | Concrete example |
|---|---|---|
| Input | Recorded observation/action/next-observation transitions | [input.json](examples/input.json) |
| Output | Dataset inspection: episode counts and tensor dimensions | [output.json](examples/output.json) |

## Use

```bash
kuafu train-dynamics --help
```

Follow the [module guide](../../../docs/DYNAMICS.md) for a complete invocation.
