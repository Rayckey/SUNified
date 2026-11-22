# Residual reinforcement learning

`core/` trains PPO residual policies; `isaac/` supplies simulator entrypoints.
Start with a verified SUN Program and an evaluated BC prior. The example shows
Kinova Stack-2 parameters and a generated training launcher.

## Install

From the repository root:

```bash
python -m pip install -r src/kuafu/rl/requirements.txt
```

See [runtime setup](../../../docs/INSTALL.md) for the CPU and Isaac environments.

## Input and output

| Direction | Contract | Concrete example |
|---|---|---|
| Input | Verified program, evaluated BC prior and parameter/weight records | [input.json](examples/input.json) |
| Output | Prepared RL launcher; executing it produces logs and a residual checkpoint | [output.json](examples/output.json) |

## Use

```bash
python scripts/residual_rl/train_residual_hparams.py --help
```

Follow the [module guide](../../../docs/WORKFLOWS.md) for a complete invocation.
