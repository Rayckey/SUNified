# DP3 and simulation-to-real interfaces

`dp3/` exports demonstrations through Kuafu2Real and trains DP3/SimpleDP3
with SUN stage conditioning. `dp3/relative.py` supplies object-relative
coordinates. Follow the runtime guide for external dependencies and deployment
calibration.

## Install

From the repository root:

```bash
python -m pip install -r src/kuafu/sim2real/requirements.txt
```

See [runtime setup](../../../docs/INSTALL.md) for the CPU and Isaac environments.

## Input and output

| Direction | Contract | Concrete example |
|---|---|---|
| Input | Point-cloud episodes, measured robot state and action conventions | [input.json](examples/input.json) |
| Output | DP3 dataset, policy checkpoint and paired rollout results | [output.json](examples/output.json) |

## Use

```bash
python -m kuafu.sim2real.dp3 --help
```

Follow the [module guide](../../../docs/STAGE_DP3_EXPERIMENT.md) for a complete invocation.
