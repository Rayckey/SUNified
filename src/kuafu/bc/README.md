# Behavior cloning

`training/` provides the native MLP trainer, checkpoint export and evaluation.
The [four-trajectory example](../../../examples/03_demonstrations_to_policy/README.md)
runs on CPU. `generated_drawer.py` also supports a nearest-neighbor BC baseline
and stage-specific MLPs for the [video-to-policy example](../../../examples/04_video_to_policy/README.md).

## Install

From the repository root:

```bash
python -m pip install -r src/kuafu/bc/requirements.txt
```

See [runtime setup](../../../docs/INSTALL.md) for the CPU and Isaac environments.

## Input and output

| Direction | Contract | Concrete example |
|---|---|---|
| Input | Recorded observations/actions and training configuration | [input.json](examples/input.json) |
| Output | Checkpoint, normalizers and validation curves | [output.json](examples/output.json) |

## Use

```bash
python examples/03_demonstrations_to_policy/run.py --output examples/my-bc-run
```

Follow the [module guide](../../../examples/03_demonstrations_to_policy/README.md) for a complete invocation.
For a generated cabinet, your agent calls `kuafu_train_drawer_policy` with a
completed collection job ID. Choose `method: "nearest_neighbor"` for the small
example or `method: "mlp"` with an `epochs` budget for neural BC. Evaluate with
`kuafu_evaluate_drawer_policy`; read `result.policy_passed` and the measured
drawer travel in its report.
