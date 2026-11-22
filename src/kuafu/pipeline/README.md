# Pipeline and handoffs

`HandoffRun` stores inputs, outputs and attempts for resumable agent runs.
`Pipeline` coordinates program formation, collection, BC and residual RL stages.

## Install

From the repository root:

```bash
python -m pip install -r src/kuafu/pipeline/requirements.txt
```

See [runtime setup](../../../docs/INSTALL.md) for the CPU and Isaac environments.

## Input and output

| Direction | Contract | Concrete example |
|---|---|---|
| Input | Task, videos and module requests | [input.json](examples/input.json) |
| Output | Run index, step receipts and artifact hashes | [output.json](examples/output.json) |

## Use

```bash
kuafu handoff status <run_id>
```

Follow the [module guide](../../../docs/PIPELINE.md) for a complete invocation.
Run the [video-to-policy example](../../../examples/04_video_to_policy/README.md)
to inspect every handoff from source clips through collection and policy rollout.
