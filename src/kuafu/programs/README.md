# SUN Program formation

`SUNProgramLearner` generates task stages; the compiler turns skill constraints
into executable predicates and residuals. External agents submit the same stage
schema.

## Install

From the repository root:

```bash
python -m pip install -r src/kuafu/programs/requirements.txt
```

See [runtime setup](../../../docs/INSTALL.md) for the CPU and Isaac environments.

## Input and output

| Direction | Contract | Concrete example |
|---|---|---|
| Input | Observed scene frames, task and skill constraints | [input.json](examples/input.json) |
| Output | Saved program/subtask_*/ stages | [output.json](examples/output.json) |

## Use

```bash
kuafu agent call kuafu_contract --input src/kuafu/programs/examples/contract-request.json
```

Follow the [module guide](../../../docs/AGENT_INTERFACE.md) for a complete invocation.

Model transcripts are saved under `$KUAFU_ARTIFACT_ROOT/llm/transcripts/`.
For offline replay, put named response buffers under `$KUAFU_ARTIFACT_ROOT/llm/responses/`.
