# Agent interface

`ToolService` exposes the same contracts through Python, MCP and JSON CLI.
Bind a `run_id` for resumable work.

## Install

From the repository root:

```bash
python -m pip install -r src/kuafu/agents/requirements.txt
```

See [runtime setup](../../../docs/INSTALL.md) for the CPU and Isaac environments.

## Input and output

| Direction | Contract | Concrete example |
|---|---|---|
| Input | Agent tool arguments and scene/program proposals | [input.json](examples/input.json) |
| Output | Structured responses, job IDs and accepted programs | [output.json](examples/output.json) |

## Use

```bash
kuafu agent tools
```

Follow the [module guide](../../../docs/AGENT_INTERFACE.md) for a complete invocation.
