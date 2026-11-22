# Connect your agent

Connect your MCP client, Python agent, or shell agent to Kuafu's shared tools.
Your agent chooses its provider and handles proposals and repairs; Kuafu builds,
compiles, simulates, and returns measured feedback.

## Install and connect

Complete the [CPU installation](INSTALL.md#assets-and-agents-cpu), then configure
the agent connection:

```bash
export KUAFU_ARTIFACT_ROOT=/absolute/path/to/KuafuArtifacts
export KUAFU_ISAAC_PYTHON=/absolute/path/to/isaac-environment/bin/python
export KUAFU_AGENT_DEVICE=cuda:0
export KUAFU_AGENT_JOB_TIMEOUT=900
```

The server uses the CPU environment and launches simulation through
`KUAFU_ISAAC_PYTHON`. Set the job timeout up to 86400 seconds for longer runs.

### MCP clients

Add this local stdio server to your client's configuration, using absolute paths:

```json
{
  "mcpServers": {
    "kuafu": {
      "command": "/absolute/path/to/Kuafu/.venv-assets/bin/python",
      "args": ["-m", "kuafu", "agent", "serve"],
      "env": {
        "KUAFU_ARTIFACT_ROOT": "/absolute/path/to/KuafuArtifacts",
        "KUAFU_ISAAC_PYTHON": "/absolute/path/to/isaac-environment/bin/python",
        "KUAFU_AGENT_DEVICE": "cuda:0",
        "KUAFU_AGENT_JOB_TIMEOUT": "900"
      }
    }
  }
}
```

Give your agent [your task, inputs, and success criterion](USAGE.md#start-a-task).

The server uses the [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
2.x. Image reads return MCP image content; your host controls provider access.

### Python agents

```python
from kuafu.agents import ToolService

kuafu = ToolService(
    artifact_root="/absolute/path/to/KuafuArtifacts",
    isaac_python="/absolute/path/to/isaac-environment/bin/python",
)
tools = kuafu.list_tools()
created = kuafu.call("kuafu_create_run", {"task": "Open the top drawer by 10 cm."})
kuafu.run_id = created["result"]["run_id"]
result = kuafu.call("kuafu_contract", {"kind": "program"})
```

Register `tools` with your framework. Dispatch each agent response through
`kuafu.call(tool_name, arguments)` and return the result as its observation.
Set revision and inference limits in your agent's loop.

### Shell agents

```bash
kuafu agent tools
kuafu handoff init --run-id my-cabinet --task "Open the top drawer by 10 cm."
export KUAFU_RUN_ID=my-cabinet
printf '%s\n' '{"kind":"program"}' | kuafu agent call kuafu_contract
kuafu agent call kuafu_submit_program --input proposal-request.json
```

Read `ok` for tool-call success and a job's `status` for execution state.
Tool errors return `ok: false` and CLI exit code 2; diagnostics go to stderr.

## Persistent runs

Start with `kuafu_create_run` and pass its `run_id` on each tool call.
Use `kuafu_run_status` to inspect all inputs, outputs and logs, and
`kuafu_retry_step` to repeat a finished step in a new attempt directory.
Set `verify_outputs: true` when checking saved files before resuming.
See the [handoff guide](PIPELINE.md).

## Tools

| Tool | Purpose |
|---|---|
| `kuafu_list_tasks` | Discover published task examples and available robots. |
| `kuafu_get_task` | Read task inputs, assets, registrations, success checks and SUN Program examples. |
| `kuafu_create_run` | Save task inputs and create one run directory. |
| `kuafu_run_status` | Inspect attempts, files, hashes and recovery state. |
| `kuafu_retry_step` | Retry saved inputs while keeping prior attempts. |
| `kuafu_contract` | Read asset/program schemas and grounding instructions. |
| `kuafu_prepare_video` | Sample videos with an object description. |
| `kuafu_build_asset` | Build scene JSON into USD, previews, joints, and semantics. |
| `kuafu_read_artifact` | Read an artifact/resource file or image. |
| `kuafu_inspect_drawer` | Observe a cabinet at a selected pose and target travel. |
| `kuafu_submit_program` | Submit stages against a successful inspection job. |
| `kuafu_verify_program` | Run the candidate in MPC and simulation. |
| `kuafu_collect_drawer` | Collect verified MPC episodes in the generated cabinet scene. |
| `kuafu_train_drawer_policy` | Fit a BC policy to those episodes, preserving the scene/program contract. |
| `kuafu_evaluate_drawer_policy` | Execute the policy and measure physical success in that same scene. |
| `kuafu_job_status` | Read progress, observations, failures, and reports. |
| `kuafu_cancel_job` | Stop a job and preserve its diagnostics. |
| `kuafu_pipeline_status` | Read a task pipeline's state. |
| `kuafu_start_stage` | Run collection, prior training/evaluation, or residual training. |

## Verify and repair

1. Read the contracts for your inputs. For video reconstruction, prepare frames,
   build scene JSON, and review previews and `semantics.json`. For an existing
   asset, check its interaction frames and physics. Give revisions new directories.
2. Inspect the scene through its task adapter and poll the job. Ground the
   program in its observed frames.
3. Submit `{"stages": [...]}` using the program contract. Each stage contains
   `task`, `atomic_actions`, and `constraints` (`Reasoning`, `Constraints`).
4. Verify, read failed predicates and joint travel, and revise against the same
   task and physical target. Inspect again after changing the asset.
5. Use `accepted_run` after all stages, MPC solves, and physical checks pass.
   Preserve its verified `program/` snapshot and `accepted_identity` path/hash
   for downstream parameter and weight metadata.

Each verification resets and executes the complete program. Jobs persist across
client disconnects; reconnect with the same artifact root and poll the job ID.

The current adapter, `kuafu_inspect_drawer`, checks `upper_drawer_joint` against
the selected top-drawer travel while the lower drawers stay still. Add scene
observations, reset behavior, and a physical acceptance check to support another
task. For a supplied asset and a Python repair loop, see the
[optional examples](EXAMPLES.md).

## Collection and learning

For the generated cabinet, pass the successful verification job ID to
`kuafu_collect_drawer`. Pass its completed job ID to `kuafu_train_drawer_policy`,
then the training job ID to `kuafu_evaluate_drawer_policy`. Training defaults to
nearest-neighbor BC; choose `method: "mlp"` and `epochs` for stage-specific neural
policies. These tools copy the
verified asset and program into each attempt, hash the handoff files and retain
the measured task target. Set `KUAFU_AGENT_JOB_TIMEOUT=3600` for collection.
Evaluation returns `result.policy_passed` separately from job completion; a
completed evaluation can report a failed policy. See the
[end-to-end example](../examples/04_video_to_policy/README.md).

Choose a registered task compatible with the accepted program, datasets, and
checkpoints. Create your [task manifest](../assets/configs/README.md) and merge these
settings to keep generation and tuning with your agent:

```toml
[learn_programs]
enabled = false

[eval]
acquire_weights = false
skip_parameter_search = true

[residual]
hparam_iterations = 0
weights_params = "/absolute/path/to/agent-weights.json"

[residual.env]
INITIAL_AGENT_HYPERPARAMETERS = "/absolute/path/to/agent-hyperparameters.json"
```

Select the verified program using the manifest's program paths. Weights carry
native program/prior identity metadata; `[eval].params_path` selects an existing
parameter file. Hyperparameter JSON accepts `clip_actions`, `residual_bounds`,
`clip_composed_actions`, `alpha_final`, and `alpha_warmup_epochs`. Match the bounds
to the action space, or use the task defaults by omitting the hyperparameter file.

Zero tuning iterations runs the initial training trial. Set `INIT_TRAIN_EPOCH`
in the generated launcher for its duration. Use `prepare_only: true` to review
the launcher. Submit revised inputs with the same `run_id` to create a new
learning attempt while retaining earlier outputs. Stage tools execute your
trusted local manifests and scripts; see [workflows](WORKFLOWS.md).
