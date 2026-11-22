# Pipeline inputs, outputs and recovery

Create one run for one task with `kuafu_create_run` or `kuafu handoff init`.
Keep its `run_id` on subsequent MCP, Python and shell tool calls.

```text
$KUAFU_ARTIFACT_ROOT/runs/<run_id>/
  run.json                         task, description and source-video hashes
  inputs/                          copied videos
  index.json                       attempts and tool-returned IDs
  steps/
    01-prepare-video/<attempt>/
    02-build-asset/<attempt>/
    03-inspect/<attempt>/
    04-form-program/<attempt>/
    05-verify-program/<attempt>/
    06-collect/<attempt>/
    07-train-prior/<attempt>/
    08-eval-prior/<attempt>/
    09-train-residual/<attempt>/
  learning/<attempt>/              datasets, checkpoints and training state
```

Each step attempt saves `request.json`, `response.json`, `receipt.json`,
`tool.log` and `artifacts.json` with file sizes and SHA-256 hashes. Long jobs also
save `jobs/<id>/request.json`, `status.json`, `run.log` and `worker.log`.
The run index resolves IDs after an agent or MCP client reconnects.
Simulator jobs expose their active stage and operation in `progress.json` and
`kuafu_job_status`; partial MPC traces survive an interrupted replay.

| Step | Concrete input | Inspectable output |
|---|---|---|
| Prepare video | Video paths + description | `input.json`, sampled frames, contact sheet |
| Build asset | Scene JSON + mass estimate | USD, preview images, `semantics.json` |
| Inspect | Asset + task + target travel | `scene.json`, initial images, frame observations |
| Form program | Inspection ID + proposed stages | `program/subtask_*/`, proposal and scene context |
| Verify | Program ID | `verification.json`, final images, MPC traces, `accepted/program/` |
| Collect | Verification job ID (generated cabinet), or verified program + task manifest | HDF5 demonstrations, provenance and checksums |
| Train BC | HDF5 + training settings | Configurations, metrics, checkpoints |
| Evaluate | Checkpoint + scene settings | Success statistics and program/prior identities |
| Train RL | Evaluated prior + parameters/weights | Launcher, logs, checkpoint and completion receipt |

## Inspect and resume

```bash
kuafu handoff status my-cabinet --verify
```

In Python, reconnect with the same artifact root and run ID:

```python
from kuafu.agents import ToolService
service = ToolService(run_id="my-cabinet")
state = service.call("kuafu_run_status")
result = service.call("kuafu_retry_step", {"attempt_id": "<saved-attempt-id>"})
```

A retry uses the saved request and creates a new attempt. To improve a module,
submit revised inputs to its regular tool. Read `stale` on downstream steps and
run them against the new asset/program revision. An active job can be polled or
cancelled with its job ID. A prepared RL launcher is recorded as `prepared` until
training executes.

For the generated cabinet, `kuafu_collect_drawer`, `kuafu_train_drawer_policy`
and `kuafu_evaluate_drawer_policy` keep immutable input snapshots and outputs
inside each job directory. Their `output/report.json` records artifact hashes;
policy evaluation reports physical success separately from job completion.
The [video-to-policy example](../examples/04_video_to_policy/README.md) connects
all three tools to asset and SUN Program formation.

For a registered-task training stage, leave generated-output overrides empty in the task
manifest. Kuafu places the copied manifest and job logs in the step directory,
and datasets, checkpoints and `pipeline_state.json` under `learning/<attempt>/`.
Each learning attempt inherits completed upstream artifacts and keeps its own
outputs. Evaluation receives a copy of the prior CSV; earlier results remain
inspectable. Retrying uses the saved manifest even if the original was edited.
Existing standalone CLI commands remain available for [individual modules](MODULES.md).

## Evidence and scope

A built mesh has geometry and semantic metadata. Physical verification comes
from the simulator report. A verified program includes its accepted identity;
collection, policy evaluation and RL preserve that identity. The
[video example](../examples/01_video_to_rollout/README.md) includes the exact
input clips, saved proposals, measured report and rollout image.
