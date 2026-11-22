# Use Kuafu with your agent

Complete [installation](INSTALL.md), then connect your agent through
[MCP or Python](AGENT_INTERFACE.md). Your agent supplies the model and proposal
loop; Kuafu exposes asset, program, simulation and learning tools.

## Start a task

Browse the [task catalog](../assets/tasks/README.md) with `kuafu tasks list`.
Use `kuafu tasks show stack2 --robot kinova_gen3` to inspect a task's registered
scene, object assets, success checks and saved SUN Programs. Agents use
`kuafu_list_tasks` and `kuafu_get_task` for the same information.

Give your agent a video, a short object description, a task and a measurable
success criterion. For example: “This is one cabinet with three sliding drawers.
Open its top drawer by 10 cm and leave the other drawers closed.”

Create a run through `kuafu_create_run`, or from the shell:

```bash
kuafu handoff init --run-id my-cabinet \
  --video /absolute/path/to/cabinet.mov \
  --description "One cabinet with three sliding drawers." \
  --task "Open the top drawer by 10 cm."
```

Pass `run_id: "my-cabinet"` with each tool call, or set `KUAFU_RUN_ID=my-cabinet`
for a dedicated agent process. Inputs and all step attempts live together under
`$KUAFU_ARTIFACT_ROOT/runs/my-cabinet/`.

## Run the agent loop

1. Read `kuafu_contract`; prepare the video with `kuafu_prepare_video`.
2. Propose scene JSON, call `kuafu_build_asset`, and review geometry and semantics.
3. Call `kuafu_inspect_drawer` and poll `kuafu_job_status` for scene observations.
4. Propose SUN Program stages, then call `kuafu_submit_program` and `kuafu_verify_program`.
5. Read measured feedback, revise the failing module, and repeat verification.
6. For the generated cabinet, call `kuafu_collect_drawer`, `kuafu_train_drawer_policy`,
   and `kuafu_evaluate_drawer_policy`, passing each completed job ID to the next tool.
   For registered scenes, use the [collection and learning workflow](WORKFLOWS.md).

The included scene adapter checks top-drawer opening. Add a task adapter with its
own observations and physical success checks for another manipulation task.

## Continue after interruption

```bash
kuafu handoff status my-cabinet --verify
```

Read the failed step's `request.json`, `response.json`, `tool.log` and job logs.
Call `kuafu_retry_step` with its `attempt_id`, or call the module again with
revised inputs. Each attempt gets a new directory. Revisions mark older
downstream results as stale; continue from the revised module.

See the [handoff layout](PIPELINE.md), [module interfaces](MODULES.md), and
[four optional examples](EXAMPLES.md).
