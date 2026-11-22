# Using Kuafu with your agent

Call the harness **Kuafu** and its task programs **SUN Programs**. Use `kuafu`
for the package, commands, and MCP prefixes; `KUAFU_*` for settings; and
`KuafuArtifacts` for external output. Keep genuine repository URLs in attribution.

Keep validation results, test logs, internal reviews and analysis writeups outside
Git in `KUAFU_ARTIFACT_ROOT`. Commit implementation, tests and user-facing guides.

Push changes to `master` by default. Push to a different branch only when the
user explicitly requests a proper release.

Use these names consistently:

| Purpose | Name |
|---|---|
| Generation stage / manifest | `learn-programs` / `[learn_programs]` |
| Bundled programs / saved stages | `resources/learned_programs/` / `program/subtask_*/` |
| Program settings | `program_path`, `program_paths`, `load_program`, `KUAFU_PROGRAM_PATH` |
| Executor generation owner | `program_learner` (`SUNProgramLearner`) |
| Evaluator stage owner | `compiled_program` (`CompiledSUNProgram`) |
| Shared executor/evaluator handle | `program_runtime` |
| Robot identifiers | `fr3_finray`, `franka_panda`, `kinova_gen3`, `ur5_robotiq` |
| Gym families | `FR3-FinRay-Table-*`, `Panda-Table-*`, `Kinova-Table-Gen3-*`, `UR5-Table-*` |

Use the [component names](docs/ARCHITECTURE.md#sun-program-components), including
`SUNProgramExecutor`, `SUNProgramEvaluator`, `SUNProgramStage`, `SUNProgramPlan`,
`ResidualRunner`, and `DemonstrationRecorder`.

Follow [agent setup](docs/AGENT_INTERFACE.md). Discover existing tasks with
`kuafu_list_tasks` and read their inputs with `kuafu_get_task`. Create a task with `kuafu_create_run`
and keep its `run_id` on tool calls. Read `kuafu_contract`; ground proposals in the user's videos, description, and scene
observations. Feed previews, errors, and physical results back into the repair
loop while preserving the target. Use `accepted_run` for verified programs.
For the generated cabinet, pass the verification job ID to `kuafu_collect_drawer`,
then continue with `kuafu_train_drawer_policy` and `kuafu_evaluate_drawer_policy`.
Read `policy_passed` separately from evaluation job completion.
Inspect `kuafu_run_status` before resuming and retain prior attempt directories.

Run agent tools in the CPU environment and physics jobs through
`KUAFU_ISAAC_PYTHON`. The host manages its provider and inference budget. The
included adapter checks top-drawer opening; add observations and physical
acceptance logic for other tasks.

Follow [CONTRIBUTING.md](CONTRIBUTING.md) for source changes. Keep documentation
concise and action-oriented. Keep the README introduction brief, with citation,
license, and third-party notices at the end. Include compact Installation and
Usage sections in that order, linking to `docs/INSTALL.md` and `docs/USAGE.md` for full
instructions. Show the minimal agent setup and keep prepared examples optional.
Keep algorithm exposition in the paper; repository guides cover setup, usage,
and interfaces. Center usage on the user's agent, task, and inputs; keep prepared
assets, saved programs, and experiment walkthroughs in `docs/EXAMPLES.md` as
optional examples.
Test asset/agent interfaces in the CPU environment
and controllers in the Isaac environment. Preserve receiver-owned tolerances,
frame conventions, and verified program identity across handoffs.
