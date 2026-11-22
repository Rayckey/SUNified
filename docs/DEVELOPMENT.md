# Develop Kuafu

Use an editable installation and start with the [module map](MODULES.md).
Import shared paths from `kuafu.paths`; save generated files in a handoff run
under `KUAFU_ARTIFACT_ROOT`.

## Repository layout

```text
docs/          installation, usage, examples and development guides
assets/        tasks, prompts, skills, configs, recipes, objects and robots
examples/      four reproducible input/output examples
scripts/       launchers and operational tools
tests/         unit, contract and simulator checks
resources/     saved programs, experiment definitions and checkpoint catalogs
src/kuafu/     modules and shared runtime utilities
pyproject.toml module installation extras and command entrypoints
```

Each module owns a README, `requirements.txt`, and `examples/input.json` plus
`examples/output.json`. Those files point to concrete repository inputs and
recorded outputs. Update them with interface changes.

## Replace a module

1. Open the affected run's latest step request and response.
2. Read the module's input/output example and public entrypoint.
3. Implement the change and run its focused tests.
4. Call that module again using the same run ID and revised inputs.
5. Continue downstream from the new attempt and inspect the physical result.

Keep units, pose frames, quaternion order and program identity explicit. Numeric
acceptance tolerances belong to the receiver. Frame sensors follow their rigid
bodies, and joint definitions determine motion axes.

## Add a task or skill

Place task data and a README in `assets/tasks/<task>/`, then add the entry to
`assets/tasks/index.json`. Include instructions, robot environment IDs, asset
references, success checks and saved program examples. Check discovery with
`kuafu tasks show <task>` and `tests/test_task_catalog.py`.

Place prompts in `assets/prompts/` and skill contracts in
`assets/skills/<operator>/`. Implement new operators in
`kuafu.skills`, program compilation in `kuafu.programs`, and scene adapters in
`kuafu.isaac`. Add observations, deterministic reset behavior and an independent
physical success check before exposing a task through agent tools.

Refresh each registered robot's `configuration` snapshot from
`kuafu.isaac.check_task_configs.configuration_inputs` when its assets, frames or
default program change. Run the configuration check in the Isaac environment:

```bash
python -m kuafu.isaac.check_task_configs --headless --device cpu --variants
```

It loads registered configurations, compares base inputs with the catalog and
checks local files. Its report goes to
`$KUAFU_ARTIFACT_ROOT/validation/task-configurations.json`.

Reset and step one task with its native assets and configured program:

```bash
python -m kuafu.isaac.check_task_runtime --task stack2 --robot franka_panda \
  --headless --device cuda:0 --output validation/stack2-panda
```

`result.json` records finite observations/rewards and actuator response. Run the
[MPC task replay](../assets/tasks/README.md) to measure physical completion.
Keep these outputs in `KUAFU_ARTIFACT_ROOT`.

## Validation

GitHub Actions checks fresh installs for assets/agents, MPC, BC and `[all]`, runs
`pip check`, and exercises the CPU contracts and learning code. A separate job
builds the source archive, installs it outside the checkout and checks bundled
tasks, prompts, skills, assets and examples. Test reports include skipped tests.

Run the simulator examples locally for physical acceptance. Configure the
external checkouts for [DP3](STAGE_DP3_EXPERIMENT.md) and
[DreamerV3](DREAMERV3.md) before their runtime tests. GitHub's hosted CPU jobs
cover the repository modules; these runtime checks add the upstream models and
Isaac physics.

In the CPU asset environment:

```bash
python -m pip install -e ".[agents,assets,asset-tests]"
python -m pytest -q tests/agents tests/asset_generation tests/test_source_integrity.py tests/test_task_catalog.py
```

In the Isaac environment, run the controller/learning tests and a replay:

```bash
python -m pytest -q tests --ignore=tests/agents --ignore=tests/asset_generation
```

Run the [cabinet replay](../examples/02_program_to_rollout/README.md) after changes
to frames, collision geometry, grippers, MPC or program compilation. Record the
asset hashes, seed, pose, joint travel and solver result. CPU tests cover agent
reconnects, interrupted steps, retries and stale downstream artifacts.
