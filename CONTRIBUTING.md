# Contributing to Kuafu

Start with the [README](README.md) and [architecture](docs/ARCHITECTURE.md).
Use an editable source installation, store run output under
`KUAFU_ARTIFACT_ROOT`, and inject provider credentials through the environment.
Keep validation reports and internal analysis in that external artifact directory.
Summarize checks in the pull request; commit the tests and user-facing instructions.

## Validate your change

In the CPU asset environment:

```bash
python -m pip install -e ".[assets,asset-tests,agents]"
python -m pytest -q tests/asset_generation tests/agents tests/test_source_integrity.py tests/test_task_catalog.py
```

In the [Isaac environment](docs/INSTALL.md):

```bash
python -m pytest -q tests --ignore=tests/asset_generation --ignore=tests/agents
```

For geometry, collision, frame, gripper, or control changes, also run the
[cabinet replay](examples/02_program_to_rollout/README.md). Record
asset hashes, task, seed, pose, joint travel, and solver results under
`KUAFU_ARTIFACT_ROOT`.

## Preserve the contracts

- Document pose reference frames, meters, and **wxyz** quaternions. Attach moving
  interaction sensors to owning rigid bodies; derive motion axes from joints.
- Keep formulation scenes responsible for physics and observations, with zero
  rewards and suppressed done signals. The generated program owns predicates
  and residuals.
- Keep numeric tolerances receiver-owned and preserve verified program identity
  through collection, evaluation, and residual training.
- Share agent schemas/handlers across MCP, Python, and JSON. Preserve the
  inspected physical target during repair and promote physically verified runs.
- Restore completed stages and the remaining plan after a restart. Count model
  attempts and control steps across controller rebuilds.
- Resolve resources through `kuafu.paths` and output through the external
  artifact root.

Write concise instructions that say what to do. Use docstrings for public
interfaces and comments for invariants, numerical choices, and compatibility
workarounds. Remove obsolete experiments, placeholders, and comments that repeat
code. Preserve extraction markers such as `# <residual>`, tool directives, and
upstream notices.

## Submit

Describe the problem, resulting behavior, and validation. Add a regression test
for correctness fixes. Record each new asset's source, terms, dimensions, and
uncertainty. Original contributions use [Apache-2.0](LICENSE); preserve upstream
terms for third-party material.
