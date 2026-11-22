# Isaac simulation adapters

`formulation/`, `learned/` and `tasks/` define simulation scenes and physical
success checks. Object and robot models live in `assets/objects/` and
`assets/robots/`. Install Isaac Sim and Isaac Lab before using these adapters.

## Install

From the repository root:

```bash
python -m pip install -r src/kuafu/isaac/requirements.txt
```

See [runtime setup](../../../docs/INSTALL.md) for the CPU and Isaac environments.

## Input and output

| Direction | Contract | Concrete example |
|---|---|---|
| Input | Task configuration, robot/object assets and interaction frames | [input.json](examples/input.json) |
| Output | Scene observations and independent physical success measurements | [output.json](examples/output.json) |

## Use

```bash
kuafu tasks list
kuafu tasks show stack2 --robot kinova_gen3
```

Choose a task from the [catalog](../../../assets/tasks/README.md).
The [cabinet replay](../../../examples/02_program_to_rollout/README.md) provides
one complete invocation; each task entry links its own configuration and programs.
