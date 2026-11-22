# SUN skill set

`schema.py` defines the operator grammar. `constraints/` contains NumPy,
CasADi and Torch implementations. The receiving controller or learner assigns
numeric tolerances.

## Install

From the repository root:

```bash
python -m pip install -r src/kuafu/skills/requirements.txt
```

See [runtime setup](../../../docs/INSTALL.md) for the CPU and Isaac environments.

## Input and output

| Direction | Contract | Concrete example |
|---|---|---|
| Input | Named operators, reference frames and geometry | [input.json](examples/input.json) |
| Output | Receiver-normalized constraints and shared grammar | [output.json](examples/output.json) |

## Use

```bash
python -m kuafu.skills --operator directional_move
```

Follow the [module guide](../../../docs/ASSET_SEMANTICS.md) for a complete invocation.
