# Asset reconstruction

Use `kuafu_prepare_video` and `kuafu_build_asset` through your agent, or the
standalone asset CLI. Export articulated meshes, collision geometry, interaction
frames and language labels from a scene proposal.

## Install

From the repository root:

```bash
python -m pip install -r src/kuafu/asset_generation/requirements.txt
```

See [runtime setup](../../../docs/INSTALL.md) for the CPU and Isaac environments.

## Input and output

| Direction | Contract | Concrete example |
|---|---|---|
| Input | Video frames, description or scene JSON | [input.json](examples/input.json) |
| Output | Articulated USD, preview images and interaction semantics | [output.json](examples/output.json) |

## Use

```bash
kuafu assets --help
```

Follow the [module guide](../../../docs/ASSET_GENERATION.md) for a complete invocation.
