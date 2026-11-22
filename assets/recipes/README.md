# Asset recipes

These optional recipes reuse our saved reconstructions. For your own inputs,
start with [your agent](../../docs/USAGE.md) or [asset generation](../../docs/ASSET_GENERATION.md).
Build a selected recipe, then validate its physics:

```bash
kuafu assets build-scene assets/recipes/0028_0029.articulated.scene.json \
  --mass-kg 2.5 --estimated-mass --out assets/cabinet
kuafu assets validate \
  --asset "$KUAFU_ARTIFACT_ROOT/assets/cabinet/asset_dynamic.usda"
```

| Recipe | Geometry |
|---|---|
| `0024.scene.json` | Rigid emergency-stop box |
| `0026.scene.json` | Hollow sage holder, approximately 80 × 80 × 112 mm |
| `0028.scene.json` | Rigid bamboo organizer with fixed drawer fronts |
| `0029.scene.json` | Rigid bamboo tray with grip |
| `0028_0029.articulated.scene.json` | Cabinet with three sliding drawers, moving handles, and six interaction frames |

Dimensions, travel, and mass fractions are video-derived estimates. Supply
measured scale and mass when available. Use the combined recipe for working
drawers; its handle/center frames include joint-derived opening and closing axes.

The four rigid inputs came from the [standalone asset project](https://github.com/Rayckey/kuafu_visual)
revision `21aec80b22ae79d4e7d32e916fdee1bc9ef3298c`; the combined recipe came from
the later reconstruction of videos 28/29. See [asset generation](../../docs/ASSET_GENERATION.md)
and [scene integration](../../docs/ASSET_SEMANTICS.md).
