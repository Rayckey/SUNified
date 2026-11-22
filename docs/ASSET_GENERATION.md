# Video-to-simulation assets

Your agent uses the [asset tools](AGENT_INTERFACE.md#tools) to turn videos and
an object description into geometry, articulated USD, interaction frames, and
previews. This guide covers the corresponding CLI commands and built-in
reconstruction loop.

## Set up

Complete the [CPU installation](INSTALL.md#assets-and-agents-cpu), then configure
the [simulation interpreter](INSTALL.md#simulation-and-learning) for physics checks:

```bash
export KUAFU_ARTIFACT_ROOT="$(cd .. && pwd)/KuafuArtifacts"
export KUAFU_ISAAC_PYTHON=/path/to/isaac/environment/bin/python
```

Run commands from the checkout. Relative output/workspace paths resolve under
`KUAFU_ARTIFACT_ROOT`; input paths resolve normally. Give builds new output
folders. Reconstruction resumes its prepared workspace and matching cached responses.

## Reconstruct an object

```bash
kuafu assets prepare /path/to/object.mov \
  --description "The hollow green holder" --frames 16 --out assets/holder
kuafu assets reconstruct assets/holder --iterations 3
```

Reconstruction uses `OPENAI_API_KEY` and up to two requests per iteration.
Select a model with `--model` or `OPENAI_MODEL` (default `gpt-6-astra`).
`run VIDEO --description TEXT --out PATH` combines both steps.

Supply a measured dimension with `--size-m` and `--size-axis`, and mass with
`--mass-kg`. Add `--estimated-mass` for an assumed mass. Other dimensions are
inferred; the default mass is a 0.2 kg placeholder.

Review `frames.jpg`, `final/preview.png`, `final/collisions.png`, and
`final/viewer.html`. Edit `final/scene.json` and rebuild:

```bash
kuafu assets build-scene \
  "$KUAFU_ARTIFACT_ROOT/assets/holder/final/scene.json" \
  --out assets/holder-edited
```

## Reconstruct articulated objects

Describe how the moving parts fit together, and supply additional views with
`--reference-video`:

```bash
kuafu assets run /path/to/object.mov \
  --reference-video /path/to/another-view.mov \
  --description "Describe the object, its moving parts, and how they connect" \
  --motion articulated --frames 16 --iterations 2 --out assets/my-object

kuafu assets validate \
  --asset "$KUAFU_ARTIFACT_ROOT/assets/my-object/final/asset_dynamic.usda"
```

`--reference-video` accepts three additional clips; `--frames` is the total.
Choose `--motion articulated` for a mechanism, `rigid` for one body, or `auto`
for inference. Supply measured dimensions and mass with the options above.

Review the closed and open previews, joint limits, interaction frames, and
motion axes. Builds sample collider clearance at rest, four positions per joint,
and all-open, with a 0.5 mm penetration tolerance. Physics validation checks
independent motion, travel stops, body following, base stability, and one-newton
pulls. Measure friction for your object, then verify robot interaction in the
[task scene](ASSET_SEMANTICS.md).

## Add photo textures

Create `photo_recipe.json` in the prepared workspace with the scene path, frame
paths, object masks, and 2D/3D landmark correspondences. Review registration,
then track and bake:

```bash
kuafu assets track-photo-views assets/holder \
  --recipe "$KUAFU_ARTIFACT_ROOT/assets/holder/photo_recipe.json" \
  --out assets/holder/multiview --max-views 12

kuafu assets bake-video-textures assets/holder \
  --recipe "$KUAFU_ARTIFACT_ROOT/assets/holder/multiview/recipe.json" \
  --out assets/holder-textured --resolution 768
```

Use at least three registered views for robust fusion, or `--fusion best` for
one or two. Track each clip in its own workspace. Inspect coverage and alignment,
then validate the textured export. For articulated GLB imports, segment the mesh
and use `build-textured --visual-links mapping.json` to assign every mesh to a
link and cover every body.

For optional Marigold de-lighting:

```bash
python -m pip install -e '.[assets,delight]'
kuafu assets delight-frames assets/holder \
  --recipe "$KUAFU_ARTIFACT_ROOT/assets/holder/multiview/recipe.json" \
  --out assets/holder-lighting --device cuda:0
```

Pass the resulting `manifest.json` to `bake-video-textures --delighting`.
Set `--cache-dir` to choose the model cache, or `--local-files-only` to use cached
weights. With `ffmpeg` installed, render a turntable:

```bash
kuafu assets turntable \
  "$KUAFU_ARTIFACT_ROOT/assets/holder-textured/asset/asset.glb" \
  --out assets/holder-turntable
```

## Validate and load

Validate your exported asset:

```bash
kuafu assets validate \
  --asset "$KUAFU_ARTIFACT_ROOT/assets/holder/final/asset_dynamic.usda"
```

For a hollow object, set `--probe-radius` and `--cavity-floor-z` to its geometry.

The rigid check runs 240 steps at 120 Hz, measures finite state and ground
contact, and optionally probes the cavity floor. `physics_smoke.json` records
results and the tested USD hashes. Revalidate each changed export.

After Isaac starts, load the appropriate configuration from `isaaclab_snippet.py`:

| Export | Configuration | Use |
|---|---|---|
| `asset_dynamic.usda` | `DYNAMIC_OBJECT` | Rigid object |
| `asset_dynamic.usda` | `ARTICULATED_OBJECT` | Moving assembly |
| `asset_static.usda` | `STATIC_OBJECT` | Frozen scenery |

Set the prim path and initial pose, and keep the asset directory together for
relative layer/texture references. USD uses meters, Z-up, and scale `(1, 1, 1)`;
GLB uses Y-up. Use `attach_semantics` to register language descriptions and moving
interaction frames; see [scene integration](ASSET_SEMANTICS.md).

## Replay and package reviewed assets

Use `build-reviewed MANIFEST` for a batch of your prepared workspaces, scenes,
physics settings, and cavity probes. Run `package-reviewed MANIFEST` after the
matching physics reports pass. To reuse our saved reconstructions, choose an
[optional asset example](../examples/01_video_to_rollout/README.md).

For SAM 3D, run `scripts/assets/setup_sam3d.sh`, then `sam3d-doctor`,
`sam3d-fetch`, `sam3d-prepare`, and `sam3d-run` with approved weights and reviewed
masks. Set `SAM3D_ROOT` for the runtime location and `SAM3D_CONDA` for its Conda
executable. Pair the resulting GLB with reviewed colliders using `build-textured`.

## Test

```bash
python -m pytest -q tests/asset_generation
kuafu assets demo --out assets/scripted-demo
```

The demo exercises reconstruction and review with synthetic video and scripted
proposals. Kuafu integrates the [standalone asset project](https://github.com/Rayckey/kuafu_visual).
