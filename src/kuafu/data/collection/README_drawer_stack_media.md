# Capture drawer-stack media

Activate the Isaac environment and run from the Kuafu checkout. Set `CHECKPOINT`,
`PRIOR_CHECKPOINT`, and `WEIGHTS_PARAMS_JSON` to your matching playback artifacts,
or use the archived defaults in `capture_drawer_stack_best_ema_media.sh`.

```bash
./src/kuafu/data/collection/capture_drawer_stack_best_ema_media.sh
DEBUG_FRAMES=1 ./src/kuafu/data/collection/capture_drawer_stack_best_ema_media.sh
```

The default task is `FR3-FinRay-Table-Drawer-Stack-fff`. Seed `20260516` keeps the
cabinet and strawberry placement consistent across captures. Camera bank `best`
uses the close negative-Y view `between_pillars_close_action`.

For a faster review render:

```bash
MAX_STEPS=700 WIDTH=960 HEIGHT=540 FPS=10 VIDEO_EVERY=5 \
  ./src/kuafu/data/collection/capture_drawer_stack_best_ema_media.sh
```

Set `CAMERA_BANK` to compare `between_pillars_close`, `between_pillars_refined`,
`between_pillars`, `drawer_close`, `sweep_a`, `sweep_b`, or `hero`. Use
`CAPTURE_BACKEND=sensor` to inspect environment cameras.

Read results under `$KUAFU_ARTIFACT_ROOT/media/drawer_stack_best_ema/`:

- `screenshots/<view>/` and `videos/<view>.mp4`: captured media.
- `camera_manifest.csv`: camera poses and focal lengths.
- `run_config.json`: checkpoints, seed, and capture settings.
