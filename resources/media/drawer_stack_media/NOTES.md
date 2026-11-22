# Drawer Stack Show Media

This folder records the exact recipe used to recreate the final drawer-stack videos that show the robot gripper, strawberry, and drawer without the table pillars blocking the view.

## Final Camera

Selected view: `between_pillars_close_action`

Camera spec in `src/kuafu/data/collection/capture_drawer_stack_best_ema_media.py`:

```python
CameraSpec("between_pillars_close_action", (0.00, -0.62, 0.56), (-0.10, 0.22, -0.03), 19.0)
```

Resolved pose from the final clean run:

```text
eye=(0.4822010397911072, -1.0628360509872437, 0.7275000214576721)
target=(0.38220104575157166, -0.22283604741096497, 0.13750001788139343)
focal_length=19.0
```

The camera is still on the negative-Y side looking toward positive-Y, but moved closer to the task so the table pillars fall out of frame.

## Final Outputs

Clean video:

```text
outputs/media/drawer_stack_best_ema/2026-05-16_03-43-25_best_clean_seed20260516/videos/00_between_pillars_close_action.mp4
```

Debug-frame video:

```text
outputs/media/drawer_stack_best_ema/2026-05-16_03-45-13_best_debugframes_seed20260516/videos/00_between_pillars_close_action.mp4
```

Both used seed `20260516`, deterministic policy playback, `1280x720 @ 30fps`, and reached success at policy step `291`.

## Recreate

From the repo root:

```bash
./resources/media/drawer_stack_media/recreate_drawer_stack_media.sh clean
./resources/media/drawer_stack_media/recreate_drawer_stack_media.sh debug
```

Or render both variants:

```bash
./resources/media/drawer_stack_media/recreate_drawer_stack_media.sh both
```

The script writes new timestamped outputs under:

```text
outputs/media/drawer_stack_best_ema/
```

## Files

- `drawer_stack_media_recipe.json`: machine-readable recipe with task, seed, checkpoints, camera, and final output paths.
- `selected_camera_manifest.csv`: exact resolved viewport camera pose from the final run.
- `recreate_drawer_stack_media.sh`: runnable wrapper for clean/debug recreation.

## Candidate Sweeps Kept For Review

```text
outputs/media/drawer_stack_best_ema/2026-05-16_02-50-09_between_pillars_clean_seed20260516
outputs/media/drawer_stack_best_ema/2026-05-16_03-34-14_between_pillars_refined_clean_seed20260516
outputs/media/drawer_stack_best_ema/2026-05-16_03-40-51_between_pillars_close_clean_seed20260516
```
