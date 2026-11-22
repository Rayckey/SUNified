# Stack-2 Show Media

This folder records the exact recipe used to recreate the final Stack-2 screenshots that show the robot gripper placing one cube on the other from a close negative-Y view.

## Final Camera

Selected view: `front_close_stack_action`

Camera spec in `src/kuafu/data/collection/capture_stack2_best_ema_media.py`:

```python
CameraSpec("front_close_stack_action", (0.08, -0.70, 0.52), (0.00, 0.08, 0.04), 22.0)
```

Resolved pose from the final clean run:

```text
eye=(0.49925506114959717, -1.0118452310562134, 0.5799999833106995)
target=(0.4192550778388977, -0.23184524476528168, 0.09999999403953552)
focal_length=22.0
```

The camera is close enough that both cubes and the gripper are readable, while still leaving table context around the final stack.

## Final Outputs

Clean final screenshot:

```text
outputs/media/stack2_best_ema/2026-05-16_14-43-09_best_clean_seed20260516/screenshots/front_close_stack_action/final.png
```

Debug-frame final screenshot:

```text
outputs/media/stack2_best_ema/2026-05-16_14-44-21_best_debugframes_seed20260516/screenshots/front_close_stack_action/final.png
```

Both used seed `20260516`, deterministic policy playback, viewport capture, `1280x720`, and reached success at policy step `226`.

## Recreate

From the repo root:

```bash
./resources/media/stack2_media/recreate_stack2_media.sh clean
./resources/media/stack2_media/recreate_stack2_media.sh debug
```

Or render both variants:

```bash
./resources/media/stack2_media/recreate_stack2_media.sh both
```

The script writes new timestamped outputs under:

```text
outputs/media/stack2_best_ema/
```

## Video Note

Viewport MP4 capture perturbed this Stack-2 rollout during testing: every-step and every-5-step video captures timed out at policy step `598` around stage `2`. The recreate wrapper therefore defaults to `NO_VIDEO=1` and captures stills around the known success window.

## Files

- `stack2_media_recipe.json`: machine-readable recipe with task, seed, checkpoints, camera, settings, and final output paths.
- `selected_camera_manifest.csv`: exact resolved viewport camera pose from the final clean run.
- `recreate_stack2_media.sh`: runnable wrapper for clean/debug screenshot recreation.

## Candidate Sweeps Kept For Review

```text
outputs/media/stack2_best_ema/2026-05-16_14-30-48_front_stack_close_clean_seed20260516
outputs/media/stack2_best_ema/2026-05-16_14-33-17_front_stack_close_clean_seed20260516
outputs/media/stack2_best_ema/2026-05-16_14-43-09_best_clean_seed20260516
outputs/media/stack2_best_ema/2026-05-16_14-44-21_best_debugframes_seed20260516
```
