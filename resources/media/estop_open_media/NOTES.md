# EStop-Open Media Recipe

This directory records the final show recipe for `FR3-FinRay-Table-EStop-Open-ff` using the best EMA/success checkpoint from `collect_and_upload_loop_estop_open_best_ema.sh`.

Selected recipe:

- Seed: `20260516`
- Weights/params: `Residuals/FR3-FinRay-Table-EStop-Open-ff/2026-04-30_12-01-28/best.json`
- Camera bank: `best`
- View: `front_close_estop_open_action`
- Camera spec: `CameraSpec("front_close_estop_open_action", (0.10, -0.72, 0.50), (0.00, 0.05, 0.05), 28.0)`
- Screenshot steps: `137`
- Viewport settle frames: `2`
- Final verified clean screenshot: `outputs/media/estop_open_best_ema/2026-05-16_17-01-10_best_clean_seed20260516/screenshots/front_close_estop_open_action/final.png`
- Final verified debug screenshot: `outputs/media/estop_open_best_ema/2026-05-16_17-02-19_best_debugframes_seed20260516/screenshots/front_close_estop_open_action/final.png`

Recreate with:

```bash
./resources/media/estop_open_media/recreate_estop_open_media.sh both
```

The EStop-Open learned-program program predates the current directional-rotate/axis-align validator. The capture script applies a playback-local compatibility patch for this task only, then loads the same checkpoint/prior/weights from the collection wrapper. Clean and debug final runs both printed `Env done at policy step 137`.

Repeated viewport captures changed this sensitive twist rollout, so the final recipe captures a single close viewport still at step `137` with `VIEWPORT_SETTLE_FRAMES=2`. The sensor backend also reaches done, but its RGB framing is not suitable for the show screenshot here.
