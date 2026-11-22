# Pen-Cup Media Recipe

This directory records the final show recipe for `FR3-FinRay-Table-Pen-Cup-ff` using the best true-success checkpoint from `collect_and_upload_loop_pen_cup_best_true_success.sh`.

Selected recipe:

- Seed: `5`
- Camera bank: `best`
- View: `front_close_pen_cup_success`
- Camera spec: `CameraSpec("front_close_pen_cup_success", (0.08, -0.92, 0.58), (0.00, 0.10, 0.05), 18.0)`
- Screenshot steps: `0,80,120,160,200,220,235,240,245,246,247`
- Final verified clean screenshot: `outputs/media/pen_cup_best_true_success/2026-05-16_15-30-30_best_clean_seed5/screenshots/front_close_pen_cup_success/final.png`
- Final verified debug screenshot: `outputs/media/pen_cup_best_true_success/2026-05-16_15-31-46_best_debugframes_seed5/screenshots/front_close_pen_cup_success/final.png`

Recreate with:

```bash
./resources/media/pen_cup_media/recreate_pen_cup_media.sh both
```

The final clean and debug runs both printed `Successfully finished #: 1` and ended at policy step 246. The recipe keeps `NO_VIDEO=1` by default because the requested show artifact is a still screenshot and the viewport stills are already verified.
