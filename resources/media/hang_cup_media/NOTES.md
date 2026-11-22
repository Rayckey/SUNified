# Hang-Cup Media Recipe

This directory records a show recipe for `FR3-FinRay-Table-Hang-Cup-fff` using the current success-epoch EMA checkpoint from `collect_and_upload_loop_hang_cup_best_ema.sh`.

Selected recipe:

- Seed: `20260516`
- Camera bank: `best`
- View: `front_close_hang_cup_success`
- Camera spec: `CameraSpec("front_close_hang_cup_success", (0.12, -0.78, 0.56), (0.00, 0.06, 0.05), 24.0)`
- Screenshot steps: `final`
- Policy checkpoint: `Residuals/FR3-FinRay-Table-Hang-Cup-fff/2026-05-22_17-09-20/rl_runs/2026-05-22_17-13-28_Run-3/nn/last_fr3-finray-hang-cup_ep_2750_rew_15378.236.pth`
- Prior checkpoint: `Priors/herm/hang_cup/hang_cup_chunking_10_state_fff/best.pth`
- Weights params JSON: `Residuals/FR3-FinRay-Table-Hang-Cup-fff/2026-05-22_17-07-48/best.json`

Recreate clean and debug-frame screenshots with:

```bash
./resources/media/hang_cup_media/recreate_hang_cup_media.sh both
```

The recipe keeps `NO_VIDEO=1` by default for quick still captures. Set `NO_VIDEO=0` when you want matching MP4s, or set `CAMERA_BANK=front_hang_cup_close` / `side_hang_cup` to sweep the other Hang-Cup camera views.
