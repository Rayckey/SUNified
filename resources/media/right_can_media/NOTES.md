# Right-Can Media Recipe

This directory records the final show recipe for `FR3-FinRay-Table-Can-ff` using the residual checkpoint from `collect_and_upload_loop_can.sh`.

Selected recipe:

- Seed: `20260516`
- Weights/params: `resources/experiment_definitions/fr3_finray/residual_rl/parameter_search/fr3_finray_learned_can/best.json`
- Camera bank: `best`
- View: `front_close_can_tight`
- Camera spec: `CameraSpec("front_close_can_tight", (0.08, -0.62, 0.46), (0.00, 0.04, 0.05), 30.0)`
- Screenshot steps: `0,100,200,300,340,360,380,390,391,394,395,396`
- Final verified clean screenshot: `outputs/media/right_can_best/2026-05-16_16-04-08_best_clean_seed20260516/screenshots/front_close_can_tight/final.png`
- Final verified debug screenshot: `outputs/media/right_can_best/2026-05-16_16-05-44_best_debugframes_seed20260516/screenshots/front_close_can_tight/final.png`

Recreate with:

```bash
./resources/media/right_can_media/recreate_right_can_media.sh both
```

`collect_and_upload_loop_can.sh` originally pointed at `Residuals/FR3-FinRay-Table-Can-ff/2026-05-15_17-44-46/best.json`, which is missing in this checkout and causes the shared collection loop to fail before launch. The collection loop and media wrapper now use `resources/experiment_definitions/fr3_finray/residual_rl/parameter_search/fr3_finray_learned_can/best.json`, which reproduced true success with the selected checkpoint. The final clean and debug runs both printed `Successfully finished #: 1` and ended at policy step 395.
