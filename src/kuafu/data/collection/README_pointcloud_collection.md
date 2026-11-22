# Collect Kinova Stack-2 point clouds

Activate the Isaac environment and run from the Kuafu checkout. Place the
checkpoint, prior, and parameters below under `KUAFU_ARTIFACT_ROOT`, or supply
your own matching paths. Start with five local demonstrations:

```bash
python src/kuafu/data/collection/launch_collect_with_checkpoint.py \
  --gpus 0 \
  --num_demos_per_worker 5 \
  --num_envs_per_worker 16 \
  --stagger_sec 0 \
  --task Kinova-Table-Gen3-Stack-2-ff3 \
  --checkpoint Residuals/Kinova-Table-Gen3-Stack-2-ff3/2026-06-22_09-01-21/checkpoints/best_global.pth \
  --prior_checkpoint Priors/kinova_gen3/stack2/stack2_2026-06-21_19-28-09_chunking_10_ff3/best.pth \
  --prior_mode residual_rl \
  --weights_params_json Residuals/Kinova-Table-Gen3-Stack-2-ff3/2026-06-22_09-01-21/best.json \
  --stride 1 \
  --skip_dense_reward_computation \
  --saver_chunk_size 16 \
  --point_cloud_num_points 16384 \
  --point_cloud_workspace_min 0.10 -0.60 -0.10 \
  --point_cloud_workspace_max 0.65 0.60 0.50 \
  --point_cloud_exclude_cylinder_center_xy 0 0 \
  --point_cloud_exclude_cylinder_radius 0.2 \
  --disable_point_cloud_table_plane_filter \
  --hide_table_visuals \
  --camera_warmup_frames_after_reset 2
```

The crop uses meters. The 0.2 m exclusion cylinder removes the robot base across
all heights. Filtering runs before downsampling to 16384 points.

Review the point clouds and configure your upload destination, then run the loop:

```bash
bash src/kuafu/data/collection/collect_and_upload_loop_kinova_stack2_ff3.sh
```

Override the batch size, GPUs, and crop as needed:

```bash
NUM_ITERS=1 DEMOS_PER_CHUNK=20 GPUS="0" \
POINT_CLOUD_WORKSPACE_MIN="0.10 -0.60 -0.10" \
POINT_CLOUD_WORKSPACE_MAX="0.65 0.60 0.50" \
POINT_CLOUD_EXCLUDE_CYLINDER_CENTER="0 0" \
POINT_CLOUD_EXCLUDE_CYLINDER_RADIUS="0.2" \
bash src/kuafu/data/collection/collect_and_upload_loop_kinova_stack2_ff3.sh
```
