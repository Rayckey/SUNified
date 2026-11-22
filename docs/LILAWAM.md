# Train LiLa-WAM on Kinova recordings

The adapter uses the [official LiLa-WAM implementation](https://github.com/teee000/LiLa-WAM)
at commit `339860f65d5cde9e229c037b672889bc4f70497c`. It learns action chunks
and future DINOv3 features jointly. Its feature decoder does not produce RGB videos.
The simulator and demonstration controllers are unchanged.

## Environment and encoder

Use `.venv-lilawam/bin/python`. This environment inherits the local PyTorch
2.7.0/CUDA 12.8 installation and installs `transformers==5.0.0rc0` and
`accelerate==1.15.0` separately from the Isaac environment. The run stores an
environment report and a source snapshot.

Get account approval for
[DINOv3 ViT-L/16](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m),
then download using the supplied local token file or an authenticated Hugging Face session:

```bash
PYTHONPATH=src .venv-lilawam/bin/python -m kuafu.dynamics.lilawam download-encoder \
  --output /absolute/external/artifact/path/pretrained/dinov3 \
  --token-file /absolute/path/to/huggingface_token.txt
```

The token is read in memory, never copied into run artifacts. Encoder revision
`ea8dc2863c51be0a264bab82070e3e8836b02d51` is pinned. Training requires these
pretrained weights; the explicitly named `smoke` command uses random encoder
weights only to test architecture, gradients, inference, and memory use.

## Data interface

Prepare the completed RGB index using the existing episode split:

```bash
PYTHONPATH=src .venv-lilawam/bin/python -m kuafu.dynamics.lilawam prepare \
  --rgb-index /absolute/path/to/rgb_datasets.json \
  --split-run /absolute/path/to/model/run.json \
  --output /absolute/external/artifact/path/data --horizon 16
```

The nine audited tasks are drawer, drawer-stack, cabinet-door, stack-2, stack-3,
hang-cup, can, E-stop open, and E-stop close. Inputs are the two RGB cameras,
nine joint positions, and a task vector fitted exclusively on training episodes.
Object poses and SUN Program stages are excluded.

Both native 128×72 views are resized to 256×144 and packed left-to-right in a
512×144 image, preserving aspect ratio and applying ImageNet normalization.
Resizing adds no visual detail. Joint positions come from `full_state[-29:-20]`;
the adapter rejects unaudited state layouts.

At anchor `t`, inputs are `RGB[t]` and `joint_positions[t]`. Targets are
`relative_joint[t:t+16]` and `RGB[t+16]`. Frames are recorded before the action at
30 Hz, so the future target is about 0.53 seconds ahead. Anchors start at 1 to
exclude reset frames, and require a full chunk and a real future frame within
the same episode. No padding or cross-episode targets are used.

Logged arm actions are scaled incremental controller commands, not radian
displacements; gripper commands are binary. Cabinet commands can have large
magnitudes. The adapter applies invertible `sign(a) * log1p(abs(a))` before
training-only min/max normalization. Apply `decode_actions` after the model's
denormalization to recover original controller units. Offline action metrics
are explicitly labeled in symlog units. No robot deployment is configured.

## Run and inspect

Each run contains `stage1.yaml`, `stage2.yaml`, `data/manifest.json`, and a
`launch.sh` using its frozen adapter snapshot. Launchers supervise encoder
download, task-vector preparation, and both training stages. While access is
pending, they poll every five minutes without reserving a GPU. They also wait
for the configured GPU to be available before each GPU job.

```bash
bash /absolute/path/to/run/launch.sh
```

Stage 1 uses 12 balanced sampling passes at LR `2e-4`; stage 2 initializes
from the best stage-1 weights and uses 4 passes at LR `4e-5`, with fresh optimizer
and scheduler state. The batch size is 128. These starting budgets follow the
upstream two-stage recommendation; they are not evidence of convergence on
Kinova. An interruption resumes from `last.pt`, restoring optimizer, scheduler,
gradient scaler, random state, and sampling position. Concurrent launches of
the same run are rejected.

Inspect `pipeline_status.json` and `stage1/progress.json` or
`stage2/progress.json`. A waiting pipeline is not training. Successful optimizer
updates are recorded separately. Each stage writes `best.pt`, `last.pt`,
`metrics.jsonl`, and `evaluation.json`.

Validation selects checkpoints using the equal-task average of action RMSE in
symlog units plus 0.5 times future-feature cosine distance. Action inference sees
only current inputs; future images are targets only. Reports include previous-action
and current-feature persistence baselines, per-task scores, and held-out test
results. Feature metrics and action prediction alone do not establish closed-loop
task success or RGB reconstruction quality.
