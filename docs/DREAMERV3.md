# Train a DreamerV3 world model

Train the recurrent world-model component of
[DreamerV3](https://github.com/danijar/dreamerv3) on existing Kuafu demonstrations.
The adapter imports upstream encoder, categorical RSSM, decoder, reward and
continuation heads, KL losses, and optimizer. It trains no actor or critic and
does not launch Isaac Sim.

## Install

Use a separate Python 3.11 environment. Keep the upstream source outside Kuafu:

```bash
git clone https://github.com/danijar/dreamerv3 /absolute/path/to/dreamerv3
git -C /absolute/path/to/dreamerv3 checkout e01491fad6434b2245a3b8ca201dd7faedcc458c
python3.11 -m venv /absolute/path/to/dreamer-env
source /absolute/path/to/dreamer-env/bin/activate
python -m pip install -e '.[dreamerv3]'
python -m pip install 'jax[cuda12]==0.4.33' 'nvidia-cuda-nvcc-cu12<=12.2'
export KUAFU_DREAMERV3_ROOT=/absolute/path/to/dreamerv3
export KUAFU_ARTIFACT_ROOT=/absolute/path/to/KuafuArtifacts
```

The CUDA installation is optional for CPU training. Keep the upstream checkout
clean; its revision and runtime versions are recorded in every run. Upstream
DreamerV3 retains its MIT license in that checkout.

If a virtual environment inherits CUDA packages from another environment,
JAX's relative library lookup may not find them. Set `LD_LIBRARY_PATH` for the
training process to that environment's `nvidia/cudnn/lib`, `nvidia/cublas/lib`,
and `nvidia/cuda_runtime/lib` directories. A standalone environment with locally
installed CUDA wheels avoids this lookup issue.

## Train on Kinova pen-in-cup

```bash
kuafu train-dreamerv3 \
  --dataset /absolute/path/to/kinova_pen_cup.hdf5 \
  --output world_models/kinova_pen_cup/dreamerv3_seed42 \
  --task Kinova-Table-Gen3-Pen-Cup \
  --device cuda:0 --model-size size1m \
  --steps 25000 --batch-size 16 --length 64 --burnin 8 \
  --context 16 --horizon 20 --seed 42
```

Use a new output directory for each run. `python -m kuafu.dynamics.dreamerv3`
exposes the same options. `--device cpu` is supported. CUDA device selection is
process-local; `cuda:3` selects physical GPU 3 before JAX initializes.

Continue a completed run with `--resume-from /absolute/path/to/previous/run`,
a new `--output`, and a larger `--steps` value (the total target update count).
Keep the dataset, model, batch, sequence, seed, and evaluation settings unchanged.
The loader restores the optimizer and sampler progression, and retains the best
validation checkpoint from the previous run.

The default `size1m` is upstream's small proprioceptive preset; this vector-only
world model has fewer parameters than the full agent. Use `size12m`, `size25m`,
or `size50m` to increase capacity. Computation uses float32 for compatibility
with Quadro RTX 8000 hardware.

## Train one model across Kinova tasks

Pass the completed collection's `datasets.json` index to train one shared model:

```bash
kuafu train-dreamerv3 \
  --dataset-index /absolute/path/to/collection/datasets.json \
  --output world_models/kinova_shared/dreamerv3_seed42 \
  --task Kinova-Shared --device cuda:0 --model-size size12m \
  --steps 25000 --batch-size 18 --length 64 --burnin 8 \
  --context 8 --horizon 16 --eval-batches 4 --report-every 500 --seed 42
```

The audited layouts cover drawer, drawer-stack, cabinet-door, stack-2, stack-3,
hang-cup, can, and E-stop open/close. The shared state has 46 robot coordinates
and three nine-coordinate object poses. Robot fields align across tasks; object
slots retain each task's recorded order. Each task's states are standardized
using its training mean and standard deviation (floored at 1e-3). Forecasts
reverse this scaling and the decoder's signed-log transform, preserving dataset
coordinates, units, and rotation-6D conventions. Task identity conditions both
observations and recurrent transitions. Missing object coordinates are masked from the encoder
and reconstruction loss. Short episodes are padded, with all padded timestep
losses excluded and each sequence weighted equally.

With nine tasks, batch size 18 samples two windows per task. Demonstrations are
split 80/10/10 within every task. One joint-command transform is fitted using
only training episodes. Checkpoint selection averages validation forecast error
equally across tasks; `evaluation.json` includes held-out errors and persistence
baselines for every task. Context 8 and horizon 16 fit the short E-stop-close
episodes without evaluation padding. Shared runs also support `--resume-from`.

`load_forecaster` recognizes shared checkpoints. Supply `task="drawer"` (or
another task key from `run.json`) to the returned callable. Inputs and outputs
use that task's native state order with its SUN stage column excluded; the
loader applies and reverses the shared mapping automatically.

## Collect additional state demonstrations

In the Isaac environment, replay a compatible saved expert with `--state_only`:

```bash
python -m kuafu.data.collection.collect_with_checkpoint \
  --task Kinova-Table-Gen3-Drawer-fff \
  --checkpoint /absolute/path/to/residual/checkpoints/best_global.pth \
  --checkpoint_params_dir /absolute/path/to/residual/params \
  --prior_mode residual_rl --prior_checkpoint /absolute/path/to/prior/best.pth \
  --weights_params_json /absolute/path/to/residual/inputs/initial_weights_params.json \
  --state_only --headless --device cuda:0 --num_envs 16 --num_demos 100 \
  --seed 42 --output_dir /absolute/path/to/new_collection
```

This writes `workers/worker_00/datasets/state_demos.hdf5` with exactly 100
successful episodes and no camera rendering. It records applied joint commands,
full observations, explicit next observations, and rewards at every control step.
Each terminal transition is omitted because automatic reset replaces its next
observation; the episode's `success` attribute retains the task outcome. A new
output file is required. Different tasks have different observation layouts;
use `--dataset-index` for shared training across the audited Kinova layouts.
For shorter tasks, reduce `--length`, `--context`, and `--horizon` to fit their
episode lengths. For example, 31–54-step E-stop-close demos support `--length 24
--burnin 4 --context 8 --horizon 16`.

## Data and evaluation

Use `data/demo_*/obs/full_state`, `next_obs/full_state`, and
`actions/relative_joint`. The defaults exclude the trailing SUN stage column and
first reset transition, yielding 58 state values and 9 action values for Kinova
pen-in-cup. See the [state layout](DYNAMICS.md#kinova-pen-in-cup). Coordinates,
meters, rotation-6D values, and joint conventions remain those of the dataset;
any quaternion conversion must use **wxyz**.

The loader checks temporal alignment, splits whole demonstrations 80/10/10, and
never samples across episode boundaries. Terminal rows are omitted because their
next observations may already be reset. Each recurrent window begins with a
reset; the first eight steps are context for the loss. `prev_action[t]` and
`reward[t]` belong to the transition **leading into** `state[t]`.

Commands are transformed using a signed logarithm and a per-coordinate scale
fitted only on training demonstrations, then bounded to `[-1, 1]`. This preserves
large legacy commands that upstream's internal action clipping would otherwise
collapse. The scale is saved and applied automatically by `load_forecaster`.
Commands outside the training range are clipped.

Rewards are read from the recording. Missing rewards are masked. Continuation
targets are one after filtering; the current Kinova dataset has no positive
termination flags. The continuation head therefore does **not** establish a
task-success or failure detector.

Validation assimilates 16 observed states, then predicts 20 future states using
only commands and sampled latent priors. Four latent trajectories are averaged
for error measurement. Checkpoint selection uses the final-horizon validation
RMSE scaled by training-state standard deviations. The final test report uses
separate demonstrations. Persistence baselines and per-coordinate errors in
original units are included; inspect these before choosing a planning horizon.

| Artifact | Contents |
|---|---|
| `best.npz` | Array-only world-model parameters selected by validation forecasts |
| `best.json` | Selected update and validation results |
| `last.npz` | Final parameters and optimizer state |
| `run.json` | Dataset hash, splits, action transform, source revision, configuration |
| `metrics.jsonl`, `progress.json` | Learning metrics and run progress |
| `evaluation.json` | Final held-out forecasts and persistence baseline |

## Predict

```python
from kuafu.dynamics.dreamerv3 import load_forecaster

forecast, run = load_forecaster('/absolute/path/to/run/best.npz', device='cuda:0')
# Float32 arrays in original dataset units:
# context_states: [B, 16, 58]
# previous_actions: [B, 16, 9], each action leading into its corresponding state
# future_actions: [B, 20, 9], starting from the last context state
samples = forecast(context_states, previous_actions, future_actions, particles=4, seed=42)
mean_prediction = samples.mean(axis=1)  # [B, 20, 58]
```

The first previous-action value is ignored by the recurrent reset. No future
observations enter forecasting. Samples represent latent model uncertainty, not
independent ensemble members. Demonstration coverage and missing simulator state
limit extrapolation; prediction metrics alone do not establish closed-loop
control performance.

## RGB world models

Use `--rgb_transitions` with the checkpoint collector to capture synchronized
128×72 RGB from both Kinova cameras, alongside applied joint commands. It saves
successful episodes to `workers/worker_00/datasets/rgb_demos.hdf5`. Images are
captured before physics, and terminal/autoreset transitions are omitted. The
cabinet expert can use `RGBTransitionRecorder` through the same recorder hooks.

Create an index with `format: "kuafu-rgb-datasets-v1"`, `phase: "complete"`, and
`tasks: {task_slug: {path, sha256, demonstrations}}`. Each path identifies a
completed RGB HDF5 dataset; hashes are verified before preprocessing.

```bash
kuafu train-dreamerv3 --rgb-index /absolute/path/to/rgb_datasets.json \
  --output world_models/kinova_shared_rgb --device cuda:0 --model-size size12m \
  --steps 25000 --batch-size 9 --length 32 --burnin 8 \
  --context 8 --horizon 16 --particles 1 --eval-batches 4 --report-every 500
```

This trains one shared visual world model from scratch. The encoder receives
the two RGB views and task identity; the RSSM also receives applied joint actions.
Object poses, robot state vectors, depth, and point clouds are not model inputs.
Camera images are resized separately to 112×64 and concatenated as six channels.
The image encoder and decoder use the pinned upstream strided-convolution option.
Image arrays are cached on disk. Tasks receive equal batch counts, with separate
80/10/10 episode splits per task and action scaling fitted on training episodes.

`best.npz` is selected by equally weighted validation RGB forecast error. Final
test reports include persistence and moving-region errors, plus camera forecast
previews in `test_previews/`. Loss and forecast metrics use RGB values in [0,1].
No actor or critic is trained.

Optimizer checkpoints are saved at each report. Continue into a new output
directory with `--resume-from /absolute/path/to/previous/run`; keep the same
dataset, seed, model, and sampling/evaluation options. `--steps` is the total
target update count. RGB cache files are reused, and optimizer/sampler pairing
is verified before resuming.

```python
from kuafu.dynamics.dreamerv3 import load_forecaster

forecast, metadata = load_forecaster("/absolute/path/to/model/best.npz", device="cuda:0")
# uint8 [batch, context, 64, 112, 6]; actions retain original relative-joint units.
rgb_predictions = forecast(context_rgb, previous_actions, future_actions,
                           task="drawer", particles=4, seed=42)
# float32 [batch, particles, horizon, 64, 112, 6], in [0,1].
```

## Point-cloud world models

Collect with `--pointcloud_transitions` to record both RGB and metric camera
depth on the same control steps. The recorder unprojects both depth images
using the camera calibration, fuses them in environment-local world axes, and
uniformly samples 1,024 XYZ points per frame. It keeps points inside the fixed
workspace `[-0.2, 1.5] × [-0.9, 0.9] × [0.01, 1.5]` meters. The HDF5 file also
stores calibration and valid-point counts; sparse or invalid frames fail capture.

Create a completed `kuafu-pointcloud-datasets-v1` index with the same task/path/hash
structure as the RGB index. The loader uses only `obs/pointcloud`, joint commands,
rewards, and task identity. It subsamples 512 XYZ points and applies a fixed
workspace normalization, preserving motion in the original coordinate system.

```bash
kuafu train-dreamerv3 --pointcloud-index /absolute/path/to/pointcloud_datasets.json \
  --output world_models/kinova_shared_pointcloud --device cuda:3 \
  --model-size size12m --num-points 512 --steps 25000 --batch-size 9 \
  --length 32 --burnin 8 --context 8 --horizon 16 --particles 1 \
  --eval-batches 4 --report-every 500
```

This adaptation uses the upstream DreamerV3 categorical RSSM, KL losses, reward
and continuation heads, and optimizer with a custom permutation-invariant point
encoder and point-set decoder. Reconstruction uses symmetric Chamfer distance.
It trains no actor or critic. Episode splits, task-balanced batches, action
scaling, and optimizer resume follow the RGB workflow. Validation selects the
checkpoint by forecast Chamfer RMSE in meters; test reports include persistence
baselines and point-set previews. Particle errors are measured separately, since
point indices do not establish correspondence between predictions.

Run the RGB command on GPU 0 and this command on GPU 3 in separate processes
with separate outputs to train one shared model per modality. Each process uses
only its selected GPU.

```python
forecast, metadata = load_forecaster("/absolute/path/to/pointcloud_model/best.npz",
                                     device="cuda:3")
# float32 [batch, context, 512, 3] in environment-local meters.
point_predictions = forecast(context_points, previous_actions, future_actions,
                             task="drawer", particles=4, seed=42)
# float32 [batch, particles, horizon, 512, 3], in the same meter coordinates.
```
