# Cosmos Predict2 — LIBERO Interactive GPU Pod

## Start the pod

```bash
# Let Karpenter pick a GPU from the pool
bash cosmos-predict2/libero/start.sh

# Request a specific GPU type
bash cosmos-predict2/libero/start.sh --instance-type g6e.xlarge    # 1× L40S 48GB
bash cosmos-predict2/libero/start.sh --instance-type g5.xlarge     # 1× A10G 24GB
bash cosmos-predict2/libero/start.sh --instance-type p4de.24xlarge # 8× A100 80GB

# Override the auto-shutdown timer (default 12h)
SHUTDOWN_AFTER=21600 bash cosmos-predict2/libero/start.sh   # 6h
SHUTDOWN_AFTER=0     bash cosmos-predict2/libero/start.sh   # no timeout
```

First run takes **5–15 minutes** — Karpenter needs to provision the EC2 node and pull the Docker image (~10 GB). Subsequent runs attach instantly.

## What runs automatically before you get the shell

`main.sh` executes inside the container at pod startup:

1. Installs `uv` to `/data/.uv-bin/` (skipped on restart if already there)
2. Installs `tmux`
3. Clones `ETHRoboticsClub/cosmos-predict2` (branch `libero`) to `/data/cosmos-predict2`
4. Prints the step-by-step guide
5. Sleeps until the timeout (keeps the pod alive)

By the time you get the shell, `uv` and the repo are already there.

## Disconnect without killing anything

You land in a `tmux` session. `tmux` keeps everything running inside the pod even if your laptop closes.

```
Ctrl+B  D      detach — leaves the session running, drops you back to your laptop
```

Run `start.sh` again to reattach. Your commands are still running.

## tmux basics inside the shell

```
Ctrl+B  C      open a new window
Ctrl+B  0      switch to window 0
Ctrl+B  1      switch to window 1
Ctrl+B  D      detach (go back to laptop)
Ctrl+B  [      scroll mode (use arrow keys, Q to exit)
```

Typical use: one window for training, one for `watch nvidia-smi`.

## Monitor the machine

```bash
nvidia-smi                    # GPU model, VRAM, running processes
watch -n1 nvidia-smi          # live GPU stats
htop                          # CPU and RAM
df -h /data                   # disk usage on the persistent volume
```

## Run the LIBERO workflow

All steps run inside the pod shell. Steps are safe to re-run — downloads resume, conversions skip if output already exists.

```bash
cd /data/cosmos-predict2

# 1. Install Python deps
uv sync --extra cu126
source .venv/bin/activate

# 2. Authenticate with Hugging Face (interactive, paste your token)
hf auth login

# 3. Download the Cosmos-Predict2-2B model
python scripts/download_checkpoints.py \
  --model_types video2world --model_sizes 2B --resolution 480 --fps 10

# 4. Download the LIBERO dataset (~27 GB, safe to re-run if interrupted)
huggingface-cli download nvidia/LIBERO-Cosmos-Policy \
  --repo-type dataset --include "all_episodes/*" \
  --local-dir datasets/libero_cosmos

# 5. Convert HDF5 → MP4 + captions, split train/val
uv run --with h5py --with pillow --with tqdm \
  python scripts/prepare_libero_cosmos_dataset.py \
  --src datasets/libero_cosmos/all_episodes \
  --out datasets/libero_cosmos_mp4 \
  --fps 10

# 6. Generate T5 embeddings (run for both splits)
python -m scripts.get_t5_embeddings --dataset_path datasets/libero_cosmos_mp4/train
python -m scripts.get_t5_embeddings --dataset_path datasets/libero_cosmos_mp4/val

# 7. Smoke test — validates the setup in ~5 minutes (1 GPU)
IMAGINAIRE_OUTPUT_ROOT=outputs torchrun \
  --nproc_per_node=1 \
  --master_port=12341 \
  -m scripts.train \
  --config=cosmos_predict2/configs/base/config.py -- \
  experiment=predict2_video2world_training_2b_libero_cosmos \
  trainer.max_iter=5 \
  trainer.validation_iter=1 \
  trainer.max_val_iter=2 \
  checkpoint.save_iter=999999

# 8. Full training (change nproc_per_node to match your GPU count)
IMAGINAIRE_OUTPUT_ROOT=/data/checkpoints torchrun \
  --nproc_per_node=1 \
  --master_port=12341 \
  -m scripts.train \
  --config=cosmos_predict2/configs/base/config.py -- \
  experiment=predict2_video2world_training_2b_libero_cosmos
```

Checkpoints are saved to `/data/checkpoints/` every 500 steps.

## Copy files from the pod to your laptop

Run these from your laptop (not inside the pod).

```bash
# Get the pod name
POD=$(kubectl -n robot-learning get pods | grep cosmos-libero-interactive | awk '{print $1}')

# Copy checkpoints to your laptop
kubectl -n robot-learning cp ${POD}:/data/checkpoints ./local-checkpoints

# Copy a specific file
kubectl -n robot-learning cp ${POD}:/data/cosmos-predict2/outputs/some-file.mp4 ./some-file.mp4
```

## Check pod status from your laptop

```bash
# Quick status
bash cosmos-predict2/libero/describe.sh

# Full pod logs (stdout from main.sh)
POD=$(kubectl -n robot-learning get pods | grep cosmos-libero-interactive | awk '{print $1}')
kubectl -n robot-learning logs ${POD}

# Follow logs live
kubectl -n robot-learning logs ${POD} -f
```

## Shutdown

```bash
# Stop the pod, keep the data (PVC stays, ~$24/month)
bash cosmos-predict2/libero/delete-job.sh

# Stop the pod and delete all data (billing stops completely)
bash cosmos-predict2/libero/delete-job.sh
kubectl -n robot-learning delete pvc cosmos-libero
```

The PVC must be deleted manually — deleting the job alone does not delete the volume.

## Auto-shutdown

**Default: 12 hours.** The pod shuts itself down after 12h, the EC2 instance is terminated, and you stop paying for compute. The PVC is not deleted — your data is safe.

```bash
# Default (12h)
bash cosmos-predict2/libero/start.sh

# Custom timeout
SHUTDOWN_AFTER=21600 bash cosmos-predict2/libero/start.sh   # 6h
SHUTDOWN_AFTER=86400 bash cosmos-predict2/libero/start.sh   # 24h

# No timeout
SHUTDOWN_AFTER=0 bash cosmos-predict2/libero/start.sh
```

## Data layout on the volume

```
/data/
  cosmos-predict2/          repo clone (branch: libero)
  datasets/
    libero_cosmos/           raw HDF5 dataset (~27 GB)
    libero_cosmos_mp4/       converted MP4 + T5 embeddings
  checkpoints/               training outputs (saved every 500 steps)
  .uv-bin/                   uv binary
  .venv/                     Python virtualenv
```
