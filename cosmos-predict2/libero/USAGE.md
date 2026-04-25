# Cosmos Predict2 — LIBERO Interactive GPU Pod

## Start the pod

```bash
# Let Karpenter pick a GPU from the pool
bash cosmos-predict2/libero/start.sh

# Request a specific instance (resources are set automatically to the machine's max)
bash cosmos-predict2/libero/start.sh --instance-type g6e.xlarge    # 1× L40S, 32GB RAM
bash cosmos-predict2/libero/start.sh --instance-type g6e.2xlarge   # 1× L40S, 64GB RAM
bash cosmos-predict2/libero/start.sh --instance-type g6e.12xlarge  # 4× L40S, 384GB RAM
bash cosmos-predict2/libero/start.sh --instance-type p5.4xlarge    # 1× H100, 192GB RAM

# Pass a WandB API key (available as WANDB_API_KEY in the shell, not stored anywhere)
bash cosmos-predict2/libero/start.sh --wandb-key <your-key>
bash cosmos-predict2/libero/start.sh --instance-type g6e.12xlarge --wandb-key <your-key>

# Smoke test: 50-iter training run → evaluate (quick sanity check, ~$2-5)
# Runs unattended, pod shuts down automatically when done
bash cosmos-predict2/libero/start.sh --instance-type g6e.12xlarge --smoketest
bash cosmos-predict2/libero/start.sh --instance-type g6e.12xlarge --smoketest --wandb-key <your-key>

# Full run unattended: train 7000 iters → evaluate (~$120-250)
# Assumes data prep steps have already been completed
bash cosmos-predict2/libero/start.sh --instance-type g6e.12xlarge --fullrun
bash cosmos-predict2/libero/start.sh --instance-type g6e.12xlarge --fullrun --wandb-key <your-key>

# Override the auto-shutdown timer for interactive mode (default 12h)
SHUTDOWN_AFTER=21600 bash cosmos-predict2/libero/start.sh   # 6h
SHUTDOWN_AFTER=0     bash cosmos-predict2/libero/start.sh   # no timeout
```

First run takes **5–15 minutes** — Karpenter needs to provision the EC2 node and pull the Docker image (~10 GB). Subsequent runs attach instantly.

## What runs automatically before you get the shell

`main.sh` executes inside the container at pod startup:

1. Installs `tmux` and `ffmpeg` if not already present
2. Installs `uv` to `/data/.uv-bin/` (skipped on restart if already there)
3. Clones `ETHRoboticsClub/cosmos-predict2` (branch `libero`) to `/data/cosmos-predict2`, or pulls latest if already cloned
4. Runs `uv sync --extra cu126` and activates the venv
5. **Smoketest/fullrun mode**: runs the pipeline then exits → pod terminates → node is shut down automatically
6. **Interactive mode**: prints the quick-start guide and sleeps until timeout (keeps the pod alive)

By the time you get the shell, `uv`, the repo, and the venv are already there.

## Monitor the machine

```bash
nvidia-smi                    # GPU model, VRAM, running processes
watch -n1 nvidia-smi          # live GPU stats
htop                          # CPU and RAM
df -h /data                   # disk usage on the persistent volume
```

## Follow logs (smoketest / fullrun)

`start.sh` tails logs automatically. If you need to reconnect:

```bash
POD=$(kubectl -n robot-learning get pods -l app=cosmos-libero-run -o jsonpath='{.items[-1].metadata.name}')
kubectl -n robot-learning logs -f ${POD}
```

Or use k9s → navigate to the pod → press `L`.

## Run the LIBERO workflow (interactive)

All steps run inside the pod shell. Steps are safe to re-run — downloads resume, conversions skip if output already exists.

```bash
cd /data/cosmos-predict2
ulimit -n 65535
# 1–5. Download everything from S3 (checkpoints + fully prepared dataset, including conversions)
aws s3 sync s3://ethrc-ml-data-916780037007/cosmos-predict2-libero/checkpoints /data/cosmos-predict2/checkpoints --region us-east-1
aws s3 sync s3://ethrc-ml-data-916780037007/cosmos-predict2-libero/datasets /data/cosmos-predict2/datasets --region us-east-1
# optional to get old checkpoins
aws s3 sync s3://ethrc-ml-data-916780037007/cosmos-predict2-libero/outputs /data/cosmos-predict2/outputs --region us-east-1


```

Alternatively, prepare from scratch (steps 1–5):

```bash
cd /data/cosmos-predict2

# 1. HF auth (paste token when prompted, or set HF_TOKEN first)
hf auth login

# 2. Download the Cosmos-Predict2-2B model
python scripts/download_checkpoints.py \
  --model_types video2world --model_sizes 2B --resolution 480 --fps 10

# 3. Download the LIBERO dataset (~27 GB, safe to re-run if interrupted)
huggingface-cli download nvidia/LIBERO-Cosmos-Policy \
  --repo-type dataset --include "all_episodes/*" \
  --local-dir datasets/libero_cosmos

# 4. Convert HDF5 → MP4 + captions, split train/val
uv run --with h5py --with pillow --with tqdm \
  python scripts/prepare_libero_cosmos_dataset.py \
  --src datasets/libero_cosmos/all_episodes \
  --out datasets/libero_cosmos_mp4 \
  --fps 10

# 5. Generate T5 embeddings (run for both splits)
python -m scripts.get_t5_embeddings --dataset_path datasets/libero_cosmos_mp4/train
python -m scripts.get_t5_embeddings --dataset_path datasets/libero_cosmos_mp4/val

# 6. Quick sanity check (4 GPUs, 50 iters)
IMAGINAIRE_OUTPUT_ROOT=outputs uv run torchrun \
  --nproc_per_node=4 --master_port=12341 \
  -m scripts.train \
  --config=cosmos_predict2/configs/base/config.py -- \
  experiment=predict2_video2world_training_2b_libero_cosmos \
  model_parallel.context_parallel_size=2 \
  dataloader_train.batch_size=4 \
  trainer.max_iter=50 trainer.validation_iter=5 \
  trainer.max_val_iter=2 checkpoint.save_iter=50

# 7. Full training (use --fullrun flag instead for unattended + auto-shutdown)
IMAGINAIRE_OUTPUT_ROOT=outputs uv run torchrun \
  --nproc_per_node=4 --master_port=12341 \
  -m scripts.train \
  --config=cosmos_predict2/configs/base/config.py -- \
  experiment=predict2_video2world_training_2b_libero_cosmos \
  model_parallel.context_parallel_size=2 \
  dataloader_train.batch_size=4 \
  dataloader_val.batch_size=2 \
  trainer.max_iter=7000 \
  trainer.grad_accum_iter=16 \
  trainer.validation_iter=50 \
  trainer.max_val_iter=10 \
  trainer.callbacks.draw_sample.every_n=100 \
  trainer.callbacks.draw_sample.is_sample=True \
  trainer.callbacks.draw_sample.show_all_frames=True \
  trainer.callbacks.draw_sample.guidance='[7.0]' \
  checkpoint.save_iter=500
```

Checkpoints are saved to `outputs/posttraining/video2world_lora/2b_libero_cosmos/checkpoints/` every 500 steps.

## Copy files from the pod to your laptop

Run these from your laptop (not inside the pod).

```bash
# Get the pod name (interactive pod)
POD=$(kubectl -n robot-learning get pods | grep cosmos-libero-interactive | awk '{print $1}')

# Copy checkpoints to your laptop
kubectl -n robot-learning cp ${POD}:/data/cosmos-predict2/outputs/posttraining ./local-checkpoints

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

## Auto-shutdown (interactive mode)

**Default: 12 hours.** The pod shuts itself down after 12h, the EC2 instance is terminated, and you stop paying for compute. The PVC is not deleted — your data is safe.

Smoketest and fullrun pods shut down automatically when the pipeline completes, regardless of this setting.

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
  cosmos-predict2/
    .venv/                              Python virtualenv
    outputs/
      posttraining/video2world_lora/
        2b_libero_cosmos/
          checkpoints/                  training checkpoints (every 500 steps)
    datasets/
      libero_cosmos/                    raw HDF5 dataset (~27 GB)
      libero_cosmos_mp4/                converted MP4 + T5 embeddings
    eval/
      base/                             baseline evaluation results
      finetuned/                        finetuned model evaluation results
  .uv-bin/                              uv binary
```