# Centralized Deployment Repository

Every division has its own subfolder for its k8s deployment files. The generic
launch tooling lives in `launch.d/`.

## Setup

Apply the persistent volume claims used by `./launch`:

```sh
./update-volumes
```

## Two ways to launch a job

There are two flavours of the launcher in this repo:

| Script | Source of truth | Use when… |
|---|---|---|
| `./launch` | A **remote git repo** that's cloned in-pod at start | The job is stable and lives in another repo (e.g. `lerobot-edit-scripts`) |
| `./launch-new` | A **local folder** in this repo (or a remote URL with a subpath) | You want fast iteration on the entrypoint/deploy spec without committing-and-pushing |

The wishlist tracked in [#10](https://github.com/ETHRoboticsClub/Hydra/issues/10)
is converging the two — `./launch-new` is the path forward.

## `./launch-new` — local folder workflow

Drop a folder under your division (e.g. `robot-learning/<name>/`) with these
files:

```
robot-learning/<name>/
├── deploy.yaml      # Kubernetes manifest(s) with {{VARS}} placeholders. Multi-doc OK (separated by `---`).
├── entrypoint.sh    # The script the main container runs. Required.
├── init.sh          # Optional: extra shell snippet appended to the initContainer.
├── pyproject.toml   # Optional: deps for `uv sync` to consume.
└── *.py / configs/  # Any other top-level files you reference from entrypoint.sh.
```

Then:

```sh
# Local folder (fast iteration, no git roundtrip):
./launch-new gpus ./robot-learning/act-new

# Or a remote repo with optional subpath:
./launch-new gpul https://github.com/ETHRoboticsClub/lerobot-edit-scripts@train/act
```

### How files get into the pod

Every regular top-level file in the folder (except `deploy.yaml`) is bundled
into a ConfigMap (`<job-name>-files`). The init container copies the whole
ConfigMap to `/workspace/`, then the main container runs `/workspace/entrypoint.sh`.

ConfigMaps are limited to ~1 MB and don't recurse into subfolders. For
larger payloads or nested directories, fetch them from S3 / git inside
`entrypoint.sh`.

### Available `{{VARS}}` in `deploy.yaml`

| Variable | Source |
|---|---|
| `{{JOB_NAME}}` | Auto-generated `launch-new-YYYYMMDD-HHMMSS` |
| `{{NAMESPACE}}`, `{{SERVICE_ACCOUNT}}`, `{{IMAGE}}`, `{{INIT_IMAGE}}` | `launch.d/generic.yaml` |
| `{{DATA_PVC}}`, `{{CHECKPOINT_PVC}}`, `{{S3_PVC}}` | `launch.d/generic.yaml` |
| `{{NODEPOOL}}`, `{{NODETIER}}` | The chosen `launch.d/instance_types/<profile>.yaml` |
| `{{CPU_REQUEST}}`, `{{CPU_LIMIT}}`, `{{MEM_REQUEST}}`, `{{MEM_LIMIT}}`, `{{GPU_REQUEST}}`, `{{GPU_LIMIT}}` | Profile |
| `{{SHM_SIZE}}`, `{{REPLICAS}}` | Profile / defaults |
| `{{LAUNCHER_ID}}`, `{{LAUNCHER_HASH}}`, `{{USER}}` | `aws sts get-caller-identity` (falls back to `$USER@hostname`) |
| `{{CONFIGMAP_NAME}}` | `<job-name>-files` |
| `{{INIT_SCRIPT}}` | The bash snippet that copies job files into `/workspace` (and runs `init.sh` if you provided one) |

### Multi-resource deploys

`deploy.yaml` can hold any number of Kubernetes resources separated by `---`.
`./launch-new` substitutes vars across the whole file in one pass, then runs a
single `kubectl apply` so the documents land atomically.

`./kill-new` cleans up across all kinds it knows about: TrainJob, Job,
Deployment, InferenceService, Service, ConfigMap. If you add a resource of a
new kind, also add it to the `KINDS` array in `kill-new`.

### Examples in this repo

| Folder | Demonstrates |
|---|---|
| `robot-learning/act-new/` | Single `Job` (one-shot training) — the simplest case |
| `robot-learning/act-inference/` | Multi-resource: `Deployment` + `Service` for serving a trained policy |
| `robot-learning/act-with-embedder/` | The validation case from issue [#10](https://github.com/ETHRoboticsClub/Hydra/issues/10) — `Deployment` + `Service` + `Job` in one launch, training pod waits on embedder via initContainer health probe, then queries it via the Service DNS name |
| `sample/training/` | Reference for the legacy `./launch` (TrainJob via remote clone) |

### Common commands

```sh
./launch-new --list-profiles                 # gpus, gpul, gpum, a100, a100-80
./launch-new --dry-run gpus ./robot-learning/act-new   # render manifest, don't apply
./launch-new gpus ./robot-learning/act-new   # apply
./kill-new mine                              # delete jobs you launched
./kill-new <job-name>                        # delete one job (and its ConfigMap, Service, etc.)
./kill-new robot-learning                    # delete every launch-new job in the namespace (prompts)
```

### Logs

```sh
kubectl -n robot-learning get pods -l ethrobotics.ch/job-name=<job-name>
kubectl -n robot-learning logs -f -l ethrobotics.ch/job-name=<job-name>
```

For a Deployment with a Service (e.g. `act-inference`), reach the server with:

```sh
kubectl -n robot-learning port-forward svc/<job-name> 8080:80
curl http://localhost:8080/healthz
```
