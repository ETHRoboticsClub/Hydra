# Hydra Quickstart

Hydra contains Kubernetes deployment manifests for training, inference, and simulation workloads on the EKS cluster.

## Prerequisites

- AWS CLI configured with valid credentials (`aws sts get-caller-identity`)
- `kubectl` connected to the cluster: `aws eks update-kubeconfig --name ethrc-prod-1 --region us-east-1`
- Your IAM principal added to the cluster access list (see Hercules `cluster_access` variable)

## Repository structure
