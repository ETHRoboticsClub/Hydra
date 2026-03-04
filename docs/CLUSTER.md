# Configure your local kubectl

## IAM Authentication

aws login
aws eks update-kubeconfig --name ethrc-prod-1 --region us-east-1

# Install the Kubeflow Trainer

kubectl apply --server-side -k "https://github.com/kubeflow/trainer.git/manifests/overlays/manager?ref=v2.1.0"
kubectl apply --server-side -k "https://github.com/kubeflow/trainer.git/manifests/overlays/runtimes?ref=v2.1.0"

# Configure the W&B API key

1. Go to https://wandb.ai/authorize.
2. Copy the API key.
3. Run the following command on the cluster
   kubectl create secret generic training-secrets \
   --from-literal=wandb-api-key=YOUR_API_KEY \
   -n hydra

# Enable GPU support in Kubernetes using the NVIDIA device plugin

kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.1/deployments/static/nvidia-device-plugin.yml
