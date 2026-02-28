# Deployment commands on our EKS cluster

# Training Deployment

kubectl apply -f training/job.yaml

Minimum resources: 
    - 4 CPUs
    - 16 GiBs RAM 
    - 1 GPU

# Inference Deployment

kubectl apply -f inference/deployment.yaml

Minimum resources: 
    - 2 CPUs
    - 8 GiBs RAM 
    - 1 GPU


# Simulation Deployment

kubectl apply -f simulation/job.yaml

Minimum resources:
    - 4 CPUs
    - 16 GiBs RAM
    - 1 GPU




