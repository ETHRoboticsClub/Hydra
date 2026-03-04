# Deployment commands on our EKS cluster

# Training Deployment

Train ACT policy:
kubectl apply -f training/act.yaml

Minimum resources: 
    - 4 CPUs
    - 16 GiBs RAM 
    - 1 GPU

Training data is stored in the PersistentVolume 'training-data-pv' (read-only)
Pods then access this data using the PersistentVolumeClaim 'training-data-pvc'

Checkpoints are read from/written to using a PersistentVolumeClaim 'checkpoints-pvc' which is based on a custom StorageClass object 'gp3'

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




