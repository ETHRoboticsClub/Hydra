# Centralized Deployment Repository
Hydra contains the tooling and configuration files to deploy workloads to the Kubernetes cluster. Each division has their own subfolder to store their deployment files.

For instructions on how to use the cluster: [Hercules Wiki](https://github.com/ETHRoboticsClub/Hercules/wiki/Deploying)

## Scripts
`launch`: Deploys an existing configuration directory to the cluster. Analog `kubectl apply` with some extra features.

`kill`: Kills a running workload on the cluster

`create`: Utility that allows creating PVCs (storage); jobs and deployments; and launching an interactive session for debugging.

The versions of these scripts with the `-new` suffix are experimental. We plan to merge these features into the regular scripts in the future.

## Reading
### Overview
[Why Kubernetes](https://kubernetes.io/docs/concepts/overview/)

### Storage and Services
[PVs and PVCs (Storage)](https://kubernetes.io/docs/concepts/storage/persistent-volumes/)

[Service](https://kubernetes.io/docs/concepts/services-networking/service/)

### Workloads
[Jobs](https://kubernetes.io/docs/concepts/workloads/controllers/job/)

[Deployments](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/)