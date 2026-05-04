# Centralized Deployment Repository
Hydra contains the tooling and configuration files to deploy workloads to the Kubernetes cluster. Each division has their own subfolder to store their deployment files.

For instructions on how to use the cluster: [Hercules Wiki](https://github.com/ETHRoboticsClub/Hercules/wiki/Deploying)

## Scripts
`launch`: Deploys an existing configuration directory to the cluster. Analog `kubectl apply` with some extra features.

`kill`: Kills a running workload on the cluster

`create`: Utility that allows creating PVCs (storage); jobs and deployments; and launching an interactive session for debugging.

The versions of these scripts with the `-new` suffix are experimental, providing a user interface. We plan to merge these features into the regular scripts in the future.