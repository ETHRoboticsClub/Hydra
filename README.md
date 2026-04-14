# Centralized Deployment Repository
Every division has its own subfolder to store its k8s deployment files.

Generic training workflow files live in `launch.d`.

Apply the persistent volume claims used by `./launch`:

```sh
./update-volumes
```
