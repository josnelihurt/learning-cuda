# Scripts Overview

This directory consolidates operational tooling used during development, CI builds, and deployments. Each subfolder groups scripts by lifecycle stage; the sections below summarize intent and entry points.

## Directory Catalog
- `build/`: language-specific build helpers (`golang.sh`, `protos.sh`, `frontend.sh`) wired into CI jobs or local automation.
- `deployment/`: infrastructure launchers for staging, production, GitHub runners, and ARM64 hardware. See [Deployment Tooling](#deployment-tooling) for deeper coverage.
- `dev/`: developer convenience wrappers (`start.sh`, `stop.sh`, `clean.sh`, `grafana-mcp-server.sh`) for local iterative flows with hot reload.
- `docker/`: utilities for building images locally, generating certs, and validating host prerequisites, highlighted in [Local Docker Tooling](#local-docker-tooling).
- `hooks/`: git hook installers and language-specific linters run before commits or pushes.
- `linters/`: language-agnostic lint orchestration, including the Dockerfile used by the language compliance pipeline.
- `secrets/`: bootstrap scripts for encrypted secrets material.
- `test/`: coverage, unit, integration, BDD, E2E, and workflow-local runners mirroring CI behaviour.
- `tools/`: ad-hoc video and frame analysis helpers for experimentation and debugging.

## Interaction Map
```mermaid
flowchart LR
    DevStart[scripts/dev/start.sh]
    BuildProtos[scripts/build/protos.sh]
    DockerLocal[scripts/docker/build-local.sh]
    Tests[scripts/test/*.sh]
    Staging[scripts/deployment/staging_local/start.sh]
    Runners[scripts/deployment/github-runner/start.sh]
    Radxa[scripts/deployment/radxa/deploy-runner.sh]
    Prox4[scripts/deployment/prox4/ansible/site.yml]

    DevStart --> DockerLocal
    BuildProtos --> Tests
    DockerLocal --> Staging
    Tests --> Staging
    Runners --> CI[Self-hosted runners]
    Prox4 --> Runners
    Radxa --> CI
```

## Deployment Tooling
### `deployment/github-runner`
- **Purpose**: provisions self-hosted GitHub Actions runners for ARM64 and AMD64 workloads using Terraform (Proxmox VMs) and Ansible (runtime configuration, runner registration).
- **Pipeline**: `start.sh` verifies dependencies (`terraform`, `gh`, `ansible-playbook`, `jq`, `python3`), applies Terraform, renders per-runner Ansible variable files, executes `site.yml`, then polls the GitHub API until each runner reports `online`.
- **Outputs**: self-hosted runners labeled for workflow selection (`jetson-nano`, architecture-specific tags) and preloaded with Docker, NVIDIA toolkit, and registry credentials required by CI.
- **Maintenance**: `stop.sh` tears down instances via Terraform destroy; secrets such as `proxmox-api.key` live under `.secrets/` and must exist before provisioning.

### `deployment/staging_local`
- Ships a docker-compose stack mirroring production, pulling AMD64 images from GHCR. Useful for validating image outputs from CI on a laptop or workstation.

### `deployment/jetson-nano`
- Automates production deployment on Jetson Nano hardware using Ansible playbooks (`deploy.sh` orchestrates `init`, `sync`, and `start`). Aligns with the ARM64 CI workflow outputs.

### `deployment/cloud-vm`
- Automates production deployment of the Go server to a cloud VM (x86_64) using Ansible playbooks. The Go server runs separately from the Jetson Nano deployment, which only hosts the gRPC server (C++ with CUDA) and infrastructure services.
- **Pipeline**: `deploy.sh` orchestrates Ansible playbooks: `ansible/sync.yml` (sync configuration and secrets) and `ansible/start.yml` (pull images and start services). Also includes `deploy-runner.sh` for provisioning a self-hosted GitHub Actions runner (`learning-cuda-cloud-vm-1`, labels: `self-hosted,Linux,X64,prod,cloud-vm`).
- **Automation**: Integrated into the x86 CI workflow (`docker-monorepo-build-x86.yml`) to automatically deploy after building and pushing images to GHCR.
- **Requirements**: SSH access to the cloud VM, Docker installed, user in docker group. Secrets configured in GitHub Actions: `CLOUD_VM_HOST`, `CLOUD_VM_USER`, `CLOUD_VM_SSH_KEY`.
- **Configuration**: Uses `docker-compose.go-cloud.yml` to deploy only the Go server service, connecting to existing Traefik instance via `public-wan` Docker network.

### `deployment/local_dev`
- Provides `start.sh` for fast local stacks without cloud dependencies, targeting developers iterating on scripts or runtime settings.

### `deployment/radxa`
- Automates GitHub Actions runner provisioning on Radxa ARM64 hardware. `deploy-runner.sh` registers runner `learning-cuda-radxa-1` with labels `self-hosted,Linux,ARM64,radxa`. Supports both `RADXA_*` and `JETSON_*` environment variables for compatibility. `test.sh` validates SSH connectivity and Ansible availability. Includes Ansible playbooks for application deployment and Docker orchestration. See [`deployment/radxa/README.md`](radxa/README.md) for full documentation.

## Local Docker Tooling
- `docker/build-local.sh`: builds the monorepo Docker image using the same Dockerfiles as CI, enabling preflight validation before opening pull requests.
- `docker/install-nvidia-toolkit.sh`: configures host NVIDIA drivers and container toolkit, matching the requirements enforced on self-hosted runners.
- `docker/generate-certs.sh`: issues local TLS certs consumed by `scripts/dev/start.sh` and staging stacks.
- `docker/validate-env.sh`: validates Docker environment prerequisites (SSL certs, Docker daemon, NVIDIA toolkit, GPU availability).

## Runner Operations Reference
For provisioning internals, troubleshooting workflows, and label conventions, refer to the dedicated CI documentation in [`docs/ci-workflows.md`](../docs/ci-workflows.md). The two documents complement each other: this file covers script entry points, while the CI doc explains how the workflows consume the resulting infrastructure.
