# Radxa ARM Runner Deployment

This directory contains Ansible playbooks and scripts to deploy a GitHub Actions self-hosted runner on a Radxa ARM server.

## Prerequisites

1. Ansible installed on your local machine
2. SSH access to the Radxa server (configure via RADXA_HOST or JETSON_HOST)
3. GitHub CLI (`gh`) installed and authenticated
4. Environment variables configured (supports both RADXA_* and JETSON_* variables)
   The deployment will use JETSON_* variables as fallback if RADXA_* are not set.

## Configuration

The deployment uses environment variables. You can either:

1. **Use RADXA_* variables** (recommended):
   ```bash
   export RADXA_HOST=<your-radxa-ip-address>
   export RADXA_USER=<your-username>
   export RADXA_SUDO_PASSWORD=<your-password>
   export RADXA_APP_DIRECTORY=/path/to/app
   ```

2. **Use JETSON_* variables** (for compatibility):
   The script will automatically use JETSON_* variables if RADXA_* are not set.
   Current documented standard uses JETSON_* variables in `.secrets/production.env.example`.

## Usage

### Deploy/Start Runner

```bash
./deploy-runner.sh
```

This will:
- Install required packages on the Radxa server
- Download and configure the GitHub Actions runner
- Register the runner with GitHub
- Start the runner as a systemd service

### Stop Runner

```bash
./deploy-runner.sh --stop
```

This will:
- Stop the runner service
- Unregister the runner from GitHub
- Remove runner files

### Test Connection

```bash
./test.sh
```

This validates:
- Required environment variables are set
- Ansible is installed
- SSH connection to the Radxa server works

## Runner Details

- **Name**: `learning-cuda-radxa-1`
- **Labels**: `self-hosted,Linux,ARM64,radxa`
- **Installation Path**: `/opt/actions-runner-radxa`
- **Service Name**: `actions.runner.josnelihurt-learning-cuda.learning-cuda-radxa-1`

## Advanced Usage

### Additional Ansible Playbooks

The `ansible/` directory contains additional playbooks for manual operations:

- `init.yml` - Initialize Radxa server with required dependencies
- `sync.yml` - Sync project files to Radxa server
- `start.yml` - Start Docker services on Radxa server
- `clean.yml` - Clean up deployment artifacts

These can be run directly with `ansible-playbook` if needed:

```bash
cd ansible
ansible-playbook -i inventory.yml init.yml
```

### Ansible Roles

The deployment uses three Ansible roles:

- `roles/application/` - Git clone and application setup
- `roles/docker-deploy/` - Docker deployment and service management
- `roles/setup/` - System setup and dependency installation

## Notes

- The runner runs as root (required for Docker operations)
- The runner service is configured to auto-restart on failure
- The runner will automatically download the latest ARM64 runner binary

