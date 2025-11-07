pm_api_url        = "https://192.168.10.45:8006/api2/json"
pm_target_node    = "prox4"
pm_lxc_template   = "local:vztmpl/debian-12-standard_12.12-1_amd64.tar.zst"
pm_rootfs_storage = "local-lvm"
pm_network_bridge = "vmbr0"

runner_vm_id                = 990
runner_repo                 = "josnelihurt/learning-cuda"
runner_name                 = "learning-cuda-prox4-x86"
runner_ssh_public_key       = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPa6IR4wZReEICQYWFTI8+YcLtygFvdjy9511N7Cpb/K github-runner-prox4"
runner_ssh_private_key_path = ".secrets/keys/github-runner-prox4_ed25519"
runner_ipv4_cidr            = "192.168.10.90/24"
runner_ipv4_gateway         = "192.168.10.1"

