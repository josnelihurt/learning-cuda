pm_api_url        = "https://192.168.10.45:8006/api2/json"
pm_target_node    = "prox4"
pm_lxc_template   = "local:vztmpl/debian-12-standard_12.12-1_amd64.tar.zst"
pm_rootfs_storage = "local-lvm"
pm_network_bridge = "vmbr0"

runner_repo                 = "josnelihurt/learning-cuda"
runner_ssh_public_key       = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPa6IR4wZReEICQYWFTI8+YcLtygFvdjy9511N7Cpb/K github-runner-prox4"
runner_ssh_private_key_path = ".secrets/keys/github-runner-prox4_ed25519"
runner_cores                = 8

runner_instances = [
  {
    name         = "learning-cuda-prox4-x86-1"
    vm_id        = 990
    ipv4_cidr    = "192.168.10.90/24"
    ipv4_gateway = "192.168.10.1"
  },
  {
    name         = "learning-cuda-prox4-x86-2"
    vm_id        = 991
    ipv4_cidr    = "192.168.10.91/24"
    ipv4_gateway = "192.168.10.1"
  },
  {
    name         = "learning-cuda-prox4-x86-3"
    vm_id        = 992
    ipv4_cidr    = "192.168.10.92/24"
    ipv4_gateway = "192.168.10.1"
  },
  {
    name         = "learning-cuda-prox4-x86-4"
    vm_id        = 993
    ipv4_cidr    = "192.168.10.93/24"
    ipv4_gateway = "192.168.10.1"
  },
]

