variable "pm_api_url" {
  type        = string
  description = "Proxmox API endpoint (e.g., https://192.168.1.10:8006/api2/json)."
  default     = "https://192.168.168.10.45:8006/api2/json"
}

variable "pm_user" {
  type        = string
  description = "Proxmox API user with token access."
  default     = "root@pam"
}

variable "pm_api_token_id" {
  type        = string
  description = "Proxmox API token identifier (user@realm!token-name)."
  sensitive   = true
  default     = null
}

variable "pm_api_token_secret" {
  type        = string
  description = "Proxmox API token secret."
  sensitive   = true
  default     = null
}

variable "pm_target_node" {
  type        = string
  description = "Target Proxmox node to host the runner container."
}

variable "pm_lxc_template" {
  type        = string
  description = "Proxmox storage/template identifier (e.g., local:vztmpl/debian-12-standard_12.0-1_amd64.tar.zst)."
}

variable "pm_rootfs_storage" {
  type        = string
  description = "Proxmox storage location for the runner root filesystem (e.g., local-lvm)."
}

variable "pm_network_bridge" {
  type        = string
  description = "Network bridge to attach to the LXC container."
}

variable "runner_repo" {
  type        = string
  description = "GitHub repository to bind the self-hosted runner against (owner/name)."

  validation {
    condition     = can(regex("^[^/]+/[^/]+$", var.runner_repo))
    error_message = "runner_repo must be provided in the form owner/name."
  }
}

variable "runner_ssh_public_key" {
  type        = string
  description = "SSH public key injected into the runner container for provisioning access."

  validation {
    condition     = can(regex("^ssh-(rsa|ed25519) ", var.runner_ssh_public_key))
    error_message = "runner_ssh_public_key must be a valid OpenSSH public key in authorized_keys format."
  }
}

variable "runner_ssh_private_key_path" {
  type        = string
  description = "Filesystem path to the SSH private key matching runner_ssh_public_key. May be relative to the project root."
}

variable "runner_memory" {
  type        = number
  description = "Memory allocation (MB) for the LXC container."
  default     = 32768
}

variable "runner_cores" {
  type        = number
  description = "CPU core allocation for the LXC container."
  default     = 4
}

variable "runner_rootfs_size" {
  type        = string
  description = "Disk size for the runner container rootfs."
  default     = "256G"
}

variable "runner_timezone" {
  type        = string
  description = "Timezone to configure inside the LXC container."
  default     = "UTC"
}

variable "runner_labels" {
  type        = list(string)
  description = "Extra labels to assign to the self-hosted runner."
  default     = []
}

variable "runner_image" {
  type        = string
  description = "GitHub Actions runner tarball URL."
  default     = "https://github.com/actions/runner/releases/download/v2.321.0/actions-runner-linux-x64-2.321.0.tar.gz"
}

variable "bazel_cache_instance" {
  description = "Configuration for the Bazel remote cache LXC container."
  type = object({
    name         = string
    vm_id        = number
    ipv4_cidr    = string
    ipv4_gateway = optional(string)
    cores        = number
    memory       = number
    rootfs_size  = string
  })
  default = null
}

variable "runner_instances" {
  description = "Runner instances to provision on Proxmox."
  type = list(object({
    name         = string
    vm_id        = number
    ipv4_cidr    = string
    ipv4_gateway = optional(string)
  }))
}

