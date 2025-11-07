output "runner_instance_id" {
  description = "Proxmox VMID of the runner container."
  value       = proxmox_lxc.runner.id
}

output "proxmox_host" {
  description = "Hostname or IP address of the Proxmox node managing the runner."
  value       = local.proxmox_host
}

output "runner_ipv4" {
  description = "IPv4 address assigned to the runner container (without CIDR)."
  value       = local.runner_ipv4_address
}

output "runner_ipv4_cidr" {
  description = "CIDR notation used for the runner container address."
  value       = var.runner_ipv4_cidr
}

output "runner_ipv4_gateway" {
  description = "Gateway configured for the runner container."
  value       = var.runner_ipv4_gateway
}

output "runner_private_key_path" {
  description = "Absolute path to the SSH private key for the runner container."
  value       = local.runner_private_key_path
}

output "runner_connection" {
  description = "Connection parameters for provisioning the runner via Ansible."
  value = {
    host             = local.runner_ipv4_address
    user             = "root"
    private_key_path = local.runner_private_key_path
  }
}

output "runner_template" {
  description = "Proxmox LXC template identifier used to create the runner."
  value       = var.pm_lxc_template
}

output "runner_image_url" {
  description = "URL for the GitHub Actions runner binary used during provisioning."
  value       = var.runner_image
}

output "runner_repo" {
  description = "GitHub repository bound to this runner."
  value       = var.runner_repo
}

output "runner_name" {
  description = "Logical name assigned to the runner."
  value       = var.runner_name
}

output "runner_labels" {
  description = "Labels that should be applied to the runner."
  value       = local.runner_label_list
}

output "runner_labels_csv" {
  description = "Comma-separated representation of runner labels."
  value       = local.runner_label_string
}

