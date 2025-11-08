output "proxmox_host" {
  description = "Hostname or IP address of the Proxmox node managing the runner."
  value       = local.proxmox_host
}

output "runner_private_key_path" {
  description = "Absolute path to the SSH private key for the runner container."
  value       = local.runner_private_key_path
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

output "runner_labels" {
  description = "Labels that should be applied to the runner."
  value       = local.runner_label_list
}

output "runner_labels_csv" {
  description = "Comma-separated representation of runner labels."
  value       = local.runner_label_string
}

output "runner_names" {
  description = "List of runner names defined for this deployment."
  value       = keys(local.runner_instances_map)
}

output "runner_instances" {
  description = "Runner instance details keyed by runner name."
  value = {
    for name, details in local.runner_instance_details :
    name => merge(details, {
      proxmox_id = proxmox_lxc.runner[name].id
    })
  }
}

output "runner_connections" {
  description = "Connection parameters for provisioning each runner via Ansible."
  value       = local.runner_connections
}

