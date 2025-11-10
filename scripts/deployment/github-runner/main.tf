terraform {
  required_version = ">= 1.5.0"

  required_providers {
    proxmox = {
      source  = "Telmate/proxmox"
      version = "~> 2.9"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }
}

# Run Terraform from a host with API access to Proxmox and an authenticated `gh` CLI session.
# The file `.secrets/proxmox-api.key` should contain JSON with `token_id` and `token_secret` for the configured Proxmox user.
# Typical workflow:
#   terraform init
#   terraform apply -var='runner_repo=owner/repo' -var='runner_ssh_public_key=ssh-rsa AAA...' -var='pm_lxc_template=storage:vztmpl/template.tar.zst' -var='pm_target_node=pve-node1' -var='pm_rootfs_storage=local-lvm' -var='runner_vm_id=1234'

locals {
  project_root        = abspath("${path.root}/../../..")
  secrets_path        = "${local.project_root}/.secrets/proxmox-api.key"
  proxmox_secret_raw  = try(file(local.secrets_path), null)
  proxmox_credentials = local.proxmox_secret_raw == null ? null : try(jsondecode(local.proxmox_secret_raw), null)

  pm_api_token_id     = try(coalesce(var.pm_api_token_id, try(local.proxmox_credentials.token_id, null)), null)
  pm_api_token_secret = try(coalesce(var.pm_api_token_secret, try(local.proxmox_credentials.token_secret, null)), null)

  pm_api_url_clean = replace(replace(var.pm_api_url, "https://", ""), "http://", "")
  pm_api_host_port = element(split("/", local.pm_api_url_clean), 0)
  proxmox_host     = element(split(":", local.pm_api_host_port), 0)

  runner_private_key_path = can(regex("^/", var.runner_ssh_private_key_path)) ? var.runner_ssh_private_key_path : abspath("${local.project_root}/${var.runner_ssh_private_key_path}")
  runner_label_list       = distinct(compact(concat(["proxmox", "lxc"], var.runner_labels)))
  runner_label_string     = length(local.runner_label_list) > 0 ? join(",", local.runner_label_list) : ""

  runner_instances_map = {
    for instance in var.runner_instances :
    instance.name => instance
  }

  runner_connections = {
    for name, instance in local.runner_instances_map :
    name => {
      host             = element(split("/", instance.ipv4_cidr), 0)
      user             = "root"
      private_key_path = local.runner_private_key_path
    }
  }

  runner_instance_details = {
    for name, instance in local.runner_instances_map :
    name => {
      vm_id        = instance.vm_id
      ipv4_cidr    = instance.ipv4_cidr
      ipv4_address = element(split("/", instance.ipv4_cidr), 0)
      ipv4_gateway = try(instance.ipv4_gateway, null)
    }
  }

  bazel_cache_instance = var.bazel_cache_instance
  bazel_cache_details = local.bazel_cache_instance == null ? null : {
    vm_id        = local.bazel_cache_instance.vm_id
    ipv4_cidr    = local.bazel_cache_instance.ipv4_cidr
    ipv4_address = element(split("/", local.bazel_cache_instance.ipv4_cidr), 0)
    ipv4_gateway = try(local.bazel_cache_instance.ipv4_gateway, null)
    cores        = local.bazel_cache_instance.cores
    memory       = local.bazel_cache_instance.memory
    rootfs_size  = local.bazel_cache_instance.rootfs_size
  }
  bazel_cache_connection = local.bazel_cache_instance == null ? null : {
    host             = element(split("/", local.bazel_cache_instance.ipv4_cidr), 0)
    user             = "root"
    private_key_path = local.runner_private_key_path
  }
}

provider "proxmox" {
  pm_api_url          = var.pm_api_url
  pm_user             = var.pm_user
  pm_api_token_id     = local.pm_api_token_id
  pm_api_token_secret = local.pm_api_token_secret
  pm_tls_insecure     = true
}

resource "random_password" "runner_root" {
  for_each         = local.runner_instances_map
  length           = 24
  special          = true
  override_special = "!#%^*-_"
}

resource "proxmox_lxc" "runner" {
  for_each     = local.runner_instances_map
  target_node  = var.pm_target_node
  vmid         = each.value.vm_id
  hostname     = each.value.name
  ostemplate   = var.pm_lxc_template
  password     = random_password.runner_root[each.key].result
  start        = true
  onboot       = true
  unprivileged = true
  cores        = var.runner_cores
  memory       = var.runner_memory
  swap         = 0
  ostype       = "ubuntu"
  description  = "Terraform-managed GitHub Actions runner"

  features {
    nesting = true
  }

  rootfs {
    storage = var.pm_rootfs_storage
    size    = var.runner_rootfs_size
  }

  network {
    name   = "eth0"
    bridge = var.pm_network_bridge
    ip     = each.value.ipv4_cidr
    gw     = try(each.value.ipv4_gateway, null)
  }

  ssh_public_keys = var.runner_ssh_public_key
}

resource "random_password" "bazel_cache_root" {
  count            = local.bazel_cache_instance == null ? 0 : 1
  length           = 24
  special          = true
  override_special = "!#%^*-_"
}

resource "proxmox_lxc" "bazel_cache" {
  count        = local.bazel_cache_instance == null ? 0 : 1
  target_node  = var.pm_target_node
  vmid         = local.bazel_cache_instance.vm_id
  hostname     = local.bazel_cache_instance.name
  ostemplate   = var.pm_lxc_template
  password     = random_password.bazel_cache_root[count.index].result
  start        = true
  onboot       = true
  unprivileged = true
  cores        = local.bazel_cache_instance.cores
  memory       = local.bazel_cache_instance.memory
  swap         = 0
  ostype       = "ubuntu"
  description  = "Terraform-managed Bazel remote cache"

  features {
    nesting = true
  }

  rootfs {
    storage = var.pm_rootfs_storage
    size    = local.bazel_cache_instance.rootfs_size
  }

  network {
    name   = "eth0"
    bridge = var.pm_network_bridge
    ip     = local.bazel_cache_instance.ipv4_cidr
    gw     = try(local.bazel_cache_instance.ipv4_gateway, null)
  }

  ssh_public_keys = var.runner_ssh_public_key
}

