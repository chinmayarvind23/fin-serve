variable "name" {
  description = "Unique environment name, also used for scoped resource names."
  type        = string
  default     = "finserve-staging"
  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{2,25}$", var.name))
    error_message = "Use 3-26 lowercase letters, numbers and hyphens."
  }
}
variable "aws_account_id" {
  description = "Explicit account fence; provider refuses credentials for a different account."
  type        = string
  validation {
    condition     = can(regex("^[0-9]{12}$", var.aws_account_id))
    error_message = "A 12-digit intended AWS account ID is required."
  }
}
variable "region" {
  description = "AWS region; GPU capacity, quotas and service versions need authenticated preflight."
  type        = string
  default     = "us-east-1"
}
variable "availability_zones" {
  description = "Two distinct AZ names in the selected region; no hidden discovery at validation."
  type        = list(string)
  validation {
    condition     = length(var.availability_zones) == 2 && length(distinct(var.availability_zones)) == 2
    error_message = "Supply exactly two distinct availability zones."
  }
}
variable "vpc_cidr" {
  description = "IPv4 /16 leaves disjoint /24 public and private subnets."
  type        = string
  default     = "10.72.0.0/16"
  validation {
    condition     = can(cidrnetmask(var.vpc_cidr)) && endswith(var.vpc_cidr, "/16")
    error_message = "Supply a valid IPv4 /16 CIDR."
  }
}
variable "kubernetes_version" {
  description = "Pinned EKS minor, verified in AWS standard-support documentation."
  type        = string
  default     = "1.35"
  validation {
    condition     = var.kubernetes_version == "1.35"
    error_message = "This verified configuration profile targets EKS 1.35."
  }
}
variable "cluster_admin_role_arn" {
  description = "Existing operator role; private API also requires VPC/VPN/SSM network access."
  type        = string
  validation {
    condition     = can(regex("^arn:aws:iam::[0-9]{12}:role/.+$", var.cluster_admin_role_arn))
    error_message = "Provide an existing IAM role ARN for explicit EKS API access."
  }
}
variable "cpu_ami_release_version" {
  description = "Exact EKS AL2023 x86_64 standard AMI release, resolved during authenticated preflight."
  type        = string
  validation {
    condition     = can(regex("^[0-9]+\\.[0-9]+\\.[0-9]+-[0-9]{8}$", var.cpu_ami_release_version))
    error_message = "Pin an actual CPU AMI release version."
  }
}
variable "gpu_ami_release_version" {
  description = "Exact AL2023 x86_64 NVIDIA AMI release compatible with the chosen EKS minor."
  type        = string
  validation {
    condition     = can(regex("^[0-9]+\\.[0-9]+\\.[0-9]+-[0-9]{8}$", var.gpu_ami_release_version))
    error_message = "Pin an actual GPU AMI release version."
  }
}
variable "addon_versions" {
  description = "Exact EKS add-on builds; caller must resolve compatibility before plan/apply."
  type        = object({ coredns = string, kube_proxy = string, vpc_cni = string, ebs_csi = string })
  validation {
    condition     = alltrue([for value in values(var.addon_versions) : can(regex("^v[0-9]+\\.[0-9]+\\.[0-9]+-eksbuild\\.[0-9]+$", value))])
    error_message = "Each add-on must use an exact vX.Y.Z-eksbuild.N version."
  }
}
variable "gpu_instance_type" {
  description = "Single-GPU NVIDIA instance family; capacity and price remain unverified."
  type        = string
  default     = "g5.xlarge"
  validation {
    condition     = contains(["g5.xlarge", "g5.2xlarge", "g6.xlarge", "g6.2xlarge"], var.gpu_instance_type)
    error_message = "This bounded profile supports only listed single-GPU instance sizes."
  }
}
variable "gpu_desired_nodes" {
  description = "0 keeps staging GPU idle; set 1 before installing the GPU engine workload."
  type        = number
  default     = 0
  validation {
    condition     = contains([0, 1], var.gpu_desired_nodes)
    error_message = "GPU desired capacity is bounded to zero or one node."
  }
}
variable "postgres_version" {
  description = "Exact supported PostgreSQL engine minor; validate regional availability before plan."
  type        = string
  validation {
    condition     = can(regex("^[0-9]+\\.[0-9]+$", var.postgres_version))
    error_message = "Specify an exact PostgreSQL major.minor version."
  }
}
variable "redis_auth_token" {
  description = "Externally generated Redis auth token; Terraform state must use encrypted restricted S3."
  type        = string
  sensitive   = true
  validation {
    condition     = can(regex("^[A-Za-z0-9]{32,128}$", var.redis_auth_token))
    error_message = "Supply a 32-128 character alphanumeric token through a secret input channel."
  }
}
