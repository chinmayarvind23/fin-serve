# Foundation intentionally has no Kubernetes provider: cluster access is a later stage.
terraform {
  required_version = "= 1.16.2"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "= 6.61.0"
    }
  }
  backend "s3" {
    encrypt      = true
    use_lockfile = true
  }
}

provider "aws" {
  region              = var.region
  allowed_account_ids = [var.aws_account_id]
  default_tags {
    tags = { Project = "FinServe", Environment = var.name, ManagedBy = "Terraform" }
  }
}
