# Mocked provider plans check configuration invariants without AWS credentials or service calls.
mock_provider "aws" {
  mock_resource "aws_eks_cluster" {
    defaults = { identity = [{ oidc = [{ issuer = "https://oidc.eks.us-east-1.amazonaws.com/id/FIXTURE" }] }] }
  }
  mock_resource "aws_db_instance" {
    defaults = { master_user_secret = [{ secret_arn = "arn:aws:secretsmanager:us-east-1:123456789012:secret:fixture" }] }
  }
}
variables {
  aws_account_id          = "123456789012"
  availability_zones      = ["us-east-1a", "us-east-1b"]
  cluster_admin_role_arn  = "arn:aws:iam::123456789012:role/fixture-admin"
  cpu_ami_release_version = "1.35.0-20260101"
  gpu_ami_release_version = "1.35.0-20260101"
  postgres_version        = "17.6"
  redis_auth_token        = "fixtureOnlyNotASecret00000000000000"
  addon_versions = {
    coredns = "v1.12.0-eksbuild.1", kube_proxy = "v1.35.0-eksbuild.1", vpc_cni = "v1.20.0-eksbuild.1"
  }
}
run "private_bounded_foundation" {
  command = plan
  assert {
    condition     = !aws_eks_cluster.main.vpc_config[0].endpoint_public_access && aws_eks_cluster.main.vpc_config[0].endpoint_private_access
    error_message = "The Kubernetes API must remain private."
  }
  assert {
    condition     = aws_eks_node_group.gpu.scaling_config[0].max_size == 1 && aws_eks_node_group.gpu.scaling_config[0].desired_size == 0
    error_message = "Idle staging must not allocate an unbounded GPU fleet."
  }
  assert {
    condition     = aws_db_instance.registry.storage_encrypted && !aws_db_instance.registry.publicly_accessible && aws_db_instance.registry.deletion_protection
    error_message = "Registry storage must be encrypted, private and protected from routine deletion."
  }
  assert {
    condition     = aws_elasticache_replication_group.redis.transit_encryption_enabled && aws_elasticache_replication_group.redis.at_rest_encryption_enabled
    error_message = "Redis requires transport and storage encryption."
  }
  assert {
    condition     = aws_s3_bucket_public_access_block.evidence.block_public_policy && aws_s3_bucket_versioning.evidence.versioning_configuration[0].status == "Enabled"
    error_message = "Evidence must block public policies and retain versions."
  }
}
run "reject_unbounded_gpu_input" {
  command = plan
  variables { gpu_desired_nodes = 20 }
  expect_failures = [var.gpu_desired_nodes]
}
