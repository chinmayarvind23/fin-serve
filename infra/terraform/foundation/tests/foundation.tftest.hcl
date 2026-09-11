# Mocked provider plans check configuration invariants without AWS credentials or service calls.
mock_provider "aws" {
  override_during = plan
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
    coredns = "v1.12.0-eksbuild.1", kube_proxy = "v1.35.0-eksbuild.1", vpc_cni = "v1.20.0-eksbuild.1", ebs_csi = "v1.60.0-eksbuild.1"
  }
}
run "private_bounded_foundation" {
  command = plan
  assert {
    condition     = length(aws_iam_role.node_autoscaler) == 0 && length(aws_autoscaling_group_tag.autoscaler) == 0
    error_message = "Node autoscaling IAM and discovery must require an explicit opt-in."
  }
  assert {
    condition     = aws_eks_node_group.cpu.scaling_config[0].min_size == 2 && aws_eks_node_group.cpu.scaling_config[0].max_size == 3
    error_message = "CPU nodes must stay bounded independently of controller installation."
  }
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
  assert {
    condition     = aws_eks_addon.ebs_csi.addon_name == "aws-ebs-csi-driver" && aws_eks_addon.ebs_csi.addon_version == var.addon_versions.ebs_csi
    error_message = "Model volumes require the explicitly pinned standard EBS CSI add-on."
  }
  assert {
    condition     = aws_iam_role_policy_attachment.ebs_csi.role == aws_iam_role.ebs_csi.name && aws_iam_role_policy_attachment.ebs_csi.policy_arn == "arn:aws:iam::aws:policy/AmazonEBSCSIDriverPolicyV2"
    error_message = "The managed-volume policy belongs to the dedicated controller role."
  }
}
run "scoped_node_autoscaler" {
  command = plan
  variables { enable_node_autoscaling = true }
  override_resource {
    target          = aws_iam_openid_connect_provider.cluster
    override_during = plan
    values = {
      arn = "arn:aws:iam::123456789012:oidc-provider/oidc.eks.us-east-1.amazonaws.com/id/FIXTURE"
    }
  }
  override_resource {
    target          = aws_eks_node_group.cpu
    override_during = plan
    values = {
      arn       = "arn:aws:eks:us-east-1:123456789012:nodegroup/finserve-staging/cpu/fixture"
      resources = [{ autoscaling_groups = [{ name = "fixture-cpu-asg" }] }]
    }
  }
  override_resource {
    target          = aws_eks_node_group.gpu
    override_during = plan
    values = {
      arn       = "arn:aws:eks:us-east-1:123456789012:nodegroup/finserve-staging/gpu/fixture"
      resources = [{ autoscaling_groups = [{ name = "fixture-gpu-asg" }] }]
    }
  }
  override_resource {
    target          = aws_subnet.private["0"]
    override_during = plan
    values          = { id = "subnet-fixture-first" }
  }
  assert {
    condition     = aws_eks_node_group.gpu.subnet_ids == toset(["subnet-fixture-first"])
    error_message = "A GPU returning from zero must remain in the model volume's chosen AZ."
  }
  assert {
    condition     = aws_eks_node_group.gpu.scaling_config[0].max_size == 1 && aws_eks_node_group.gpu.scaling_config[0].min_size == 0
    error_message = "GPU capacity must remain zero-to-one."
  }
  assert {
    condition = (
      jsondecode(aws_iam_role_policy.node_autoscaler[0].policy).Statement[0].Resource == [
        "arn:aws:autoscaling:us-east-1:123456789012:autoScalingGroup:*:autoScalingGroupName/fixture-cpu-asg",
        "arn:aws:autoscaling:us-east-1:123456789012:autoScalingGroup:*:autoScalingGroupName/fixture-gpu-asg"
      ] &&
      jsondecode(aws_iam_role_policy.node_autoscaler[0].policy).Statement[0].Condition.StringEquals == {
        "aws:ResourceTag/k8s.io/cluster-autoscaler/enabled"          = "true",
        "aws:ResourceTag/k8s.io/cluster-autoscaler/finserve-staging" = "owned"
      }
    )
    error_message = "Both actual ASG names and both cluster discovery tags must fence scaling writes."
  }
  assert {
    condition = jsondecode(aws_iam_role.node_autoscaler[0].assume_role_policy).Statement[0].Condition.StringEquals == {
      "${local.oidc_host}:aud" = "sts.amazonaws.com",
      "${local.oidc_host}:sub" = "system:serviceaccount:kube-system:finserve-cluster-autoscaler"
    }
    error_message = "Only the dedicated controller account may assume the node scaling role."
  }
  assert {
    condition = (
      jsondecode(aws_iam_role_policy.node_autoscaler[0].policy).Statement[0].Action == [
        "autoscaling:SetDesiredCapacity", "autoscaling:TerminateInstanceInAutoScalingGroup"
      ] &&
      alltrue([for action in jsondecode(aws_iam_role_policy.node_autoscaler[0].policy).Statement[1].Action :
        startswith(action, "autoscaling:Describe") || startswith(action, "ec2:Describe") || action == "ec2:GetInstanceTypesFromInstanceRequirements"
      ]) &&
      jsondecode(aws_iam_role_policy.node_autoscaler[0].policy).Statement[2].Action == "eks:DescribeNodegroup" &&
      jsondecode(aws_iam_role_policy.node_autoscaler[0].policy).Statement[2].Resource == [
        "arn:aws:eks:us-east-1:123456789012:nodegroup/finserve-staging/cpu/fixture",
        "arn:aws:eks:us-east-1:123456789012:nodegroup/finserve-staging/gpu/fixture"
      ]
    )
    error_message = "Capacity discovery must remain read-only; scaling cannot grant tagging or node-group configuration writes."
  }
  assert {
    condition = (
      aws_autoscaling_group_tag.autoscaler["gpu:k8s.io/cluster-autoscaler/node-template/taint/nvidia.com/gpu"].tag[0].value == "true:NoSchedule" &&
      aws_autoscaling_group_tag.autoscaler["gpu:k8s.io/cluster-autoscaler/node-template/resources/nvidia.com/gpu"].tag[0].value == "1" &&
      aws_autoscaling_group_tag.autoscaler["gpu:k8s.io/cluster-autoscaler/node-template/label/finserve.io/pool"].tag[0].value == "gpu" &&
      alltrue([for tag in aws_autoscaling_group_tag.autoscaler : tag.tag[0].propagate_at_launch])
    )
    error_message = "Scale-from-zero requires matching GPU labels, taint and resource metadata."
  }
  assert {
    condition = alltrue([for key, tag in aws_autoscaling_group_tag.autoscaler :
      tag.autoscaling_group_name == (startswith(key, "gpu:") ? "fixture-gpu-asg" : "fixture-cpu-asg")
    ])
    error_message = "Discovery tags must attach to each actual managed ASG."
  }
}
run "reject_unbounded_gpu_input" {
  command = plan
  variables { gpu_desired_nodes = 20 }
  expect_failures = [var.gpu_desired_nodes]
}
run "reject_unpinned_storage_driver" {
  command = plan
  variables {
    addon_versions = {
      coredns = "v1.12.0-eksbuild.1", kube_proxy = "v1.35.0-eksbuild.1", vpc_cni = "v1.20.0-eksbuild.1", ebs_csi = "latest"
    }
  }
  expect_failures = [var.addon_versions]
}
run "reject_malformed_addon_version" {
  command = plan
  variables {
    addon_versions = {
      coredns = "v1.12.0-eksbuild.1", kube_proxy = "v1.35.0-eksbuild.1", vpc_cni = "v1.20.0-eksbuild.1", ebs_csi = "v1 not-a-version-eksbuild.1"
    }
  }
  expect_failures = [var.addon_versions]
}
