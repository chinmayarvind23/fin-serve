# Discovery tags are applied to the actual EKS-managed ASGs, not merely the node groups.
locals {
  autoscaler_groups = {
    cpu = aws_eks_node_group.cpu.resources[0].autoscaling_groups[0].name
    gpu = aws_eks_node_group.gpu.resources[0].autoscaling_groups[0].name
  }
  autoscaler_common_tags = {
    "k8s.io/cluster-autoscaler/enabled"     = "true"
    "k8s.io/cluster-autoscaler/${var.name}" = "owned"
  }
  autoscaler_pool_tags = {
    cpu = merge(local.autoscaler_common_tags, {
      "k8s.io/cluster-autoscaler/node-template/label/finserve.io/pool" = "cpu"
    })
    gpu = merge(local.autoscaler_common_tags, {
      "k8s.io/cluster-autoscaler/node-template/label/finserve.io/pool"              = "gpu"
      "k8s.io/cluster-autoscaler/node-template/label/k8s.amazonaws.com/accelerator" = aws_eks_node_group.gpu.labels["k8s.amazonaws.com/accelerator"]
      "k8s.io/cluster-autoscaler/node-template/taint/nvidia.com/gpu"                = "true:NoSchedule"
      "k8s.io/cluster-autoscaler/node-template/resources/nvidia.com/gpu"            = "1"
    })
  }
  autoscaler_tags = merge([for pool, tags in local.autoscaler_pool_tags : {
    for key, value in tags : "${pool}:${key}" => { pool = pool, key = key, value = value }
  }]...)
}

resource "aws_autoscaling_group_tag" "autoscaler" {
  for_each               = var.enable_node_autoscaling ? local.autoscaler_tags : {}
  autoscaling_group_name = local.autoscaler_groups[each.value.pool]
  tag {
    key                 = each.value.key
    value               = each.value.value
    propagate_at_launch = true
  }
}

resource "aws_iam_role" "node_autoscaler" {
  count = var.enable_node_autoscaling ? 1 : 0
  name  = "${var.name}-node-autoscaler"
  assume_role_policy = jsonencode({ Version = "2012-10-17", Statement = [{
    Effect    = "Allow", Action = "sts:AssumeRoleWithWebIdentity",
    Principal = { Federated = aws_iam_openid_connect_provider.cluster.arn },
    Condition = { StringEquals = {
      "${local.oidc_host}:aud" = "sts.amazonaws.com",
      "${local.oidc_host}:sub" = "system:serviceaccount:kube-system:finserve-cluster-autoscaler"
    } }
  }] })
}

resource "aws_iam_role_policy" "node_autoscaler" {
  count = var.enable_node_autoscaling ? 1 : 0
  role  = aws_iam_role.node_autoscaler[0].name
  # Both actual ASG names and discovery tags fence mutation to this cluster's two groups.
  policy = jsonencode({ Version = "2012-10-17", Statement = [
    {
      Sid    = "ScaleOwnedGroups", Effect = "Allow",
      Action = ["autoscaling:SetDesiredCapacity", "autoscaling:TerminateInstanceInAutoScalingGroup"],
      Resource = [for name in values(local.autoscaler_groups) :
        "arn:aws:autoscaling:${var.region}:${var.aws_account_id}:autoScalingGroup:*:autoScalingGroupName/${name}"
      ],
      Condition = { StringEquals = {
        "aws:ResourceTag/k8s.io/cluster-autoscaler/enabled"     = "true",
        "aws:ResourceTag/k8s.io/cluster-autoscaler/${var.name}" = "owned"
      } }
    },
    {
      Sid = "DiscoverCapacity", Effect = "Allow", Resource = "*",
      Action = [
        "autoscaling:DescribeAutoScalingGroups", "autoscaling:DescribeAutoScalingInstances",
        "autoscaling:DescribeLaunchConfigurations", "autoscaling:DescribeScalingActivities",
        "autoscaling:DescribeTags", "ec2:DescribeImages", "ec2:DescribeInstanceTypes",
        "ec2:DescribeLaunchTemplateVersions", "ec2:GetInstanceTypesFromInstanceRequirements"
      ]
    },
    {
      Sid      = "DescribeOwnedNodeGroups", Effect = "Allow", Action = "eks:DescribeNodegroup",
      Resource = [aws_eks_node_group.cpu.arn, aws_eks_node_group.gpu.arn]
    }
  ] })
}
