resource "aws_iam_role" "cluster" {
  name = "${var.name}-eks"
  assume_role_policy = jsonencode({ Version = "2012-10-17", Statement = [{
    Effect = "Allow", Action = "sts:AssumeRole", Principal = { Service = "eks.amazonaws.com" }
  }] })
}
resource "aws_iam_role_policy_attachment" "cluster" {
  role       = aws_iam_role.cluster.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
}
resource "aws_cloudwatch_log_group" "cluster" {
  name              = "/aws/eks/${var.name}/cluster"
  retention_in_days = 30
}
resource "aws_eks_cluster" "main" {
  name     = var.name
  role_arn = aws_iam_role.cluster.arn
  version  = var.kubernetes_version
  access_config {
    authentication_mode                         = "API"
    bootstrap_cluster_creator_admin_permissions = false
  }
  vpc_config {
    subnet_ids              = [for subnet in aws_subnet.private : subnet.id]
    endpoint_private_access = true
    endpoint_public_access  = false
  }
  enabled_cluster_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]
  depends_on                = [aws_iam_role_policy_attachment.cluster, aws_cloudwatch_log_group.cluster]
}
resource "aws_eks_access_entry" "operator" {
  cluster_name  = aws_eks_cluster.main.name
  principal_arn = var.cluster_admin_role_arn
  type          = "STANDARD"
}
resource "aws_eks_access_policy_association" "operator" {
  cluster_name  = aws_eks_cluster.main.name
  principal_arn = aws_eks_access_entry.operator.principal_arn
  policy_arn    = "arn:aws:eks::aws:cluster-access-policy/AmazonEKSClusterAdminPolicy"
  access_scope { type = "cluster" }
}
resource "aws_iam_role" "nodes" {
  name = "${var.name}-nodes"
  assume_role_policy = jsonencode({ Version = "2012-10-17", Statement = [{
    Effect = "Allow", Action = "sts:AssumeRole", Principal = { Service = "ec2.amazonaws.com" }
  }] })
}
resource "aws_iam_role_policy_attachment" "nodes" {
  for_each   = toset(["AmazonEKSWorkerNodePolicy", "AmazonEC2ContainerRegistryPullOnly"])
  role       = aws_iam_role.nodes.name
  policy_arn = "arn:aws:iam::aws:policy/${each.key}"
}
resource "aws_launch_template" "nodes" {
  for_each    = toset(["cpu", "gpu"])
  name_prefix = "${var.name}-${each.key}-"
  metadata_options {
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
  }
  block_device_mappings {
    device_name = "/dev/xvda"
    ebs {
      volume_size           = each.key == "gpu" ? 100 : 40
      volume_type           = "gp3"
      encrypted             = true
      delete_on_termination = true
    }
  }
}
resource "aws_eks_node_group" "cpu" {
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "cpu"
  node_role_arn   = aws_iam_role.nodes.arn
  subnet_ids      = [for subnet in aws_subnet.private : subnet.id]
  instance_types  = ["m6i.large"]
  ami_type        = "AL2023_x86_64_STANDARD"
  release_version = var.cpu_ami_release_version
  version         = var.kubernetes_version
  capacity_type   = "ON_DEMAND"
  labels          = { "finserve.io/pool" = "cpu" }
  scaling_config {
    desired_size = 2
    min_size     = 2
    max_size     = 2
  }
  update_config { max_unavailable = 1 }
  launch_template {
    id      = aws_launch_template.nodes["cpu"].id
    version = aws_launch_template.nodes["cpu"].latest_version
  }
  depends_on = [aws_iam_role_policy_attachment.nodes, aws_eks_addon.cni]
}
resource "aws_eks_node_group" "gpu" {
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "gpu"
  node_role_arn   = aws_iam_role.nodes.arn
  subnet_ids      = [for subnet in aws_subnet.private : subnet.id]
  instance_types  = [var.gpu_instance_type]
  ami_type        = "AL2023_x86_64_NVIDIA"
  release_version = var.gpu_ami_release_version
  version         = var.kubernetes_version
  capacity_type   = "ON_DEMAND"
  labels          = { "finserve.io/pool" = "gpu" }
  taint {
    key    = "nvidia.com/gpu"
    value  = "true"
    effect = "NO_SCHEDULE"
  }
  scaling_config {
    desired_size = var.gpu_desired_nodes
    min_size     = 0
    max_size     = 1
  }
  update_config { max_unavailable = 1 }
  launch_template {
    id      = aws_launch_template.nodes["gpu"].id
    version = aws_launch_template.nodes["gpu"].latest_version
  }
  depends_on = [aws_iam_role_policy_attachment.nodes, aws_eks_addon.cni]
}
resource "aws_iam_openid_connect_provider" "cluster" {
  url            = aws_eks_cluster.main.identity[0].oidc[0].issuer
  client_id_list = ["sts.amazonaws.com"]
}
locals {
  oidc_host = replace(aws_iam_openid_connect_provider.cluster.url, "https://", "")
}
resource "aws_iam_role" "cni" {
  name = "${var.name}-cni"
  assume_role_policy = jsonencode({ Version = "2012-10-17", Statement = [{
    Effect    = "Allow", Action = "sts:AssumeRoleWithWebIdentity",
    Principal = { Federated = aws_iam_openid_connect_provider.cluster.arn },
    Condition = { StringEquals = {
      "${local.oidc_host}:aud" = "sts.amazonaws.com",
      "${local.oidc_host}:sub" = "system:serviceaccount:kube-system:aws-node"
    } }
  }] })
}
resource "aws_iam_role_policy_attachment" "cni" {
  role       = aws_iam_role.cni.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
}
resource "aws_eks_addon" "cni" {
  cluster_name             = aws_eks_cluster.main.name
  addon_name               = "vpc-cni"
  addon_version            = var.addon_versions.vpc_cni
  service_account_role_arn = aws_iam_role.cni.arn
  configuration_values     = jsonencode({ enableNetworkPolicy = "true" })
  depends_on               = [aws_iam_role_policy_attachment.cni]
}
resource "aws_eks_addon" "system" {
  for_each      = { coredns = var.addon_versions.coredns, kube-proxy = var.addon_versions.kube_proxy }
  cluster_name  = aws_eks_cluster.main.name
  addon_name    = each.key
  addon_version = each.value
  depends_on    = [aws_eks_node_group.cpu]
}
