# Keep volume credentials on the controller service account, outside the node role.
resource "aws_iam_role" "ebs_csi" {
  name = "${var.name}-ebs-csi"
  assume_role_policy = jsonencode({ Version = "2012-10-17", Statement = [{
    Effect    = "Allow", Action = "sts:AssumeRoleWithWebIdentity",
    Principal = { Federated = aws_iam_openid_connect_provider.cluster.arn },
    Condition = { StringEquals = {
      "${local.oidc_host}:aud" = "sts.amazonaws.com",
      "${local.oidc_host}:sub" = "system:serviceaccount:kube-system:ebs-csi-controller-sa"
    } }
  }] })
}
resource "aws_iam_role_policy_attachment" "ebs_csi" {
  role       = aws_iam_role.ebs_csi.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEBSCSIDriverPolicyV2"
}
# The exact regional build is an input, never an unverified latest-version lookup.
resource "aws_eks_addon" "ebs_csi" {
  cluster_name             = aws_eks_cluster.main.name
  addon_name               = "aws-ebs-csi-driver"
  addon_version            = var.addon_versions.ebs_csi
  service_account_role_arn = aws_iam_role.ebs_csi.arn
  depends_on               = [aws_iam_role_policy_attachment.ebs_csi, aws_eks_node_group.cpu]
}
