output "cluster_name" {
  description = "Use an authorized operator from the private network for the Kubernetes stage."
  value       = aws_eks_cluster.main.name
}
output "node_autoscaler_role_arn" {
  description = "Bind only kube-system/finserve-cluster-autoscaler; null until explicitly enabled."
  value       = try(aws_iam_role.node_autoscaler[0].arn, null)
}
output "gpu_availability_zone" {
  description = "Retained model PVs must match this single GPU node-group AZ before deployment."
  value       = var.availability_zones[0]
}
output "cluster_endpoint" {
  description = "Private Kubernetes API endpoint, not a public model endpoint."
  value       = aws_eks_cluster.main.endpoint
}
output "evidence_bucket" {
  value = aws_s3_bucket.evidence.id
}
output "evidence_role_arn" {
  value = aws_iam_role.evidence.arn
}
output "registry_endpoint" {
  value = aws_db_instance.registry.address
}
output "registry_secret_arn" {
  description = "Reference only; obtain secret bytes through separately authorized deployment tooling."
  value       = aws_db_instance.registry.master_user_secret[0].secret_arn
}
output "redis_endpoint" {
  value = aws_elasticache_replication_group.redis.primary_endpoint_address
}
output "application_repository" {
  value = aws_ecr_repository.application.repository_url
}
