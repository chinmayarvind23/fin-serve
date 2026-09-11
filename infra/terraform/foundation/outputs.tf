output "cluster_name" {
  description = "Use an authorized operator from the private network for the Kubernetes stage."
  value       = aws_eks_cluster.main.name
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
