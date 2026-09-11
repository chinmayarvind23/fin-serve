resource "aws_security_group" "data" {
  name        = "${var.name}-data"
  description = "Only EKS nodes can initiate database and cache connections."
  vpc_id      = aws_vpc.main.id
}
resource "aws_vpc_security_group_ingress_rule" "postgres" {
  security_group_id            = aws_security_group.data.id
  referenced_security_group_id = aws_eks_cluster.main.vpc_config[0].cluster_security_group_id
  from_port                    = 5432
  to_port                      = 5432
  ip_protocol                  = "tcp"
}
resource "aws_vpc_security_group_ingress_rule" "redis" {
  security_group_id            = aws_security_group.data.id
  referenced_security_group_id = aws_eks_cluster.main.vpc_config[0].cluster_security_group_id
  from_port                    = 6379
  to_port                      = 6379
  ip_protocol                  = "tcp"
}
resource "aws_db_subnet_group" "main" {
  name       = var.name
  subnet_ids = [for subnet in aws_subnet.private : subnet.id]
}
resource "aws_db_instance" "registry" {
  identifier                          = "${var.name}-registry"
  engine                              = "postgres"
  engine_version                      = var.postgres_version
  instance_class                      = "db.t4g.micro"
  allocated_storage                   = 20
  max_allocated_storage               = 50
  storage_type                        = "gp3"
  storage_encrypted                   = true
  db_name                             = "finserve"
  username                            = "finserve_admin"
  manage_master_user_password         = true
  iam_database_authentication_enabled = true
  db_subnet_group_name                = aws_db_subnet_group.main.name
  vpc_security_group_ids              = [aws_security_group.data.id]
  publicly_accessible                 = false
  multi_az                            = false
  backup_retention_period             = 7
  deletion_protection                 = true
  skip_final_snapshot                 = false
  final_snapshot_identifier           = "${var.name}-registry-final"
  auto_minor_version_upgrade          = false
  copy_tags_to_snapshot               = true
  enabled_cloudwatch_logs_exports     = ["postgresql", "upgrade"]
}
resource "aws_elasticache_subnet_group" "main" {
  name       = var.name
  subnet_ids = [for subnet in aws_subnet.private : subnet.id]
}
resource "aws_elasticache_replication_group" "redis" {
  replication_group_id       = "${var.name}-redis"
  description                = "FinServe bounded rate policy and cache; SQL owns durable truth"
  engine                     = "redis"
  engine_version             = "7.1"
  node_type                  = "cache.t4g.micro"
  num_cache_clusters         = 1
  port                       = 6379
  parameter_group_name       = "default.redis7"
  subnet_group_name          = aws_elasticache_subnet_group.main.name
  security_group_ids         = [aws_security_group.data.id]
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  auth_token                 = var.redis_auth_token
  automatic_failover_enabled = false
  snapshot_retention_limit   = 1
  auto_minor_version_upgrade = false
}
resource "aws_s3_bucket" "evidence" {
  bucket        = "${var.name}-${var.aws_account_id}-${var.region}-evidence"
  force_destroy = false
}
resource "aws_s3_bucket_public_access_block" "evidence" {
  bucket                  = aws_s3_bucket.evidence.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
resource "aws_s3_bucket_versioning" "evidence" {
  bucket = aws_s3_bucket.evidence.id
  versioning_configuration { status = "Enabled" }
}
resource "aws_s3_bucket_server_side_encryption_configuration" "evidence" {
  bucket = aws_s3_bucket.evidence.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "AES256" }
  }
}
resource "aws_s3_bucket_policy" "evidence" {
  bucket = aws_s3_bucket.evidence.id
  policy = jsonencode({ Version = "2012-10-17", Statement = [{
    Sid       = "DenyUnencryptedTransport", Effect = "Deny", Principal = "*", Action = "s3:*",
    Resource  = [aws_s3_bucket.evidence.arn, "${aws_s3_bucket.evidence.arn}/*"],
    Condition = { Bool = { "aws:SecureTransport" = "false" } }
  }] })
}
resource "aws_ecr_repository" "application" {
  name                 = var.name
  image_tag_mutability = "IMMUTABLE"
  force_delete         = false
  image_scanning_configuration { scan_on_push = true }
  encryption_configuration { encryption_type = "AES256" }
}
# Only the evidence writer service account receives these object permissions.
resource "aws_iam_role" "evidence" {
  name = "${var.name}-evidence"
  assume_role_policy = jsonencode({ Version = "2012-10-17", Statement = [{
    Effect    = "Allow", Action = "sts:AssumeRoleWithWebIdentity",
    Principal = { Federated = aws_iam_openid_connect_provider.cluster.arn },
    Condition = { StringEquals = {
      "${local.oidc_host}:aud" = "sts.amazonaws.com",
      "${local.oidc_host}:sub" = "system:serviceaccount:finserve:finserve-evidence"
    } }
  }] })
}
resource "aws_iam_role_policy" "evidence" {
  name = "immutable-evidence"
  role = aws_iam_role.evidence.id
  policy = jsonencode({ Version = "2012-10-17", Statement = [
    { Effect = "Allow", Action = ["s3:GetObject", "s3:PutObject"], Resource = "${aws_s3_bucket.evidence.arn}/evidence/sha256/*" },
    { Effect = "Allow", Action = "s3:ListBucket", Resource = aws_s3_bucket.evidence.arn,
    Condition = { StringLike = { "s3:prefix" = ["evidence/sha256/*"] } } }
  ] })
}
