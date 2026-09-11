locals {
  zones = { for index, zone in var.availability_zones : tostring(index) => zone }
}
resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_support   = true
  enable_dns_hostnames = true
  tags                 = { Name = var.name }
}
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
}
resource "aws_subnet" "public" {
  for_each                = local.zones
  vpc_id                  = aws_vpc.main.id
  availability_zone       = each.value
  cidr_block              = cidrsubnet(var.vpc_cidr, 8, tonumber(each.key))
  map_public_ip_on_launch = false
  tags                    = { Name = "${var.name}-public-${each.key}", "kubernetes.io/role/elb" = "1" }
}
resource "aws_subnet" "private" {
  for_each          = local.zones
  vpc_id            = aws_vpc.main.id
  availability_zone = each.value
  cidr_block        = cidrsubnet(var.vpc_cidr, 8, 10 + tonumber(each.key))
  tags              = { Name = "${var.name}-private-${each.key}", "kubernetes.io/role/internal-elb" = "1" }
}
resource "aws_eip" "nat" {
  domain = "vpc"
}
# One NAT is a deliberate staging cost/availability tradeoff; neither AZ is egress-independent.
resource "aws_nat_gateway" "main" {
  allocation_id = aws_eip.nat.id
  subnet_id     = aws_subnet.public["0"].id
  depends_on    = [aws_internet_gateway.main]
}
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }
}
resource "aws_route_table" "private" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.main.id
  }
}
resource "aws_route_table_association" "public" {
  for_each       = aws_subnet.public
  subnet_id      = each.value.id
  route_table_id = aws_route_table.public.id
}
resource "aws_route_table_association" "private" {
  for_each       = aws_subnet.private
  subnet_id      = each.value.id
  route_table_id = aws_route_table.private.id
}
resource "aws_vpc_endpoint" "s3" {
  vpc_id            = aws_vpc.main.id
  service_name      = "com.amazonaws.${var.region}.s3"
  vpc_endpoint_type = "Gateway"
  route_table_ids   = [aws_route_table.private.id]
}
