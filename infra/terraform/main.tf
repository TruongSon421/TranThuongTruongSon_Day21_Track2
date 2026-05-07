terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

resource "random_id" "bucket_suffix" {
  byte_length = 3
}

locals {
  bucket_name = "${var.project_name}-${var.environment}-model-store-${random_id.bucket_suffix.hex}"
}

resource "aws_s3_bucket" "model_store" {
  bucket = local.bucket_name

  tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

resource "aws_s3_bucket_versioning" "model_store" {
  bucket = aws_s3_bucket.model_store.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "model_store" {
  bucket = aws_s3_bucket.model_store.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "model_store" {
  bucket = aws_s3_bucket.model_store.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# --- IAM Role for EC2 (no static credentials needed) ---

resource "aws_iam_role" "ec2_serve" {
  name = "${var.project_name}-${var.environment}-ec2-serve-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })

  tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

resource "aws_iam_role_policy" "ec2_s3_model_access" {
  name = "s3-model-access"
  role = aws_iam_role.ec2_serve.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ]
      Resource = [
        aws_s3_bucket.model_store.arn,
        "${aws_s3_bucket.model_store.arn}/*"
      ]
    }]
  })
}

resource "aws_iam_instance_profile" "ec2_serve" {
  name = "${var.project_name}-${var.environment}-ec2-serve-profile"
  role = aws_iam_role.ec2_serve.name
}

# --- EC2 for inference API ---

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

resource "aws_security_group" "mlops_serve" {
  name        = "${var.project_name}-${var.environment}-serve-sg"
  description = "Allow inbound on port 8000 (inference API) and SSH"

  ingress {
    description = "Inference API"
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "SSH for deployment"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

resource "aws_instance" "mlops_serve" {
  ami                         = data.aws_ami.ubuntu.id
  instance_type               = var.ec2_instance_type
  key_name                    = var.ec2_key_name
  vpc_security_group_ids      = [aws_security_group.mlops_serve.id]
  iam_instance_profile        = aws_iam_instance_profile.ec2_serve.name
  associate_public_ip_address = true
  instance_market_options {
    market_type = "spot"

    spot_options {
      spot_instance_type             = "one-time"
      instance_interruption_behavior = "terminate"
    }
  }

  user_data = <<-EOF
    #!/bin/bash
    apt-get update -y
    apt-get install -y python3-pip
    pip3 install fastapi uvicorn scikit-learn joblib boto3
    mkdir -p /home/ubuntu/models /home/ubuntu/src
    chown -R ubuntu:ubuntu /home/ubuntu/models /home/ubuntu/src
  EOF

  tags = {
    Name        = "${var.project_name}-${var.environment}-serve"
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }

  lifecycle {
    ignore_changes = [user_data]
  }
}
