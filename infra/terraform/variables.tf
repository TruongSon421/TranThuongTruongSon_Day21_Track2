variable "aws_region" {
  description = "AWS region to create infrastructure in."
  type        = string
  default     = "ap-southeast-1"
}

variable "project_name" {
  description = "Project slug used in resource naming."
  type        = string
  default     = "mlops-track2"
}

variable "environment" {
  description = "Deployment environment (dev, staging, prod)."
  type        = string
  default     = "dev"
}

variable "model_key" {
  description = "Default S3 object key for the deployed model."
  type        = string
  default     = "models/latest/model.pkl"
}

variable "ec2_instance_type" {
  description = "EC2 instance type for the inference server."
  type        = string
  default     = "t2.small"
}

variable "ec2_key_name" {
  description = "Name of the EC2 key pair for SSH access (must exist in AWS before apply)."
  type        = string
  default     = ""
}
