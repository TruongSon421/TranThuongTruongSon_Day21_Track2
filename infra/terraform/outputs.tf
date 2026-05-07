output "s3_bucket_name" {
  description = "S3 bucket name to store model artifacts."
  value       = aws_s3_bucket.model_store.bucket
}

output "aws_region" {
  description = "AWS region for CI/CD and serving app."
  value       = var.aws_region
}

output "model_key" {
  description = "Model object key consumed by the serving app."
  value       = var.model_key
}

output "github_actions_required_secrets" {
  description = "Secrets required by current GitHub Actions workflow."
  value = [
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_REGION",
    "S3_BUCKET",
    "EC2_HOST",
    "EC2_USER",
    "EC2_SSH_KEY"
  ]
}

output "serve_runtime_env_vars" {
  description = "Environment variables your serving process needs on VM (no credentials needed — IAM Role handles auth)."
  value = {
    AWS_REGION   = var.aws_region
    S3_BUCKET    = aws_s3_bucket.model_store.bucket
    S3_MODEL_KEY = var.model_key
  }
}

output "ec2_iam_role_arn" {
  description = "IAM Role ARN attached to EC2. boto3 uses this automatically via instance metadata."
  value       = aws_iam_role.ec2_serve.arn
}

output "ec2_public_ip" {
  description = "Public IP of the inference server. Use this as EC2_HOST in GitHub Secrets."
  value       = aws_instance.mlops_serve.public_ip
}

output "inference_api_url" {
  description = "Public URL for the inference API."
  value       = "http://${aws_instance.mlops_serve.public_ip}:8000"
}
