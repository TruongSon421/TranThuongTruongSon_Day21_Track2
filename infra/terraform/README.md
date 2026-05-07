# Terraform for model artifact storage

This Terraform stack creates an S3 bucket for model artifacts and exposes values needed by CI/CD and serving.

## 1) Apply infrastructure

```bash
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars
terraform init
terraform plan
terraform apply
```

## 2) Read outputs

```bash
terraform output
```

Use these outputs:

- `s3_bucket_name` -> GitHub secret `S3_BUCKET`
- `aws_region` -> GitHub secret `AWS_REGION`
- `model_key` -> runtime env var `S3_MODEL_KEY` on VM
- `serve_runtime_env_vars` -> all env vars your serving process needs

## 3) GitHub Actions secrets

Current workflow `.github/workflows/mlops.yml` still expects:

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`
- `S3_BUCKET`
- `EC2_HOST`
- `EC2_USER`
- `EC2_SSH_KEY`

`AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` come from an IAM user (or switch workflow to OIDC role later).

## 4) EC2 runtime env vars for serving

Set these on your EC2 service (`mlops-serve`):

- `AWS_REGION`
- `S3_BUCKET`
- `S3_MODEL_KEY` (default: `models/latest/model.pkl`)

Your `src/serve.py` can load model from `s3://$S3_BUCKET/$S3_MODEL_KEY`.
