# Make sure to create state bucket beforehand
terraform {
  required_version = ">= 1.0"
  backend "s3" {
    bucket  = "tf-state-mlops-project-coupon-accepting"
    key     = "mlops-zoomcamp-stg.tfstate"
    region  = "eu-west-2"
    encrypt = true
  }
}

provider "aws" {
  region = var.aws_region
}

# can assume 'aws..' = class, 'current..' = instance, 'account_id' = property
data "aws_caller_identity" "current_identity" {}

locals {
  account_id = data.aws_caller_identity.current_identity.account_id
}

# mlflow s3 bucket
module "s3_bucket" {
  source = "./modules/s3"
  bucket_name = "${var.model_bucket}-${var.project_id}"
}

# mlflow tracking server ec2
module "aws_instance" {
  source = "./modules/ec2"

}

# mlflow backend rds
module "db_instance" {
  source = "./modules/rds"

}

# For CI/CD
output "model_bucket" {
  value = module.s3_bucket.name
}

