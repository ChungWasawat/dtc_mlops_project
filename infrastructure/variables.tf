variable "aws_region" {
  description = "AWS region to create resources"
  default     = "eu-west-2"
}

variable "project_id" {
  description = "project_id"
  default = "mlops-proj-coupon-accepting"
}

# s3
variable "model_bucket" {
  description = "s3_bucket"
  default = "model-storage"
}


# variable "lambda_function_local_path" {
#   description = ""
# }

# variable "docker_image_local_path" {
#   description = ""
# }

# variable "ecr_repo_name" {
#   description = ""
# }

# variable "lambda_function_name" {
#   description = ""
# }