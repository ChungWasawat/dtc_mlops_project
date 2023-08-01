variable "ami_id" {
  description = "amazon machine image id"
  default = ""
}

variable "aws_instance_type" {
  description = "aws instance type"
  default = ""
}

variable "ec2_instance_name" {
  description = "Name of ec2 instance"
  default = "mlflow-tracking-server"
}