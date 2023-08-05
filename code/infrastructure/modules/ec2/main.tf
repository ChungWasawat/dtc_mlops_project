resource "aws_instance" "ec2_instance" {
  ami = var.ami_id
  instance_type = var.aws_instance_type
  # force_destroy = true
  tags = {
    Name = "${var.ec2_instance_name}"
  }
}
