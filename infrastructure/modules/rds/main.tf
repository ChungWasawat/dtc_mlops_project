resource "aws_db_instance" "mlflow_backend" {
  engine                        = var.engine_type
  instance_class                = var.instance_type

  allocated_storage             = var.storage_size
  db_name                       = var.database_name
  username                      = var.master_name
  password                      = var.db_password
  skip_final_snapshot           = true
}
