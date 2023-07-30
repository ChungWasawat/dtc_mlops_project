variable "engine_type" {
  description = "engine type of database instance: postgres, mysql, etc."
  default = "postgres"
}

variable "instance_type" {
    description = "instance type"
    default = ""
}

variable "storage_size" {
    description = "allocated storage size"
    default = 100
}

variable "database_name" {
    description = "database name"
    default = ""
}

variable "master_name" {
    description = "master username"
    default = ""
  
}

variable "db_password" {
    description = "master password"
    default = ""
}



