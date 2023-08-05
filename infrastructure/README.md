1. install terraform
    - On Windows 11, download the zip file and extract the file at `C:\terraform`
    - add this path to `Path` of system environment variable
    - `terraform --version` to check if it is installed successfully
2. to keep terraform state on S3, need to create a bucket manually first
3. set values in varaibles.tf to set name of bucket, region, etc.
    - and other variables.tf in modules folder, for each AWS service
4. after that, use these terraform commands to create or delete resources
    - `terraform init`
    - `terraform plan`
    - `terraform apply`
    - `terraform destroy` 
