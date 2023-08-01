
1. install pipenv   
    ```pip install pipenv```
    - install dependencies       
        ```pipenv install pandas scikit-learn mlflow prefect --python==3.10```
    - install dependencies for development (test, formatiing)   
        ```pipenv install --dev pylint black isort pre-commit```
2. set values in `pyproject.toml` and `.pre-commit-config.yaml` for formatting and linting
3. set up MLflow tracking server
    * use Terraform to create resources on AWS (EC2, S3, RDS) [Terraform setup](https://github.com/ChungWasawat/dtc_mlops_project/tree/main/infrastructure)
    * set up those services like [this](https://github.com/DataTalksClub/mlops-zoomcamp/blob/main/02-experiment-tracking/mlflow_on_aws.md)
4. use 


