# instructions for running the code
1. install pipenv   
    ```pip install pipenv```
    - install dependencies       
        ```pipenv install pandas scikit-learn mlflow prefect --python==3.10```
    - install dependencies for development (test, formatiing)   
        ```pipenv install --dev pylint black isort pre-commit```
2. set values in `pyproject.toml` and `.pre-commit-config.yaml` for formatting and linting
    * need to `pre-commit install` to install pre-commit hook
3. set up MLflow tracking server
    * use Terraform to create resources on AWS (EC2, S3, RDS) [Terraform setup](https://github.com/ChungWasawat/dtc_mlops_project/tree/main/infrastructure)
    * set up those services like [this](https://github.com/DataTalksClub/mlops-zoomcamp/blob/main/02-experiment-tracking/mlflow_on_aws.md)
4. use Prefect deployment to help tuning hyperparameters 
    * use `prefect server start` or `prefect cloud login` to start Prefect server
    * use `prefect project init` (it's necessary to create a prefect deployment)
    * use `prefect work-pool create my-pool` to create work pool for a worker
    * use `prefect worker start -p my-pool -t process` to start a worker 
    * use `prefect deploy code/model.py:flow -n 'deployment' -p my-pool` to create deployment to tune hyperparameters
    * use MLFlow UI to track the experiment on `http://aws_bucket_uri:5000/`
5. after some experiments are done, can register the best model with `best_model.py` 
    * can create a Prefect deployment of this file or just `python best_model.py` on terminal


