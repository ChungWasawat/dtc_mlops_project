# instructions for running the code
1. install pipenv   
    ```pip install pipenv```
    - install dependencies       
        ```pipenv install pandas scikit-learn mlflow prefect --python==3.9```
    - install dependencies for development (test, formatiing)   
        ```pipenv install --dev pylint black isort pre-commit pytest```
2. set values in `pyproject.toml` and `.pre-commit-config.yaml` for formatting and linting
    * need to `pre-commit install` to install pre-commit hook
3. set up MLflow tracking server
    * use Terraform to create resources on AWS (EC2, S3, RDS) [Terraform setup](https://github.com/ChungWasawat/dtc_mlops_project/tree/main/code/infrastructure)
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
6. create a server for sending request to predict the result
    * `docker build -t coupon-accepting-prediction-service:v1 .`
    * `docker run -it --rm -p 9696:9696 coupon-accepting-prediction-service:v1`
    * `docker run -it --rm -p 9696:9696 -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY  coupon-accepting-prediction-service:v1` to pass credentials to load model from AWS S3
        -   `AWS_ACCESS_KEY_ID=$(aws --profile default configure get aws_access_key_id)`
        -   `AWS_SECRET_ACCESS_KEY=$(aws --profile default configure get aws_secret_access_key)`
    * 

