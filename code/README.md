
1. install pipenv   
    ```pip install pipenv```
    - install dependencies       
        ```pipenv install pandas scikit-learn mlflow boto3 prefect --python==3.10```
    - install dependencies for development (test, formatiing)   
        ```pipenv install --dev pylint black isort pre-commit```
2. set up MLflow tracking server on EC2
    * set up AWS services like [this](https://github.com/DataTalksClub/mlops-zoomcamp/blob/main/02-experiment-tracking/mlflow_on_aws.md)
    * run `python model.py` to train the model
    


