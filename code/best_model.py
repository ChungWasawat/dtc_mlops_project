# pylint: disable=import-error, ungrouped-imports
"""train final models and find the best model to register on mlflow"""
import os
import pickle
from datetime import date

import mlflow
import xgboost as xgb
from prefect import flow, task
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.metrics import roc_auc_score
from prefect.artifacts import create_markdown_artifact

# variables
VAL_EXPERIMENT_NAME = "mlops-project-coupon-accepting-experiment"
EXPERIMENT_NAME = "final-model-mlops-project-coupon-accepting"
TRACKING_SERVER_HOST = os.environ.get('MLFLOW_SERVER_HOST')
TRACKING_SERVER_URI = f"http://{TRACKING_SERVER_HOST}:5000"
XGB_PARAMS_FLOAT = ['learning_rate', 'min_child_weight', 'reg_alpha', 'reg_lambda']
XGB_PARAMS_INT = ['max_depth', 'seed', 'verbosity']

mlflow.set_tracking_uri(TRACKING_SERVER_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


# functions
def load_pickle(filename):
    """load pickle file"""
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@task
def train_and_log_final_model(
    data_path: str,
    params: list,
):
    """predict test set and report a result"""
    x_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    x_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

    with mlflow.start_run():
        mlflow.set_tag("model", "xgboost")
        mlflow.set_tag("eval", "test set")

        train = xgb.DMatrix(x_train, label=y_train)
        test = xgb.DMatrix(x_test, label=y_test)

        for param in XGB_PARAMS_FLOAT:
            params[param] = float(params[param])
        for param in XGB_PARAMS_INT:
            params[param] = int(params[param])

        # log hyperparams
        mlflow.log_params(params)

        booster = xgb.train(
            params=params,
            dtrain=train,
            num_boost_round=500,
            evals=[(test, "test")],
            early_stopping_rounds=50,
        )

        # log score
        pred_target = booster.predict(test)
        auc_score = roc_auc_score(y_test, pred_target)
        mlflow.log_metric("auc_score", auc_score)

        # save model
        mlflow.xgboost.log_model(booster, artifact_path="final_models")
        print(f"default artifacts URI: '{mlflow.get_artifact_uri()}'")

        # create prefect markdown report
        markdown_auc_report = f"""# AUC Report TEST SET

        ## Summary

        Duration Prediction 

        ## AUC XGBoost Model

        | Region    | AUC |
        |:----------|-------:|
        | {date.today()} | {auc_score:.2f} |
        """

        create_markdown_artifact(
            key="coupon-model-report", markdown=markdown_auc_report
        )


@flow
def main_flow(top_n=5):
    """main flow for final training and to register the best model"""

    client = MlflowClient()

    experiment = client.get_experiment_by_name(VAL_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        filter_string="tags.eval = 'validation set'",
        run_view_type=ViewType.ACTIVE_ONLY,
        # how many runs to return
        max_results=top_n,
        order_by=["metrics.auc_score DESC"],
    )

    for run in runs:
        # run.info.run_id is the id of the run
        # run.data.metrics["rmse"]
        print("This is the run id: ", run.info.run_id)
        train_and_log_final_model(data_path="others", params=run.data.params)

    # Select the model with the lowest test RMSE
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.auc_score DESC"],
    )[0]

    best_model_uri = f"runs:/{best_run.info.run_id}/final_models"
    print("This is the best model's uri: ", best_model_uri)
    # Register the best model
    mlflow.register_model(
        model_uri=best_model_uri,
        name="coupon-accepting-xgb-model",
    )


if __name__ == "__main__":
    TOP_NUMBER = 5

    main_flow(TOP_NUMBER)
