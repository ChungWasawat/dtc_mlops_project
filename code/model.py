# pylint: disable=import-error, ungrouped-imports, invalid-name, line-too-long, too-many-locals, too-many-arguments
"""Model training pipeline"""
import pickle
import argparse
from pathlib import Path
from datetime import date

import numpy as np
import scipy
import mlflow
import pandas as pd
import sklearn
import xgboost as xgb
from prefect import flow, task
from sklearn.metrics import roc_auc_score
from prefect.artifacts import create_markdown_artifact
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer


@task(retries=3, retry_delay_seconds=2)
def read_data(filename: str) -> pd.DataFrame:
    """Read data into DataFrame"""
    # create dataframe from the file
    coupon_rec_df = pd.read_csv(filename, compression='zip')

    coupon_rec_df.rename(
        columns={'direction_same': 'same_direction', 'Y': 'coupon_accepting'},
        inplace=True,
    )

    # choose interested features and target
    categorical = ['destination', 'weather', 'time', 'coupon', 'expiration']
    boolean = ['same_direction', 'coupon_accepting']

    # convert to categorical when the columns are used repeatedly,
    # it's helpful in memory management and fast processing
    coupon_rec_df[categorical] = coupon_rec_df[categorical].astype(str)

    return coupon_rec_df[categorical + boolean]


@task
def create_dict_vector(
    whole_df: pd.DataFrame,
) -> tuple(
    [
        scipy.sparse._csr.csr_matrix,
        scipy.sparse._csr.csr_matrix,
        scipy.sparse._csr.csr_matrix,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        sklearn.feature_extraction.DictVectorizer,
    ]
):
    """Split train, val, test dataset and convert dataframe to dict"""
    df_full_train, df_test = train_test_split(whole_df, test_size=0.2, random_state=42)
    df_train, df_val = train_test_split(df_full_train, test_size=0.2, random_state=42)

    categorical = ['destination', 'weather', 'time', 'coupon', 'expiration']
    boolean = ['same_direction']

    # convert dataframe to dict
    dict_vector = DictVectorizer()

    train_dicts = df_train[categorical + boolean].to_dict(orient="records")
    X_train = dict_vector.fit_transform(train_dicts)

    val_dicts = df_val[categorical + boolean].to_dict(orient="records")
    X_val = dict_vector.transform(val_dicts)

    test_dicts = df_test[categorical + boolean].to_dict(orient="records")
    X_test = dict_vector.fit_transform(test_dicts)

    y_train = df_train['coupon_accepting'].values
    y_val = df_val['coupon_accepting'].values
    y_test = df_test['coupon_accepting'].values
    return X_train, X_val, X_test, y_train, y_val, y_test, dict_vector


@task(log_prints=True)
def train_best_model(
    train_features: scipy.sparse._csr.csr_matrix,
    val_features: scipy.sparse._csr.csr_matrix,
    train_target: np.ndarray,
    val_target: np.ndarray,
    dict_vector: sklearn.feature_extraction.DictVectorizer,
    learning_rate: float,
    min_child_weight: float,
    max_depth: int,
    reg_lambda: float,
    reg_alpha: float,
):
    """train a model with best hyperparams and write everything out"""

    with mlflow.start_run():
        mlflow.set_tag("model", "xgboost")

        train = xgb.DMatrix(train_features, label=train_target)
        valid = xgb.DMatrix(val_features, label=val_target)

        best_params = {
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "min_child_weight": min_child_weight,
            "objective": "binary:logistic",
            "reg_alpha": reg_lambda,
            "reg_lambda": reg_alpha,
            "seed": 42,
            'verbosity': 1,
        }

        # log hyperparams
        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=500,
            evals=[(valid, "validation")],
            early_stopping_rounds=50,
        )

        # log score
        pred_target = booster.predict(valid)
        auc_score = roc_auc_score(val_target, pred_target)
        mlflow.log_metric("auc_score", auc_score)

        # save preprocessor
        Path("others").mkdir(exist_ok=True)
        with open("others/preprocessor.b", "wb") as f_out:
            pickle.dump(dict_vector, f_out)
        mlflow.log_artifact("others/preprocessor.b", artifact_path="preprocessor")

        # save model
        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")
        print(f"default artifacts URI: '{mlflow.get_artifact_uri()}'")

        # create prefect markdown report
        markdown_auc_report = f"""# AUC Report VALIDATION SET

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


@task
def predict_test_set(feature_test, target_test):
    """predict test set and report a result"""
    if feature_test == target_test:
        print("Okay!")


# dangerous-default-value:
# https://stackoverflow.com/questions/26320899/why-is-the-empty-dictionary-a-dangerous-default-value-in-python
@flow
def main_flow_coupon_accepting(lr, mc, md, rl, ra) -> None:
    """The main model training pipeline"""
    # input path
    input_path = "https://archive.ics.uci.edu/static/public/603/in+vehicle+coupon+recommendation.zip"
    # train_path = f"./data/coupon_recommendation.parquet"

    print("train path: ", input_path)
    # MLflow settings
    TRACKING_SERVER_HOST = "ec2-13-51-56-182.eu-north-1.compute.amazonaws.com"
    mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")
    mlflow.set_experiment("mlops-project-coupon-accepting-experiment")

    # Load
    coupon_df = read_data(input_path)

    # Transform
    X_train, X_val, X_test, y_train, y_val, y_test, dv = create_dict_vector(coupon_df)

    # Train
    train_best_model(X_train, X_val, y_train, y_val, dv, lr, mc, md, rl, ra)

    # save test data to s3 for future prediction
    predict_test_set(X_test, y_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Change parameters to train or Predict test set.'
    )
    parser.add_argument(
        '-lr', default=0.01, type=float, help='enter float number for learning_rate'
    )
    parser.add_argument(
        '-md', default=30, type=int, help='enter decimal number for min_child_weight'
    )
    parser.add_argument(
        '-mc', default=1.0, type=float, help='enter decimal number for max_depth'
    )
    parser.add_argument(
        '-rl', default=0.02, type=float, help='enter float number for reg_lambda'
    )
    parser.add_argument(
        '-ra', default=0.02, type=float, help='enter float number for reg_alpha'
    )
    args = parser.parse_args()

    LEARNING_RATE = args.lr
    MAX_DEPTH = args.md
    MIN_CHILD = args.mc
    REG_LAMBDA = args.rl
    REG_ALPHA = args.ra

    main_flow_coupon_accepting(
        LEARNING_RATE, MIN_CHILD, MAX_DEPTH, REG_LAMBDA, REG_ALPHA
    )
