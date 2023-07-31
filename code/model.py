# pylint: disable=import-error, ungrouped-imports, invalid-name, line-too-long, too-many-locals, too-many-arguments
"""Model training pipeline"""
# import argparse
import os
import pickle
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

# variables
EXPERIMENT_NAME = "mlops-project-coupon-accepting-experiment"
TRACKING_SERVER_HOST = (
    "ec2-13-51-56-182.eu-north-1.compute.amazonaws.com"  # shouldn't be showed like this
)
TRACKING_SERVER_URI = f"http://{TRACKING_SERVER_HOST}:5000"

DATA_BUCKET = "state-persist"


# functions
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
def prepare_data_valid_set(
    whole_df: pd.DataFrame,
) -> tuple(
    [
        scipy.sparse._csr.csr_matrix,
        scipy.sparse._csr.csr_matrix,
        np.ndarray,
        np.ndarray,
        sklearn.feature_extraction.DictVectorizer,
        np.ndarray,
        np.ndarray,
    ]
):
    """Split train, val dataset and convert dataframe to dict"""
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

    y_train = df_train['coupon_accepting'].values
    y_val = df_val['coupon_accepting'].values

    return X_train, X_val, y_train, y_val, dict_vector, df_full_train, df_test


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
        mlflow.set_tag("eval", "validation set")

        train = xgb.DMatrix(train_features, label=train_target)
        valid = xgb.DMatrix(val_features, label=val_target)

        # save preprocessor
        Path("others").mkdir(exist_ok=True)
        with open("others/val_preprocessor.b", "wb") as f_out:
            pickle.dump(dict_vector, f_out)
        mlflow.log_artifact(
            "others/val_preprocessor.b", artifact_path="val_preprocessor"
        )

        params = {
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
        mlflow.log_params(params)

        booster = xgb.train(
            params=params,
            dtrain=train,
            num_boost_round=500,
            evals=[(valid, "validation")],
            early_stopping_rounds=50,
        )

        # log score
        pred_target = booster.predict(valid)
        auc_score = roc_auc_score(val_target, pred_target)
        mlflow.log_metric("auc_score", auc_score)

        # save model
        mlflow.xgboost.log_model(booster, artifact_path="val_models")
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


def dump_pickle(obj, filename: str):
    """create pickle file"""
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)


@task
def prepare_data_test_set(df_train, df_test):
    """Split train, test dataset and convert dataframe to dict"""
    categorical = ['destination', 'weather', 'time', 'coupon', 'expiration']
    boolean = ['same_direction']

    # convert dataframe to dict
    dict_vector = DictVectorizer()

    train_dicts = df_train[categorical + boolean].to_dict(orient="records")
    X_train = dict_vector.fit_transform(train_dicts)

    test_dicts = df_test[categorical + boolean].to_dict(orient="records")
    X_test = dict_vector.transform(test_dicts)

    y_train = df_train['coupon_accepting'].values
    y_test = df_test['coupon_accepting'].values

    dump_pickle((X_train, y_train), os.path.join("others", "train.pkl"))
    dump_pickle((X_test, y_test), os.path.join("others", "test.pkl"))

    with mlflow.start_run():
        with open("others/final_preprocessor.b", "wb") as f_out:
            pickle.dump(dict_vector, f_out)
        mlflow.log_artifact(
            "others/final_preprocessor.b", artifact_path="final_preprocessor"
        )


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
    mlflow.set_tracking_uri(TRACKING_SERVER_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load
    coupon_df = read_data(input_path)

    # Transform
    X_train, X_val, y_train, y_val, dv, df_full_train, df_test = prepare_data_valid_set(
        coupon_df
    )

    # Train
    train_best_model(X_train, X_val, y_train, y_val, dv, lr, mc, md, rl, ra)

    # save full_train and test set on local machine
    prepare_data_test_set(df_full_train, df_test)


if __name__ == "__main__":
    ## use argparse to tune hyperparams from command line
    # parser = argparse.ArgumentParser(
    #     description='Change parameters to train or Predict test set.'
    # )
    # parser.add_argument(
    #     '-lr', default=0.01, type=float, help='enter float number for learning_rate'
    # )
    # parser.add_argument(
    #     '-md', default=30, type=int, help='enter decimal number for min_child_weight'
    # )
    # parser.add_argument(
    #     '-mc', default=1.0, type=float, help='enter decimal number for max_depth'
    # )
    # parser.add_argument(
    #     '-rl', default=0.02, type=float, help='enter float number for reg_lambda'
    # )
    # parser.add_argument(
    #     '-ra', default=0.02, type=float, help='enter float number for reg_alpha'
    # )
    # args = parser.parse_args()

    # LEARNING_RATE = args.lr
    # MAX_DEPTH = args.md
    # MIN_CHILD = args.mc
    # REG_LAMBDA = args.rl
    # REG_ALPHA = args.ra

    # can tune from prefect deployment
    LEARNING_RATE = 0.01
    MAX_DEPTH = 30
    MIN_CHILD = 1.0
    REG_LAMBDA = 0.02
    REG_ALPHA = 0.02

    main_flow_coupon_accepting(
        LEARNING_RATE, MIN_CHILD, MAX_DEPTH, REG_LAMBDA, REG_ALPHA
    )
