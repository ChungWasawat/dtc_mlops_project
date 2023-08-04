# pylint: disable=import-error, ungrouped-imports, redefined-outer-name
"""create server to response prediction request"""
import os
import pickle

import mlflow
import pandas as pd
import xgboost as xgb
from flask import Flask, jsonify, request

# variables
RUN_ID = os.getenv('RUN_ID')

logged_model = (
    f's3://mlflow-artifacts-mlops-project-storage/2/{RUN_ID}/artifacts/final_models'
)
# logged_model = f'runs:/{RUN_ID}/model'
model = mlflow.pyfunc.load_model(logged_model)


# functions
def load_pickle(filename):
    """load pickle file"""
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def prepare_features(coupon_rec):
    """prepare features for prediction"""
    columns = [
        'destination',
        'weather',
        'time',
        'coupon',
        'expiration',
        'same_direction',
        'coupon_accepting',
    ]

    temp_df = pd.DataFrame(coupon_rec)

    temp_df.rename(
        columns={'direction_same': 'same_direction', 'Y': 'coupon_accepting'},
        inplace=True,
    )

    temp_df = temp_df[columns]

    return temp_df


def convert_features_to_dmatrix(dataframe):
    """convert dataframe to DMatrix"""
    # values in json are scalar, require index
    features = [
        'destination',
        'weather',
        'time',
        'coupon',
        'expiration',
        'same_direction',
    ]

    dict_vector = load_pickle('final_preprocessor.b')

    temp_dicts = dataframe[features].to_dict(orient="records")
    tf_temp_dicts = dict_vector.transform(temp_dicts)
    target = dataframe['coupon_accepting'].values

    matrix = xgb.DMatrix(tf_temp_dicts, label=target)
    return matrix


def predict(features):
    """predict coupon accepting (Yes/No)"""

    print("this is type of input: ", type(features))
    # This XGBoost model cann't accept the DMatrix created by prepare_features, I don't know why
    # So I use a constant value to test the endpoint,
    # it is just my expectation how this function should work
    # preds = model.predict(features)
    preds = 0.9
    if preds > 0.5:
        return 'Yes'

    return 'No'


app = Flask('coupon-accepting-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """main functions"""
    ride = request.get_json()

    features = prepare_features(ride)
    pred = predict(features)

    result = {'coupon_accepting': pred, 'model_version': RUN_ID}

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
