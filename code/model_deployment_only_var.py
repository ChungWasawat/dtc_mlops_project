# pylint: disable=import-error
"""only variables for unit testing"""
import os

import mlflow
from flask import Flask, jsonify, request

import model_deployment_only_model as model

# variables
RUN_ID = os.getenv('RUN_ID')
app = Flask('coupon-accepting-prediction')


# functions
def get_model_location(run_id):
    """get model location"""
    model_location = os.getenv('MODEL_LOCATION')

    if model_location is not None:
        return model_location

    model_bucket = os.getenv('MODEL_BUCKET', 'mlflow-artifacts-mlops-project-storage')
    experiment_id = os.getenv('EXPERIMENT_ID', '2')

    model_location = (
        f's3://{model_bucket}/{experiment_id}/{run_id}/artifacts/final_models'
    )
    return model_location


def load_model():
    """load model"""
    model_location = get_model_location(RUN_ID)
    loaded_model = mlflow.pyfunc.load_model(model_location)
    return loaded_model


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """main functions"""
    ride = request.get_json()

    loaded_model = load_model()
    model_service = model.ModelService(loaded_model, RUN_ID)
    features = model_service.prepare_features(ride)
    pred = model_service.predict(features)

    result = {'coupon_accepting': pred, 'model_version': RUN_ID}

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
