# pylint: disable=import-error
"""only variables for unit testing"""
import os

import mlflow
from flask import Flask, jsonify, request

import model_deployment_only_model as model

# variables
RUN_ID = os.getenv('RUN_ID')

logged_model = (
    f's3://mlflow-artifacts-mlops-project-storage/2/{RUN_ID}/artifacts/final_models'
)
# logged_model = f'runs:/{RUN_ID}/model'
loaded_model = mlflow.pyfunc.load_model(logged_model)

app = Flask('coupon-accepting-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """main functions"""
    ride = request.get_json()

    model_service = model.ModelService(loaded_model, RUN_ID)
    features = model_service.prepare_features(ride)
    pred = model_service.predict(features)

    result = {'coupon_accepting': pred, 'model_version': RUN_ID}

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
