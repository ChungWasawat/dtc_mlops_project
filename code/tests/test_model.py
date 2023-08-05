# pylint: disable=import-error
"""To do unit testing"""
import pandas as pd
import xgboost as xgb

import model_deployment_only_model as model


def test_prepare_features():
    """to test that prepare_features() returns the expected result"""
    model_service = model.ModelService(model=None, model_version=None)

    test_data = {
        "destination": ["No Urgent Place", "Home", "Work"],
        "weather": ["Sunny", "Rainy", "Snowy"],
        "time": ["10AM", "10PM", "7AM"],
        "coupon": ["Coffee House", "Coffee House", "Coffee House"],
        "expiration": ["2h", "2h", "1d"],
        "direction_same": [0, 1, 1],
        "direction_opp": [1, 0, 0],
        "Y": [0, 0, 0],
    }

    actual_result = model_service.prepare_features(test_data)

    expected_result = pd.DataFrame(
        {
            "destination": ["No Urgent Place", "Home", "Work"],
            "weather": ["Sunny", "Rainy", "Snowy"],
            "time": ["10AM", "10PM", "7AM"],
            "coupon": ["Coffee House", "Coffee House", "Coffee House"],
            "expiration": ["2h", "2h", "1d"],
            "same_direction": [0, 1, 1],
            "coupon_accepting": [0, 0, 0],
        }
    )

    assert expected_result.equals(actual_result)


def test_convert_to_dmatrix():
    """to test that convert_features_to_dmatrix() returns a correct type of the result"""
    model_service = model.ModelService(model=None, model_version=None)

    test_df = pd.DataFrame(
        {
            "destination": ["No Urgent Place", "Home", "Work"],
            "weather": ["Sunny", "Rainy", "Snowy"],
            "time": ["10AM", "10PM", "7AM"],
            "coupon": ["Coffee House", "Coffee House", "Coffee House"],
            "expiration": ["2h", "2h", "1d"],
            "same_direction": [0, 1, 1],
            "coupon_accepting": [0, 0, 0],
        }
    )

    actual_result = model_service.convert_features_to_dmatrix(test_df)

    assert isinstance(actual_result, xgb.DMatrix)
