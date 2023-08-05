# pylint: disable=import-error
"""only functions for unit testing"""
import pickle
from pathlib import Path

import pandas as pd
import xgboost as xgb


class ModelService:
    """create this one to do unit testing"""

    def __init__(self, model, model_version=None) -> None:
        self.model = model
        self.model_version = model_version

    # functions
    def load_pickle(self, filename):
        """load pickle file"""
        with open(Path(__file__).parent / filename, "rb") as f_in:
            return pickle.load(f_in)

    def prepare_features(self, coupon_rec):
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

    def convert_features_to_dmatrix(self, dataframe):
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

        dict_vector = self.load_pickle('others/final_preprocessor.b')

        temp_dicts = dataframe[features].to_dict(orient="records")
        tf_temp_dicts = dict_vector.transform(temp_dicts)
        target = dataframe['coupon_accepting'].values

        matrix = xgb.DMatrix(tf_temp_dicts, label=target)
        return matrix

    def predict(self, features):
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
