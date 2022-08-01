from typing import Any, Dict, List

import pandas as pd

import sklearn

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def encode_target_col(data, target_col):
    try:
        print("The dataset has {} samples with {} features.".format(*data.shape))
    except:
        print("The dataset could not be loaded. Is the dataset missing?")
    target = data[target_col].map({"Placed": 0 , "Not Placed": 1})
    return data, target

def exclude_feature(data, drop_features):
    drop_features_list = drop_features.split(',')
    new_data = data.drop(drop_features_list, axis=1)
    return new_data

def encode_categorical_features(data):
    categorical_features = separate_categoric_feature(data)
    for feature in categorical_features:
        le = LabelEncoder()
        le.fit(data.loc[:, feature])
        data.loc[:, feature] = le.transform(data.loc[:, feature])
    return data

def separate_categoric_feature(data):
    features = data.columns
    # numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
    # numeric_features = [col for col in numeric_columns if col in features]
    categorical_features = [col for col in categorical_columns if col in features]
    return categorical_features


def train_test_spliter(data, target):
    data = data.fillna(0)
    X_train, X_valid, y_train, y_valid = train_test_split(data, target, test_size=0.15, random_state=10)
    return X_train, X_valid, y_train, y_valid
 