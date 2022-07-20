from typing import Any, Dict, List

import pandas as pd

import sklearn

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def read_files(data) -> pd.DataFrame:
    try:
        print("The dataset has {} samples with {} features.".format(*data.shape))
    except:
        print("The dataset could not be loaded. Is the dataset missing?")
    return data

def exclude_feature(data):
    exclude_feature = ['sl_no', 'salary', 'status']
    columns = data.columns.tolist()
    features = [col for col in columns if col not in exclude_feature]
    return features

def separate_numeric_categoric_feature(data, features):
    numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
    numeric_features = [col for col in numeric_columns if col in features]
    categorical_features = [col for col in categorical_columns if col in features]
    return numeric_features, categorical_features


def train_test_spliter(data, numeric_features, categorical_features, target_col):
    target = data[target_col].map({"Placed": 0 , "Not Placed": 1})

    features = numeric_features + categorical_features
    data = data[features]
    data = data.fillna(0)
    X_train, X_valid, y_train, y_valid = train_test_split(data, target, test_size=0.15, random_state=10)
    return X_train, X_valid, y_train, y_valid

def encode_categorical_features(categorical_features, train, valid):
    # train.head()
    # valid.head()
    # return train, valid
    for feature in categorical_features:
        le = LabelEncoder()
        le.fit(train.loc[:, feature])
        train.loc[:, feature] = le.transform(train.loc[:, feature])
        le.fit(valid.loc[:, feature])
        valid.loc[:, feature] = le.transform(valid.loc[:, feature])
    return train, valid
