"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import exclude_feature, read_files, separate_numeric_categoric_feature, train_test_spliter, encode_categorical_features


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(read_files,
            ['raw_data'],
            'data',
            name = 'read_files'),
        node(exclude_feature,
            ['data'],
            'features',
            name = 'exclude'),
        node(separate_numeric_categoric_feature,
            ['data', 'features'],
            ['numeric_features', 'categorical_features'],
            name = 'separate'),
        node(train_test_spliter,
            ['data', 'numeric_features', 'categorical_features', 'params:target_col_pre'],
            ['train', 'valid', 'y_train', 'y_valid'],
            name = 'split'),
        node(encode_categorical_features,
            ['categorical_features', 'train', 'valid'],
            ['X_train', 'X_valid'],
            name = 'encode'),
    ])
